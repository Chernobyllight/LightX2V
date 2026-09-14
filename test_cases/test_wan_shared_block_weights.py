import hashlib
import os
from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import save_file

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

import lightx2v.models.networks.wan.shared_block_weights as shared_block_weights  # noqa: E402
from lightx2v.models.networks.wan.shared_block_weights import WanFp8VllmSharedBlockAdapter  # noqa: E402


def _block_tensors(block_index: int) -> dict[str, torch.Tensor]:
    prefix = f"blocks.{block_index}.self_attn.q"
    return {
        f"{prefix}.weight": torch.tensor(
            [[1.0, -2.0, 0.5], [3.0, 0.25, -0.5]],
            dtype=torch.float8_e4m3fn,
        ),
        f"{prefix}.weight_scale": torch.tensor(
            [[1.0003], [0.3333]],
            dtype=torch.float32,
        ),
        f"{prefix}.bias": torch.tensor([0.12345, -0.98765], dtype=torch.float32),
    }


@pytest.fixture
def tiny_checkpoint(tmp_path):
    for block_index in range(2):
        save_file(_block_tensors(block_index), tmp_path / f"block_{block_index}.safetensors")
    return tmp_path


@pytest.fixture
def fp16_dtypes(monkeypatch):
    monkeypatch.setattr(shared_block_weights, "GET_DTYPE", lambda: torch.float16)
    monkeypatch.setattr(shared_block_weights, "GET_SENSITIVE_DTYPE", lambda: torch.float16)


def _config(checkpoint, **overrides):
    config = {
        "cpu_offload": True,
        "offload_granularity": "block",
        "lazy_load": False,
        "dit_quantized": True,
        "dit_quant_scheme": "fp8-vllm",
        "dit_quantized_ckpt": str(checkpoint),
        "num_layers": 2,
        "tensor_parallel": False,
        "weight_auto_quant": False,
        "shared_cpu_weight_backend": "sysv",
    }
    config.update(overrides)
    return config


def test_manifest_preserves_fp8_and_matches_baseline_target_dtypes(tiny_checkpoint, fp16_dtypes):
    adapter = WanFp8VllmSharedBlockAdapter(_config(tiny_checkpoint))
    specs = adapter.manifest.by_name

    for block_index in range(2):
        prefix = f"blocks.{block_index}.self_attn.q"
        weight = specs[f"{prefix}.weight"]
        scale = specs[f"{prefix}.weight_scale"]
        bias = specs[f"{prefix}.bias"]

        assert weight.dtype == "float8_e4m3fn"
        assert weight.storage_shape == (2, 3)
        assert weight.logical_shape == (2, 3)
        assert scale.dtype == "float32"
        assert scale.storage_shape == (2, 1)
        assert bias.dtype == "float16"
        assert bias.storage_shape == (2,)

    assert len(adapter.manifest.tensors) == 6
    assert len(adapter.manifest.weight_signature) == 64
    assert adapter.manifest.weight_signature == WanFp8VllmSharedBlockAdapter(_config(tiny_checkpoint)).manifest.weight_signature


def test_checkpoint_signature_changes_when_payload_changes(tiny_checkpoint, fp16_dtypes):
    original_signature = WanFp8VllmSharedBlockAdapter(_config(tiny_checkpoint)).manifest.weight_signature
    checkpoint = tiny_checkpoint / "block_0.safetensors"
    with checkpoint.open("r+b") as checkpoint_file:
        checkpoint_file.seek(-1, os.SEEK_END)
        last_byte = checkpoint_file.read(1)
        checkpoint_file.seek(-1, os.SEEK_END)
        checkpoint_file.write(bytes([last_byte[0] ^ 1]))

    changed_signature = WanFp8VllmSharedBlockAdapter(_config(tiny_checkpoint)).manifest.weight_signature

    assert changed_signature != original_signature


def test_checkpoint_digest_prefers_valid_huggingface_metadata(tiny_checkpoint):
    checkpoint = tiny_checkpoint / "block_0.safetensors"
    metadata = tiny_checkpoint / ".cache/huggingface/download/block_0.safetensors.metadata"
    metadata.parent.mkdir(parents=True)
    expected_digest = "a" * 64
    metadata.write_text(f"commit\n{expected_digest}\n{checkpoint.stat().st_mtime}\n", encoding="utf-8")

    assert shared_block_weights._checkpoint_content_digest(checkpoint) == expected_digest

    metadata.write_text("commit\nnot-a-sha256\ntimestamp\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid SHA-256 on line 2"):
        shared_block_weights._checkpoint_content_digest(checkpoint)

    metadata.unlink()
    assert shared_block_weights._checkpoint_content_digest(checkpoint) == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_checkpoint_digest_ignores_stale_huggingface_metadata(tiny_checkpoint):
    checkpoint = tiny_checkpoint / "block_0.safetensors"
    metadata = tiny_checkpoint / ".cache/huggingface/download/block_0.safetensors.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"commit\n{'a' * 64}\n{checkpoint.stat().st_mtime - 10}\n", encoding="utf-8")

    assert shared_block_weights._checkpoint_content_digest(checkpoint) == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_populate_accepts_plain_cpu_views_and_reproduces_scale_round_trip(tiny_checkpoint, fp16_dtypes):
    adapter = WanFp8VllmSharedBlockAdapter(_config(tiny_checkpoint))
    views = {
        spec.name: torch.empty(spec.logical_shape, dtype=getattr(torch, spec.dtype))
        for spec in adapter.manifest.tensors
    }

    adapter._populate(views)

    for block_index in range(2):
        source = _block_tensors(block_index)
        prefix = f"blocks.{block_index}.self_attn.q"
        scale_name = f"{prefix}.weight_scale"
        bias_name = f"{prefix}.bias"

        assert torch.equal(views[f"{prefix}.weight"], source[f"{prefix}.weight"])
        # The existing loader first casts all F32 tensors to inference dtype;
        # fp8-vllm post-processing then stores scales as F32 again.
        expected_scale = source[scale_name].to(torch.float16).to(torch.float32)
        assert torch.equal(views[scale_name], expected_scale)
        assert not torch.equal(views[scale_name], source[scale_name])
        assert torch.equal(views[bias_name], source[bias_name].to(torch.float16))


def test_bfloat16_setting_uses_the_same_cast_rules(tiny_checkpoint, monkeypatch):
    monkeypatch.setattr(shared_block_weights, "GET_DTYPE", lambda: torch.bfloat16)
    monkeypatch.setattr(shared_block_weights, "GET_SENSITIVE_DTYPE", lambda: torch.bfloat16)
    adapter = WanFp8VllmSharedBlockAdapter(_config(tiny_checkpoint))
    views = {spec.name: torch.empty(spec.logical_shape, dtype=getattr(torch, spec.dtype)) for spec in adapter.manifest.tensors}

    adapter._populate(views)

    prefix = "blocks.0.self_attn.q"
    source = _block_tensors(0)
    assert views[f"{prefix}.bias"].dtype == torch.bfloat16
    expected_scale = source[f"{prefix}.weight_scale"].to(torch.bfloat16).to(torch.float32)
    assert torch.equal(views[f"{prefix}.weight_scale"], expected_scale)


@dataclass
class _FakeAllocation:
    views: dict[str, torch.Tensor]

    def tensor_views(self):
        return self.views


def test_materialize_callback_can_be_driven_without_cuda(tiny_checkpoint, fp16_dtypes, monkeypatch):
    adapter = WanFp8VllmSharedBlockAdapter(
        _config(
            tiny_checkpoint,
            shared_cpu_weight_scope="numa",
            shared_cpu_weight_strict_numa=False,
            shared_cpu_weight_register_chunk_mb=7,
        )
    )
    captured = {}

    def fake_materialize(manifest, populate, **kwargs):
        views = {
            spec.name: torch.empty(spec.logical_shape, dtype=getattr(torch, spec.dtype))
            for spec in manifest.tensors
        }
        populate(views)
        captured.update(kwargs)
        return _FakeAllocation(views)

    monkeypatch.setattr(shared_block_weights, "materialize_shared_weight_arena", fake_materialize)

    allocation = adapter.materialize()

    assert allocation is adapter.allocation
    assert captured == {
        "scope": "numa",
        "strict_numa": False,
        "register_chunk_bytes": 7 * 1024 * 1024,
    }
    scale = allocation.views["blocks.0.self_attn.q.weight_scale"]
    expected = _block_tensors(0)["blocks.0.self_attn.q.weight_scale"].to(torch.float16).to(torch.float32)
    assert torch.equal(scale, expected)
    with pytest.raises(RuntimeError, match="already been materialized"):
        adapter.materialize()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cpu_offload": False}, "cpu_offload must be true"),
        ({"offload_granularity": "phase"}, "offload_granularity must be 'block'"),
        ({"lazy_load": True}, "lazy_load must be false"),
        ({"dit_quant_scheme": "fp8-q8f"}, "requires dit_quantized=true"),
        ({"tensor_parallel": True}, "tensor_parallel is not supported yet"),
        ({"weight_auto_quant": True}, "weight_auto_quant is not supported"),
        ({"adapter_model_path": "adapter.safetensors"}, "adapter checkpoints are not supported"),
        ({"lora_configs": [{"path": "lora.safetensors"}]}, "LoRA/diff weights are not supported"),
        ({"shared_cpu_weight_backend": "posix"}, "shared_cpu_weight_backend must be 'sysv'"),
    ],
)
def test_invalid_shared_weight_config_is_rejected(tiny_checkpoint, fp16_dtypes, overrides, message):
    with pytest.raises(ValueError, match=message):
        WanFp8VllmSharedBlockAdapter(_config(tiny_checkpoint, **overrides))


def test_mismatched_sensitive_dtype_is_rejected(tiny_checkpoint, fp16_dtypes, monkeypatch):
    monkeypatch.setattr(shared_block_weights, "GET_SENSITIVE_DTYPE", lambda: torch.float32)

    with pytest.raises(ValueError, match="DTYPE and SENSITIVE_LAYER_DTYPE to match"):
        WanFp8VllmSharedBlockAdapter(_config(tiny_checkpoint))


def test_missing_or_incomplete_checkpoint_is_rejected(tmp_path, fp16_dtypes):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="dit_quantized_ckpt must be a directory"):
        WanFp8VllmSharedBlockAdapter(_config(missing))

    save_file(_block_tensors(0), tmp_path / "block_0.safetensors")
    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        WanFp8VllmSharedBlockAdapter(_config(tmp_path))
