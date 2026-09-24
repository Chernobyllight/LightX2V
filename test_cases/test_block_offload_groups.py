"""Exercise model-independent block loading and the actual offload scheduler."""

import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from lightx2v.common.offload.block_loader import prepare_contiguous_groups, validate_contiguous_config, validate_group_checkpoints
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.common.ops.tensor.tensor import DefaultTensor
from lightx2v.models.networks.qwen_image.weights.transformer_weights import QwenImageTransformerWeights
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerWeights
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.ops.weight_storage import TensorMetadata

device_module = getattr(torch, AI_DEVICE)
pytestmark = pytest.mark.skipif(not device_module.is_available(), reason="requires an accelerator")


@pytest.fixture(autouse=True)
def prepare_platform():
    from lightx2v_platform.base.offload import get_block_offload_backend

    get_block_offload_backend().prepare()


def config(scheme="Default"):
    attention = "npu_flash_attn" if AI_DEVICE == "npu" else "torch_sdpa"
    return dict(
        model_cls="wan2.1",
        task="i2v",
        num_layers=3,
        num_attention_heads=4,
        cpu_offload=True,
        offload_granularity="block",
        seq_parallel=False,
        dit_quantized=scheme != "Default",
        dit_quant_scheme=scheme,
        self_attn_1_type=attention,
        cross_attn_1_type=attention,
        cross_attn_2_type=attention,
        attn_type=attention,
        rms_norm_type="torch",
        layer_norm_type="torch",
        cpu_offload_layout="contiguous",
    )


def weights_for(owner):
    weights = {}
    for _, leaf in owner.named_weight_leaves():
        for name, attr, transpose in getattr(leaf, "base_attrs", ()):
            index = int(re.search(r"\.(\d+)\.", name)[1]) if re.search(r"\.(\d+)\.", name) else 0
            if attr == "weight" and (transpose or hasattr(leaf, "act_quant_func")):
                dtype = torch.bfloat16
                if hasattr(leaf, "act_quant_func"):
                    dtype = torch.int8 if "Npu" in type(leaf).__name__ else torch.float8_e4m3fn
                value = (torch.arange(32 * 32).reshape(32, 32) % 7 + index).to(dtype)
            elif attr == "weight_scale":
                value = torch.full((32, 1), 0.125 + index / 32, dtype=torch.bfloat16)
            else:
                value = torch.linspace(0.5, 1.5, 32, dtype=torch.bfloat16) + index
            weights[name] = value
        if isinstance(leaf, DefaultTensor):
            weights[leaf.tensor_name] = torch.ones((1, 6, 32), dtype=torch.bfloat16)
    return weights


def state(block):
    return {re.sub(r"\.\d+\.", ".0.", name, count=1): value for name, value in block.state_dict().items() if value is not None}


def assert_same(actual, expected):
    a, b = state(actual), state(expected)
    assert a.keys() == b.keys()
    for name in a:
        x, y = a[name], b[name]
        assert (x.dtype, x.shape, x.stride()) == (y.dtype, y.shape, y.stride()), name
        torch.testing.assert_close(x.reshape(-1).view(torch.uint8).cpu(), y.reshape(-1).view(torch.uint8).cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("factory", [WanTransformerWeights, QwenImageTransformerWeights])
@pytest.mark.parametrize("scheme", ["Default", "fp8-vllm"])
def test_groups_match_baseline_and_reuse_device_views(factory, scheme):
    if AI_DEVICE != "cuda" and scheme == "fp8-vllm":
        pytest.skip("FP8-vLLM requires CUDA")
    options = config(scheme)
    if factory is QwenImageTransformerWeights:
        options["model_cls"] = "qwen_image"
    baseline = factory(options)
    baseline.load(weights_for(baseline))
    model = factory(options)
    weights = weights_for(model)
    groups = tuple(model.iter_offload_groups())
    prepare_contiguous_groups(groups, weights)
    model.load(weights)
    assert not weights
    for actual, expected in zip(model.blocks, baseline.blocks):
        assert_same(actual, expected)
        storage = actual.block_buffer.storage
        for tensor in state(actual).values():
            if tensor.device.type == "cpu":
                assert tensor.is_pinned()
                assert tensor.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
                assert tensor.data_ptr() % 256 == 0

    manager = WeightAsyncStreamManager("block")
    manager.init_contiguous_groups(groups)
    manager.init_first_buffer(model.blocks)
    versions = [block.block_buffer.storage._version for block in model.blocks]
    reference = baseline.offload_block_cuda_buffers[0]
    for index in (0, 1, 2, 0, 1, 2):
        reference.load_state_dict(baseline.blocks[index].state_dict(), index)
        device_module.synchronize()
        assert_same(manager.cuda_buffers[0], reference)
        phase = manager.cuda_buffers[0].compute_phases[0]
        ref_phase = reference.compute_phases[0]
        projection = "self_attn_q" if factory is WanTransformerWeights else "to_q"
        x = torch.arange(64, device=AI_DEVICE).to(torch.bfloat16).reshape(2, 32) / 32
        with torch.no_grad(), device_module.stream(manager.compute_stream):
            actual = getattr(phase, projection).apply(x)
            expected = getattr(ref_phase, projection).apply(x)
        manager.prefetch_weights((index + 1) % 3, model.blocks)
        manager.swap_blocks()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert versions == [block.block_buffer.storage._version for block in model.blocks]
    manager.contiguous_transfer.close()


@pytest.mark.parametrize("factory", [WanTransformerWeights, QwenImageTransformerWeights])
def test_per_tensor_manager_preserves_block_order(factory):
    options = config()
    options.pop("cpu_offload_layout")
    model = factory(options)
    model.load(weights_for(model))
    manager = WeightAsyncStreamManager("block")
    manager.init_cuda_buffer(blocks_cuda_buffer=model.offload_block_cuda_buffers)
    manager.init_first_buffer(model.blocks)
    for index in (0, 1, 2, 0):
        device_module.synchronize()
        assert_same(manager.cuda_buffers[0], model.blocks[index])
        manager.prefetch_weights((index + 1) % len(model.blocks), model.blocks)
        manager.swap_blocks()


def test_different_model_groups_can_share_a_manager():
    models = [WanTransformerWeights(config()), QwenImageTransformerWeights(config())]
    groups = [next(model.iter_offload_groups()) for model in models]
    for model, group in zip(models, groups):
        weights = weights_for(model)
        prepare_contiguous_groups([group], weights)
        model.load(weights)
    manager = WeightAsyncStreamManager("block")
    manager.init_contiguous_groups(groups)
    for model in (models[0], models[1], models[0]):
        manager.init_first_buffer(model.blocks)
        device_module.synchronize()
        assert_same(manager.cuda_buffers[0], model.blocks[0])
        manager.prefetch_weights(1, model.blocks)
        manager.swap_blocks()
        assert_same(manager.cuda_buffers[0], model.blocks[1])
    transfers = [transfer for _, transfer in manager.contiguous_groups.values()]
    del manager
    assert all(transfer.closed for transfer in transfers)


@pytest.mark.parametrize("problem", ["extra", "missing", "shape", "precision", "unknown_operator", "overlap"])
def test_invalid_groups_fail_before_consuming_weights(problem):
    model = WanTransformerWeights(config())
    weights = weights_for(model)
    groups = tuple(model.iter_offload_groups())
    if problem == "extra":
        weights["blocks.0.unmapped"] = torch.ones(1)
    elif problem == "missing":
        del weights["blocks.1.self_attn.q.weight"]
    elif problem == "shape":
        weights["blocks.1.self_attn.q.weight"] = torch.ones((16, 32), dtype=torch.bfloat16)
    elif problem == "precision":
        weights["blocks.0.self_attn.q.weight"] = weights["blocks.0.self_attn.q.weight"].to(torch.float8_e4m3fn)
    elif problem == "unknown_operator":
        model.blocks[0].add_module("custom_weight", SimpleNamespace(state_dict=lambda: {"custom": torch.ones(1)}))
    else:
        groups += groups
    keys = set(weights)
    with pytest.raises((ValueError, KeyError)):
        prepare_contiguous_groups(groups, weights)
    assert set(weights) == keys
    assert all(not hasattr(block, "block_buffer") for block in model.blocks)


def test_raw_dtype_validation_catches_prior_casts():
    model = WanTransformerWeights(config())
    weights = weights_for(model)
    metadata = {name: TensorMetadata(tuple(tensor.shape), tensor.dtype) for name, tensor in weights.items()}
    name = "blocks.0.self_attn.q.weight"
    metadata[name] = TensorMetadata(tuple(weights[name].shape), torch.float8_e4m3fn)
    assert weights[name].dtype == torch.bfloat16
    with pytest.raises(ValueError, match="checkpoint dtype"):
        prepare_contiguous_groups(model.iter_offload_groups(), weights, metadata)
    with pytest.raises(ValueError, match="checkpoint dtype"):
        validate_group_checkpoints(model.iter_offload_groups(), metadata)


@pytest.mark.parametrize("extension", ["safetensors", "pth"])
def test_checkpoint_reader_keeps_original_dtype_before_casting(tmp_path, extension):
    from safetensors.torch import save_file

    from lightx2v.models.networks.wan.model import WanModel

    model = object.__new__(WanModel)
    model.config = {}
    model.device = torch.device("cpu")
    model._checkpoint_metadata = {}
    checkpoint = tmp_path / f"weights.{extension}"
    source = {"unrelated_model.layer.q.weight": torch.zeros((32, 32)).to(torch.float8_e4m3fn)}
    if extension == "safetensors":
        save_file(source, checkpoint)
    else:
        torch.save(source, checkpoint)
    weights = model._load_safetensor_to_dict(str(checkpoint), True, ())
    name = next(iter(source))
    assert weights[name].dtype == torch.bfloat16
    assert model._checkpoint_metadata[name].dtype == torch.float8_e4m3fn
    assert model._checkpoint_metadata[name].shape == (32, 32)


def test_npu_format_setup_and_strided_host_copy(monkeypatch):
    from lightx2v_platform.base.ascend_npu import NpuBlockOffload, NpuDevice

    if AI_DEVICE != "npu":
        npu = SimpleNamespace(config=SimpleNamespace(allow_internal_format=True))
        monkeypatch.setattr(torch, "npu", npu, raising=False)
    NpuBlockOffload.prepare()
    if AI_DEVICE != "npu":
        assert torch.npu.config.allow_internal_format is False
    parent = torch.full((4, 8), -1, dtype=torch.bfloat16).pin_memory()
    target = parent[:, ::2].t()
    source = torch.arange(16, dtype=torch.bfloat16, device=AI_DEVICE).reshape(4, 4)
    pointer = target.data_ptr()
    NpuDevice.copy_to_cpu(target, source, non_blocking=True)
    torch.testing.assert_close(target, source.cpu(), rtol=0, atol=0)
    assert target.data_ptr() == pointer
    assert target.is_pinned()
    assert torch.all(parent[:, 1::2] == -1)


def test_bind_cannot_escape_planned_storage(monkeypatch):
    from lightx2v.common.ops.mm.mm_weight import MMWeight

    original = MMWeight.load

    def escaping(self, context):
        original(self, context)
        if not self.create_cuda_buffer:
            self.pin_weight = self.pin_weight.clone()

    monkeypatch.setattr(MMWeight, "load", escaping)
    model = WanTransformerWeights(config())
    weights = weights_for(model)
    prepare_contiguous_groups(model.iter_offload_groups(), weights)
    with pytest.raises(ValueError, match="escaped"):
        model.load(weights)


def test_stateless_operator_initialization_is_preserved():
    initialized = []
    model = WanTransformerWeights(config())
    model.blocks[0].add_module("setup", SimpleNamespace(state_dict=lambda destination=None: {}, load=lambda context: initialized.append(True)))
    weights = weights_for(model)
    prepare_contiguous_groups(model.iter_offload_groups(), weights)
    model.load(weights)
    assert initialized == [True]


@pytest.mark.parametrize("key", ["shared_cpu_weights", "lazy_load", "seq_parallel", "lora_dynamic_apply", "enable_cuda_graph"])
def test_unsupported_execution_modes_remain_rejected(key):
    options = config()
    options[key] = True
    with pytest.raises(ValueError):
        validate_contiguous_config(options)


def test_qwen_torch_modulation_does_not_load_triton_kernels():
    code = """
import importlib.abc
import sys
import torch
class RejectQwenTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lightx2v.models.networks.qwen_image.infer.triton_ops":
            raise RuntimeError(f"Torch modulation imported {fullname}")
sys.meta_path.insert(0, RejectQwenTriton())
from lightx2v.models.networks.qwen_image.infer.transformer_infer import QwenImageTransformerInfer
infer = QwenImageTransformerInfer({"seq_parallel": False, "modulate_type": "torch"})
x = torch.randn(1, 5, 8)
params = torch.randn(2, 24)
out, gate = infer._modulate(x, params[:1])
shift, scale, expected_gate = params[:1].chunk(3, dim=-1)
torch.testing.assert_close(out, (x * (1 + scale) + shift).squeeze(0))
torch.testing.assert_close(gate, expected_gate)
index = torch.tensor([[0, 1, 0, 1, 1]])
out, gate = infer._modulate(x, params, index)
selected = params[index.squeeze(0)]
shift, scale, expected_gate = selected.chunk(3, dim=-1)
torch.testing.assert_close(out, x.squeeze(0) * (1 + scale) + shift)
torch.testing.assert_close(gate, expected_gate)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cuda_import_does_not_load_ascend_operators():
    if AI_DEVICE != "cuda":
        pytest.skip("CUDA import isolation")
    code = """
import importlib.abc
import sys
class RejectAscend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("lightx2v_platform.ops.") and ".ascend_npu" in fullname:
            raise RuntimeError(f"CUDA imported {fullname}")
sys.meta_path.insert(0, RejectAscend())
from lightx2v.common.offload.block_loader import prepare_contiguous_groups
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerWeights
from lightx2v.models.networks.qwen_image.weights.transformer_weights import QwenImageTransformerWeights
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PLATFORM="cuda"))
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("scheme", ["Default", "int8-npu"])
def test_ascend_operator_storage_contracts(monkeypatch, scheme):
    from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER
    from lightx2v_platform.ops.mm.ascend_npu.mm_weight import MMWeightWint8channelAint8channeldynamicNpu
    from lightx2v_platform.ops.norm.ascend_npu.npu_layer_norm import NpuLayerNormWeight
    from lightx2v_platform.ops.norm.ascend_npu.npu_rms_norm import NpuRmsNormWeight

    monkeypatch.setitem(MM_WEIGHT_REGISTER, "int8-npu", MMWeightWint8channelAint8channeldynamicNpu)
    monkeypatch.setitem(LN_WEIGHT_REGISTER, "npu_layer_norm", NpuLayerNormWeight)
    monkeypatch.setitem(RMS_WEIGHT_REGISTER, "npu_rms_norm", NpuRmsNormWeight)
    options = dict(config(scheme), rms_norm_type="npu_rms_norm", layer_norm_type="npu_layer_norm")
    model = WanTransformerWeights(options)
    baseline = WanTransformerWeights(options)
    baseline.load(weights_for(baseline))
    weights = weights_for(model)
    prepare_contiguous_groups(model.iter_offload_groups(), weights)
    model.load(weights)
    for actual, expected in zip(model.blocks, baseline.blocks):
        assert_same(actual, expected)
    manager = WeightAsyncStreamManager("block")
    manager.init_contiguous_groups(model.iter_offload_groups())
    manager.init_first_buffer(model.blocks)
    manager.prefetch_weights(2, model.blocks)
    manager.swap_blocks()
    assert_same(manager.cuda_buffers[0], model.blocks[2])
    if AI_DEVICE == "npu":
        reference = baseline.offload_block_cuda_buffers[0]
        reference.load_state_dict(baseline.blocks[2].state_dict(), 2)
        x = torch.ones((2, 32), device=AI_DEVICE, dtype=torch.bfloat16)
        torch.testing.assert_close(manager.cuda_buffers[0].compute_phases[0].self_attn_q.apply(x), reference.compute_phases[0].self_attn_q.apply(x), rtol=0, atol=0)
    manager.contiguous_transfer.close()
