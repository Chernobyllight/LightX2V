import gc
import importlib.util
import os
import sys
import types
import weakref
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_source_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, _REPOSITORY_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Importing lightx2v.common.ops normally imports every optional GPU attention
# backend.  Load the three units under test with only their direct dependencies
# so this remains a small, deterministic CPU-only test.
shared_weight_map = _load_source_module(
    "_shared_weight_map_under_test",
    "lightx2v/common/offload/shared_weight_map.py",
)

fake_envs = types.ModuleType("lightx2v.utils.envs")
fake_envs.GET_DTYPE = lambda: torch.float16
fake_envs.GET_SENSITIVE_DTYPE = lambda: torch.float32
fake_global_var = types.ModuleType("lightx2v_platform.base.global_var")
fake_global_var.AI_DEVICE = "cuda"
fake_registry = types.ModuleType("lightx2v.utils.registry_factory")
fake_registry.TENSOR_REGISTER = lambda _name: lambda cls: cls
fake_registry.EMBEDDING_WEIGHT_REGISTER = lambda _name: lambda cls: cls

with patch.dict(
    sys.modules,
    {
        "lightx2v.common.offload.shared_weight_map": shared_weight_map,
        "lightx2v.utils.envs": fake_envs,
        "lightx2v.utils.registry_factory": fake_registry,
        "lightx2v_platform.base.global_var": fake_global_var,
    },
):
    ops_utils = _load_source_module(
        "_ops_utils_under_test",
        "lightx2v/common/ops/utils.py",
    )
    tensor_module = _load_source_module(
        "_default_tensor_under_test",
        "lightx2v/common/ops/tensor/tensor.py",
    )
    embedding_module = _load_source_module(
        "_embedding_weight_under_test",
        "lightx2v/common/ops/embedding/embedding_weight.py",
    )

SharedWeightViewMap = shared_weight_map.SharedWeightViewMap
create_default_tensors = ops_utils.create_default_tensors
DefaultTensor = tensor_module.DefaultTensor
EmbeddingWeightTemplate = embedding_module.EmbeddingWeightTemplate


class _Owner:
    pass


def test_shared_entries_remain_available_to_multiple_consumers():
    owner = _Owner()
    shared = torch.arange(6, dtype=torch.float32).view(2, 3)
    weights = SharedWeightViewMap({"private": torch.tensor(1)}, {"shared": shared}, owner=owner)

    first = weights.take("shared")
    second = weights["shared"]

    assert first is shared
    assert second is shared
    assert weights.is_shared("shared")
    assert weights.consumed_shared_keys == frozenset({"shared"})
    assert weights.take("private").item() == 1
    assert "private" not in weights
    assert not hasattr(weights, "pop")


def test_shared_mapping_rejects_missing_owner_and_key_overlap():
    tensor = torch.tensor(1)
    with pytest.raises(ValueError, match="owner is required"):
        SharedWeightViewMap(shared={"weight": tensor})
    with pytest.raises(ValueError, match="overlap"):
        SharedWeightViewMap({"weight": tensor}, {"weight": tensor}, owner=_Owner())
    with pytest.raises(ValueError, match="non-empty"):
        SharedWeightViewMap(owner=_Owner())


def test_mapping_retains_owner_until_mapping_is_released():
    owner = _Owner()
    owner_ref = weakref.ref(owner)
    weights = SharedWeightViewMap(shared={"weight": torch.tensor(1)}, owner=owner)
    del owner
    gc.collect()

    assert owner_ref() is weights.owner

    del weights
    gc.collect()
    assert owner_ref() is None


def test_create_default_tensors_adopts_shared_view_without_copy():
    owner = _Owner()
    source = torch.arange(6, dtype=torch.float32).view(2, 3)
    weights = SharedWeightViewMap(shared={"block.0.weight": source}, owner=owner)

    device_tensors, pin_tensors = create_default_tensors(
        [("block.0.weight", "weight", True)],
        weights,
    )

    adopted = pin_tensors["weight"]
    assert not device_tensors
    assert adopted.shape == (3, 2)
    assert adopted.stride() == (1, 3)
    assert adopted.data_ptr() == source.data_ptr()
    assert weights["block.0.weight"] is source
    assert weights.consumed_shared_keys == frozenset({"block.0.weight"})


def test_create_default_tensors_keeps_plain_dict_consumption(monkeypatch):
    source = torch.arange(6, dtype=torch.float32).view(2, 3)
    weights = {"block.0.weight": source}
    allocated = []

    def _fake_create_pin_tensor(tensor, transpose=False, dtype=None):
        result = tensor.clone()
        allocated.append(result)
        return result.t() if transpose else result

    monkeypatch.setattr(ops_utils, "create_pin_tensor", _fake_create_pin_tensor)
    _, pin_tensors = create_default_tensors(
        [("block.0.weight", "weight", True)],
        weights,
    )

    assert "block.0.weight" not in weights
    assert pin_tensors["weight"].data_ptr() != source.data_ptr()
    assert allocated


def test_default_tensor_adopts_shared_view_and_keeps_it_reusable():
    owner = _Owner()
    source = torch.arange(4, dtype=torch.float32)
    weights = SharedWeightViewMap(shared={"blocks.0.modulation": source}, owner=owner)
    first = DefaultTensor("blocks.0.modulation")
    second = DefaultTensor("blocks.0.modulation")

    first.load(weights)
    second.load(weights)

    assert first.pin_tensor is source
    assert second.pin_tensor is source
    assert first.pin_tensor.data_ptr() == source.data_ptr()
    assert weights["blocks.0.modulation"] is source
    assert weights.consumed_shared_keys == frozenset({"blocks.0.modulation"})


def test_embedding_adopts_shared_view_without_copy():
    source = torch.arange(12, dtype=torch.float32).view(4, 3)
    weights = SharedWeightViewMap(shared={"blocks.0.embedding": source}, owner=_Owner())
    embedding = EmbeddingWeightTemplate("blocks.0.embedding")

    embedding.load(weights)

    assert embedding.pin_weight is source
    assert weights["blocks.0.embedding"] is source
    assert weights.consumed_shared_keys == frozenset({"blocks.0.embedding"})


def test_embedding_keeps_plain_dict_copy_and_consumption(monkeypatch):
    source = torch.arange(12, dtype=torch.float32).view(4, 3)
    weights = {"blocks.0.embedding": source}
    embedding = EmbeddingWeightTemplate("blocks.0.embedding")
    copied = source.clone()
    monkeypatch.setattr(embedding, "_create_cpu_pin_weight", lambda tensor: copied)

    embedding.load(weights)

    assert embedding.pin_weight is copied
    assert "blocks.0.embedding" not in weights
