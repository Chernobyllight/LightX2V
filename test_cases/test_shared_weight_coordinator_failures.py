import os

import pytest
import torch

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

from lightx2v.common.offload import shared_weight_coordinator as coordinator  # noqa: E402
from lightx2v.common.offload.shared_pinned_arena import (  # noqa: E402
    ReplicaGroup,
    ReplicaKey,
    ReplicaPlan,
    SharedWeightManifest,
    TopologyRecord,
)


def _record(rank):
    return TopologyRecord(
        rank=rank,
        host_id="host-a",
        ipc_namespace="ipc:[1]",
        numa_node=0,
        weight_signature="weights-v1",
        local_rank=rank,
        cuda_device=rank,
        pci_bus_id=f"0000:{rank + 1:02x}:00.0",
    )


def _plan():
    key = ReplicaKey(host_id="host-a", ipc_namespace="ipc:[1]", weight_signature="weights-v1", numa_node=0)
    return ReplicaPlan(groups=(ReplicaGroup(key=key, ranks=(0, 1)),))


def _patch_rank_one_preflight(monkeypatch, manifest):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(coordinator, "_distributed_rank_and_world", lambda: (1, 2))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(coordinator.TopologyRecord, "discover", classmethod(lambda cls, *args, **kwargs: _record(1)))
    monkeypatch.setattr(coordinator.ReplicaPlanner, "plan", classmethod(lambda cls, records, scope: _plan()))

    def peer_discovery(local_status):
        peer = dict(local_status)
        peer.update(rank=0, topology=_record(0), manifest_digest=manifest.digest, manifest_nbytes=manifest.nbytes)
        return peer

    return peer_discovery


def test_coordinate_rank_local_error_reraises_original_without_distributed_runtime(monkeypatch):
    error = ValueError("local failure")
    monkeypatch.setattr(coordinator, "_distributed_rank_and_world", lambda: (0, 1))

    with pytest.raises(ValueError, match="local failure") as exc_info:
        coordinator.coordinate_rank_local_error("test stage", error)

    assert exc_info.value is error


def test_coordinate_rank_local_error_propagates_peer_failure(monkeypatch):
    monkeypatch.setattr(coordinator, "_distributed_rank_and_world", lambda: (1, 2))
    monkeypatch.setattr(
        coordinator,
        "_all_gather_object",
        lambda status, world_size: [
            {"rank": 0, "ok": False, "error": "RuntimeError: peer failed"},
            status,
        ],
    )

    with pytest.raises(coordinator.SharedWeightCoordinationError, match="rank 0: RuntimeError: peer failed"):
        coordinator.coordinate_rank_local_error("model initialization", None)


def test_invalid_local_input_is_reported_through_first_collective(monkeypatch):
    manifest = SharedWeightManifest.from_tensors({"value": torch.empty(1)}, "weights-v1")
    monkeypatch.setattr(coordinator, "_distributed_rank_and_world", lambda: (1, 2))
    gathered = []

    def fake_all_gather(value, world_size):
        assert world_size == 2
        gathered.append(value)
        return [{"rank": 0, "ok": True}, value]

    monkeypatch.setattr(coordinator, "_all_gather_object", fake_all_gather)

    with pytest.raises(coordinator.SharedWeightCoordinationError, match="populate must be callable"):
        coordinator.materialize_shared_weight_arena(manifest, None)

    assert len(gathered) == 1
    assert gathered[0]["ok"] is False


def test_missing_group_descriptor_is_reported_through_attach_collective(monkeypatch):
    manifest = SharedWeightManifest.from_tensors({"value": torch.empty(1)}, "weights-v1")
    peer_discovery = _patch_rank_one_preflight(monkeypatch, manifest)
    gathered_values = []

    def fake_all_gather(value, world_size):
        assert world_size == 2
        gathered_values.append(value)
        stage = len(gathered_values)
        if stage == 1:
            return [peer_discovery(value), value]
        if stage == 2:
            return [{"rank": 0, "ok": True, "descriptor": None}, value]
        if stage == 3:
            assert not value["ok"]
            assert "no arena descriptor" in value["error"]
            return [{"rank": 0, "ok": False, "error": "missing descriptor"}, value]
        raise AssertionError(f"unexpected collective stage {stage}")

    monkeypatch.setattr(coordinator, "_all_gather_object", fake_all_gather)

    with pytest.raises(coordinator.SharedWeightCoordinationError, match="attach/CUDA registration"):
        coordinator.materialize_shared_weight_arena(manifest, lambda views: None)

    assert len(gathered_values) == 3


def test_collective_exception_after_creation_closes_local_leader_arena(monkeypatch):
    manifest = SharedWeightManifest.from_tensors({"value": torch.empty(1)}, "weights-v1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(coordinator, "_distributed_rank_and_world", lambda: (0, 2))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(coordinator.TopologyRecord, "discover", classmethod(lambda cls, *args, **kwargs: _record(0)))
    monkeypatch.setattr(coordinator.ReplicaPlanner, "plan", classmethod(lambda cls, records, scope: _plan()))

    class FakeArena:
        shmid = 77
        nbytes = manifest.nbytes

        def __init__(self):
            self.close_calls = []

        def tensor_views(self):
            return {"value": torch.empty(1)}

        def close(self, *, remove, synchronize_cuda=True):
            self.close_calls.append((remove, synchronize_cuda))

    arena = FakeArena()
    create_kwargs = {}

    def fake_create(**kwargs):
        create_kwargs.update(kwargs)
        return arena

    monkeypatch.setattr(coordinator.SharedPinnedArena, "create", fake_create)
    collective_count = 0

    def fake_all_gather(value, world_size):
        nonlocal collective_count
        collective_count += 1
        if collective_count == 1:
            peer = dict(value)
            peer.update(rank=1, topology=_record(1))
            return [value, peer]
        if collective_count == 2:
            raise RuntimeError("injected collective failure")
        raise AssertionError("unexpected collective")

    monkeypatch.setattr(coordinator, "_all_gather_object", fake_all_gather)

    with pytest.raises(RuntimeError, match="injected collective failure"):
        coordinator.materialize_shared_weight_arena(manifest, lambda views: None)

    assert create_kwargs["auto_remove"] is True
    assert arena.close_calls == [(True, False)]
