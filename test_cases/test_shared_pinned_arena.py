import gc
import multiprocessing
import os
import sys

import pytest
import torch

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

from lightx2v.common.offload import shared_pinned_arena as arena_module  # noqa: E402
from lightx2v.common.offload.shared_pinned_arena import (  # noqa: E402
    CudaHostRegistration,
    CudaHostRegistrationError,
    ReplicaPlanner,
    SharedPinnedArena,
    SharedWeightManifest,
    SysVSegment,
    SysVSharedMemoryError,
    TensorSpec,
    TopologyRecord,
    build_tensor_aware_registration_regions,
    discover_gpu_numa_node,
)


def _record(rank, numa_node, *, host="host-a", ipc="ipc:[1]", signature="weights-v1"):
    return TopologyRecord(
        rank=rank,
        local_rank=rank,
        cuda_device=rank,
        pci_bus_id=f"0000:{rank + 1:02x}:00.0",
        host_id=host,
        ipc_namespace=ipc,
        numa_node=numa_node,
        weight_signature=signature,
    )


def _attach_and_mutate_in_subprocess(shmid, nbytes, connection):
    arena = None
    try:
        arena = SharedPinnedArena.attach(shmid, nbytes, register_cuda=False)
        observed = int(arena.raw[0].item())
        arena.raw[1] = 29
        connection.send(("ok", observed))
        if not connection.poll(10):
            raise TimeoutError("parent did not release shared-memory child")
        if connection.recv() != "close":
            raise RuntimeError("unexpected command from parent")
    except BaseException as error:
        try:
            connection.send(("error", repr(error)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        if arena is not None:
            arena.close(remove=False)
        connection.close()


@pytest.mark.parametrize(
    ("numa_nodes", "expected_groups"),
    [
        ([0] * 8, ((0, 1, 2, 3, 4, 5, 6, 7),)),
        ([0] * 4 + [1] * 4, ((0, 1, 2, 3), (4, 5, 6, 7))),
        ([0, 0, 1, 1, 2, 2, 3, 3], ((0, 1), (2, 3), (4, 5), (6, 7))),
    ],
)
def test_replica_planner_generalizes_over_numa_count(numa_nodes, expected_groups):
    records = [_record(rank, node) for rank, node in reversed(list(enumerate(numa_nodes)))]

    plan = ReplicaPlanner.plan(records, scope="auto")

    assert tuple(group.ranks for group in plan.groups) == expected_groups
    assert tuple(group.leader_rank for group in plan.groups) == tuple(group[0] for group in expected_groups)
    for rank in range(len(records)):
        assert rank in plan.group_for_rank(rank).ranks


def test_replica_planner_never_crosses_host_ipc_or_weight_boundaries():
    records = [
        _record(0, 0),
        _record(1, 1),
        _record(2, 0, host="host-b"),
        _record(3, 0, ipc="ipc:[2]"),
        _record(4, 0, signature="weights-v2"),
    ]

    plan = ReplicaPlanner.plan(records, scope="host")

    assert tuple(group.ranks for group in plan.groups) == ((0, 1), (2,), (3,), (4,))


def test_replica_planner_auto_falls_back_to_host_for_unknown_numa():
    records = [_record(0, 0), _record(1, None)]

    auto_plan = ReplicaPlanner.plan(records, scope="auto")

    assert len(auto_plan.groups) == 1
    assert auto_plan.groups[0].ranks == (0, 1)
    assert auto_plan.groups[0].key.numa_node is None
    with pytest.raises(ValueError, match="unknown for ranks"):
        ReplicaPlanner.plan(records, scope="numa")


def test_gpu_numa_discovery_normalizes_eight_digit_domain(tmp_path):
    device = tmp_path / "0000:18:00.0"
    device.mkdir()
    (device / "numa_node").write_text("3\n")

    assert discover_gpu_numa_node("00000000:18:00.0", sysfs_root=tmp_path) == 3
    (device / "numa_node").write_text("-1\n")
    assert discover_gpu_numa_node("0000:18:00.0", sysfs_root=tmp_path) is None
    (device / "numa_node").write_text("-2\n")
    with pytest.raises(arena_module.SharedPinnedArenaError, match="invalid NUMA node"):
        discover_gpu_numa_node("0000:18:00.0", sysfs_root=tmp_path)


def test_topology_record_rejects_invalid_negative_numa_node():
    with pytest.raises(ValueError, match="numa_node must be non-negative"):
        _record(0, -2)


def test_manifest_round_trip_and_transposed_float8_view():
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("this torch build has no float8_e4m3fn dtype")

    fp8_itemsize = torch.empty((), dtype=fp8_dtype).element_size()
    fp8_spec = TensorSpec(
        name="group.0.weight",
        offset=64,
        nbytes=6 * fp8_itemsize,
        dtype="float8_e4m3fn",
        storage_shape=(2, 3),
        shape=(3, 2),
        stride=(1, 3),
    )
    scale_spec = TensorSpec(
        name="group.0.weight_scale",
        offset=0,
        nbytes=2 * torch.empty((), dtype=torch.float32).element_size(),
        dtype="float32",
        storage_shape=(2,),
    )
    manifest = SharedWeightManifest(
        nbytes=128,
        tensors=(fp8_spec, scale_spec),
        weight_signature="checkpoint:fp8-vllm",
    )

    decoded = SharedWeightManifest.from_json(manifest.to_json())

    assert decoded == manifest
    assert decoded.digest == manifest.digest
    with SharedPinnedArena.create(manifest=manifest, register_cuda=False) as arena:
        views = arena.tensor_views()
        assert views["group.0.weight"].dtype == fp8_dtype
        assert views["group.0.weight"].shape == (3, 2)
        assert views["group.0.weight"].stride() == (1, 3)
        assert views["group.0.weight"].data_ptr() == arena.address + fp8_spec.offset
        assert views["group.0.weight_scale"].dtype == torch.float32


def test_logical_view_storage_offset_is_relative_to_tensor_region():
    itemsize = torch.empty((), dtype=torch.float32).element_size()
    spec = TensorSpec(
        name="logical",
        offset=64,
        nbytes=9 * itemsize,
        dtype="float32",
        storage_shape=(3, 3),
        shape=(2, 2),
        stride=(3, 1),
        storage_offset=4,
    )
    manifest = SharedWeightManifest(nbytes=128, tensors=(spec,), weight_signature="logical-offset")

    with SharedPinnedArena.create(manifest=manifest, register_cuda=False) as arena:
        view = arena.tensor_views()["logical"]

        assert view.data_ptr() == arena.address + spec.offset + spec.storage_offset * itemsize


def test_manifest_builder_is_deterministic():
    tensors_a = {"z": torch.empty(7, dtype=torch.uint8), "a": torch.empty((2, 3), dtype=torch.float32)}
    tensors_b = {"a": tensors_a["a"], "z": tensors_a["z"]}

    manifest_a = SharedWeightManifest.from_tensors(tensors_a, "same-weights")
    manifest_b = SharedWeightManifest.from_tensors(tensors_b, "same-weights")

    assert manifest_a == manifest_b
    assert manifest_a.digest == manifest_b.digest
    with pytest.raises(TypeError):
        manifest_a.by_name["new"] = manifest_a.tensors[0]


def test_manifest_rejects_empty_layout_and_lossy_deserialization():
    with pytest.raises(ValueError, match="at least one tensor"):
        SharedWeightManifest.from_tensors({}, "empty")

    manifest = SharedWeightManifest.from_tensors({"x": torch.empty(1)}, "strict-json")
    encoded = manifest.to_dict()
    encoded["nbytes"] = float(encoded["nbytes"])
    with pytest.raises(ValueError, match="nbytes must be positive"):
        SharedWeightManifest.from_dict(encoded)


def test_arena_cannot_reinterpret_a_different_same_size_manifest():
    first = SharedWeightManifest.from_tensors({"x": torch.empty(4, dtype=torch.uint8)}, "first")
    second = SharedWeightManifest.from_tensors({"y": torch.empty(4, dtype=torch.uint8)}, "second")

    with SharedPinnedArena.create(manifest=first, register_cuda=False) as arena:
        assert set(arena.tensor_views()) == {"x"}
        with pytest.raises(TypeError):
            arena.tensor_views(second)


def test_sysv_arena_attach_shares_physical_data_and_is_removed():
    source = {"x": torch.arange(8, dtype=torch.float32), "flag": torch.tensor([17], dtype=torch.uint8)}
    manifest = SharedWeightManifest.from_tensors(source, "sysv-test")
    creator = SharedPinnedArena.create(manifest=manifest, register_cuda=False)
    attacher = None
    shmid = creator.shmid
    try:
        assert creator.registered_nbytes == 0
        assert creator.registration_chunk_count == 0
        creator.copy_from(source)
        attacher = SharedPinnedArena.attach(shmid, manifest=manifest, register_cuda=False)

        attached_views = attacher.tensor_views()
        assert torch.equal(attached_views["x"], source["x"])
        assert attached_views["flag"].item() == 17
        assert creator.stat().attach_count >= 2

        attached_views["flag"].fill_(29)
        assert creator.tensor_views()["flag"].item() == 29

        creator.mark_for_deletion()
        assert creator._lifetime.segment.marked_for_deletion
    finally:
        if attacher is not None:
            attacher.close(remove=False)
        creator.close(remove=False)

    with pytest.raises(SysVSharedMemoryError):
        SysVSegment.stat_by_id(shmid)


@pytest.mark.skipif(sys.platform != "linux", reason="SysV shared-memory process test requires Linux")
def test_sysv_arena_is_shared_across_spawned_processes_and_cleaned_up():
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    creator = SharedPinnedArena.create(4096, register_cuda=False, auto_remove=True)
    process = context.Process(
        target=_attach_and_mutate_in_subprocess,
        args=(creator.shmid, creator.nbytes, child_connection),
    )
    shmid = creator.shmid
    creator.raw[0] = 17

    try:
        process.start()
        child_connection.close()
        assert parent_connection.poll(15), "shared-memory child did not report readiness"
        status, detail = parent_connection.recv()
        assert status == "ok", detail
        assert detail == 17
        assert creator.raw[1].item() == 29
        assert creator.stat().attach_count >= 2

        parent_connection.send("close")
        process.join(15)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent_connection.close()
        child_connection.close()
        creator.close(remove=False)

    with pytest.raises(SysVSharedMemoryError):
        SysVSegment.stat_by_id(shmid)


class _FakeCudaRuntime:
    def __init__(self, fail_registration_index=None, *, fail_sync_count=0, fail_unregister_once=None, fail_unregister_always=False):
        self.fail_registration_index = fail_registration_index
        self.fail_sync_count = fail_sync_count
        self.fail_unregister_once = set(fail_unregister_once or ())
        self.fail_unregister_always = fail_unregister_always
        self.devices = []
        self.registered = []
        self.unregistered = []
        self.synchronize_count = 0

    def set_device(self, device):
        self.devices.append(device)

    def synchronize(self):
        if self.fail_sync_count:
            self.fail_sync_count -= 1
            raise CudaHostRegistrationError("injected synchronize failure")
        self.synchronize_count += 1

    def host_register(self, address, size, flags):
        if len(self.registered) == self.fail_registration_index:
            raise CudaHostRegistrationError("injected cudaHostRegister failure")
        self.registered.append((address, size, flags))

    def host_unregister(self, address):
        if self.fail_unregister_always:
            raise CudaHostRegistrationError("injected persistent cudaHostUnregister failure")
        if address in self.fail_unregister_once:
            self.fail_unregister_once.remove(address)
            raise CudaHostRegistrationError("injected cudaHostUnregister failure")
        self.unregistered.append(address)


def test_cuda_registration_chunks_and_close_are_idempotent():
    page_size = os.sysconf("SC_PAGE_SIZE")
    runtime = _FakeCudaRuntime()
    registration = CudaHostRegistration(
        0x100000,
        page_size * 2 + 123,
        device=3,
        chunk_bytes=page_size,
        runtime=runtime,
    )
    registration.register()

    assert tuple(size for _, size in registration.chunks) == (page_size, page_size, 123)
    assert registration.registered_nbytes == page_size * 2 + 123
    assert registration.chunk_count == 3
    registration.close()
    registration.close()

    assert runtime.synchronize_count == 1
    assert runtime.unregistered == [0x102000, 0x101000, 0x100000]


def test_cuda_registration_failure_rolls_back_completed_chunks():
    page_size = os.sysconf("SC_PAGE_SIZE")
    runtime = _FakeCudaRuntime(fail_registration_index=2)

    registration = CudaHostRegistration(
        0x200000,
        page_size * 4,
        device=0,
        chunk_bytes=page_size,
        runtime=runtime,
    )
    with pytest.raises(CudaHostRegistrationError, match="injected"):
        registration.register()

    gc.collect()
    assert runtime.unregistered == [0x201000, 0x200000]


def test_arena_keeps_mapping_when_registration_and_cleanup_both_fail():
    page_size = os.sysconf("SC_PAGE_SIZE")
    runtime = _FakeCudaRuntime(fail_registration_index=1, fail_unregister_always=True)
    quarantine_size = len(arena_module._QUARANTINED_LIFETIMES)

    with pytest.raises(CudaHostRegistrationError, match="injected cudaHostRegister") as exc_info:
        SharedPinnedArena.create(
            page_size * 2,
            register_cuda=True,
            cuda_device=0,
            register_chunk_bytes=page_size,
            cuda_runtime=runtime,
            auto_remove=True,
        )

    assert any("rollback also failed" in note for note in exc_info.value.__notes__)
    assert any("arena cleanup also failed" in note for note in exc_info.value.__notes__)
    assert len(arena_module._QUARANTINED_LIFETIMES) == quarantine_size + 1
    lifetime = arena_module._QUARANTINED_LIFETIMES[-1]
    assert lifetime.segment.is_attached
    assert lifetime.registration.chunk_count == 1

    runtime.fail_unregister_always = False
    lifetime.close(remove=True, synchronize_cuda=False)
    arena_module._QUARANTINED_LIFETIMES.pop()


def test_cuda_sync_failure_keeps_all_registration_state_for_retry():
    page_size = os.sysconf("SC_PAGE_SIZE")
    runtime = _FakeCudaRuntime(fail_sync_count=1)
    registration = CudaHostRegistration(
        0x300000,
        page_size * 2,
        device=0,
        chunk_bytes=page_size,
        runtime=runtime,
    )
    registration.register()

    with pytest.raises(CudaHostRegistrationError, match="synchronize"):
        registration.close()

    assert registration.chunk_count == 2
    assert runtime.unregistered == []
    registration.close()
    assert registration.chunk_count == 0


def test_cuda_unregister_failure_retains_only_failed_chunk_for_retry():
    page_size = os.sysconf("SC_PAGE_SIZE")
    failed_address = 0x401000
    runtime = _FakeCudaRuntime(fail_unregister_once={failed_address})
    registration = CudaHostRegistration(
        0x400000,
        page_size * 3,
        device=0,
        chunk_bytes=page_size,
        runtime=runtime,
    )
    registration.register()

    with pytest.raises(CudaHostRegistrationError, match="unregister"):
        registration.close()

    assert registration.chunks == ((failed_address, page_size),)
    assert runtime.unregistered == [0x402000, 0x400000]
    registration.close()
    assert runtime.unregistered[-1] == failed_address
    assert registration.chunk_count == 0


def test_arena_cuda_registration_uses_injected_runtime():
    runtime = _FakeCudaRuntime()
    arena = SharedPinnedArena.create(
        os.sysconf("SC_PAGE_SIZE") + 7,
        register_cuda=True,
        cuda_device=2,
        register_chunk_bytes=os.sysconf("SC_PAGE_SIZE"),
        cuda_runtime=runtime,
    )
    shmid = arena.shmid

    assert arena.is_cuda_registered
    assert arena.registered_nbytes == os.sysconf("SC_PAGE_SIZE") + 7
    assert arena.registration_chunk_count == 2
    arena.raw.fill_(11)
    arena.close()
    arena.close()

    assert arena.registered_nbytes == 0
    assert arena.registration_chunk_count == 0
    assert runtime.synchronize_count == 1
    assert runtime.unregistered
    with pytest.raises(SysVSharedMemoryError):
        SysVSegment.stat_by_id(shmid)


def test_arena_close_does_not_detach_when_cuda_sync_fails():
    page_size = os.sysconf("SC_PAGE_SIZE")
    runtime = _FakeCudaRuntime(fail_sync_count=1)
    arena = SharedPinnedArena.create(
        page_size,
        register_cuda=True,
        cuda_device=0,
        register_chunk_bytes=page_size,
        cuda_runtime=runtime,
        auto_remove=True,
    )
    shmid = arena.shmid
    address = arena.address

    with pytest.raises(CudaHostRegistrationError, match="synchronize"):
        arena.close()

    assert arena.address == address
    assert arena.raw.numel() == page_size
    assert arena.registered_nbytes == page_size
    assert arena.stat().attach_count >= 1

    arena.close()
    with pytest.raises(SysVSharedMemoryError):
        SysVSegment.stat_by_id(shmid)


def test_arena_close_does_not_detach_when_cuda_unregister_fails():
    page_size = os.sysconf("SC_PAGE_SIZE")
    runtime = _FakeCudaRuntime()
    arena = SharedPinnedArena.create(
        page_size * 2,
        register_cuda=True,
        cuda_device=0,
        register_chunk_bytes=page_size,
        cuda_runtime=runtime,
        auto_remove=True,
    )
    shmid = arena.shmid
    failed_address = runtime.registered[0][0]
    runtime.fail_unregister_once.add(failed_address)

    with pytest.raises(CudaHostRegistrationError, match="unregister"):
        arena.close()

    assert arena.registered_nbytes == page_size
    assert arena.registration_chunk_count == 1
    assert arena.stat().attach_count >= 1
    arena.close()
    with pytest.raises(SysVSharedMemoryError):
        SysVSegment.stat_by_id(shmid)


def test_auto_removed_sysv_segment_remains_attachable_until_last_detach():
    creator = SharedPinnedArena.create(4096, register_cuda=False, auto_remove=True)
    shmid = creator.shmid
    attacher = None
    try:
        assert creator._lifetime.segment.marked_for_deletion
        attacher = SharedPinnedArena.attach(shmid, 4096, register_cuda=False)
        creator.raw[0] = 73
        assert attacher.raw[0].item() == 73
    finally:
        if attacher is not None:
            attacher.close(remove=False)
        creator.close(remove=False)

    with pytest.raises(SysVSharedMemoryError):
        SysVSegment.stat_by_id(shmid)


def test_sysv_detach_failure_keeps_address_for_retry(monkeypatch):
    class FakeLibc:
        def __init__(self):
            self.calls = 0

        def shmdt(self, address):
            self.calls += 1
            if self.calls == 1:
                return -1
            return 0

    libc = FakeLibc()
    segment = SysVSegment(123, 4096, 0x500000, creator=False)
    monkeypatch.setattr(arena_module, "_load_libc", lambda: libc)

    with pytest.raises(SysVSharedMemoryError, match="shmdt"):
        segment.detach()

    assert segment.is_attached
    assert segment.address == 0x500000
    segment.detach()
    assert not segment.is_attached
    assert segment.address == 0


def test_sysv_rmid_failure_does_not_detach_creator(monkeypatch):
    class FakeLibc:
        def __init__(self):
            self.rmid_calls = 0
            self.detach_calls = 0

        def shmctl(self, shmid, command, value):
            self.rmid_calls += 1
            if self.rmid_calls == 1:
                return -1
            return 0

        def shmdt(self, address):
            self.detach_calls += 1
            return 0

    libc = FakeLibc()
    segment = SysVSegment(456, 4096, 0x600000, creator=True)
    monkeypatch.setattr(arena_module, "_load_libc", lambda: libc)

    with pytest.raises(SysVSharedMemoryError, match="IPC_RMID"):
        segment.close()

    assert segment.is_attached
    assert libc.detach_calls == 0
    segment.close()
    assert not segment.is_attached
    assert libc.detach_calls == 1


def test_registration_regions_do_not_split_tensor_at_128_mib_boundary():
    mib = 1024 * 1024
    tensor_start = 120 * mib + 64
    tensor_size = 20 * mib
    manifest = SharedWeightManifest(
        nbytes=260 * mib,
        tensors=(
            TensorSpec(
                name="large.weight",
                offset=tensor_start,
                nbytes=tensor_size,
                dtype="uint8",
                storage_shape=(tensor_size,),
            ),
        ),
        weight_signature="large-weight-boundary-test",
    )
    relative_regions = build_tensor_aware_registration_regions(
        manifest.nbytes,
        manifest.tensors,
        target_chunk_bytes=128 * mib,
    )
    internal_boundaries = [offset + size for offset, size in relative_regions[:-1]]

    assert sum(size for _, size in relative_regions) == manifest.nbytes
    assert relative_regions[0][0] == 0
    assert all(next_offset == offset + size for (offset, size), (next_offset, _) in zip(relative_regions, relative_regions[1:]))
    assert 128 * mib not in internal_boundaries
    assert not any(tensor_start < boundary < tensor_start + tensor_size for boundary in internal_boundaries)
    containing_regions = [
        (offset, size)
        for offset, size in relative_regions
        if offset <= tensor_start and tensor_start + tensor_size <= offset + size
    ]
    assert len(containing_regions) == 1
