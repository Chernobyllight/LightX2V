import os

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")

from lightx2v.common.offload.shared_pinned_arena import ReplicaPlanner, TopologyRecord  # noqa: E402


def _record(rank, host, numa, *, local_rank, cuda_device):
    return TopologyRecord(
        rank=rank,
        host_id=host,
        ipc_namespace=f"ipc:{host}",
        numa_node=numa,
        weight_signature="same-weights",
        local_rank=local_rank,
        cuda_device=cuda_device,
        pci_bus_id=f"0000:{cuda_device + 1:02x}:00.0",
    )


def test_auto_scope_is_per_host_and_independent_of_rank_device_order():
    records = [
        _record(9, "host-b", None, local_rank=0, cuda_device=6),
        _record(5, "host-a", 0, local_rank=3, cuda_device=7),
        _record(2, "host-b", 1, local_rank=1, cuda_device=2),
        _record(7, "host-a", 1, local_rank=2, cuda_device=0),
        _record(1, "host-a", 0, local_rank=0, cuda_device=5),
    ]

    plan = ReplicaPlanner.plan(records, scope="auto")

    # host-a has complete NUMA data and splits by memory domain.  host-b has
    # an unknown NUMA node, so only that host falls back to host scope.  The
    # deliberately shuffled ranks/local ranks/CUDA ordinals do not affect it.
    assert tuple(group.ranks for group in plan.groups) == ((1, 5), (2, 9), (7,))
    assert tuple(group.leader_rank for group in plan.groups) == (1, 2, 7)
