import pytest

from tools.benchmark.shared_arena_cuda_smoke import _validate_statuses


def _status(rank, numa_node, group_ranks, leader_rank, shmid, populate_calls):
    return {
        "rank": rank,
        "host_id": "host-a",
        "ipc_namespace": "ipc:[1]",
        "numa_node": numa_node,
        "group_ranks": list(group_ranks),
        "leader_rank": leader_rank,
        "shmid": shmid,
        "arena_bytes": 4096,
        "registered_bytes": 4096,
        "manifest_hash": "manifest-a",
        "populate_calls": populate_calls,
        "h2d_ok": True,
    }


def test_auto_scope_validates_one_segment_and_populator_per_numa_group():
    statuses = [
        _status(0, 0, (0, 2), 0, 10, 1),
        _status(1, 1, (1, 3), 1, 11, 1),
        _status(2, 0, (0, 2), 0, 10, 0),
        _status(3, 1, (1, 3), 1, 11, 0),
    ]

    groups = _validate_statuses(statuses, "auto", world_size=4)

    assert [group["ranks"] for group in groups] == [[0, 2], [1, 3]]
    assert [group["shmid"] for group in groups] == [10, 11]


def test_host_scope_requires_one_group_even_when_gpu_numa_nodes_differ():
    statuses = [
        _status(0, 0, (0, 1), 0, 10, 1),
        _status(1, 1, (0, 1), 0, 10, 0),
    ]

    groups = _validate_statuses(statuses, "host", world_size=2)

    assert groups == [{"ranks": [0, 1], "leader_rank": 0, "host_id": "host-a", "shmid": 10, "numa_nodes": [0, 1]}]


def test_status_validation_rejects_a_follower_on_a_private_segment():
    statuses = [
        _status(0, 0, (0, 1), 0, 10, 1),
        _status(1, 0, (0, 1), 0, 12, 0),
    ]

    with pytest.raises(RuntimeError, match="does not share one SysV segment"):
        _validate_statuses(statuses, "auto", world_size=2)
