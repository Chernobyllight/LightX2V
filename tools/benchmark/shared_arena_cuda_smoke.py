"""Small distributed CUDA smoke test for the shared weight arena."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=32)
    parser.add_argument("--scope", choices=("auto", "host", "numa"), default="auto")
    return parser.parse_args()


def _expected_groups(statuses, scope):
    cohorts = {}
    for status in statuses:
        key = (status["host_id"], status["ipc_namespace"], status["manifest_hash"])
        cohorts.setdefault(key, []).append(status)

    groups = []
    for cohort in cohorts.values():
        split_by_numa = scope == "numa" or (scope == "auto" and all(status["numa_node"] is not None for status in cohort))
        if split_by_numa:
            by_numa = {}
            for status in cohort:
                if status["numa_node"] is None:
                    raise RuntimeError("numa scope requires a known NUMA node for every rank")
                by_numa.setdefault(status["numa_node"], []).append(status["rank"])
            groups.extend(tuple(sorted(ranks)) for ranks in by_numa.values())
        else:
            groups.append(tuple(sorted(status["rank"] for status in cohort)))
    return sorted(groups, key=lambda ranks: ranks[0])


def _validate_statuses(statuses, scope, world_size):
    statuses = sorted(statuses, key=lambda status: status["rank"])
    ranks = [status["rank"] for status in statuses]
    if ranks != list(range(world_size)):
        raise RuntimeError(f"expected one status for ranks 0..{world_size - 1}, got {ranks}")
    if len({status["manifest_hash"] for status in statuses}) != 1:
        raise RuntimeError("ranks built different shared-weight manifests")
    if len({status["arena_bytes"] for status in statuses}) != 1:
        raise RuntimeError("ranks report different arena sizes")

    failures = [status for status in statuses if not status["h2d_ok"]]
    if failures:
        raise RuntimeError(f"H2D validation failed: {failures}")

    expected_groups = _expected_groups(statuses, scope)
    reported_groups = sorted({tuple(status["group_ranks"]) for status in statuses}, key=lambda ranks: ranks[0])
    if reported_groups != expected_groups:
        raise RuntimeError(f"expected replica groups {expected_groups}, got {reported_groups}")

    summaries = []
    for group_ranks in expected_groups:
        members = [status for status in statuses if status["rank"] in group_ranks]
        if any(tuple(status["group_ranks"]) != group_ranks for status in members):
            raise RuntimeError(f"ranks disagree about replica group {group_ranks}")

        leader_rank = min(group_ranks)
        leaders = [status for status in members if status["populate_calls"] == 1]
        followers = [status for status in members if status["populate_calls"] == 0]
        if [status["rank"] for status in leaders] != [leader_rank] or len(followers) != len(members) - 1:
            calls = {status["rank"]: status["populate_calls"] for status in members}
            raise RuntimeError(f"group {group_ranks} expected only rank {leader_rank} to populate, got {calls}")
        if any(status["leader_rank"] != leader_rank for status in members):
            raise RuntimeError(f"group {group_ranks} reports an inconsistent leader")

        ipc_segments = {(status["host_id"], status["ipc_namespace"], status["shmid"]) for status in members}
        if len(ipc_segments) != 1:
            raise RuntimeError(f"group {group_ranks} does not share one SysV segment: {sorted(ipc_segments)}")
        if any(status["registered_bytes"] != status["arena_bytes"] for status in members):
            raise RuntimeError(f"group {group_ranks} did not register its complete arena")

        summaries.append(
            {
                "ranks": list(group_ranks),
                "leader_rank": leader_rank,
                "host_id": members[0]["host_id"],
                "shmid": members[0]["shmid"],
                "numa_nodes": sorted({status["numa_node"] for status in members if status["numa_node"] is not None}),
            }
        )

    segment_keys = [(status["host_id"], status["ipc_namespace"], status["shmid"]) for status in statuses if status["rank"] == status["leader_rank"]]
    if len(segment_keys) != len(set(segment_keys)):
        raise RuntimeError("different replica groups unexpectedly report the same SysV segment")
    return summaries


def main():
    args = parse_args()
    if args.size_mib <= 0:
        raise ValueError("--size-mib must be positive")

    from lightx2v.common.offload.shared_pinned_arena import SharedWeightManifest
    from lightx2v.common.offload.shared_weight_coordinator import materialize_shared_weight_arena

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    allocation = None
    views = None
    source = None
    device_copy = None
    try:
        if distributed:
            dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

        tensor_meta = {"payload": torch.empty(args.size_mib * 1024 * 1024, dtype=torch.uint8, device="meta")}
        manifest = SharedWeightManifest.from_tensors(
            tensor_meta,
            weight_signature=f"shared-arena-cuda-smoke:{args.size_mib}MiB:v1",
            alignment=4096,
        )

        populate_calls = 0

        def populate(shared_views):
            nonlocal populate_calls
            populate_calls += 1
            shared_views["payload"].fill_(37)

        allocation = materialize_shared_weight_arena(manifest, populate, scope=args.scope)
        views = allocation.tensor_views()
        source = views["payload"]
        pinned = source.is_pinned()

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            device_copy = source.to("cuda", non_blocking=True)
        stream.synchronize()
        h2d_ok = pinned and bool(torch.all(device_copy == 37).item())

        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1
        status = {
            "rank": rank,
            "host_id": allocation.topology.host_id,
            "ipc_namespace": allocation.topology.ipc_namespace,
            "numa_node": allocation.topology.numa_node,
            "group_ranks": list(allocation.group.ranks),
            "leader_rank": allocation.group.leader_rank,
            "shmid": allocation.arena.shmid,
            "arena_bytes": allocation.arena.nbytes,
            "registered_bytes": allocation.arena.registered_nbytes,
            "manifest_hash": manifest.digest,
            "populate_calls": populate_calls,
            "h2d_ok": h2d_ok,
        }
        if distributed:
            statuses = [None] * world_size
            dist.all_gather_object(statuses, status)
        else:
            statuses = [status]

        groups = _validate_statuses(statuses, args.scope, world_size)
        if rank == 0:
            summary = {
                "arena_bytes": manifest.nbytes,
                "groups": groups,
                "h2d": "ok",
                "scope": args.scope,
                "world_size": world_size,
            }
            print(f"SHARED_ARENA_CUDA_SMOKE_SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)
    finally:
        device_copy = None
        source = None
        views = None
        try:
            if allocation is not None:
                allocation.close()
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
