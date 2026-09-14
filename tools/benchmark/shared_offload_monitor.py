#!/usr/bin/env python3
"""Inspect one host's process-shared offload arenas through Linux ``/proc``.

The runtime emits one JSON event per rank after a shared arena is attached::

    [SharedCPUWeights] {"pid": 123, "shmid": 42, ...}

This tool can discover the relevant PIDs and SysV shared-memory IDs from those
events, accept either value explicitly, and correlate ``maps``, ``smaps`` and
``numa_maps``.  It intentionally uses only the Python standard library so it
can run next to a live inference process without importing torch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_MAPS_HEADER_RE = re.compile(
    r"^(?P<start>[0-9a-fA-F]+)-(?P<end>[0-9a-fA-F]+)\s+"
    r"(?P<perms>\S+)\s+(?P<offset>[0-9a-fA-F]+)\s+"
    r"(?P<device>\S+)\s+(?P<inode>\d+)(?:\s+(?P<path>.*))?$"
)
_SMAPS_VALUE_RE = re.compile(r"^(?P<key>[A-Za-z_]+):\s+(?P<value>\d+)(?:\s+kB)?$")
_NUMA_NODE_RE = re.compile(r"^N(?P<node>\d+)=(?P<pages>\d+)$")
_SHARED_EVENT_MARKER = "[SharedCPUWeights]"


@dataclass(frozen=True)
class ProcMap:
    start: int
    end: int
    perms: str
    offset: int
    device: str
    inode: int
    path: str

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def is_sysv(self) -> bool:
        return self.path.startswith("/SYSV")

    @property
    def shmid(self) -> int | None:
        # Linux exposes the SysV shmid as the maps inode value.
        return self.inode if self.is_sysv else None


@dataclass(frozen=True)
class NumaMap:
    start: int
    policy: str
    pages_by_node: Mapping[int, int]
    kernel_page_kb: int | None


def parse_maps(text: str) -> list[ProcMap]:
    """Parse ``/proc/PID/maps`` and return all well-formed mappings."""

    mappings = []
    for line in text.splitlines():
        match = _MAPS_HEADER_RE.match(line)
        if match is None:
            continue
        mappings.append(
            ProcMap(
                start=int(match.group("start"), 16),
                end=int(match.group("end"), 16),
                perms=match.group("perms"),
                offset=int(match.group("offset"), 16),
                device=match.group("device"),
                inode=int(match.group("inode")),
                path=match.group("path") or "",
            )
        )
    return mappings


def parse_smaps(text: str) -> dict[int, dict[str, int]]:
    """Return smaps kB counters keyed by mapping start address."""

    counters: dict[int, dict[str, int]] = {}
    current: dict[str, int] | None = None
    for line in text.splitlines():
        header = _MAPS_HEADER_RE.match(line)
        if header is not None:
            current = counters.setdefault(int(header.group("start"), 16), {})
            continue
        if current is None:
            continue
        value = _SMAPS_VALUE_RE.match(line)
        if value is not None:
            current[value.group("key")] = int(value.group("value"))
    return counters


def parse_numa_maps(text: str) -> dict[int, NumaMap]:
    """Return NUMA residency counters keyed by mapping start address."""

    mappings = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            start = int(fields[0], 16)
        except ValueError:
            continue
        pages_by_node = {}
        kernel_page_kb = None
        for field in fields[2:]:
            node = _NUMA_NODE_RE.match(field)
            if node is not None:
                pages_by_node[int(node.group("node"))] = int(node.group("pages"))
            elif field.startswith("kernelpagesize_kB="):
                try:
                    kernel_page_kb = int(field.split("=", 1)[1])
                except ValueError:
                    pass
        mappings[start] = NumaMap(start=start, policy=fields[1], pages_by_node=pages_by_node, kernel_page_kb=kernel_page_kb)
    return mappings


def parse_shared_events(text: str, *, source: str = "<memory>") -> list[dict[str, Any]]:
    """Extract valid SharedCPUWeights JSON objects from arbitrary log lines."""

    events = []
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        marker_index = text.find(_SHARED_EVENT_MARKER, cursor)
        if marker_index < 0:
            break
        line_number = text.count("\n", 0, marker_index) + 1
        payload_start = marker_index + len(_SHARED_EVENT_MARKER)
        while payload_start < len(text) and text[payload_start].isspace():
            payload_start += 1
        try:
            event, payload_end = decoder.raw_decode(text, payload_start)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid SharedCPUWeights JSON in {source}:{line_number}: {error.msg}") from error
        if not isinstance(event, dict):
            raise ValueError(f"SharedCPUWeights event in {source}:{line_number} is not an object")
        if "pid" not in event or "shmid" not in event:
            raise ValueError(f"SharedCPUWeights event in {source}:{line_number} lacks pid or shmid")
        event = dict(event)
        event["pid"] = int(event["pid"])
        event["shmid"] = int(event["shmid"])
        event["_source"] = source
        event["_line"] = line_number
        events.append(event)
        cursor = payload_end
    return events


def _sum_smaps(counters: Mapping[str, int]) -> dict[str, int]:
    shared_kb = counters.get("Shared_Clean", 0) + counters.get("Shared_Dirty", 0)
    private_kb = counters.get("Private_Clean", 0) + counters.get("Private_Dirty", 0)
    return {
        "size_kb": counters.get("Size", 0),
        "rss_kb": counters.get("Rss", 0),
        "pss_kb": counters.get("Pss", 0),
        "shared_kb": shared_kb,
        "private_kb": private_kb,
        "locked_kb": counters.get("Locked", 0),
    }


def _add_numeric(target: dict[str, int], values: Mapping[str, int]) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0) + value


def inspect_process(pid: int, *, shmids: set[int] | None = None, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Inspect one process.  ``shmids=None`` selects every SysV mapping."""

    proc_dir = proc_root / str(pid)
    result: dict[str, Any] = {"pid": pid, "mappings": [], "totals": {}}
    try:
        maps = parse_maps((proc_dir / "maps").read_text())
        smaps = parse_smaps((proc_dir / "smaps").read_text())
        numa_maps = parse_numa_maps((proc_dir / "numa_maps").read_text())
    except (OSError, PermissionError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result

    selected = [mapping for mapping in maps if mapping.is_sysv and (shmids is None or mapping.shmid in shmids)]
    totals: dict[str, int] = {"mapping_count": len(selected), "virtual_bytes": 0}
    numa_pages: dict[int, int] = defaultdict(int)
    for mapping in selected:
        metrics = _sum_smaps(smaps.get(mapping.start, {}))
        numa = numa_maps.get(mapping.start)
        page_counts = {} if numa is None else dict(sorted(numa.pages_by_node.items()))
        detail: dict[str, Any] = {
            "shmid": mapping.shmid,
            "address_start": mapping.start,
            "address_end": mapping.end,
            "address_range": f"{mapping.start:x}-{mapping.end:x}",
            "virtual_bytes": mapping.nbytes,
            "perms": mapping.perms,
            "path": mapping.path,
            **metrics,
            "numa_policy": None if numa is None else numa.policy,
            "numa_pages": {str(node): pages for node, pages in page_counts.items()},
            "kernel_page_kb": None if numa is None else numa.kernel_page_kb,
        }
        result["mappings"].append(detail)
        totals["virtual_bytes"] += mapping.nbytes
        _add_numeric(totals, metrics)
        for node, pages in page_counts.items():
            numa_pages[node] += pages
    totals["numa_pages"] = {str(node): pages for node, pages in sorted(numa_pages.items())}
    result["totals"] = totals
    return result


def aggregate_by_shmid(processes: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate mapping metrics across ranks, keeping NUMA counts explicit."""

    expected_sizes: dict[int, set[int]] = defaultdict(set)
    event_metadata: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        shmid = int(event["shmid"])
        if "arena_bytes" in event:
            expected_sizes[shmid].add(int(event["arena_bytes"]))
        event_metadata[shmid].append({key: value for key, value in event.items() if not key.startswith("_")})

    grouped: dict[int, dict[str, Any]] = {}
    max_pages: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    sum_pages: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for process in processes:
        pid = int(process["pid"])
        for mapping in process.get("mappings", []):
            shmid = int(mapping["shmid"])
            entry = grouped.setdefault(
                shmid,
                {
                    "shmid": shmid,
                    "pids": [],
                    "mapping_count": 0,
                    "sum_virtual_bytes": 0,
                    "sum_size_kb": 0,
                    "sum_rss_kb": 0,
                    "sum_pss_kb": 0,
                    "sum_shared_kb": 0,
                    "sum_private_kb": 0,
                    "sum_locked_kb": 0,
                },
            )
            if pid not in entry["pids"]:
                entry["pids"].append(pid)
            entry["mapping_count"] += 1
            entry["sum_virtual_bytes"] += int(mapping["virtual_bytes"])
            for name in ("size_kb", "rss_kb", "pss_kb", "shared_kb", "private_kb", "locked_kb"):
                entry[f"sum_{name}"] += int(mapping[name])
            for node_text, pages in mapping["numa_pages"].items():
                node = int(node_text)
                sum_pages[shmid][node] += int(pages)
                max_pages[shmid][node] = max(max_pages[shmid][node], int(pages))

    for shmid in set(event_metadata) - set(grouped):
        grouped[shmid] = {
            "shmid": shmid,
            "pids": [],
            "mapping_count": 0,
            "sum_virtual_bytes": 0,
            "sum_size_kb": 0,
            "sum_rss_kb": 0,
            "sum_pss_kb": 0,
            "sum_shared_kb": 0,
            "sum_private_kb": 0,
            "sum_locked_kb": 0,
        }

    output = []
    for shmid, entry in sorted(grouped.items()):
        entry["pids"].sort()
        sizes = sorted(expected_sizes.get(shmid, ()))
        entry["expected_arena_bytes"] = sizes[0] if len(sizes) == 1 else (sizes or None)
        entry["numa_pages_sum_across_mappings"] = {str(node): pages for node, pages in sorted(sum_pages[shmid].items())}
        # Each process maps the same physical segment.  Per-node maxima avoid
        # naively multiplying identical residency by the rank count, while PSS
        # remains the kernel's preferred additive physical-memory estimate.
        entry["numa_pages_nonadditive_max"] = {str(node): pages for node, pages in sorted(max_pages[shmid].items())}
        entry["events"] = event_metadata.get(shmid, [])
        output.append(entry)
    return output


def build_report(
    *,
    explicit_pids: Iterable[int] = (),
    events: Sequence[Mapping[str, Any]] = (),
    explicit_shmids: Iterable[int] = (),
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Build the serializable report used by both the CLI and unit tests."""

    domains = {(event["host_id"], event["ipc_namespace"]) for event in events if event.get("host_id") is not None and event.get("ipc_namespace") is not None}
    if len(domains) > 1:
        raise ValueError("a /proc report can cover only one host and IPC namespace")

    explicit_pid_set = {int(pid) for pid in explicit_pids}
    selected_shmids = {int(shmid) for shmid in explicit_shmids}
    event_shmids: dict[int, set[int]] = defaultdict(set)
    for event in events:
        event_shmids[int(event["pid"])].add(int(event["shmid"]))
    pids = sorted(explicit_pid_set | set(event_shmids))
    processes = []
    for pid in pids:
        if selected_shmids:
            filters = selected_shmids
        elif pid in event_shmids:
            filters = event_shmids[pid]
        else:
            filters = None
        processes.append(inspect_process(pid, shmids=filters, proc_root=proc_root))

    totals: dict[str, int] = {"process_count": len(processes), "process_error_count": sum("error" in process for process in processes)}
    for process in processes:
        process_totals = process.get("totals", {})
        for name in ("mapping_count", "virtual_bytes", "size_kb", "rss_kb", "pss_kb", "shared_kb", "private_kb", "locked_kb"):
            totals[f"sum_{name}"] = totals.get(f"sum_{name}", 0) + int(process_totals.get(name, 0))

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested": {"pids": pids, "shmids": sorted(selected_shmids)},
        "event_count": len(events),
        "processes": processes,
        "shmids": aggregate_by_shmid(processes, events),
        "totals": totals,
        "notes": {
            "pss": "sum_pss_kb is additive across process mappings and is the preferred physical-memory estimate.",
            "numa": "numa_pages_sum_across_mappings counts every VMA; numa_pages_nonadditive_max avoids multiplying identical shared residency but may underestimate disjoint faults.",
        },
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", action="append", default=[], type=_positive_int, help="process ID to inspect; repeat for multiple ranks")
    parser.add_argument("--log", action="append", default=[], type=Path, help="log containing [SharedCPUWeights] JSON events; repeatable")
    parser.add_argument("--shmid", action="append", default=[], type=_nonnegative_int, help="restrict inspection to a SysV shmid; repeatable")
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help="write JSON to this file instead of stdout")
    parser.add_argument("--compact", action="store_true", help="emit compact rather than indented JSON")
    args = parser.parse_args(argv)
    if not args.pid and not args.log:
        parser.error("at least one --pid or --log is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    events = []
    try:
        for log_path in args.log:
            events.extend(parse_shared_events(log_path.read_text(), source=str(log_path)))
        report = build_report(explicit_pids=args.pid, events=events, explicit_shmids=args.shmid, proc_root=args.proc_root)
    except (OSError, ValueError) as error:
        print(f"shared_offload_monitor: {error}", file=sys.stderr)
        return 2

    payload = json.dumps(report, indent=None if args.compact else 2, sort_keys=True)
    try:
        if args.output is None:
            print(payload)
        else:
            args.output.write_text(payload + "\n")
    except OSError as error:
        print(f"shared_offload_monitor: {error}", file=sys.stderr)
        return 2
    return 1 if report["totals"]["process_error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
