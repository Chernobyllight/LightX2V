import json

import pytest

from tools.benchmark.shared_offload_monitor import build_report, parse_maps, parse_numa_maps, parse_shared_events, parse_smaps

MAPS = """\
10000000-10010000 rw-s 00000000 00:01 42 /SYSV00000000 (deleted)
20000000-20001000 rw-p 00000000 00:00 0 [heap]
30000000-30008000 rw-s 00000000 00:01 99 /SYSV00000000 (deleted)
"""

SMAPS = """\
10000000-10010000 rw-s 00000000 00:01 42 /SYSV00000000 (deleted)
Size:                 64 kB
Rss:                  64 kB
Pss:                  32 kB
Shared_Clean:          8 kB
Shared_Dirty:         48 kB
Private_Clean:         4 kB
Private_Dirty:         4 kB
Locked:               64 kB
VmFlags: rd wr sh mr mw me ms lo
20000000-20001000 rw-p 00000000 00:00 0 [heap]
Size:                  4 kB
Rss:                   4 kB
Pss:                   4 kB
Private_Dirty:         4 kB
30000000-30008000 rw-s 00000000 00:01 99 /SYSV00000000 (deleted)
Size:                 32 kB
Rss:                  16 kB
Pss:                   8 kB
Shared_Dirty:         16 kB
"""

NUMA_MAPS = """\
10000000 bind:0 file=/SYSV00000000\\040(deleted) dirty=16 mapped=16 N0=12 N1=4 kernelpagesize_kB=4
20000000 default heap anon=1 dirty=1 N0=1 kernelpagesize_kB=4
30000000 default file=/SYSV00000000\\040(deleted) dirty=4 mapped=4 N1=4 kernelpagesize_kB=4
"""


def _write_proc(proc_root, pid, *, address="10000000"):
    process_dir = proc_root / str(pid)
    process_dir.mkdir()
    process_dir.joinpath("maps").write_text(MAPS.replace("10000000", address).replace("10010000", f"{int(address, 16) + 0x10000:x}"))
    process_dir.joinpath("smaps").write_text(SMAPS.replace("10000000", address).replace("10010000", f"{int(address, 16) + 0x10000:x}"))
    process_dir.joinpath("numa_maps").write_text(NUMA_MAPS.replace("10000000", address))


def test_proc_parsers_correlate_sysv_smaps_and_numa_fields():
    mappings = parse_maps(MAPS)
    smaps = parse_smaps(SMAPS)
    numa = parse_numa_maps(NUMA_MAPS)

    assert [(mapping.shmid, mapping.nbytes) for mapping in mappings if mapping.is_sysv] == [(42, 64 * 1024), (99, 32 * 1024)]
    assert smaps[0x10000000]["Pss"] == 32
    assert smaps[0x10000000]["Shared_Dirty"] == 48
    assert numa[0x10000000].policy == "bind:0"
    assert numa[0x10000000].pages_by_node == {0: 12, 1: 4}
    assert numa[0x10000000].kernel_page_kb == 4


def test_log_driven_report_filters_shmid_and_aggregates_without_numa_multiplication(tmp_path):
    _write_proc(tmp_path, 100, address="10000000")
    _write_proc(tmp_path, 101, address="11000000")
    log = "\n".join(
        [
            'prefix [SharedCPUWeights] {"pid": 100, "rank": 0, "shmid": 42, "arena_bytes": 65536, "numa_node": 0} suffix',
            'prefix [SharedCPUWeights] {"pid": 101, "rank": 1, "shmid": 42, "arena_bytes": 65536, "numa_node": 0}',
        ]
    )
    events = parse_shared_events(log, source="inference.log")

    report = build_report(events=events, proc_root=tmp_path)

    assert report["requested"] == {"pids": [100, 101], "shmids": []}
    assert all(len(process["mappings"]) == 1 for process in report["processes"])
    summary = report["shmids"][0]
    assert summary["shmid"] == 42
    assert summary["pids"] == [100, 101]
    assert summary["mapping_count"] == 2
    assert summary["sum_pss_kb"] == 64
    assert summary["sum_shared_kb"] == 112
    assert summary["sum_private_kb"] == 16
    assert summary["numa_pages_sum_across_mappings"] == {"0": 24, "1": 8}
    assert summary["numa_pages_nonadditive_max"] == {"0": 12, "1": 4}
    assert summary["expected_arena_bytes"] == 65536
    assert report["totals"]["sum_mapping_count"] == 2


def test_shared_events_can_be_concatenated_on_one_log_line():
    log = 'prefix [SharedCPUWeights] {"pid": 100, "shmid": 42}[SharedCPUWeights] {"pid": 101, "shmid": 42}\n'

    events = parse_shared_events(log, source="inference.log")

    assert [(event["pid"], event["shmid"]) for event in events] == [(100, 42), (101, 42)]
    assert [event["_line"] for event in events] == [1, 1]


def test_explicit_pid_without_event_reports_all_sysv_mappings(tmp_path):
    _write_proc(tmp_path, 100)

    report = build_report(explicit_pids=[100], proc_root=tmp_path)

    assert [mapping["shmid"] for mapping in report["processes"][0]["mappings"]] == [42, 99]


def test_missing_process_is_reported_as_json_data(tmp_path):
    report = build_report(explicit_pids=[404], proc_root=tmp_path)

    assert "error" in report["processes"][0]
    assert report["totals"]["process_error_count"] == 1


def test_invalid_shared_event_is_rejected_with_source_location():
    with pytest.raises(ValueError, match=r"worker.log:2"):
        parse_shared_events("unrelated\n[SharedCPUWeights] not-json", source="worker.log")


def test_report_rejects_events_from_multiple_ipc_domains(tmp_path):
    events = [
        {"pid": 100, "shmid": 7, "host_id": "host-a", "ipc_namespace": "ipc:[1]"},
        {"pid": 101, "shmid": 7, "host_id": "host-b", "ipc_namespace": "ipc:[2]"},
    ]

    with pytest.raises(ValueError, match="only one host and IPC namespace"):
        build_report(events=events, proc_root=tmp_path)


def test_report_is_json_serializable(tmp_path):
    _write_proc(tmp_path, 100)
    report = build_report(explicit_pids=[100], explicit_shmids=[42], proc_root=tmp_path)

    decoded = json.loads(json.dumps(report))
    assert decoded["shmids"][0]["shmid"] == 42
