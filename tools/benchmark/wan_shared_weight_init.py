"""Construct only the Wan DiT to validate shared block-weight initialization."""

from __future__ import annotations

import argparse
import gc
import os
import time

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--scope", choices=("auto", "host", "numa"), help="override shared_cpu_weight_scope from the config")
    parser.add_argument("--hold-seconds", type=float, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hold_seconds < 0:
        raise ValueError("--hold-seconds must be non-negative")

    from lightx2v.models.networks.wan.model import WanModel
    from lightx2v.utils.set_config import build_startup_config, init_parallel
    from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

    config = build_startup_config(
        {
            "model_cls": "wan2.1",
            "model_path": args.model_path,
            "task": "i2v",
            "config_json": args.config_json,
        }
    )
    if args.scope is not None:
        config["shared_cpu_weight_scope"] = args.scope

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    model = None
    allocation = None
    try:
        if world_size > 1:
            config["parallel"]["seq_p_size"] = world_size
            platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
            platform_device.init_parallel_env()
            init_parallel(config)
        else:
            config["parallel"] = False
            config["seq_parallel"] = False
            torch.cuda.set_device(0)

        model = WanModel(model_path=args.model_path, config=config, device=torch.device("cpu"))
        rank = dist.get_rank() if dist.is_initialized() else 0
        allocation = model._shared_cpu_weight_owner
        print(
            f"WAN_SHARED_INIT_OK rank={rank} pid={os.getpid()} scope={config['shared_cpu_weight_scope']} shmid={allocation.arena.shmid} numa={allocation.topology.numa_node} group={allocation.group.ranks} arena_bytes={allocation.arena.nbytes}",
            flush=True,
        )
        if args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
        if dist.is_initialized():
            dist.barrier()
    finally:
        try:
            if model is not None:
                model.close_shared_cpu_weights()
                model = None
                gc.collect()
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
