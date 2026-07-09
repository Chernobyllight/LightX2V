import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

try:
    from torch.distributed import ProcessGroupNCCL
except ImportError:
    ProcessGroupNCCL = None


@PLATFORM_DEVICE_REGISTER("cuda")
class CudaDevice:
    name = "cuda"

    @staticmethod
    def init_device_env():
        pass

    @staticmethod
    def is_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def get_device() -> str:
        return "cuda"

    @staticmethod
    def init_parallel_env():
        if ProcessGroupNCCL is None:
            raise RuntimeError("ProcessGroupNCCL is not available. Please check your runtime environment.")
        pg_options = ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = True
        dist.init_process_group(backend="nccl", pg_options=pg_options)
        device_idx = dist.get_rank()
        if os.environ.get("LIGHTX2V_CFG_PARALLEL_DEVICE_SPLIT", "").lower() in ("1", "true", "yes", "on"):
            cfg_p_size = int(os.environ.get("LIGHTX2V_CFG_PARALLEL_SIZE", "1") or 1)
            device_count = torch.cuda.device_count()
            local_rank = int(os.environ.get("LOCAL_RANK", device_idx) or device_idx)
            if cfg_p_size > 1 and device_count >= cfg_p_size and device_count % cfg_p_size == 0:
                device_idx = local_rank * (device_count // cfg_p_size)
        torch.cuda.set_device(device_idx)
        logger.info(f"Initialized CUDA distributed env: rank={dist.get_rank()}, current_device=cuda:{device_idx}")
