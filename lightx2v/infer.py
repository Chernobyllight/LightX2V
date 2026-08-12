import argparse
import json
import os
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.common.ops import *
from lightx2v.models.networks.bagel.sensenova_tasks import OMNI_VISION_SUBTASK_CHOICES
from lightx2v.models.runners.bagel.bagel_runner import BagelRunner  # noqa: F401
from lightx2v.models.runners.bagel.sensenova_vision_runner import SenseNovaVisionRunner  # noqa: F401
from lightx2v.models.runners.cosmos3.cosmos3_runner import Cosmos3Runner  # noqa: F401
from lightx2v.models.runners.ernie_image.ernie_image_runner import ErnieImageRunner  # noqa: F401
from lightx2v.models.runners.flux2.flux2_runner import Flux2DevRunner, Flux2KleinRunner  # noqa: F401
from lightx2v.models.runners.hidream_o1_image.hidream_o1_image_runner import HidreamO1ImageRunner  # noqa: F401
from lightx2v.models.runners.hunyuan3d.hunyuan3d_shape_runner import Hunyuan3DShapeRunner  # noqa: F401
from lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner import HunyuanImage3Runner  # noqa: F401
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_distill_runner import HunyuanVideo15DistillRunner  # noqa: F401
from lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner import HunyuanVideo15Runner  # noqa: F401
from lightx2v.models.runners.lingbot_video.lingbot_video_runner import LingBotVideoRunner  # noqa: F401
from lightx2v.models.runners.longcat_image.longcat_image_runner import LongCatImageRunner  # noqa: F401
from lightx2v.models.runners.ltx2.ltx2_runner import LTX2ARRunner, LTX2Runner  # noqa: F401
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner  # noqa: F401
from lightx2v.models.runners.motus.motus_runner import MotusRunner  # noqa: F401
from lightx2v.models.runners.neopp.neopp_runner import NeoppRunner  # noqa: F401
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner  # noqa: F401
from lightx2v.models.runners.seedvr.seedvr_runner import SeedVRRunner  # noqa: F401
from lightx2v.models.runners.wan.fastwam_runner import FastWAMRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_animate_runner import WanAnimateRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner, WanAudioRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_dancer_runner import WanDancerRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_dreamzero_runner import WanDreamZeroRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_infinitetalk_runner import InfiniteTalkRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_lingbot_va_runner import LingbotVARunner  # noqa: F401
from lightx2v.models.runners.wan.wan_matrix_game2_runner import WanSFMtxg2Runner  # noqa: F401
from lightx2v.models.runners.wan.wan_matrix_game3_runner import WanMatrixGame3Runner  # noqa: F401
from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_s2v_runner import WanS2VRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_vace_runner import Wan22MoeVaceRunner, WanVaceRunner  # noqa: F401
from lightx2v.models.runners.worldmirror.worldmirror_runner import WorldMirrorRunner  # noqa: F401
from lightx2v.models.runners.worldplay.worldplay_ar_runner import WorldPlayARRunner  # noqa: F401
from lightx2v.models.runners.worldplay.worldplay_bi_runner import WorldPlayBIRunner  # noqa: F401
from lightx2v.models.runners.worldplay.worldplay_distill_runner import WorldPlayDistillRunner  # noqa: F401
from lightx2v.models.runners.z_image.z_image_runner import ZImageRunner  # noqa: F401
from lightx2v.utils.envs import *
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all, validate_config_paths
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def init_runner(config):
    torch.set_grad_enabled(False)
    runner = RUNNER_REGISTER[config["model_cls"]](config)
    runner.init_modules()
    return runner


def distributed_barrier():
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return False

    from lightx2v_platform.base.global_var import AI_DEVICE

    if AI_DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()

    from loguru import logger

    logger.info(f"[Barrier] synchronized all ranks")
    return True


def benchmark_synchronize():
    """Finish device work and align all ranks at a benchmark boundary."""
    from lightx2v_platform.base.global_var import AI_DEVICE

    device_module = getattr(torch, AI_DEVICE, None)
    if device_module is not None and hasattr(device_module, "synchronize"):
        device_module.synchronize()

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    if AI_DEVICE == "cuda" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def _distributed_max_values(values):
    values = [float(value) for value in values]
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return values

    backend = str(dist.get_backend()).lower()
    device = torch.device("cuda", torch.cuda.current_device()) if "nccl" in backend else torch.device("cpu")
    length_bounds = torch.tensor([len(values), -len(values)], dtype=torch.int64, device=device)
    dist.all_reduce(length_bounds, op=dist.ReduceOp.MAX)
    maximum_length = int(length_bounds[0].item())
    minimum_length = -int(length_bounds[1].item())
    if minimum_length != maximum_length:
        raise RuntimeError(f"Benchmark timing arrays differ across ranks: min={minimum_length}, max={maximum_length}.")
    if maximum_length == 0:
        return []

    reduced = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return reduced.cpu().tolist()


def aggregate_fine_grained_stage_timings(stage_timings):
    """Aggregate per-rank CUDA-event timings after the measured E2E boundary."""
    if not isinstance(stage_timings, dict):
        return stage_timings

    aggregated = dict(stage_timings)
    if "ar_prefill_seconds" in aggregated:
        aggregated["ar_prefill_seconds"] = _distributed_max_values([aggregated["ar_prefill_seconds"]])[0]

    decode_token_seconds = aggregated.get("ar_decode_token_seconds")
    if isinstance(decode_token_seconds, list):
        decode_token_seconds = _distributed_max_values(decode_token_seconds)
        decode_seconds = sum(decode_token_seconds)
        decode_token_count = len(decode_token_seconds)
        ar_timing_scope = str(aggregated.get("ar_fine_grained_timing_scope", "fine_grained_local_rank"))
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            ar_timing_scope = ar_timing_scope.replace("local_rank", "max_across_world_after_e2e")
        aggregated.update(
            {
                "ar_fine_grained_timing_scope": ar_timing_scope,
                "ar_decode_token_seconds": decode_token_seconds,
                "ar_decode_seconds": decode_seconds,
                "ar_decode_model_token_count": decode_token_count,
            }
        )
        if decode_token_count:
            seconds_per_token = decode_seconds / decode_token_count
            aggregated["ar_decode_seconds_per_token"] = seconds_per_token
            aggregated["ar_decode_milliseconds_per_token"] = seconds_per_token * 1000.0

    denoise_step_seconds = aggregated.get("denoise_step_seconds_samples")
    if isinstance(denoise_step_seconds, list):
        denoise_step_seconds = _distributed_max_values(denoise_step_seconds)
        step_count = len(denoise_step_seconds)
        denoise_timing_scope = str(aggregated.get("denoise_fine_grained_timing_scope", "fine_grained_local_rank"))
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            denoise_timing_scope = denoise_timing_scope.replace("local_rank", "max_across_world_after_e2e")
        aggregated.update(
            {
                "denoise_fine_grained_timing_scope": denoise_timing_scope,
                "denoise_step_seconds_samples": denoise_step_seconds,
                "denoise_step_count": step_count,
            }
        )
        if step_count:
            aggregated["denoise_step_seconds"] = sum(denoise_step_seconds) / step_count
            aggregated["denoise_first_step_seconds"] = denoise_step_seconds[0]
            aggregated["denoise_remaining_step_seconds"] = sum(denoise_step_seconds[1:]) / (step_count - 1) if step_count > 1 else denoise_step_seconds[0]
    return aggregated


def is_output_rank():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _summarize_benchmark_samples(samples):
    values = [float(value) for value in samples]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "population_stddev": statistics.pstdev(values),
    }


def _is_stage_summary_metric(key, values):
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return False
    return key.endswith(("_seconds", "_seconds_per_token", "_milliseconds_per_token", "_count"))


def _summarize_stage_samples(stage_timing_samples):
    if not stage_timing_samples:
        return {}
    summary = {}
    common_keys = set.intersection(*(set(sample) for sample in stage_timing_samples))
    for key in sorted(common_keys):
        values = [sample[key] for sample in stage_timing_samples]
        if _is_stage_summary_metric(key, values):
            summary[key] = _summarize_benchmark_samples(values)
    return summary


def _summarize_ar_tokens(stage_timing_samples):
    valid_samples = [
        sample
        for sample in stage_timing_samples
        if isinstance(sample.get("ar_seconds"), (int, float)) and isinstance(sample.get("generated_token_count"), (int, float)) and sample["generated_token_count"] > 0
    ]
    if not valid_samples:
        return None

    total_ar_seconds = sum(float(sample["ar_seconds"]) for sample in valid_samples)
    total_generated_tokens = sum(int(sample["generated_token_count"]) for sample in valid_samples)
    seconds_per_token = total_ar_seconds / total_generated_tokens
    return {
        "sample_count": len(valid_samples),
        "total_generated_tokens": total_generated_tokens,
        "total_ar_seconds": total_ar_seconds,
        "average_seconds_per_token": seconds_per_token,
        "average_milliseconds_per_token": seconds_per_token * 1000.0,
    }


def _summarize_ar_decode_tokens(stage_timing_samples):
    valid_samples = [
        sample
        for sample in stage_timing_samples
        if isinstance(sample.get("ar_decode_seconds"), (int, float)) and isinstance(sample.get("ar_decode_model_token_count"), (int, float)) and sample["ar_decode_model_token_count"] > 0
    ]
    if not valid_samples:
        return None

    total_decode_seconds = sum(float(sample["ar_decode_seconds"]) for sample in valid_samples)
    total_decode_tokens = sum(int(sample["ar_decode_model_token_count"]) for sample in valid_samples)
    seconds_per_token = total_decode_seconds / total_decode_tokens
    return {
        "sample_count": len(valid_samples),
        "total_decode_model_tokens": total_decode_tokens,
        "total_decode_seconds": total_decode_seconds,
        "average_seconds_per_token": seconds_per_token,
        "average_milliseconds_per_token": seconds_per_token * 1000.0,
    }


def _summarize_denoise_steps(stage_timing_samples):
    valid_samples = [
        sample["denoise_step_seconds_samples"] for sample in stage_timing_samples if isinstance(sample.get("denoise_step_seconds_samples"), list) and sample["denoise_step_seconds_samples"]
    ]
    if not valid_samples:
        return None

    all_step_seconds = [float(value) for sample in valid_samples for value in sample]
    summary = _summarize_benchmark_samples(all_step_seconds)
    return {
        "sample_count": len(valid_samples),
        "total_step_count": len(all_step_seconds),
        "average_seconds_per_step": summary["mean"],
        "average_milliseconds_per_step": summary["mean"] * 1000.0,
        "step_seconds_summary": summary,
    }


def write_benchmark_result(
    args,
    elapsed_seconds_samples,
    runner=None,
    stage_timing_samples=None,
    runner_initialization_seconds=None,
    warmup_elapsed_seconds=None,
):
    if not args.benchmark_result_path or not is_output_rank():
        return

    stage_timing_samples = stage_timing_samples or []
    elapsed_summary = _summarize_benchmark_samples(elapsed_seconds_samples)
    result_path = Path(args.benchmark_result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_cls": args.model_cls,
        "task": args.task,
        "warmup_iterations": args.benchmark_warmup_iters,
        "measure_iterations": len(elapsed_seconds_samples),
        "elapsed_seconds": elapsed_summary["mean"],
        "elapsed_seconds_samples": elapsed_seconds_samples,
        "elapsed_seconds_summary": elapsed_summary,
        "timing_scope": "steady_state_run_pipeline_including_result_save_world_max",
        "runner_initialization_included": False,
        "world_size": dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "seed": args.seed,
        "config_json": args.config_json,
        "save_result_path": args.save_result_path,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if runner_initialization_seconds is not None:
        payload["runner_initialization_seconds"] = runner_initialization_seconds
    if warmup_elapsed_seconds is not None:
        payload["warmup_elapsed_seconds"] = warmup_elapsed_seconds
    stage_timings = getattr(runner, "last_stage_timings", None)
    if isinstance(stage_timings, dict):
        payload["stage_timings"] = stage_timings
    if stage_timing_samples:
        payload["stage_timing_samples"] = stage_timing_samples
        payload["stage_timings_summary"] = _summarize_stage_samples(stage_timing_samples)
        if any("ar_prefill_seconds" in sample or "denoise_step_seconds_samples" in sample for sample in stage_timing_samples):
            payload["fine_grained_metric_definitions"] = {
                "ar_prefill_seconds": "First model-backed AR iteration: full prompt/KV fill through first sampled-token broadcast; CUDA-event critical path, max across ranks.",
                "ar_decode_milliseconds_per_token": "Mean of later model-backed AR iterations; forced transition tokens that do not invoke the model are excluded; CUDA-event critical path, max across ranks.",
                "denoise_step_seconds": "Mean complete scheduler iteration, including both conditional and unconditional forwards under serial CFG, prediction broadcast, and latent update; CUDA-event critical path, max across ranks.",
                "elapsed_seconds": "Full run_pipeline wall time including VAE decode and result save, excluding runner/weight initialization; max across ranks.",
            }
        ar_token_summary = _summarize_ar_tokens(stage_timing_samples)
        if ar_token_summary is not None:
            payload["ar_token_summary"] = ar_token_summary
        ar_decode_token_summary = _summarize_ar_decode_tokens(stage_timing_samples)
        if ar_decode_token_summary is not None:
            payload["ar_decode_token_summary"] = ar_decode_token_summary
        denoise_step_summary = _summarize_denoise_steps(stage_timing_samples)
        if denoise_step_summary is not None:
            payload["denoise_step_summary"] = denoise_step_summary

    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"[Benchmark] timing result saved to: {result_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="The seed for random generator")
    parser.add_argument(
        "--model_cls",
        type=str,
        required=True,
        choices=[
            "wan2.1",
            "wan2.1_distill",
            "wan2.1_mean_flow_distill",
            "wan_dancer",
            "wan2.1_vace",
            "wan2.1_sf",
            "wan2.1_sf_mtxg2",
            "seko_talk",
            "seko_talk_ar",
            "wan2.2_moe",
            "lingbot_world",
            "wan2.2",
            "wan2.2_matrix_game3",
            "wan2.2_moe_audio",
            "wan2.2_audio",
            "wan2.2_moe_distill",
            "wan2.2_moe_vace",
            "qwen_image",
            "ernie_image",
            "ernie_image_turbo",
            "hidream_o1_image",
            "longcat_image",
            "cosmos3",
            "wan2.2_animate",
            "wan2.2_s2v",
            "hunyuan_video_1.5",
            "hunyuan_video_1.5_distill",
            "hunyuan_image3",
            "hunyuan3d",
            "worldplay_distill",
            "worldplay_ar",
            "worldplay_bi",
            "z_image",
            "flux2_klein",
            "flux2_dev",
            "ltx2",
            "ltx2_ar",
            "minimax_h3",
            "bagel",
            "sensenova_vision",
            "seedvr2",
            "neopp",
            "motus",
            "lingbot_world_fast",
            "worldmirror",
            "lingbot_va",
            "dreamzero",
            "infinitetalk",
            "fastwam",
            "lingbot_video",
        ],
        default="wan2.1",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "t2v",
            "i2v",
            "t2t",
            "t2i",
            "ti2t",
            "ti2i",
            "i2i",
            "flf2v",
            "vace",
            "animate",
            "s2v",
            "rs2v",
            "t2av",
            "i2av",
            "l2av",
            "fl2av",
            "ref2av",
            "i2va",
            "v2av",
            "ltx2_s2v",
            "sr",
            "recon",
            "i23d",
            "omni_vision_task",
        ],
        default="t2v",
    )
    parser.add_argument("--support_tasks", type=str, nargs="+", default=[], help="Set supported tasks for the model")
    parser.add_argument(
        "--omni_vision_subtask",
        type=str,
        choices=OMNI_VISION_SUBTASK_CHOICES,
        default=None,
        help="SenseNova-Vision subtask used with --task omni_vision_task.",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--use_prompt_enhancer", action="store_true")
    parser.add_argument("--warmup", action="store_true", help="Warm up the model before inference. Disabled by default.")
    parser.add_argument(
        "--benchmark_warmup_iters",
        type=int,
        default=0,
        help="Run this many unmeasured full-pipeline iterations after runner/weight initialization.",
    )
    parser.add_argument(
        "--benchmark_measure_iters",
        type=int,
        default=1,
        help="Number of synchronized steady-state inference iterations to measure after warmup.",
    )
    parser.add_argument(
        "--benchmark_result_path",
        type=str,
        default=None,
        help="Write rank-0 benchmark samples and summaries to this JSON file.",
    )
    parser.add_argument("--prompt", type=str, default="", help="The input prompt for text-to-video generation")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="The path to input image file(s), including HunyuanImage3 ti2t/ti2i and MiniMax-H3 ref2av reference images. Multiple paths should be comma-separated. Example: 'path1.jpg,path2.jpg'",
    )
    parser.add_argument("--state_path", type=str, default="", help="The path to input robot state file for robot i2v/i2va inference.")
    parser.add_argument("--last_frame_path", type=str, default="", help="The path to last frame file for first-last-frame-to-video (flf2v) task")
    parser.add_argument(
        "--audio_path",
        type=str,
        default="",
        help="Input audio path: Wan s2v / rs2v, LTX-2 ltx2_s2v, or MiniMax-H3 ref2av reference audio. H3 accepts comma-separated paths.",
    )
    parser.add_argument("--image_strength", type=str, default="1.0", help="i2av: single float, or comma-separated floats (one per image, or one value broadcast). Example: 1.0 or 1.0,0.85,0.9")
    parser.add_argument(
        "--i2i_denoise_strength",
        type=float,
        default=None,
        help="(i2i) Single-image edit denoising strength in [0.0, 1.0]. 0.0 preserves the source image most; 1.0 redraws most. Omit to keep the model's existing behavior.",
    )
    parser.add_argument(
        "--image_frame_idx", type=str, default="", help="i2av: comma-separated pixel frame indices (one per image). Omit or empty to evenly space frames in [0, num_frames-1]. Example: 0,40,80"
    )
    # [Warning] For vace task, need refactor.
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None.",
    )
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.",
    )
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.",
    )
    parser.add_argument(
        "--src_pose_path",
        type=str,
        default="",
        help="Pose driving video for Wan s2v / animate (e.g. examples/pose.mp4).",
    )
    parser.add_argument(
        "--src_face_path",
        type=str,
        default=None,
        help="The file of the source face. Default None.",
    )
    parser.add_argument(
        "--src_bg_path",
        type=str,
        default=None,
        help="The file of the source background. Default None.",
    )
    parser.add_argument(
        "--src_mask_path",
        type=str,
        default=None,
        help="The file of the source mask. Default None.",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default=None,
        help="Pose string (e.g., 'w-3, right-0.5') or JSON file path for WorldPlay models.",
    )
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="Directory path for lingbot camera/action control files (poses.npy, intrinsics.npy, optional action.npy).",
    )
    parser.add_argument("--action_mode", type=str, default=None, choices=["forward_dynamics", "inverse_dynamics", "policy"], help="Cosmos3 action mode.")
    parser.add_argument("--domain_name", type=str, default=None, help="Cosmos3 action embodiment domain name.")
    parser.add_argument("--view_point", type=str, default=None, help="Cosmos3 action viewpoint label.")
    parser.add_argument("--action_chunk_size", type=int, default=None, help="Cosmos3 action chunk size.")
    parser.add_argument("--action_chunk_index", type=int, default=None, help="Cosmos3 action chunk index when action_path contains action_chunks.")
    parser.add_argument(
        "--action_ckpt",
        type=str,
        default=None,
        help="Path to action model checkpoint for WorldPlay models.",
    )
    # WorldMirror (3D reconstruction) specific
    parser.add_argument("--input_path", type=str, default=None, help="(worldmirror/recon) Path to a directory of images, a video file, or a single image.")
    parser.add_argument("--strict_output_path", type=str, default=None, help="(worldmirror/recon) If set, write outputs directly here instead of under save_result_path/<subdir>/<timestamp>/.")
    parser.add_argument("--prior_cam_path", type=str, default=None, help="(worldmirror/recon) Optional camera prior JSON (extrinsics + intrinsics).")
    parser.add_argument("--prior_depth_path", type=str, default=None, help="(worldmirror/recon) Optional depth prior directory (one .npy/.png per image).")
    parser.add_argument("--subfolder", type=str, default=None, help="(worldmirror/recon) Subfolder inside model_path containing weights. Overrides config.")
    parser.add_argument("--disable_heads", type=str, nargs="*", default=None, help="(worldmirror/recon) Heads to disable: any of camera depth normal points gs.")
    parser.add_argument("--enable_bf16", action="store_true", default=False, help="(worldmirror/recon) Run the WorldMirror model in bf16.")
    parser.add_argument("--save_rendered", action="store_true", default=False, help="(worldmirror/recon) Render an interpolated fly-through video from Gaussian splats.")
    parser.add_argument("--render_interp_per_pair", type=int, default=None, help="(worldmirror/recon) Interpolated frames per camera pair for --save_rendered.")
    parser.add_argument("--render_depth", action="store_true", default=False, help="(worldmirror/recon) Also render a depth video with --save_rendered.")
    parser.add_argument("--wm_config_path", type=str, default=None, help="(worldmirror/recon) Optional training YAML (pair with --wm_ckpt_path).")
    parser.add_argument("--wm_ckpt_path", type=str, default=None, help="(worldmirror/recon) Optional .ckpt/.safetensors (pair with --wm_config_path).")

    parser.add_argument("--save_result_path", type=str, default=None, help="The path to save video path/file")
    parser.add_argument("--save_action_path", type=str, default=None, help="The path to save action predictions for Motus, LingBot-VA, or DreamZero.")
    parser.add_argument("--return_result_tensor", action="store_true", help="Whether to return result tensor. (Useful for comfyui)")
    parser.add_argument("--target_shape", type=int, nargs="+", default=[], help="Set return video or image shape")
    parser.add_argument("--aspect_ratio", type=str, default="")
    parser.add_argument(
        "--keep_original_aspect",
        action="store_true",
        help="(i2i) When exactly one reference image is provided, preserve its aspect ratio with max_size=2048.",
    )
    parser.add_argument(
        "--layout_bboxes",
        type=str,
        default="",
        help="(i2i) Layout boxes as a JSON string or JSON file path for HiDream layout-conditioned editing.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Input video path for sr/v2v/v2av, or MiniMax-H3 ref2av reference video. H3 accepts comma-separated paths. For v2av this is the pre-processed control/reference video (pose / canny / depth / motion-track for motion-transfer, or the degraded source video for ICEdit).",
    )
    parser.add_argument("--sr_ratio", type=float, default=2.0, help="super resolution ratio for sr task")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=None,
        help="Override the number of Matrix-Game-3 generation segments. Final video length follows 57 + 40 * (num_iterations - 1).",
    )
    parser.add_argument(
        "--reference_video_strength", type=float, default=1.0, help="(v2av) IC-LoRA reference-video conditioning strength in [0.0, 1.0]. 1.0 = full adherence to the control signal, 0.0 = ignore it."
    )
    parser.add_argument("--reference_video_frame_cap", type=int, default=None, help="(v2av) Maximum number of frames to read from the reference/control video. Defaults to the full clip.")
    parser.add_argument("--mux_audio_video_path", type=str, default=None, help="(v2av, optional) After saving, mux audio from this file into the output mp4 (ffmpeg). ")

    args = parser.parse_args()
    if args.benchmark_warmup_iters < 0:
        parser.error("--benchmark_warmup_iters must be greater than or equal to 0")
    if args.benchmark_measure_iters < 1:
        parser.error("--benchmark_measure_iters must be greater than or equal to 1")

    benchmark_enabled = bool(args.benchmark_result_path) or args.benchmark_warmup_iters > 0 or args.benchmark_measure_iters != 1
    seed_all(args.seed)

    # set config
    config = set_config(args)
    stage_timing_env = os.getenv("LIGHTX2V_HUNYUAN_BENCHMARK_STAGE_TIMING", "").strip().lower()
    if args.model_cls == "hunyuan_image3" and (benchmark_enabled or stage_timing_env in {"1", "true", "yes", "on"}):
        config["benchmark_stage_timing"] = True
    config["warmup"] = args.warmup
    # init input_info
    input_info = init_empty_input_info(args.task, args.support_tasks)

    if config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()
        set_parallel_config(config)

    print_config(config)

    validate_config_paths(config)

    with ProfilingContext4DebugL1("Total Cost"):
        # init runner
        runner_initialization_seconds = None
        if benchmark_enabled:
            benchmark_synchronize()
            runner_initialization_start = time.perf_counter()
        runner = init_runner(config)
        if benchmark_enabled:
            benchmark_synchronize()
            runner_initialization_seconds = time.perf_counter() - runner_initialization_start
            if is_output_rank():
                logger.info(f"[Benchmark] runner/weight initialization completed in {runner_initialization_seconds:.6f} seconds; this time is excluded from measured inference samples")
        # start to infer
        data = args.__dict__
        update_input_info_from_dict(input_info, data)

        if benchmark_enabled:
            warmup_input_info = replace(input_info, save_result_path=None, return_result_tensor=False)
            benchmark_synchronize()
            warmup_start = time.perf_counter()
            for warmup_index in range(args.benchmark_warmup_iters):
                seed_all(args.seed)
                benchmark_synchronize()
                if is_output_rank():
                    logger.info(f"[Benchmark] warmup {warmup_index + 1}/{args.benchmark_warmup_iters}, task={args.task}")
                runner.run_pipeline(warmup_input_info)
                benchmark_synchronize()
            warmup_elapsed_seconds = time.perf_counter() - warmup_start

            elapsed_seconds_samples = []
            stage_timing_samples = []
            for measure_index in range(args.benchmark_measure_iters):
                if is_output_rank():
                    logger.info(f"[Benchmark] measured inference {measure_index + 1}/{args.benchmark_measure_iters} started, task={args.task}")
                seed_all(args.seed)
                benchmark_synchronize()
                benchmark_start = time.perf_counter()
                runner.run_pipeline(input_info)
                benchmark_synchronize()
                local_elapsed_seconds = time.perf_counter() - benchmark_start
                sample_elapsed_seconds = _distributed_max_values([local_elapsed_seconds])[0]
                elapsed_seconds_samples.append(sample_elapsed_seconds)

                stage_timings = getattr(runner, "last_stage_timings", None)
                stage_timings = aggregate_fine_grained_stage_timings(stage_timings)
                if isinstance(stage_timings, dict):
                    runner.last_stage_timings = stage_timings
                if isinstance(stage_timings, dict):
                    stage_timing_samples.append(dict(stage_timings))

                if is_output_rank():
                    sample_message = f"[Benchmark] task={args.task}, sample={measure_index + 1}/{args.benchmark_measure_iters}, steady_state_e2e={sample_elapsed_seconds:.6f}s"
                    if isinstance(stage_timings, dict):
                        if "ar_seconds" in stage_timings:
                            sample_message += f", ar={stage_timings['ar_seconds']:.6f}s"
                        if "denoise_seconds" in stage_timings:
                            sample_message += f", denoise={stage_timings['denoise_seconds']:.6f}s"
                        if "ar_plus_denoise_seconds" in stage_timings:
                            sample_message += f", ar+denoise={stage_timings['ar_plus_denoise_seconds']:.6f}s"
                        if "ar_milliseconds_per_token" in stage_timings:
                            sample_message += f", ar_per_token={stage_timings['ar_milliseconds_per_token']:.6f}ms"
                        if "ar_prefill_seconds" in stage_timings:
                            sample_message += f", ar_prefill={stage_timings['ar_prefill_seconds']:.6f}s"
                        if "ar_decode_milliseconds_per_token" in stage_timings:
                            sample_message += f", ar_decode={stage_timings['ar_decode_milliseconds_per_token']:.6f}ms/token"
                        if "denoise_step_seconds" in stage_timings:
                            sample_message += f", denoise_step={stage_timings['denoise_step_seconds'] * 1000.0:.6f}ms"
                    logger.info(sample_message)

            elapsed_summary = _summarize_benchmark_samples(elapsed_seconds_samples)
            stage_summary = _summarize_stage_samples(stage_timing_samples)
            ar_token_summary = _summarize_ar_tokens(stage_timing_samples)
            ar_decode_token_summary = _summarize_ar_decode_tokens(stage_timing_samples)
            denoise_step_summary = _summarize_denoise_steps(stage_timing_samples)
            if is_output_rank():
                logger.info(
                    f"[Benchmark] aggregate steady_state_e2e: mean={elapsed_summary['mean']:.6f}s, "
                    f"median={elapsed_summary['median']:.6f}s, min={elapsed_summary['min']:.6f}s, "
                    f"max={elapsed_summary['max']:.6f}s"
                )
                for key, label in (
                    ("ar_seconds", "AR"),
                    ("denoise_seconds", "denoise"),
                    ("ar_plus_denoise_seconds", "AR+denoise"),
                    ("pipeline_before_result_save_seconds", "pipeline_before_result_save"),
                ):
                    if key in stage_summary:
                        logger.info(f"[Benchmark] aggregate {label}: mean={stage_summary[key]['mean']:.6f}s")
                if ar_token_summary is not None:
                    logger.info(
                        "[Benchmark] aggregate AR per generated token: "
                        f"{ar_token_summary['average_milliseconds_per_token']:.6f}ms/token "
                        f"({ar_token_summary['total_generated_tokens']} tokens across "
                        f"{ar_token_summary['sample_count']} samples)"
                    )
                if ar_decode_token_summary is not None:
                    logger.info(
                        "[Benchmark] aggregate AR decode: "
                        f"{ar_decode_token_summary['average_milliseconds_per_token']:.6f}ms/model-token "
                        f"({ar_decode_token_summary['total_decode_model_tokens']} tokens across "
                        f"{ar_decode_token_summary['sample_count']} samples)"
                    )
                if denoise_step_summary is not None:
                    logger.info(
                        "[Benchmark] aggregate denoise step: "
                        f"{denoise_step_summary['average_milliseconds_per_step']:.6f}ms/step "
                        f"({denoise_step_summary['total_step_count']} steps across "
                        f"{denoise_step_summary['sample_count']} samples)"
                    )

            write_benchmark_result(
                args,
                elapsed_seconds_samples,
                runner=runner,
                stage_timing_samples=stage_timing_samples,
                runner_initialization_seconds=runner_initialization_seconds,
                warmup_elapsed_seconds=warmup_elapsed_seconds,
            )
        else:
            runner.run_pipeline(input_info)

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
