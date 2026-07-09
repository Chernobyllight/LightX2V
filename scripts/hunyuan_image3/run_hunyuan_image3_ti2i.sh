#!/bin/bash
set -e

# Usage:
#   bash scripts/hunyuan_image3/run_hunyuan_image3_ti2i.sh
#   bash scripts/hunyuan_image3/run_hunyuan_image3_ti2i.sh ti2iv2
#   bash scripts/hunyuan_image3/run_hunyuan_image3_ti2i.sh 0,2,3,4,5,6,7 ti2iv2

GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
DEMO="${DEMO:-ti2i}"

if [ $# -gt 0 ]; then
    case "$1" in
        ti2i|ti2iv2)
            DEMO="$1"
            ;;
        *)
            GPU_IDS="$1"
            ;;
    esac
fi

if [ $# -gt 1 ]; then
    DEMO="$2"
fi

export lightx2v_path="${lightx2v_path:-/data/nvme0/lhd_codes/LightX2V}"
export model_path="${model_path:-/data/nvme0/lhd_codes/HunyuanImage-3.0-instruct/HunyuanImage-3-Instruct}"
export HUNYUAN_IMAGE3_REPO_PATH="${HUNYUAN_IMAGE3_REPO_PATH:-/data/nvme0/lhd_codes/HunyuanImage-3.0}"
export PYTHONPATH="${HUNYUAN_IMAGE3_REPO_PATH}:${PYTHONPATH:-}"

if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L | wc -l)
    if [ "$gpu_count" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count - 1)))
    fi
fi

if [ "$DEMO" = "ti2iv2" ]; then
    prompt="让图1的猫咪与图2的猫咪自拍，图1的猫咪说:“妈妈，我在乡下遇到了好朋喵”，背景为图3。"
    image_path="${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_2_0.png,${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_2_1.png,${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_2_2.png"
    seed=43
    save_result_path="${lightx2v_path}/save_results/hunyuan_image3_ti2iv3.png"
else
    prompt="新年宠物海报，Q版圆润的可爱标题“新年快乐汪”，副标题“HAPPY NEW YEAR”。 鱼眼镜头，背景是房间门口，近景，上传的主体歪头笑，围着红色围巾，戴着红色毛线帽，高清，绒毛细节，面部特写。 宝丽莱相纸，超现实主义，写实主义，胶片摄影，打印颗粒感肌理。肌理，超写实，复古感。"
    image_path="${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_0_0.png"
    seed=42
    save_result_path="${lightx2v_path}/save_results/hunyuan_image3_ti2i.png"
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "Running HunyuanImage3 ${DEMO}"
echo "enable_kv_cache=${enable_kv_cache:-true}"
echo "enable_text_kv_cache=${enable_text_kv_cache:-${enable_kv_cache:-true}}"
echo "use_taylor_cache=${use_taylor_cache:-false}"
echo "moe_impl=${moe_impl:-flashinfer}"
echo "attn_impl=${attn_impl:-torch_sdpa}"
HUNYUAN_CFG_MODE="${hunyuan_cfg_mode:-parallel}"
CFG_PARALLEL_SIZE="${cfg_parallel_size:-2}"
FLASHINFER_AUTOTUNE_MODE="${FLASHINFER_AUTOTUNE_MODE:-${flashinfer_autotune_mode:-tune}}"
FLASHINFER_AUTOTUNE_CACHE="${FLASHINFER_AUTOTUNE_CACHE:-${flashinfer_autotune_cache:-${lightx2v_path}/save_results/hunyuan_image3_flashinfer_autotune_${DEMO}.json}}"
echo "hunyuan_cfg_mode=${HUNYUAN_CFG_MODE}"
echo "cfg_parallel_size=${CFG_PARALLEL_SIZE}"
echo "flashinfer_autotune_mode=${FLASHINFER_AUTOTUNE_MODE}"
echo "flashinfer_autotune_cache=${FLASHINFER_AUTOTUNE_CACHE}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-${lightx2v_path}/save_results/torchrun_logs/hunyuan_image3_${DEMO}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
echo "torchrun_log_dir=${TORCHRUN_LOG_DIR}"

source "${lightx2v_path}/scripts/base/base.sh"

config_json="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_i2i.json"
runtime_config="$config_json"
launcher=(python)
if [ "$HUNYUAN_CFG_MODE" = "parallel" ]; then
    export LIGHTX2V_CFG_PARALLEL_DEVICE_SPLIT=1
    export LIGHTX2V_CFG_PARALLEL_SIZE="$CFG_PARALLEL_SIZE"
    runtime_config="$(mktemp /tmp/hunyuan_image3_ti2i_cfg_parallel.XXXXXX.json)"
    trap 'rm -f "$runtime_config"' EXIT
    python - "$config_json" "$runtime_config" "$CFG_PARALLEL_SIZE" <<'PY'
import json
import sys

base_config, runtime_config, cfg_parallel_size = sys.argv[1:4]
with open(base_config, "r") as f:
    config = json.load(f)

parallel = dict(config.get("parallel") or {})
parallel["cfg_p_size"] = int(cfg_parallel_size)
parallel.setdefault("seq_p_size", 1)
parallel.setdefault("tensor_p_size", 1)
config["parallel"] = parallel
config["enable_cfg"] = True

with open(runtime_config, "w") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
PY
    echo "runtime_config=${runtime_config}"
    launcher=(
        torchrun
        --standalone
        --nnodes=1
        --nproc_per_node="$CFG_PARALLEL_SIZE"
        --log_dir="$TORCHRUN_LOG_DIR"
        --tee=3
    )
fi

"${launcher[@]}" -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task i2i \
    --model_path "$model_path" \
    --config_json "$runtime_config" \
    --prompt "$prompt" \
    --image_path "$image_path" \
    --save_result_path "$save_result_path" \
    --seed "$seed" \
    --moe_impl "${moe_impl:-flashinfer}" \
    --hunyuan_cfg_mode "${HUNYUAN_CFG_MODE}" \
    --attn_impl "${attn_impl:-torch_sdpa}" \
    --flashinfer_autotune_mode "${FLASHINFER_AUTOTUNE_MODE}" \
    --flashinfer_autotune_cache "${FLASHINFER_AUTOTUNE_CACHE}" \
    --flashinfer_tune_max_num_tokens "${flashinfer_tune_max_num_tokens:-16384}" \
    --flashinfer_tuning_buckets "${flashinfer_tuning_buckets:-128,256,512,1024,2048,4096,8192,12288,16384}" \
    --flashinfer_autotune_round_up "${flashinfer_autotune_round_up:-true}" \
    --enable_kv_cache "${enable_kv_cache:-true}" \
    --enable_text_kv_cache "${enable_text_kv_cache:-${enable_kv_cache:-true}}" \
    --use_taylor_cache "${use_taylor_cache:-false}" \
    --taylor_cache_interval "${taylor_cache_interval:-5}" \
    --taylor_cache_order "${taylor_cache_order:-2}" \
    --taylor_cache_enable_first_enhance "${taylor_cache_enable_first_enhance:-false}" \
    --taylor_cache_first_enhance_steps "${taylor_cache_first_enhance_steps:-3}" \
    --taylor_cache_enable_tailing_enhance "${taylor_cache_enable_tailing_enhance:-false}" \
    --taylor_cache_tailing_enhance_steps "${taylor_cache_tailing_enhance_steps:-1}" \
    --taylor_cache_low_freqs_order "${taylor_cache_low_freqs_order:-2}" \
    --taylor_cache_high_freqs_order "${taylor_cache_high_freqs_order:-2}"
