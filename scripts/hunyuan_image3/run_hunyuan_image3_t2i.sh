#!/bin/bash
set -e

# 用法：
#   bash run_hunyuan.sh 0
#   bash run_hunyuan.sh 1,2
#   bash run_hunyuan.sh 3,5,6
#
# 如果不传参数，则使用已有的 CUDA_VISIBLE_DEVICES
# 如果也没有设置 CUDA_VISIBLE_DEVICES，则自动使用所有 GPU
# Cache 开关直接写在 python 入口参数里，测试时改入口处的 true/false/数字即可。

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"

export lightx2v_path="/data/nvme0/lhd_codes/LightX2V"
export model_path="/data/nvme0/lhd_codes/HunyuanImage-3.0-instruct/HunyuanImage-3-Instruct"
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

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "enable_kv_cache=${enable_kv_cache:-true}"
echo "enable_text_kv_cache=${enable_text_kv_cache:-${enable_kv_cache:-true}}"
echo "use_taylor_cache=${use_taylor_cache:-false}"
echo "moe_impl=${moe_impl:-flashinfer}"
echo "attn_impl=${attn_impl:-torch_sdpa}"
HUNYUAN_CFG_MODE="${hunyuan_cfg_mode:-parallel}"
CFG_PARALLEL_SIZE="${cfg_parallel_size:-2}"
FLASHINFER_AUTOTUNE_MODE="${FLASHINFER_AUTOTUNE_MODE:-${flashinfer_autotune_mode:-tune}}"
FLASHINFER_AUTOTUNE_CACHE="${FLASHINFER_AUTOTUNE_CACHE:-${flashinfer_autotune_cache:-${lightx2v_path}/save_results/hunyuan_image3_flashinfer_autotune_t2i.json}}"
echo "hunyuan_cfg_mode=${HUNYUAN_CFG_MODE}"
echo "cfg_parallel_size=${CFG_PARALLEL_SIZE}"
echo "flashinfer_autotune_mode=${FLASHINFER_AUTOTUNE_MODE}"
echo "flashinfer_autotune_cache=${FLASHINFER_AUTOTUNE_CACHE}"
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-${lightx2v_path}/save_results/torchrun_logs/hunyuan_image3_t2i}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
echo "torchrun_log_dir=${TORCHRUN_LOG_DIR}"

source ${lightx2v_path}/scripts/base/base.sh

config_json="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_t2i.json"
runtime_config="$config_json"
launcher=(python)
if [ "$HUNYUAN_CFG_MODE" = "parallel" ]; then
    export LIGHTX2V_CFG_PARALLEL_DEVICE_SPLIT=1
    export LIGHTX2V_CFG_PARALLEL_SIZE="$CFG_PARALLEL_SIZE"
    runtime_config="$(mktemp /tmp/hunyuan_image3_t2i_cfg_parallel.XXXXXX.json)"
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
    --task t2i \
    --model_path "$model_path" \
    --config_json "$runtime_config" \
    --prompt "生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_t2i.png" \
    --seed 42 \
    --moe_impl "${moe_impl:-flashinfer}" \
    --hunyuan_cfg_mode "${HUNYUAN_CFG_MODE}" \
    --attn_impl "${attn_impl:-sdpa}" \
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
