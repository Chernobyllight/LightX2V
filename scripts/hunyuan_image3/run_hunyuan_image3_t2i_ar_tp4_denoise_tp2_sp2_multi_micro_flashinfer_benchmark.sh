#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_multi_micro_flashinfer_benchmark.sh [options]

Topology: 4 GPUs; AR TP4; denoise TP2+SP2; CFG runs serially (no CFG parallel group).

Options:
  --warmup-iters N         Full-pipeline warmup iterations (default: 2).
  --measure-iters N        Timed full-pipeline iterations (default: 3).
  --save-result-path PATH  Generated image path.
  --result-path PATH       Benchmark JSON path.
  -h, --help               Show this help message.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="$(cd -- "${script_dir}/../.." && pwd)"
model_path="${HUNYUAN_IMAGE3_MODEL_PATH:-${MODEL_PATH:-/data/liuhongda/HunyuanImage-3-Instruct}}"
hunyuan_image3_path="${HUNYUAN_IMAGE3_PATH:-/data/liuhongda/HunyuanImage-3.0}"

benchmark_warmup_iters="${LIGHTX2V_BENCHMARK_WARMUP_ITERS:-2}"
benchmark_measure_iters="${LIGHTX2V_BENCHMARK_MEASURE_ITERS:-3}"
benchmark_name="hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_cfg1_serial_gpu4_multi_micro_flashinfer_benchmark"
benchmark_save_path="${LIGHTX2V_BENCHMARK_SAVE_PATH:-${lightx2v_path}/save_results/${benchmark_name}.png}"
benchmark_result_path="${LIGHTX2V_BENCHMARK_RESULT_PATH:-${lightx2v_path}/inference_time/${benchmark_name}.json}"

while (($# > 0)); do
    case "$1" in
        --warmup-iters)
            [[ $# -ge 2 ]] || { echo "Error: --warmup-iters requires a value." >&2; exit 2; }
            benchmark_warmup_iters="$2"
            shift 2
            ;;
        --measure-iters)
            [[ $# -ge 2 ]] || { echo "Error: --measure-iters requires a value." >&2; exit 2; }
            benchmark_measure_iters="$2"
            shift 2
            ;;
        --save-result-path)
            [[ $# -ge 2 ]] || { echo "Error: --save-result-path requires a value." >&2; exit 2; }
            benchmark_save_path="$2"
            shift 2
            ;;
        --result-path)
            [[ $# -ge 2 ]] || { echo "Error: --result-path requires a value." >&2; exit 2; }
            benchmark_result_path="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "${benchmark_warmup_iters}" =~ ^[0-9]+$ ]] || { echo "Error: --warmup-iters must be an integer greater than or equal to 0." >&2; exit 2; }
[[ "${benchmark_measure_iters}" =~ ^[1-9][0-9]*$ ]] || { echo "Error: --measure-iters must be an integer greater than or equal to 1." >&2; exit 2; }
[[ -n "${benchmark_save_path}" ]] || { echo "Error: --save-result-path must not be empty." >&2; exit 2; }
[[ -n "${benchmark_result_path}" ]] || { echo "Error: --result-path must not be empty." >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ ${#visible_gpus[@]} -ne 4 ]]; then
    echo "Error: hybrid AR TP4 / denoise TP2+SP2 benchmark requires exactly 4 visible GPUs; got '${CUDA_VISIBLE_DEVICES}'." >&2
    exit 2
fi

export PYTHONPATH="${hunyuan_image3_path}:${PYTHONPATH:-}"
export PATH="/opt/conda/bin:${PATH}"
export LIGHTX2V_HUNYUAN_BENCHMARK_STAGE_TIMING=1
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/bin/torchrun}"

source "${lightx2v_path}/scripts/base/base.sh"
export PROFILING_DEBUG_LEVEL=0

mkdir -p "$(dirname -- "${benchmark_save_path}")" "$(dirname -- "${benchmark_result_path}")"

"${TORCHRUN_BIN}" --standalone --nproc_per_node=4 -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_t2i_ar_tp4_denoise_tp2_sp2_multi_micro_flashinfer.json" \
    --prompt "生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗" \
    --save_result_path "${benchmark_save_path}" \
    --benchmark_warmup_iters "${benchmark_warmup_iters}" \
    --benchmark_measure_iters "${benchmark_measure_iters}" \
    --benchmark_result_path "${benchmark_result_path}" \
    --seed 42
