#!/bin/bash
set -euo pipefail

ROOT="/scratch2/mjgwak/rl-data-contamination-mj"

# --- Config (override via env vars if needed) ---
MODELS=(
  "${MODEL_ID_1:-talzoomanzoo/qwen2.5-7b-instruct-kk-best}"
  "${MODEL_ID_2:-Qwen/Qwen2.5-Math-7B}"
)
DATASET_PATH="${DATASET_PATH:-${ROOT}/benchmarks/KK/RLMIA_kk.parquet}"
OUT_DIR="${OUT_DIR:-${ROOT}/reasoning_eval_scripts/data}"

# vLLM / decoding config (match eurus_and_base.sh defaults)
USE_VLLM="${USE_VLLM:-1}"
ASYNC_VLLM="${ASYNC_VLLM:-1}"
BATCH_SIZE="${BATCH_SIZE:-60}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-8}"
SCORE_MAX_IN_FLIGHT="${SCORE_MAX_IN_FLIGHT:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

NUM_SAMPLES="${NUM_SAMPLES:-5}"
DO_SAMPLE="${DO_SAMPLE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

mkdir -p "$OUT_DIR"

dataset_tag="$(basename "$DATASET_PATH")"
dataset_tag="${dataset_tag%.parquet}"

sanitize_tag() {
  # Replace anything not safe for filenames.
  echo "$1" | sed -E 's/[^a-zA-Z0-9._-]+/_/g; s/^_+//; s/_+$//'
}

for MODEL_ID in "${MODELS[@]}"; do
  model_tag="$(sanitize_tag "$MODEL_ID")"
  gen_output="${OUT_DIR}/${dataset_tag}__${model_tag}__generations.json"
  eval_output="${OUT_DIR}/${dataset_tag}__${model_tag}__evaluated.json"

  EXTRA_ARGS=()
  if [ "$USE_VLLM" -eq 1 ]; then
    EXTRA_ARGS+=(--use_vllm --batch_size "$BATCH_SIZE")
    if [ "$ASYNC_VLLM" -eq 1 ]; then
      EXTRA_ARGS+=(--async_vllm --max_in_flight "$MAX_IN_FLIGHT" --score_max_in_flight "$SCORE_MAX_IN_FLIGHT")
      EXTRA_ARGS+=(--gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" --max_num_seqs "$MAX_NUM_SEQS")
    fi
  fi
  if [ "$DO_SAMPLE" -eq 1 ]; then
    EXTRA_ARGS+=(--do_sample --temperature "$TEMPERATURE" --top_p "$TOP_P")
  fi

  python "${ROOT}/reasoning_eval_scripts/src/eurus_evals.py" \
    --model "$MODEL_ID" \
    --dataset_path "$DATASET_PATH" \
    --gen_output "$gen_output" \
    --eval_output "$eval_output" \
    --num_samples "$NUM_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    "${EXTRA_ARGS[@]}"
done
