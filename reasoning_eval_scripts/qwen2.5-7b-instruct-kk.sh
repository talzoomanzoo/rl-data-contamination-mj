#!/bin/bash
set -euo pipefail

ROOT="/scratch2/mjgwak/rl-data-contamination-mj"

# --- Config (override via env vars if needed) ---
MODELS=(
  "${MODEL_ID_1:-Qwen/Qwen2.5-7B-Instruct}"
  "${MODEL_ID_2:-talzoomanzoo/qwen2.5-7b-instruct-kk-best}"
)
SAVE_DIR="${SAVE_DIR:-${ROOT}/reasoning_eval_scripts/result_qa}"
CONFIG="${CONFIG:-}"
NTRAIN="${NTRAIN:-0}"
MAX_TOKEN="${MAX_TOKEN:-1024}"
ARCH="${ARCH:-}"
DATASET_PATH="${DATASET_PATH:-${ROOT}/benchmarks/KK/RLMIA_kk.parquet}"
SPLIT="${SPLIT:-test}"
EVAL_NPPL="${EVAL_NPPL:-4}"
PROBLEM_TYPE="${PROBLEM_TYPE:-clean}"
USE_VLLM="${USE_VLLM:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
COT="${COT:-0}"
NO_LINEBREAK="${NO_LINEBREAK:-0}"

for MODEL_ID in "${MODELS[@]}"; do
  EXTRA_ARGS=()
  if [ -n "$ARCH" ]; then
    EXTRA_ARGS+=(--arch "$ARCH")
  fi
  if [ "$USE_VLLM" -eq 1 ]; then
    EXTRA_ARGS+=(--use_vllm --batch_size "$BATCH_SIZE")
  fi
  if [ "$COT" -eq 1 ]; then
    EXTRA_ARGS+=(--cot)
  fi
  if [ "$NO_LINEBREAK" -eq 1 ]; then
    EXTRA_ARGS+=(--no_linebreak)
  fi

  python "${ROOT}/reasoning_eval_scripts/src/kk_evals.py" \
    --model "$MODEL_ID" \
    --save_dir "$SAVE_DIR" \
    --config "$CONFIG" \
    --ntrain "$NTRAIN" \
    --max_token "$MAX_TOKEN" \
    --dataset_path "$DATASET_PATH" \
    --split "$SPLIT" \
    --eval_nppl "$EVAL_NPPL" \
    --problem_type "$PROBLEM_TYPE" \
    "${EXTRA_ARGS[@]}"
done
