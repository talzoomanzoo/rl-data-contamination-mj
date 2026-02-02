#!/bin/bash
set -euo pipefail

ROOT="/scratch2/mjgwak/rl-data-contamination-mj"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
HF_TRANSFER="${HF_TRANSFER:-1}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${ROOT}/reasoning_eval_scripts/src:${PYTHONPATH:-}"

export HF_HUB_DISABLE_PROGRESS_BARS=0
export TQDM_DISABLE=0
export VLLM_DISABLE_TQDM=0

if [ "$SHOW_PROGRESS" -eq 0 ]; then
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  export TQDM_DISABLE=1
  export VLLM_DISABLE_TQDM=1
fi

if [ "$HF_TRANSFER" -eq 1 ]; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi

# --- Config (override via env vars if needed) ---
MODELS=(
  "${MODEL_ID_1:-Qwen/Qwen2.5-7B-Instruct}"
  "${MODEL_ID_2:-talzoomanzoo/qwen2.5-7b-instruct-sat-best}"
)
DATASET_PATH="${DATASET_PATH:-${ROOT}/benchmarks/SAT/RLMIA_SAT.parquet}"
OUT_DIR="${OUT_DIR:-${ROOT}/reasoning_eval_scripts/sat_outputs}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
USE_VLLM="${USE_VLLM:-1}"
BATCH_SIZE="${BATCH_SIZE:-100}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

START_IDX="${START_IDX:-0}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUT_DIR"

for MODEL_ID in "${MODELS[@]}"; do
  model_tag="$(echo "$MODEL_ID" | tr '/:' '__' | tr -cd 'a-zA-Z0-9._-')"
  GEN_JSON="${GEN_JSON_BASE:-${OUT_DIR}/${model_tag}__generations.json}"
  EVAL_JSON="${EVAL_JSON_BASE:-${OUT_DIR}/${model_tag}__evaluated.json}"
  ACC_JSON="${ACC_JSON_BASE:-${OUT_DIR}/${model_tag}__accuracy.json}"

  if [ "$SKIP_IF_EXISTS" -eq 1 ] && [ -s "$GEN_JSON" ] && [ -s "$EVAL_JSON" ] && [ -s "$ACC_JSON" ]; then
    echo "Skipping $MODEL_ID (outputs already exist)"
    continue
  fi

  EXTRA_ARGS=()
  if [ "$DO_SAMPLE" -eq 1 ]; then
    EXTRA_ARGS+=(--do_sample --temperature "$TEMPERATURE" --top_p "$TOP_P")
  fi
  if [ "$USE_VLLM" -eq 1 ]; then
    EXTRA_ARGS+=(--use_vllm --batch_size "$BATCH_SIZE" --tensor_parallel_size "$TP_SIZE")
    EXTRA_ARGS+=(--max_model_len "$MAX_MODEL_LEN")
  fi
  if [ -n "$LIMIT" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
  fi

  python "${ROOT}/reasoning_eval_scripts/src/sat_evals.py" \
    --model "$MODEL_ID" \
    --dataset_path "$DATASET_PATH" \
    --gen_output "$GEN_JSON" \
    --eval_output "$EVAL_JSON" \
    --acc_output "$ACC_JSON" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --start_idx "$START_IDX" \
    "${EXTRA_ARGS[@]}"
done
