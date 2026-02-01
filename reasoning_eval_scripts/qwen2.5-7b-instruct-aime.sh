#!/bin/bash
set -euo pipefail

ROOT="/scratch2/mjgwak/rl-data-contamination-mj"

# --- Config (override via env vars if needed) ---
MODELS=(
  "${MODEL_ID_1:-Qwen/Qwen2.5-7B-Instruct}"
  "${MODEL_ID_2:-talzoomanzoo/qwen2.5-7b-instruct-aime-5k-best}"
)
DATASET_PATH="${DATASET_PATH:-${ROOT}/benchmarks/AIME24/aime24.parquet}"
ANSWERS_JSON="${ANSWERS_JSON:-}"
OUT_DIR="${OUT_DIR:-${ROOT}/reasoning_eval_scripts/data}"
GEN_JSON="${GEN_JSON:-${OUT_DIR}/aime__qwen2.5-7b-instruct__generations.json}"
EVAL_JSON="${EVAL_JSON:-${OUT_DIR}/aime__qwen2.5-7b-instruct__evaluated.json}"

for MODEL_ID in "${MODELS[@]}"; do
  model_tag="$(echo "$MODEL_ID" | tr '/:' '__' | tr -cd 'a-zA-Z0-9._-')"
  GEN_JSON="${GEN_JSON_BASE:-${OUT_DIR}/aime__${model_tag}__generations.json}"
  EVAL_JSON="${EVAL_JSON_BASE:-${OUT_DIR}/aime__${model_tag}__evaluated.json}"

  EXTRA_ARGS=()
  if [ -n "$ANSWERS_JSON" ] && [ -f "$ANSWERS_JSON" ]; then
    EXTRA_ARGS+=(--answers_json "$ANSWERS_JSON")
  fi

  python "${ROOT}/reasoning_eval_scripts/src/aime_evals.py" \
    --model "$MODEL_ID" \
    --dataset_path "$DATASET_PATH" \
    --gen_output "$GEN_JSON" \
    --eval_output "$EVAL_JSON" \
    --do_sample \
    "${EXTRA_ARGS[@]}"
done
