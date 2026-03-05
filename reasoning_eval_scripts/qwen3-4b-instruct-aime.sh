#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Allow overriding ROOT from environment; otherwise infer repo root from script location.
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# --- Config (override via env vars if needed) ---
MODEL_1="${MODEL_ID_1:-Qwen/Qwen3-4B-Instruct-2507}"
if [ -n "${MODEL_ID_2:-}" ]; then
  MODEL_2="$MODEL_ID_2"
else
  AUTO_MERGED_ACTOR="/scratch2/mjgwak/rl-data-contamination-mj/models/qwen3-4b-instruct-2507-aime-actor-merged"
  if [ -d "$AUTO_MERGED_ACTOR" ]; then
    MODEL_2="$AUTO_MERGED_ACTOR"
  else
    MODEL_2="talzoomanzoo/qwen3-4b-instruct-2507-aime-actor"
  fi
fi

MODELS=("$MODEL_1" "$MODEL_2")

# Run both AIME24 and AIME25 by default (override with DATASET_PATH to run only one).
DATASET_PATH="${DATASET_PATH:-}"
ANSWERS_JSON="${ANSWERS_JSON:-}"
OUT_DIR="${OUT_DIR:-${ROOT}/reasoning_eval_scripts/data}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

DATASETS=()
if [ -n "$DATASET_PATH" ]; then
  DATASETS+=("$DATASET_PATH")
else
  DATASETS+=("${ROOT}/benchmarks/AIME24/aime24.parquet")
  DATASETS+=("${ROOT}/benchmarks/AIME25/aime25.parquet")
fi

mkdir -p "$OUT_DIR"

for DATASET in "${DATASETS[@]}"; do
  dataset_stem="$(basename "$DATASET")"
  dataset_tag="${dataset_stem%.parquet}"
  for MODEL_ID in "${MODELS[@]}"; do
    model_tag="$(echo "$MODEL_ID" | tr '/:' '__' | tr -cd 'a-zA-Z0-9._-')"
    GEN_JSON="${GEN_JSON_BASE:-${OUT_DIR}/${dataset_tag}__${model_tag}__generations.json}"
    EVAL_JSON="${EVAL_JSON_BASE:-${OUT_DIR}/${dataset_tag}__${model_tag}__evaluated.json}"

    EXTRA_ARGS=()
    if [ -n "$ANSWERS_JSON" ] && [ -f "$ANSWERS_JSON" ]; then
      EXTRA_ARGS+=(--answers_json "$ANSWERS_JSON")
    fi

    echo "======================================================"
    echo "Model:   $MODEL_ID"
    echo "Dataset: $DATASET"
    echo "Out:     $GEN_JSON"
    echo "======================================================"

    if [ -s "$GEN_JSON" ]; then
      echo "--> Found existing generations: $GEN_JSON"
      if [ -s "$EVAL_JSON" ]; then
        echo "--> Found existing evaluation:  $EVAL_JSON"
        echo "--> Skipping (already generated + evaluated)."
        continue
      fi
      echo "--> Skipping generation; running evaluation only..."
      python "${ROOT}/reasoning_eval_scripts/src/aime_evals.py" \
        --input "$GEN_JSON" \
        --output "$EVAL_JSON" \
        "${EXTRA_ARGS[@]}"
      continue
    fi

    python "${ROOT}/reasoning_eval_scripts/src/aime_evals.py" \
      --model "$MODEL_ID" \
      --dataset_path "$DATASET" \
      --gen_output "$GEN_JSON" \
      --eval_output "$EVAL_JSON" \
      --do_sample \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      "${EXTRA_ARGS[@]}"
  done
done