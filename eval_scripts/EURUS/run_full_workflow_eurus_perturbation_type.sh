#!/bin/bash
set -e

# Resolve repo root relative to this script so it works from any cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# --- Hugging Face download acceleration ---
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- vLLM multiprocessing start method ---
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- OpenRouter concurrency (used by RepStiff) ---
OPENROUTER_WORKERS="${OPENROUTER_WORKERS:-8}"
export OPENROUTER_SIMILAR_WORKERS="$OPENROUTER_WORKERS"
export OPENROUTER_INCOMPLETE_WORKERS="$OPENROUTER_WORKERS"
export OPENROUTER_PARAPHRASE_WORKERS="$OPENROUTER_WORKERS"

# --- CONFIGURATION ---
HF_MODEL_ID="PRIME-RL/Eurus-2-7B-PRIME"
MODEL_PATH="$HF_MODEL_ID"
MODEL_NAME="Eurus-2-7B-PRIME_eurus"
DATA_ROOT_DIR="${ROOT}/benchmarks/EURUS"
DATASET_TAG="eurus_member"

METHODS_TO_RUN=("self_critique" "dime" "consistency")

TEMPERATURE_RANDOM=0.8
NUM_RANDOM_SAMPLES=10
TENSOR_PARALLEL_SIZE=1
MAX_NEW_TOKENS=4096
BATCH_SIZE=100

SUBSET_SOURCE=""
NUM_SAMPLES_PER_SOURCE=-1

SUBSET_TAG="_${SUBSET_SOURCE:-all}"
SAMPLE_TAG="_n${NUM_SAMPLES_PER_SOURCE}"
if [ "$NUM_SAMPLES_PER_SOURCE" -lt 0 ]; then
    SAMPLE_TAG="_all_samples"
fi

RESULTS_DIR="${ROOT}/final_results/${MODEL_NAME}/${DATASET_TAG}/${SUBSET_TAG}_${SAMPLE_TAG}"
mkdir -p "$RESULTS_DIR"

GENERATED_DATA_FILE="${RESULTS_DIR}/generated_data.jsonl"

REP_STIFF_COMBINED_FIXED="${REP_STIFF_COMBINED_FIXED:-1}"
REP_STIFF_COMBINED_RULE="${REP_STIFF_COMBINED_RULE:-trend_v1}"
REP_STIFF_COMBINED_WEIGHTS_JSON="${REP_STIFF_COMBINED_WEIGHTS_JSON:-}"

K_BLANKS=1
V4_ALPHA_DEFAULT="0.0"

NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-28}"
REP_STIFF_LAYERS="$(python -c "import sys; print(','.join(f'L{i}' for i in range(int(sys.argv[1]))))" "$NUM_HIDDEN_LAYERS")"

REP_STIFF_LARA_EPS="${REP_STIFF_LARA_EPS:-1e-8}"
REP_STIFF_LARA_CLEAN_REF="${REP_STIFF_LARA_CLEAN_REF:-}"
REP_STIFF_LARA_MIX_BETA="${REP_STIFF_LARA_MIX_BETA:-0.65}"
REP_STIFF_LARA_ROBUST_LAYER_WINDOW="${REP_STIFF_LARA_ROBUST_LAYER_WINDOW:-all}"
REP_STIFF_LARA_ROBUST_DC_WEIGHT="${REP_STIFF_LARA_ROBUST_DC_WEIGHT:-1.0}"

# Perturbation-type ablation: slug -> --rep_stiff_incomplete_blank_strategy value
#   info_rem          : [BLANK] information removal (importance-based)
#   num_replace       : replace one number with another
#   var_rename        : replace one variable with a random word
#   distractor_insert : insert one distractor sentence
PERTURBATION_TYPES=(
    "info_rem:important"
    "num_replace:num_replace"
    "var_rename:var_rename"
    "distractor_insert:distractor"
)

LARA_SUMMARY_JSON="${RESULTS_DIR}/perturbation_type/lara_scores_summary.json"

echo "======================================================"
echo "  EURUS workflow — perturbation-type LaRA ablation"
echo "  Subset: ${SUBSET_SOURCE:-all}, Samples: ${NUM_SAMPLES_PER_SOURCE}"
echo "======================================================"

echo "--> Resolving Hugging Face model snapshot..."
MODEL_PATH="$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("PRIME-RL/Eurus-2-7B-PRIME"))
PY
)"

CMD_ARGS=""
if [ -n "$SUBSET_SOURCE" ]; then
    CMD_ARGS="$CMD_ARGS --subset_source $SUBSET_SOURCE"
fi
if [ "$NUM_SAMPLES_PER_SOURCE" -ge 0 ]; then
    CMD_ARGS="$CMD_ARGS --num_samples_per_source $NUM_SAMPLES_PER_SOURCE"
fi

if [ ! -s "$GENERATED_DATA_FILE" ]; then
    echo "--> Step 1: Generating model responses (shared across perturbation types)..."
    python "${ROOT}/generate_full_data.py" \
        --model_path "$MODEL_PATH" \
        --data_root_dir "$DATA_ROOT_DIR" \
        --output_file "$GENERATED_DATA_FILE" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --max_tokens "$MAX_NEW_TOKENS" \
        --temperature_random "$TEMPERATURE_RANDOM" \
        --num_random_samples "$NUM_RANDOM_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --methods_to_run "${METHODS_TO_RUN[@]}" \
        $CMD_ARGS

    if [ ! -s "$GENERATED_DATA_FILE" ]; then
        echo "[error] Step 1 produced no output: $GENERATED_DATA_FILE"
        exit 1
    fi
else
    echo "--> Step 1: Reusing existing generated data: $GENERATED_DATA_FILE"
fi

FREE_GPU_BEFORE_STEP2="${FREE_GPU_BEFORE_STEP2:-1}"
REQUIRED_FREE_MB="${REQUIRED_FREE_MB:-2000}"
if [ "$FREE_GPU_BEFORE_STEP2" = "1" ]; then
    echo "--> Cleaning up vLLM/Ray workers before Step 2..."
    pkill -f "vllm" || true
    pkill -f "VLLM" || true
    pkill -f "ray::" || true
    pkill -f "raylet" || true

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "--> Waiting for GPU memory to free (>= ${REQUIRED_FREE_MB} MB)..."
        for _ in $(seq 1 60); do
            free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1{print $1}')"
            if [ -n "$free_mb" ] && [ "$free_mb" -ge "$REQUIRED_FREE_MB" ]; then
                echo "--> GPU free memory: ${free_mb} MB"
                break
            fi
            sleep 2
        done
    fi
fi

COMBINED_ARGS=""
if [ -n "$REP_STIFF_COMBINED_WEIGHTS_JSON" ]; then
    COMBINED_ARGS="$COMBINED_ARGS --rep_stiff_combined_weights \"$REP_STIFF_COMBINED_WEIGHTS_JSON\""
fi
if [ "$REP_STIFF_COMBINED_FIXED" = "1" ]; then
    COMBINED_ARGS="$COMBINED_ARGS --rep_stiff_combined_fixed --rep_stiff_combined_rule \"$REP_STIFF_COMBINED_RULE\""
fi

LARA_ARGS=""
if [ -n "$REP_STIFF_LARA_CLEAN_REF" ]; then
    LARA_ARGS="$LARA_ARGS --rep_stiff_lara_clean_ref \"$REP_STIFF_LARA_CLEAN_REF\""
fi

mkdir -p "${RESULTS_DIR}/perturbation_type"

for entry in "${PERTURBATION_TYPES[@]}"; do
    PERTURB_SLUG="${entry%%:*}"
    PERTURB_STRATEGY="${entry##*:}"

    RUN_DIR="${RESULTS_DIR}/perturbation_type/${PERTURB_SLUG}/k${K_BLANKS}"
    mkdir -p "$RUN_DIR"

    echo ""
    echo "======================================================"
    echo "  Perturbation: ${PERTURB_SLUG} (strategy=${PERTURB_STRATEGY})"
    echo "  Output: ${RUN_DIR}"
    echo "======================================================"

    python "${ROOT}/evaluate_all_methods.py" \
        --input_file "$GENERATED_DATA_FILE" \
        --output_summary_json "${RUN_DIR}/evaluation_summary.json" \
        --output_plot "${RUN_DIR}/performance_plot.png" \
        --rep_stiff_model_name "$HF_MODEL_ID" \
        --rep_stiff_max_workers "$OPENROUTER_WORKERS" \
        --rep_stiff_layers "$REP_STIFF_LAYERS" \
        --rep_stiff_scores_json "${RUN_DIR}/rep_stiff_scores.json" \
        --rep_stiff_output_dir "${ROOT}/rep_stiff_outputs_eurus_perturb_${PERTURB_SLUG}_k${K_BLANKS}" \
        --rep_stiff_incomplete_blank_strategy "$PERTURB_STRATEGY" \
        --rep_stiff_incomplete_num_blanks "$K_BLANKS" \
        --rep_stiff_combined_v4_alpha "$V4_ALPHA_DEFAULT" \
        --rep_stiff_lara_eps "$REP_STIFF_LARA_EPS" \
        --rep_stiff_lara_mix_beta "$REP_STIFF_LARA_MIX_BETA" \
        --rep_stiff_lara_robust_layer_window "$REP_STIFF_LARA_ROBUST_LAYER_WINDOW" \
        --rep_stiff_lara_robust_dc_weight "$REP_STIFF_LARA_ROBUST_DC_WEIGHT" \
        $(eval echo "$COMBINED_ARGS") \
        $(eval echo "$LARA_ARGS")
done

echo ""
echo "--> Aggregating LaRA scores across perturbation types..."
RESULTS_DIR="$RESULTS_DIR" K_BLANKS="$K_BLANKS" LARA_SUMMARY_JSON="$LARA_SUMMARY_JSON" python - <<'PY'
import json
import os
from pathlib import Path

results_dir = Path(os.environ["RESULTS_DIR"]) / "perturbation_type"
k_blanks = os.environ["K_BLANKS"]
out_path = Path(os.environ["LARA_SUMMARY_JSON"])

slugs = ("info_rem", "num_replace", "var_rename", "distractor_insert")
lara_keys = (
    "rep_stiff_lara",
    "rep_stiff_lara_robust",
    "self_critique_rep_stiff_lara_mix",
    "self_critique_rep_stiff_lara_robust_mix",
)

summary = {}
for slug in slugs:
    eval_path = results_dir / slug / f"k{k_blanks}" / "evaluation_summary.json"
    if not eval_path.is_file():
        print(f"[warn] missing {eval_path}")
        continue
    with open(eval_path, encoding="utf-8") as f:
        data = json.load(f)
    row = {"evaluation_summary": str(eval_path)}
    for key in lara_keys:
        if key in data:
            row[key] = data[key].get("overall_performance", {})
    summary[slug] = row

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Wrote LaRA summary -> {out_path}")
for slug, row in summary.items():
    lara = row.get("rep_stiff_lara", {})
    robust = row.get("rep_stiff_lara_robust", {})
    print(
        f"  {slug:20s}  LaRA AUC={lara.get('roc_auc')}  TPR@5%FPR={lara.get('tpr_at_fpr_5')}"
        f"  |  robust AUC={robust.get('roc_auc')}  TPR@5%FPR={robust.get('tpr_at_fpr_5')}"
    )
PY

echo ""
echo "======================================================"
echo "  Workflow completed."
echo "  Per-type results: ${RESULTS_DIR}/perturbation_type/<type>/k${K_BLANKS}/"
echo "  LaRA summary:     ${LARA_SUMMARY_JSON}"
echo "======================================================"
