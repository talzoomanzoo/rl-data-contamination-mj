#!/bin/bash
set -e 

# Resolve repo root relative to this script so it works from any cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# --- Hugging Face download acceleration ---
# NOTE: set to 0 unless you have `hf_transfer` installed.
export HF_HUB_ENABLE_HF_TRANSFER=0

# --- vLLM multiprocessing start method ---
# With tensor_parallel_size > 1, vLLM uses multiprocessing; CUDA requires "spawn" (not fork).
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- OpenRouter concurrency (used by RepStiff) ---
# Increase this if OpenRouter is slow and you want parallel requests.
# Be mindful of OpenRouter rate limits.
OPENROUTER_WORKERS="${OPENROUTER_WORKERS:-8}"
export OPENROUTER_SIMILAR_WORKERS="$OPENROUTER_WORKERS"
export OPENROUTER_INCOMPLETE_WORKERS="$OPENROUTER_WORKERS"
export OPENROUTER_PARAPHRASE_WORKERS="$OPENROUTER_WORKERS"

# --- CONFIGURATION ---
HF_MODEL_ID="talzoomanzoo/eurus-epoch0-step8"
MODEL_PATH="$HF_MODEL_ID"
MODEL_NAME="eurus-epoch0-step8"
# EURUS benchmark dataset root.
# `generate_full_data.py` scans recursively for *.jsonl/*.parquet under this directory.
# Use the local eurus_member parquet generated in `benchmarks/EURUS/`.
DATA_ROOT_DIR="${ROOT}/benchmarks/EURUS"
# Used only to namespace outputs (prevents reusing old generated_data.jsonl from other benchmarks).
DATASET_TAG="eurus_member"

METHODS_TO_RUN=("self_critique" "dime" "consistency")

# --- Sampling & VLLM Configuration ---
TEMPERATURE_RANDOM=0.8
NUM_RANDOM_SAMPLES=10
TENSOR_PARALLEL_SIZE=1
MAX_NEW_TOKENS=4096
BATCH_SIZE=100

# Leave empty to run on all sources found under DATA_ROOT_DIR.
# If you want a specific source, set this to match the dataset's `data_source` value.
SUBSET_SOURCE=""
NUM_SAMPLES_PER_SOURCE=-1

SUBSET_TAG="_${SUBSET_SOURCE:-all}"
SAMPLE_TAG="_n${NUM_SAMPLES_PER_SOURCE}"
if [ "$NUM_SAMPLES_PER_SOURCE" -lt 0 ]; then
    SAMPLE_TAG="_all_samples"
fi
FILENAME_TAG="${SUBSET_TAG}${SAMPLE_TAG}"

# --- Output Configuration ---
RESULTS_DIR="${ROOT}/final_results/${MODEL_NAME}/${DATASET_TAG}/${SUBSET_TAG}_${SAMPLE_TAG}"
mkdir -p "$RESULTS_DIR"

GENERATED_DATA_FILE="${RESULTS_DIR}/generated_data.jsonl"
EVAL_SUMMARY_JSON="${RESULTS_DIR}/evaluation_summary.json"
PLOT_PNG="${RESULTS_DIR}/performance_plot.png"
DIME_DETAIL_JSONL="${RESULTS_DIR}/dime_detail_report.jsonl"
REP_STIFF_SCORES_JSON="${RESULTS_DIR}/rep_stiff_scores.json"

# --- RepStiff combined score options ---
# Enables RepStiff "combined" scores (e.g. rep_stiff_combined_score and combined_trend_v*_score).
# - If you have learned weights, set REP_STIFF_COMBINED_WEIGHTS_JSON to a JSON path and it will be used.
# - Otherwise, keep REP_STIFF_COMBINED_FIXED=1 to use fixed rules.
REP_STIFF_COMBINED_FIXED="${REP_STIFF_COMBINED_FIXED:-1}"
REP_STIFF_COMBINED_RULE="${REP_STIFF_COMBINED_RULE:-trend_v1}"
REP_STIFF_COMBINED_WEIGHTS_JSON="${REP_STIFF_COMBINED_WEIGHTS_JSON:-}"

# --- RepStiff incomplete-question blanking controls ---
REP_STIFF_INCOMPLETE_BLANK_STRATEGY="${REP_STIFF_INCOMPLETE_BLANK_STRATEGY:-important}"
REP_STIFF_INCOMPLETE_NUM_BLANKS="${REP_STIFF_INCOMPLETE_NUM_BLANKS:-1}"
# --- WORKFLOW ---
echo "======================================================"
echo "    Starting Final Contamination Detection Workflow"
echo "    Config -> Subset: ${SUBSET_SOURCE:-all}, Samples: ${NUM_SAMPLES_PER_SOURCE}"
echo "======================================================"

echo "--> Resolving Hugging Face model snapshot..."
MODEL_PATH="$(python - <<'PY'
from huggingface_hub import snapshot_download

model_id = "talzoomanzoo/eurus-epoch0-step8"
path = snapshot_download(model_id)
print(path)
PY
)"

CMD_ARGS=""
if [ -n "$SUBSET_SOURCE" ]; then
    CMD_ARGS="$CMD_ARGS --subset_source $SUBSET_SOURCE"
fi
if [ "$NUM_SAMPLES_PER_SOURCE" -ge 0 ]; then
    CMD_ARGS="$CMD_ARGS --num_samples_per_source $NUM_SAMPLES_PER_SOURCE"
fi

PERTURBATION_PREFIX="hello, what's your name?" 
PERTURBATION_SUFFIX="I'm fine, thank you."

echo "--> Step 1: Generating all necessary data..."
python "${ROOT}/generate_full_data.py" \
    --model_path "$MODEL_PATH" \
    --data_root_dir "$DATA_ROOT_DIR" \
    --output_file "$GENERATED_DATA_FILE" \
    --perturbation_prefix "$PERTURBATION_PREFIX" \
    --perturbation_suffix "$PERTURBATION_SUFFIX" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_tokens "$MAX_NEW_TOKENS" \
    --temperature_random "$TEMPERATURE_RANDOM" \
    --num_random_samples "$NUM_RANDOM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --methods_to_run "${METHODS_TO_RUN[@]}" \
    $CMD_ARGS

if [ ! -s "$GENERATED_DATA_FILE" ]; then
    echo "[error] Step 1 produced no output file (or it's empty): $GENERATED_DATA_FILE"
    echo "[error] Check DATA_ROOT_DIR for *.jsonl/*.parquet and that rows contain `prompt`."
    exit 1
fi

echo "--> Step 1 finished. All data saved to '$GENERATED_DATA_FILE'."
echo ""

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

echo "--> Step 2: Evaluating DIME and all baseline methods..."
COMBINED_ARGS=""
if [ -n "$REP_STIFF_COMBINED_WEIGHTS_JSON" ]; then
    COMBINED_ARGS="$COMBINED_ARGS --rep_stiff_combined_weights \"$REP_STIFF_COMBINED_WEIGHTS_JSON\""
fi
if [ "$REP_STIFF_COMBINED_FIXED" = "1" ]; then
    COMBINED_ARGS="$COMBINED_ARGS --rep_stiff_combined_fixed --rep_stiff_combined_rule \"$REP_STIFF_COMBINED_RULE\""
fi

# --- Compare selected RepStiff layer sets for rep_stiff_combined_v4 alpha=0 ---
V4_ALPHA_DEFAULT="0.0"
LAYER_COMPARISON_CSV="${RESULTS_DIR}/rep_stiff_layer_selection_comparison__alpha0.csv"
LAYER_COMPARISON_JSONL="${RESULTS_DIR}/rep_stiff_layer_selection_comparison__alpha0.jsonl"

mkdir -p "${RESULTS_DIR}/layer_selection"

LAYER_CASE_NAMES=("first_last" "quartiles" "fifths")
LAYER_CASE_SPECS=(
    "first,last"
    "first,1/4,2/4,last"
    "first,1/5,2/5,3/5,4/5,last"
)

for i in "${!LAYER_CASE_NAMES[@]}"; do
    CASE_NAME="${LAYER_CASE_NAMES[$i]}"
    LAYER_SPEC="${LAYER_CASE_SPECS[$i]}"
    echo "--> RepStiff layer selection: ${CASE_NAME} (${LAYER_SPEC}), alpha=${V4_ALPHA_DEFAULT}"
    RUN_DIR="${RESULTS_DIR}/layer_selection/${CASE_NAME}"
    mkdir -p "$RUN_DIR"

    python "${ROOT}/evaluate_all_methods.py" \
        --input_file "$GENERATED_DATA_FILE" \
        --output_summary_json "${RUN_DIR}/evaluation_summary.json" \
        --output_plot "${RUN_DIR}/performance_plot.png" \
        --rep_stiff_model_name "$HF_MODEL_ID" \
        --rep_stiff_max_workers "$OPENROUTER_WORKERS" \
        --rep_stiff_layers "$LAYER_SPEC" \
        --rep_stiff_scores_json "${RUN_DIR}/rep_stiff_scores.json" \
        --rep_stiff_output_dir "${ROOT}/rep_stiff_outputs_eurus_epoch0_layers_${CASE_NAME}" \
        --rep_stiff_incomplete_blank_strategy "$REP_STIFF_INCOMPLETE_BLANK_STRATEGY" \
        --rep_stiff_incomplete_num_blanks "$REP_STIFF_INCOMPLETE_NUM_BLANKS" \
        --rep_stiff_combined_v4_alpha "$V4_ALPHA_DEFAULT" \
        $(eval echo "$COMBINED_ARGS")
done

# Consolidate layer-selection results into a single CSV/JSONL.
python - <<'PY' "$RESULTS_DIR" "$LAYER_COMPARISON_CSV" "$LAYER_COMPARISON_JSONL" "$REP_STIFF_INCOMPLETE_BLANK_STRATEGY" "$REP_STIFF_INCOMPLETE_NUM_BLANKS"
import csv
import json
import math
import os
import sys

results_dir, csv_path, jsonl_path, blank_strategy, num_blanks = sys.argv[1:6]

def _num(x):
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None

cases = [
    ("first_last", "first,last"),
    ("quartiles", "first,1/4,2/4,last"),
    ("fifths", "first,1/5,2/5,3/5,4/5,last"),
]
method_names = [
    "rep_stiff_combined_score",
    "rep_stiff_combined_new",
    "rep_stiff_combined_new3",
    "rep_stiff_combined_v4",
    "self_critique_rep_stiff_v4_mix",
    "rep_stiff_combined_v5",
    "self_critique_rep_stiff_v5_mix",
    "rep_stiff_combined_v6",
    "self_critique_rep_stiff_v6_mix",
    "rep_stiff_combined_score_new2",
]
metric_names = [
    "roc_auc",
    "tpr_at_fpr_5",
    "best_f1_score",
    "accuracy_at_best_f1",
]

rows = []
for case_name, layer_spec in cases:
    summ_path = os.path.join(results_dir, "layer_selection", case_name, "evaluation_summary.json")
    if not os.path.exists(summ_path):
        continue
    with open(summ_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    def pull(method_name: str):
        payload = summary.get(method_name, {})
        overall = payload.get("overall_performance", {}) if isinstance(payload, dict) else {}
        return {
            "roc_auc": _num(overall.get("roc_auc")),
            "tpr_at_fpr_5": _num(overall.get("tpr_at_fpr_5")),
            "best_f1_score": _num(overall.get("best_f1_score")),
            "accuracy_at_best_f1": _num(overall.get("accuracy_at_best_f1")),
        }

    base = {
        "alpha": 0.0,
        "layer_case": case_name,
        "rep_stiff_layers": layer_spec,
        "num_layers": len(layer_spec.split(",")),
        "blank_strategy": blank_strategy,
        "num_blanks": num_blanks,
    }
    for method_name in method_names:
        base.update({f"{method_name}_{kk}": vv for kk, vv in pull(method_name).items()})
    rows.append(base)

fieldnames = ["alpha", "layer_case", "rep_stiff_layers", "num_layers", "blank_strategy", "num_blanks"]
fieldnames.extend(f"{method_name}_{metric_name}" for method_name in method_names for metric_name in metric_names)

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(jsonl_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
PY
    
echo "======================================================"
echo "           Workflow Completed!"
echo "======================================================"