#!/bin/bash
set -e 

# Resolve repo root relative to this script so it works from any cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# --- Hugging Face download acceleration ---
# NOTE: set to 0 unless you have `hf_transfer` installed.
export HF_HUB_ENABLE_HF_TRANSFER=1

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
HF_MODEL_ID="PRIME-RL/Eurus-2-7B-PRIME"
MODEL_PATH="$HF_MODEL_ID"
MODEL_NAME="Eurus-2-7B-PRIME_eurus"
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
# 'important' blanks the most important word(s)/phrase(s); 'guided' blanks an identified info-type.
REP_STIFF_INCOMPLETE_BLANK_STRATEGY="${REP_STIFF_INCOMPLETE_BLANK_STRATEGY:-important}"
# In 'important' mode, enforce exactly this many [BLANK] tokens in each incomplete question.
REP_STIFF_INCOMPLETE_NUM_BLANKS="${REP_STIFF_INCOMPLETE_NUM_BLANKS:-1}"

# --- RepStiff layer probing ---
# Number of transformer layers in the base model (Qwen2.5-Math-7B = 28).
# Used to build "L0,L1,...,L{N-1}" so RepStiff probes every layer individually.
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-28}"
REP_STIFF_LAYERS="$(python -c "import sys; print(','.join(f'L{i}' for i in range(int(sys.argv[1]))))" "$NUM_HIDDEN_LAYERS")"

# --- LaRA (clean-reference standardized geometric anomaly) options ---
# - REP_STIFF_LARA_EPS: numerical epsilon for z = (m - mu)/(sigma + eps).
# - REP_STIFF_LARA_CLEAN_REF: optional path to a JSON of clean reference stats
#   {layer: {metric: {mean, std}}} from a held-out clean validation set.
#   When unset, mu/sigma are estimated from rows with ground_truth_label == 0
#   in the current eval set.
# - REP_STIFF_LARA_MIX_BETA: rank-mix weight on self_critique for
#   self_critique_rep_stiff_lara_mix.
# - REP_STIFF_LARA_ROBUST_LAYER_WINDOW: which layer-position subset the robust
#   variant aggregates over. One of {all, early_mid, early, mid, late}; defaults
#   to 'all' (paper-spec aggregation). Tighter windows are explored as variants
#   in the auxiliary lara_robust_variants.json / lara_self_critique_beta_sweep.json
#   files written next to evaluation_summary.json.
# - REP_STIFF_LARA_ROBUST_DC_WEIGHT: multiplicative weight on the DC metric in
#   the robust aggregation (1.0 = uniform, 2.0 = double DC influence). Default
#   1.0 (uniform). Variant files explore higher weights.
REP_STIFF_LARA_EPS="${REP_STIFF_LARA_EPS:-1e-8}"
REP_STIFF_LARA_CLEAN_REF="${REP_STIFF_LARA_CLEAN_REF:-}"
REP_STIFF_LARA_MIX_BETA="${REP_STIFF_LARA_MIX_BETA:-0.65}"
REP_STIFF_LARA_ROBUST_LAYER_WINDOW="${REP_STIFF_LARA_ROBUST_LAYER_WINDOW:-all}"
REP_STIFF_LARA_ROBUST_DC_WEIGHT="${REP_STIFF_LARA_ROBUST_DC_WEIGHT:-1.0}"
# --- WORKFLOW ---
echo "======================================================"
echo "    Starting Final Contamination Detection Workflow"
echo "    Config -> Subset: ${SUBSET_SOURCE:-all}, Samples: ${NUM_SAMPLES_PER_SOURCE}"
echo "======================================================"

echo "--> Resolving Hugging Face model snapshot..."
MODEL_PATH="$(python - <<'PY'
from huggingface_hub import snapshot_download

model_id = "PRIME-RL/Eurus-2-7B-PRIME"
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

# --- Single RepStiff run with k=1 incomplete-question blanks ---
# The k=1 output dir is preserved (final_results/.../blank_k_sweep/k1/) because
# downstream tooling (e.g. final_results_main_analyses/visualize.py) expects
# rep_stiff_scores.json to live at that path.
V4_ALPHA_DEFAULT="0.0"
K_BLANKS=1
RUN_DIR="${RESULTS_DIR}/blank_k_sweep/k${K_BLANKS}"
mkdir -p "$RUN_DIR"

echo "--> RepStiff: k_blanks=${K_BLANKS}, alpha=${V4_ALPHA_DEFAULT}"

LARA_ARGS=""
if [ -n "$REP_STIFF_LARA_CLEAN_REF" ]; then
    LARA_ARGS="$LARA_ARGS --rep_stiff_lara_clean_ref \"$REP_STIFF_LARA_CLEAN_REF\""
fi

python "${ROOT}/evaluate_all_methods.py" \
    --input_file "$GENERATED_DATA_FILE" \
    --output_summary_json "${RUN_DIR}/evaluation_summary.json" \
    --output_plot "${RUN_DIR}/performance_plot.png" \
    --rep_stiff_model_name "$HF_MODEL_ID" \
    --rep_stiff_max_workers "$OPENROUTER_WORKERS" \
    --rep_stiff_layers "$REP_STIFF_LAYERS" \
    --rep_stiff_scores_json "${RUN_DIR}/rep_stiff_scores.json" \
    --rep_stiff_output_dir "${ROOT}/rep_stiff_outputs_eurus_blank_k${K_BLANKS}" \
    --rep_stiff_incomplete_blank_strategy "$REP_STIFF_INCOMPLETE_BLANK_STRATEGY" \
    --rep_stiff_incomplete_num_blanks "$K_BLANKS" \
    --rep_stiff_combined_v4_alpha "$V4_ALPHA_DEFAULT" \
    --rep_stiff_lara_eps "$REP_STIFF_LARA_EPS" \
    --rep_stiff_lara_mix_beta "$REP_STIFF_LARA_MIX_BETA" \
    --rep_stiff_lara_robust_layer_window "$REP_STIFF_LARA_ROBUST_LAYER_WINDOW" \
    --rep_stiff_lara_robust_dc_weight "$REP_STIFF_LARA_ROBUST_DC_WEIGHT" \
    $(eval echo "$COMBINED_ARGS") \
    $(eval echo "$LARA_ARGS")

echo "======================================================"
echo "           Workflow Completed!"
echo "======================================================"