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
# Hugging Face id for vLLM (Step 1) and RepStiff / OpenRouter (Step 2).
HF_MODEL_ID="talzoomanzoo/eurus_grpo_rlmia_epoch_1"
MODEL_PATH="$HF_MODEL_ID"
# Slug for local paths (HF id contains '/').
MODEL_NAME="${HF_MODEL_ID//\//__}"
# RepStiff disk cache root pattern (keep distinct from EURUS `rep_stiff_outputs_eurus_blank_k*`).
REP_STIFF_OUTPUT_DIR_BASENAME="rep_stiff_outputs_talzoomanzoo_limr_grpo_rlmia_epoch_1"
# LIMR benchmark dataset root.
# `generate_full_data.py` scans recursively for *.jsonl/*.parquet under this directory.
DATA_ROOT_DIR="${ROOT}/benchmarks/LIMR"
# Used only to namespace outputs (prevents reusing old generated_data.jsonl from other benchmarks).
DATASET_TAG="limr_grpo_rlmia_epoch_1"

METHODS_TO_RUN=("self_critique" "dime" "consistency")

# --- Sampling & vLLM configuration ---
# Smaller batches + more CPU swap reduce KV preemption / "lack of CPU swap space" under n>1 sampling.
TEMPERATURE_RANDOM=0.8
NUM_RANDOM_SAMPLES=10
TENSOR_PARALLEL_SIZE=1
MAX_NEW_TOKENS=4096
BATCH_SIZE="${BATCH_SIZE:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
# CPU KV swap per GPU (GiB); raise if vLLM aborts on swap exhaustion.
VLLM_SWAP_SPACE="${VLLM_SWAP_SPACE:-32}"
# Cap concurrent sequences (lower = less KV pressure). Export VLLM_MAX_NUM_SEQS="" to omit flag.
if [[ ! -v VLLM_MAX_NUM_SEQS ]]; then
    VLLM_MAX_NUM_SEQS=64
fi
VLLM_RANDOM_MICROBATCH="${VLLM_RANDOM_MICROBATCH:-4}"
VLLM_CRITIQUE_MICROBATCH="${VLLM_CRITIQUE_MICROBATCH:-4}"

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

# --- RepStiff: full transformer layer sweep (L0..L{N-1}) ---
# Default N=28 matches typical LIMR / Qwen-class `num_hidden_layers`. Set NUM_REP_STIFF_LAYERS if your model differs.
# Much slower than `early,mid,late` alone (one RepStiff scoring path per layer × metric).
NUM_REP_STIFF_LAYERS="${NUM_REP_STIFF_LAYERS:-28}"
REP_STIFF_LAYERS="$(python3 -c "import sys; n=int(sys.argv[1]); print(','.join(f'L{i}' for i in range(n)))" "${NUM_REP_STIFF_LAYERS}")"

# --- LaRA (clean-reference standardized geometric anomaly) options ---
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

echo "--> Resolving Hugging Face model snapshot (${HF_MODEL_ID})..."
MODEL_PATH="$(HF_MODEL_ID="$HF_MODEL_ID" python - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(os.environ["HF_MODEL_ID"])
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
if [ -n "$VLLM_MAX_NUM_SEQS" ]; then
    CMD_ARGS="$CMD_ARGS --max_num_seqs $VLLM_MAX_NUM_SEQS"
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
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --swap_space "$VLLM_SWAP_SPACE" \
    --vllm_random_microbatch "$VLLM_RANDOM_MICROBATCH" \
    --vllm_critique_microbatch "$VLLM_CRITIQUE_MICROBATCH" \
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
echo "--> RepStiff layers: full sweep ${NUM_REP_STIFF_LAYERS} layers (L0..L$((NUM_REP_STIFF_LAYERS - 1)))"

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
    --rep_stiff_output_dir "${ROOT}/${REP_STIFF_OUTPUT_DIR_BASENAME}_blank_k${K_BLANKS}" \
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
