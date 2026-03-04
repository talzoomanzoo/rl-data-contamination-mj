#!/bin/bash
set -e 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Allow overriding ROOT from environment; otherwise infer repo root from script location.
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# --- Hugging Face download acceleration ---
export HF_HUB_ENABLE_HF_TRANSFER=1
# --- vLLM multiprocessing mode for CUDA ---
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- CONFIGURATION ---
HF_MODEL_ID="talzoomanzoo/qwen3-4b-instruct-2507-kk_mia_test-actor"
# If set to an existing directory, vLLM will load from here instead of HF_MODEL_ID.
# Use this after converting/merging DTensor/FSDP shards into a standard HF folder.
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-}"
if [ -z "$MERGED_MODEL_DIR" ]; then
    AUTO_MERGED_MODEL_DIR="/scratch2/mjgwak/rl-data-contamination-mj/models/${HF_MODEL_ID##*/}-merged"
    if [ -d "$AUTO_MERGED_MODEL_DIR" ]; then
        MERGED_MODEL_DIR="$AUTO_MERGED_MODEL_DIR"
    fi
fi
MODEL_PATH="$HF_MODEL_ID"
MODEL_NAME="Qwen3-4B-Instruct_kk_mia_test_hf"
DATA_ROOT_DIR="${ROOT}/benchmarks/KK" 

METHODS_TO_RUN=("self_critique" "dime" "consistency")

# --- Sampling & VLLM Configuration ---
TEMPERATURE_RANDOM=0.8
NUM_RANDOM_SAMPLES=10
TENSOR_PARALLEL_SIZE=4
MAX_NEW_TOKENS=4096
BATCH_SIZE=100

SUBSET_SOURCE="kk_logic"
NUM_SAMPLES_PER_SOURCE=-1

SUBSET_TAG="_${SUBSET_SOURCE:-all}"
SAMPLE_TAG="_n${NUM_SAMPLES_PER_SOURCE}"
if [ "$NUM_SAMPLES_PER_SOURCE" -lt 0 ]; then
    SAMPLE_TAG="_all_samples"
fi
FILENAME_TAG="${SUBSET_TAG}${SAMPLE_TAG}"

# --- Output Configuration ---
RESULTS_DIR="${ROOT}/final_results/${MODEL_NAME}/${SUBSET_TAG}_${SAMPLE_TAG}"
mkdir -p "$RESULTS_DIR"

GENERATED_DATA_FILE="${RESULTS_DIR}/generated_data.jsonl"
EVAL_SUMMARY_JSON="${RESULTS_DIR}/evaluation_summary.json"
PLOT_PNG="${RESULTS_DIR}/performance_plot.png"
DIME_DETAIL_JSONL="${RESULTS_DIR}/dime_detail_report.jsonl"
# --- WORKFLOW ---
echo "======================================================"
echo "    Starting Final Contamination Detection Workflow"
echo "    Config -> Subset: ${SUBSET_SOURCE:-all}, Samples: ${NUM_SAMPLES_PER_SOURCE}"
echo "======================================================"

if [ -n "$MERGED_MODEL_DIR" ] && [ -d "$MERGED_MODEL_DIR" ]; then
    echo "--> Using merged local model path: $MERGED_MODEL_DIR"
    MODEL_PATH="$MERGED_MODEL_DIR"
    export MODEL_PATH
    # Patch tokenizer_config.json for older transformers (expects dict, not list).
    python - <<'PY'
import json, os
mp = os.environ.get("MODEL_PATH") or ""
tc = os.path.join(mp, "tokenizer_config.json")
if os.path.exists(tc):
    with open(tc, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    x = cfg.get("extra_special_tokens")
    if isinstance(x, list):
        cfg["extra_special_tokens"] = {}
        with open(tc, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"--> Patched tokenizer_config.json extra_special_tokens to dict: {tc}")
PY
elif [ -d "$HF_MODEL_ID" ]; then
    echo "--> Using local model path: $HF_MODEL_ID"
    MODEL_PATH="$HF_MODEL_ID"
else
    echo "--> Resolving Hugging Face model snapshot (for cache patch only)..."
    export HF_MODEL_ID
    python - <<'PY'
from huggingface_hub import snapshot_download
import json
import os

model_id = os.environ["HF_MODEL_ID"]
path = snapshot_download(model_id)
config_path = os.path.join(path, "config.json")
if not os.path.exists(config_path):
    config_path = os.path.join(path, "huggingface", "config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    rope = cfg.get("rope_scaling")
    if rope is None:
        cfg["rope_scaling"] = {"type": "dynamic", "factor": 1.0}
    elif isinstance(rope, dict):
        rope["type"] = "dynamic"
        rope.setdefault("factor", 1.0)
        cfg["rope_scaling"] = rope
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
print(f"--> Snapshot cached at: {path}")
PY

    # Keep MODEL_PATH as a Hugging Face repo id when not using local.
    MODEL_PATH="$HF_MODEL_ID"
fi

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
    echo "--> Step 1 finished, but no data file was created (or it is empty): '$GENERATED_DATA_FILE'."
    echo "--> Skipping Step 2."
    exit 0
fi

echo "--> Step 1 finished. Data saved to '$GENERATED_DATA_FILE'."
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
python "${ROOT}/evaluate_all_methods.py" \
    --input_file "$GENERATED_DATA_FILE" \
    --output_summary_json "$EVAL_SUMMARY_JSON" \
    --output_plot "$PLOT_PNG" \
    --rep_stiff_model_name "$MODEL_PATH" \
    --rep_stiff_max_workers 4 \
    --rep_stiff_layers "early,mid,late" \
    --rep_stiff_output_dir "${ROOT}/rep_stiff_outputs_kk" \
    --rep_stiff_combined_fixed \
    --rep_stiff_combined_rule "trend_v2" \
    --rep_stiff_combined_weights "${ROOT}/final_results/Qwen3-4B-Instruct_kk_hf/_kk_logic__all_samples/rep_stiff_combined_metrics.json" \
    --rep_stiff_combined_rule "trend_v3" \
    --rep_stiff_combined_weights "${ROOT}/final_results/Qwen3-4B-Instruct_kk_hf/_kk_logic__all_samples/rep_stiff_combined_metrics_v3.json" \
    --rep_stiff_combined_rule "trend_v4" \
    --rep_stiff_combined_weights "${ROOT}/final_results/Qwen3-4B-Instruct_kk_hf/_kk_logic__all_samples/rep_stiff_combined_metrics_v4.json" \
    
echo "======================================================"
echo "           Workflow Completed!"
echo "======================================================"