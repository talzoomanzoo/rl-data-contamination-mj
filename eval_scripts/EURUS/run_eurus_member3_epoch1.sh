#!/bin/bash
set -e 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Allow overriding ROOT from environment; otherwise infer repo root from script location.
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

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
HF_MODEL_ID="talzoomanzoo/eurus_member3_new_epoch1"
MODEL_PATH="$HF_MODEL_ID"
MODEL_NAME="eurus_member3_new_epoch1"
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
# --- WORKFLOW ---
echo "======================================================"
echo "    Starting Final Contamination Detection Workflow"
echo "    Config -> Subset: ${SUBSET_SOURCE:-all}, Samples: ${NUM_SAMPLES_PER_SOURCE}"
echo "======================================================"

echo "--> Resolving Hugging Face model snapshot..."
MODEL_PATH="$(python - <<'PY'
import json
import os
import shutil
import sys
import tempfile

from huggingface_hub import snapshot_download
from huggingface_hub.utils import logging as hf_logging
from safetensors import safe_open

model_id = "talzoomanzoo/eurus_member3_new_epoch1"
ref_model_id = "talzoomanzoo/eurus_member3_new_epoch0"
hf_logging.set_verbosity_error()

local_override = os.environ.get("EURUS_MEMBER3_EPOCH1_MODEL_PATH", "").strip()
if local_override:
    path = os.path.abspath(local_override)
    if not os.path.isdir(path):
        hint = ""
        if "path/to" in local_override:
            hint = (
                " (you passed the documentation placeholder; set this to a real "
                "checkpoint directory on disk)"
            )
        raise FileNotFoundError(
            f"EURUS_MEMBER3_EPOCH1_MODEL_PATH is not a directory: {path}{hint}"
        )
    print(f"[info] Using local model path override: {path}", flush=True)
else:
    path = snapshot_download(model_id)

ref_dir = snapshot_download(ref_model_id)

# HF repo epoch1 was uploaded without tokenizer/index metadata (same weights family as epoch0).
_TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "model.safetensors.index.json",
)
_LM_HEAD_SHARD = "model-00007-of-00007.safetensors"
_LM_HEAD_EXPECTED_BYTES = 2_179_989_632

def _missing_tokenizer_files(model_dir):
    return [f for f in _TOKENIZER_FILES if not os.path.isfile(os.path.join(model_dir, f))]

def _link_or_copy(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst, symlinks=True)
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst, follow_symlinks=True)

def _validate_shards(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            weight_map = json.load(f).get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = sorted(
            f
            for f in os.listdir(model_dir)
            if f.endswith(".safetensors") and f.startswith("model-")
        )

    bad = []
    for shard in shard_names:
        fp = os.path.join(model_dir, shard)
        size = os.path.getsize(fp) if os.path.exists(fp) else -1
        try:
            with safe_open(fp, framework="pt") as f:
                list(f.keys())
        except Exception as e:
            bad.append((shard, size, str(e)))
    return bad

def _prepare_model_dir(source_dir, reference_dir, *, patch_lm_head=False):
    """Build a writable model dir; copy tokenizer/index from epoch0 when missing."""
    missing_tok = _missing_tokenizer_files(source_dir)
    if not missing_tok and not patch_lm_head:
        return source_dir

    out = tempfile.mkdtemp(prefix="eurus_member3_epoch1_prepared_")
    skip = set(missing_tok)
    if patch_lm_head:
        skip.add(_LM_HEAD_SHARD)

    for name in os.listdir(source_dir):
        if name in skip:
            continue
        _link_or_copy(os.path.join(source_dir, name), os.path.join(out, name))

    for name in missing_tok:
        src = os.path.join(source_dir, name)
        if not os.path.isfile(src):
            src = os.path.join(reference_dir, name)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Missing tokenizer file {name} under {source_dir} and {reference_dir}")
        shutil.copy2(src, os.path.join(out, name), follow_symlinks=True)
    if missing_tok:
        print(
            f"[info] Materialized tokenizer/index files from {ref_model_id}: {missing_tok}",
            flush=True,
        )

    if patch_lm_head:
        src_shard = os.path.join(reference_dir, _LM_HEAD_SHARD)
        if not os.path.isfile(src_shard):
            raise FileNotFoundError(f"Reference model missing {_LM_HEAD_SHARD}: {src_shard}")
        shutil.copy2(src_shard, os.path.join(out, _LM_HEAD_SHARD), follow_symlinks=True)
        print(
            f"[warn] Patched {_LM_HEAD_SHARD} from {ref_model_id} into {out}. "
            "Generation/logits will NOT match true epoch1 until the HF upload is fixed.",
            flush=True,
        )

    still_bad = _validate_shards(out)
    if still_bad:
        raise RuntimeError(f"Prepared model still has corrupt shards: {still_bad}")
    return out

# Fail fast before vLLM spends ~1 min loading shards 1-6.
bad = _validate_shards(path)
patch_lm_head = False

if bad:
    patch_lm_head = (
        len(bad) == 1
        and bad[0][0] == _LM_HEAD_SHARD
        and os.environ.get("EURUS_MEMBER3_EPOCH1_PATCH_LM_HEAD_FROM_EPOCH0", "").strip() == "1"
    )
    level = "[warn]" if patch_lm_head else "[error]"
    print(f"{level} Corrupt safetensors shard(s) under {path}:", file=sys.stderr, flush=True)
    for shard, size, err in bad:
        print(f"  - {shard}: size={size:,} bytes; {err}", file=sys.stderr, flush=True)

    if not patch_lm_head:
        if any(shard == _LM_HEAD_SHARD for shard, _, _ in bad):
            print(
                f"[error] {_LM_HEAD_SHARD} should be ~{_LM_HEAD_EXPECTED_BYTES:,} bytes "
                f"(contains lm_head.weight). The Hugging Face upload for {model_id} is truncated.",
                file=sys.stderr,
                flush=True,
            )
            print(
                "[error] Fix options:\n"
                "  1) Re-upload the correct shard to Hugging Face, then clear cache:\n"
                "     huggingface-cli upload talzoomanzoo/eurus_member3_new_epoch1 \\\n"
                "       /actual/checkpoint/dir/model-00007-of-00007.safetensors \\\n"
                "       model-00007-of-00007.safetensors\n"
                "     rm -f \"$HF_HOME/hub/models--talzoomanzoo--eurus_member3_new_epoch1/"
                "snapshots/*/model-00007-of-00007.safetensors\"\n"
                "  2) Point to a local full checkpoint:\n"
                "     EURUS_MEMBER3_EPOCH1_MODEL_PATH=/actual/checkpoint/dir bash run_eurus_member3_epoch1.sh\n"
                "  3) Temporary workaround (wrong lm_head; not for final eval):\n"
                "     EURUS_MEMBER3_EPOCH1_PATCH_LM_HEAD_FROM_EPOCH0=1 bash run_eurus_member3_epoch1.sh",
                file=sys.stderr,
                flush=True,
            )
        raise SystemExit(1)

path = _prepare_model_dir(path, ref_dir, patch_lm_head=patch_lm_head)

print(f"[info] Using model path: {path}", file=sys.stderr, flush=True)
print(path)
PY
)"
MODEL_PATH="$(printf "%s\n" "$MODEL_PATH" | tail -n 1)"
echo "--> Model path: $MODEL_PATH"

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

python "${ROOT}/evaluate_all_methods.py" \
    --input_file "$GENERATED_DATA_FILE" \
    --output_summary_json "$EVAL_SUMMARY_JSON" \
    --output_plot "$PLOT_PNG" \
    --rep_stiff_model_name "$MODEL_PATH" \
    --rep_stiff_max_workers "$OPENROUTER_WORKERS" \
    --rep_stiff_layers "early,mid,late" \
    --rep_stiff_scores_json "$REP_STIFF_SCORES_JSON" \
    --rep_stiff_output_dir "${ROOT}/rep_stiff_outputs_eurus" \
    $(eval echo "$COMBINED_ARGS")
    
echo "======================================================"
echo "           Workflow Completed!"
echo "======================================================"