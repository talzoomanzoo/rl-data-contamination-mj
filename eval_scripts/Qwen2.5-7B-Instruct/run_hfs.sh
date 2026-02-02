set -euo pipefail

cleanup_gpu() {
  pkill -f "vllm" || true
  pkill -f "VLLM" || true
  pkill -f "ray::" || true
  pkill -f "raylet" || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    # Wait until at least 2000 MB is free.
    for _ in $(seq 1 60); do
      free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1{print $1}')"
      if [ -n "$free_mb" ] && [ "$free_mb" -ge 2000 ]; then
        break
      fi
      sleep 2
    done
  fi
}

bash run_full_workflow_qwen_instruct_sat_hf.sh
cleanup_gpu
bash run_full_workflow_qwen_instruct_kk_hf.sh
cleanup_gpu
bash run_full_workflow_qwen-instruct_aime_hf.sh
cleanup_gpu
bash run_full_workflow_qwen-instruct_aime25_hf.sh