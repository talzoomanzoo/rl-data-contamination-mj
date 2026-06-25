set -euo pipefail

ROOT="/scratch2/mjgwak/rl-data-contamination-mj"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${ROOT}/compat_site:${ROOT}/reasoning_eval_scripts/src:${PYTHONPATH:-}"

mkdir -p data

python src/eurus_evals.py \
  --model "talzoomanzoo/eurus_grpo_rlmia_epoch_1" \
  --dataset_path "/scratch2/mjgwak/rl-data-contamination-mj/benchmarks/LIMR/limr_member.parquet" \
  --gen_output "data/limr_member__limr_grpo_rlmia_epoch_1__generations.json" \
  --eval_output "data/limr_member__limr_grpo_rlmia_epoch_1__evaluated.json" \
  --use_vllm \
  --prompt_logprobs_k 20 \
  --batch_size 60 \
  --gpu_memory_utilization 0.8 \
  --max_num_seqs 16 \
  --num_samples 10 \
  --do_sample \
  --max_new_tokens 4096