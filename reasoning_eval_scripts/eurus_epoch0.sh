set -euo pipefail

mkdir -p data

python src/eurus_evals.py \
  --model "talzoomanzoo/eurus-epoch0-step8" \
  --dataset_path "/scratch2/mjgwak/rl-data-contamination-mj/benchmarks/EURUS/eurus_member.parquet" \
  --gen_output "data/eurus_member__eurus-epoch0-step8__generations.json" \
  --eval_output "data/eurus_member__eurus-epoch0-step8__evaluated.json" \
  --use_vllm \
  --prompt_logprobs_k 20 \
  --batch_size 60 \
  --gpu_memory_utilization 0.8 \
  --max_num_seqs 16 \
  --num_samples 10 \
  --do_sample \
  --max_new_tokens 4096