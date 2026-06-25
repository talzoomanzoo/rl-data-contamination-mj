set -euo pipefail

mkdir -p data

python src/eurus_evals.py \
  --model "RUC-AIBOX/STILL-3-1.5B-preview" \
  --dataset_path "/scratch2/mjgwak/rl-data-contamination-mj/benchmarks/STILL/still_member.parquet" \
  --gen_output "data/still_member__still-3-1.5b-preview__generations.json" \
  --eval_output "data/still_member__still-3-1.5b-preview__evaluated.json" \
  --use_vllm \
  --batch_size 60 \
  --gpu_memory_utilization 0.8 \
  --max_num_seqs 16 \
  --num_samples 5 \
  --do_sample \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_new_tokens 32768

python src/eurus_evals.py \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  --dataset_path "/scratch2/mjgwak/rl-data-contamination-mj/benchmarks/STILL/still_member.parquet" \
  --gen_output "data/still_member__deepseek-r1-distill-qwen-1.5b__generations.json" \
  --eval_output "data/still_member__deepseek-r1-distill-qwen-1.5b__evaluated.json" \
  --use_vllm \
  --batch_size 60 \
  --gpu_memory_utilization 0.8 \
  --max_num_seqs 16 \
  --num_samples 5 \
  --do_sample \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_new_tokens 32768