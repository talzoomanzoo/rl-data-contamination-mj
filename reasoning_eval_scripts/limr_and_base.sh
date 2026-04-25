set -euo pipefail

mkdir -p data

python src/eurus_evals.py \
  --model "GAIR/LIMR" \
  --dataset_path "/scratch2/mjgwak/rl-data-contamination-mj/benchmarks/LIMR/limr_member.parquet" \
  --gen_output "data/limr_member__limr__generations.json" \
  --eval_output "data/limr_member__limr__evaluated.json" \
  --use_vllm \
  --async_vllm \
  --batch_size 60 \
  --max_in_flight 8 \
  --score_max_in_flight 1 \
  --gpu_memory_utilization 0.8 \
  --max_num_seqs 16 \
  --num_samples 5 \
  --do_sample \
  --max_new_tokens 3072

python src/eurus_evals.py \
  --model "Qwen/Qwen2.5-Math-7B" \
  --dataset_path "/scratch2/mjgwak/rl-data-contamination-mj/benchmarks/LIMR/limr_member.parquet" \
  --gen_output "data/limr_member__qwen2.5-math-7b__generations.json" \
  --eval_output "data/limr_member__qwen2.5-math-7b__evaluated.json" \
  --use_vllm \
  --async_vllm \
  --batch_size 60 \
  --max_in_flight 8 \
  --score_max_in_flight 1 \
  --gpu_memory_utilization 0.8 \
  --max_num_seqs 16 \
  --num_samples 5 \
  --do_sample \
  --max_new_tokens 3072