ROOT="/scratch2/mjgwak/rl-data-contamination-mj"
export PYTHONPATH="${ROOT}/reasoning_eval_scripts/src:${PYTHONPATH:-}"

bash run_eurus_member2_epoch0.sh && \
bash run_eurus_member2_epoch1.sh && \
bash run_eurus_member3_epoch0.sh && \
bash run_eurus_member3_epoch1.sh