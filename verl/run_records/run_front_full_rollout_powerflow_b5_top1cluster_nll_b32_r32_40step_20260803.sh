#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
RUN_ID="ttrl_full_rollout_powerflow_b5_top1cluster_nll_b32_r32_40step_20260803"
LOG="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}.log"

mkdir -p "$(dirname "$LOG")"
rm -f "$LOG"

cd "$ROOT"
RUN_ID="$RUN_ID" \
EXPERIMENT="$RUN_ID" \
LOG_NAME="0803-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullrollout-b5-top1cluster-nll-40step" \
OUTPUT_DIR="/tmp/ttrl_b200/checkpoints/${RUN_ID}" \
TTRL_RUNTIME_DIR="/tmp/frb5n40" \
TOTAL_TRAINING_STEPS=40 \
TOTAL_EPOCHS=3 \
TEST_FREQ=40 \
FINAL_VAL_ENABLE=True \
VAL_BEFORE_TRAIN=False \
SAVE_FREQ=-1 \
bash run_records/ttrl_full_rollout_powerflow_b5_top1cluster_nll_b32_r32_20step_20260802.sh 2>&1 | tee "$LOG"
