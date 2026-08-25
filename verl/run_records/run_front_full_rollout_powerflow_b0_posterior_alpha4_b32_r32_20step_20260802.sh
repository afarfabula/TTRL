#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
RUN_ID="ttrl_full_rollout_powerflow_b0_posterior_alpha4_b32_r32_20step_20260802"
LOG="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}.log"
VAL="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}_val_metrics.json"

mkdir -p "$(dirname "$LOG")"
rm -f "$LOG" "$VAL"

cd "$ROOT"
RUN_ID="$RUN_ID" \
EXPERIMENT="$RUN_ID" \
LOG_NAME="0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullrollout-b0-20step" \
TTRL_RUNTIME_DIR="/tmp/frb0p" \
TOTAL_TRAINING_STEPS=20 \
TOTAL_EPOCHS=2 \
TEST_FREQ=20 \
FINAL_VAL_ENABLE=True \
VAL_BEFORE_TRAIN=False \
SAVE_FREQ=-1 \
bash run_records/ttrl_full_rollout_powerflow_b0_posterior_alpha4_b32_r32_20step_20260802.sh 2>&1 | tee "$LOG"
