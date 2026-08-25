#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
RUN_ID="ttrl_sharpened_grpo_b32_r32_v64_20step_h100_restore_mvanchor_soft002_20260803"
LOG="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}.log"

mkdir -p "$(dirname "$LOG")"
rm -f "$LOG"

cd "$ROOT"
RUN_ID="$RUN_ID" \
EXPERIMENT="$RUN_ID" \
LOG_NAME="0803-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-grpo-h100-restore" \
TTRL_RUNTIME_DIR="/tmp/smva02h10020" \
OUTPUT_DIR="/tmp/ttrl_h100/checkpoints/${RUN_ID}" \
bash run_records/ttrl_sharpened_grpo_b32_r32_v64_20step_h100_restore_mvanchor_soft002_20260803.sh 2>&1 | tee "$LOG"
