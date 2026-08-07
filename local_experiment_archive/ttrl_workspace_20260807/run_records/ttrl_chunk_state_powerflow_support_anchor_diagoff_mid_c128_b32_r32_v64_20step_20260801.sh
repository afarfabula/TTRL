#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_support_anchor_diagoff_mid_c128_b32_r32_v64_3step_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_support_anchor_diagoff_mid_c128_b32_r32_v64_20step_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-support-anchor-diagoff-20step}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cssa12820diagoff}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
export TEST_FREQ="${TEST_FREQ:-20}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-True}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export DIAG_JSONL="${DIAG_JSONL:-}"

exec bash "$BASE" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.final_val_enable="$FINAL_VAL_ENABLE" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.log_val_generations=0 \
  "$@"
