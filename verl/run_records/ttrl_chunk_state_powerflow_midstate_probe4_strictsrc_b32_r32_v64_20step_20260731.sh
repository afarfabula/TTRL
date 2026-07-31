#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_midstate_b32_r32_v64_3step_20260731.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_midstate_probe4_strictsrc_b32_r32_v64_20step_20260731}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0731-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-midstate-probe4-strictsrc-powerflow}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cpmp420}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
export TEST_FREQ="${TEST_FREQ:-20}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

mkdir -p "$(dirname "$DIAG_JSONL")"
rm -f "$DIAG_JSONL"

exec bash "$BASE" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.final_val_enable=True \
  trainer.save_freq=2000000 \
  ttrl.chunk_state_probe_samples=4 \
  ttrl.chunk_state_probe_max_tokens=1024 \
  ttrl.chunk_state_diag_jsonl="$DIAG_JSONL" \
  "$@"
