#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_b32_r32_v64_20step_20260731.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_diag_b32_r32_v64_3step_20260731}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0731-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-chunk-diag}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cspfdiagb32r32v64s3}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TEST_FREQ="${TEST_FREQ:-2000000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export LOGGER="${LOGGER:-console}"

exec bash "$BASE" \
  ttrl.chunk_state_diag_enable=True \
  trainer.test_freq="$TEST_FREQ" \
  "$@"
