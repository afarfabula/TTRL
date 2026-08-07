#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_future_support_gain_src3_mid_c128_probe4_b32_r32_v64_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_future_support_gain_strict_src3_mid_c128_probe4_b32_r32_v64_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-future-support-gain-strict-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cfsg3s}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TEST_FREQ="${TEST_FREQ:-2000000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_future_support_gain_slack=0.05 \
  ttrl.chunk_state_future_support_source_prior_weight=1.25 \
  ttrl.chunk_state_min_answer_coverage=0.50 \
  ttrl.chunk_state_min_informative_gap=0.02 \
  ttrl.chunk_state_skip_uniform=True \
  "$@"
