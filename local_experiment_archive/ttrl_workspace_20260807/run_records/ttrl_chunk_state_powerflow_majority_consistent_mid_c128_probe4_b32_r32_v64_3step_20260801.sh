#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_label_consistent_c128_probe4_b32_r32_v64_3step_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_majority_consistent_mid_c128_probe4_b32_r32_v64_3step_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-majority-consistent-mid-c128-powerflow}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/csmc1283}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TEST_FREQ="${TEST_FREQ:-2000000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_source_mode=majority_consistent \
  ttrl.chunk_state_boundary_mode=mid \
  ttrl.chunk_state_mid_boundary_min_ratio=0.25 \
  ttrl.chunk_state_mid_boundary_max_ratio=0.80 \
  ttrl.chunk_state_source_chunk_enable=True \
  ttrl.chunk_state_source_chunk_candidate_index=0 \
  "$@"
