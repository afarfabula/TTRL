#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v23_fullguard_balanced_sourceweight_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_supportflow_posteriorgain_nosplit_v24_fullguard_balanced_sourceweight_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-supportflow-posteriorgain-nosplit-v24-fullguard-balanced-sourceweight-targetonly-smoke}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TEST_FREQ="${TEST_FREQ:--1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"

exec bash "$BASE" \
  ttrl.chunk_state_support_flow_score_type=posterior_gain \
  ttrl.chunk_state_support_flow_baseline_scale=0.5 \
  ttrl.chunk_state_support_flow_gain_slack=0.05 \
  ttrl.chunk_state_support_flow_min_positive_margin=0.0 \
  "$@"
