#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v16_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v17_guardcand_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-supportflow-posteriormass-nosplit-v17-guardcand-targetonly-smoke}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TEST_FREQ="${TEST_FREQ:--1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"

exec bash "$BASE" \
  ttrl.chunk_state_target_guard_enable=True \
  ttrl.chunk_state_target_guard_candidate_enable=True \
  ttrl.chunk_state_target_guard_candidate_max_boxed_count=1 \
  ttrl.chunk_state_target_guard_candidate_assistant_marker=True \
  ttrl.chunk_state_zero_inconsistent_candidates=True \
  ttrl.chunk_state_prune_zero_weight_samples=True \
  "$@"
