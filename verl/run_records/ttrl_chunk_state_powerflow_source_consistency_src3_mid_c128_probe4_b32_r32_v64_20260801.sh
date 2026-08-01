#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_answer_value_margin_top2_mid_c128_probe4_b32_r32_v64_3step_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_source_consistency_src3_mid_c128_probe4_b32_r32_v64_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-source-answer-consistency-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/csrc3}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TEST_FREQ="${TEST_FREQ:-2000000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_score_mode=answer_source_consistency \
  ttrl.chunk_state_source_mode=majority_consistent \
  ttrl.chunk_state_source_chunk_enable=True \
  ttrl.chunk_state_teacher_anchor_enable=False \
  ttrl.chunk_state_target_guard_enable=True \
  ttrl.chunk_state_target_guard_min_answer_mass=0.03125 \
  ttrl.chunk_state_target_guard_use_distribution_score=False \
  ttrl.chunk_state_target_guard_use_mass_gain=False \
  ttrl.chunk_state_target_guard_candidate_enable=True \
  ttrl.chunk_state_target_guard_candidate_max_boxed_count=1 \
  ttrl.chunk_state_target_guard_candidate_assistant_marker=True \
  "$@"
