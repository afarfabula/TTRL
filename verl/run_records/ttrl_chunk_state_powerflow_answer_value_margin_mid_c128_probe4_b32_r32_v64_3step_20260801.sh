#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_majority_consistent_mid_c128_probe4_b32_r32_v64_3step_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_answer_value_margin_mid_c128_probe4_b32_r32_v64_3step_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-answer-value-margin-mid-c128-powerflow}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/csavm1283}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TEST_FREQ="${TEST_FREQ:-2000000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_score_mode=answer_value_margin \
  ttrl.chunk_state_source_chunk_enable=False \
  ttrl.chunk_state_teacher_anchor_enable=False \
  ttrl.chunk_state_probe_max_tokens=1024 \
  +ttrl.chunk_state_value_margin=0.25 \
  +ttrl.chunk_state_format_guard=True \
  +ttrl.chunk_state_max_boxed_count=8 \
  +ttrl.chunk_state_max_answer_chars=128 \
  ttrl.chunk_state_min_majority_ratio=0.20 \
  ttrl.chunk_state_min_answer_coverage=0.60 \
  ttrl.chunk_state_min_informative_gap=0.0 \
  ttrl.chunk_state_label_consistent_only=True \
  "$@"
