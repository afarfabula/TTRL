#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_weighted_successdiag_b32_r32_v64_3step_20260731.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_midstate_b32_r32_v64_3step_20260731}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0731-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-midstate-powerflow}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cpms3}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

mkdir -p "$(dirname "$DIAG_JSONL")"
rm -f "$DIAG_JSONL"

exec bash "$BASE" \
  ttrl.chunk_state_boundaries='[256,512,768]' \
  ttrl.chunk_state_min_boundary=256 \
  ttrl.chunk_state_teacher_anchor_enable=True \
  ttrl.chunk_state_teacher_anchor_score=1.0 \
  ttrl.chunk_state_teacher_anchor_candidate_index=0 \
  ttrl.chunk_state_diag_jsonl="$DIAG_JSONL" \
  "$@"
