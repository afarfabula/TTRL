#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_weighted_successdiag_b32_r32_v64_3step_20260731.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_anchor_nonzero_b32_r32_v64_3step_20260731}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0731-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-anchor-nonzero-chunk}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cpa3}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"

exec bash "$BASE" \
  ttrl.chunk_state_boundaries='[256,512,768]' \
  ttrl.chunk_state_teacher_anchor_enable=True \
  ttrl.chunk_state_teacher_anchor_score=1.0 \
  ttrl.chunk_state_teacher_anchor_candidate_index=0 \
  "$@"
