#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_successdiag_b32_r32_v64_3step_20260731.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_weighted_successdiag_b32_r32_v64_3step_20260731}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0731-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-weighted-success-chunk-diag}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cpw3}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"

exec bash "$BASE" \
  actor_rollout_ref.actor.powerflow_use_chunk_weights=True \
  "$@"
