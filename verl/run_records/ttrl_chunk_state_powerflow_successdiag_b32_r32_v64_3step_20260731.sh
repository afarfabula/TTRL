#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_diag_b32_r32_v64_3step_20260731.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_successdiag_b32_r32_v64_3step_20260731}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0731-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-success-chunk-diag}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cspfsuccessdiagb32r32v64s3}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"

exec bash "$BASE" \
  ttrl.chunk_state_source_mode=success \
  ttrl.chunk_state_skip_all_negative=True \
  "$@"
