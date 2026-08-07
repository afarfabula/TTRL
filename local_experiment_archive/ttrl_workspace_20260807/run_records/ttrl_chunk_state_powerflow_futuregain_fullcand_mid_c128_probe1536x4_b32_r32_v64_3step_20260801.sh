#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_futuregain_smoothed_transport_mid_c128_probe1024x4_b32_r32_v64_3step_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_3step_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullcand-longprobe-3step-smoke}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_probe_samples=4 \
  ttrl.chunk_state_probe_max_tokens=1536 \
  ttrl.chunk_state_candidates=8 \
  ttrl.chunk_state_staged_probe_enable=False \
  ttrl.chunk_state_support_anchor_count=3 \
  ttrl.chunk_state_support_anchor_candidate_start=5 \
  ttrl.chunk_state_future_support_prior_smoothing=4.0 \
  ttrl.chunk_state_future_support_source_prior_weight=1.0 \
  "$@"
