#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softprefix_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_posterior_support_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-posterior-support-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cfsp82psm}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_future_support_score_type=posterior_support_match \
  ttrl.chunk_state_future_support_prior_smoothing=16.0 \
  ttrl.chunk_state_future_support_anchor_prior_weight=2.0 \
  ttrl.chunk_state_future_support_source_prior_weight=1.0 \
  ttrl.chunk_state_future_support_keep_mode=soft \
  ttrl.chunk_state_future_support_soft_weight_floor=0.05 \
  ttrl.chunk_state_powerflow_weight_clip=4.0 \
  ttrl.chunk_state_powerflow_weight_clip_renorm=True \
  "$@"
