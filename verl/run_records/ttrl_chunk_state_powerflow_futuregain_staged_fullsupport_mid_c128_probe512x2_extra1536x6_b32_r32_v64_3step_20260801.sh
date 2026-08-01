#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_futuregain_smoothed_transport_mid_c128_probe1024x4_b32_r32_v64_3step_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_futuregain_staged_fullsupport_mid_c128_probe512x2_extra1536x6_b32_r32_v64_3step_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-staged-fullsupport-3step-smoke}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_support_anchor_count=3 \
  ttrl.chunk_state_support_anchor_candidate_start=5 \
  ttrl.chunk_state_probe_samples=2 \
  ttrl.chunk_state_probe_max_tokens=512 \
  ttrl.chunk_state_staged_probe_enable=True \
  ttrl.chunk_state_staged_probe_topk=2 \
  ttrl.chunk_state_staged_probe_extra_samples=6 \
  ttrl.chunk_state_staged_probe_extra_max_tokens=1536 \
  ttrl.chunk_state_staged_probe_merge_mode=repeat_base \
  ttrl.chunk_state_future_support_prior_smoothing=6.0 \
  ttrl.chunk_state_future_support_source_prior_weight=0.5 \
  ttrl.chunk_state_future_support_min_state_coverage=0.125 \
  ttrl.chunk_state_future_support_max_state_oov=0.875 \
  ttrl.chunk_state_future_support_min_state_mean_mass=0.010 \
  ttrl.chunk_state_future_support_min_state_max_mass=0.03125 \
  ttrl.chunk_state_future_support_min_state_top_margin=0.001 \
  "$@"
