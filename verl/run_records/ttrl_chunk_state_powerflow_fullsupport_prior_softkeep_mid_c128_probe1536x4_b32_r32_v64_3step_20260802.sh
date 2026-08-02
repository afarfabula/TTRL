#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_3step_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullsupport-prior-softkeep-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cfsp82}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_future_support_score_type=support_distribution_match \
  ttrl.chunk_state_future_support_anchor_prior_weight=4.0 \
  ttrl.chunk_state_future_support_anchor_prior_power=1.0 \
  ttrl.chunk_state_future_support_source_prior_weight=1.0 \
  ttrl.chunk_state_future_support_prior_smoothing=8.0 \
  ttrl.chunk_state_future_support_keep_mode=soft \
  ttrl.chunk_state_future_support_soft_weight_floor=0.05 \
  ttrl.chunk_state_future_support_min_positive_margin=0.0 \
  ttrl.chunk_state_future_support_min_state_coverage=0.0 \
  ttrl.chunk_state_future_support_max_state_oov=1.0 \
  ttrl.chunk_state_future_support_min_state_mean_mass=0.0 \
  ttrl.chunk_state_future_support_min_state_max_mass=0.0 \
  ttrl.chunk_state_future_support_min_state_top_margin=0.0 \
  ttrl.chunk_state_min_answer_coverage=0.0 \
  ttrl.chunk_state_min_informative_gap=0.0 \
  ttrl.chunk_state_support_anchor_enable=True \
  ttrl.chunk_state_support_anchor_count=4 \
  ttrl.chunk_state_support_anchor_candidate_start=4 \
  ttrl.chunk_state_support_anchor_min_mass=0.03125 \
  ttrl.chunk_state_support_anchor_skip_source=True \
  ttrl.chunk_state_probe_samples=4 \
  ttrl.chunk_state_probe_max_tokens=1536 \
  ttrl.chunk_state_candidates=8 \
  "$@"
