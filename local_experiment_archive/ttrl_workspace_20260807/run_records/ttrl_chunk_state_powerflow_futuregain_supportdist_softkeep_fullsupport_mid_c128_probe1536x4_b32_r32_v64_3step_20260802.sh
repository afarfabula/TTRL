#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_3step_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_futuregain_supportdist_softkeep_fullsupport_mid_c128_probe1536x4_b32_r32_v64_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-full-support-distribution-softkeep-smoke}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_future_support_score_type=support_distribution_match \
  ttrl.chunk_state_future_support_keep_mode=soft \
  ttrl.chunk_state_future_support_soft_weight_floor=0.05 \
  ttrl.chunk_state_future_support_min_positive_margin=0.0 \
  ttrl.chunk_state_future_support_min_state_coverage=0.0 \
  ttrl.chunk_state_future_support_max_state_oov=1.0 \
  ttrl.chunk_state_future_support_min_state_mean_mass=0.0 \
  ttrl.chunk_state_future_support_min_state_max_mass=0.0 \
  ttrl.chunk_state_future_support_min_state_top_margin=0.0 \
  ttrl.chunk_state_future_support_min_candidate_coverage=0.0 \
  ttrl.chunk_state_future_support_min_candidate_mean_mass=0.0 \
  ttrl.chunk_state_min_answer_coverage=0.0 \
  ttrl.chunk_state_min_informative_gap=0.0 \
  ttrl.chunk_state_prune_zero_weight_samples=False \
  ttrl.chunk_state_source_quality_weight_mode=group_quality \
  ttrl.chunk_state_source_quality_weight_floor=0.20 \
  ttrl.chunk_state_source_quality_weight_power=0.5 \
  ttrl.chunk_state_source_quality_max_entropy=2.0 \
  "$@"
