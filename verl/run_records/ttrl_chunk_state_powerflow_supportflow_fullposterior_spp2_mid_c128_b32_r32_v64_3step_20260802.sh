#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_futuregain_smoothed_transport_mid_c128_probe1024x4_b32_r32_v64_3step_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_supportflow_fullposterior_spp2_mid_c128_b32_r32_v64_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-supportflow-fullposterior-spp2-smoke}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_score_mode=support_flow \
  ttrl.chunk_state_source_mode=majority_consistent \
  ttrl.chunk_state_source_select_by_mass=True \
  ttrl.chunk_state_min_prompt_top_mass=0.45 \
  ttrl.chunk_state_min_prompt_valid_answer_coverage=0.65 \
  ttrl.chunk_state_min_prompt_top_margin=0.20 \
  ttrl.chunk_state_min_source_answer_mass=0.40 \
  ttrl.chunk_state_states_per_prompt=2 \
  ttrl.chunk_state_support_anchor_enable=True \
  ttrl.chunk_state_support_anchor_count=4 \
  ttrl.chunk_state_support_anchor_candidate_start=4 \
  ttrl.chunk_state_support_anchor_min_mass=0.03125 \
  ttrl.chunk_state_support_anchor_selection_mode=answer_stratified \
  ttrl.chunk_state_support_anchor_skip_source=True \
  ttrl.chunk_state_source_chunk_enable=True \
  ttrl.chunk_state_source_chunk_candidate_index=0 \
  ttrl.chunk_state_support_flow_score_type=soft_mass \
  ttrl.chunk_state_support_flow_source_prior_weight=1.0 \
  ttrl.chunk_state_support_flow_split_mass_by_answer=True \
  ttrl.chunk_state_support_flow_answer_split_power=1.0 \
  ttrl.chunk_state_future_support_keep_mode=soft \
  ttrl.chunk_state_future_support_soft_weight_floor=0.05 \
  ttrl.chunk_state_source_quality_weight_mode=group_quality \
  ttrl.chunk_state_source_quality_weight_floor=0.30 \
  ttrl.chunk_state_source_quality_weight_power=0.5 \
  ttrl.chunk_state_source_quality_max_entropy=1.8 \
  ttrl.chunk_state_prune_zero_weight_samples=True \
  "$@"
