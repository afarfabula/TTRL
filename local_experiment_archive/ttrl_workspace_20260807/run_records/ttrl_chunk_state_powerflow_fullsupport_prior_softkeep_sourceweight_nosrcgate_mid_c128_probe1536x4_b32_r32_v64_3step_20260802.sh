#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_sourceweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullsupport-prior-softkeep-sourceweight-nosrcgate-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cfsp82c}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_min_source_answer_mass=0.0 \
  ttrl.chunk_state_min_prompt_top_mass=0.30 \
  ttrl.chunk_state_source_select_by_mass=True \
  ttrl.chunk_state_source_quality_weight_mode=source_mass \
  ttrl.chunk_state_source_quality_weight_floor=0.05 \
  ttrl.chunk_state_source_quality_weight_power=0.5 \
  "$@"
