#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_futuregain_posteriormatch_softkeep_fullsupport_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_futuregain_posteriormatch_supportmixed_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-posterior-support-match-supportmixed-smoke}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_source_mode=support_mixed \
  ttrl.chunk_state_source_mixed_low_ratio=0.50 \
  ttrl.chunk_state_source_low_max_answer_mass=0.35 \
  ttrl.chunk_state_source_min_valid_answer_mass=0.03125 \
  ttrl.chunk_state_min_source_answer_mass=0.0 \
  ttrl.chunk_state_min_prompt_top_mass=0.30 \
  ttrl.chunk_state_min_prompt_valid_answer_coverage=0.50 \
  ttrl.chunk_state_source_select_by_mass=False \
  "$@"
