#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softprefix_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_fullsupport_prior_softprefix_highsupport_midlate_c128_probe1536x4_b32_r32_v64_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-highsupport-midlate-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cfsp82hsm}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_min_prompt_top_mass=0.35 \
  ttrl.chunk_state_min_prompt_valid_answer_coverage=0.75 \
  ttrl.chunk_state_min_prompt_top_margin=0.05 \
  ttrl.chunk_state_max_prompt_answer_entropy=2.2 \
  ttrl.chunk_state_min_boundary=128 \
  ttrl.chunk_state_mid_boundary_min_ratio=0.45 \
  ttrl.chunk_state_mid_boundary_max_ratio=0.85 \
  ttrl.chunk_state_future_support_min_state_coverage=0.50 \
  ttrl.chunk_state_future_support_max_state_oov=0.50 \
  ttrl.chunk_state_future_support_min_state_mean_mass=0.12 \
  ttrl.chunk_state_future_support_min_state_top_margin=0.02 \
  "$@"
