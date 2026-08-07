#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_posterior_support_sharp_a6s8_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_posterior_support_longprobe2560_a6s8_prompt020_productweight_nosrcgate_mid_c128_probe2560x4_b32_r32_v64_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-posterior-support-longprobe2560-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cfsp82pslp}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  ttrl.chunk_state_probe_max_tokens=2560 \
  ttrl.chunk_state_staged_probe_enable=False \
  ttrl.chunk_state_future_support_min_candidate_coverage=0.0 \
  ttrl.chunk_state_future_support_min_state_coverage=0.0 \
  ttrl.chunk_state_future_support_max_state_oov=1.0 \
  ttrl.chunk_state_future_support_min_state_max_mass=0.0 \
  ttrl.chunk_state_future_support_min_state_top_margin=0.0 \
  ttrl.chunk_state_future_support_keep_mode=soft \
  ttrl.chunk_state_future_support_soft_weight_floor=0.05 \
  "$@"
