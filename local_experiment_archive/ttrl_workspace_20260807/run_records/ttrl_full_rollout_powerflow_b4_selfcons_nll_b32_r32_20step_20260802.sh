#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_full_rollout_powerflow_b0_posterior_alpha4_b32_r32_20step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_full_rollout_powerflow_b4_selfcons_nll_b32_r32_20step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullrollout-b4-selfcons-nll-20step}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export FULL_ROLLOUT_MODE="${FULL_ROLLOUT_MODE:-self_consistency_advantage}"
export FULL_ROLLOUT_BETA="${FULL_ROLLOUT_BETA:-4.0}"
export FULL_ROLLOUT_WEIGHT_CLIP="${FULL_ROLLOUT_WEIGHT_CLIP:-4.0}"
export FULL_ROLLOUT_WEIGHT_CLIP_RENORM="${FULL_ROLLOUT_WEIGHT_CLIP_RENORM:-True}"

exec bash "$BASE" \
  actor_rollout_ref.actor.powerflow_enable=False \
  actor_rollout_ref.actor.chunk_weighted_nll_enable=True \
  "$@"
