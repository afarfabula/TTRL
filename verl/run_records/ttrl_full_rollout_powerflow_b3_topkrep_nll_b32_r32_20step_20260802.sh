#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_full_rollout_powerflow_b0_posterior_alpha4_b32_r32_20step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_full_rollout_powerflow_b3_topkrep_nll_b32_r32_20step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullrollout-b3-topkrep-nll-20step}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export FULL_ROLLOUT_MODE="${FULL_ROLLOUT_MODE:-topk_representative}"
export FULL_ROLLOUT_ALPHA="${FULL_ROLLOUT_ALPHA:-4.0}"
export FULL_ROLLOUT_TOP_K_ANSWERS="${FULL_ROLLOUT_TOP_K_ANSWERS:-2}"
export FULL_ROLLOUT_REPRESENTATIVES_PER_ANSWER="${FULL_ROLLOUT_REPRESENTATIVES_PER_ANSWER:-2}"

exec bash "$BASE" \
  actor_rollout_ref.actor.powerflow_enable=False \
  actor_rollout_ref.actor.chunk_weighted_nll_enable=True \
  "$@"
