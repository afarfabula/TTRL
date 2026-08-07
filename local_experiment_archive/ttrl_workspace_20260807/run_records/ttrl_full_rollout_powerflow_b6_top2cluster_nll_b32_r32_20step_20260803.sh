#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_full_rollout_powerflow_b1_postgain_nll_b32_r32_20step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_full_rollout_powerflow_b6_top2cluster_nll_b32_r32_20step_20260803}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0803-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullrollout-b6-top2cluster-nll-20step}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export FULL_ROLLOUT_MODE="${FULL_ROLLOUT_MODE:-topk_answer_cluster}"
export FULL_ROLLOUT_TOP_K_ANSWERS="${FULL_ROLLOUT_TOP_K_ANSWERS:-2}"
export FULL_ROLLOUT_ALPHA="${FULL_ROLLOUT_ALPHA:-6.0}"
export FULL_ROLLOUT_SLACK="${FULL_ROLLOUT_SLACK:-0.02}"
export FULL_ROLLOUT_EPS="${FULL_ROLLOUT_EPS:-0.0}"

exec bash "$BASE" "$@"
