#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_full_rollout_powerflow_b1_postgain_nll_b32_r32_20step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_full_rollout_powerflow_b5_top1cluster_nll_b32_r32_20step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullrollout-b5-top1cluster-nll-20step}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export FULL_ROLLOUT_MODE="${FULL_ROLLOUT_MODE:-top1_answer_cluster}"
export FULL_ROLLOUT_ALPHA="${FULL_ROLLOUT_ALPHA:-6.0}"
export FULL_ROLLOUT_SLACK="${FULL_ROLLOUT_SLACK:-0.02}"
export FULL_ROLLOUT_EPS="${FULL_ROLLOUT_EPS:-0.0}"

exec bash "$BASE" "$@"
