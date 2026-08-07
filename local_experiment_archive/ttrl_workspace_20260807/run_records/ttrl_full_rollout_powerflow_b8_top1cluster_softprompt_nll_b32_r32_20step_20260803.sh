#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_full_rollout_powerflow_b1_postgain_nll_b32_r32_20step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_full_rollout_powerflow_b8_top1cluster_softprompt_nll_b32_r32_20step_20260803}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0803-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullrollout-b8-top1cluster-softprompt-nll-20step}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export FULL_ROLLOUT_MODE="${FULL_ROLLOUT_MODE:-top1_answer_cluster}"
export FULL_ROLLOUT_ALPHA="${FULL_ROLLOUT_ALPHA:-6.0}"
export FULL_ROLLOUT_SLACK="${FULL_ROLLOUT_SLACK:-0.02}"
export FULL_ROLLOUT_EPS="${FULL_ROLLOUT_EPS:-0.0}"
export FULL_ROLLOUT_QUALITY_GATE_ENABLE="${FULL_ROLLOUT_QUALITY_GATE_ENABLE:-False}"
export FULL_ROLLOUT_SOFT_PROMPT_WEIGHT_ENABLE="${FULL_ROLLOUT_SOFT_PROMPT_WEIGHT_ENABLE:-True}"
export FULL_ROLLOUT_MARGIN_TAU="${FULL_ROLLOUT_MARGIN_TAU:-0.45}"
export FULL_ROLLOUT_PROMPT_WEIGHT_FLOOR="${FULL_ROLLOUT_PROMPT_WEIGHT_FLOOR:-0.35}"
export FULL_ROLLOUT_MIN_VALID_COVERAGE="${FULL_ROLLOUT_MIN_VALID_COVERAGE:-0.0}"
export FULL_ROLLOUT_MIN_TOP_MASS="${FULL_ROLLOUT_MIN_TOP_MASS:-0.0}"
export FULL_ROLLOUT_MIN_MARGIN="${FULL_ROLLOUT_MIN_MARGIN:-0.0}"

exec bash "$BASE" "$@"
