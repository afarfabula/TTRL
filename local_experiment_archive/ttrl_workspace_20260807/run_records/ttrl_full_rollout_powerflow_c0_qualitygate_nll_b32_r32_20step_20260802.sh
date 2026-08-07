#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_full_rollout_powerflow_b1_postgain_nll_b32_r32_20step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_full_rollout_powerflow_c0_qualitygate_nll_b32_r32_20step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullrollout-c0-qualitygate-nll-20step}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export FULL_ROLLOUT_QUALITY_GATE_ENABLE="${FULL_ROLLOUT_QUALITY_GATE_ENABLE:-True}"
export FULL_ROLLOUT_MIN_VALID_COVERAGE="${FULL_ROLLOUT_MIN_VALID_COVERAGE:-0.50}"
export FULL_ROLLOUT_MIN_TOP_MASS="${FULL_ROLLOUT_MIN_TOP_MASS:-0.35}"
export FULL_ROLLOUT_MIN_MARGIN="${FULL_ROLLOUT_MIN_MARGIN:-0.15}"

exec bash "$BASE" "$@"
