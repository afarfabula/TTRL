#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_full_rollout_powerflow_c1_postgain_eps005_nll_b32_r32_20step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_full_rollout_powerflow_e0_postgain_eps005_n64_nll_b32_20step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullrollout-e0-postgain-eps005-n64-nll-20step}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-64}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.86}"

exec bash "$BASE" "$@"
