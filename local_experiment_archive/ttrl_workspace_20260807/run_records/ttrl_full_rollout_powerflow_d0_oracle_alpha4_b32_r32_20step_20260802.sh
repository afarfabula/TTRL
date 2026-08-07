#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_full_rollout_powerflow_b0_posterior_alpha4_b32_r32_3step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_full_rollout_powerflow_d0_oracle_alpha4_b32_r32_20step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullrollout-d0-oracle-20step}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export TEST_FREQ="${TEST_FREQ:-20}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-True}"
export FULL_ROLLOUT_MODE="${FULL_ROLLOUT_MODE:-oracle_correctness}"
export FULL_ROLLOUT_ALPHA="${FULL_ROLLOUT_ALPHA:-4.0}"
export FULL_ROLLOUT_EPS="${FULL_ROLLOUT_EPS:-0.0}"
export FULL_ROLLOUT_WEIGHT_CLIP="${FULL_ROLLOUT_WEIGHT_CLIP:-4.0}"
export FULL_ROLLOUT_WEIGHT_CLIP_RENORM="${FULL_ROLLOUT_WEIGHT_CLIP_RENORM:-True}"
export FULL_ROLLOUT_QUALITY_GATE_ENABLE="${FULL_ROLLOUT_QUALITY_GATE_ENABLE:-False}"
export FULL_ROLLOUT_POLLUTION_GUARD_ENABLE="${FULL_ROLLOUT_POLLUTION_GUARD_ENABLE:-True}"

exec bash "$BASE" "$@"
