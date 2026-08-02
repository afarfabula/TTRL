#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_3step_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_20step_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullcand-longprobe-20step-gate}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-True}"
export TEST_FREQ="${TEST_FREQ:-20}"
export SAVE_FREQ="${SAVE_FREQ:--1}"

exec bash "$BASE" "$@"
