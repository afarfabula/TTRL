#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_suffixsupport_v27_mid_suffix_eos_b32_r32_v64_3step_20260802.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_suffixsupport_v28_dedup_mid_suffix_eos_b32_r32_v64_20step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-suffixsupport-v28-dedup-20step}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export TEST_FREQ="${TEST_FREQ:-20}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-True}"

exec bash "$BASE" \
  trainer.validation_metric_dump_path="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}_val_metrics.json" \
  ttrl.chunk_state_suffix_support_split_duplicates=True \
  ttrl.chunk_state_suffix_support_answer_split_power=1.0 \
  "$@"
