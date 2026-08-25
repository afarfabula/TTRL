#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
RUN_ID="ttrl_chunk_state_powerflow_suffixsupport_v29_answeragg_mid_suffix_eos_b32_r32_v64_20step_20260802_rerun1"
LOG="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}.log"
DIAG="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl"
VAL="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}_val_metrics.json"

mkdir -p "$(dirname "$LOG")" "$(dirname "$DIAG")"
rm -f "$LOG" "$DIAG" "$VAL"

cd "$ROOT"
RUN_ID="$RUN_ID" \
EXPERIMENT="$RUN_ID" \
LOG_NAME="0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-suffixsupport-v29-answeragg-20step-rerun1" \
TOTAL_EPOCHS=2 \
TOTAL_TRAINING_STEPS=20 \
TEST_FREQ=20 \
FINAL_VAL_ENABLE=True \
VAL_BEFORE_TRAIN=False \
SAVE_FREQ=-1 \
bash run_records/ttrl_chunk_state_powerflow_suffixsupport_v29_answeragg_mid_suffix_eos_b32_r32_v64_20step_20260802.sh 2>&1 | tee "$LOG"
