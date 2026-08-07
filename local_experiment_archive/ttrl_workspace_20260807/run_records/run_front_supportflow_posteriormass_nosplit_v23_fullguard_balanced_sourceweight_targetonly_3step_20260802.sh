#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
RUN_ID="ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v23_fullguard_balanced_sourceweight_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802"
LOG="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}.log"
DIAG="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl"
VAL="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}_val_metrics.json"

mkdir -p "$(dirname "$LOG")" "$(dirname "$DIAG")"
rm -f "$LOG" "$DIAG" "$VAL"

cd "$ROOT"
bash "run_records/${RUN_ID}.sh" 2>&1 | tee "$LOG"
