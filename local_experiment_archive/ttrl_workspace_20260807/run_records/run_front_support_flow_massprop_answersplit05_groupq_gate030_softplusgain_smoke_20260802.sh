#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
RUN_ID="ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_groupq_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802"
LOG="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}.log"
DIAG="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl"

mkdir -p "$(dirname "$LOG")" "$(dirname "$DIAG")"
rm -f "$LOG" "$DIAG"

cd "$ROOT"
bash "run_records/${RUN_ID}.sh" 2>&1 | tee "$LOG"
