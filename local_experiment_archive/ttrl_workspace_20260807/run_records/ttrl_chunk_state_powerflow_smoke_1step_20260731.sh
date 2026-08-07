#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="ttrl_chunk_state_powerflow_smoke_1step_20260731"
ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
SCRIPT="$ROOT/run_records/ttrl_chunk_state_powerflow_b32_r32_v64_20step_20260731.sh"

export DATA_TRAIN_BATCH_SIZE=8
export N_VOTES_PER_PROMPT=2
export N_SAMPLES_PER_PROMPT=1
export MINI_BATCH_SIZE=8
export MICRO_BATCH_SIZE=1
export VAL_N=2
export TEST_FREQ=-1
export TOTAL_TRAINING_STEPS=1
export TTRL_RUNTIME_DIR="/tmp/cspfsmoke1"
export OUTPUT_DIR="/tmp/ttrl_b200/checkpoints/${RUN_ID}"
export EXPERIMENT="${RUN_ID}"
export LOG_NAME="0731-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-chunk-state"

exec bash "$SCRIPT" \
  ttrl.chunk_state_candidates=2 \
  ttrl.chunk_state_states_per_prompt=1 \
  ttrl.chunk_state_chunk_size=128 \
  ttrl.chunk_state_probe_max_tokens=512 \
  ttrl.chunk_state_boundaries='[0,128,256]'
