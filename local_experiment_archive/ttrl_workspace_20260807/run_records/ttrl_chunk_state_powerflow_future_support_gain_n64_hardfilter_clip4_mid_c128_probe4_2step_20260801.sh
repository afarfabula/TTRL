#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_future_support_gain_hardfilter_clip4_src3_mid_c128_probe4_b32_r32_v64_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_future_support_gain_n64_hardfilter_clip4_mid_c128_probe4_2step_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-future-support-n64-hardfilter-clip4-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cfsg_n64_hc4_2step}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
export TEST_FREQ="${TEST_FREQ:-2000000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-64}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.86}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  trainer.log_val_generations=0 \
  ttrl.chunk_state_source_mode=majority_consistent \
  ttrl.chunk_state_boundary_mode=mid \
  ttrl.chunk_state_chunk_size=128 \
  ttrl.chunk_state_probe_samples=4 \
  ttrl.chunk_state_probe_max_tokens=1024 \
  "$@"
