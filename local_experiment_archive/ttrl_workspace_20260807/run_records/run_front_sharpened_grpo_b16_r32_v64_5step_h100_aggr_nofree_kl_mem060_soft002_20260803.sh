#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
RUN_ID="ttrl_sharpened_grpo_b16_r32_v64_5step_h100_aggr_nofree_kl_mem060_soft002_20260803"
LOG="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}.log"

mkdir -p "$(dirname "$LOG")"
rm -f "$LOG"

cd "$ROOT"
RUN_ID="$RUN_ID" \
EXPERIMENT="$RUN_ID" \
LOG_NAME="0803-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-grpo-h100-aggr" \
TTRL_RUNTIME_DIR="/tmp/smva02h1005b16aggr060kl" \
OUTPUT_DIR="/tmp/ttrl_h100/checkpoints/${RUN_ID}" \
GPU_MEMORY_UTILIZATION=0.60 \
ROLLOUT_FREE_CACHE_ENGINE=False \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192 \
ACTOR_USE_KL_LOSS=True \
TOTAL_TRAINING_STEPS=5 \
TEST_FREQ=2000000 \
SAVE_FREQ=2000000 \
bash run_records/ttrl_sharpened_grpo_b16_r32_v64_20step_h100_aggr_nofree_nokl_soft002_20260803.sh 2>&1 | tee "$LOG"
