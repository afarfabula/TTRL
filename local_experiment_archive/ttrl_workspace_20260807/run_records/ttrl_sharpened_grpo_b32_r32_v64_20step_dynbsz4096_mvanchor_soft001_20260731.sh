#!/usr/bin/env bash
set -euo pipefail

RUN_ID="ttrl_sharpened_grpo_b32_r32_v64_20step_dynbsz4096_mvanchor_soft001_20260731"
ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
SCRIPT="$ROOT/examples/ttrl/Qwen2.5-Math/math500_7b_8gpu.sh"

export PYTHON_BIN="/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python"
export BACKBONE_PATH="/models/Qwen2.5-Math-7B"
export DATA_LOCAL_DIR="$ROOT/data"

export TTRL_RUNTIME_DIR="/tmp/smva0120"
export OUTPUT_DIR="/tmp/ttrl_b200/checkpoints/${RUN_ID}"
export EXPERIMENT="${RUN_ID}"
export LOG_NAME="0731-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-grpo"

export DATA_TRAIN_BATCH_SIZE=32
export EPISODE=2
export N_VOTES_PER_PROMPT=64
export N_SAMPLES_PER_PROMPT=32
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE=2
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=3072
export VAL_N=16
export TP_SIZE=1

export GPU_MEMORY_UTILIZATION=0.80
export REF_PARAM_OFFLOAD=True
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096
export ACTOR_USE_DYNAMIC_BSZ=True
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=4096
export ATTN_IMPLEMENTATION=flash_attention_2
export LOGGER=console

export SAVE_FREQ=2000000
export TEST_FREQ=20
export VAL_BEFORE_TRAIN=False

export TTRL_FORCE_LOCAL_NCCL=1
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="=eth0"
export NCCL_SOCKET_FAMILY=AF_INET6

rm -rf "$TTRL_RUNTIME_DIR"
mkdir -p "$TTRL_RUNTIME_DIR" "$OUTPUT_DIR" /tmp/ttrl_b200/logs

exec bash "$SCRIPT" \
  trainer.total_training_steps=20 \
  trainer.total_epochs=2 \
  ttrl.sharpened_enable=True \
  ttrl.sharpened_alpha=2.0 \
  ttrl.sharpened_tau_pos=0.375 \
  ttrl.sharpened_tau_marg=0.125 \
  ttrl.sharpened_use_confidence=True \
  ttrl.sharpened_reward_mode=mv_anchor \
  ttrl.sharpened_soft_coef=0.01 \
  ttrl.sharpened_negative_enable=False \
  ttrl.sharpened_tau_low=0.125 \
  ttrl.sharpened_negative_reward=0.0 \
  trainer.resume_mode=disable
