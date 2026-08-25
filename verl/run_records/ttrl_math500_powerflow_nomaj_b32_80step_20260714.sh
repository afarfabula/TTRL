#!/usr/bin/env bash
set -euo pipefail

RUN_ID="ttrl_math500_powerflow_nomaj_b32_80step_20260714"
ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
SCRIPT="$ROOT/examples/ttrl/Qwen2.5-Math/math500_7b_8gpu.sh"

export PYTHON_BIN="/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python"
export BACKBONE_PATH="/models/Qwen2.5-Math-7B"
export DATA_LOCAL_DIR="$ROOT/data"

export TTRL_RUNTIME_DIR="/tmp/pf80b32"
export OUTPUT_DIR="/tmp/ttrl_b200/checkpoints/${RUN_ID}"
export EXPERIMENT="${RUN_ID}"
export LOG_NAME="0714-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-powerflow"

export DATA_TRAIN_BATCH_SIZE=32
export EPISODE=10
export N_VOTES_PER_PROMPT=32
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
export ACTOR_USE_DYNAMIC_BSZ=False
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=4096
export ATTN_IMPLEMENTATION=sdpa
export LOGGER=console

export SAVE_FREQ=80
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
  trainer.total_training_steps=80 \
  trainer.total_epochs=10 \
  trainer.resume_mode=disable \
  ttrl.powerflow_no_majority=True \
  actor_rollout_ref.actor.powerflow_enable=True \
  actor_rollout_ref.actor.powerflow_use_boxed_reward=False \
  actor_rollout_ref.actor.powerflow_beta_coef=4.0 \
  actor_rollout_ref.actor.powerflow_init_ref_log_prob=0.36 \
  actor_rollout_ref.actor.powerflow_proj_layers=3 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  algorithm.norm_adv_by_std_in_grpo=False
