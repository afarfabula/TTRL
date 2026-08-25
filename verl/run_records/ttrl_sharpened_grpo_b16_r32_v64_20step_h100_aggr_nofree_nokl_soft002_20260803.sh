#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-ttrl_sharpened_grpo_b16_r32_v64_20step_h100_aggr_nofree_nokl_soft002_20260803}"
ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
SCRIPT="$ROOT/examples/ttrl/Qwen2.5-Math/math500_7b_8gpu.sh"

export PYTHON_BIN="/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_h100_cu129/bin/python"
export BACKBONE_PATH="/models/Qwen2.5-Math-7B"
export DATA_LOCAL_DIR="$ROOT/data"

H100_VENV="/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_h100_cu129"
H100_NVIDIA_LIBS="$(find "$H100_VENV/lib/python3.11/site-packages/nvidia" -maxdepth 2 -type d -name lib | paste -sd: -)"
export CUDA_HOME="/usr/local/cuda-12.9"
export VENV_CUDA_LIB_PATH="$H100_NVIDIA_LIBS"
export VLLM_ATTENTION_BACKEND_CFG=""
export VLLM_USE_FLASHINFER_SAMPLER=1

export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/smva02h10020b16aggr}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_h100/checkpoints/${RUN_ID}}"
export EXPERIMENT="${EXPERIMENT:-$RUN_ID}"
export LOG_NAME="${LOG_NAME:-0803-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-grpo-h100-aggr}"

export DATA_TRAIN_BATCH_SIZE=16
export EPISODE=10
export N_VOTES_PER_PROMPT=64
export N_SAMPLES_PER_PROMPT=32
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=3072
export VAL_N=16
export TP_SIZE=1

export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
export ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
export REF_PARAM_OFFLOAD=True
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
export ACTOR_USE_DYNAMIC_BSZ=False
export ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-False}"
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=4096
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export LOGGER=console

export SAVE_FREQ=2000000
export TEST_FREQ=20
export VAL_BEFORE_TRAIN=False

export TTRL_FORCE_LOCAL_NCCL=1
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export MY_HOST_IP=127.0.0.1
unset MY_HOST_IPV6
export MASTER_ADDR=127.0.0.1
export RAY_OVERRIDE_NODE_IP=127.0.0.1
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=eth0
if [[ -z "${NCCL_SOCKET_FAMILY+x}" ]]; then
  unset NCCL_SOCKET_FAMILY
fi

rm -rf "$TTRL_RUNTIME_DIR"
mkdir -p "$TTRL_RUNTIME_DIR" "$OUTPUT_DIR" /tmp/ttrl_h100/logs

cd "$ROOT"
exec bash "$SCRIPT" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-20}" \
  trainer.total_epochs=10 \
  ttrl.sharpened_enable=True \
  ttrl.sharpened_alpha=2.0 \
  ttrl.sharpened_tau_pos=0.375 \
  ttrl.sharpened_tau_marg=0.125 \
  ttrl.sharpened_use_confidence=True \
  ttrl.sharpened_reward_mode=mv_anchor \
  ttrl.sharpened_soft_coef=0.02 \
  ttrl.sharpened_negative_enable=False \
  ttrl.sharpened_tau_low=0.125 \
  ttrl.sharpened_negative_reward=0.0 \
  trainer.resume_mode=disable
