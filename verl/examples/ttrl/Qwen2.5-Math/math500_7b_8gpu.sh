#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKSPACE_DIR="$(cd "$ROOT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_DIR/.venvs/ttrl_b200/bin/python}"
TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/ttrl_b200/runtime}"
MODEL_DIR="${BACKBONE_PATH:-/models/Qwen2.5-Math-7B}"

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$WORKSPACE_DIR/.venvs/ttrl_b200/lib/python3.11/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$WORKSPACE_DIR/.venvs/ttrl_b200/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
unset VLLM_USE_V1

"$PYTHON_BIN" "$ROOT_DIR/scripts/check_hf_model_ready.py" "$MODEL_DIR"

BACKBONE="${BACKBONE:-Qwen2.5-Math-7B}" \
BACKBONE_PATH="$MODEL_DIR" \
N_GPUS="${N_GPUS:-8}" \
DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-32}" \
N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-64}" \
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}" \
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}" \
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}" \
K="${K:-3}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}" \
VAL_N="${VAL_N:-16}" \
EPISODE="${EPISODE:-1}" \
TP_SIZE="${TP_SIZE:-1}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}" \
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-False}" \
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}" \
ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-True}" \
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-8192}" \
SAVE_FREQ="${SAVE_FREQ:--1}" \
TEST_FREQ="${TEST_FREQ:--1}" \
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}" \
EXPERIMENT="${EXPERIMENT:-TTRL-Math500-7B-8GPU-b200}" \
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/TTRL-Math500-7B-8GPU-b200}" \
USE_LIBUV="${USE_LIBUV:-0}" \
TTRL_FORCE_LOCAL_NCCL="${TTRL_FORCE_LOCAL_NCCL:-1}" \
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}" \
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=eth0}" \
NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}" \
LOGGER="${LOGGER:-console}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
PYTHON_BIN="$PYTHON_BIN" \
TTRL_RUNTIME_DIR="$TTRL_RUNTIME_DIR" \
bash "$ROOT_DIR/examples/ttrl/Qwen2.5-Math/math500_local.sh" \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=FLASH_ATTN \
  reward_model.reward_manager=prime \
  reward_model.num_processes=128 \
  ttrl.majority_vote_num_processes=64 \
  +actor_rollout_ref.rollout.logprobs=1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.use_rollout_log_probs_as_old=True \
  "$@"
