#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-ttrl_qwen3_30b_moe_math500_grpo_1step_smoke_20260806}"
ROOT="${ROOT:-/mlx_devbox/users/quyanyi/playground/TTRL/verl}"
SCRIPT="$ROOT/examples/ttrl/Qwen2.5-Math/math500_local.sh"
MODEL_PATH="${MODEL_PATH:-/tmp/Qwen3-30B-A3B-Base}"
PYTHON_BIN="${PYTHON_BIN:-/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_h100_cu129/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SHORT_ID="q3moe_${STAMP}_$$"

export BACKBONE="Qwen3-30B-A3B-Base"
export BACKBONE_PATH="$MODEL_PATH"
export DATA_LOCAL_DIR="$ROOT/data"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/${SHORT_ID}/r}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/${SHORT_ID}/ckpt}"
export EXPERIMENT="$RUN_ID"
export LOG_NAME="${STAMP}-${RUN_ID}-MATH-TTT-Qwen3-30B-A3B-grpo"

export PYTHON_BIN
export N_GPUS="${N_GPUS:-8}"
export DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-8}"
export EPISODE="${EPISODE:-1}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-1}"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
export VAL_N="${VAL_N:-2}"
export TP_SIZE="${TP_SIZE:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1536}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
export ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"

export ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-False}"
export ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-False}"
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-1536}"
export ACTOR_GRADIENT_CHECKPOINTING="${ACTOR_GRADIENT_CHECKPOINTING:-False}"
export ACTOR_GRADIENT_CHECKPOINTING_USE_REENTRANT="${ACTOR_GRADIENT_CHECKPOINTING_USE_REENTRANT:-}"
export ACTOR_GRADIENT_CHECKPOINTING_DETERMINISM_CHECK="${ACTOR_GRADIENT_CHECKPOINTING_DETERMINISM_CHECK:-}"
export ACTOR_GRADIENT_CHECKPOINTING_EARLY_STOP="${ACTOR_GRADIENT_CHECKPOINTING_EARLY_STOP:-}"
export MODEL_LORA_RANK="${MODEL_LORA_RANK:-8}"
export MODEL_LORA_ALPHA="${MODEL_LORA_ALPHA:-16}"
export MODEL_TARGET_MODULES="${MODEL_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
export ACTOR_ACTIVATION_OFFLOAD="${ACTOR_ACTIVATION_OFFLOAD:-False}"
export ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp2}"
export ACTOR_FSDP_OFFLOAD_POLICY="${ACTOR_FSDP_OFFLOAD_POLICY:-False}"
export ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-False}"

export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
export REF_STRATEGY="${REF_STRATEGY:-$ACTOR_STRATEGY}"
export CRITIC_STRATEGY="${CRITIC_STRATEGY:-$ACTOR_STRATEGY}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export TEST_FREQ="${TEST_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"
export LOGGER="${LOGGER:-console}"

export TTRL_FORCE_LOCAL_NCCL="${TTRL_FORCE_LOCAL_NCCL:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export MY_HOST_IP="${MY_HOST_IP:-127.0.0.1}"
unset MY_HOST_IPV6
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_OVERRIDE_NODE_IP="${RAY_OVERRIDE_NODE_IP:-127.0.0.1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_SOCKET_FAMILY="AF_INET"
export NCCL_DEBUG="WARN"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export RAY_SKIP_ENV_HOOK="${RAY_SKIP_ENV_HOOK:-1}"

CUDA_HOME_DEFAULT="/usr/local/cuda"
export CUDA_HOME="${CUDA_HOME:-$CUDA_HOME_DEFAULT}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

mkdir -p "$ROOT/run_records/logs" "$TTRL_RUNTIME_DIR" "$OUTPUT_DIR"
LOG_FILE="$ROOT/run_records/logs/${RUN_ID}_${STAMP}.log"

echo "RUN_ID=$RUN_ID"
echo "LOG_FILE=$LOG_FILE"
echo "MODEL_PATH=$MODEL_PATH"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "TTRL_RUNTIME_DIR=$TTRL_RUNTIME_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "N_GPUS=$N_GPUS TP_SIZE=$TP_SIZE train_batch=$DATA_TRAIN_BATCH_SIZE rollout_n=$N_SAMPLES_PER_PROMPT response_len=$MAX_RESPONSE_LENGTH"
date
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
"$PYTHON_BIN" "$ROOT/scripts/check_hf_model_ready.py" "$MODEL_PATH"

exec bash "$SCRIPT" \
  algorithm.adv_estimator=grpo \
  trainer.total_training_steps=1 \
  trainer.resume_mode=disable \
  trainer.final_val_enable=False \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.critic_warmup=0 \
  actor_rollout_ref.model.trust_remote_code=True \
  data.trust_remote_code=True \
  ttrl.enable=True \
  ttrl.sharpened_enable=False \
  ttrl.full_rollout_powerflow_enable=False \
  ttrl.chunk_state_enable=False \
  "$@"
