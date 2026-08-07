#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_b32_r32_v64_20step_20260731}"
ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
SCRIPT="$ROOT/examples/ttrl/Qwen2.5-Math/math500_7b_8gpu.sh"

export PYTHON_BIN="/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python"
export BACKBONE_PATH="/models/Qwen2.5-Math-7B"
export DATA_LOCAL_DIR="$ROOT/data"

export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cspfb32r32v64s20}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0731-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-chunk-state}"

export DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-32}"
export EPISODE="${EPISODE:-10}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
export VAL_N="${VAL_N:-16}"
export TP_SIZE="${TP_SIZE:-1}"

export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}"
export ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-False}"
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-2048}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export LOGGER="${LOGGER:-console}"

export SAVE_FREQ="${SAVE_FREQ:-2000000}"
export TEST_FREQ="${TEST_FREQ:-20}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"

export TTRL_FORCE_LOCAL_NCCL=1
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="=eth0"
export NCCL_SOCKET_FAMILY=AF_INET6

rm -rf "$TTRL_RUNTIME_DIR"
mkdir -p "$TTRL_RUNTIME_DIR" "$OUTPUT_DIR" /tmp/ttrl_b200/logs

exec bash "$SCRIPT" \
  trainer.resume_mode=disable \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  ttrl.chunk_state_enable=True \
  ttrl.chunk_state_candidates=8 \
  ttrl.chunk_state_chunk_size=256 \
  ttrl.chunk_state_probe_max_tokens=3072 \
  ttrl.chunk_state_states_per_prompt=1 \
  ttrl.chunk_state_max_prefix_tokens=1024 \
  ttrl.chunk_state_boundaries='[0,256,512,768,1024]' \
  ttrl.chunk_state_alpha=2.0 \
  ttrl.chunk_state_eps=0.05 \
  ttrl.chunk_state_skip_uniform=False \
  actor_rollout_ref.actor.chunk_weighted_nll_enable=False \
  actor_rollout_ref.actor.powerflow_enable=True \
  actor_rollout_ref.actor.powerflow_use_boxed_reward=True \
  actor_rollout_ref.actor.powerflow_beta_coef=4.0 \
  actor_rollout_ref.actor.powerflow_init_ref_log_prob=0.36 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_epochs=1 \
  "$@"
