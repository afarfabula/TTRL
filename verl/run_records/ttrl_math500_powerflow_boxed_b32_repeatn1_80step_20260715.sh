#!/usr/bin/env bash
set -euo pipefail

RUN_ID="ttrl_math500_powerflow_boxed_b32_repeatn1_80step_20260715"
ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
SCRIPT="${ROOT}/examples/ttrl/Qwen2.5-Math/math500_local.sh"
PY="/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python"

export PYTHON_BIN="${PY}"
export BACKBONE="Qwen2.5-Math-7B"
export BACKBONE_PATH="/models/Qwen2.5-Math-7B"
export DATA_LOCAL_DIR="${ROOT}/data"

export TTRL_RUNTIME_DIR="/tmp/ttrl_pf_boxed80_repeatn1"
export TMPDIR="${TTRL_RUNTIME_DIR}/tmp"
export RAY_TMPDIR="${TTRL_RUNTIME_DIR}/ray"
export OUTPUT_DIR="/tmp/ttrl_b200/checkpoints/${RUN_ID}"
export WANDB_PROJECT="TTRL-PowerFlow-local"
export EXPERIMENT="${RUN_ID}"
export LOG_NAME="0715-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME="/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${CUDA_HOME}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${TTRL_RUNTIME_DIR}/hf"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export VLLM_CACHE_ROOT="${TTRL_RUNTIME_DIR}/vllm"
export TRITON_CACHE_DIR="${TTRL_RUNTIME_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${TTRL_RUNTIME_DIR}/ti"

export DATA_TRAIN_BATCH_SIZE=32
export N_VOTES_PER_PROMPT=32
export N_SAMPLES_PER_PROMPT=32
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE=2
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=3072
export VAL_N=16
export EPISODE=10
export N_GPUS=8
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

export RAY_DASHBOARD_STARTUP_TIMEOUT_S=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=true
export TTRL_FORCE_LOCAL_NCCL=1
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="=eth0"
export NCCL_SOCKET_FAMILY=AF_INET6
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_USE_V1

rm -rf "${RAY_TMPDIR}"
mkdir -p "${TTRL_RUNTIME_DIR}" "${TMPDIR}" "${RAY_TMPDIR}" \
  "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  /tmp/ttrl_b200/logs "${OUTPUT_DIR}"

cd "${ROOT}"

exec bash "${SCRIPT}" \
  trainer.total_training_steps=80 \
  trainer.total_epochs=10 \
  trainer.resume_mode=disable \
  trainer.test_freq=20 \
  trainer.save_freq=80 \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  ttrl.powerflow_no_majority=True \
  actor_rollout_ref.actor.powerflow_enable=True \
  actor_rollout_ref.actor.powerflow_use_boxed_reward=True \
  actor_rollout_ref.actor.powerflow_beta_coef=4.0 \
  actor_rollout_ref.actor.powerflow_init_ref_log_prob=0.36 \
  actor_rollout_ref.actor.powerflow_proj_layers=3 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.use_rollout_log_probs_as_old=False \
  actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=TRITON_ATTN \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=32 \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.attn_implementation=sdpa \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=False \
  reward_model.reward_manager=dapo \
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
  +reward_model.reward_kwargs.overlong_buffer_cfg.len=0 \
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=0.0 \
  +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
  +reward_model.reward_kwargs.max_resp_len=3072 \
  custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func
