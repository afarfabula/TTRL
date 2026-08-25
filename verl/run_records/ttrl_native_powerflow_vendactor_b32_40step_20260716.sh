#!/usr/bin/env bash
set -euo pipefail

RUN_ID="ttrl_native_powerflow_vendactor_b32_40step_20260716"
ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
PY="/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME="/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${CUDA_HOME}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TTRL_RUNTIME_DIR="/tmp/pfttrl40va"
export TMPDIR="${TTRL_RUNTIME_DIR}/tmp"
export RAY_TMPDIR="${TTRL_RUNTIME_DIR}/ray"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_DASHBOARD_STARTUP_TIMEOUT_S=1
export RAY_USAGE_STATS_ENABLED=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="=eth0"
export NCCL_SOCKET_FAMILY=AF_INET6
export NCCL_NVLS_ENABLE=1
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_USE_FLASHINFER_SAMPLER=0
export POWERFLOW_ATTN_IMPL=sdpa
unset VLLM_USE_V1

rm -rf "${RAY_TMPDIR}"
mkdir -p "${TTRL_RUNTIME_DIR}" "${TMPDIR}" "${RAY_TMPDIR}" /tmp/ttrl_b200/logs /tmp/ttrl_b200/checkpoints
cd "${ROOT}"

exec "${PY}" -m powerflow.main_powerflow \
  +ray_kwargs.ray_init.no_runtime_env=True \
  +ray_kwargs.ray_init.include_dashboard=False \
  data.train_files="[/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT/train.parquet]" \
  data.val_files="[/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT/test.parquet]" \
  data.prompt_key=prompt \
  data.truncation=error \
  data.max_prompt_length=1024 \
  data.max_response_length=3072 \
  data.gen_batch_size=32 \
  data.train_batch_size=32 \
  +data.seed=null \
  actor_rollout_ref.rollout.n=32 \
  algorithm.adv_estimator=grpo_ref \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.model.path=/models/Qwen2.5-Math-7B \
  actor_rollout_ref.model.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.beta_coef=4.0 \
  actor_rollout_ref.actor.use_boxed_reward=True \
  actor_rollout_ref.actor.init_ref_log_prob=0.36 \
  actor_rollout_ref.actor.powerflow_enable=True \
  actor_rollout_ref.actor.powerflow_beta_coef=4.0 \
  actor_rollout_ref.actor.powerflow_use_boxed_reward=True \
  actor_rollout_ref.actor.powerflow_init_ref_log_prob=0.36 \
  actor_rollout_ref.actor.powerflow_proj_layers=3 \
  actor_rollout_ref.actor.powerflow_on_policy=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=False \
  actor_rollout_ref.ref.fsdp_config.forward_prefetch=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=TRITON_ATTN \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  reward_model.reward_manager=dapo \
  custom_reward_function.path=/mlx_devbox/users/quyanyi/playground/TTRL/verl/verl/utils/reward_score/ttrl_math/__init__.py \
  custom_reward_function.name=reward_func \
  trainer.logger="[console]" \
  trainer.project_name=TTRL-PowerFlow-local \
  trainer.experiment_name="${RUN_ID}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.test_freq=20 \
  trainer.save_freq=40 \
  trainer.total_training_steps=40 \
  trainer.total_epochs=10 \
  trainer.default_local_dir=/tmp/ttrl_b200/checkpoints/${RUN_ID} \
  trainer.resume_mode=disable
