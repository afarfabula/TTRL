#!/bin/bash
# 8x B200 TTRL run (~1h) for throughput & stability on modelchef venv.
# Uses TTRL's own verl (PYTHONPATH) + modelchef venv deps. No modelchef changes.
set -x

export PYTHONPATH=/opt/tiger/TTRL/verl:${PYTHONPATH}
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
export TOKENIZERS_PARALLELISM=false

# IPv6-only container: c10d store over loopback, NCCL bootstrap over eth0/IPv6.
export MY_HOST_IP=127.0.0.1
unset MY_HOST_IPV6
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29512
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET6
export TP_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=1

PY=/opt/tiger/modelchef/.venv/bin/python

DATA_DIR=/opt/tiger/TTRL/verl/data/AIME-TTT
MODEL=/mnt/hdfs/models/qwen3_4b

# TTRL sampling
N_VOTES=16
N_SAMPLES=16
MAX_PROMPT=1024
MAX_RESP=3072

cd /opt/tiger/TTRL/verl

$PY -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  data.train_files=["$DATA_DIR/train.parquet"] \
  data.val_files=["$DATA_DIR/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT \
  data.max_response_length=$MAX_RESP \
  data.train_batch_size=8 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT + MAX_RESP)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=$N_SAMPLES \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT + MAX_RESP)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT + MAX_RESP)) \
  ~actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.adv_estimator=grpo \
  custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  ttrl.enable=True \
  ttrl.n_votes_per_prompt=$N_VOTES \
  ttrl.n_samples_per_prompt=$N_SAMPLES \
  trainer.logger=['console'] \
  trainer.project_name=TTRL-8gpu \
  trainer.experiment_name=aime-qwen3_4b-8gpu \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=10 \
  trainer.val_before_train=False \
  trainer.total_epochs=200 \
  "$@"
