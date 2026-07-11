#!/bin/bash
# Paper-aligned full TTRL run on MATH-TTT (500 prompts, full set), 8x B200.
# TTRL's own verl (PYTHONPATH) + modelchef venv. No modelchef changes.
# Hyperparams aligned with examples/ttrl/Qwen2.5-Math/aime.sh (paper config).
set -x

export PYTHONPATH=/opt/tiger/TTRL/verl:${PYTHONPATH}
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
export TOKENIZERS_PARALLELISM=false

# IPv6-only container: c10d store over loopback, NCCL bootstrap over eth0/IPv6.
export MY_HOST_IP=127.0.0.1
unset MY_HOST_IPV6
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29513
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET6
export TP_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=1

PY=/opt/tiger/modelchef/.venv/bin/python

# Full dataset: MATH-TTT has 500 prompts (largest). Test-time train on it.
TASK="MATH-TTT"
DATA_DIR=/opt/tiger/TTRL/verl/data/$TASK
MODEL=/mnt/hdfs/models/qwen3_4b

# Paper config (Qwen2.5-Math aime.sh): K=3
K=3
MAX_PROMPT=512
MAX_RESP=$((1024 * K))      # 3072
N_VOTES=64                  # rollouts per prompt for majority voting
N_SAMPLES=32                # rollouts per prompt used for training
EPISODE=80
TRAIN_BATCH=8

cd /opt/tiger/TTRL/verl

$PY -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  data.train_files=["$DATA_DIR/train.parquet"] \
  data.val_files=["$DATA_DIR/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT \
  data.max_response_length=$MAX_RESP \
  data.train_batch_size=$TRAIN_BATCH \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  +data.suffix_prompt='"\nPlease reason step by step, and put your final answer within \boxed{}."' \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.n=$N_SAMPLES \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
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
  trainer.project_name=TTRL-paper \
  trainer.experiment_name=math-qwen3_4b-full \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.val_before_train=True \
  trainer.total_epochs=$EPISODE \
  "$@"
