#!/bin/bash
# SPS-weighted self-consistency TTRL on MATH-TTT (Math500), 4x GPU (0/1/2/3).
# Experiment goal: 50 training steps without mid-run validation. Final validation
# still runs at the last step via trainer.test_freq because is_last_step triggers
# validation; val_before_train is disabled to avoid a startup validation pass.
#
# Internal-feedback algorithm:
#   1. Sample 64 high-temperature rollouts per prompt for exploration.
#   2. Score each rollout with base-model logprob and proposal logprob.
#   3. Cluster rollouts by extracted final answer and choose the pseudo label
#      with answer-level logsumexp SPS weight.
#   4. Train 32 selected rollouts with the normal 0/1 math reward against that
#      pseudo label. No ground-truth labels are used for training.
set -x

export PYTHONPATH=/opt/tiger/TTRL/verl:${PYTHONPATH}
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Keep this run separate from the long SPS run and the 4-7 MajVote run.
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_sps_weighted_vote}

export MY_HOST_IP=127.0.0.1
unset MY_HOST_IPV6
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29519}
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET6
export TP_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=1

PY=/opt/tiger/modelchef/.venv/bin/python

TASK="MATH-TTT"
DATA_DIR=/opt/tiger/TTRL/verl/data/$TASK
MODEL=/mnt/hdfs/models/qwen3_4b

K=3
MAX_PROMPT=512
MAX_RESP=$((1024 * K))
N_VOTES=64
N_SAMPLES=32
TRAIN_BATCH=8
TOTAL_STEPS=50

# High-temperature rollout for exploration; low-temperature SPS prior for
# answer-level weighting.
ROLLOUT_TEMP=1.0
SPS_WEIGHT_TEMP_BASE=0.4

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
  actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
  actor_rollout_ref.rollout.calculate_log_probs=True \
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
  ttrl.sps_enable=True \
  ttrl.sps_proposal_temperature=$ROLLOUT_TEMP \
  ttrl.sps_weight_temperature_base=$SPS_WEIGHT_TEMP_BASE \
  ttrl.sps_reward_mode=answer_weighted_vote \
  ttrl.sps_length_normalize=True \
  ttrl.sps_weight_temperature=1.0 \
  ttrl.n_votes_per_prompt=$N_VOTES \
  ttrl.n_samples_per_prompt=$N_SAMPLES \
  trainer.logger=['console'] \
  trainer.project_name=TTRL-SPS \
  trainer.experiment_name=math-qwen3_4b-sps-weighted-vote-50step \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=$TOTAL_STEPS \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.total_training_steps=$TOTAL_STEPS \
  "$@"
