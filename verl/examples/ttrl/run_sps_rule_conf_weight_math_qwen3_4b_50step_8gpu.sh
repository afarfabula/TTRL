#!/bin/bash
# Rule-reward confidence-weighted majority TTRL on MATH-TTT (Math500), 8x GPU.
# Final-only validation: val_before_train=false, test_freq=50.
#
# Internal-feedback algorithm:
#   1. Sample 64 rollouts per prompt.
#   2. Use majority vote as the pseudo label.
#   3. Compute answer-level SPS agreement confidence from base/proposal logprobs.
#   4. Train with the rule-based 0/1 reward against the majority pseudo label,
#      multiplied by a continuous prompt weight from majority confidence,
#      SPS-majority agreement confidence, and an internal truncation penalty.
#      No ground-truth labels are used for training.
set -x

export PYTHONPATH=/opt/tiger/TTRL/verl:${PYTHONPATH}
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_sps_rule_conf_weight8}

export MY_HOST_IP=${MY_HOST_IP:-127.0.0.1}
unset MY_HOST_IPV6
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29565}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME#=}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME#=}
TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME#=}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
if [ "${TTRL_UNSET_NCCL_SOCKET_FAMILY:-0}" = "1" ]; then
  unset NCCL_SOCKET_FAMILY
else
  export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET6}
fi
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
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

ROLLOUT_TEMP=1.0
SPS_WEIGHT_TEMP_BASE=0.4
SPS_WEIGHT_FLOOR=0.35
SPS_CLIP_PENALTY=0.25

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
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.warmup_style='constant' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT + MAX_RESP)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
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
  ttrl.sps_reward_mode=answer_rule_conf_weight \
  ttrl.sps_length_normalize=True \
  ttrl.sps_weight_temperature=1.0 \
  ttrl.sps_weight_floor=$SPS_WEIGHT_FLOOR \
  ttrl.sps_clip_penalty=$SPS_CLIP_PENALTY \
  ttrl.n_votes_per_prompt=$N_VOTES \
  ttrl.n_samples_per_prompt=$N_SAMPLES \
  trainer.logger=['console'] \
  trainer.project_name=TTRL-SPS \
  trainer.experiment_name=math-qwen3_4b-sps-rule-conf-weight-50step-8gpu \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=$TOTAL_STEPS \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.total_training_steps=$TOTAL_STEPS \
  "$@"
