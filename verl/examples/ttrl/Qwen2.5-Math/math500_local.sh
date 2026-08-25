#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/mnt/local/localcache00/quyanyi/ttrl}"
PYTHON_BIN="${PYTHON_BIN:-$TTRL_RUNTIME_DIR/venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$TTRL_RUNTIME_DIR/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TMPDIR="${TMPDIR:-$TTRL_RUNTIME_DIR/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-$TTRL_RUNTIME_DIR/ray}"
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export USE_LIBUV="${USE_LIBUV:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ "${TTRL_FORCE_LOCAL_NCCL:-1}" == "1" ]]; then
  export NCCL_NET=Socket
  export NCCL_NET_PLUGIN=none
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=eth0}"
  export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}"
else
  export NCCL_NET="${NCCL_NET:-Socket}"
  export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=eth0}"
  export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}"
fi
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TMPDIR" "$RAY_TMPDIR" \
  "$TTRL_RUNTIME_DIR/models" "$TTRL_RUNTIME_DIR/logs"

DATE="$(date +%m%d)"
TIME_TAG="$(date +%H%M%S)"

TASK="${TASK:-MATH-TTT}"
BACKBONE="${BACKBONE:-Qwen2.5-Math-1.5B}"
BACKBONE_PATH="${BACKBONE_PATH:-Qwen/Qwen2.5-Math-1.5B}"
ADVANTAGE="${ADVANTAGE:-grpo}"

K="${K:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$((1024 * K))}"
VAL_N="${VAL_N:-1}"
EPISODE="${EPISODE:-1}"

DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-8}"
N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-$MICRO_BATCH_SIZE}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-$MICRO_BATCH_SIZE}"
N_GPUS="${N_GPUS:-8}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-$N_GPUS}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-False}"
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ACTOR_GRADIENT_CHECKPOINTING="${ACTOR_GRADIENT_CHECKPOINTING:-True}"
ACTOR_GRADIENT_CHECKPOINTING_DETERMINISM_CHECK="${ACTOR_GRADIENT_CHECKPOINTING_DETERMINISM_CHECK:-}"
ACTOR_GRADIENT_CHECKPOINTING_EARLY_STOP="${ACTOR_GRADIENT_CHECKPOINTING_EARLY_STOP:-}"
ACTOR_GRADIENT_CHECKPOINTING_USE_REENTRANT="${ACTOR_GRADIENT_CHECKPOINTING_USE_REENTRANT:-}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
ACTOR_ACTIVATION_OFFLOAD="${ACTOR_ACTIVATION_OFFLOAD:-False}"
MODEL_LORA_RANK="${MODEL_LORA_RANK:-0}"
MODEL_LORA_ALPHA="${MODEL_LORA_ALPHA:-16}"
MODEL_TARGET_MODULES="${MODEL_TARGET_MODULES:-all-linear}"
ACTOR_FSDP_FORWARD_PREFETCH="${ACTOR_FSDP_FORWARD_PREFETCH:-False}"
ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp}"
ACTOR_FSDP_OFFLOAD_POLICY="${ACTOR_FSDP_OFFLOAD_POLICY:-False}"
ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-True}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-dummy_dtensor}"
CRITIC_PARAM_OFFLOAD="${CRITIC_PARAM_OFFLOAD:-False}"
CRITIC_OPTIMIZER_OFFLOAD="${CRITIC_OPTIMIZER_OFFLOAD:-False}"
CRITIC_ACTIVATION_OFFLOAD="${CRITIC_ACTIVATION_OFFLOAD:-False}"
CRITIC_GRADIENT_CHECKPOINTING="${CRITIC_GRADIENT_CHECKPOINTING:-True}"
CRITIC_FSDP_FORWARD_PREFETCH="${CRITIC_FSDP_FORWARD_PREFETCH:-False}"
CRITIC_STRATEGY="${CRITIC_STRATEGY:-$ACTOR_STRATEGY}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
REF_FSDP_FORWARD_PREFETCH="${REF_FSDP_FORWARD_PREFETCH:-False}"
REF_STRATEGY="${REF_STRATEGY:-$ACTOR_STRATEGY}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-1024}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

DATA_LOCAL_DIR="${DATA_LOCAL_DIR:-$ROOT_DIR/data}"
WANDB_PROJECT="${WANDB_PROJECT:-TTRL-verl-local}"
LOGGER="${LOGGER:-console}"
MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="${EXPERIMENT:-TTRL-Math500-Len@${K}k-smoke}"
LOG_NAME="${LOG_NAME:-${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}}"

if [[ ! -f "$DATA_LOCAL_DIR/$TASK/train.parquet" || ! -f "$DATA_LOCAL_DIR/$TASK/test.parquet" ]]; then
  "$PYTHON_BIN" scripts/prepare_ttrl_json.py --data-dir "$DATA_LOCAL_DIR/$TASK" --source "$TASK"
fi

echo "Using python: $PYTHON_BIN"
echo "Using model: $BACKBONE_PATH"
echo "Using data: $DATA_LOCAL_DIR/$TASK"
echo "Output directory: $OUTPUT_DIR"
echo "Actor offload: param=$ACTOR_PARAM_OFFLOAD optimizer=$ACTOR_OPTIMIZER_OFFLOAD activation=$ACTOR_ACTIVATION_OFFLOAD"
echo "Rollout: free_cache_engine=$ROLLOUT_FREE_CACHE_ENGINE enforce_eager=$ROLLOUT_ENFORCE_EAGER gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "Critic offload: param=$CRITIC_PARAM_OFFLOAD optimizer=$CRITIC_OPTIMIZER_OFFLOAD activation=$CRITIC_ACTIVATION_OFFLOAD"

MODEL_TARGET_MODULES_OVERRIDE="actor_rollout_ref.model.target_modules=$MODEL_TARGET_MODULES"
if [[ "$MODEL_TARGET_MODULES" == *,* ]]; then
  MODEL_TARGET_MODULES_OVERRIDE="actor_rollout_ref.model.target_modules=[$MODEL_TARGET_MODULES]"
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  data.train_files="[$DATA_LOCAL_DIR/$TASK/train.parquet]" \
  data.val_files="[$DATA_LOCAL_DIR/$TASK/test.parquet]" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.train_batch_size="$DATA_TRAIN_BATCH_SIZE" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$BACKBONE_PATH" \
  actor_rollout_ref.model.attn_implementation="${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  actor_rollout_ref.model.enable_gradient_checkpointing="$ACTOR_GRADIENT_CHECKPOINTING" \
  ${ACTOR_GRADIENT_CHECKPOINTING_USE_REENTRANT:++actor_rollout_ref.model.gradient_checkpointing_kwargs.use_reentrant="$ACTOR_GRADIENT_CHECKPOINTING_USE_REENTRANT"} \
  ${ACTOR_GRADIENT_CHECKPOINTING_DETERMINISM_CHECK:++actor_rollout_ref.model.gradient_checkpointing_kwargs.determinism_check="$ACTOR_GRADIENT_CHECKPOINTING_DETERMINISM_CHECK"} \
  ${ACTOR_GRADIENT_CHECKPOINTING_EARLY_STOP:++actor_rollout_ref.model.gradient_checkpointing_kwargs.early_stop="$ACTOR_GRADIENT_CHECKPOINTING_EARLY_STOP"} \
  actor_rollout_ref.model.enable_activation_offload="$ACTOR_ACTIVATION_OFFLOAD" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.lora_rank="$MODEL_LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$MODEL_LORA_ALPHA" \
  "$MODEL_TARGET_MODULES_OVERRIDE" \
  actor_rollout_ref.actor.strategy="$ACTOR_STRATEGY" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz="$ACTOR_USE_DYNAMIC_BSZ" \
  actor_rollout_ref.actor.use_kl_loss="${ACTOR_USE_KL_LOSS:-True}" \
  actor_rollout_ref.actor.use_torch_compile="$ACTOR_USE_TORCH_COMPILE" \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.offload_policy="$ACTOR_FSDP_OFFLOAD_POLICY" \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch="$ACTOR_FSDP_FORWARD_PREFETCH" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.strategy="$REF_STRATEGY" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
  actor_rollout_ref.ref.fsdp_config.forward_prefetch="$REF_FSDP_FORWARD_PREFETCH" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  actor_rollout_ref.rollout.free_cache_engine="$ROLLOUT_FREE_CACHE_ENGINE" \
  actor_rollout_ref.rollout.load_format="$ROLLOUT_LOAD_FORMAT" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.n="$N_SAMPLES_PER_PROMPT" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  critic.optim.lr=9e-6 \
  critic.strategy="$CRITIC_STRATEGY" \
  critic.model.use_remove_padding=True \
  critic.model.path="$BACKBONE_PATH" \
  critic.model.attn_implementation="${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  critic.model.enable_gradient_checkpointing="$CRITIC_GRADIENT_CHECKPOINTING" \
  critic.model.enable_activation_offload="$CRITIC_ACTIVATION_OFFLOAD" \
  critic.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  critic.model.fsdp_config.param_offload="$CRITIC_PARAM_OFFLOAD" \
  critic.model.fsdp_config.optimizer_offload="$CRITIC_OPTIMIZER_OFFLOAD" \
  critic.model.fsdp_config.forward_prefetch="$CRITIC_FSDP_FORWARD_PREFETCH" \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator="$ADVANTAGE" \
  custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  ttrl.enable=True \
  ttrl.n_votes_per_prompt="$N_VOTES_PER_PROMPT" \
  ttrl.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT" \
  trainer.logger="[$LOGGER]" \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$LOG_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.final_val_enable="${FINAL_VAL_ENABLE:-True}" \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.total_epochs="$EPISODE" \
  "$@"
