#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TTRL_DIR="$(cd "$ROOT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$ROOT_DIR/../.." && pwd)"
VENV="${VENV:-$WORKSPACE_DIR/.venvs/ttrl_b200}"
PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"
MODEL_DIR="${BACKBONE_PATH:-/tmp/Qwen3-30B-A3B-Base}"
RUN_TAG="${RUN_TAG:-qwen3_30b_a3b_b200_2gpu_ttrl_smoke_$(date +%Y%m%d_%H%M%S)}"
TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-$WORKSPACE_DIR/ttrl_runtime/$RUN_TAG}"
SHORT_RUNTIME_DIR="${SHORT_RUNTIME_DIR:-/tmp/q3b2_$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$TTRL_DIR/important_experiment_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_TAG}.log}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$TTRL_RUNTIME_DIR"/{hf,checkpoints} "$SHORT_RUNTIME_DIR"/{tmp,ray} "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_HOME="${CUDA_HOME:-$VENV/lib/python3.11/site-packages/nvidia/cu13}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib:$VENV/lib/python3.11/site-packages/nvidia/nccl/lib:$VENV/lib/python3.11/site-packages/nvidia/cublas/lib:$VENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$VENV/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$VENV/lib/python3.11/site-packages/nvidia/cudnn/lib:$VENV/lib/python3.11/site-packages/nvidia/cusparse/lib:$VENV/lib/python3.11/site-packages/nvidia/cusparselt/lib:$VENV/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$TTRL_RUNTIME_DIR/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TMPDIR="${TMPDIR:-$SHORT_RUNTIME_DIR/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-$SHORT_RUNTIME_DIR/ray}"
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

cd "$ROOT_DIR"

echo "Using python: $PYTHON_BIN"
echo "Using model: $MODEL_DIR"
echo "Using runtime dir: $TTRL_RUNTIME_DIR"
echo "Writing log: $LOG_FILE"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  data.train_files="[$ROOT_DIR/data/MATH-TTT/train.parquet]" \
  data.val_files="[$ROOT_DIR/data/MATH-TTT/test.parquet]" \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.train_batch_size="${DATA_TRAIN_BATCH_SIZE:-2}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_DIR" \
  actor_rollout_ref.model.attn_implementation="${ATTN_IMPLEMENTATION:-sdpa}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.enable_activation_offload=False \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.strategy="${ACTOR_STRATEGY:-fsdp}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE:-2}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-1}" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-False}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}" \
  actor_rollout_ref.actor.fsdp_config.offload_policy="${ACTOR_FSDP_OFFLOAD_POLICY:-False}" \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024 \
  actor_rollout_ref.ref.strategy="${REF_STRATEGY:-${ACTOR_STRATEGY:-fsdp}}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE:-1}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.forward_prefetch=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-1}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.55}" \
  actor_rollout_ref.rollout.n="${N_SAMPLES_PER_PROMPT:-2}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}" \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  +actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=FLASH_ATTN \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.moe_backend=triton \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes="${KV_CACHE_MEMORY_BYTES:-$((2 * 1024 * 1024 * 1024))}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_hybrid_kv_cache_manager=True \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path="$MODEL_DIR" \
  critic.model.attn_implementation="${ATTN_IMPLEMENTATION:-sdpa}" \
  critic.model.enable_gradient_checkpointing=True \
  critic.model.enable_activation_offload=False \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  critic.model.fsdp_config.forward_prefetch=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=grpo \
  custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  ttrl.enable=True \
  ttrl.n_votes_per_prompt="${N_VOTES_PER_PROMPT:-2}" \
  ttrl.n_samples_per_prompt="${N_SAMPLES_PER_PROMPT:-2}" \
  ttrl.majority_vote_num_processes="${MAJORITY_VOTE_NUM_PROCESSES:-8}" \
  reward_model.reward_manager=prime \
  reward_model.num_processes="${REWARD_NUM_PROCESSES:-8}" \
  trainer.logger="[console]" \
  trainer.project_name="TTRL-Qwen3-B200-Smoke" \
  trainer.experiment_name="$RUN_TAG" \
  trainer.n_gpus_per_node="${N_GPUS:-2}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.final_val_enable=False \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir="$TTRL_RUNTIME_DIR/checkpoints" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-1}" \
  trainer.total_epochs=1 \
  ray_init.num_cpus="${RAY_NUM_CPUS:-32}" \
  "$@" 2>&1 | tee "$LOG_FILE"
