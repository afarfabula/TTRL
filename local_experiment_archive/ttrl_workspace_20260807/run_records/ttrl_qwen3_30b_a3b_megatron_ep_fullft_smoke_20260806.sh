#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-ttrl_qwen3_30b_a3b_megatron_ep_fullft_smoke_20260806}"
ROOT="${ROOT:-/mlx_devbox/users/quyanyi/playground/TTRL/verl}"
MODEL_PATH="${MODEL_PATH:-/tmp/Qwen3-30B-A3B-Base}"
PYTHON_BIN="${PYTHON_BIN:-/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_h100_cu129/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SHORT_ID="q3moe_megatron_${STAMP}_$$"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT"

MEGATRON_DEPS_DIR="${MEGATRON_DEPS_DIR:-$ROOT/.python_deps/megatron}"
VENV_SITE_PACKAGES="$("$PYTHON_BIN" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-$VENV_SITE_PACKAGES/nvidia/nccl/include}"
NCCL_LIB_DIR="${NCCL_LIB_DIR:-$VENV_SITE_PACKAGES/nvidia/nccl/lib}"
NVRTC_LIB_DIR="${NVRTC_LIB_DIR:-$VENV_SITE_PACKAGES/nvidia/cuda_nvrtc/lib}"
CUDNN_LIB_DIR="${CUDNN_LIB_DIR:-$VENV_SITE_PACKAGES/nvidia/cudnn/lib}"
CURAND_LIB_DIR="${CURAND_LIB_DIR:-$VENV_SITE_PACKAGES/nvidia/curand/lib}"
CUDA_WHEEL_LIB_DIRS="${CUDA_WHEEL_LIB_DIRS:-$NVRTC_LIB_DIR:$VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib:$VENV_SITE_PACKAGES/nvidia/cublas/lib:$CUDNN_LIB_DIR:$CURAND_LIB_DIR:$VENV_SITE_PACKAGES/nvidia/cusparse/lib:$VENV_SITE_PACKAGES/nvidia/cusolver/lib:$NCCL_LIB_DIR}"

export PYTHONPATH="$MEGATRON_DEPS_DIR:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/tmp/${SHORT_ID}/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TMPDIR="${TMPDIR:-/tmp/${SHORT_ID}/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/r$$}"
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
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
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-lo}"
export MY_HOST_IP="${MY_HOST_IP:-127.0.0.1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_OVERRIDE_NODE_IP="${RAY_OVERRIDE_NODE_IP:-127.0.0.1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export RAY_SKIP_ENV_HOOK="${RAY_SKIP_ENV_HOOK:-1}"
export VERL_MEGATRON_SKIP_INITIAL_BROADCAST="${VERL_MEGATRON_SKIP_INITIAL_BROADCAST:-1}"
unset MY_HOST_IPV6

CUDA_HOME_DEFAULT="/usr/local/cuda"
export CUDA_HOME="${CUDA_HOME:-$CUDA_HOME_DEFAULT}"
export CPATH="$NCCL_INCLUDE_DIR${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$NCCL_INCLUDE_DIR${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$NCCL_LIB_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CUDNN_PATH="$CUDNN_LIB_DIR"
export NVRTC_PATH="$NVRTC_LIB_DIR"
export CURAND_PATH="$CURAND_LIB_DIR"
export LD_LIBRARY_PATH="$CUDA_WHEEL_LIB_DIRS:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"

NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TOTAL_TRAIN_STEPS="${TOTAL_TRAIN_STEPS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-1536}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
TRAIN_TP="${TRAIN_TP:-2}"
TRAIN_EP="${TRAIN_EP:-4}"
TRAIN_ETP="${TRAIN_ETP:-null}"
TRAIN_PP="${TRAIN_PP:-1}"
TRAIN_VPP="${TRAIN_VPP:-null}"
TRAIN_CP="${TRAIN_CP:-1}"
USE_DIST_CKPT="${USE_DIST_CKPT:-False}"
DIST_CKPT_PATH="${DIST_CKPT_PATH:-/tmp/${SHORT_ID}/dist_ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/${SHORT_ID}/ckpt}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1536}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
LOGGER="${LOGGER:-console}"

TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/MATH-TTT/train.parquet}"
VAL_FILES="${VAL_FILES:-$ROOT/data/MATH-TTT/test.parquet}"

mkdir -p "$ROOT/run_records/logs" "$TMPDIR" "$RAY_TMPDIR" "$OUTPUT_DIR" "$HF_HOME"
LOG_FILE="$ROOT/run_records/logs/${RUN_ID}_${STAMP}.log"

echo "RUN_ID=$RUN_ID"
echo "LOG_FILE=$LOG_FILE"
echo "MODEL_PATH=$MODEL_PATH"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "MEGATRON_DEPS_DIR=$MEGATRON_DEPS_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_GPUS=$NUM_GPUS TRAIN_TP=$TRAIN_TP TRAIN_EP=$TRAIN_EP TRAIN_PP=$TRAIN_PP TRAIN_CP=$TRAIN_CP ROLLOUT_TP=$ROLLOUT_TP"
echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE rollout_votes=$N_VOTES_PER_PROMPT train_samples=$N_SAMPLES_PER_PROMPT max_prompt=$MAX_PROMPT_LENGTH max_response=$MAX_RESPONSE_LENGTH"
date
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
"$PYTHON_BIN" "$ROOT/scripts/check_hf_model_ready.py" "$MODEL_PATH"
"$PYTHON_BIN" - <<'PY'
import importlib
import torch

print("preflight torch", torch.__version__, "cuda", torch.version.cuda)
for name in ["transformer_engine.pytorch", "megatron.core", "vllm"]:
    mod = importlib.import_module(name)
    print("preflight import", name, "ok", getattr(mod, "__version__", ""))
PY

if [[ "$USE_DIST_CKPT" == "True" ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/converter_hf_to_mcore.py" \
    --hf_model_path "$MODEL_PATH" \
    --output_path "$DIST_CKPT_PATH"
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-name='ppo_megatron_trainer_ttrl.yaml' \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.use_kl_in_reward=False \
  data.train_files="[$TRAIN_FILES]" \
  data.val_files="[$VAL_FILES]" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.trust_remote_code=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.override_config.moe_config.freeze_moe_router=False \
  actor_rollout_ref.actor.strategy=megatron \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$TRAIN_TP" \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size="$TRAIN_EP" \
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="$TRAIN_ETP" \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="$TRAIN_PP" \
  actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size="$TRAIN_VPP" \
  actor_rollout_ref.actor.megatron.context_parallel_size="$TRAIN_CP" \
  actor_rollout_ref.actor.megatron.sequence_parallel=True \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing="$USE_DIST_CKPT" \
  actor_rollout_ref.actor.megatron.dist_checkpointing_path="$DIST_CKPT_PATH" \
  actor_rollout_ref.actor.megatron.param_offload=False \
  actor_rollout_ref.actor.megatron.grad_offload=False \
  actor_rollout_ref.actor.megatron.optimizer_offload=False \
  actor_rollout_ref.ref.strategy=megatron \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="$TRAIN_TP" \
  actor_rollout_ref.ref.megatron.expert_model_parallel_size="$TRAIN_EP" \
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size="$TRAIN_ETP" \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="$TRAIN_PP" \
  actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size="$TRAIN_VPP" \
  actor_rollout_ref.ref.megatron.context_parallel_size="$TRAIN_CP" \
  actor_rollout_ref.ref.megatron.sequence_parallel=True \
  actor_rollout_ref.ref.megatron.use_dist_checkpointing="$USE_DIST_CKPT" \
  actor_rollout_ref.ref.megatron.dist_checkpointing_path="$DIST_CKPT_PATH" \
  actor_rollout_ref.ref.megatron.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.load_format=dummy_megatron \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.n="$N_SAMPLES_PER_PROMPT" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  critic.strategy=megatron \
  critic.model.path="$MODEL_PATH" \
  critic.model.trust_remote_code=True \
  critic.model.override_config.moe_config.freeze_moe_router=False \
  critic.megatron.tensor_model_parallel_size="$TRAIN_TP" \
  critic.megatron.expert_model_parallel_size="$TRAIN_EP" \
  critic.megatron.expert_tensor_parallel_size="$TRAIN_ETP" \
  critic.megatron.pipeline_model_parallel_size="$TRAIN_PP" \
  critic.megatron.virtual_pipeline_model_parallel_size="$TRAIN_VPP" \
  critic.megatron.context_parallel_size="$TRAIN_CP" \
  critic.megatron.sequence_parallel=True \
  critic.megatron.use_dist_checkpointing="$USE_DIST_CKPT" \
  critic.megatron.dist_checkpointing_path="$DIST_CKPT_PATH" \
  reward_model.enable=False \
  custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  ttrl.enable=True \
  ttrl.n_votes_per_prompt="$N_VOTES_PER_PROMPT" \
  ttrl.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT" \
  ttrl.majority_vote_num_processes=0 \
  ttrl.sharpened_enable=False \
  ttrl.full_rollout_powerflow_enable=False \
  ttrl.chunk_state_enable=False \
  trainer.logger="[$LOGGER]" \
  trainer.project_name=TTRL-qwen3-moe \
  trainer.experiment_name="$RUN_ID" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.balance_batch=False \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable \
  +trainer.final_val_enable=False \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$TOTAL_TRAIN_STEPS" \
  ray_init.num_cpus=64 \
  "$@" 2>&1 | tee "$LOG_FILE"
