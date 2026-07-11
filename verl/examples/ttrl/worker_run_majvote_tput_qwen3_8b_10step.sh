#!/bin/bash
set -euo pipefail

EXP_NAME=${EXP_NAME:-tput_majvote_qwen3_8b_10step}
MASTER_PORT=${MASTER_PORT:-29666}
RAY_DIR=${RAY_DIR:-/tmp/ttrl_${EXP_NAME}}
LOG=${LOG:-/opt/tiger/TTRL/verl/${EXP_NAME}.log}
RAY_LOG_SNAPSHOT=${RAY_LOG_SNAPSHOT:-/opt/tiger/TTRL/verl/${EXP_NAME}_ray_taskrunner.log}
METRICS_SNAPSHOT=${METRICS_SNAPSHOT:-/opt/tiger/TTRL/verl/${EXP_NAME}_metrics.txt}
GPU_CSV=${GPU_CSV:-/opt/tiger/TTRL/verl/${EXP_NAME}_gpu.csv}
SUMMARY=${SUMMARY:-/opt/tiger/TTRL/verl/${EXP_NAME}_throughput_summary.txt}
SRC_MODEL=${SRC_MODEL:-/opt/tiger/qwen3_8b}
LOCAL_MODEL=${LOCAL_MODEL:-/tmp/qwen3_8b_local_v21_answer_sharpen}

echo "WORKER_QWEN3_8B_MAJVOTE_TPUT_START exp=${EXP_NAME} $(date '+%F %T')" | tee "$LOG"
echo "HOST $(hostname)" | tee -a "$LOG"
if [ ! -e /proc/self ] || [ ! -e /proc/meminfo ]; then
  echo "WORKER_QWEN3_8B_MAJVOTE_TPUT_PROC_BAD $(date '+%F %T')" | tee -a "$LOG"
  ls -ld /proc /proc/self /proc/meminfo 2>&1 | tee -a "$LOG" || true
  exit 97
fi
ls -ld /proc /proc/self /proc/meminfo | tee -a "$LOG"
df -h /opt/tiger /tmp | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG"

if [ -s "$LOCAL_MODEL/config.json" ]; then
  echo "MODEL_LOCAL_COPY_REUSE $(date '+%F %T') dst=${LOCAL_MODEL}" | tee -a "$LOG"
else
  echo "MODEL_LOCAL_COPY_START $(date '+%F %T') src=${SRC_MODEL} dst=${LOCAL_MODEL}" | tee -a "$LOG"
  rm -rf "${LOCAL_MODEL}.tmp"
  mkdir -p "${LOCAL_MODEL}.tmp"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${SRC_MODEL}/" "${LOCAL_MODEL}.tmp/" 2>&1 | tee -a "$LOG"
  else
    cp -a "${SRC_MODEL}/." "${LOCAL_MODEL}.tmp/"
  fi
  rm -rf "$LOCAL_MODEL"
  mv "${LOCAL_MODEL}.tmp" "$LOCAL_MODEL"
  echo "MODEL_LOCAL_COPY_DONE $(date '+%F %T')" | tee -a "$LOG"
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/opt/tiger/TTRL/verl:${PYTHONPATH:-}
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export RAY_TMPDIR="$RAY_DIR"
export MY_HOST_IP=${MY_HOST_IP:-127.0.0.1}
unset MY_HOST_IPV6
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET6}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=1
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/ti_${EXP_NAME}}
export TRANSFORMERS_OFFLINE=1

rm -rf "$RAY_DIR"
rm -f "$RAY_LOG_SNAPSHOT" "$METRICS_SNAPSHOT" "$GPU_CSV" "$SUMMARY"

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,nounits \
  -l 5 > "$GPU_CSV" &
gpu_sampler_pid=$!
cleanup() {
  kill "$gpu_sampler_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

PY=/opt/tiger/modelchef/.venv/bin/python
DATA_DIR=/opt/tiger/TTRL/verl/data/MATH-TTT
MAX_PROMPT=512
MAX_RESP=3072

cd /opt/tiger/TTRL/verl
set +e
$PY -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  data.train_files=["$DATA_DIR/train.parquet"] \
  data.val_files=["$DATA_DIR/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT \
  data.max_response_length=$MAX_RESP \
  data.train_batch_size=8 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  +data.suffix_prompt='"\nPlease reason step by step, and put your final answer within \boxed{}."' \
  actor_rollout_ref.model.path="$LOCAL_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.warmup_style='constant' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT + MAX_RESP)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.n=32 \
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
  ttrl.sps_enable=False \
  ttrl.n_votes_per_prompt=64 \
  ttrl.n_samples_per_prompt=32 \
  trainer.logger=['console'] \
  trainer.project_name=TTRL-MajVote \
  trainer.experiment_name="math-qwen3_8b-${EXP_NAME}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.total_training_steps=10 \
  "$@" \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
cleanup

task_log=$(
  find "$RAY_DIR/ray/session_latest/logs" -maxdepth 1 -name 'worker-*-01000000-*.out' -type f -printf '%s %p\n' 2>/dev/null \
    | sort -nr \
    | awk '{print $2}' \
    | while read -r candidate; do
        if grep -a -q "training/global_step:10" "$candidate"; then
          echo "$candidate"
          break
        fi
      done
)

if [ -n "$task_log" ] && [ -f "$task_log" ]; then
  cp "$task_log" "$RAY_LOG_SNAPSHOT" || true
  {
    echo "TASK_LOG $task_log"
    grep -a -E "training/global_step|train/ground_truth_reward|train/pass@32|train/majority_ratio|response_length/mean|response_length/clip_ratio|timing_s/step|timing_s/gen|timing_s/generate_sequences|timing_s/old_log_prob|timing_s/ref|timing_s/update_actor|perf/total_num_tokens|perf/throughput" "$task_log" || true
  } > "$METRICS_SNAPSHOT"
  /opt/tiger/TTRL/verl/examples/ttrl/parse_ttrl_throughput.py "$task_log" --gpu-csv "$GPU_CSV" --last 10 > "$SUMMARY" || true
  echo "WORKER_QWEN3_8B_MAJVOTE_TPUT_RAY_LOG_SNAPSHOT $RAY_LOG_SNAPSHOT" | tee -a "$LOG"
  echo "WORKER_QWEN3_8B_MAJVOTE_TPUT_METRICS_SNAPSHOT $METRICS_SNAPSHOT" | tee -a "$LOG"
  echo "WORKER_QWEN3_8B_MAJVOTE_TPUT_SUMMARY $SUMMARY" | tee -a "$LOG"
  cat "$SUMMARY" | tee -a "$LOG"
else
  echo "WORKER_QWEN3_8B_MAJVOTE_TPUT_RAY_LOG_MISSING" | tee -a "$LOG"
fi

echo "WORKER_QWEN3_8B_MAJVOTE_TPUT_EXIT exp=${EXP_NAME} status=${status} $(date '+%F %T')" | tee -a "$LOG"
exit "$status"
