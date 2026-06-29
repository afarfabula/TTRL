#!/bin/bash
set -euo pipefail

EXP_NAME=${EXP_NAME:-tput_baseline_v28_alg_10step}
MASTER_PORT=${MASTER_PORT:-29640}
RAY_DIR=${RAY_DIR:-/tmp/ttrl_${EXP_NAME}}
LOG=${LOG:-/opt/tiger/TTRL/verl/${EXP_NAME}.log}
RAY_LOG_SNAPSHOT=${RAY_LOG_SNAPSHOT:-/opt/tiger/TTRL/verl/${EXP_NAME}_ray_taskrunner.log}
METRICS_SNAPSHOT=${METRICS_SNAPSHOT:-/opt/tiger/TTRL/verl/${EXP_NAME}_metrics.txt}
GPU_CSV=${GPU_CSV:-/opt/tiger/TTRL/verl/${EXP_NAME}_gpu.csv}
SUMMARY=${SUMMARY:-/opt/tiger/TTRL/verl/${EXP_NAME}_throughput_summary.txt}
SRC_MODEL=${SRC_MODEL:-/opt/tiger/qwen3_8b}
LOCAL_MODEL=${LOCAL_MODEL:-/tmp/qwen3_8b_local_v21_answer_sharpen}

echo "WORKER_QWEN3_8B_TPUT_START exp=${EXP_NAME} $(date '+%F %T')" | tee "$LOG"
echo "HOST $(hostname)" | tee -a "$LOG"
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
export RAY_TMPDIR="$RAY_DIR"
export MASTER_PORT
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET6}
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

set +e
bash /opt/tiger/TTRL/verl/examples/ttrl/run_sps_rule_conf_weight_math_qwen3_4b_50step_8gpu.sh \
  actor_rollout_ref.model.path="$LOCAL_MODEL" \
  ttrl.sps_weight_floor=0.15 \
  ttrl.sps_clip_penalty=0.5 \
  ttrl.sps_weight_power=1.5 \
  ttrl.sps_base_logprob_source=ref \
  ttrl.sps_answer_sharpen_beta=2.0 \
  ttrl.sps_answer_sharpen_capacity=False \
  ttrl.sps_rollout_selection=sharpened_cluster \
  ttrl.sps_selection_temperature=0.4 \
  ttrl.sps_selection_require_majority=False \
  ttrl.sps_selection_cluster_bonus=1.0 \
  ttrl.sps_selection_parseable_bonus=3.0 \
  ttrl.sps_selection_nonclip_bonus=4.0 \
  ttrl.sps_selection_priority=nonclip_parseable_bucket \
  ttrl.sps_selection_capacity=True \
  ttrl.sps_selection_capacity_floor=0.55 \
  ttrl.sps_selection_capacity_power=0.5 \
  ttrl.sps_format_reward_coef=0.0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  trainer.total_epochs=1 \
  trainer.total_training_steps=10 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.experiment_name="math-qwen3_8b-${EXP_NAME}" \
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
    grep -a -E "training/global_step|train/sps/selected_|train/sps/selection_fallback|train/sps/selection_capacity|train/sps/train_weight|train/ground_truth_reward|train/pass@32|train/majority_ratio|response_length/mean|response_length/clip_ratio|timing_s/step|timing_s/gen|timing_s/generate_sequences|timing_s/old_log_prob|timing_s/ref|timing_s/update_actor|perf/total_num_tokens|perf/throughput" "$task_log" || true
  } > "$METRICS_SNAPSHOT"
  /opt/tiger/TTRL/verl/examples/ttrl/parse_ttrl_throughput.py "$task_log" --gpu-csv "$GPU_CSV" --last 10 > "$SUMMARY" || true
  echo "WORKER_QWEN3_8B_TPUT_RAY_LOG_SNAPSHOT $RAY_LOG_SNAPSHOT" | tee -a "$LOG"
  echo "WORKER_QWEN3_8B_TPUT_METRICS_SNAPSHOT $METRICS_SNAPSHOT" | tee -a "$LOG"
  echo "WORKER_QWEN3_8B_TPUT_SUMMARY $SUMMARY" | tee -a "$LOG"
  cat "$SUMMARY" | tee -a "$LOG"
else
  echo "WORKER_QWEN3_8B_TPUT_RAY_LOG_MISSING" | tee -a "$LOG"
fi

echo "WORKER_QWEN3_8B_TPUT_EXIT exp=${EXP_NAME} status=${status} $(date '+%F %T')" | tee -a "$LOG"
exit "$status"
