#!/bin/bash
set -euo pipefail

EXP_NAME=procfix_v26_bucket_10step
LOG=/opt/tiger/TTRL/verl/${EXP_NAME}.log
RAY_LOG_SNAPSHOT=/opt/tiger/TTRL/verl/${EXP_NAME}_ray_taskrunner.log
METRICS_SNAPSHOT=/opt/tiger/TTRL/verl/${EXP_NAME}_metrics.txt
PROC_SNAPSHOT=/opt/tiger/TTRL/verl/${EXP_NAME}_proc_health.txt
RAY_DIR=/tmp/ray_${EXP_NAME}
SRC_MODEL=/opt/tiger/qwen3_8b
LOCAL_MODEL=/tmp/qwen3_8b_local_v21_answer_sharpen

echo "PROC_FIX_VERIFY_START $(date '+%F %T')" | tee "$LOG"
echo "HOST $(hostname)" | tee -a "$LOG"
if [ ! -e /proc/self ] || [ ! -e /proc/meminfo ]; then
  echo "PROC_FIX_VERIFY_PROC_BAD_BEFORE" | tee -a "$LOG"
  exit 97
fi
ls -ld /proc /proc/self /proc/meminfo | tee -a "$LOG"
python - <<'PY' | tee -a "$LOG"
import os
print("PROC_COUNT_BEFORE", len(os.listdir("/proc")))
PY
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
export MASTER_PORT=29681
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET6
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${EXP_NAME}
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND

rm -rf "$RAY_DIR"
rm -f "$RAY_LOG_SNAPSHOT" "$METRICS_SNAPSHOT" "$PROC_SNAPSHOT"

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
  ttrl.sps_format_reward_coef=0.0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  trainer.total_epochs=1 \
  trainer.total_training_steps=10 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.experiment_name=math-qwen3_8b-procfix-v26-bucket-10step \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

{
  echo "PROC_FIX_VERIFY_STATUS ${status} $(date '+%F %T')"
  test -e /proc/self && echo PROC_SELF_OK_AFTER || echo PROC_SELF_BAD_AFTER
  test -e /proc/meminfo && echo PROC_MEMINFO_OK_AFTER || echo PROC_MEMINFO_BAD_AFTER
  python - <<'PY'
import os
print("PROC_COUNT_AFTER", len(os.listdir("/proc")))
PY
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
} 2>&1 | tee "$PROC_SNAPSHOT" | tee -a "$LOG"

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
    grep -a -E "training/global_step|timing_s/step|perf/throughput|train/sps/selected_|response_length/clip_ratio|train/pass@32" "$task_log" || true
  } > "$METRICS_SNAPSHOT"
  echo "PROC_FIX_VERIFY_RAY_LOG_SNAPSHOT $RAY_LOG_SNAPSHOT" | tee -a "$LOG"
  echo "PROC_FIX_VERIFY_METRICS_SNAPSHOT $METRICS_SNAPSHOT" | tee -a "$LOG"
else
  echo "PROC_FIX_VERIFY_RAY_LOG_MISSING" | tee -a "$LOG"
fi

if grep -a -E "Fail to open /proc|Worker exits unexpectedly|receives a SIGTERM|PROC_SELF_BAD_AFTER|PROC_MEMINFO_BAD_AFTER|PROC_COUNT_AFTER 0" "$LOG" "$RAY_LOG_SNAPSHOT" "$PROC_SNAPSHOT" >/tmp/${EXP_NAME}_bad_patterns.txt 2>/dev/null; then
  echo "PROC_FIX_VERIFY_BAD_PATTERN_FOUND" | tee -a "$LOG"
  cat /tmp/${EXP_NAME}_bad_patterns.txt | tee -a "$LOG"
else
  echo "PROC_FIX_VERIFY_NO_BAD_PATTERN" | tee -a "$LOG"
fi

echo "PROC_FIX_VERIFY_EXIT status=${status} $(date '+%F %T')" | tee -a "$LOG"
exit "$status"
