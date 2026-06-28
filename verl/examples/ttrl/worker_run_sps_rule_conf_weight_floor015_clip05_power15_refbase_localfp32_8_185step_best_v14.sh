#!/bin/bash
set -euo pipefail

LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_refbase_localfp32_8_185step_best_v14.log
RAY_LOG_SNAPSHOT=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_refbase_localfp32_8_185step_best_v14_ray_taskrunner.log
METRICS_SNAPSHOT=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_refbase_localfp32_8_185step_best_v14_metrics.txt
RAY_DIR=/tmp/r185_v14
SRC_MODEL=/mnt/hdfs/models/qwen3_4b
LOCAL_MODEL=/tmp/qwen3_4b_local_185_v14

echo "WORKER_BEST185_V14_START $(date '+%F %T')" | tee "$LOG"
echo "HOST $(hostname)" | tee -a "$LOG"
ls -ld /proc /proc/self /proc/meminfo | tee -a "$LOG"
for attempt in 1 2 3; do
  if nvidia-smi -L | tee -a "$LOG"; then
    break
  fi
  echo "NVIDIA_SMI_RETRY attempt=${attempt} $(date '+%F %T')" | tee -a "$LOG"
  sleep 5
done
nvidia-smi -L >/dev/null

echo "MODEL_LOCAL_COPY_START $(date '+%F %T') src=${SRC_MODEL} dst=${LOCAL_MODEL}" | tee -a "$LOG"
rm -rf "${LOCAL_MODEL}.tmp"
mkdir -p "${LOCAL_MODEL}.tmp"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "${SRC_MODEL}/" "${LOCAL_MODEL}.tmp/" 2>&1 | tee -a "$LOG"
else
  cp -a "${SRC_MODEL}/." "${LOCAL_MODEL}.tmp/"
fi
find "${LOCAL_MODEL}.tmp" -maxdepth 1 -type f -printf '%f %s\n' | sort | tee -a "$LOG"
rm -rf "$LOCAL_MODEL"
mv "${LOCAL_MODEL}.tmp" "$LOCAL_MODEL"
echo "MODEL_LOCAL_COPY_DONE $(date '+%F %T')" | tee -a "$LOG"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR="$RAY_DIR"
export MASTER_PORT=29591
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET6
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_best185_v14_floor015_clip05_power15_refbase_localfp32
export TRANSFORMERS_OFFLINE=1
rm -rf "$RAY_DIR"
rm -f "$RAY_LOG_SNAPSHOT" "$METRICS_SNAPSHOT"

bash /opt/tiger/TTRL/verl/examples/ttrl/run_sps_rule_conf_weight_math_qwen3_4b_50step_8gpu.sh \
  actor_rollout_ref.model.path="$LOCAL_MODEL" \
  ttrl.sps_weight_floor=0.15 \
  ttrl.sps_clip_penalty=0.5 \
  ttrl.sps_weight_power=1.5 \
  ttrl.sps_base_logprob_source=ref \
  actor_rollout_ref.actor.use_kl_loss=True \
  trainer.total_epochs=4 \
  trainer.total_training_steps=185 \
  trainer.test_freq=185 \
  trainer.experiment_name=math-qwen3_4b-sps-rule-conf-weight-floor015-clip05-power15-refbase-localfp32-best-v14-185step-8gpu \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}

task_log=$(
  grep -R -l -a "Final validation metrics\|training/global_step:185\|val-core/MATH-TTT/acc/mean@4" \
    "$RAY_DIR/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
)
if [ -z "$task_log" ]; then
  task_log=$(
    grep -R -l -a "Training from scratch\|Total training steps: 185" \
      "$RAY_DIR/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
  )
fi

if [ -n "$task_log" ] && [ -f "$task_log" ]; then
  cp "$task_log" "$RAY_LOG_SNAPSHOT" || true
  {
    echo "TASK_LOG $task_log"
    grep -a -E "training/global_step|val-core/MATH-TTT/acc|val-aux/MATH-TTT/acc|Final validation metrics|global_step:185" "$task_log" || true
  } > "$METRICS_SNAPSHOT"
  echo "WORKER_BEST185_V14_RAY_LOG_SNAPSHOT $RAY_LOG_SNAPSHOT" | tee -a "$LOG"
  echo "WORKER_BEST185_V14_METRICS_SNAPSHOT $METRICS_SNAPSHOT" | tee -a "$LOG"
  tail -80 "$METRICS_SNAPSHOT" | tee -a "$LOG"
else
  echo "WORKER_BEST185_V14_RAY_LOG_MISSING" | tee -a "$LOG"
fi
echo "WORKER_BEST185_V14_EXIT status=${status} $(date '+%F %T')" | tee -a "$LOG"
exit "$status"
