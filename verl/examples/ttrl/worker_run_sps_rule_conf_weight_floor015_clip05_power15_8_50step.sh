#!/bin/bash
set -euo pipefail

LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_50step.log
RAY_LOG_SNAPSHOT=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_ray_taskrunner.log
METRICS_SNAPSHOT=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_metrics.txt
RAY_DIR=/tmp/ray_v12

echo "WORKER_V12_START $(date '+%F %T')" | tee "$LOG"
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

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR="$RAY_DIR"
export MASTER_PORT=29571
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_v12
rm -f "$RAY_LOG_SNAPSHOT" "$METRICS_SNAPSHOT"

bash /opt/tiger/TTRL/verl/examples/ttrl/run_sps_rule_conf_weight_math_qwen3_4b_50step_8gpu.sh \
  ttrl.sps_weight_floor=0.15 \
  ttrl.sps_clip_penalty=0.5 \
  ttrl.sps_weight_power=1.5 \
  trainer.experiment_name=math-qwen3_4b-sps-rule-conf-weight-floor015-clip05-power15-50step-8gpu \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}

task_log=$(
  grep -R -l -a "Final validation metrics\\|training/global_step:50\\|val-core/MATH-TTT/acc/mean@4" \
    "$RAY_DIR/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
)
if [ -z "$task_log" ]; then
  task_log=$(
    grep -R -l -a "Training from scratch\\|Total training steps: 50" \
      "$RAY_DIR/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
  )
fi

if [ -n "$task_log" ] && [ -f "$task_log" ]; then
  cp "$task_log" "$RAY_LOG_SNAPSHOT" || true
  {
    echo "TASK_LOG $task_log"
    grep -a -E "training/global_step|val-core/MATH-TTT/acc|val-aux/MATH-TTT/acc|Final validation metrics|global_step:50" "$task_log" || true
  } > "$METRICS_SNAPSHOT"
  echo "WORKER_V12_RAY_LOG_SNAPSHOT $RAY_LOG_SNAPSHOT" | tee -a "$LOG"
  echo "WORKER_V12_METRICS_SNAPSHOT $METRICS_SNAPSHOT" | tee -a "$LOG"
  tail -80 "$METRICS_SNAPSHOT" | tee -a "$LOG"
else
  echo "WORKER_V12_RAY_LOG_MISSING" | tee -a "$LOG"
fi
echo "WORKER_V12_EXIT status=${status} $(date '+%F %T')" | tee -a "$LOG"
exit "$status"
