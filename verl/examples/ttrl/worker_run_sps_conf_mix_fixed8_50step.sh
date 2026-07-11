#!/bin/bash
set -euo pipefail

LOG=/opt/tiger/TTRL/verl/sps_conf_mix_fixed8_50step.log
RAY_LOG_SNAPSHOT=/opt/tiger/TTRL/verl/sps_conf_mix_fixed8_ray_taskrunner.log
METRICS_SNAPSHOT=/opt/tiger/TTRL/verl/sps_conf_mix_fixed8_metrics.txt
echo "WORKER_V5B_START $(date '+%F %T')" | tee "$LOG"
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
export RAY_TMPDIR=/tmp/ray_sps_conf_mix_fixed8
export MASTER_PORT=29555
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_sps_mix_fixed8
rm -f "$RAY_LOG_SNAPSHOT" "$METRICS_SNAPSHOT"

bash /opt/tiger/TTRL/verl/examples/ttrl/run_sps_conf_weight_majority_mix_fixed_math_qwen3_4b_50step_8gpu.sh 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
task_log=$(find /tmp/ray_sps_conf_mix_fixed8/ray/session_latest/logs -maxdepth 1 -name 'worker-*-01000000-*.out' -type f 2>/dev/null | sort | tail -1 || true)
if [ -n "$task_log" ] && [ -f "$task_log" ]; then
  cp "$task_log" "$RAY_LOG_SNAPSHOT" || true
  {
    echo "TASK_LOG $task_log"
    grep -a -E "training/global_step|val-core/MATH-TTT/acc|val-aux/MATH-TTT/acc|global_step:50" "$task_log" || true
  } > "$METRICS_SNAPSHOT"
  echo "WORKER_V5B_RAY_LOG_SNAPSHOT $RAY_LOG_SNAPSHOT" | tee -a "$LOG"
  echo "WORKER_V5B_METRICS_SNAPSHOT $METRICS_SNAPSHOT" | tee -a "$LOG"
  tail -80 "$METRICS_SNAPSHOT" | tee -a "$LOG"
else
  echo "WORKER_V5B_RAY_LOG_MISSING" | tee -a "$LOG"
fi
echo "WORKER_V5B_EXIT status=${status} $(date '+%F %T')" | tee -a "$LOG"
exit "$status"
