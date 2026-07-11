#!/bin/bash
set -euo pipefail

LOG=/opt/tiger/TTRL/verl/sps_conf_mix8_50step.log
echo "WORKER_V5_START $(date '+%F %T')" | tee "$LOG"
echo "HOST $(hostname)" | tee -a "$LOG"
ls -ld /proc /proc/self /proc/meminfo | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/tmp/ray_sps_conf_mix8
export MASTER_PORT=29545

bash /opt/tiger/TTRL/verl/examples/ttrl/run_sps_conf_weight_majority_mix_math_qwen3_4b_50step_8gpu.sh 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
echo "WORKER_V5_EXIT status=${status} $(date '+%F %T')" | tee -a "$LOG"
exit "$status"
