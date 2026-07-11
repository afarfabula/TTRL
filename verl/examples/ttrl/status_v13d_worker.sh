#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT=/tmp/ray_v13d_localbf16/ray/session_latest/logs
TASK_LOG="$LOG_ROOT/worker-7ebd60f12d30728214b0468bfc898902adedcf21adf9cd3b85e53c27-01000000-33589.out"

echo "STATUS_V13D_DATE $(date '+%F %T %Z')"
echo "STATUS_V13D_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,pstate --format=csv,noheader,nounits || true
echo "STATUS_V13D_PROCS"
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E 'main_ppo|TaskRunner|WorkerDict|ray::' | grep -v grep | head -80 || true
echo "STATUS_V13D_TASK_TAIL"
tail -120 "$TASK_LOG" || true
echo "STATUS_V13D_RECENT_ERRS"
find "$LOG_ROOT" -maxdepth 1 -name '*.err' -mmin -15 -type f -print -exec tail -30 {} \; || true
