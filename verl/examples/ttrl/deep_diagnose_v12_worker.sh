#!/bin/bash
set -u

RAY_DIR=/tmp/ray_v12/ray/session_latest
LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_50step.log

echo "DEEP_DIAG_DATE $(date '+%F %T %Z')"
echo "DEEP_DIAG_PROC"
ls -ld /proc /proc/self /proc/meminfo || true

echo "DEEP_DIAG_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true

echo "DEEP_DIAG_MAIN_LOG"
ls -l "$LOG" 2>/dev/null || true
tail -80 "$LOG" 2>/dev/null || true

echo "DEEP_DIAG_MAIN_PY_STACK"
main_pid=$(pgrep -f "verl.trainer.main_ppo" | head -1 || true)
if [ -n "$main_pid" ]; then
  echo "MAIN_PID $main_pid"
  py-spy dump --pid "$main_pid" --native --nonblocking 2>&1 | head -240 || true
else
  echo "MAIN_PID_MISSING"
fi

echo "DEEP_DIAG_RAY_TASKS"
RAY_ADDRESS=127.0.0.1:8265 /opt/tiger/modelchef/.venv/bin/ray list tasks --detail 2>&1 | head -240 || true

echo "DEEP_DIAG_RAY_ACTORS"
RAY_ADDRESS=127.0.0.1:8265 /opt/tiger/modelchef/.venv/bin/ray list actors --detail 2>&1 | head -240 || true

echo "DEEP_DIAG_ERROR_GREP"
grep -R -n -a -E "Traceback|RuntimeError|ValueError|Exception|NCCL|CUDA|Killed|OutOfMemory|timeout" \
  "$RAY_DIR/logs" "$LOG" 2>/dev/null | tail -200 || true

echo "DEEP_DIAG_NONEMPTY_RAY_LOGS"
find "$RAY_DIR/logs" -maxdepth 1 -type f \( -name "*.out" -o -name "*.err" \) -size +0c \
  -printf "%p %s %TY-%Tm-%Td %TH:%TM:%TS\n" 2>/dev/null | sort | tail -40 || true

echo "DEEP_DIAG_RAY_LOG_CONTENT"
find "$RAY_DIR/logs" -maxdepth 1 -type f \( -name "worker-*-01000000-*.out" -o -name "worker-*-01000000-*.err" \) -size +0c \
  -print 2>/dev/null | sort | tail -12 | while read -r f; do
    echo "===== $f"
    sed -n '1,160p' "$f" 2>/dev/null || true
  done
