#!/bin/bash
set -u

RAY_DIR=/tmp/ray_v12b/ray/session_latest
LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_50step_v12b.log

echo "DIAG_V12B_DATE $(date '+%F %T %Z')"
echo "DIAG_V12B_PROC"
ls -ld /proc /proc/self /proc/meminfo || true

echo "DIAG_V12B_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits || true

echo "DIAG_V12B_PROCS"
ps -eo pid,ppid,stat,etime,cmd | grep -E "verl.trainer.main_ppo|ray|TaskRunner|WorkerDict|python -m" | grep -v grep | head -180 || true

echo "DIAG_V12B_MAIN_LOG"
ls -l "$LOG" 2>/dev/null || true
tail -120 "$LOG" 2>/dev/null || true

echo "DIAG_V12B_RAY_LOGS"
find "$RAY_DIR/logs" -maxdepth 1 -type f \( -name "*.out" -o -name "*.err" \) \
  -printf "%p %s %TY-%Tm-%Td %TH:%TM:%TS\n" 2>/dev/null | sort | tail -100 || true

echo "DIAG_V12B_ERRORS"
grep -R -n -a -E "Traceback|RuntimeError|ValueError|Exception|Error|failed|timeout|TimedOut|TaskRunner|WorkerDict|Scheduling|GCS" \
  "$RAY_DIR/logs" "$LOG" 2>/dev/null | tail -200 || true

echo "DIAG_V12B_WORKER_LOG_CONTENT"
find "$RAY_DIR/logs" -maxdepth 1 -type f \( -name "worker-*-01000000-*.out" -o -name "worker-*-01000000-*.err" \) -size +0c \
  -print 2>/dev/null | sort | tail -16 | while read -r f; do
    echo "===== $f"
    sed -n '1,120p' "$f" 2>/dev/null || true
  done

echo "DIAG_V12B_TASKRUNNER_TAIL"
task_log=$(grep -R -l -a ":actor_name:TaskRunner" "$RAY_DIR/logs"/worker-*.out 2>/dev/null | head -1 || true)
task_err=$(grep -R -l -a ":actor_name:TaskRunner" "$RAY_DIR/logs"/worker-*.err 2>/dev/null | head -1 || true)
echo "TASK_LOG $task_log"
echo "TASK_ERR $task_err"
if [ -n "$task_log" ]; then
  tail -240 "$task_log" 2>/dev/null || true
fi
if [ -n "$task_err" ]; then
  tail -160 "$task_err" 2>/dev/null || true
fi

echo "DIAG_V12B_RANK0_TAIL"
rank0_log=$(grep -R -l -a "WorkerDict_0:0" "$RAY_DIR/logs"/worker-*.out 2>/dev/null | head -1 || true)
rank0_err=$(grep -R -l -a "WorkerDict_0:0" "$RAY_DIR/logs"/worker-*.err 2>/dev/null | head -1 || true)
echo "RANK0_LOG $rank0_log"
echo "RANK0_ERR $rank0_err"
if [ -n "$rank0_log" ]; then
  tail -180 "$rank0_log" 2>/dev/null || true
fi
if [ -n "$rank0_err" ]; then
  tail -120 "$rank0_err" 2>/dev/null || true
fi

echo "DIAG_V12B_RAY_STACK"
/opt/tiger/modelchef/.venv/bin/ray stack 2>&1 | head -260 || true
