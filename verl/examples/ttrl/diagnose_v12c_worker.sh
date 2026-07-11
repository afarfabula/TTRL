#!/bin/bash
set -u

RAY_DIR=/tmp/ray_v12c/ray/session_latest
LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_50step_v12c.log

echo "DIAG_V12C_DATE $(date '+%F %T %Z')"
echo "DIAG_V12C_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits || true

echo "DIAG_V12C_PROCS"
ps -eo pid,ppid,stat,etime,cmd | grep -E "verl.trainer.main_ppo|ray|TaskRunner|WorkerDict|python -m" | grep -v grep | head -160 || true

echo "DIAG_V12C_MAIN_LOG"
ls -l "$LOG" 2>/dev/null || true
tail -100 "$LOG" 2>/dev/null || true

echo "DIAG_V12C_RAY_LOGS"
find "$RAY_DIR/logs" -maxdepth 1 -type f \( -name "*.out" -o -name "*.err" \) \
  -printf "%p %s %TY-%Tm-%Td %TH:%TM:%TS\n" 2>/dev/null | sort | tail -100 || true

echo "DIAG_V12C_ERRORS"
grep -R -n -a -E "Traceback|RuntimeError|ValueError|Exception|Error|failed|timeout|TimedOut|TaskRunner|WorkerDict|Waiting for register|Training from scratch|training/global_step" \
  "$RAY_DIR/logs" "$LOG" 2>/dev/null | tail -200 || true

echo "DIAG_V12C_TASKRUNNER_TAIL"
task_log=$(grep -R -l -a ":actor_name:TaskRunner" "$RAY_DIR/logs"/worker-*.out 2>/dev/null | head -1 || true)
task_err=$(grep -R -l -a ":actor_name:TaskRunner" "$RAY_DIR/logs"/worker-*.err 2>/dev/null | head -1 || true)
echo "TASK_LOG $task_log"
echo "TASK_ERR $task_err"
if [ -n "$task_log" ]; then
  tail -220 "$task_log" 2>/dev/null || true
fi
if [ -n "$task_err" ]; then
  tail -160 "$task_err" 2>/dev/null || true
fi

