#!/bin/bash
set -u

RAY_DIR=/tmp/ray_v12e/ray/session_latest
LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_50step_v12e.log

echo "DIAG_V12E_DATE $(date '+%F %T %Z')"
echo "DIAG_V12E_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits || true

echo "DIAG_V12E_PROCS"
ps -eo pid,ppid,stat,etime,cmd | grep -E "verl.trainer.main_ppo|ray|TaskRunner|WorkerDict|python -m" | grep -v grep | head -160 || true

echo "DIAG_V12E_MAIN_KEY_LINES"
grep -a -n -E "WORKER_V12E_EXIT|Traceback|TimeoutError|Failed to get register_center|Training from scratch|Total training steps|training/global_step|val-core/MATH-TTT|After actor FSDP|After ref FSDP|Before building|After building" \
  "$LOG" 2>/dev/null | tail -160 || true

echo "DIAG_V12E_RAY_KEY_LINES"
grep -R -n -a -E "TimeoutError|Failed to get register_center|Waiting for register center|Traceback|RuntimeError|ValueError|ActorDied|OutOfMemory|WorkerDict|register_center|Training from scratch|Total training steps|training/global_step|val-core/MATH-TTT|NCCL|Gloo|After actor FSDP|After ref FSDP|Before building|After building" \
  "$RAY_DIR/logs" 2>/dev/null | tail -320 || true

echo "DIAG_V12E_TASKRUNNER_TAIL"
task_log=$(grep -R -l -a ":actor_name:TaskRunner" "$RAY_DIR/logs"/worker-*.out 2>/dev/null | head -1 || true)
task_err=$(grep -R -l -a ":actor_name:TaskRunner" "$RAY_DIR/logs"/worker-*.err 2>/dev/null | head -1 || true)
echo "TASK_LOG $task_log"
echo "TASK_ERR $task_err"
if [ -n "$task_log" ]; then
  tail -260 "$task_log" 2>/dev/null || true
fi
if [ -n "$task_err" ]; then
  tail -180 "$task_err" 2>/dev/null || true
fi

echo "DIAG_V12E_RANK_TAILS"
for f in "$RAY_DIR"/logs/worker-*.out "$RAY_DIR"/logs/worker-*.err; do
  [ -f "$f" ] || continue
  if grep -a -q "user_actor_name:.*WorkerDict" "$f"; then
    echo "===== $f ====="
    tail -100 "$f" || true
  fi
done
