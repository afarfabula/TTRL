#!/bin/bash
set +e

LOG_DIR=/tmp/ray_sps_conf_mix8/ray/session_latest/logs
echo "SCAN_START $(date '+%F %T')"
echo "LOG_DIR $LOG_DIR"

echo "TRAIN_LOG_STAT"
stat -c "%y %s %n" /opt/tiger/TTRL/verl/sps_conf_mix8_50step.log

echo "GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo "TASK_PROCS"
ps -eo pid,ppid,pcpu,pmem,etime,cmd | grep -E "main_ppo|TaskRunner|ref_init_model|actor_init_model|WorkerDict|vllm|ray::" | grep -v grep | head -n 80

echo "LOG_MATCHES"
grep -R -n -E "Traceback|Error|Exception|CUDA|NCCL|loading|Loaded|ref_init_model|actor_init_model|vllm|FSDP|global_step|val-core|mean@4" "$LOG_DIR" 2>/dev/null | tail -n 160

echo "HOT_LOGS"
find "$LOG_DIR" -maxdepth 1 -type f \( -name "*.out" -o -name "*.err" -o -name "*.log" \) -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" 2>/dev/null | sort | tail -n 30

echo "DRIVER_TAIL"
driver=$(find "$LOG_DIR" -maxdepth 1 -type f -name "python-core-driver*.log" | head -n 1)
if [ -n "$driver" ]; then
  echo "--- $driver"
  tail -n 120 "$driver"
fi

echo "ACTIVE_WORKER_TAILS"
for pid in $(ps -eo pid,cmd | awk '/WorkerDict\.ref_init_model|TaskRunner.run|main_ppo/ && !/awk/ {print $1}' | head -n 12); do
  echo "PID $pid"
  for f in "$LOG_DIR"/*_"$pid".log "$LOG_DIR"/worker-*-"$pid".out "$LOG_DIR"/worker-*-"$pid".err; do
    [ -f "$f" ] || continue
    echo "--- $f"
    tail -n 80 "$f"
  done
done

echo "SCAN_DONE $(date '+%F %T')"
