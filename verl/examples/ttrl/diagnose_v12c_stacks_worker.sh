#!/bin/bash
set -u

RAY_DIR=/tmp/ray_v12c/ray/session_latest

echo "STACK_V12C_DATE $(date '+%F %T %Z')"
echo "STACK_V12C_WORKER_PROCS"
ps -eo pid,ppid,stat,etime,wchan:32,cmd | grep -E "ray::WorkerDict|ray::IDLE|verl.trainer.main_ppo" | grep -v grep | head -120 || true

echo "STACK_V12C_RANK_LOG_TAILS"
for f in "$RAY_DIR"/logs/worker-*.out "$RAY_DIR"/logs/worker-*.err; do
  [ -f "$f" ] || continue
  if grep -a -q "user_actor_name:ZN0mUcWorkerDict_0:" "$f"; then
    echo "===== $f ====="
    tail -80 "$f" || true
  fi
done

echo "STACK_V12C_CORE_WORKER_RECENT"
for f in "$RAY_DIR"/logs/python-core-worker-*.log; do
  [ -f "$f" ] || continue
  if grep -a -q "WorkerDict.ref_init_model" "$f"; then
    echo "===== $f ====="
    tail -80 "$f" || true
  fi
done

echo "STACK_V12C_PYSPY"
if command -v py-spy >/dev/null 2>&1; then
  for pid in $(ps -eo pid,cmd | awk '/ray::WorkerDict/ && !/awk/ {print $1}' | head -8); do
    echo "===== py-spy pid=$pid ====="
    timeout 20 py-spy dump --pid "$pid" --native || true
  done
else
  echo "py-spy not found"
fi

echo "STACK_V12C_PROC_STACK"
for pid in $(ps -eo pid,cmd | awk '/ray::WorkerDict/ && !/awk/ {print $1}' | head -8); do
  echo "===== /proc/$pid/status ====="
  grep -E "Name|State|Threads|voluntary|nonvoluntary" "/proc/$pid/status" 2>/dev/null || true
  echo "===== /proc/$pid/wchan ====="
  cat "/proc/$pid/wchan" 2>/dev/null || true
  echo
done
