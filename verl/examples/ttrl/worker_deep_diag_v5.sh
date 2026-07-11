#!/bin/bash
set +e

LOG_DIR=/tmp/ray_sps_conf_mix8/ray/session_latest/logs
echo "DEEP_DIAG_START $(date '+%F %T')"

echo "BASIC"
hostname
stat -c "%y %s %n" /opt/tiger/TTRL/verl/sps_conf_mix8_50step.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,pstate --format=csv,noheader

echo "PROCESS_TREE"
ps -eo pid,ppid,stat,pcpu,pmem,etime,wchan:32,cmd | grep -E "main_ppo|WorkerDict|TaskRunner|raylet|gcs_server|vllm|python -m verl" | grep -v grep

echo "BLOCKED_PROC_DETAILS"
for pid in $(ps -eo pid,cmd | awk '/ray::WorkerDict.ref_init_model/ {print $1}'); do
  echo "--- pid $pid"
  ps -p "$pid" -o pid,ppid,stat,pcpu,pmem,etime,wchan:40,cmd
  echo "wchan=$(cat /proc/$pid/wchan 2>/dev/null)"
  echo "io"
  cat /proc/$pid/io 2>/dev/null | sed -n '1,20p'
  echo "stack"
  cat /proc/$pid/stack 2>/dev/null | sed -n '1,40p'
  echo "threads"
  ps -L -p "$pid" -o pid,tid,stat,pcpu,wchan:32,comm | sed -n '1,80p'
  echo "fd-sample"
  ls -l /proc/$pid/fd 2>/dev/null | sed -n '1,40p'
done

echo "RAY_TASKS"
/opt/tiger/modelchef/.venv/bin/python - <<'PY'
import json
try:
    import ray
    ray.init(address="auto", ignore_reinit_error=True)
    from ray.util.state import list_tasks, list_actors
    tasks = list_tasks(limit=40, detail=True)
    print("tasks", len(tasks))
    for t in tasks[:40]:
        print("TASK", getattr(t, "task_id", None), getattr(t, "name", None), getattr(t, "state", None), getattr(t, "func_or_class_name", None))
    actors = list_actors(limit=40, detail=True)
    print("actors", len(actors))
    for a in actors[:40]:
        print("ACTOR", getattr(a, "actor_id", None), getattr(a, "name", None), getattr(a, "state", None), getattr(a, "class_name", None))
except Exception as e:
    print("RAY_STATE_ERROR", type(e).__name__, e)
PY

echo "RANK_WORKER_LOG_TAILS"
for f in "$LOG_DIR"/worker-*-01000000-*.out "$LOG_DIR"/worker-*-01000000-*.err; do
  [ -f "$f" ] || continue
  if grep -q "WorkerDict_0:" "$f"; then
    echo "--- $f"
    tail -n 120 "$f"
  fi
done

echo "DRIVER_LOG_TAIL"
for f in "$LOG_DIR"/python-core-driver*.log "$LOG_DIR"/worker-*-01000000-33084.out "$LOG_DIR"/worker-*-01000000-33084.err; do
  [ -f "$f" ] || continue
  echo "--- $f"
  tail -n 160 "$f"
done

echo "PY_STACK_TOOL_CHECK"
command -v py-spy || true
command -v gdb || true
command -v pstack || true

echo "DEEP_DIAG_DONE $(date '+%F %T')"
