#!/bin/bash
set -u

RAY_DIR=/tmp/ray_v12c/ray/session_latest
LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_50step_v12c.log

echo "NARROW_V12C_DATE $(date '+%F %T %Z')"
echo "NARROW_V12C_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits || true

echo "NARROW_V12C_MAIN_KEY_LINES"
grep -a -n -E "WORKER_V12C_EXIT|Traceback|TimeoutError|Failed to get register_center|Training from scratch|training/global_step|val-core/MATH-TTT" \
  "$LOG" 2>/dev/null | tail -80 || true

echo "NARROW_V12C_RAY_KEY_LINES"
grep -R -n -a -E "TimeoutError|Failed to get register_center|Waiting for register center|Traceback|RuntimeError|ValueError|ActorDied|OutOfMemory|WorkerDict|register_center|training/global_step|val-core/MATH-TTT" \
  "$RAY_DIR/logs" 2>/dev/null | tail -220 || true

echo "NARROW_V12C_ACTOR_STATE"
/opt/tiger/modelchef/.venv/bin/python - <<'PY' || true
import ray

ray.init(address="auto", ignore_reinit_error=True)

try:
    print("namespace", ray.get_runtime_context().namespace)
except Exception as exc:
    print("namespace_error", type(exc).__name__, exc)

try:
    from ray.util import list_named_actors

    print("named_default", list_named_actors())
    print("named_all", list_named_actors(all_namespaces=True))
except Exception as exc:
    print("list_named_actors_error", type(exc).__name__, exc)

for name in ["ZN0mUc_register_center"]:
    try:
        actor = ray.get_actor(name)
        print("get_actor_ok", name, actor)
        print("rank_zero_info", ray.get(actor.get_rank_zero_info.remote(), timeout=5))
        print("worker_info", ray.get(actor.get_worker_info.remote(), timeout=5))
    except Exception as exc:
        print("get_actor_fail", name, type(exc).__name__, exc)
PY
