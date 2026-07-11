#!/bin/bash
set -euo pipefail

echo "DIAG_V13C_NARROW_DATE $(date '+%F %T %Z')"
echo "DIAG_V13C_NARROW_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,pstate --format=csv,noheader,nounits || true

echo "DIAG_V13C_NARROW_PROCS"
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd \
  | grep -E 'ray::TaskRunner|ray::WorkerDict|setup_worker.py|main_ppo|gcs_server|raylet' \
  | grep -v grep \
  | head -n 120 || true

echo "DIAG_V13C_NARROW_KEYS"
grep -R -a -n -E \
  'Total training steps|WorkerDict|actor_rollout_init_model|ref_init_model|Before init actor from HF AutoModel|Model config after override|Qwen3ForCausalLM|After init actor from HF AutoModel|After actor FSDP init|Before building vllm rollout|After building vllm rollout|Training from scratch|training/global_step|val-core/MATH-TTT|Traceback|RuntimeError|NCCL version|WORKER_V13C_EXIT' \
  /tmp/ray_v13c_actorbase/ray/session_latest/logs \
  /opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_actorbase_8_50step_v13c.log \
  2>/dev/null | tail -n 240 || true

echo "DIAG_V13C_NARROW_WORKER_OUT"
for f in /tmp/ray_v13c_actorbase/ray/session_latest/logs/worker-*-01000000-*.out; do
  grep -q -a 'WorkerDict' "$f" || continue
  echo "===== $f ====="
  tail -n 80 "$f" || true
done

echo "DIAG_V13C_NARROW_WORKER_ERR"
for f in /tmp/ray_v13c_actorbase/ray/session_latest/logs/worker-*-01000000-*.err; do
  grep -q -a -E 'WorkerDict|Traceback|RuntimeError|Flash Attention|NCCL' "$f" || continue
  echo "===== $f ====="
  tail -n 80 "$f" || true
done
