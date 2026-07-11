#!/bin/bash
set -euo pipefail

echo "DIAG_V13D_DATE $(date '+%F %T %Z')"
echo "DIAG_V13D_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,pstate --format=csv,noheader,nounits || true

echo "DIAG_V13D_PROCS"
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd \
  | grep -E 'ray::TaskRunner|ray::WorkerDict|setup_worker.py|main_ppo|rsync|cp |gcs_server|raylet' \
  | grep -v grep \
  | head -n 140 || true

echo "DIAG_V13D_LOCAL_MODEL"
du -sh /tmp/qwen3_4b_local_v13d /tmp/qwen3_4b_local_v13d.tmp 2>/dev/null || true
find /tmp/qwen3_4b_local_v13d -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null | sort | head -n 40 || true

echo "DIAG_V13D_KEYS"
grep -R -a -n -E \
  'MODEL_LOCAL_COPY|Total training steps|WorkerDict|actor_rollout_init_model|ref_init_model|Before init actor from HF AutoModel|Model config after override|Qwen3ForCausalLM|After init actor from HF AutoModel|After actor FSDP init|Before building vllm rollout|After building vllm rollout|Training from scratch|training/global_step|val-core/MATH-TTT|Traceback|RuntimeError|NCCL version|WORKER_V13D_EXIT' \
  /tmp/ray_v13d_localbf16/ray/session_latest/logs \
  /opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_actorbase_localbf16_8_50step_v13d.log \
  2>/dev/null | tail -n 260 || true

echo "DIAG_V13D_WORKER_OUT"
for f in /tmp/ray_v13d_localbf16/ray/session_latest/logs/worker-*-01000000-*.out; do
  grep -q -a 'WorkerDict\|TaskRunner' "$f" || continue
  echo "===== $f ====="
  tail -n 80 "$f" || true
done

echo "DIAG_V13D_WORKER_ERR"
for f in /tmp/ray_v13d_localbf16/ray/session_latest/logs/worker-*-01000000-*.err; do
  grep -q -a -E 'WorkerDict|Traceback|RuntimeError|Flash Attention|NCCL|Loading checkpoint' "$f" || continue
  echo "===== $f ====="
  tail -n 80 "$f" || true
done
