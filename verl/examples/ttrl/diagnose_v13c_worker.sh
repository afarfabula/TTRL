#!/bin/bash
set -euo pipefail

echo "DIAG_V13C_DATE $(date '+%F %T %Z')"
echo "DIAG_V13C_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits || true

echo "DIAG_V13C_PROCS"
ps -eo pid,ppid,state,etime,cmd | grep -E 'verl.trainer.main_ppo|WorkerDict|ray::|v13c_actorbase' | grep -v grep || true

echo "DIAG_V13C_LOCAL_KEY"
grep -a -E "Started a local Ray instance|TaskRunner|Waiting for register|Total training steps|Training from scratch|training/global_step|After actor FSDP|Before building|After building|Traceback|RuntimeError|WORKER_V13C_EXIT|val-core/MATH-TTT" \
  /opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_actorbase_8_50step_v13c.log 2>/dev/null || true

echo "DIAG_V13C_RAY_KEY"
grep -R -a -E "TaskRunner|WorkerDict|Total training steps|Training from scratch|training/global_step|After actor FSDP|Before building|After building|NCCL version|Qwen3ForCausalLM|Traceback|RuntimeError|val-core/MATH-TTT" \
  /tmp/ray_v13c_actorbase/ray/session_latest/logs 2>/dev/null | tail -120 || true

echo "DIAG_V13C_RANK_TAILS"
find /tmp/ray_v13c_actorbase/ray/session_latest/logs -maxdepth 1 -type f \( -name 'worker-*.out' -o -name 'worker-*.err' \) 2>/dev/null \
  | sort | tail -20 | while read -r f; do
    echo "===== $f ====="
    tail -40 "$f" || true
  done
