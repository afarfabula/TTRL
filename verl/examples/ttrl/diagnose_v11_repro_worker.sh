#!/bin/bash
set -euo pipefail

RAY_DIR=/tmp/ray_v11_repro/ray/session_latest
LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_8_50step_v11_repro.log

echo "DIAG_V11_REPRO_DATE $(date '+%F %T %Z')"
echo "DIAG_V11_REPRO_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits || true

echo "DIAG_V11_REPRO_PROCS"
ps -eo pid,ppid,stat,etime,cmd | grep -E "verl.trainer.main_ppo|ray::TaskRunner|ray::WorkerDict|raylet|gcs_server" | grep -v grep | head -160 || true

echo "DIAG_V11_REPRO_LOCAL_KEY"
grep -a -E "Started a local Ray instance|TaskRunner|Waiting for register|Training from scratch|Total training steps|training/global_step|Traceback|RuntimeError|WORKER_V11_REPRO_EXIT|val-core/MATH-TTT|After actor FSDP|Before building|After building|NCCL version|Qwen3ForCausalLM" "$LOG" 2>/dev/null | tail -120 || true

echo "DIAG_V11_REPRO_RAY_KEY"
grep -R -a -E "TaskRunner|Waiting for register|Training from scratch|Total training steps|training/global_step|Traceback|RuntimeError|val-core/MATH-TTT|After actor FSDP|Before building|After building|NCCL version|Qwen3ForCausalLM|Gloo|WorkerDict" "$RAY_DIR/logs" 2>/dev/null | tail -220 || true

echo "DIAG_V11_REPRO_RANK_TAILS"
for f in $(find "$RAY_DIR/logs" -maxdepth 1 -type f \( -name "worker-*.out" -o -name "worker-*.err" \) 2>/dev/null | sort | tail -20); do
  echo "===== $f ====="
  tail -60 "$f" || true
done
