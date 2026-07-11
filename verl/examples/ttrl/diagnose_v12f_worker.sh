#!/bin/bash
set -euo pipefail

RAY_DIR=/tmp/ray_v12f/ray/session_latest
LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_50step_v12f.log

echo "DIAG_V12F_DATE $(date '+%F %T %Z')"
echo "DIAG_V12F_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits || true

echo "DIAG_V12F_PROCS"
ps -eo pid,ppid,stat,etime,cmd | grep -E "verl.trainer.main_ppo|ray::TaskRunner|ray::WorkerDict|raylet|gcs_server" | grep -v grep | head -160 || true

echo "DIAG_V12F_LOCAL_KEY"
grep -a -E "Started a local Ray instance|TaskRunner|Waiting for register|Training from scratch|Total training steps|training/global_step|Traceback|RuntimeError|WORKER_V12F_EXIT|val-core/MATH-TTT|After actor FSDP|Before building|After building|NCCL version|Qwen3ForCausalLM" "$LOG" 2>/dev/null | tail -120 || true

echo "DIAG_V12F_RAY_FILES"
find "$RAY_DIR/logs" -maxdepth 1 -type f \( -name "*.out" -o -name "*.err" \) -printf "%p %s %TY-%Tm-%Td %TH:%TM:%TS\n" 2>/dev/null | sort | tail -80 || true

echo "DIAG_V12F_RAY_KEY"
grep -R -a -E "TaskRunner|Waiting for register|Training from scratch|Total training steps|training/global_step|Traceback|RuntimeError|val-core/MATH-TTT|After actor FSDP|Before building|After building|NCCL version|Qwen3ForCausalLM|Gloo|WorkerDict" "$RAY_DIR/logs" 2>/dev/null | tail -220 || true

echo "DIAG_V12F_RANK_TAILS"
for f in $(find "$RAY_DIR/logs" -maxdepth 1 -type f \( -name "worker-*.out" -o -name "worker-*.err" \) 2>/dev/null | sort | tail -20); do
  echo "===== $f ====="
  tail -60 "$f" || true
done
