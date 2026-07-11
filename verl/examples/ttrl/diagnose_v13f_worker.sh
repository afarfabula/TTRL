#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT=/tmp/ray_v13f_refbase_localfp32/ray/session_latest/logs
MAIN_LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_refbase_localfp32_8_50step_v13f.log

echo "DIAG_V13F_DATE $(date '+%F %T %Z')"
echo "DIAG_V13F_HOST $(hostname)"
echo "DIAG_V13F_PROC"
ls -ld /proc /proc/self /proc/meminfo 2>&1 || true
echo "DIAG_V13F_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,pstate --format=csv,noheader,nounits || true
echo "DIAG_V13F_PROCS"
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E 'main_ppo|TaskRunner|WorkerDict|ray::' | grep -v grep | head -120 || true
echo "DIAG_V13F_LOG_FILES"
find "$LOG_ROOT" -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null | sort | head -120 || true
echo "DIAG_V13F_KEYS"
grep -R -n -a -E 'Traceback|Error|Exception|Total training steps|Training from scratch|After init ref from HF AutoModel|After ref FSDP init|After init actor from HF AutoModel|After actor FSDP init|Before building vllm rollout|After building vllm rollout|training/global_step|val-core/MATH-TTT/acc|Final validation metrics' "$LOG_ROOT" "$MAIN_LOG" 2>/dev/null | tail -180 || true
