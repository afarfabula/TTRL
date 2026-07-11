#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT=/tmp/ray_v13e_refbase_localbf16/ray/session_latest/logs
MAIN_LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_refbase_localbf16_8_50step_v13e.log

echo "DIAG_V13E_DATE $(date '+%F %T %Z')"
echo "DIAG_V13E_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,pstate --format=csv,noheader,nounits || true
echo "DIAG_V13E_PROCS"
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E 'main_ppo|TaskRunner|WorkerDict|ray::' | grep -v grep | head -80 || true
echo "DIAG_V13E_LOCAL_MODEL"
du -sh /tmp/qwen3_4b_local_v13e 2>/dev/null || true
find /tmp/qwen3_4b_local_v13e -maxdepth 1 -type f -printf '%f %s\n' 2>/dev/null | sort || true
echo "DIAG_V13E_KEYS"
grep -R -n -a -E 'Traceback|Error|Exception|Total training steps|Training from scratch|After init ref from HF AutoModel|After ref FSDP init|After init actor from HF AutoModel|After actor FSDP init|Before building vllm rollout|After building vllm rollout|training/global_step|val-core/MATH-TTT/acc|Final validation metrics' "$LOG_ROOT" "$MAIN_LOG" 2>/dev/null | tail -120 || true
