#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT=/tmp/ray_v14_power15_refbase_localfp32/ray/session_latest/logs
MAIN_LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_refbase_localfp32_8_50step_v14.log

echo "DIAG_V14_DATE $(date '+%F %T %Z')"
hostname || true
ls -ld /proc /proc/self /proc/meminfo || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits || true
ps -eo pid,ppid,pcpu,pmem,etime,cmd | grep -E 'main_ppo|TaskRunner|WorkerDict|ray::|vllm' | grep -v grep | head -80 || true
echo "LOG_TAIL_BEGIN"
grep -R -h -a -E 'TaskRunner|Total training steps|Training from scratch|training/global_step|WorkerDict|Traceback|RuntimeError|val-core/MATH-TTT|Qwen3ForCausalLM|After actor FSDP|Before building|After building|NCCL version' "$LOG_ROOT" "$MAIN_LOG" 2>/dev/null | tail -160 || true
echo "LOG_TAIL_END"
