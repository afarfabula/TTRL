#!/bin/bash
set -u

echo "DIAG_DATE $(date '+%F %T %Z')"
echo "DIAG_PROC"
ls -ld /proc /proc/self /proc/meminfo || true

echo "DIAG_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits || true

echo "DIAG_PROCS"
ps -eo pid,ppid,stat,etime,cmd | grep -E "verl.trainer.main_ppo|ray|TaskRunner|python -m" | grep -v grep | head -120 || true

echo "DIAG_RAY_LOGS"
find /tmp/ray_v12 -maxdepth 5 -type f \( -name "*.out" -o -name "*.err" \) -printf "%p %s %TY-%Tm-%Td %TH:%TM:%TS\n" 2>/dev/null | sort | tail -80 || true

echo "DIAG_RUN_LOG"
ls -l /opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_50step.log /opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_metrics.txt 2>/dev/null || true
tail -120 /opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_8_50step.log 2>/dev/null || true
