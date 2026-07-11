#!/bin/bash
set +e

echo "DIAG_START $(date '+%F %T')"
echo "HOST $(hostname)"

echo "PROC"
ls -ld /proc /proc/self /proc/meminfo /proc/cpuinfo

echo "PY_PROCS"
ps -ef | grep -E "python|ray|vllm|verl" | grep -v grep

echo "GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo "LOG_STAT"
stat -c "%y %s %n" /opt/tiger/TTRL/verl/sps_conf_mix8_50step.log

echo "LOG_TAIL"
tail -n 120 /opt/tiger/TTRL/verl/sps_conf_mix8_50step.log

echo "RAY_TMP"
find /tmp/ray_sps_conf_mix8 -maxdepth 3 -type f -name "*.log" -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" 2>/dev/null | sort | tail -n 40

echo "RAY_ALL_LOGS"
find /tmp/ray_sps_conf_mix8 -maxdepth 5 -type f \( -name "*.out" -o -name "*.err" -o -name "*.log" \) -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" 2>/dev/null | sort | tail -n 80

echo "RECENT_RAY_TAILS"
for f in $(find /tmp/ray_sps_conf_mix8 -maxdepth 5 -type f \( -name "worker*.out" -o -name "worker*.err" -o -name "python-core-worker*.log" -o -name "raylet.err" -o -name "raylet.out" \) 2>/dev/null | sort | tail -n 12); do
  echo "--- $f"
  tail -n 40 "$f"
done

echo "DIAG_DONE $(date '+%F %T')"
