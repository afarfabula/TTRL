#!/usr/bin/env bash
set -euo pipefail

echo "COPY_DIAG_V13E_DATE $(date '+%F %T %Z')"
echo "COPY_DIAG_V13E_PROCS"
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E 'rsync|cp -a|worker_run|qwen3_4b|main_ppo|python|bash' | grep -v grep || true
echo "COPY_DIAG_V13E_MODEL_DIRS"
ls -ld /tmp/qwen3_4b_local_v13e /tmp/qwen3_4b_local_v13e.tmp 2>&1 || true
du -sh /tmp/qwen3_4b_local_v13e.tmp /tmp/qwen3_4b_local_v13e 2>&1 || true
echo "COPY_DIAG_V13E_LOCAL_TMP_FILES"
find /tmp/qwen3_4b_local_v13e.tmp -maxdepth 1 -type f -printf '%f %s\n' 2>&1 | sort || true
echo "COPY_DIAG_V13E_HDFS_LS"
timeout 15 ls -lh /mnt/hdfs/models/qwen3_4b 2>&1 | head -40 || true
