#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT=/tmp/ray_v13d_localbf16/ray/session_latest/logs
MAIN_LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_actorbase_localbf16_8_50step_v13d.log

echo "TAIL_V13D_DATE $(date '+%F %T %Z')"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,pstate --format=csv,noheader,nounits || true

tmp_file=$(mktemp)
trap 'rm -f "$tmp_file"' EXIT

grep -R -h -e 'training/global_step' -e 'val-core/MATH-TTT/acc' "$LOG_ROOT" "$MAIN_LOG" 2>/dev/null >"$tmp_file" || true

echo "TAIL_V13D_LATEST_TRAIN"
grep 'training/global_step' "$tmp_file" | tail -1 || true

echo "TAIL_V13D_VAL"
grep 'val-core/MATH-TTT/acc' "$tmp_file" | tail -20 || true
