#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT=/tmp/ray_v13e_refbase_localbf16/ray/session_latest/logs
MAIN_LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_refbase_localbf16_8_50step_v13e.log
METRICS_OUT=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_refbase_localbf16_8_v13e_metrics.txt

echo "TAIL_V13E_DATE $(date '+%F %T %Z')"
echo "TAIL_V13E_GPU"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,pstate --format=csv,noheader,nounits || true

tmp_file=$(mktemp)
trap 'rm -f "$tmp_file"' EXIT

grep -R -h -a \
  -e 'training/global_step' \
  -e 'val-core/MATH-TTT/acc' \
  -e 'Final validation metrics' \
  -e 'Total training steps' \
  "$LOG_ROOT" "$MAIN_LOG" 2>/dev/null >"$tmp_file" || true

{
  echo "TAIL_V13E_LATEST_TRAIN"
  grep 'training/global_step' "$tmp_file" | tail -1 || true
  echo "TAIL_V13E_VAL"
  grep -e 'val-core/MATH-TTT/acc' -e 'Final validation metrics' "$tmp_file" | tail -30 || true
} | tee "$METRICS_OUT"
