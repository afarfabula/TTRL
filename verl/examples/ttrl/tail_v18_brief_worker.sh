#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT=/tmp/r18/ray/session_latest/logs
MAIN_LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power125_refbase_localfp32_8_50step_v18.log

extract_metric() {
  local line=$1
  local key=$2
  printf '%s\n' "$line" | tr ' ' '\n' | grep -F -m1 "${key}:" | cut -d: -f2
}

latest_train=$(
  grep -R -h -a 'training/global_step' "$LOG_ROOT" "$MAIN_LOG" 2>/dev/null | tail -1 || true
)
latest_val=$(
  grep -R -h -a -e 'val-core/MATH-TTT/acc' -e 'Final validation metrics' "$LOG_ROOT" "$MAIN_LOG" 2>/dev/null | tail -20 || true
)

echo "TAIL_V18_BRIEF_DATE $(date '+%F %T %Z')"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd ';' - || true

if [[ -n "$latest_train" ]]; then
  printf 'STEP=%s TRAIN_WEIGHT=%s GT_REWARD=%s LABEL_ACC=%s REF_S=%s STEP_S=%s THROUGHPUT=%s\n' \
    "$(extract_metric "$latest_train" 'training/global_step')" \
    "$(extract_metric "$latest_train" 'train/sps/train_weight')" \
    "$(extract_metric "$latest_train" 'train/ground_truth_reward')" \
    "$(extract_metric "$latest_train" 'train/label_accuracy')" \
    "$(extract_metric "$latest_train" 'timing_s/ref')" \
    "$(extract_metric "$latest_train" 'timing_s/step')" \
    "$(extract_metric "$latest_train" 'perf/throughput')"
else
  echo "STEP=NA"
fi

if [[ -n "$latest_val" ]]; then
  echo "VAL_LINES_BEGIN"
  printf '%s\n' "$latest_val"
  echo "VAL_LINES_END"
else
  echo "VAL_LINES=NA"
fi
