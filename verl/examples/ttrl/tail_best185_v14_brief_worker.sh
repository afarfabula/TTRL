#!/bin/bash
set -euo pipefail

LOG_ROOT=/tmp/r185_v14/ray/session_latest/logs
MAIN_LOG=/opt/tiger/TTRL/verl/sps_rule_conf_weight_floor015_clip05_power15_refbase_localfp32_8_185step_best_v14.log

extract_metric() {
  local line=$1
  local key=$2
  printf '%s\n' "$line" | awk -v key="${key}:" '
    {
      n = split($0, parts, key)
      if (n > 1) {
        split(parts[2], value, " ")
        print value[1]
      }
    }'
}

latest_train=$(grep -R -h -a 'training/global_step' "$LOG_ROOT" "$MAIN_LOG" 2>/dev/null | tail -1 || true)
latest_val=$(grep -R -h -a -e 'val-core/MATH-TTT/acc' -e 'Final validation metrics' "$LOG_ROOT" "$MAIN_LOG" 2>/dev/null | tail -20 || true)

if [ -n "$latest_train" ]; then
  printf 'step=%s train_weight=%s ground_truth_reward=%s label_accuracy=%s step_s=%s throughput=%s\n' \
    "$(extract_metric "$latest_train" 'training/global_step')" \
    "$(extract_metric "$latest_train" 'train/sps/train_weight')" \
    "$(extract_metric "$latest_train" 'train/ground_truth_reward')" \
    "$(extract_metric "$latest_train" 'train/label_accuracy')" \
    "$(extract_metric "$latest_train" 'timing_s/step')" \
    "$(extract_metric "$latest_train" 'perf/throughput')"
else
  echo "no training step yet"
fi

if [ -n "$latest_val" ]; then
  printf '%s\n' "$latest_val"
fi
