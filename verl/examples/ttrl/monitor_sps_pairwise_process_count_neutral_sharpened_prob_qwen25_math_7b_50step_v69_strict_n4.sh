#!/usr/bin/env bash
set -euo pipefail

EXP=sps_pairwise_process_count_neutral_sharpened_prob_qwen25_math_7b_50step_v69_strict_n4
LOG=/opt/tiger/TTRL/verl/${EXP}.log
LOG_ROOT=/tmp/r69/ray/session_latest/logs
PROC_HEALTH=/opt/tiger/TTRL/verl/${EXP}_proc_health.txt

extract_metric() {
  local line=$1
  local key=$2
  printf '%s\n' "$line" | tr ' ' '\n' | grep -F -m1 "${key}:" | cut -d: -f2 || true
}

latest_train=$(
  grep -R -h -a 'training/global_step' "$LOG_ROOT" "$LOG" 2>/dev/null | tail -1 || true
)
latest_val=$(
  grep -R -h -a -e 'val-core/MATH-TTT/acc' -e 'val-aux/MATH-TTT' -e 'Final validation metrics' "$LOG_ROOT" "$LOG" 2>/dev/null | tail -40 || true
)
latest_runner_status=$(
  grep -h -a -e 'WORKER_QWEN25_MATH_7B_V69_PAIRWISE_PROCESS_COUNT_NEUTRAL_SHARPENED_STRICT_N4_' "$LOG" "$PROC_HEALTH" 2>/dev/null | tail -20 || true
)

echo "MONITOR_V69_READ_ONLY_DATE $(date '+%F %T %Z')"
echo "RUN_FROM_SEPARATE_WORKER_LOGIN_TERMINAL=1"
echo "DO_NOT_CTRL_C_TRAINING_FOREGROUND_TERMINAL=1"

if [[ -e /proc/self && -e /proc/meminfo ]]; then
  proc_count=$(python3 - <<'PY' 2>/dev/null || true
import os
print(len(os.listdir("/proc")))
PY
)
  echo "PROC_STATUS=OK PROC_COUNT=${proc_count:-NA}"
else
  echo "PROC_STATUS=BAD"
fi

nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd ';' - || true

if [[ -n "$latest_train" ]]; then
  printf 'STEP=%s MODE=%s COUNT_NEUTRAL=%s PAIRWISE_STD=%s PAIRWISE_TOP=%s TARGET_EK=%s TARGET_CONF=%s MAJ_MASS=%s BASE_AGREE=%s PICK_ACC=%s CORRECT_MASS=%s CLIP=%s STEP_S=%s TOKENS=%s THROUGHPUT=%s\n' \
    "$(extract_metric "$latest_train" 'training/global_step')" \
    "$(extract_metric "$latest_train" 'train/sps/reward_mode')" \
    "$(extract_metric "$latest_train" 'train/sps/direct_count_neutral_aggregation')" \
    "$(extract_metric "$latest_train" 'train/sps/direct_pairwise_process_preference_std')" \
    "$(extract_metric "$latest_train" 'train/sps/direct_pairwise_process_top_preference')" \
    "$(extract_metric "$latest_train" 'train/sps/direct_target_effective_K')" \
    "$(extract_metric "$latest_train" 'train/sps/direct_target_confidence')" \
    "$(extract_metric "$latest_train" 'train/sps/direct_majority_target_mass')" \
    "$(extract_metric "$latest_train" 'train/sps/direct_base_agreement')" \
    "$(extract_metric "$latest_train" 'train/sps/pick_accuracy')" \
    "$(extract_metric "$latest_train" 'train/sps/correct_weight_mass')" \
    "$(extract_metric "$latest_train" 'response_length/clip_ratio')" \
    "$(extract_metric "$latest_train" 'timing_s/step')" \
    "$(extract_metric "$latest_train" 'perf/total_num_tokens')" \
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

if [[ -n "$latest_runner_status" ]]; then
  echo "RUNNER_STATUS_LINES_BEGIN"
  printf '%s\n' "$latest_runner_status"
  echo "RUNNER_STATUS_LINES_END"
else
  echo "RUNNER_STATUS_LINES=NA"
fi
