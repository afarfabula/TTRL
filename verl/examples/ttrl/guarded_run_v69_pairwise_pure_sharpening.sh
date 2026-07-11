#!/usr/bin/env bash
set -euo pipefail

READINESS="/opt/tiger/TTRL/verl/examples/ttrl/check_worker_readiness_for_v69.sh"
RUNNER="/opt/tiger/TTRL/verl/examples/ttrl/worker_run_sps_pairwise_process_count_neutral_sharpened_prob_qwen25_math_7b_50step_v69_strict_n4.sh"

echo "GUARDED_V69_READINESS_BEGIN $(date '+%F %T')"
bash "$READINESS"
echo "GUARDED_V69_READINESS_PASS $(date '+%F %T')"

if [ "${RUN_V69_AFTER_READINESS:-0}" != "1" ]; then
  echo "GUARDED_V69_DRY_RUN=1"
  echo "NEXT_STEP=login to the selected worker printed above, then run:"
  echo "RUN_V69_AFTER_READINESS=1 bash $0"
  echo "DIRECT_RUNNER=bash $RUNNER"
  exit 0
fi

echo "GUARDED_V69_RUNNER_BEGIN $(date '+%F %T')"
bash "$RUNNER"
