#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/tiger/TTRL/verl"
PY="/opt/tiger/modelchef/.venv/bin/python3"
RUNNER="$ROOT/examples/ttrl/worker_run_sps_basin_contrast_process_sharpened_prob_qwen25_math_7b_50step_v81_strict_n4.sh"

echo "SMOKE_V81_LOCAL_GUARDS_BEGIN $(date '+%F %T')"

echo "SMOKE py_compile"
"$PY" -m py_compile \
  "$ROOT/verl/trainer/ppo/ttrl_utils.py" \
  "$ROOT/verl/trainer/ppo/ray_trainer.py" \
  "$ROOT/examples/ttrl/smoke_v81_basin_contrast_cpu.py" \
  "$ROOT/examples/ttrl/verify_pure_sharpening_strict_configs.py" \
  "$ROOT/examples/ttrl/audit_v68_v69_no_majority_target.py" \
  "$ROOT/examples/ttrl/audit_pure_sharpening_result.py" \
  "$ROOT/examples/ttrl/summarize_pure_sharpening_ablation.py" \
  "$ROOT/examples/ttrl/check_pure_sharpening_goal_completion.py"

echo "SMOKE bash_syntax"
bash -n \
  "$RUNNER" \
  "$ROOT/examples/ttrl/check_worker_readiness_for_v81.sh" \
  "$ROOT/examples/ttrl/smoke_v81_worker_readiness_guard.sh"

echo "SMOKE v81_basin_contrast_cpu"
"$PY" "$ROOT/examples/ttrl/smoke_v81_basin_contrast_cpu.py"

echo "SMOKE v81_worker_readiness_guard"
bash "$ROOT/examples/ttrl/smoke_v81_worker_readiness_guard.sh"

echo "SMOKE strict_config_audit"
"$PY" "$ROOT/examples/ttrl/verify_pure_sharpening_strict_configs.py"

echo "SMOKE no_majority_target_audit"
"$PY" "$ROOT/examples/ttrl/audit_v68_v69_no_majority_target.py"

echo "SMOKE ablation_summary_refresh"
"$PY" "$ROOT/examples/ttrl/summarize_pure_sharpening_ablation.py" \
  | tee "$ROOT/pure_sharpening_ablation_summary.tsv" >/dev/null
grep -F "v81_basin_contrast" "$ROOT/pure_sharpening_ablation_summary.tsv"

echo "SMOKE completion_gate_expected_current_fail"
set +e
"$PY" "$ROOT/examples/ttrl/check_pure_sharpening_goal_completion.py"
completion_status=$?
set -e
echo "SMOKE completion_gate_exit=${completion_status}"
if [ "$completion_status" -eq 0 ]; then
  echo "SMOKE completion_gate_unexpected_pass"
  exit 1
fi
echo "SMOKE completion_gate_not_complete"

echo "SMOKE worker_readiness_current"
set +e
bash "$ROOT/examples/ttrl/check_worker_readiness_for_v81.sh"
readiness_status=$?
set -e
echo "SMOKE worker_readiness_exit=${readiness_status}"

echo "SMOKE_V81_LOCAL_GUARDS_DONE $(date '+%F %T')"
