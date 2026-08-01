#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_b32_r32_v64_20step_20260731.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_voteinfo_probe4_b32_r32_v64_20step_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-voteinfo-probe4-powerflow}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/csp_voteinfo_probe4}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
export TEST_FREQ="${TEST_FREQ:-20}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-True}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

mkdir -p "$(dirname "$DIAG_JSONL")"
rm -f "$DIAG_JSONL"

exec bash "$BASE" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.final_val_enable="$FINAL_VAL_ENABLE" \
  trainer.save_freq=2000000 \
  ttrl.chunk_state_diag_enable=True \
  ttrl.chunk_state_diag_jsonl="$DIAG_JSONL" \
  ttrl.chunk_state_source_mode=random \
  ttrl.chunk_state_boundaries='[0,256,512,768,1024]' \
  ttrl.chunk_state_min_boundary=0 \
  ttrl.chunk_state_teacher_anchor_enable=False \
  ttrl.chunk_state_source_chunk_enable=False \
  ttrl.chunk_state_probe_samples=4 \
  ttrl.chunk_state_probe_max_tokens=1024 \
  ttrl.chunk_state_skip_all_negative=True \
  ttrl.chunk_state_skip_uniform=True \
  ttrl.chunk_state_min_informative_gap=0.0 \
  actor_rollout_ref.actor.powerflow_use_chunk_weights=True \
  actor_rollout_ref.actor.chunk_weighted_nll_enable=False \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  "$@"
