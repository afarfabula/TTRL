#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
BASE="$ROOT/run_records/ttrl_chunk_state_powerflow_voteinfo_probe4_b32_r32_v64_20step_20260801.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_support_anchor_mid_c128_b32_r32_v64_3step_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-support-anchor-no-probe-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cssa1283}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TEST_FREQ="${TEST_FREQ:-2000000}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$BASE" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.final_val_enable="$FINAL_VAL_ENABLE" \
  +ttrl.chunk_state_score_mode=support_anchor \
  ttrl.chunk_state_source_mode=majority_consistent \
  ttrl.chunk_state_boundary_mode=mid \
  ttrl.chunk_state_mid_boundary_min_ratio=0.25 \
  ttrl.chunk_state_mid_boundary_max_ratio=0.80 \
  ttrl.chunk_state_chunk_size=128 \
  ttrl.chunk_state_boundaries='[0,128,256,384,512,640,768,896,1024]' \
  ttrl.chunk_state_min_boundary=0 \
  ttrl.chunk_state_support_anchor_count=4 \
  ttrl.chunk_state_support_anchor_candidate_start=0 \
  ttrl.chunk_state_support_anchor_min_mass=0.03125 \
  ttrl.chunk_state_source_chunk_enable=False \
  ttrl.chunk_state_teacher_anchor_enable=False \
  ttrl.chunk_state_target_guard_enable=False \
  ttrl.chunk_state_target_guard_candidate_enable=False \
  ttrl.chunk_state_probe_samples=1 \
  ttrl.chunk_state_probe_max_tokens=1 \
  ttrl.chunk_state_skip_all_negative=True \
  ttrl.chunk_state_skip_uniform=True \
  ttrl.chunk_state_min_answer_coverage=0.25 \
  ttrl.chunk_state_min_informative_gap=0.02 \
  ttrl.chunk_state_label_consistent_only=True \
  ttrl.chunk_state_zero_inconsistent_candidates=True \
  ttrl.chunk_state_prune_zero_weight_samples=True \
  ttrl.chunk_state_powerflow_weight_clip=4.0 \
  ttrl.chunk_state_powerflow_weight_clip_renorm=True \
  actor_rollout_ref.actor.powerflow_use_chunk_weights=True \
  actor_rollout_ref.actor.chunk_weighted_nll_enable=False \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  "$@"
