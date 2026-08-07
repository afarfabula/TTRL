#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
LAUNCHER="$ROOT/examples/ttrl/Qwen2.5-Math/math500_7b_8gpu.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_future_support_highmass_late_transport_support_gain_c128_probe4_b32_r32_v64_1step_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-highmass-late-transport-support-gain-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cfshl1}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export BACKBONE="${BACKBONE:-Qwen2.5-Math-7B}"
export BACKBONE_PATH="${BACKBONE_PATH:-/models/Qwen2.5-Math-7B}"
export DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-32}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
export VAL_N="${VAL_N:-16}"

export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.86}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
export TEST_FREQ="${TEST_FREQ:-2000000}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$LAUNCHER" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.total_epochs=1 \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.final_val_enable="$FINAL_VAL_ENABLE" \
  trainer.log_val_generations=0 \
  ttrl.chunk_state_score_mode=future_support_gain \
  ttrl.chunk_state_enable=True \
  ttrl.chunk_state_source_mode=majority_consistent \
  ttrl.chunk_state_source_select_by_mass=True \
  ttrl.chunk_state_min_prompt_top_mass=0.40 \
  ttrl.chunk_state_min_source_answer_mass=0.50 \
  ttrl.chunk_state_source_chunk_enable=True \
  ttrl.chunk_state_teacher_anchor_enable=False \
  ttrl.chunk_state_boundary_mode=mid \
  ttrl.chunk_state_mid_boundary_min_ratio=0.50 \
  ttrl.chunk_state_mid_boundary_max_ratio=0.90 \
  ttrl.chunk_state_future_support_score_type=transport_support_gain \
  ttrl.chunk_state_future_support_min_positive_margin=0.001 \
  ttrl.chunk_state_future_support_source_prior_weight=1.05 \
  ttrl.chunk_state_future_support_min_mass=0.03125 \
  ttrl.chunk_state_future_support_max_mass_coef=0.25 \
  ttrl.chunk_state_future_support_gain_slack=0.0 \
  ttrl.chunk_state_future_support_baseline_scale=1.0 \
  ttrl.chunk_state_future_support_min_state_coverage=0.50 \
  ttrl.chunk_state_future_support_max_state_oov=0.50 \
  ttrl.chunk_state_future_support_min_state_top_margin=0.02 \
  ttrl.chunk_state_future_support_min_candidate_coverage=0.25 \
  ttrl.chunk_state_future_support_min_candidate_mean_mass=0.02 \
  ttrl.chunk_state_min_answer_coverage=0.0 \
  ttrl.chunk_state_min_informative_gap=0.001 \
  ttrl.chunk_state_probe_samples=4 \
  ttrl.chunk_state_probe_max_tokens=3072 \
  ttrl.chunk_state_staged_probe_enable=False \
  ttrl.chunk_state_candidates=8 \
  ttrl.chunk_state_chunk_size=128 \
  ttrl.chunk_state_powerflow_weight_clip=4.0 \
  ttrl.chunk_state_powerflow_weight_clip_renorm=True \
  ttrl.chunk_state_zero_inconsistent_candidates=True \
  ttrl.chunk_state_prune_zero_weight_samples=True \
  ttrl.chunk_state_label_consistent_only=True \
  ttrl.chunk_state_skip_all_negative=True \
  ttrl.chunk_state_diag_enable=True \
  ttrl.chunk_state_diag_jsonl="$DIAG_JSONL" \
  actor_rollout_ref.actor.powerflow_enable=True \
  actor_rollout_ref.actor.powerflow_use_boxed_reward=True \
  actor_rollout_ref.actor.powerflow_use_chunk_weights=True \
  actor_rollout_ref.actor.powerflow_init_ref_log_prob=0.36 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.chunk_weighted_nll_enable=False \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  "$@"
