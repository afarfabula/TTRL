#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
LAUNCHER="$ROOT/examples/ttrl/Qwen2.5-Math/math500_7b_8gpu.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_support_flow_massprop_answersplit_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-support-flow-massprop-answersplit-gate030-softplusgain-3step-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/cfsfs1}"
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
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TEST_FREQ="${TEST_FREQ:--1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"
export DIAG_JSONL="${DIAG_JSONL:-/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl}"

exec bash "$LAUNCHER" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.final_val_enable="$FINAL_VAL_ENABLE" \
  trainer.log_val_generations=0 \
  ttrl.chunk_state_score_mode=support_flow \
  ttrl.chunk_state_enable=True \
  ttrl.chunk_state_source_mode=support_mixed \
  ttrl.chunk_state_source_select_by_mass=True \
  ttrl.chunk_state_min_prompt_top_mass=0.30 \
  ttrl.chunk_state_min_source_answer_mass=0.0 \
  ttrl.chunk_state_source_mixed_low_ratio=0.75 \
  ttrl.chunk_state_source_low_max_answer_mass=0.35 \
  ttrl.chunk_state_source_min_valid_answer_mass=0.03125 \
  ttrl.chunk_state_source_chunk_enable=True \
  ttrl.chunk_state_source_chunk_candidate_index=0 \
  ttrl.chunk_state_teacher_anchor_enable=False \
  ttrl.chunk_state_support_anchor_enable=True \
  ttrl.chunk_state_support_anchor_count=7 \
  ttrl.chunk_state_support_anchor_candidate_start=1 \
  ttrl.chunk_state_support_anchor_min_mass=0.03125 \
  ttrl.chunk_state_support_anchor_prefix_compat_enable=False \
  ttrl.chunk_state_support_anchor_skip_source=True \
  ttrl.chunk_state_support_anchor_selection_mode=mass_ranked \
  ttrl.chunk_state_support_flow_split_mass_by_answer=True \
  ttrl.chunk_state_boundary_mode=mid \
  ttrl.chunk_state_mid_boundary_min_ratio=0.25 \
  ttrl.chunk_state_mid_boundary_max_ratio=0.70 \
  ttrl.chunk_state_support_flow_score_type=softplus_gain \
  ttrl.chunk_state_support_flow_softplus_temperature=0.125 \
  ttrl.chunk_state_support_flow_gain_slack=0.0 \
  ttrl.chunk_state_support_flow_source_prior_weight=1.0 \
  ttrl.chunk_state_support_flow_min_positive_margin=0.0 \
  ttrl.chunk_state_min_answer_coverage=0.0 \
  ttrl.chunk_state_min_informative_gap=0.001 \
  ttrl.chunk_state_probe_samples=1 \
  ttrl.chunk_state_probe_max_tokens=1 \
  ttrl.chunk_state_candidates=8 \
  ttrl.chunk_state_chunk_size=128 \
  ttrl.chunk_state_powerflow_weight_clip=4.0 \
  ttrl.chunk_state_powerflow_weight_clip_renorm=True \
  ttrl.chunk_state_zero_inconsistent_candidates=False \
  ttrl.chunk_state_prune_zero_weight_samples=True \
  ttrl.chunk_state_label_consistent_only=False \
  ttrl.chunk_state_skip_all_negative=False \
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
