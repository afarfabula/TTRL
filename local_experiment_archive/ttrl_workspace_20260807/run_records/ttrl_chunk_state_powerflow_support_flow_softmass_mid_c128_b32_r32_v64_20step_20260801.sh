#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
LAUNCHER="$ROOT/examples/ttrl/Qwen2.5-Math/math500_7b_8gpu.sh"

export RUN_ID="${RUN_ID:-ttrl_chunk_state_powerflow_support_flow_softmass_mid_c128_b32_r32_v64_20step_20260801}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0801-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-support-flow-softmass-20step}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/csf_softmass_20step}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export BACKBONE="${BACKBONE:-Qwen2.5-Math-7B}"
export BACKBONE_PATH="${BACKBONE_PATH:-/models/Qwen2.5-Math-7B}"
export DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-32}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export VAL_N="${VAL_N:-16}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
export TEST_FREQ="${TEST_FREQ:-20}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

exec bash "$LAUNCHER" \
  trainer.total_training_steps=20 \
  trainer.total_epochs=2 \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.val_before_train=False \
  trainer.final_val_enable=True \
  trainer.log_val_generations=0 \
  trainer.validation_metric_dump_path="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/${RUN_ID}_val_metrics.json" \
  ttrl.chunk_state_enable=True \
  ttrl.chunk_state_score_mode=support_flow \
  ttrl.chunk_state_source_mode=majority_consistent \
  ttrl.chunk_state_boundary_mode=mid \
  ttrl.chunk_state_chunk_size=128 \
  ttrl.chunk_state_candidates=8 \
  ttrl.chunk_state_support_anchor_count=4 \
  ttrl.chunk_state_support_anchor_candidate_start=0 \
  ttrl.chunk_state_support_anchor_min_mass=0.03125 \
  ttrl.chunk_state_support_flow_score_type=soft_mass \
  ttrl.chunk_state_support_flow_gain_slack=0.0 \
  ttrl.chunk_state_support_flow_baseline_scale=1.0 \
  ttrl.chunk_state_support_flow_source_prior_weight=1.0 \
  ttrl.chunk_state_support_flow_min_positive_margin=0.0 \
  ttrl.chunk_state_label_consistent_only=True \
  ttrl.chunk_state_zero_inconsistent_candidates=True \
  ttrl.chunk_state_prune_zero_weight_samples=True \
  ttrl.chunk_state_powerflow_weight_clip=4.0 \
  ttrl.chunk_state_powerflow_weight_clip_renorm=True \
  actor_rollout_ref.actor.powerflow_enable=True \
  actor_rollout_ref.actor.powerflow_use_boxed_reward=True \
  actor_rollout_ref.actor.powerflow_use_chunk_weights=True \
  actor_rollout_ref.actor.powerflow_init_ref_log_prob=0.36 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.chunk_weighted_nll_enable=False \
  ttrl.chunk_state_skip_all_negative=True \
  ttrl.chunk_state_diag_enable=True \
  ttrl.chunk_state_diag_jsonl="/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  "$@"
