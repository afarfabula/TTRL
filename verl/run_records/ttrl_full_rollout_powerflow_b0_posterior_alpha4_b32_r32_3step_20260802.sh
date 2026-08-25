#!/usr/bin/env bash
set -euo pipefail

ROOT="/mlx_devbox/users/quyanyi/playground/TTRL/verl"
LAUNCHER="$ROOT/examples/ttrl/Qwen2.5-Math/math500_7b_8gpu.sh"

export RUN_ID="${RUN_ID:-ttrl_full_rollout_powerflow_b0_posterior_alpha4_b32_r32_3step_20260802}"
export EXPERIMENT="${EXPERIMENT:-${RUN_ID}}"
export LOG_NAME="${LOG_NAME:-0802-${RUN_ID}-MATH-TTT-Qwen2.5-Math-7B-fullrollout-b0-smoke}"
export TTRL_RUNTIME_DIR="${TTRL_RUNTIME_DIR:-/tmp/ttrl_b200/runtime/${RUN_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ttrl_b200/checkpoints/${RUN_ID}}"
export BACKBONE="${BACKBONE:-Qwen2.5-Math-7B}"
export BACKBONE_PATH="${BACKBONE_PATH:-/models/Qwen2.5-Math-7B}"
export DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-32}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
export VAL_N="${VAL_N:-16}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.86}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TEST_FREQ="${TEST_FREQ:--1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export FINAL_VAL_ENABLE="${FINAL_VAL_ENABLE:-False}"
export FULL_ROLLOUT_MODE="${FULL_ROLLOUT_MODE:-posterior_sharpen}"
export FULL_ROLLOUT_ALPHA="${FULL_ROLLOUT_ALPHA:-4.0}"
export FULL_ROLLOUT_BETA="${FULL_ROLLOUT_BETA:-4.0}"
export FULL_ROLLOUT_SLACK="${FULL_ROLLOUT_SLACK:-0.02}"
export FULL_ROLLOUT_EPS="${FULL_ROLLOUT_EPS:-0.0}"
export FULL_ROLLOUT_WEIGHT_CLIP="${FULL_ROLLOUT_WEIGHT_CLIP:-4.0}"
export FULL_ROLLOUT_WEIGHT_CLIP_RENORM="${FULL_ROLLOUT_WEIGHT_CLIP_RENORM:-True}"
export FULL_ROLLOUT_INVALID_WEIGHT="${FULL_ROLLOUT_INVALID_WEIGHT:-0.0}"
export FULL_ROLLOUT_QUALITY_GATE_ENABLE="${FULL_ROLLOUT_QUALITY_GATE_ENABLE:-False}"
export FULL_ROLLOUT_MIN_VALID_COVERAGE="${FULL_ROLLOUT_MIN_VALID_COVERAGE:-0.0}"
export FULL_ROLLOUT_MIN_TOP_MASS="${FULL_ROLLOUT_MIN_TOP_MASS:-0.0}"
export FULL_ROLLOUT_MIN_MARGIN="${FULL_ROLLOUT_MIN_MARGIN:-0.0}"
export FULL_ROLLOUT_MARGIN_TAU="${FULL_ROLLOUT_MARGIN_TAU:-0.25}"
export FULL_ROLLOUT_SOFT_PROMPT_WEIGHT_ENABLE="${FULL_ROLLOUT_SOFT_PROMPT_WEIGHT_ENABLE:-False}"
export FULL_ROLLOUT_PROMPT_WEIGHT_FLOOR="${FULL_ROLLOUT_PROMPT_WEIGHT_FLOOR:-0.2}"
export FULL_ROLLOUT_TOP_K_ANSWERS="${FULL_ROLLOUT_TOP_K_ANSWERS:-2}"
export FULL_ROLLOUT_REPRESENTATIVES_PER_ANSWER="${FULL_ROLLOUT_REPRESENTATIVES_PER_ANSWER:-2}"
export FULL_ROLLOUT_POLLUTION_GUARD_ENABLE="${FULL_ROLLOUT_POLLUTION_GUARD_ENABLE:-True}"

exec bash "$LAUNCHER" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.final_val_enable="$FINAL_VAL_ENABLE" \
  trainer.log_val_generations=0 \
  trainer.balance_batch=False \
  ttrl.enable=True \
  ttrl.powerflow_no_majority=True \
  ttrl.full_rollout_powerflow_enable=True \
  ttrl.full_rollout_powerflow_mode="$FULL_ROLLOUT_MODE" \
  ttrl.full_rollout_powerflow_alpha="$FULL_ROLLOUT_ALPHA" \
  ttrl.full_rollout_powerflow_beta="$FULL_ROLLOUT_BETA" \
  ttrl.full_rollout_powerflow_slack="$FULL_ROLLOUT_SLACK" \
  ttrl.full_rollout_powerflow_eps="$FULL_ROLLOUT_EPS" \
  ttrl.full_rollout_powerflow_weight_clip="$FULL_ROLLOUT_WEIGHT_CLIP" \
  ttrl.full_rollout_powerflow_weight_clip_renorm="$FULL_ROLLOUT_WEIGHT_CLIP_RENORM" \
  ttrl.full_rollout_powerflow_invalid_weight="$FULL_ROLLOUT_INVALID_WEIGHT" \
  ttrl.full_rollout_powerflow_quality_gate_enable="$FULL_ROLLOUT_QUALITY_GATE_ENABLE" \
  ttrl.full_rollout_powerflow_min_valid_answer_coverage="$FULL_ROLLOUT_MIN_VALID_COVERAGE" \
  ttrl.full_rollout_powerflow_min_top_mass="$FULL_ROLLOUT_MIN_TOP_MASS" \
  ttrl.full_rollout_powerflow_min_margin="$FULL_ROLLOUT_MIN_MARGIN" \
  ttrl.full_rollout_powerflow_margin_tau="$FULL_ROLLOUT_MARGIN_TAU" \
  ttrl.full_rollout_powerflow_soft_prompt_weight_enable="$FULL_ROLLOUT_SOFT_PROMPT_WEIGHT_ENABLE" \
  ttrl.full_rollout_powerflow_prompt_weight_floor="$FULL_ROLLOUT_PROMPT_WEIGHT_FLOOR" \
  ttrl.full_rollout_powerflow_top_k_answers="$FULL_ROLLOUT_TOP_K_ANSWERS" \
  ttrl.full_rollout_powerflow_representatives_per_answer="$FULL_ROLLOUT_REPRESENTATIVES_PER_ANSWER" \
  ttrl.full_rollout_powerflow_pollution_guard_enable="$FULL_ROLLOUT_POLLUTION_GUARD_ENABLE" \
  ttrl.chunk_state_enable=False \
  actor_rollout_ref.actor.powerflow_enable=True \
  actor_rollout_ref.actor.powerflow_use_boxed_reward=False \
  actor_rollout_ref.actor.powerflow_use_chunk_weights=True \
  actor_rollout_ref.actor.powerflow_chunk_loss_mode=target_only \
  actor_rollout_ref.actor.powerflow_init_ref_log_prob=0.36 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.chunk_weighted_nll_enable=False \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  "$@"
