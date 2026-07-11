#!/bin/bash
set -euo pipefail

EXP_NAME=sps_bucket_select_beta20_floor015_clip05_power15_refbase_localfp32_qwen3_8b_8_620step_v26_tput
LOG=/opt/tiger/TTRL/verl/${EXP_NAME}.log
RAY_LOG_SNAPSHOT=/opt/tiger/TTRL/verl/${EXP_NAME}_ray_taskrunner.log
METRICS_SNAPSHOT=/opt/tiger/TTRL/verl/${EXP_NAME}_metrics.txt
SUMMARY=/opt/tiger/TTRL/verl/${EXP_NAME}_throughput_summary.txt
PROC_SNAPSHOT=/opt/tiger/TTRL/verl/${EXP_NAME}_proc_health.txt
RAY_DIR=/tmp/r26_620_tput
SRC_MODEL=/opt/tiger/qwen3_8b
LOCAL_MODEL=/tmp/qwen3_8b_local_v21_answer_sharpen

echo "WORKER_QWEN3_8B_V26_620_TPUT_START $(date '+%F %T')" | tee "$LOG"
echo "HOST $(hostname)" | tee -a "$LOG"
bash /opt/tiger/TTRL/verl/examples/ttrl/check_cuda_compat_preflight.sh 2>&1 | tee -a "$LOG"
if [ ! -e /proc/self ] || [ ! -e /proc/meminfo ]; then
  echo "WORKER_QWEN3_8B_V26_620_TPUT_PROC_BAD_BEFORE" | tee -a "$LOG"
  exit 97
fi
ls -ld /proc /proc/self /proc/meminfo | tee -a "$LOG"
python - <<'PY' | tee -a "$LOG"
import os
print("PROC_COUNT_BEFORE", len(os.listdir("/proc")))
PY
df -h /opt/tiger /tmp | tee -a "$LOG"
for attempt in 1 2 3; do
  if nvidia-smi -L | tee -a "$LOG"; then
    break
  fi
  echo "NVIDIA_SMI_RETRY attempt=${attempt} $(date '+%F %T')" | tee -a "$LOG"
  sleep 5
done
nvidia-smi -L >/dev/null

if [ -s "$LOCAL_MODEL/config.json" ]; then
  echo "MODEL_LOCAL_COPY_REUSE $(date '+%F %T') dst=${LOCAL_MODEL}" | tee -a "$LOG"
else
  echo "MODEL_LOCAL_COPY_START $(date '+%F %T') src=${SRC_MODEL} dst=${LOCAL_MODEL}" | tee -a "$LOG"
  rm -rf "${LOCAL_MODEL}.tmp"
  mkdir -p "${LOCAL_MODEL}.tmp"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${SRC_MODEL}/" "${LOCAL_MODEL}.tmp/" 2>&1 | tee -a "$LOG"
  else
    cp -a "${SRC_MODEL}/." "${LOCAL_MODEL}.tmp/"
  fi
  find "${LOCAL_MODEL}.tmp" -maxdepth 1 -type f -printf '%f %s\n' | sort | tee -a "$LOG"
  rm -rf "$LOCAL_MODEL"
  mv "${LOCAL_MODEL}.tmp" "$LOCAL_MODEL"
  echo "MODEL_LOCAL_COPY_DONE $(date '+%F %T')" | tee -a "$LOG"
fi
df -h /opt/tiger /tmp | tee -a "$LOG"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR="$RAY_DIR"
export MASTER_PORT=29683
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET6
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_qwen3_8b_v26_bucket_select_620_tput
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
rm -rf "$RAY_DIR"
rm -f "$RAY_LOG_SNAPSHOT" "$METRICS_SNAPSHOT" "$SUMMARY" "$PROC_SNAPSHOT"

set +e
bash /opt/tiger/TTRL/verl/examples/ttrl/run_sps_rule_conf_weight_math_qwen3_4b_50step_8gpu.sh \
  actor_rollout_ref.model.path="$LOCAL_MODEL" \
  ttrl.sps_weight_floor=0.15 \
  ttrl.sps_clip_penalty=0.5 \
  ttrl.sps_weight_power=1.5 \
  ttrl.sps_base_logprob_source=ref \
  ttrl.sps_answer_sharpen_beta=2.0 \
  ttrl.sps_answer_sharpen_capacity=False \
  ttrl.sps_rollout_selection=sharpened_cluster \
  ttrl.sps_selection_temperature=0.4 \
  ttrl.sps_selection_require_majority=False \
  ttrl.sps_selection_cluster_bonus=1.0 \
  ttrl.sps_selection_parseable_bonus=3.0 \
  ttrl.sps_selection_nonclip_bonus=4.0 \
  ttrl.sps_selection_priority=nonclip_parseable_bucket \
  ttrl.sps_format_reward_coef=0.0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=14 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  ttrl.sps_reuse_rollout_log_probs_as_old=True \
  ttrl.sps_reuse_base_log_probs_as_ref=True \
  trainer.total_epochs=10 \
  trainer.total_training_steps=620 \
  trainer.test_freq=620 \
  trainer.val_before_train=False \
  trainer.experiment_name=math-qwen3_8b-sps-bucket-select-beta20-floor015-clip05-power15-refbase-localfp32-v26-620step-tput-8gpu \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

{
  echo "WORKER_QWEN3_8B_V26_620_TPUT_STATUS ${status} $(date '+%F %T')"
  test -e /proc/self && echo PROC_SELF_OK_AFTER || echo PROC_SELF_BAD_AFTER
  test -e /proc/meminfo && echo PROC_MEMINFO_OK_AFTER || echo PROC_MEMINFO_BAD_AFTER
  python - <<'PY'
import os
print("PROC_COUNT_AFTER", len(os.listdir("/proc")))
PY
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
} 2>&1 | tee "$PROC_SNAPSHOT" | tee -a "$LOG"

task_log=$(
  grep -R -l -a "Final validation metrics\|training/global_step:620\|val-core/MATH-TTT/acc/mean@4" \
    "$RAY_DIR/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
)
if [ -z "$task_log" ]; then
  task_log=$(
    grep -R -l -a "Training from scratch\|Total training steps: 620" \
      "$RAY_DIR/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
  )
fi

if [ -n "$task_log" ] && [ -f "$task_log" ]; then
  cp "$task_log" "$RAY_LOG_SNAPSHOT" || true
  {
    echo "TASK_LOG $task_log"
    grep -a -E "training/global_step|train/sps/selected_|train/sps/selection_fallback|train/sps/answer_|train/sps/majority_sharp|train/sps/train_weight|train/sps/weighted_label_confidence|train/ground_truth_reward|train/pass@32|train/majority_ratio|response_length/clip_ratio|timing_s/step|timing_s/gen|timing_s/generate_sequences|timing_s/sps_|timing_s/old_log_prob|timing_s/ref|timing_s/update_actor|perf/total_num_tokens|perf/throughput|val-core/MATH-TTT/acc|val-aux/MATH-TTT/acc|val-aux/MATH-TTT/format_score|Final validation metrics|global_step:620" "$task_log" || true
  } > "$METRICS_SNAPSHOT"
  /opt/tiger/TTRL/verl/examples/ttrl/parse_ttrl_throughput.py "$task_log" --last 50 > "$SUMMARY" || true
  echo "WORKER_QWEN3_8B_V26_620_TPUT_RAY_LOG_SNAPSHOT $RAY_LOG_SNAPSHOT" | tee -a "$LOG"
  echo "WORKER_QWEN3_8B_V26_620_TPUT_METRICS_SNAPSHOT $METRICS_SNAPSHOT" | tee -a "$LOG"
  echo "WORKER_QWEN3_8B_V26_620_TPUT_SUMMARY $SUMMARY" | tee -a "$LOG"
  tail -100 "$METRICS_SNAPSHOT" | tee -a "$LOG"
else
  echo "WORKER_QWEN3_8B_V26_620_TPUT_RAY_LOG_MISSING" | tee -a "$LOG"
fi
echo "WORKER_QWEN3_8B_V26_620_TPUT_EXIT status=${status} $(date '+%F %T')" | tee -a "$LOG"
exit "$status"
