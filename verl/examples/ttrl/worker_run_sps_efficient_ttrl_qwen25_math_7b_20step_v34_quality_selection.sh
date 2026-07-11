#!/bin/bash
set -euo pipefail

LOG=/opt/tiger/TTRL/verl/sps_efficient_ttrl_qwen25_math_7b_20step_v34_quality_selection.log
RAY_LOG_SNAPSHOT=/opt/tiger/TTRL/verl/sps_efficient_ttrl_qwen25_math_7b_20step_v34_quality_selection_ray_taskrunner.log
METRICS_SNAPSHOT=/opt/tiger/TTRL/verl/sps_efficient_ttrl_qwen25_math_7b_20step_v34_quality_selection_metrics.txt
THROUGHPUT_SUMMARY=/opt/tiger/TTRL/verl/sps_efficient_ttrl_qwen25_math_7b_20step_v34_quality_selection_throughput_summary.txt
PROC_HEALTH=/opt/tiger/TTRL/verl/sps_efficient_ttrl_qwen25_math_7b_20step_v34_quality_selection_proc_health.txt
RAY_DIR=/tmp/r25v34q
SRC_MODEL=/opt/tiger/qwen2.5_math_7b
LOCAL_MODEL=/tmp/qwen2_5_math_7b_local_v34_quality_selection_20step

echo "WORKER_QWEN25_MATH_7B_V34_QUALITY_SELECTION_START $(date '+%F %T')" | tee "$LOG"
echo "HOST $(hostname)" | tee -a "$LOG"
ls -ld /proc /proc/self /proc/meminfo | tee -a "$LOG"
python - <<'PY' | tee -a "$LOG"
import os
print("PROC_COUNT_BEFORE", len(os.listdir("/proc")))
PY
df -h /opt/tiger /tmp | tee -a "$LOG"

bash /opt/tiger/TTRL/verl/examples/ttrl/check_cuda_compat_preflight.sh 2>&1 | tee -a "$LOG"

for attempt in 1 2 3; do
  if nvidia-smi -L | tee -a "$LOG"; then
    break
  fi
  echo "NVIDIA_SMI_RETRY attempt=${attempt} $(date '+%F %T')" | tee -a "$LOG"
  sleep 5
done
nvidia-smi -L >/dev/null
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader | tee -a "$LOG"

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
df -h /opt/tiger /tmp | tee -a "$LOG"
echo "MODEL_LOCAL_COPY_DONE $(date '+%F %T')" | tee -a "$LOG"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR="$RAY_DIR"
export MASTER_PORT=29696
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET6
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_qwen25_math_7b_v34_quality_selection_20step
export TRANSFORMERS_OFFLINE=1
rm -rf "$RAY_DIR"
rm -f "$RAY_LOG_SNAPSHOT" "$METRICS_SNAPSHOT" "$THROUGHPUT_SUMMARY" "$PROC_HEALTH"

bash /opt/tiger/TTRL/verl/examples/ttrl/run_sps_rule_conf_weight_math_qwen3_4b_50step_8gpu.sh \
  actor_rollout_ref.model.path="$LOCAL_MODEL" \
  ttrl.sps_weight_floor=0.15 \
  ttrl.sps_clip_penalty=0.5 \
  ttrl.sps_weight_power=1.5 \
  ttrl.sps_base_logprob_source=ref \
  ttrl.sps_answer_sharpen_beta=2.0 \
  ttrl.sps_answer_sharpen_capacity=False \
  ttrl.sps_rollout_selection=first \
  ttrl.sps_format_reward_coef=0.0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=14 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.val_kwargs.n=32 \
  trainer.validation_answer_selection_enable=True \
  trainer.validation_answer_selection_repeats=4 \
  trainer.validation_answer_selection_strategy=quality_weighted \
  trainer.validation_answer_selection_logprob_beta=1.0 \
  trainer.validation_answer_selection_clip_penalty=0.25 \
  ttrl.sps_reuse_rollout_log_probs_as_old=True \
  ttrl.sps_reuse_base_log_probs_as_ref=True \
  trainer.total_training_steps=20 \
  trainer.test_freq=20 \
  trainer.val_before_train=False \
  trainer.experiment_name=math-qwen25_math_7b-efficient-ttrl-v34-20step-quality-selection \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}

task_log=$(
  grep -R -l -a "Final validation metrics\|training/global_step:20\|val-core/MATH-TTT/acc/mean@4" \
    "$RAY_DIR/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
)
if [ -z "$task_log" ]; then
  task_log=$(
    grep -R -l -a "Training from scratch\|Total training steps: 20" \
      "$RAY_DIR/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
  )
fi

if [ -n "$task_log" ] && [ -f "$task_log" ]; then
  cp "$task_log" "$RAY_LOG_SNAPSHOT" || true
  {
    echo "TASK_LOG $task_log"
    grep -a -E "training/global_step|train/sps/|train/label_accuracy|train/reward_accuracy|train/majority_voting_reward|train/ground_truth_reward|train/pass@32|train/majority_ratio|response_length/clip_ratio|response_avg_logprob|response_clip|timing_s/|perf/total_num_tokens|perf/throughput|val-core/MATH-TTT/acc|val-raw/MATH-TTT/acc|val-aux/MATH-TTT/(acc|format_score|response_len|response_clip|response_avg_logprob)|Final validation metrics|global_step:20" "$task_log" || true
  } > "$METRICS_SNAPSHOT"
  echo "WORKER_QWEN25_MATH_7B_V34_QUALITY_SELECTION_RAY_LOG_SNAPSHOT $RAY_LOG_SNAPSHOT" | tee -a "$LOG"
  echo "WORKER_QWEN25_MATH_7B_V34_QUALITY_SELECTION_METRICS_SNAPSHOT $METRICS_SNAPSHOT" | tee -a "$LOG"
  tail -180 "$METRICS_SNAPSHOT" | tee -a "$LOG"

  python - "$task_log" "$THROUGHPUT_SUMMARY" <<'PY' || true
import re, sys
task_log, out_path = sys.argv[1], sys.argv[2]
rows = []
pat = re.compile(r"step:(\d+).*?timing_s/step:([0-9.]+).*?perf/total_num_tokens:([0-9.]+)")
for line in open(task_log, "r", errors="ignore"):
    m = pat.search(line)
    if m:
        step = int(m.group(1))
        if step <= 20:
            rows.append((step, float(m.group(2)), float(m.group(3))))
last = rows[-10:] if len(rows) >= 10 else rows
with open(out_path, "w") as f:
    f.write(f"task_log={task_log}\n")
    if last:
        f.write(f"steps={last[0][0]}-{last[-1][0]}\n")
        f.write(f"timing_s/step={sum(x[1] for x in last)/len(last):.3f}\n")
        f.write(f"perf/total_num_tokens={sum(x[2] for x in last)/len(last):.3f}\n")
        f.write(f"whole_machine_tokens_per_s={sum(x[2] for x in last)/max(sum(x[1] for x in last), 1e-9):.3f}\n")
    else:
        f.write("NO_STEP_ROWS\n")
PY
  cat "$THROUGHPUT_SUMMARY" | tee -a "$LOG"
else
  echo "WORKER_QWEN25_MATH_7B_V34_QUALITY_SELECTION_RAY_LOG_MISSING" | tee -a "$LOG"
fi

{
  echo "WORKER_QWEN25_MATH_7B_V34_QUALITY_SELECTION_STATUS $status $(date '+%F %T')"
  test -e /proc/self && echo PROC_SELF_OK_AFTER || echo PROC_SELF_BAD_AFTER
  test -e /proc/meminfo && echo PROC_MEMINFO_OK_AFTER || echo PROC_MEMINFO_BAD_AFTER
  python - <<'PY'
import os
print("PROC_COUNT_AFTER", len(os.listdir("/proc")))
PY
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
} | tee "$PROC_HEALTH" | tee -a "$LOG"

echo "WORKER_QWEN25_MATH_7B_V34_QUALITY_SELECTION_EXIT status=${status} $(date '+%F %T')" | tee -a "$LOG"
exit "$status"
