#!/bin/bash
set -euo pipefail

EXP=sps_support_residual_process_sharpened_prob_qwen25_math_7b_50step_v78_strict_n4
LOG=/opt/tiger/TTRL/verl/${EXP}.log
RAY_LOG_SNAPSHOT=/opt/tiger/TTRL/verl/${EXP}_ray_taskrunner.log
METRICS_SNAPSHOT=/opt/tiger/TTRL/verl/${EXP}_metrics.txt
THROUGHPUT_SUMMARY=/opt/tiger/TTRL/verl/${EXP}_throughput_summary.txt
PROC_HEALTH=/opt/tiger/TTRL/verl/${EXP}_proc_health.txt
CACHE_ROOT=/tmp/ttrl_cache/${EXP}
RAY_DIR=/tmp/r78
SHORT_TMPDIR=/tmp/t78
SRC_MODEL=/opt/tiger/qwen2.5_math_7b
LOCAL_MODEL=${CACHE_ROOT}/model

final_status_snapshot() {
  local status="$1"
  {
    echo "WORKER_QWEN25_MATH_7B_V78_SUPPORT_RESIDUAL_PROCESS_SHARPENED_STRICT_N4_FINAL_STATUS ${status} $(date '+%F %T')"
    test -e /proc/self && echo PROC_SELF_OK_FINAL || echo PROC_SELF_BAD_FINAL
    test -e /proc/meminfo && echo PROC_MEMINFO_OK_FINAL || echo PROC_MEMINFO_BAD_FINAL
    python - <<'PY' || true
import os
try:
    print("PROC_COUNT_FINAL", len(os.listdir("/proc")))
except Exception as exc:
    print("PROC_COUNT_FINAL_ERROR", repr(exc))
PY
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
  } | tee -a "$PROC_HEALTH" | tee -a "$LOG" >/dev/null
}

on_exit() {
  local status=$?
  final_status_snapshot "$status"
}
trap on_exit EXIT

echo "WORKER_QWEN25_MATH_7B_V78_SUPPORT_RESIDUAL_PROCESS_SHARPENED_STRICT_N4_START $(date '+%F %T')" | tee "$LOG"
echo "HOST $(hostname)" | tee -a "$LOG"
ls -ld /proc /proc/self /proc/meminfo | tee -a "$LOG"
python - <<'PY' | tee -a "$LOG"
import os
print("PROC_COUNT_BEFORE", len(os.listdir("/proc")))
PY
df -h / /opt/tiger /tmp | tee -a "$LOG"

source /opt/tiger/TTRL/verl/examples/ttrl/setup_ttrl_cuda_env.sh
ttrl_setup_cuda_env "$EXP" "$LOG"

bash /opt/tiger/TTRL/verl/examples/ttrl/check_cuda_compat_preflight.sh 2>&1 | tee -a "$LOG"
ttrl_record_cuda_preflight "$LOG" 1
nvidia-smi -L | tee -a "$LOG"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader | tee -a "$LOG"
GPU_COUNT=$(
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l
)
echo "GPU_COUNT_BEFORE_TRAIN ${GPU_COUNT}" | tee -a "$LOG"
if [ "$GPU_COUNT" -lt 8 ]; then
  echo "GPU_COUNT_BEFORE_TRAIN_BAD expected=8 actual=${GPU_COUNT}" | tee -a "$LOG"
  exit 86
fi

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
  rm -rf "$LOCAL_MODEL"
  mv "${LOCAL_MODEL}.tmp" "$LOCAL_MODEL"
  df -h / /opt/tiger /tmp | tee -a "$LOG"
  echo "MODEL_LOCAL_COPY_DONE $(date '+%F %T')" | tee -a "$LOG"
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TTRL_CACHE_ROOT="$CACHE_ROOT"
export RAY_TMPDIR="$RAY_DIR"
export HF_HOME="$CACHE_ROOT/hf_home"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf_datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$SHORT_TMPDIR"
export MY_HOST_IP=127.0.0.1
unset MY_HOST_IPV6
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29717
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export TTRL_UNSET_NCCL_SOCKET_FAMILY=1
unset NCCL_SOCKET_FAMILY
export TRANSFORMERS_OFFLINE=1
rm -rf "$RAY_DIR" "$SHORT_TMPDIR"
mkdir -p "$RAY_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
rm -f "$RAY_LOG_SNAPSHOT" "$METRICS_SNAPSHOT" "$THROUGHPUT_SUMMARY" "$PROC_HEALTH"

set +e
bash /opt/tiger/TTRL/verl/examples/ttrl/run_sps_rule_conf_weight_math_qwen3_4b_50step_8gpu.sh \
  actor_rollout_ref.model.path="$LOCAL_MODEL" \
  actor_rollout_ref.rollout.temperature=1.0 \
  ttrl.sps_proposal_temperature=1.0 \
  ttrl.sps_base_logprob_source=ref \
  ttrl.sps_reward_mode=support_residual_process_sharpened_prob \
  ttrl.sps_direct_beta=4.0 \
  ttrl.sps_direct_target_temperature=1.0 \
  ttrl.sps_direct_reward_scale=1.0 \
  ttrl.sps_direct_per_sample_mode=answer_mass \
  ttrl.sps_direct_reward_floor=0.0 \
  ttrl.sps_direct_process_tilt_strength=0.6 \
  ttrl.sps_direct_process_cluster_tilt_strength=0.8 \
  ttrl.sps_direct_base_contrast_strength=0.0 \
  ttrl.sps_direct_base_contrast_gate_strength=0.0 \
  ttrl.sps_direct_base_contrast_process_margin=0.0 \
  ttrl.sps_direct_target_effective_k_min=0.0 \
  ttrl.sps_direct_target_effective_k_max=0.0 \
  ttrl.sps_direct_target_effective_k_strength=0.0 \
  ttrl.sps_low_budget_k=4 \
  ttrl.sps_process_tail_fraction=0.50 \
  ttrl.sps_rollout_selection=first \
  ttrl.sps_majority_reward_coef=0.0 \
  ttrl.sps_format_reward_coef=0.0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=14 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  trainer.validation_answer_selection_enable=False \
  ttrl.sps_reuse_rollout_log_probs_as_old=True \
  ttrl.sps_reuse_base_log_probs_as_ref=True \
  trainer.total_training_steps=50 \
  trainer.test_freq=50 \
  trainer.val_before_train=False \
  +ray_init.no_runtime_env=True \
  +ray_init.include_dashboard=False \
  +ray_init.node_ip_address=127.0.0.1 \
  trainer.experiment_name=math-qwen25_math_7b-support-residual-process-sharpened-prob-v78-strict-n4 \
  ttrl.sps_direct_count_neutral_aggregation=False \
  ttrl.sps_direct_pairwise_process_preference_strength=0.0 \
  ttrl.sps_direct_pairwise_process_preference_temperature=1.0 \
  ttrl.sps_direct_support_gate_strength=1.0 \
  ttrl.sps_direct_support_gate_temperature=1.0 \
  ttrl.sps_direct_support_gate_floor=0.05 \
  ttrl.sps_direct_support_gate_base_weight=1.0 \
  ttrl.sps_direct_support_gate_low_budget_weight=1.0 \
  ttrl.sps_direct_support_gate_process_weight=1.0 \
  ttrl.sps_direct_support_residual_strength=0.5 \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

task_log=$(
  grep -R -l -a "Final validation metrics\|training/global_step:50\|val-core/MATH-TTT/acc/mean@4" \
    "$RAY_DIR/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
)
if [ -z "$task_log" ]; then
  task_log=$(
    grep -R -l -a "Training from scratch\|Total training steps: 50" \
      "$RAY_DIR/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
  )
fi

if [ -n "$task_log" ] && [ -f "$task_log" ]; then
  cp "$task_log" "$RAY_LOG_SNAPSHOT" || true
  {
    echo "TASK_LOG $task_log"
    grep -a -E "training/global_step|train/sps/|train/pass@32|train/majority_ratio|response_length/clip_ratio|timing_s/|perf/total_num_tokens|perf/throughput|val-core/MATH-TTT/acc|val-aux/MATH-TTT/(acc|format_score|response_clip)|Final validation metrics|global_step:50" "$task_log" || true
  } > "$METRICS_SNAPSHOT"
  echo "WORKER_QWEN25_MATH_7B_V78_SUPPORT_RESIDUAL_PROCESS_SHARPENED_STRICT_N4_RAY_LOG_SNAPSHOT $RAY_LOG_SNAPSHOT" | tee -a "$LOG"
  echo "WORKER_QWEN25_MATH_7B_V78_SUPPORT_RESIDUAL_PROCESS_SHARPENED_STRICT_N4_METRICS_SNAPSHOT $METRICS_SNAPSHOT" | tee -a "$LOG"
  tail -220 "$METRICS_SNAPSHOT" | tee -a "$LOG"

  python - "$task_log" "$THROUGHPUT_SUMMARY" <<'PY' || true
import re, sys
task_log, out_path = sys.argv[1], sys.argv[2]
rows = []
pat = re.compile(r"step:(\d+).*?timing_s/step:([0-9.]+).*?perf/total_num_tokens:([0-9.]+)")
for line in open(task_log, "r", errors="ignore"):
    m = pat.search(line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
nonval = [row for row in rows if row[0] < 50]
last = [row for row in rows if 41 <= row[0] <= 50]
with open(out_path, "w") as f:
    f.write(f"task_log={task_log}\n")
    f.write(f"step_rows={len(rows)}\n")
    if nonval:
        f.write(f"non_validation_steps_count={len(nonval)}\n")
        f.write(f"non_validation_timing_s/step={sum(x[1] for x in nonval)/len(nonval):.3f}\n")
        f.write(f"non_validation_whole_machine_tokens_per_s={sum(x[2] for x in nonval)/max(sum(x[1] for x in nonval), 1e-9):.3f}\n")
    if last:
        f.write(f"steps_41_50_timing_s/step={sum(x[1] for x in last)/len(last):.3f}\n")
        f.write(f"steps_41_50_whole_machine_tokens_per_s={sum(x[2] for x in last)/max(sum(x[1] for x in last), 1e-9):.3f}\n")
    if not rows:
        f.write("NO_STEP_ROWS\n")
PY
  cat "$THROUGHPUT_SUMMARY" | tee -a "$LOG"
else
  echo "WORKER_QWEN25_MATH_7B_V78_SUPPORT_RESIDUAL_PROCESS_SHARPENED_STRICT_N4_RAY_LOG_MISSING" | tee -a "$LOG"
fi

{
  echo "WORKER_QWEN25_MATH_7B_V78_SUPPORT_RESIDUAL_PROCESS_SHARPENED_STRICT_N4_STATUS $status $(date '+%F %T')"
  test -e /proc/self && echo PROC_SELF_OK_AFTER || echo PROC_SELF_BAD_AFTER
  test -e /proc/meminfo && echo PROC_MEMINFO_OK_AFTER || echo PROC_MEMINFO_BAD_AFTER
  python - <<'PY'
import os
print("PROC_COUNT_AFTER", len(os.listdir("/proc")))
PY
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
} | tee "$PROC_HEALTH" | tee -a "$LOG"

echo "WORKER_QWEN25_MATH_7B_V78_SUPPORT_RESIDUAL_PROCESS_SHARPENED_STRICT_N4_ABLATION_SUMMARY_BEGIN $(date '+%F %T')" | tee -a "$LOG"
set +e
/opt/tiger/modelchef/.venv/bin/python3 \
  /opt/tiger/TTRL/verl/examples/ttrl/summarize_pure_sharpening_ablation.py \
  | tee /opt/tiger/TTRL/verl/pure_sharpening_ablation_summary.tsv \
  | tee -a "$LOG"
summary_status=${PIPESTATUS[0]}
echo "WORKER_QWEN25_MATH_7B_V78_SUPPORT_RESIDUAL_PROCESS_SHARPENED_STRICT_N4_ABLATION_SUMMARY_STATUS ${summary_status}" | tee -a "$LOG"
echo "WORKER_QWEN25_MATH_7B_V78_SUPPORT_RESIDUAL_PROCESS_SHARPENED_STRICT_N4_COMPLETION_AUDIT_BEGIN $(date '+%F %T')" | tee -a "$LOG"
/opt/tiger/modelchef/.venv/bin/python3 \
  /opt/tiger/TTRL/verl/examples/ttrl/check_pure_sharpening_goal_completion.py \
  2>&1 | tee -a "$LOG"
completion_audit_status=${PIPESTATUS[0]}
echo "WORKER_QWEN25_MATH_7B_V78_SUPPORT_RESIDUAL_PROCESS_SHARPENED_STRICT_N4_COMPLETION_AUDIT_STATUS ${completion_audit_status}" | tee -a "$LOG"
set -e

echo "WORKER_QWEN25_MATH_7B_V78_SUPPORT_RESIDUAL_PROCESS_SHARPENED_STRICT_N4_EXIT status=${status} $(date '+%F %T')" | tee -a "$LOG"
trap - EXIT
final_status_snapshot "$status"
exit "$status"
