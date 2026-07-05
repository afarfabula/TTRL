#!/bin/bash
set -euo pipefail

ROUNDS=${ROUNDS:-10}
EXP_NAME=${EXP_NAME:-v37_teardown_stress_noenv_10x}
LOG=${LOG:-/opt/tiger/TTRL/verl/${EXP_NAME}.log}
SUMMARY=${SUMMARY:-/opt/tiger/TTRL/verl/${EXP_NAME}_summary.tsv}
SRC_MODEL=${SRC_MODEL:-/opt/tiger/qwen2.5_math_7b}
LOCAL_MODEL=${LOCAL_MODEL:-/tmp/qwen2_5_math_7b_local_v37_bucket_select_strict_n4_20step}
RUN_CUDA_COMPAT_PREFLIGHT=${RUN_CUDA_COMPAT_PREFLIGHT:-0}
RAY_NO_RUNTIME_ENV=${RAY_NO_RUNTIME_ENV:-1}
RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD:-False}
RAY_NODE_IP_ADDRESS=${RAY_NODE_IP_ADDRESS:-127.0.0.1}

health_snapshot() {
  local label="$1"
  python - "$label" <<'PY'
import json
import os
import sys
import time

label = sys.argv[1]
health = {
    "label": label,
    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "proc_self": os.path.exists("/proc/self"),
    "proc_meminfo": os.path.exists("/proc/meminfo"),
}
try:
    health["proc_count"] = len(os.listdir("/proc"))
except Exception as exc:
    health["proc_count_error"] = repr(exc)
print("PROC_HEALTH", json.dumps(health, sort_keys=True), flush=True)
PY
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
  ps -eo pid,ppid,stat,etime,cmd | grep -E 'ray|WorkerDict|vllm|main_ppo|python -m verl' | grep -v grep | head -n 80 || true
}

proc_ok() {
  python - <<'PY'
import os
raise SystemExit(0 if os.path.exists("/proc/self") and os.path.exists("/proc/meminfo") and len(os.listdir("/proc")) > 10 else 1)
PY
}

echo "V37_TEARDOWN_STRESS_START rounds=${ROUNDS} $(date '+%F %T')" | tee "$LOG"
echo -e "round\tstatus\telapsed_s\tproc_ok\tproc_count\tgpu_used_mib_total\tstep_rows\tlast_step_s" > "$SUMMARY"

if [ "$RUN_CUDA_COMPAT_PREFLIGHT" = "1" ]; then
  bash /opt/tiger/TTRL/verl/examples/ttrl/check_cuda_compat_preflight.sh 2>&1 | tee -a "$LOG"
else
  echo "CUDA_COMPAT_PREFLIGHT_SKIPPED $(date '+%F %T')" | tee -a "$LOG"
fi
health_snapshot "before_stress" 2>&1 | tee -a "$LOG"
proc_ok

if [ -s "$LOCAL_MODEL/config.json" ]; then
  echo "MODEL_LOCAL_COPY_REUSE dst=${LOCAL_MODEL}" | tee -a "$LOG"
else
  echo "MODEL_LOCAL_COPY_START src=${SRC_MODEL} dst=${LOCAL_MODEL}" | tee -a "$LOG"
  rm -rf "${LOCAL_MODEL}.tmp"
  mkdir -p "${LOCAL_MODEL}.tmp"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${SRC_MODEL}/" "${LOCAL_MODEL}.tmp/" 2>&1 | tee -a "$LOG"
  else
    cp -a "${SRC_MODEL}/." "${LOCAL_MODEL}.tmp/"
  fi
  rm -rf "$LOCAL_MODEL"
  mv "${LOCAL_MODEL}.tmp" "$LOCAL_MODEL"
  echo "MODEL_LOCAL_COPY_DONE dst=${LOCAL_MODEL}" | tee -a "$LOG"
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET6}
export TRANSFORMERS_OFFLINE=1

for round in $(seq 1 "$ROUNDS"); do
  if ! proc_ok; then
    echo "V37_TEARDOWN_STRESS_PROC_BAD_BEFORE_ROUND round=${round}" | tee -a "$LOG"
    health_snapshot "before_round_${round}_bad" 2>&1 | tee -a "$LOG"
    exit 97
  fi

  RAY_DIR="/tmp/r37stress${round}"
  ROUND_LOG="/opt/tiger/TTRL/verl/${EXP_NAME}_round${round}.log"
  export RAY_TMPDIR="$RAY_DIR"
  export MASTER_PORT=$((29700 + round))
  export TORCHINDUCTOR_CACHE_DIR="/tmp/ti_${EXP_NAME}_${round}"
  rm -rf "$RAY_DIR"

  echo "V37_TEARDOWN_STRESS_ROUND_START round=${round} $(date '+%F %T')" | tee -a "$LOG"
  health_snapshot "before_round_${round}" 2>&1 | tee -a "$LOG"

  start_s=$(date +%s)
  set +e
  bash /opt/tiger/TTRL/verl/examples/ttrl/run_sps_rule_conf_weight_math_qwen3_4b_50step_8gpu.sh \
    actor_rollout_ref.model.path="$LOCAL_MODEL" \
    ttrl.sps_weight_floor=0.15 \
    ttrl.sps_clip_penalty=0.5 \
    ttrl.sps_weight_power=1.5 \
    ttrl.sps_base_logprob_source=ref \
    ttrl.sps_answer_sharpen_beta=2.0 \
    ttrl.sps_answer_sharpen_capacity=True \
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
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    trainer.validation_answer_selection_enable=False \
    ttrl.sps_reuse_rollout_log_probs_as_old=True \
    ttrl.sps_reuse_base_log_probs_as_ref=True \
    trainer.total_training_steps=1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    +ray_init.no_runtime_env="$RAY_NO_RUNTIME_ENV" \
    +ray_init.include_dashboard="$RAY_INCLUDE_DASHBOARD" \
    +ray_init.node_ip_address="$RAY_NODE_IP_ADDRESS" \
    trainer.experiment_name="math-qwen25_math_7b-v37-teardown-stress-round${round}" \
    2>&1 | tee "$ROUND_LOG"
  status=${PIPESTATUS[0]}
  set -e
  end_s=$(date +%s)
  elapsed=$((end_s - start_s))

  task_log=$(
    find "$RAY_DIR/ray/session_latest/logs" -maxdepth 1 -name 'worker-*-01000000-*.out' -type f -printf '%s %p\n' 2>/dev/null \
      | sort -nr \
      | awk '{print $2}' \
      | head -1
  )
  step_rows=0
  last_step_s="NA"
  if [ -n "$task_log" ] && [ -f "$task_log" ]; then
    step_rows=$(grep -a -c 'training/global_step' "$task_log" || true)
    last_step_s=$(python - "$task_log" <<'PY'
import re
import sys

last = "NA"
for line in open(sys.argv[1], "r", errors="ignore"):
    if "training/global_step" not in line or "timing_s/step" not in line:
        continue
    m = re.search(r"timing_s/step:([0-9.]+)", line)
    if m:
        last = m.group(1)
print(last)
PY
)
  fi

  if proc_ok; then
    proc_status="OK"
  else
    proc_status="BAD"
  fi
  proc_count=$(python - <<'PY'
import os
try:
    print(len(os.listdir("/proc")))
except Exception:
    print("ERR")
PY
)
  gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  echo -e "${round}\t${status}\t${elapsed}\t${proc_status}\t${proc_count}\t${gpu_used}\t${step_rows}\t${last_step_s}" | tee -a "$SUMMARY" | tee -a "$LOG"
  health_snapshot "after_round_${round}" 2>&1 | tee -a "$LOG"

  if [ "$proc_status" != "OK" ]; then
    echo "V37_TEARDOWN_STRESS_PROC_BAD_AFTER_ROUND round=${round}" | tee -a "$LOG"
    exit 98
  fi
done

echo "V37_TEARDOWN_STRESS_DONE $(date '+%F %T')" | tee -a "$LOG"
cat "$SUMMARY" | tee -a "$LOG"
