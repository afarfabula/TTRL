#!/bin/bash
set -euo pipefail

SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
RUN_TAG="${RUN_TAG:-v33_seed_sweep_10x_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/opt/tiger/TTRL/verl/${RUN_TAG}}"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
SUMMARY_TXT="${OUT_DIR}/summary.txt"
SRC_MODEL=/mnt/hdfs/models/qwen3_4b
LOCAL_MODEL="${LOCAL_MODEL:-/tmp/qwen3_4b_local_v33_seed_sweep}"
BASE_PORT="${BASE_PORT:-29700}"

mkdir -p "$OUT_DIR"

echo "V33_SEED_SWEEP_START $(date '+%F %T') tag=${RUN_TAG} seeds=${SEEDS}" | tee "${OUT_DIR}/sweep.log"
echo "HOST $(hostname)" | tee -a "${OUT_DIR}/sweep.log"
ls -ld /proc /proc/self /proc/meminfo | tee -a "${OUT_DIR}/sweep.log"
python - <<'PY' | tee -a "${OUT_DIR}/sweep.log"
import os
print("PROC_COUNT_BEFORE", len(os.listdir("/proc")))
PY
df -h /opt/tiger /tmp | tee -a "${OUT_DIR}/sweep.log"

bash /opt/tiger/TTRL/verl/examples/ttrl/check_cuda_compat_preflight.sh 2>&1 | tee -a "${OUT_DIR}/sweep.log"
nvidia-smi -L | tee -a "${OUT_DIR}/sweep.log"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader | tee -a "${OUT_DIR}/sweep.log"

if [ ! -d "$LOCAL_MODEL" ] || [ ! -f "$LOCAL_MODEL/config.json" ]; then
  echo "MODEL_LOCAL_COPY_START $(date '+%F %T') src=${SRC_MODEL} dst=${LOCAL_MODEL}" | tee -a "${OUT_DIR}/sweep.log"
  rm -rf "${LOCAL_MODEL}.tmp"
  mkdir -p "${LOCAL_MODEL}.tmp"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${SRC_MODEL}/" "${LOCAL_MODEL}.tmp/" 2>&1 | tee -a "${OUT_DIR}/sweep.log"
  else
    cp -a "${SRC_MODEL}/." "${LOCAL_MODEL}.tmp/"
  fi
  find "${LOCAL_MODEL}.tmp" -maxdepth 1 -type f -printf '%f %s\n' | sort | tee -a "${OUT_DIR}/sweep.log"
  rm -rf "$LOCAL_MODEL"
  mv "${LOCAL_MODEL}.tmp" "$LOCAL_MODEL"
  echo "MODEL_LOCAL_COPY_DONE $(date '+%F %T')" | tee -a "${OUT_DIR}/sweep.log"
else
  echo "MODEL_LOCAL_COPY_REUSE $(date '+%F %T') dst=${LOCAL_MODEL}" | tee -a "${OUT_DIR}/sweep.log"
fi
df -h /opt/tiger /tmp | tee -a "${OUT_DIR}/sweep.log"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET6
export TRANSFORMERS_OFFLINE=1

echo "seed,status,mean4,best4,maj4,raw_mean32,raw_best32,raw_maj32,task_log,log,metrics,proc_health" > "$SUMMARY_CSV"

run_one_seed() {
  local seed="$1"
  local port=$((BASE_PORT + seed))
  local name="sps_efficient_ttrl_qwen3_4b_20step_v33_seed${seed}"
  local log="${OUT_DIR}/${name}.log"
  local ray_log_snapshot="${OUT_DIR}/${name}_ray_taskrunner.log"
  local metrics_snapshot="${OUT_DIR}/${name}_metrics.txt"
  local throughput_summary="${OUT_DIR}/${name}_throughput_summary.txt"
  local proc_health="${OUT_DIR}/${name}_proc_health.txt"
  local ray_dir="/tmp/r3s${seed}"

  echo "V33_SEED_START seed=${seed} $(date '+%F %T')" | tee -a "${OUT_DIR}/sweep.log" | tee "$log"
  test -e /proc/self && echo PROC_SELF_OK_BEFORE || echo PROC_SELF_BAD_BEFORE | tee -a "$log"
  test -e /proc/meminfo && echo PROC_MEMINFO_OK_BEFORE || echo PROC_MEMINFO_BAD_BEFORE | tee -a "$log"
  python - <<'PY' | tee -a "$log"
import os
print("PROC_COUNT_BEFORE_SEED", len(os.listdir("/proc")))
PY
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader | tee -a "$log"

  export RAY_TMPDIR="$ray_dir"
  export MASTER_PORT="$port"
  export TORCHINDUCTOR_CACHE_DIR="/tmp/ti_qwen3_4b_v33_seed_${seed}"
  rm -rf "$ray_dir"
  rm -f "$ray_log_snapshot" "$metrics_snapshot" "$throughput_summary" "$proc_health"

  set +e
  bash /opt/tiger/TTRL/verl/examples/ttrl/run_sps_rule_conf_weight_math_qwen3_4b_50step_8gpu.sh \
    actor_rollout_ref.model.path="$LOCAL_MODEL" \
    +data.seed="$seed" \
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
    ttrl.sps_reuse_rollout_log_probs_as_old=True \
    ttrl.sps_reuse_base_log_probs_as_ref=True \
    trainer.total_training_steps=20 \
    trainer.test_freq=20 \
    trainer.val_before_train=False \
    trainer.experiment_name="math-qwen3_4b-efficient-ttrl-v33-20step-seed${seed}" \
    2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  set -e

  local task_log=""
  task_log=$(
    grep -R -l -a "Final validation metrics\|training/global_step:20\|val-core/MATH-TTT/acc/mean@4" \
      "$ray_dir/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
  )
  if [ -z "$task_log" ]; then
    task_log=$(
      grep -R -l -a "Training from scratch\|Total training steps: 20" \
        "$ray_dir/ray/session_latest/logs" 2>/dev/null | sort | head -1 || true
    )
  fi

  if [ -n "$task_log" ] && [ -f "$task_log" ]; then
    cp "$task_log" "$ray_log_snapshot" || true
    {
      echo "SEED $seed"
      echo "TASK_LOG $task_log"
      grep -a -E "training/global_step|train/sps/|train/label_accuracy|train/reward_accuracy|train/majority_voting_reward|train/ground_truth_reward|train/pass@32|train/majority_ratio|response_length/clip_ratio|timing_s/|perf/total_num_tokens|perf/throughput|val-core/MATH-TTT/acc|val-raw/MATH-TTT/acc|val-aux/MATH-TTT/acc|Final validation metrics|global_step:20" "$task_log" || true
    } > "$metrics_snapshot"

    python - "$task_log" "$throughput_summary" <<'PY' || true
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
  else
    echo "SEED $seed TASK_LOG_MISSING" > "$metrics_snapshot"
    echo "NO_TASK_LOG" > "$throughput_summary"
  fi

  {
    echo "V33_SEED_STATUS seed=${seed} status=${status} $(date '+%F %T')"
    test -e /proc/self && echo PROC_SELF_OK_AFTER || echo PROC_SELF_BAD_AFTER
    test -e /proc/meminfo && echo PROC_MEMINFO_OK_AFTER || echo PROC_MEMINFO_BAD_AFTER
    python - <<'PY'
import os
print("PROC_COUNT_AFTER_SEED", len(os.listdir("/proc")))
PY
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
  } | tee "$proc_health" | tee -a "$log"

  python - "$seed" "$status" "$task_log" "$log" "$metrics_snapshot" "$proc_health" "$SUMMARY_CSV" <<'PY'
import csv, math, re, sys
seed, status, task_log, log, metrics, proc_health, summary = sys.argv[1:]
keys = {
    "mean4": r"val-core/MATH-TTT/acc/mean@4:([0-9.]+)",
    "best4": r"val-core/MATH-TTT/acc/best@4/mean:([0-9.]+)",
    "maj4": r"val-core/MATH-TTT/acc/maj@4/mean:([0-9.]+)",
    "raw_mean32": r"val-raw/MATH-TTT/acc/mean@32:([0-9.]+)",
    "raw_best32": r"val-raw/MATH-TTT/acc/best@32/mean:([0-9.]+)",
    "raw_maj32": r"val-raw/MATH-TTT/acc/maj@32/mean:([0-9.]+)",
}
text = ""
try:
    if task_log:
        text = open(task_log, "r", errors="ignore").read()
except OSError:
    pass
vals = {}
for key, pat in keys.items():
    m = list(re.finditer(pat, text))
    vals[key] = m[-1].group(1) if m else ""
with open(summary, "a", newline="") as f:
    csv.writer(f).writerow([
        seed, status, vals["mean4"], vals["best4"], vals["maj4"],
        vals["raw_mean32"], vals["raw_best32"], vals["raw_maj32"],
        task_log, log, metrics, proc_health,
    ])
PY

  echo "V33_SEED_DONE seed=${seed} status=${status} $(date '+%F %T')" | tee -a "${OUT_DIR}/sweep.log" | tee -a "$log"
  return "$status"
}

overall_status=0
for seed in $SEEDS; do
  if ! run_one_seed "$seed"; then
    overall_status=1
    echo "V33_SEED_FAILED seed=${seed}; continuing sweep" | tee -a "${OUT_DIR}/sweep.log"
  fi
done

python - "$SUMMARY_CSV" "$SUMMARY_TXT" <<'PY'
import csv, statistics, sys
summary_csv, summary_txt = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(summary_csv)))
ok = [r for r in rows if r["status"] == "0" and r["mean4"]]
vals = [float(r["mean4"]) for r in ok]
with open(summary_txt, "w") as f:
    f.write(f"rows={len(rows)} ok={len(ok)}\n")
    if vals:
        f.write(f"mean4_mean={statistics.mean(vals):.12f}\n")
        f.write(f"mean4_min={min(vals):.12f}\n")
        f.write(f"mean4_max={max(vals):.12f}\n")
        if len(vals) > 1:
            f.write(f"mean4_stdev={statistics.stdev(vals):.12f}\n")
    for r in rows:
        f.write(
            f"seed={r['seed']} status={r['status']} mean4={r['mean4']} "
            f"raw_mean32={r['raw_mean32']} raw_best32={r['raw_best32']} raw_maj32={r['raw_maj32']}\n"
        )
PY

cat "$SUMMARY_TXT" | tee -a "${OUT_DIR}/sweep.log"
echo "V33_SEED_SWEEP_DONE status=${overall_status} out_dir=${OUT_DIR} $(date '+%F %T')" | tee -a "${OUT_DIR}/sweep.log"
exit "$overall_status"
