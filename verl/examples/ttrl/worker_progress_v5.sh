#!/bin/bash
set +e

LOG_DIR=/tmp/ray_sps_conf_mix8/ray/session_latest/logs
TASK_LOG="$LOG_DIR/worker-a57a7f4499a23cf4ac741102eb5b50a79ad7b7d3a612cb813fdc289b-01000000-33084.out"
TASK_ERR="$LOG_DIR/worker-a57a7f4499a23cf4ac741102eb5b50a79ad7b7d3a612cb813fdc289b-01000000-33084.err"

echo "PROGRESS_START $(date '+%F %T')"
echo "GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "WORKER"
ps -eo pid,ppid,stat,pcpu,pmem,etime,comm,args | \
  grep -E "main_ppo|TaskRunner|WorkerDict.actor|WorkerDict.ref" | grep -v grep | \
  awk '{print $1, $2, $3, $4, $5, $6, $7}' | sed -n '1,16p'
echo "TASK_LOG_STAT"
stat -c "%y %s %n" "$TASK_LOG" 2>/dev/null
echo "LATEST_STEPS"
grep -a "training/global_step" "$TASK_LOG" 2>/dev/null | tail -5 | sed -E '
  s/^step:([0-9]+).*train\/label_accuracy:([0-9.]+).*train\/reward_accuracy:([0-9.]+).*train\/majority_voting_reward:([0-9.]+).*train\/ground_truth_reward:([0-9.]+).*training\/global_step:([0-9.]+).*timing_s\/step:([0-9.]+).*perf\/throughput:([0-9.]+).*/step:\1 global_step:\6 label_acc:\2 reward_acc:\3 maj_reward:\4 gt_reward:\5 step_s:\7 throughput:\8/
'
echo "MAJORITY_MIX_METRIC"
grep -a "sps_majority_reward" "$TASK_LOG" 2>/dev/null | tail -5
echo "LATEST_VALIDATION"
grep -a -E "val-core/MATH-TTT/acc/(mean@4|maj@4|best@4)|val-aux/MATH-TTT/acc/worst@4|final|Validation|global_step:50" "$TASK_LOG" 2>/dev/null | tail -40
echo "TRAINING_PROGRESS"
grep -a "Training Progress" "$TASK_ERR" 2>/dev/null | tail -5
echo "ERRORS"
find "$LOG_DIR" -maxdepth 1 -name 'worker-*-01000000-*.err' -mmin -5 -print0 2>/dev/null | \
  xargs -0 grep -a -E "Traceback|Error|Exception|FAILED|NCCL|CUDA" 2>/dev/null | tail -20
echo "PROGRESS_DONE $(date '+%F %T')"
