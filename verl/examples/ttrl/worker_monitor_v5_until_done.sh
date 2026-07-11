#!/bin/bash
set +e

LOG_DIR=/tmp/ray_sps_conf_mix8/ray/session_latest/logs
TASK_LOG="$LOG_DIR/worker-a57a7f4499a23cf4ac741102eb5b50a79ad7b7d3a612cb813fdc289b-01000000-33084.out"
TASK_ERR="$LOG_DIR/worker-a57a7f4499a23cf4ac741102eb5b50a79ad7b7d3a612cb813fdc289b-01000000-33084.err"
last_step=""

echo "MONITOR_START $(date '+%F %T')"
while true; do
  now=$(date '+%F %T')
  latest_line=$(grep -a "training/global_step" "$TASK_LOG" 2>/dev/null | tail -1)
  latest_step=$(printf "%s\n" "$latest_line" | sed -nE 's/.*training\/global_step:([0-9.]+).*/\1/p')
  if [ -n "$latest_step" ] && [ "$latest_step" != "$last_step" ]; then
    short=$(printf "%s\n" "$latest_line" | sed -E '
      s/^step:([0-9]+).*train\/label_accuracy:([0-9.]+).*train\/reward_accuracy:([0-9.]+).*train\/majority_voting_reward:([0-9.]+).*train\/ground_truth_reward:([0-9.]+).*training\/global_step:([0-9.]+).*timing_s\/step:([0-9.]+).*perf\/throughput:([0-9.]+).*/step:\1 global_step:\6 label_acc:\2 reward_acc:\3 maj_reward:\4 gt_reward:\5 step_s:\7 throughput:\8/
    ')
    echo "$now $short"
    last_step="$latest_step"
  fi

  val=$(grep -a -E "val-core/MATH-TTT/acc/(mean@4|maj@4|best@4)|val-aux/MATH-TTT/acc/worst@4" "$TASK_LOG" 2>/dev/null | tail -20)
  if [ -n "$val" ]; then
    echo "VALIDATION_FOUND $now"
    printf "%s\n" "$val"
    break
  fi

  if ! pgrep -f "verl.trainer.main_ppo" >/dev/null; then
    echo "TRAIN_PROCESS_EXITED $now"
    tail -80 "$TASK_LOG" 2>/dev/null
    tail -80 "$TASK_ERR" 2>/dev/null
    break
  fi

  sleep 30
done
echo "MONITOR_DONE $(date '+%F %T')"
