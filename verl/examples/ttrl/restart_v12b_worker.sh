#!/bin/bash
set -euo pipefail

echo "RESTART_V12B_BEGIN $(date '+%F %T %Z')"
echo "RESTART_V12B_BEFORE"
ps -eo pid,ppid,stat,etime,cmd | grep -E "verl.trainer.main_ppo|/tmp/ray_v12|ray_v12|ray::" | grep -v grep | head -120 || true

pkill -TERM -f "verl.trainer.main_ppo" || true
sleep 10
pkill -KILL -f "verl.trainer.main_ppo" || true
pkill -TERM -f "/tmp/ray_v12" || true
pkill -TERM -f "ray::" || true
sleep 5
pkill -KILL -f "/tmp/ray_v12" || true
pkill -KILL -f "ray::" || true

echo "RESTART_V12B_AFTER_KILL"
ps -eo pid,ppid,stat,etime,cmd | grep -E "verl.trainer.main_ppo|/tmp/ray_v12|ray_v12|ray::" | grep -v grep | head -120 || true

bash /opt/tiger/TTRL/verl/examples/ttrl/worker_run_sps_rule_conf_weight_floor015_clip05_power15_8_50step_v12b.sh

