#!/bin/bash
set -euo pipefail

echo "RESTART_V12C_BEGIN $(date '+%F %T %Z')"
pkill -TERM -f "verl.trainer.main_ppo" || true
sleep 10
pkill -KILL -f "verl.trainer.main_ppo" || true
pkill -TERM -f "/tmp/ray_v12b" || true
pkill -TERM -f "ray::" || true
sleep 5
pkill -KILL -f "/tmp/ray_v12b" || true
pkill -KILL -f "ray::" || true

echo "RESTART_V12C_AFTER_KILL"
ps -eo pid,ppid,stat,etime,cmd | grep -E "verl.trainer.main_ppo|ray_v12b|ray::" | grep -v grep | head -80 || true

bash /opt/tiger/TTRL/verl/examples/ttrl/worker_run_sps_rule_conf_weight_floor015_clip05_power15_8_50step_v12c.sh

