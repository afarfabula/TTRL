#!/bin/bash
set -euo pipefail

echo "HEALTH_START $(date '+%F %T')"
hostname

echo "PROC_CHECK"
ls -ld /proc /proc/self /proc/meminfo

echo "PY_CHECK"
/opt/tiger/modelchef/.venv/bin/python -c 'import psutil, torch; print("pids", len(psutil.pids())); print("cuda", torch.cuda.is_available(), torch.cuda.device_count()); x=torch.ones(1, device="cuda"); print("cuda_tensor", x.item())'

echo "RAY_CHECK"
/opt/tiger/modelchef/.venv/bin/python -c 'import ray; ray.init(local_mode=True, ignore_reinit_error=True); print("ray_ok"); ray.shutdown()'

echo "NVIDIA_CHECK"
nvidia-smi -L

echo "HEALTH_DONE $(date '+%F %T')"
