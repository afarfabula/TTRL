#!/bin/bash
set -euo pipefail

echo "CUDA_COMPAT_PREFLIGHT_START $(date '+%F %T')"

if [ ! -e /proc/self ] || [ ! -e /proc/meminfo ]; then
  echo "CUDA_COMPAT_PREFLIGHT_PROC_BAD"
  exit 304
fi

DRV=$(
  grep -oE '[0-9]{3}\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version \
    | head -1
)
DRV_MAJOR=${DRV%%.*}
echo "CUDA_COMPAT_PREFLIGHT_DRIVER ${DRV}"

if [ "$DRV_MAJOR" -ge 580 ]; then
  echo "CUDA_COMPAT_PREFLIGHT_ACTION clear_compat"
  for f in /etc/ld.so.conf.d/00-compat-*.conf; do
    [ -e "$f" ] && sudo truncate -s 0 "$f"
  done
else
  echo "CUDA_COMPAT_PREFLIGHT_ACTION enable_cuda_12_9_compat"
  echo "/usr/local/cuda-12.9/compat" | sudo tee /etc/ld.so.conf.d/00-compat-active.conf
fi

sudo ldconfig
ldconfig -p | grep -E 'libcuda\.so\.1' | head -5 || true

/opt/tiger/modelchef/.venv/bin/python3 - <<'PY'
import ctypes
code = ctypes.CDLL("libcuda.so.1").cuInit(0)
print("cuInit:", code)
raise SystemExit(0 if code == 0 else code)
PY

echo "CUDA_COMPAT_PREFLIGHT_DONE $(date '+%F %T')"
