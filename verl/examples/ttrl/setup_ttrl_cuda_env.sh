#!/bin/bash
# Source this file from TTRL worker runners before starting Python/Ray.
# It keeps CUDA user-space libraries on the venv cu12.9 stack while leaving
# libcuda.so.1 to the driver/compat preflight.

ttrl_setup_cuda_env() {
  local exp_name="${1:?exp_name is required}"
  local log_file="${2:?log_file is required}"
  local venv_site="/opt/tiger/modelchef/.venv/lib/python3.11/site-packages"
  local cache_root="/tmp/ttrl_cache/${exp_name}"

  export TTRL_VENV_SITE="$venv_site"
  export TTRL_CACHE_ROOT="$cache_root"

  mkdir -p "$cache_root"/{tmp,hf,torch,xdg,vllm,triton,ti}

  export TMPDIR="$cache_root/tmp"
  export HF_HOME="$cache_root/hf"
  export HUGGINGFACE_HUB_CACHE="$cache_root/hf/hub"
  export TRANSFORMERS_CACHE="$cache_root/hf/transformers"
  export TORCH_HOME="$cache_root/torch"
  export XDG_CACHE_HOME="$cache_root/xdg"
  export VLLM_CACHE_ROOT="$cache_root/vllm"
  export TRITON_CACHE_DIR="$cache_root/triton"
  export TORCHINDUCTOR_CACHE_DIR="$cache_root/ti"

  local cuda_libs=(
    "$venv_site/nvidia/nvjitlink/lib"
    "$venv_site/nvidia/cublas/lib"
    "$venv_site/nvidia/cuda_runtime/lib"
    "$venv_site/nvidia/cudnn/lib"
    "$venv_site/nvidia/nccl/lib"
    "$venv_site/nvidia/cuda_nvrtc/lib"
    "$venv_site/nvidia/cusolver/lib"
    "$venv_site/nvidia/cusparse/lib"
    "$venv_site/nvidia/cuda_cupti/lib"
  )
  local prefix=""
  local d
  for d in "${cuda_libs[@]}"; do
    if [ -d "$d" ]; then
      if [ -z "$prefix" ]; then
        prefix="$d"
      else
        prefix="${prefix}:$d"
      fi
    fi
  done
  export LD_LIBRARY_PATH="${prefix}:${LD_LIBRARY_PATH:-}"

  {
    echo "TTRL_CUDA_ENV_SETUP_START $(date '+%F %T')"
    echo "TTRL_CACHE_ROOT ${TTRL_CACHE_ROOT}"
    echo "TMPDIR ${TMPDIR}"
    echo "HF_HOME ${HF_HOME}"
    echo "TORCH_HOME ${TORCH_HOME}"
    echo "XDG_CACHE_HOME ${XDG_CACHE_HOME}"
    echo "VLLM_CACHE_ROOT ${VLLM_CACHE_ROOT}"
    echo "TRITON_CACHE_DIR ${TRITON_CACHE_DIR}"
    echo "TORCHINDUCTOR_CACHE_DIR ${TORCHINDUCTOR_CACHE_DIR}"
    echo "LD_LIBRARY_PATH ${LD_LIBRARY_PATH}"
  } | tee -a "$log_file"
}

ttrl_record_cuda_preflight() {
  local log_file="${1:?log_file is required}"
  local do_gemm="${2:-1}"

  {
    echo "TTRL_CUDA_PREFLIGHT_START $(date '+%F %T')"
    test -e /proc/self && echo "PROC_SELF_OK" || echo "PROC_SELF_BAD"
    test -e /proc/meminfo && echo "PROC_MEMINFO_OK" || echo "PROC_MEMINFO_BAD"
    python - <<'PY'
import os
try:
    print("PROC_COUNT", len(os.listdir("/proc")))
except Exception as exc:
    print("PROC_COUNT_ERROR", repr(exc))
PY
    df -h / /opt/tiger /tmp
    /opt/tiger/modelchef/.venv/bin/python3 - <<'PY'
import importlib.metadata as md
import os
import sys
names = [
    "torch", "vllm", "ray", "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12",
    "nvidia-nccl-cu12", "nvidia-cuda-nvrtc-cu12",
    "nvidia-cusolver-cu12", "nvidia-cusparse-cu12",
    "nvidia-nvjitlink-cu12", "triton", "flash-attn", "flash-attn-3",
]
print("python", sys.version.replace("\n", " "))
for name in names:
    try:
        print(f"{name}=={md.version(name)}")
    except Exception:
        print(f"{name}=MISSING")
print("LD_LIBRARY_PATH=", os.environ.get("LD_LIBRARY_PATH", ""))
print("CUDA_HOME=", os.environ.get("CUDA_HOME", ""))
PY
    echo "LDD_LIBTORCH_CUDA_BEGIN"
    ldd /opt/tiger/modelchef/.venv/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so \
      | grep -Ei 'cublas|cuda|cudnn|nccl|nvJitLink|not found' || true
    echo "LDD_LIBTORCH_CUDA_END"
  } | tee -a "$log_file"

  if [ "$do_gemm" = "1" ]; then
    /opt/tiger/modelchef/.venv/bin/python3 - <<'PY' | tee -a "$log_file"
import os
import torch

print("TTRL_GEMM_SMOKE_BEGIN")
print("torch", torch.__version__, "cuda", torch.version.cuda)
a = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
b = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
c = a @ b
torch.cuda.synchronize()
print("GEMM_OK", tuple(c.shape), "pid", os.getpid())
keys = ("libcuda", "libcudart", "libcublas", "libcublasLt", "libcudnn", "libnccl", "libnvJitLink")
seen = []
with open(f"/proc/{os.getpid()}/maps", "r", errors="ignore") as f:
    for line in f:
        if any(key in line for key in keys):
            path = line.rstrip().split()[-1]
            if path not in seen:
                seen.append(path)
for path in seen:
    print("LOADED_LIB", path)
print("TTRL_GEMM_SMOKE_END")
PY
  fi

  {
    echo "TTRL_CUDA_PREFLIGHT_DONE $(date '+%F %T')"
    test -e /proc/self && echo "PROC_SELF_OK_AFTER_PREFLIGHT" || echo "PROC_SELF_BAD_AFTER_PREFLIGHT"
    test -e /proc/meminfo && echo "PROC_MEMINFO_OK_AFTER_PREFLIGHT" || echo "PROC_MEMINFO_BAD_AFTER_PREFLIGHT"
    python - <<'PY'
import os
try:
    print("PROC_COUNT_AFTER_PREFLIGHT", len(os.listdir("/proc")))
except Exception as exc:
    print("PROC_COUNT_AFTER_PREFLIGHT_ERROR", repr(exc))
PY
  } | tee -a "$log_file"
}
