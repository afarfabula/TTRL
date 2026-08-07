#!/usr/bin/env bash
set -u

for p in \
  /mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python \
  /opt/tiger/modelchef/.venv/bin/python \
  /usr/bin/python3 \
  /bin/python \
  /opt/tiger/ss_lib/python_package/bin/python
do
  if [ -x "$p" ]; then
    echo "PY=$p"
    "$p" -c 'import importlib.util, sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch_err", repr(e))
for m in ["ray", "vllm", "flash_attn"]:
    try:
        mod = __import__(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m + "_err", repr(e))
'
  fi
done
