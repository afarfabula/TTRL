import sys

import torch

print("python", sys.executable)
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))

b200_site = "/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/lib/python3.11/site-packages"
if b200_site not in sys.path:
    sys.path.append(b200_site)

for name in ["ray", "vllm", "flash_attn", "transformers", "verl"]:
    try:
        module = __import__(name)
        print(name, getattr(module, "__version__", "ok"), getattr(module, "__file__", ""))
    except Exception as exc:
        print(name + "_err", repr(exc))
