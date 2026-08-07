# B200 TTRL Environment Usage Notes

This file records the reusable environment used for the B200 TTRL Math500 runs. It is intended for future algorithm experiments on this machine that need to share the same Python, CUDA, Ray, vLLM, and verl setup.

## Persistent Environment

Use this Python environment:

```bash
/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python
```

Current verified versions:

```text
python: 3.11.2
torch: 2.11.0+cu130
cuda reported by torch: 13.0
ray: 2.56.0
transformers: 5.12.1
vllm: 0.24.0
flash_attn Python import: not installed
```

Important note:

- `flash_attn` is not importable as a Python package in this environment.
- The successful run used vLLM with `FLASH_ATTN` backend and fused Triton kernels. Do not use `import flash_attn` as the only test for whether the working inference/training path is using optimized attention.

Environment size:

```text
/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200: about 13G
```

The environment is intentionally under `/mlx_devbox/users/quyanyi/playground`, not under `/tmp`, so it should survive worker/runtime cleanup.

## Repo And Model Paths

Main repo:

```bash
/mlx_devbox/users/quyanyi/playground/TTRL
```

Model:

```bash
/models/Qwen2.5-Math-7B
```

Math500/TTRL data:

```bash
/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT/train.parquet
/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT/test.parquet
```

Useful launchers:

```bash
/mlx_devbox/users/quyanyi/playground/TTRL/verl/examples/ttrl/Qwen2.5-Math/math500_7b_8gpu.sh
/mlx_devbox/users/quyanyi/playground/TTRL/verl/examples/ttrl/Qwen2.5-Math/math500_local.sh
```

Paper-style run script kept for reference:

```bash
/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_math500_paper_b32_20260714_101500.sh
```

Experiment record:

```bash
/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_math500_paper_b32_20260714_101500/
```

## Runtime Directories

Use short `/tmp` runtime paths for Ray to avoid Unix socket path length errors:

```bash
export TTRL_RUNTIME_DIR=/tmp/pb32
export RAY_TMPDIR=/tmp/pb32/ray
```

Recommended runtime output layout:

```bash
/tmp/ttrl_b200/logs
/tmp/ttrl_b200/checkpoints
```

Current completed paper-style run artifacts:

```bash
/tmp/ttrl_b200/logs/ttrl_math500_paper_b32_20260714_101500.log
/tmp/ttrl_b200/checkpoints/ttrl_math500_paper_b32_20260714_101500/global_step_150
```

The checkpoint/log area is large and should stay out of git:

```text
/tmp/ttrl_b200: about 86G after the paper-style run
global_step_150 checkpoint: about 86G
raw log: 121092624 bytes
```

At the end of the run, `/tmp` had:

```text
3.5T total, 377G used, 2.9T available, 12% used
```

## Common Environment Variables

The launcher sets most variables, but these are the important ones for reuse:

```bash
export PYTHON_BIN=/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python
export BACKBONE_PATH=/models/Qwen2.5-Math-7B
export DATA_LOCAL_DIR=/mlx_devbox/users/quyanyi/playground/TTRL/verl/data
export TTRL_RUNTIME_DIR=/tmp/pb32
export OUTPUT_DIR=/tmp/ttrl_b200/checkpoints/<run_id>
export LOGGER=console
```

CUDA/vLLM settings in the wrapper:

```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/lib/python3.11/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_USE_V1
```

NCCL/Ray settings used by the working single-node B200 path:

```bash
export TTRL_FORCE_LOCAL_NCCL=1
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="=eth0"
export NCCL_SOCKET_FAMILY=AF_INET6
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=true
```

## Paper-Style Math500 Baseline Command

The successful paper-style run used the persisted script:

```bash
bash /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_math500_paper_b32_20260714_101500.sh
```

Core training semantics:

```text
data.train_batch_size=32
trainer.total_epochs=10
trainer.total_training_steps=None
computed total training steps=150
ttrl.n_votes_per_prompt=64
ttrl.n_samples_per_prompt=32
max_prompt_length=1024
max_response_length=3072
rollout.temperature=1.0
val_kwargs.n=16
val_kwargs.temperature=0.6
val_kwargs.top_p=0.95
actor ppo_mini_batch_size=1
actor/rollout/ref micro_batch_size_per_gpu=2
ref param_offload=True
trainer.test_freq=20
trainer.save_freq=2000000
trainer.resume_mode=disable
trainer.val_before_train=False
```

This run produced:

```text
step20   mean@16=0.7630  maj@16=0.825  best@16=0.912
step40   mean@16=0.8010  maj@16=0.849  best@16=0.911
step60   mean@16=0.8100  maj@16=0.846  best@16=0.899
step80   mean@16=0.8240  maj@16=0.852  best@16=0.889
step100  mean@16=0.8260  maj@16=0.853  best@16=0.881
step120  mean@16=0.8270  maj@16=0.853  best@16=0.884
step140  mean@16=0.8300  maj@16=0.853  best@16=0.887
step150  mean@16=0.8275  maj@16=0.853  best@16=0.885
```

## 80-Step Iteration Baseline

For future algorithm iteration, using the first `80` training steps is a reasonable development loop because it already reaches near-paper performance while avoiding the full 150-step runtime.

From the completed run:

```text
tqdm progress at 79/150: 2:28:30<2:09:33, 109.49s/it
tqdm progress at 80/150: 2:35:09<3:48:48, 196.12s/it
```

Interpretation:

- Reaching the end of step 79, before the step80 validation is fully counted, took about `2h28m30s`.
- Reaching progress `80/150` took about `2h35m09s`.
- Step 80 itself included validation and had `timing_s/step=398.215s`.
- The validation part of step 80 was `timing_s/testing=289.366s`.
- A typical nearby non-validation step was around `109-113s`.

Step 80 metrics:

```text
training/global_step=80
timing_s/step=398.215
timing_s/gen=27.230
timing_s/reward=17.594
timing_s/ref=11.242
timing_s/update_actor=34.341
timing_s/testing=289.366
perf/total_num_tokens=772885
perf/throughput=242.609
val-core/math/acc/mean@16=0.824
val-core/math/acc/maj@16/mean=0.852
val-core/math/acc/best@16/mean=0.889
```

If an algorithm loop only needs an 80-step checkpoint-like signal, budget roughly:

```text
2.5 to 2.6 hours including the step80 validation
```

If validation is disabled until the end or moved to a separate eval job, the 80-step loop can be shorter by roughly the validation cost at step80, about `4.8 minutes` in this run.

## Health Checks Before Starting A New Run

Check GPU state:

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
```

Check disk:

```bash
df -h /tmp /mlx_devbox/users/quyanyi/playground
```

Check model:

```bash
/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python \
  /mlx_devbox/users/quyanyi/playground/TTRL/verl/scripts/check_hf_model_ready.py \
  /models/Qwen2.5-Math-7B
```

Check process conflicts:

```bash
ps -ef | rg 'verl.trainer.main_ppo|ray::TaskRunner|raylet|ttrl_math500'
```

## Monitoring Commands

Parse latest progress and validation:

```bash
python - <<'PY'
import pathlib, re, time
log = pathlib.Path('/tmp/ttrl_b200/logs/<run_id>.log')
text = log.read_text(errors='ignore')
progress = list(re.finditer(r'Training Progress:\s+[^\r\n]*?(\d+)/(\d+)\s+\[([^\]]+)\]', text))
if progress:
    m = progress[-1]
    print('progress', f'{m.group(1)}/{m.group(2)}', m.group(3))
for key in ['val-core/math/acc/mean@16', 'val-core/math/acc/maj@16/mean', 'val-core/math/acc/best@16/mean']:
    vals = re.findall(re.escape(key) + r"(?:': np\.float64\(|:)(-?\d+(?:\.\d+)?)", text)
    if vals:
        print(key, vals[-1])
PY
```

Look for fatal markers:

```bash
rg -n 'RayTaskError|OutOfMemoryError|CUDA out of memory|No space left|Error executing job|RuntimeError:' /tmp/ttrl_b200/logs/<run_id>.log
```

## Practical Notes For Future Algorithms

- Keep dependencies in `/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200`, not `/tmp`.
- Keep transient Ray/runtime/log/checkpoint data in `/tmp` with short path names.
- Prefer a short `TTRL_RUNTIME_DIR`, for example `/tmp/pb32`, because Ray Unix socket paths can fail if too long.
- Do not use the raw `flash_attn` import as the sole attention check in this environment.
- For paper-style TTRL Math500, `batch_size=32` and `10` epochs naturally computed `150` total steps here.
- For fast algorithm iteration, `80` steps is a good default budget: it reached `mean@16=82.4%` in this run.
- For final reporting, always distinguish:
  - `val-core/math/acc/mean@16`
  - `val-core/math/acc/maj@16/mean`
  - `val-core/math/acc/best@16/mean`
