# Qwen3-30B-A3B 两卡 B200 TTRL Smoke 记录

日期：2026-08-05

## 目标

先在只有 2 张 B200 的 worker 上把 `/tmp/Qwen3-30B-A3B-Base` 的 TTRL 主链路跑通，验证权重、vLLM、verl hybrid engine、rollout、ref logprob、actor update 是否能闭环。后续主力训练仍计划迁移到 8 卡 H100。

## 环境

- 工作目录：`/mlx_devbox/users/quyanyi/playground/TTRL/verl`
- Python venv：`/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200`
- 关键版本：
  - torch `2.11.0+cu130`
  - vLLM `0.24.0`
  - transformers `5.12.1`
- 模型路径：`/tmp/Qwen3-30B-A3B-Base`
- GPU：2 x NVIDIA B200，每卡约 183GB
- 日志目录：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs`

## 代码适配

本轮只做 infra/兼容性适配，不改 TTRL 训练语义。

1. `verl/utils/vllm_utils.py`
   - `patch_vllm_moe_model_weight_loader()` 兼容 vLLM 0.24 的 Qwen3 MoE 结构。
   - 旧逻辑假设 MoE 层存在 `mlp.experts.weight_loader`，vLLM 0.24 中对应对象变为 `MoERunner`，没有这个属性。
   - 新逻辑仅在旧结构存在时补 `w13_weight/w2_weight` loader，否则跳过。

2. `verl/workers/sharding_manager/fsdp_vllm.py`
   - 新增 `_get_vllm_tp_device_group()`。
   - 兼容旧 vLLM 的 `get_tensor_model_parallel_group()` 和 vLLM 0.24 的 `get_tp_group()`。

3. `run_records/qwen3_30b_a3b_b200_2gpu_ttrl_smoke.sh`
   - 新增环境变量覆盖：
     - `ACTOR_PARAM_OFFLOAD`
     - `ACTOR_OPTIMIZER_OFFLOAD`
     - `ACTOR_STRATEGY`
     - `ACTOR_FSDP_OFFLOAD_POLICY`
     - `KV_CACHE_MEMORY_BYTES`
   - 默认行为仍保持 FSDP1，不影响旧命令。

## vLLM 关键配置

能稳定初始化 Qwen3-30B-A3B 的 vLLM 配置：

```bash
TP_SIZE=2
actor_rollout_ref.rollout.enforce_eager=True
actor_rollout_ref.rollout.free_cache_engine=False
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=FLASH_ATTN
actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.moe_backend=triton
actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False
actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=536870912
```

已确认日志：

- vLLM 使用 `FLASH_ATTN` backend / FlashAttention v4。
- MoE backend 使用 `TRITON Unquantized MoE backend`。
- `free_cache_engine=True` 不可用，vLLM 0.24 sleep mode 报 `cumem allocator is not supported on current platform`。
- `vllm.third_party.deep_gemm` 会因 `libnvrtc.so.13` 缺失 warning，但当前强制 `moe_backend=triton`，不是 fatal。

## 失败路径

### FSDP1 + optimizer_offload 仍然 OOM

失败运行：

- `q3b2_smoke_t9.log`
- `q3b2_smoke_t10.log`

配置：

```bash
TP_SIZE=2
ACTOR_OPTIMIZER_OFFLOAD=True
ACTOR_PARAM_OFFLOAD=True/False
KV_CACHE_MEMORY_BYTES=1073741824 或 536870912
```

失败点：

```text
actor_rollout_update_actor -> Adam.step -> torch.optim.adam._init_group
state["exp_avg"] / state["exp_avg_sq"] = torch.zeros_like(...)
torch.OutOfMemoryError: Tried to allocate 1.16 GiB
```

原因判断：

- verl 当前 FSDP1 的 `optimizer_offload` helper 只会把“已经存在的 optimizer state”搬到 CPU。
- 但 Adam 第一次 `step()` 前 `optimizer.state` 为空，`exp_avg/exp_avg_sq` 仍按参数设备先在 GPU 上创建。
- 因此它不是完整的 CPU Adam / ZeRO-Offload，不能解决 30B 全参 Adam 首次 state 初始化峰值。

## 成功路径

成功运行：

```bash
RUN_TAG=q3b2_smoke_t11 \
ACTOR_STRATEGY=fsdp2 \
ACTOR_FSDP_OFFLOAD_POLICY=True \
ACTOR_PARAM_OFFLOAD=False \
ACTOR_OPTIMIZER_OFFLOAD=False \
TP_SIZE=2 \
KV_CACHE_MEMORY_BYTES=536870912 \
/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/qwen3_30b_a3b_b200_2gpu_ttrl_smoke.sh
```

日志：

```text
/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/q3b2_smoke_t11.log
```

结果：

- 完成 1 个 TTRL training step。
- 没有 OOM。
- GPU peak 从 FSDP1 路径的约 177GB/卡降到日志指标里的约 45GB reserved / 36GB allocated。
- CPU memory used 约 1374GB，说明 offload 压力主要转移到了 CPU 内存侧。

关键 step 指标：

```text
step: 1
global_seqlen/mean: 1084
prompt_length/mean: 30
response_length/mean: 512
response_length/clip_ratio: 1.0
timing_s/generate_sequences: 61.377
timing_s/gen: 62.878
timing_s/reward: 0.096
timing_s/old_log_prob: 4.104
timing_s/ref: 1.216
timing_s/update_actor: 256.548
timing_s/step: 325.003
perf/total_num_tokens: 2168
perf/throughput: 3.335
perf/max_memory_allocated_gb: 36.165
perf/max_memory_reserved_gb: 44.807
perf/cpu_memory_used_gb: 1374.232
```

## 当前结论

1. 两卡 B200 可以跑通 Qwen3-30B-A3B 的 TTRL 主链路，但必须使用 FSDP2 `offload_policy=True`。
2. FSDP1 的 `optimizer_offload=True` 对 30B 全参 Adam 不够，因为首次 Adam state 初始化仍发生在 GPU。
3. 当前两卡 B200 smoke 的 step 很慢，主要慢在 `update_actor=256.5s`，这是 train-time CPU offload 的代价，不代表 8 卡 H100 主力配置的最终速度。
4. 两卡 B200 更适合做兼容性 smoke，不适合做这个模型的高吞吐全参 RL 训练。
5. 后续上 8 卡 H100 时优先用更多 FSDP shard 降低 optimizer state/GPU 峰值；若仍要跑 30B 全参，必须优先验证 FSDP2 offload、ZeRO/CPU Adam 或更大卡数，而不是依赖 FSDP1 `optimizer_offload`。

## 下一步建议

1. 在 8 卡 H100 上先跑 1-step smoke：

```bash
RUN_TAG=qwen3_30b_a3b_h100_8gpu_smoke \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
N_GPUS=8 \
TP_SIZE=4 \
ACTOR_STRATEGY=fsdp \
ACTOR_PARAM_OFFLOAD=False \
ACTOR_OPTIMIZER_OFFLOAD=False \
KV_CACHE_MEMORY_BYTES=1073741824 \
/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/qwen3_30b_a3b_b200_2gpu_ttrl_smoke.sh
```

2. 如果 H100 FSDP1 仍在 Adam state 初始化 OOM，改：

```bash
ACTOR_STRATEGY=fsdp2
ACTOR_FSDP_OFFLOAD_POLICY=True
```

3. B200 两卡若继续调试，仅建议跑 smoke，不建议跑长训练；当前端到端 1 step 约 325s，训练效率不可接受。
