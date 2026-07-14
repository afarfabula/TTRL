# TTRL Math500 7B 八卡 H100 Infra 优化进度

## 目标

- 模型不变：`/models/Qwen2.5-Math-7B`。
- 数据不变：`data/MATH-TTT/train.parquet` 和 `data/MATH-TTT/test.parquet`，math500 数据。
- 训练语义不变：`10 episodes / 150 steps`，`ttrl.enable=True`，`n_votes_per_prompt=64`，`n_samples_per_prompt=32`，`train_batch_size=32`，`max_prompt_length=1024`，`max_response_length=3072`，GRPO、reward 和 validation 逻辑不改。
- 只优化 infra/runtime：Ray/env、vLLM runtime、FSDP/offload、CUDA/NCCL、cache/CUDA graph、显存放置、进程清理、等效 micro-batch 实现。
- 成功标准：多个完整训练 step 的 steady-state 平均耗时 `<= 40s/step`，无 Ray/runtime/NCCL/OOM fatal error，GPU 利用率保持高位，然后再启动完整 10 episodes 训练。

## 2026-07-12 FlashAttention2 硬验证

代码改动：

- `verl/workers/fsdp_workers.py` 的 actor/ref `from_pretrained` 显式传入 `attn_implementation="flash_attention_2"`。
- actor/ref 模型构建后读取 `model.config._attn_implementation`，不是 `flash_attention_2` 则直接 `RuntimeError` 退出，避免 silent fallback。
- critic 构建路径也增加同样校验；当前 GRPO 不使用 critic，但后续若启用也不会 silent fallback。

验证日志：

- `/mnt/local/localcache00/quyanyi/ttrl/logs/ttrl_math500_7b_8gpu_verify_fa2_micro4_reuse_20260712_105737.log`

验证结果：

- 日志明确包含：`ref attention implementation verified: flash_attention_2`。
- 日志明确包含：`actor attention implementation verified: flash_attention_2`。
- 因此当前 actor/ref 的 HF attention backend 已硬性确认是 FlashAttention2；如果后续不是 FA2，任务会直接失败退出。

## 2026-07-12 Baseline

已停止的慢跑日志：

- `/mnt/local/localcache00/quyanyi/ttrl/logs/ttrl_math500_7b_8gpu_paper10ep_offload_kwargs_20260712_075332.log`
- 停止位置：`14/150`，没有最终测试分。
- 最近 10 个完整 step 平均：`165.5908s/step`。
- 最近 10 个 step 拆分：
  - `timing_s/gen`: `50.5859s`
  - `timing_s/generate_sequences`: `39.0836s`
  - `timing_s/reward`: `4.9271s`
  - `timing_s/old_log_prob`: `14.319s`
  - `timing_s/ref`: `27.0555s`
  - `timing_s/update_actor`: `63.6186s`
  - `perf/throughput`: `809.1342 tokens/s`
  - `perf/mfu/actor`: `0.102`
  - `perf/max_memory_reserved_gb`: `89.9194`
  - `response_length/mean`: `951.5416`

结论：

- 当前慢点不是 Ray runtime_env；Ray/env blocker 已经绕过，训练可以稳定跑 step。
- 主要慢在 heavy offload 和保守 vLLM runtime：
  - `update_actor` 约 64s，是最大单项耗时，actor MFU 只有约 0.10。
  - `gen` 约 51s，受 TTRL 64 vote、32 sample、3k response 上限影响。
  - `ref`/`old_log_prob` 合计约 41s，也受 micro-batch 和 offload 影响。
- 当前 benchmark 与原始 `examples/ttrl/Qwen2.5/math.sh` 相比，牺牲了吞吐：
  - 开启了 actor optimizer offload。
  - 开启了 actor activation offload。
  - 开启了 critic param/optimizer/activation offload。
  - 开启了 vLLM `free_cache_engine`。
  - 开启了 vLLM `enforce_eager`。
  - 降低了 vLLM `gpu_memory_utilization` 到 `0.72`。
  - `MICRO_BATCH_SIZE=1`，原始脚本为 `2`。

## 实验原则

- 不通过降低 `n_votes_per_prompt`、`n_samples_per_prompt`、`train_batch_size`、长度、episode、模型或数据来提速。
- 每轮 benchmark 至少等到能产出多个完整 `step:` metric 后再判断；如果 OOM 或 fatal error，要记录具体错误。
- 每轮只改少数 infra 变量，便于定位收益或风险。

## 下一轮实验矩阵

1. `bench_no_offload_orig_runtime`
   - 目标：恢复接近原始 TTRL math 脚本的高性能 runtime。
   - 保持语义参数不变。
   - 尝试：`ACTOR_OPTIMIZER_OFFLOAD=False`，`ACTOR_ACTIVATION_OFFLOAD=False`，`CRITIC_PARAM_OFFLOAD=False`，`CRITIC_OPTIMIZER_OFFLOAD=False`，`CRITIC_ACTIVATION_OFFLOAD=False`，`ROLLOUT_FREE_CACHE_ENGINE=False`，`ROLLOUT_ENFORCE_EAGER=False`，`GPU_MEMORY_UTILIZATION=0.8`，`MICRO_BATCH_SIZE=2`。
   - 预期：如果显存能撑住，`update_actor/ref/old_log_prob/gen` 都应明显下降；如果 OOM，再逐项回退。

2. `bench_no_offload_mem_safe`
   - 目标：保留 no-offload 训练，降低 vLLM 显存压力。
   - 尝试：`GPU_MEMORY_UTILIZATION=0.72` 或只开 `ROLLOUT_FREE_CACHE_ENGINE=True`。
   - 预期：如果第一轮 OOM，可判断是 vLLM KV/cache 压力还是训练状态压力。

3. `bench_minimal_offload`
   - 目标：只打开必要 offload，不回到 heavy offload。
   - 尝试顺序：只开 activation offload；只开 optimizer offload；critic 单独 offload。
   - 预期：找到能稳定跑且 `update_actor` 不超过 40s 的组合。

4. `bench_nccl_default`
   - 目标：确认单机 H100 上是否应去掉 socket fallback。
   - 尝试：`TTRL_FORCE_LOCAL_NCCL=0`，让 NCCL 使用默认选择。
   - 预期：如果通信更快且无 UCX/NCCL 错误，保留默认 NCCL。

## 当前状态

- 当前没有正在运行的 TTRL 训练。
- 8 张 H100 显存已释放到空闲状态。
- 下一步：生成 benchmark launcher，先跑 `bench_no_offload_orig_runtime`。

## 2026-07-12 实验 1：bench_no_offload_orig_runtime

日志：

- `/mnt/local/localcache00/quyanyi/ttrl/logs/ttrl_math500_7b_8gpu_bench_no_offload_orig_runtime_foreground_20260712_090250.log`

配置：

- 语义参数不变：`Qwen2.5-Math-7B`、math500、`10 episodes / 150 steps`、`train_batch_size=32`、`n_votes_per_prompt=64`、`n_samples_per_prompt=32`、`max_prompt_length=1024`、`max_response_length=3072`。
- Infra 参数：actor/critic offload 全关，`ROLLOUT_FREE_CACHE_ENGINE=False`，`ROLLOUT_ENFORCE_EAGER=False`，`GPU_MEMORY_UTILIZATION=0.8`，`MICRO_BATCH_SIZE=2`，`MINI_BATCH_SIZE=1`。

结果：

- 能成功进入 Ray 和 FSDP/vLLM 初始化。
- 能到 `Training Progress: 0/150`。
- 在第一个训练 step 的 `actor_rollout_update_actor` backward 阶段 OOM，没有产出完整 `step:` metric。
- 错误摘要：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 934.00 MiB. GPU 0 ... 856.56 MiB is free ... 75.54 GiB is allocated by PyTorch ...`

结论：

- 完全 no-offload 在当前语义参数下不可行；训练侧显存边界太紧。
- OOM 出现在 actor backward，不是 Ray runtime_env、NCCL 或 vLLM 初始化问题。
- 下一轮应保留 vLLM 高性能项 `free_cache_engine=False/enforce_eager=False`，但只开启最小必要的训练侧省显存项，优先尝试 actor activation offload 或降低 vLLM `gpu_memory_utilization`，不要直接恢复 heavy optimizer offload。

## 2026-07-12 实验 2：bench_actor_activation_offload

日志：

- `/mnt/local/localcache00/quyanyi/ttrl/logs/ttrl_math500_7b_8gpu_bench_actor_activation_offload_20260712_091050.log`

配置：

- 语义参数不变。
- Infra 参数：`ACTOR_ACTIVATION_OFFLOAD=True`，actor optimizer offload 关闭，critic offload 关闭，`ROLLOUT_FREE_CACHE_ENGINE=False`，`ROLLOUT_ENFORCE_EAGER=False`，`GPU_MEMORY_UTILIZATION=0.8`，`MICRO_BATCH_SIZE=2`，`MINI_BATCH_SIZE=1`。

结果：

- 能成功进入 `Training Progress: 0/150`。
- 在第一个训练 step 的 `actor_rollout_update_actor` 阶段仍 OOM，没有产出完整 `step:` metric。
- 错误摘要：FSDP reduce-scatter 时 `torch.OutOfMemoryError: Tried to allocate 520.00 MiB`，多张 GPU 仅剩 `70-384 MiB` 空闲。

结论：

- 只开 actor activation offload 不够；OOM 点是 FSDP backward/reduce-scatter 的额外 grad buffer。
- 当前 `GPU_MEMORY_UTILIZATION=0.8` 给 vLLM/KV/cache 留得过多，训练侧剩余显存太窄。
- 下一轮先把 `GPU_MEMORY_UTILIZATION` 降到 `0.72`，保持 optimizer offload 关闭和 vLLM 非 eager/非 free-cache，验证是否能释放足够显存并保留较高吞吐。

## 2026-07-12 实验 3：bench_actoff_gmem072

日志：

- `/mnt/local/localcache00/quyanyi/ttrl/logs/ttrl_math500_7b_8gpu_bench_actoff_gmem072_20260712_091857.log`

配置：

- 语义参数不变。
- Infra 参数：`ACTOR_ACTIVATION_OFFLOAD=True`，actor optimizer offload 关闭，critic offload 关闭，`ROLLOUT_FREE_CACHE_ENGINE=False`，`ROLLOUT_ENFORCE_EAGER=False`，`GPU_MEMORY_UTILIZATION=0.72`，`MICRO_BATCH_SIZE=2`，`MINI_BATCH_SIZE=1`。

结果：

- 能稳定跑过多个训练 step，说明降低 vLLM 显存占比后解决了首步 backward OOM。
- 已得到完整 step metrics：
  - step 1: `157.096s`
  - step 2: `144.983s`
  - step 3: `143.786s`
- 最近 3 个 step 拆分大致为：
  - `timing_s/gen`: `45-47s`
  - `timing_s/generate_sequences`: `35-36s`
  - `timing_s/reward`: `4.5-5.5s`
  - `timing_s/old_log_prob`: `10.6-11.2s`
  - `timing_s/ref`: `14.8-18.7s`
  - `timing_s/update_actor`: `64-69s`
  - `perf/max_memory_reserved_gb`: `76.805`
  - `perf/mfu/actor`: `0.09-0.105`

结论：

- 这组配置可跑，但仍远高于 `<=40s/step` 目标。
- 最大瓶颈仍是 `update_actor`，其次是 TTRL 投票生成和 ref/old logprob。
- 下一轮优先测试更大的 per-GPU micro batch，保持有效 batch/采样语义不变，目标是减少 actor update microbatch 数并提高 MFU；如果显存不足，再考虑更细粒度的 offload/显存释放。

## 2026-07-12 实验 4：bench_actoff_gmem072_micro4

日志：

- `/mnt/local/localcache00/quyanyi/ttrl/logs/ttrl_math500_7b_8gpu_bench_actoff_gmem072_micro4_20260712_093653.log`

配置：

- 语义参数不变。
- Infra 参数：实验 3 基础上把 `MICRO_BATCH_SIZE=2` 提到 `4`，其余保持：`ACTOR_ACTIVATION_OFFLOAD=True`，actor optimizer offload 关闭，critic offload 关闭，`ROLLOUT_FREE_CACHE_ENGINE=False`，`ROLLOUT_ENFORCE_EAGER=False`，`GPU_MEMORY_UTILIZATION=0.72`。

结果：

- 能稳定跑过多个训练 step，无 OOM/fatal error；进程由 benchmark `timeout` 停止，退出码 `124` 属于预期停止。
- 已得到完整 step metrics：近 3 step 为 `131.151s`、`119.182s`、`118.118s`。
- 相比实验 3，主要改善：`update_actor` 从 `64-69s` 降到 `46-51s`，`old_log_prob/ref` 也下降到约 `8-12s`。
- 近 3 个 step 拆分：\n  - `timing_s/gen`: `44.7-47.0s`\n  - `timing_s/generate_sequences`: `35.1-36.6s`\n  - `timing_s/reward`: `4.4-5.5s`\n  - `timing_s/old_log_prob`: `8.1-8.9s`\n  - `timing_s/ref`: `8.8-12.5s`\n  - `timing_s/update_actor`: `46.7-51.2s`\n  - `perf/max_memory_reserved_gb`: `76.805`\n  - `perf/mfu/actor`: `0.125-0.143`

结论：

- `MICRO_BATCH_SIZE=4` 是当前 `MINI_BATCH_SIZE=1`、8 卡、rollout.n=32 下不改变 PPO mini-batch 语义的最大本地 micro batch，收益明确但仍远不到 `<=40s/step`。
- 剩余主要成本：TTRL 生成约 `45s` 已单独超过目标，actor update 仍约 `47s`，old/ref 合计约 `17-21s`。
- 下一步不应继续只调 micro batch；需要代码级 runtime 优化，例如复用 vLLM 生成时 token logprob 跳过 actor old_log_prob 重算，或重构 actor/ref/logprob 的串行调度。

## 2026-07-12 实验 5：bench_micro4_reuse_rollout_logprobs

代码改动：

- `verl/trainer/config/ppo_trainer.yaml` 新增默认关闭的 `actor_rollout_ref.rollout.use_rollout_log_probs_as_old`。
- `verl/trainer/ppo/ray_trainer.py` 在该开关开启且 batch 内有 `rollout_log_probs` 时，直接用 `rollout_log_probs` 作为 `old_log_probs`，跳过 actor 的 `compute_log_prob`。
- 该路径要求 rollout 采样参数与 actor old-logprob 定义一致。本实验中 `temperature=1.0`、`top_p=1`、`top_k=-1`，并通过 `+actor_rollout_ref.rollout.logprobs=1` 让 vLLM 返回 token logprob。
- 默认配置不启用该优化，避免影响常规训练。

日志：

- 第一次失败：`/mnt/local/localcache00/quyanyi/ttrl/logs/ttrl_math500_7b_8gpu_bench_micro4_reuse_rollout_logprobs_20260712_095412.log`
  - 原因：跳过 old logprob 后未设置 `batch.meta_info["temperature"]`，actor update 触发 `KeyError: 'temperature'`。
- 第二次失败：`/mnt/local/localcache00/quyanyi/ttrl/logs/ttrl_math500_7b_8gpu_bench_micro4_reuse_rollout_logprobs_fix_20260712_100132.log`
  - 原因：Hydra struct 不允许直接覆写 `actor_rollout_ref.rollout.logprobs=1`，需要用 `+actor_rollout_ref.rollout.logprobs=1`。
- 有效 benchmark：`/mnt/local/localcache00/quyanyi/ttrl/logs/ttrl_math500_7b_8gpu_bench_micro4_reuse_rollout_logprobs_fix2_20260712_100212.log`

有效 benchmark 结果：

- 无 OOM/fatal error；进程由人工停止。
- 已得到 4 个完整 step metrics；最近 3 个 step 平均 `116.671s/step`，最后一个完整 step `114.221s`。
- `old_log_prob` 成功从 micro4 baseline 的约 `8s` 降到约 `0.02-0.04s`。
- 但 `gen` 从 micro4 baseline 的约 `44.7-47.0s` 变为约 `47.0-54.7s`，因为 vLLM 生成时需要返回 token logprob。
- `update_actor` 仍为约 `47-52s`。
- 日志中 `training/rollout_probs_diff_max/mean/std` 为 `0.000`，说明该开关生效后 `old_log_probs` 与 `rollout_log_probs` 完全一致。

结论：

- 该优化有效消除了 actor old-logprob 重算，但净收益只有数秒，不足以接近 `<=40s/step`。
- 目前硬下限由 TTRL 生成和 actor update 同时决定：生成本身已经约 `45-55s`，actor update 约 `47-52s`。
- 继续达成 40s 需要更大结构改动，例如改变并行拓扑 / 分离资源池 / 更快 rollout 后端 / 更深的 FSDP update 优化；单机 8 卡、当前 colocated FSDP+vLLM 串行路径很难达到。

## 2026-07-12 实验 6：bench_micro4_reuse_logprobs_fused_triton

日志：

- `/mnt/local/localcache00/quyanyi/ttrl/logs/ttrl_math500_7b_8gpu_bench_micro4_reuse_logprobs_fused_triton_20260712_101924.log`

配置：

- 语义参数不变。
- 在实验 5 基础上开启 `actor_rollout_ref.model.use_fused_kernels=True`，`actor_rollout_ref.model.fused_kernel_options.impl_backend=triton`。
- 日志确认 `Using Triton backend for fused kernels in Qwen2ForCausalLM` 和 `Actor use_fused_kernels=True`。

结果：

- 无 OOM/fatal error；进程由 benchmark `timeout` 停止，退出码 `124` 属于预期停止。
- 已得到 3 个完整 step metrics：`181.801s`、`124.478s`、`110.269s`。
- 第一步有明显 fused kernel 编译/热身惩罚；第三步比实验 5 的最后一个完整 step `114.221s` 略好。
- 第三步拆分：`gen=47.424s`，`old_log_prob=0.035s`，`ref=9.167s`，`update_actor=44.266s`，`step=110.269s`。

结论：

- fused triton 有小幅 steady-state 收益，主要降低 actor/ref 路径，但幅度仍远低于 40s 目标所需。
- 当前即使 old-logprob 基本归零，`gen` 约 `47s` 已经单项超过 40s，`update_actor` 仍约 `44s`，二者串行叠加使 40s 不可达。
- 后续若必须追 40s，需要架构级改变：把 rollout 与 actor update 解耦/流水并行，或使用更多 GPU/多节点资源池拆分 actor/ref/rollout，或更换可用的更高吞吐 rollout 后端。当前本机 venv 没有 `sglang`，单机 8 卡 colocated vLLM+FSDP 路径已经接近普通 infra 调参上限。

## 2026-07-12 实验 7：bench_sp4_micro16_reuse_logprobs

日志：

- `/mnt/local/localcache00/quyanyi/ttrl/logs/ttrl_math500_7b_8gpu_bench_sp4_micro16_reuse_logprobs_20260712_110705.log`

配置：

- 语义参数不变。
- 尝试 Ulysses sequence parallel：`actor_rollout_ref.actor.ulysses_sequence_parallel_size=4`，`actor_rollout_ref.ref.ulysses_sequence_parallel_size=4`。
- 保持 `MICRO_BATCH_SIZE=16`，`rollout_log_probs` 复用开关开启。
- Qwen2.5-Math-7B 的 `num_attention_heads=28`、`num_key_value_heads=4`，`sp_size=4` 通过 divisibility 检查。

结果：

- FA2 硬验证通过：日志包含 `ref attention implementation verified: flash_attention_2` 和 `actor attention implementation verified: flash_attention_2`。
- 能进入训练进度，但没有产出完整 `step:` metrics；人工停止，退出码 `143`。
- tqdm 进度显示：`1/150 [02:32]`，`2/150 [04:32]`，`3/150 [06:32]`，约 `127-153s/step`。
- 进程状态显示仍在 `actor_rollout_update_actor`，并未看到明显收益。

结论：

- SP4 在这套 colocated FSDP+vLLM 单机 8 卡链路上没有带来预期收益，反而初始化更慢，step 仍在 120s+。
- 后续不继续沿 SP4 方向投入，除非有多节点 / 更大 batch / 不同并行拓扑支持。
