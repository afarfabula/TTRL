# Chunk-Level Search-State TTRL 实验日志

## 目标

本轮实验从 `chunk_level_search_state_ttrl_24h_goal.md` 出发，验证一个比 majority-vote reward shaping 更有方法变化的 TTRL 方向：

> 先采样完整 on-policy rollout，再从真实 rollout 中截取中间 state，对该 state 重采 next chunk，用 future probe 得到 search-improved chunk distribution，并蒸馏回 actor。

## 当前 MVP 设计

第一版不做逐 chunk 在线更新，也不主推 GRPO。训练流程为：

1. 按当前 TTRL 语义先对每个 prompt 采完整 rollout。
2. 从完整 rollout 的 chunk boundary 构造 state：`query + response[:t]`。
3. 每个 state 重采 `K=8` 个 next chunk，chunk 长度 `256`。
4. 对 `state + chunk` 再 probe 到可用剩余长度，用现有 math rule reward 得到终局 success score。
5. 将 probe score 写成 chunk-level boxed reward。
6. actor 优先使用 PowerFlow trajectory-balance loss，在 chunk response 上更新局部 continuation policy。

PowerFlow 主路径：

```text
delta = logZ(state) + avg_log pi_theta(chunk | state)
        - beta * (avg_log pi_ref(chunk | state) + (boxed_reward - 1) / 2)
L = delta^2
```

保留一个 weighted NLL fallback / ablation：

```text
q_i = normalize((score_i + eps) ** alpha)
L = -sum_i q_i * log pi_theta(chunk_i | state)
```

这版的主叙事是 PowerFlow-style search-state improvement，而不是 chunk-GRPO。

## 已落地的代码改动

- `verl/trainer/config/ppo_trainer_ttrl.yaml`
  - 增加 `ttrl.chunk_state_*` 配置开关，默认关闭。
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
  - 支持 per-call `max_tokens` 后按实际短生成长度 padding response，避免 chunk/probe 被全局 `MAX_RESPONSE_LENGTH=3072` 拖慢。
- `verl/workers/actor/dp_actor.py`
  - 增加 `actor_rollout_ref.actor.chunk_weighted_nll_enable=True` 分支。
  - 支持 `chunk_weights`，只对 response/chunk token 做 weighted NLL。
  - weighted NLL 保留为 fallback；主实验优先使用已有 `powerflow_enable=True` 分支。
- `verl/trainer/ppo/ray_trainer.py`
  - 增加 chunk-state helper：
    - 从 full rollout 构造 state prompt。
    - 对 state 重采 next chunk。
    - 对 `state + chunk` 做 probe。
    - 用 reward tensor 求 probe score。
    - 将 probe score 写入 chunk-level `boxed_reward`。
    - 计算 chunk actor batch 的 `ref_log_prob`，再调用 PowerFlow actor update。
- `verl/run_records/ttrl_chunk_state_powerflow_b32_r32_v64_20step_20260731.sh`
  - 20-step pilot 启动脚本。

## Pilot 配置

```text
model=/models/Qwen2.5-Math-7B
data=MATH-TTT
full rollout: DATA_TRAIN_BATCH_SIZE=32, N_VOTES_PER_PROMPT=64, N_SAMPLES_PER_PROMPT=32
validation: VAL_N=16, TEST_FREQ=20
chunk_state_candidates=8
chunk_state_chunk_size=256
chunk_state_states_per_prompt=1
chunk_state_max_prefix_tokens=1024
chunk_state_boundaries=[0,256,512,768,1024]
chunk_state_alpha=2.0
chunk_state_eps=0.05
actor dynamic batch=False
actor powerflow_enable=True
actor powerflow_use_boxed_reward=True
actor powerflow_beta_coef=4.0
actor chunk weighted NLL=False
actor KL loss=False
```

## 关键监控指标

训练指标：

- `chunk_state/num_states`
- `chunk_state/num_candidates`
- `chunk_state/num_actor_samples`
- `chunk_state/positive_ratio`
- `chunk_state/informative_ratio`
- `chunk_state/target_entropy`
- `actor/powerflow_loss`
- `actor/boxed_reward/mean`
- `actor/log_z`
- `timing_s/chunk_state_chunks`
- `timing_s/chunk_state_probe`
- `timing_s/update_actor`

Fallback / ablation 指标：

- `actor/chunk_weighted_nll_loss`
- `actor/chunk_weight_mean`

验证指标：

- `val-core/math/acc/mean@16`
- `val-core/math/acc/maj@16/mean`
- `val-core/math/acc/best@16/mean`

## 20-Step Gate

当前对齐基线：

```text
MV step20: mean@16 ~= 0.760, maj@16 ~= 0.820, best@16 ~= 0.901
soft0.02 step20: mean@16 ~= 0.757, maj@16 ~= 0.821, best@16 ~= 0.911
```

判断规则：

- 如果 step20 mean@16 明显低于 MV 2 个点以上，先停，检查 score 构造、state 截取和 weighted loss。
- 如果 step20 接近 MV，继续扩到 80 step。
- 如果 step20 超过 MV 或 chunk-state 统计明显健康，优先扩到 80 step 并做 `K=16` / `chunk_size=128` ablation。

## 当前风险

- Probe 成本可能超过 actor update 节省的时间，需要看 `chunk_state_probe` timing。
- State prefix 过长会让 actor forward 成本仍然偏高，后续可试 `max_state_response_prefix=512`。
- 如果同一 state 下 candidate 全对或全错太多，说明 chunk supervision 信息量不足，需要改 state sampling 或 probe budget。
- 第一版关闭 KL loss，若训练发散，需要加入轻量 reference regularization 或切 pairwise preference。

## 2026-07-31 Smoke 结果

8 卡 B200 worker 上的 1-step smoke 已跑通，脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_smoke_1step_20260731.sh
```

确认点：

- 使用模型 `/models/Qwen2.5-Math-7B`。
- `actor.powerflow_enable=True`。
- `actor.powerflow_use_boxed_reward=True`。
- `actor.chunk_weighted_nll_enable=False`。
- `actor.use_kl_loss=False`。
- chunk-state smoke override 生效：`candidates=2, chunk_size=128, probe_max_tokens=512`。
- PowerFlow actor update 已完成，日志出现 `actor/powerflow_loss`，不是 weighted NLL 或 GRPO actor loss。

smoke step1 关键指标：

```text
chunk_state/num_states=8
chunk_state/num_candidates=16
chunk_state/num_actor_samples=16
chunk_state/positive_ratio=0.062
chunk_state/informative_ratio=0.125
chunk_state/target_entropy=0.609
actor/powerflow_loss=1.341
actor/pg_loss=1.341
actor/log_prob=-0.366
actor/ref_log_prob=-0.366
actor/log_z=-0.752
actor/grad_norm=374.667
timing_s/gen=25.744
timing_s/chunk_state_chunks=0.927
timing_s/chunk_state_probe=2.364
timing_s/chunk_state_score=1.483
timing_s/chunk_state_ref=4.016
timing_s/update_actor=1.062
```

为跑通 smoke 修复的问题：

- vLLM rollout 对短 `max_tokens` 调用按本次调用长度 padding，不再强制 pad 到全局 3072。
- vLLM rollout 在 `n>1` 时对所有长度匹配的 non-tensor 字段统一 repeat，避免 `answer` 与 batch size 不一致。
- vLLM rollout 返回 `response_mask`，供 chunk/probe 拼接使用。
- PowerFlow 开启时强制创建 RefPolicy worker，保证 `ref_log_prob` 存在。
- PowerFlow 主路径不再读取 weighted-NLL fallback 专用的 `chunk_weights`。
- chunk-state early-continue 分支不调用普通 PPO `compute_data_metrics`，避免要求不存在的 `token_level_scores/advantages/returns`。
- chunk-state early-continue 分支仅在 `timing_raw["step"]` 存在时计算 throughput，避免日志字段阻塞训练。

下一步：

- 启动 20-step pilot：`ttrl_chunk_state_powerflow_b32_r32_v64_20step_20260731.sh`。
- 重点观察 step20 的 `val-core/math/acc/mean@16`、chunk informative ratio、PowerFlow loss、probe/ref/update_actor timing。

## 2026-07-31 20-Step Pilot 结果

脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_b32_r32_v64_20step_20260731.sh
```

核心配置：

```text
model=/models/Qwen2.5-Math-7B
train_batch_size=32
rollout.n=32
val.n=16
total_training_steps=20
test_freq=20
chunk_state_candidates=8
chunk_state_chunk_size=256
chunk_state_probe_max_tokens=3072
actor.powerflow_enable=True
actor.powerflow_use_boxed_reward=True
actor.chunk_weighted_nll_enable=False
actor.use_kl_loss=False
actor dynamic batch=False
```

结论：

- 8 卡 B200 正式 20-step pilot 已跑完，确认 actor update 走的是 `actor/powerflow_loss`，不是 weighted NLL，也不是 GRPO。
- 链路工程上有效：每步 32 个 state，256 个 chunk candidates，256 个 actor samples；chunk supervision 不是全 0/全 1。
- 指标失败：step20 `mean@16=0.471`、`maj@16=0.596`、`best@16=0.856`，明显低于当前 MV step20 gate，不应直接扩到 80 step。
- 训练时有 SymPy verifier warning 和一次 `Timeout during comparison`，但进程没有崩溃。

step20 validation：

```text
val-core/math/acc/mean@16=0.471125
val-core/math/acc/maj@16/mean=0.595646
val-core/math/acc/best@16/mean=0.855968
val-aux/math/acc/maj@8/mean=0.575206
val-aux/math/acc/best@8/mean=0.806562
val-aux/math/format_score/mean@16=0.897250
val-aux/math/format_score/maj@16/mean=0.877422
```

step20 chunk / PowerFlow 健康指标：

```text
chunk_state/num_states=32
chunk_state/num_candidates=256
chunk_state/num_actor_samples=256
chunk_state/positive_ratio=0.250
chunk_state/informative_ratio=0.625
chunk_state/target_entropy=1.267
chunk_state/weight_max=0.984
chunk_state/weight_min=0.000
actor/powerflow_loss=0.426
actor/pg_loss=0.426
actor/log_prob=-0.513
actor/ref_log_prob=-0.513
actor/boxed_reward/mean=0.156
actor/boxed_reward/max=1.000
actor/log_z=-1.781
actor/importance_weight=0.800
actor/grad_norm=9.537
```

step20 timing：

```text
timing_s/gen=21.266
timing_s/chunk_state_chunks=1.474
timing_s/chunk_state_probe=13.496
timing_s/chunk_state_score=2.561
timing_s/chunk_state_ref=1.886
timing_s/update_actor=6.050
timing_s/testing=302.242
```

训练中段观察：

- 训练 step 平均约 50-55s，不含 step20 validation。
- full rollout `gen` 通常约 22-24s，偶发到 32-34s。
- chunk candidate generation 约 1.5s。
- probe generation 约 13.5-14.1s。
- chunk score 约 2.5-4.3s。
- ref logprob 约 1.8-2.7s。
- actor update 约 5.7-8.6s。

初步诊断：

- 这个 pilot 的 best@16 仍有 0.856，说明 base sampling/search 本身还有成功轨迹；mean@16/maj@16 低说明更新后的模型分布没有被有效推向正确 chunk。
- `format_score/mean@16=0.897` 但 `acc/mean@16=0.471`，说明模型大多能输出格式化答案，但 chunk-level supervision 对数学正确性提升不足。
- 当前 `chunk_state_probe_max_tokens=3072` 本质上仍在用较长 future probe 定义 chunk label，成本不低，同时标签可能把完整答案的噪声折回局部 chunk。
- `actor/ref_log_prob` 与 `actor/log_prob` 在日志中几乎相同，符合第一轮 close-to-ref 的状态；如果继续训练需要观察 PowerFlow importance / CISPO 是否过早截断有效 token。

下一步建议：

- 先不要扩 80 step；优先做 2-3 个 20-step 诊断实验。
- 检查 state 构造是否过多抽到很短或低价值 prefix；记录 boundary 分布和原 full rollout 正确性。
- 把 candidate chunk 的 probe correctness、原 full rollout correctness、同 prompt majority correctness 同时落日志，确认 chunk label 是否和最终答案方向一致。
- 尝试减少 probe horizon 或改成 top candidates full-probe，以降低噪声和成本。
- PowerFlow loss 主线保留，但需要把 improved distribution 从 hard boxed reward 改成更稳定的 sharpened distribution / ranking distribution，而不是直接把 sparse binary probe reward 写入 chunk update。
