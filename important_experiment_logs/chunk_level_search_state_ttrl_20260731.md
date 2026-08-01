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

## 2026-07-31 3-Step 诊断实验

目的：

- 20-step pilot 明显低于 MV gate 后，不继续盲跑 80 step。
- 增加 `ttrl.chunk_state_diag_enable=True` 诊断字段，只打点，不改变 chunk PowerFlow 训练语义。
- 同时记录 state boundary、source rollout correctness、prompt-level pass、probe label 分布，判断失败来自 state sampling、probe scoring 还是 PowerFlow loss。

新增诊断代码：

- `ttrl.chunk_state_diag_enable` 默认关闭。
- `_make_chunk_state_prompts` 给 state 记录来源：
  - `chunk_state_source_index`
  - `chunk_state_source_prompt_index`
  - `chunk_state_source_local`
  - `chunk_state_boundary`
  - `chunk_state_source_response_len`
- `_compute_chunk_state_diag_metrics` 在开关打开时额外计算：
  - full rollout pseudo-label reward
  - full rollout original-GT reward
  - selected source rollout correctness
  - prompt-level original mean / pass
  - probe mean 在 source correct / source wrong 上的差异

诊断脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_diag_b32_r32_v64_3step_20260731.sh
```

配置：

```text
train_batch_size=32
rollout.n=32
val.n=16
total_training_steps=3
chunk_state_candidates=8
chunk_state_chunk_size=256
chunk_state_probe_max_tokens=3072
actor.powerflow_enable=True
actor.powerflow_use_boxed_reward=True
actor.chunk_weighted_nll_enable=False
actor.use_kl_loss=False
ttrl.chunk_state_diag_enable=True
```

Infra 观察：

- vLLM rollout config 显示 `attention_config.backend=FLASH_ATTN`。
- FlashInfer JIT autotune 有启动日志。
- vLLM 捕获了 mixed prefill-decode 和 decode CUDA graphs。
- Actor 日志显示 flash attention monkey patch 和 Triton fused kernels。
- NCCL 日志显示 `NCCL_NVLS_ENABLE=1`、NVLS multicast available、P2P direct available。
- 这次诊断保留了 NCCL DEBUG 输出用于确认 infra，后续正式实验应降回 WARN，避免 stdout 过大。

诊断 step 指标：

```text
step1:
boundary_mean=344.000
boundary_zero_ratio=0.281
source_response_len_mean=1190.875
source_original_acc_mean=0.406
prompt_original_mean=0.334
prompt_original_pass=0.906
probe_score_std=0.374
state_all_negative_ratio=0.438
state_mixed_ratio=0.562
probe_mean_source_original_correct=0.135
probe_mean_source_original_wrong=0.191
chunk_state/positive_ratio=0.168
actor/boxed_reward/max=1.000
timing_s/chunk_state_score=11.087

step2:
boundary_mean=360.000
boundary_zero_ratio=0.375
source_original_acc_mean=0.250
prompt_original_mean=0.334
prompt_original_pass=0.906
probe_score_std=0.367
state_all_positive_ratio=0.031
state_all_negative_ratio=0.562
state_mixed_ratio=0.406
probe_mean_source_original_correct=0.344
probe_mean_source_original_wrong=0.099
chunk_state/positive_ratio=0.160
actor/boxed_reward/max=0.000
timing_s/chunk_state_score=9.480

step3:
boundary_mean=288.000
boundary_zero_ratio=0.438
source_response_len_mean=848.062
source_original_acc_mean=0.281
prompt_original_mean=0.330
prompt_original_pass=0.969
probe_score_std=0.419
state_all_positive_ratio=0.000
state_all_negative_ratio=0.469
state_mixed_ratio=0.531
probe_mean_source_original_correct=0.444
probe_mean_source_original_wrong=0.141
chunk_state/positive_ratio=0.227
actor/boxed_reward/max=1.000
timing_s/chunk_state_score=9.274
```

step3 validation：

```text
val-core/math/acc/mean@16=0.496375
val-core/math/acc/maj@16/mean=0.626066
val-core/math/acc/best@16/mean=0.863062
val-aux/math/format_score/mean@16=0.910375
```

诊断结论：

- 这不是 infra 失败。B200 上 FlashAttention / FlashInfer / CUDA graph / NVLS / P2P 都有实际日志证据。
- 失败主要来自训练语义：prompt-level `pass@32` 很高，step1/2/3 分别约 0.906/0.906/0.969，但当前 state 采样选中的 source rollout original acc 只有 0.406/0.250/0.281。
- 随机 source + 随机 boundary 浪费了大量可用成功轨迹。很多 state 来自错误 rollout，导致 next-chunk probe label 噪声很大。
- `state_all_negative_ratio` 在 0.438/0.562/0.469，说明接近一半 state 的 8 个 candidates 全错；这些 state 对 PowerFlow 只提供低信息量甚至反向的局部监督。
- step1 中 `probe_mean_source_original_correct < probe_mean_source_original_wrong`，说明单次 future probe 的 label 与完整 source correctness 不稳定；step2/3 转为正相关，但波动很大。
- `chunk_state_score` 从原来的约 2.5s 增到约 9-11s，是因为诊断额外计算了 full rollout pseudo reward + original-GT reward。该成本只用于诊断，不应常开。
- step2 的 `actor/boxed_reward/max=0` 需要继续查：同 step `chunk_state/positive_ratio=0.160`，但 actor 侧看到的 boxed reward 全 0，可能是 CISPO/PowerFlow token 过滤或 reward 落点统计存在边界问题。

下一轮方法调整：

- 不再用“随机 rollout source”作为主路径。
- 改成 success-conditioned / contrastive state sampling：
  - 对每个 prompt 的 32 条 full rollout 先按 original-GT reward 分成 success / fail。
  - 优先从 success rollout 上截 state，保证局部 state 至少位于一条可成功轨迹上。
  - 同时采 fail rollout 的相同或相近 boundary，构造 contrastive state/chunk 对。
  - 对 success-state 的 candidates 用 probe success 构造 PowerFlow target；全错 state 降权或跳过。
- 增加 `skip_all_negative` 或 state weight，避免全错 state 批量拉低局部分布。
- 保留 PowerFlow loss 主线，但 target 从 sparse boxed reward 进一步改为 per-state sharpened distribution / ranking distribution。
- 下一次实验先跑 3-step smoke 验证：
  - source_original_acc_mean 是否显著高于当前随机版。
  - state_all_negative_ratio 是否下降。
  - actor/boxed_reward/max 是否稳定为 1。
  - step3/20 val 是否不再快速跌到 0.49/0.47。

## 2026-07-31 Success-Conditioned 3-Step 诊断

目的：

- 保持 chunk actor update 主路径为 PowerFlow loss，不切到 GRPO，也不使用 weighted NLL fallback。
- 修正上一轮随机 state source 的明显语义问题：同一 prompt 下 `pass@32` 很高，但随机选中的 source rollout original correctness 很低。
- 优先从 original-GT correct 的 full rollout 上截取 chunk state；如果该 prompt 没有 correct rollout，再 fallback 到原来的 round-robin/random source。
- 跳过 8 个 chunk candidates 全部 probe 失败的 state，避免 PowerFlow 在低信息量 all-negative group 上更新。

脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_successdiag_b32_r32_v64_3step_20260731.sh
```

相对随机诊断新增配置：

```text
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_skip_all_negative=True
actor.powerflow_enable=True
actor.powerflow_use_boxed_reward=True
actor.chunk_weighted_nll_enable=False
actor.use_kl_loss=False
```

step 指标：

```text
step1:
selected_original_acc_mean=0.906
prompt_original_pass=0.906
prompt_original_mean=0.334
state_all_negative_ratio=0.406
state_mixed_ratio=0.594
chunk_state/num_actor_samples=152
chunk_state/positive_ratio=0.215
chunk_state/kept_state_ratio=0.594
actor/powerflow_loss=0.711
actor/boxed_reward/max=1.000
timing_s/chunk_state_probe=14.605
timing_s/chunk_state_score=10.093
timing_s/update_actor=3.578

step2:
selected_original_acc_mean=0.875
prompt_original_pass=0.875
prompt_original_mean=0.314
state_all_negative_ratio=0.281
state_mixed_ratio=0.656
chunk_state/num_actor_samples=184
chunk_state/positive_ratio=0.289
chunk_state/kept_state_ratio=0.719
actor/powerflow_loss=0.545
actor/boxed_reward/max=1.000
timing_s/chunk_state_probe=13.682
timing_s/chunk_state_score=9.797
timing_s/update_actor=4.788

step3:
selected_original_acc_mean=0.906
prompt_original_pass=0.906
prompt_original_mean=0.313
state_all_negative_ratio=0.312
state_mixed_ratio=0.688
chunk_state/num_actor_samples=176
chunk_state/positive_ratio=0.266
chunk_state/kept_state_ratio=0.688
actor/powerflow_loss=0.654
actor/boxed_reward/max=1.000
timing_s/chunk_state_probe=13.739
timing_s/chunk_state_score=9.408
timing_s/update_actor=4.829
```

step3 validation：

```text
val-core/math/acc/mean@16=0.457500
val-core/math/acc/maj@16/mean=0.585826
val-core/math/acc/best@16/mean=0.848662
val-aux/math/format_score/mean@16=0.903125
```

结论：

- 这次确认 actor update 仍是 `actor/powerflow_loss`，不是 weighted NLL，也不是 GRPO。
- success-conditioned source selection 生效：selected source original correctness 从随机诊断的 0.250-0.406 提升到 0.875-0.906。
- `skip_all_negative` 生效：actor samples 从固定 256 降到 152/184/176，update_actor 降到 3.6-4.8s，比 20-step pilot 的 5.7-8.6s 更友好。
- `actor/boxed_reward/max` 三步都为 1，上一轮 step2 的全 0 actor reward 问题在该配置下没有复现。
- 但 validation 没有改善，3-step `mean@16=0.4575` 仍明显低于 MV gate。这说明修复 source correctness 只是必要条件，不是充分条件。

下一步判断：

- 失败主因不再是“选错 source rollout”这一项；现在更像是 chunk-level target 分布本身不够可靠。
- 当前 PowerFlow target 仍是 sparse boxed reward：只要 probe 成功就是 1，否则 0。这会把 full-answer verifier 的高方差信号直接压到局部 chunk 上，credit assignment 仍然很粗。
- 下一轮应继续沿 PowerFlow loss 主线改 target，而不是切 GRPO：
  - 用 per-state probe scores 构造 sharpened distribution，而不是只写 binary boxed reward。
  - 对 success source 的 original next chunk 保留 teacher anchor，避免模型只追逐短 probe 偶然成功的 chunk。
  - boundary 先收窄到非零中早段，例如 256/512/768，减少 `boundary_zero_ratio` 过高时退化成普通 full-answer update。
  - `chunk_state_score` 诊断开销只在 smoke 打开；正式 20-step 应关闭重诊断或只低频采样。

## 2026-07-31 Weighted PowerFlow Target 设计

动机：

- 用户明确要求 chunk actor update 优先使用 PowerFlow loss。
- success-conditioned 诊断说明 source selection 已明显改善，但 sparse boxed reward 仍然不能让 validation 变好。
- 因此下一版不切 GRPO，不切 weighted NLL，而是在 PowerFlow residual 上引入 per-state search-improved target distribution。

实现思路：

```text
q_j = normalize((score_j + eps) ** alpha)
w_j = q_j * K
L = mean_j stopgrad(w_j) * delta_j^2
```

其中 `delta_j` 仍然是 PowerFlow trajectory-balance residual：

```text
delta_j = logZ(s) + avg_log pi_theta(c_j | s)
          - beta * (avg_log pi_ref(c_j | s) + (boxed_reward_j - 1) / 2)
```

设计约束：

- `actor.powerflow_enable=True` 仍是主路径。
- `actor.chunk_weighted_nll_enable=False`，weighted NLL 仍只作为 fallback / ablation。
- 新开关 `actor.powerflow_use_chunk_weights=True` 显式启用，不影响普通 PowerFlow/TTRL runs。
- `chunk_weights=q_j` 保留给 weighted NLL fallback；新增 `powerflow_chunk_weights=q_j*K` 给 PowerFlow，保证 uniform target 时 loss scale 近似不变。
- 配套记录 `actor/powerflow_weight/*` 和 `chunk_state/powerflow_weight_*`，确认训练是否实际使用 sharpened target。

待跑 smoke：

```text
verl/run_records/ttrl_chunk_state_powerflow_weighted_successdiag_b32_r32_v64_3step_20260731.sh
```

gate：

- 3-step 必须出现 `actor/powerflow_weight/max > 1`，否则说明 sharpened distribution 没进入 PowerFlow loss。
- `actor/powerflow_loss` 不为 NaN，`update_actor` 不明显劣化。
- 如果 smoke 正常，再开 20-step weighted-success pilot；若 step20 仍明显低于 MV gate，下一刀才考虑 teacher anchor / 非零 boundary ablation。

## 2026-07-31 Weighted Success 3-Step Smoke 结果

脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_weighted_successdiag_b32_r32_v64_3step_20260731.sh
```

配置确认：

```text
actor.powerflow_enable=True
actor.powerflow_use_boxed_reward=True
actor.powerflow_use_chunk_weights=True
actor.chunk_weighted_nll_enable=False
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_skip_all_negative=True
```

实现/infra 修复：

- 第一次启动失败在 Ray AF_UNIX socket path 过长：`/tmp/cspfweightedsuccessdiagb32r32v64s3/.../plasma_store` 超过 107 bytes。
- 已把 weighted smoke 默认 runtime dir 改为 `/tmp/cpw3`，并把 successdiag 默认 runtime dir 改为 `/tmp/cps3`，避免后续同类问题。
- 静态检查已通过：
  - `python -m py_compile verl/workers/actor/dp_actor.py verl/trainer/ppo/ray_trainer.py`
  - `bash -n run_records/ttrl_chunk_state_powerflow_weighted_successdiag_b32_r32_v64_3step_20260731.sh`
  - `git diff --check`

step 指标：

```text
step1:
selected_original_acc_mean=0.906
state_all_negative_ratio=0.406
chunk_state/num_actor_samples=152
chunk_state/positive_ratio=0.215
chunk_state/powerflow_weight_mean=1.000
chunk_state/powerflow_weight_max=7.875
actor/powerflow_weight/max=3.973
actor/powerflow_loss=0.259
timing_s/gen=51.876
timing_s/chunk_state_probe=13.846
timing_s/chunk_state_score=10.203
timing_s/chunk_state_ref=6.896
timing_s/update_actor=4.658

step2:
selected_original_acc_mean=0.938
state_all_negative_ratio=0.406
chunk_state/num_actor_samples=152
chunk_state/positive_ratio=0.238
chunk_state/powerflow_weight_mean=1.000
chunk_state/powerflow_weight_max=7.875
actor/powerflow_weight/max=7.875
actor/powerflow_loss=1.107
timing_s/gen=22.860
timing_s/chunk_state_probe=13.702
timing_s/chunk_state_score=10.302
timing_s/chunk_state_ref=1.208
timing_s/update_actor=3.674

step3:
selected_original_acc_mean=0.938
state_all_negative_ratio=0.312
chunk_state/num_actor_samples=176
chunk_state/positive_ratio=0.219
chunk_state/powerflow_weight_mean=1.000
chunk_state/powerflow_weight_max=7.875
actor/powerflow_weight/max=7.875
actor/powerflow_loss=1.573
timing_s/gen=23.438
timing_s/chunk_state_probe=13.753
timing_s/chunk_state_score=8.869
timing_s/chunk_state_ref=1.531
timing_s/update_actor=4.719
```

step3 validation：

```text
val-core/math/acc/mean@16=0.459875
val-core/math/acc/maj@16/mean=0.588092
val-core/math/acc/best@16/mean=0.848188
val-aux/math/format_score/mean@16=0.900125
timing_s/testing=297.709
```

结论：

- Weighted PowerFlow target 路径已真实生效：trainer 侧 `chunk_state/powerflow_weight_max=7.875`，actor 侧 `actor/powerflow_weight/max=7.875`。
- actor update 正常，loss 非 NaN，update_actor 约 3.7-4.7s，未比 successdiag 明显劣化。
- 3-step validation 仍未改善，和 successdiag 基本同档：mean@16 约 0.46，maj@16 约 0.59。
- `actor/powerflow_weight/mean` 是按 micro-batch reduce 后的均值，step3 显示 0.673 不代表全 batch 权重均值偏移；全 batch 指标看 `chunk_state/powerflow_weight_mean=1.000`。
- 当前最明显的工程瓶颈仍是 probe/scoring：`chunk_state_probe` 约 13.7s，诊断版 `chunk_state_score` 约 8.9-10.3s，最终 validation 约 298s。

下一步：

- 不再继续扩这个 3-step 诊断配置到 80 step。
- 如果要跑 20-step gate，应使用 weighted-success 主路径但关闭重诊断，保留轻量 target statistics。
- 算法下一刀优先 teacher anchor / nonzero boundary：
  - success source 的 original next chunk 加入 PowerFlow target 或作为 anchor weight。
  - boundary 先试 `[256,512,768]`，避免 boundary=0 退化成普通 full-answer sampling。
  - 只低频打开 full original-GT diagnostic，避免每步多算 full rollout reward。

## 2026-07-31 Teacher Anchor + Nonzero Boundary 设计

动机：

- Weighted PowerFlow target 已经确认进入 actor loss，但 3-step validation 仍未改善。
- 当前 target 完全来自 `state + sampled chunk + probe` 的 verifier 结果，仍可能追逐短 probe 的偶然成功。
- success-conditioned source rollout 本身已经是 original-GT correct 的完整轨迹，应把它在当前 boundary 后的 original next chunk 作为局部 teacher anchor。

实现：

```text
state = query + source_response[:t]
anchor_chunk = source_response[t : t + chunk_size]
```

在每个 state 的 K 个 candidates 中，不额外增加 candidate 数，而是把 `candidate[0]` 替换为 `anchor_chunk`，然后照常做 probe scoring。scoring 后对 anchor candidate 加 score floor：

```text
score_anchor = max(score_anchor_from_probe, teacher_anchor_score)
```

默认配置仍关闭，实验脚本显式打开：

```text
ttrl.chunk_state_teacher_anchor_enable=True
ttrl.chunk_state_teacher_anchor_score=1.0
ttrl.chunk_state_teacher_anchor_candidate_index=0
actor.powerflow_use_chunk_weights=True
```

同时把 boundary 收窄到非零中早段：

```text
ttrl.chunk_state_boundaries=[256,512,768]
```

预期：

- `boundary_zero_ratio` 应显著低于 random-boundary 版本；如果 source response 短于所有指定 boundary，当前实现会 fallback 到 0。
- `teacher_anchor/replaced_ratio` 应接近 1；如果明显低，说明很多 success source 长度短于 boundary。
- `chunk_state/powerflow_weight_max` 仍应大于 1，说明 target distribution 不是 uniform。
- `update_actor` 不应明显慢于 weighted-success smoke。

待跑脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_anchor_nonzero_b32_r32_v64_3step_20260731.sh
```

## 2026-07-31 Teacher Anchor + Nonzero Boundary 3-step 结果

配置：

```text
train_batch_size=32
rollout.n=32
val_kwargs.n=16
total_training_steps=3
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=256
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_skip_all_negative=True
ttrl.chunk_state_boundaries=[256,512,768]
ttrl.chunk_state_teacher_anchor_enable=True
ttrl.chunk_state_teacher_anchor_score=1.0
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.chunk_weighted_nll_enable=False
actor.use_kl_loss=False
```

关键 step 指标：

```text
step1:
selected_original_acc_mean=0.906
teacher_anchor/replaced_ratio=1.000
teacher_anchor/mean_len=177.281
boundary_zero_ratio=0.031
state_all_negative_ratio=0.000
state_mixed_ratio=1.000
positive_ratio=0.355
target_entropy=0.837
powerflow_weight_max=7.875
actor/powerflow_loss=1.931
timing_s/gen=52.047
timing_s/chunk_state_probe=13.867
timing_s/chunk_state_score=10.198
timing_s/chunk_state_ref=5.850
timing_s/update_actor=6.456

step2:
selected_original_acc_mean=0.906
teacher_anchor/replaced_ratio=1.000
teacher_anchor/mean_len=174.906
boundary_zero_ratio=0.094
state_all_negative_ratio=0.000
state_mixed_ratio=1.000
positive_ratio=0.309
target_entropy=0.736
powerflow_weight_max=7.875
actor/powerflow_loss=1.757
timing_s/gen=23.501
timing_s/chunk_state_probe=13.798
timing_s/chunk_state_score=9.090
timing_s/chunk_state_ref=2.068
timing_s/update_actor=6.562

step3:
selected_original_acc_mean=0.906
teacher_anchor/replaced_ratio=1.000
teacher_anchor/mean_len=181.062
boundary_zero_ratio=0.031
state_all_negative_ratio=0.000
state_mixed_ratio=1.000
positive_ratio=0.328
target_entropy=0.788
powerflow_weight_max=7.875
actor/powerflow_loss=1.665
timing_s/gen=23.615
timing_s/chunk_state_probe=14.070
timing_s/chunk_state_score=9.289
timing_s/chunk_state_ref=2.130
timing_s/update_actor=6.798
timing_s/testing=300.769
```

final validation：

```text
val-core/math/acc/mean@16=0.456125
val-core/math/acc/maj@16/mean=0.579148
val-core/math/acc/best@16/mean=0.844980
val-aux/math/format_score/mean@16=0.897500
```

结论：

- Teacher anchor 真实生效：每步 `replaced_ratio=1.0`，anchor score floor 生效，PowerFlow weighted loss 仍是 actor update 主路径。
- target 诊断明显变健康：`state_all_negative_ratio=0`，`state_mixed_ratio=1.0`，说明 anchor 解决了 all-negative state 的训练信号缺失。
- 但 final validation 仍和前两个 chunk-state 失败版本同档，低于 base/MV gate，不能升 20-step。
- `boundary_zero_ratio` 仍有 3%-9%，原因是部分 source response 长度不够，`_make_chunk_state_prompts` fallback 到 0；这不是主要失败原因。
- 这条结果说明：仅靠 success original chunk 作为 local teacher anchor，会把局部 target 变干净，但没有解决“局部 chunk 更新破坏 full-answer policy / target 与最终任务分布错配”的问题。

下一步方向：

- 不继续扩 teacher-anchor 3-step 到 20/80 step。
- 优先改训练对象：不要只训练短 chunk 本身，考虑把 PowerFlow loss 作用在 `state + chunk + short continuation` 的可评分 span，或加入 full-answer distillation/regularization，避免模型只学局部补丁。
- 降低 final validation 频率，后续 3-step smoke 只在关键 gate 做 val；诊断阶段记录 target stats 和少量 held-out prompts，减少 300s validation 固定成本。

## 2026-07-31 Scored Span PowerFlow 设计

动机：

- 之前三版 chunk-state PowerFlow 都只在 `next chunk` token 上做 actor update，但 score 来自 `state + chunk + probe continuation` 的最终 correctness。
- 这会产生训练对象错配：被 verifier 打分的是完整可评分 span，实际更新的却只有短 chunk，模型可能学到局部补丁而破坏完整回答分布。
- 下一版仍然保持 PowerFlow loss 主路径，不切 GRPO / weighted NLL；只把 actor response span 从 `chunk` 改成 `chunk + probe continuation`。

实现计划：

```text
ttrl.chunk_state_actor_span=chunk        # default, old behavior
ttrl.chunk_state_actor_span=chunk_probe  # new smoke
```

`chunk_probe` 模式下：

```text
actor_response = concat(chunk_tokens, probe_tokens)
actor_response = actor_response[:data.max_response_length]
boxed_reward   = probe final correctness
powerflow_weight = sharpened q_j * K
```

关键点：

- 仍然只把 `query + source_response[:t]` 作为 prompt，不训练 state prefix token。
- reward 仍来自同一个 probe correctness，因此 target 语义不变。
- response 最长截断到现有 `data.max_response_length=3072`，避免超过 `max_model_len=4096`。
- 默认配置仍保持 `chunk`，不会影响已有 MV / PowerFlow / teacher-anchor 实验。

第一版 smoke 隔离变量：

```text
source_mode=success
skip_all_negative=True
actor.powerflow_use_chunk_weights=True
teacher_anchor=False
actor_span=chunk_probe
```

预期：

- `chunk_state_actor_span/mode_chunk_probe=1`
- `chunk_state_actor_span/response_len_mean` 明显大于 256。
- `update_actor` 会慢于 chunk-only，但应仍低于 full-response GRPO 级别；如果显著爆炸，说明 scored span 成本不可接受。
- 如果 3-step final val 仍是 mean@16 约 0.46，则说明问题不只是 chunk-only mismatch，需要加入 full-answer regularization 或重新定义 state sampling。

待跑脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_scoredspan_b32_r32_v64_3step_20260731.sh
```

## 2026-07-31 Scored Span PowerFlow 3-step 结果

配置：

```text
train_batch_size=32
rollout.n=32
val_kwargs.n=16
total_training_steps=3
source_mode=success
skip_all_negative=True
actor.powerflow_use_chunk_weights=True
teacher_anchor=False
ttrl.chunk_state_actor_span=chunk_probe
```

关键 step 指标：

```text
step1:
selected_original_acc_mean=0.906
state_all_negative_ratio=0.406
state_mixed_ratio=0.594
num_actor_samples=152
positive_ratio=0.215
actor_span/response_len_mean=1560.223
actor_span/truncated_ratio=0.195
powerflow_weight_max=7.875
actor/powerflow_loss=1.237
timing_s/gen=51.865
timing_s/chunk_state_probe=15.590
timing_s/chunk_state_score=10.251
timing_s/chunk_state_ref=7.496
timing_s/update_actor=14.254

step2:
selected_original_acc_mean=0.875
state_all_negative_ratio=0.344
state_mixed_ratio=0.594
num_actor_samples=168
positive_ratio=0.281
actor_span/response_len_mean=1356.617
actor_span/truncated_ratio=0.168
powerflow_weight_max=7.875
actor/powerflow_loss=1.501
timing_s/gen=23.703
timing_s/chunk_state_probe=13.776
timing_s/chunk_state_score=11.260
timing_s/chunk_state_ref=4.264
timing_s/update_actor=14.853

step3:
selected_original_acc_mean=0.938
state_all_negative_ratio=0.312
state_mixed_ratio=0.688
num_actor_samples=176
positive_ratio=0.285
actor_span/response_len_mean=1398.367
actor_span/truncated_ratio=0.164
powerflow_weight_max=7.875
actor/powerflow_loss=0.748
timing_s/gen=34.290
timing_s/chunk_state_probe=13.711
timing_s/chunk_state_score=9.264
timing_s/chunk_state_ref=4.917
timing_s/update_actor=17.188
timing_s/testing=302.175
```

final validation：

```text
val-core/math/acc/mean@16=0.446125
val-core/math/acc/maj@16/mean=0.565996
val-core/math/acc/best@16/mean=0.852924
val-aux/math/format_score/mean@16=0.895625
```

结论：

- `chunk_probe` actor span 已真实生效：平均训练 response 从 256 附近变成 1350-1560 tokens，说明 actor update 覆盖了被 scorer 评估的 chunk+probe span。
- 但这条比 chunk-only 更差，`mean@16=0.446`，低于 successdiag / weighted / teacher-anchor 三个失败版本。
- update_actor 从 chunk-only 的约 3.7-6.8s 上升到 14-17s，ref 也上升到 4-7s；成本明显变差。
- 截断率 16%-20%，说明很多 probe span 已经超过 3072 response budget，训练对象仍不是完整 scored trajectory。
- 因此“把 loss 直接扩到 chunk+probe span”不是当前可行方向，不升 20-step。

下一步判断：

- 目前失败不是单一工程 bug，而是 chunk-level supervision 的语义还没稳定：success source、weighted target、teacher anchor、scored span 都无法把 mean@16 拉回 base/MV gate。
- 继续做 chunk-state 前，应先加一个保护项：full-answer behavior regularization / base-policy distillation，限制局部更新不能破坏完整回答分布。
- 另一个方向是先不更新 actor，只做 offline analysis：比较 source chunk、sampled chunk、probe success 与最终 answer type，确认 chunk score 是否真的能预测 full-answer correctness。
- Infra 上不要继续扩大 scored-span，因为它同时更慢且更差；后续 smoke 应减少 final val 频率，并优先降低 scoring 长尾。

## 2026-07-31 Chunk Score JSONL Diagnostic 设计

动机：

- 目前连续几条训练变体失败，不能继续盲目跑 20-step。
- 需要先验证 `state -> K chunks -> probe score` 是否真的有可学习信号：哪些 boundary 容易 all-negative，source correctness 高是否对应更高 probe mean，probe score 分布是否有区分度。

实现：

- 新增默认关闭配置：

```text
ttrl.chunk_state_diag_jsonl=null
```

- 显式设置路径时，每个 state 写一行 JSONL：

```text
global_step
state_index
source_index / source_prompt_index / source_local
boundary
source_response_len
source_original_correct
probe_mean / probe_max / probe_min
probe_positive_count
probe_scores
all_negative / all_positive / mixed
```

- 这个 dump 只记录 tensor 上已有的 score 摘要，不改变训练 loss，不改变 sampling 语义。

待跑脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_diagjson_b32_r32_v64_1step_20260731.sh
```

目标：

- 先跑 1-step，生成 `/tmp/ttrl_b200/chunk_state_diag/*.jsonl`。
- 用这个文件做离线统计，决定下一步是改 state sampling、改 verifier/probe，还是加入 full-answer regularization。

## 2026-07-31 Chunk Score JSONL Diagnostic 结果

运行脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_diagjson_b32_r32_v64_1step_20260731.sh
```

配置要点：

```text
train_batch_size=32
rollout.n=32
ttrl.chunk_state_enable=True
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_actor_span=chunk
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=False
ttrl.chunk_state_diag_jsonl=/tmp/ttrl_b200/chunk_state_diag/ttrl_chunk_state_powerflow_diagjson_b32_r32_v64_1step_20260731.jsonl
```

产物：

```text
/tmp/ttrl_b200/chunk_state_diag/ttrl_chunk_state_powerflow_diagjson_b32_r32_v64_1step_20260731.jsonl
/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_diagjson_b32_r32_v64_1step_20260731.jsonl
```

注意：首次运行时 `DIAG_JSONL` 默认在 worker `/tmp`，CPU devbox 看不到；已经手工复制到持久目录，并把脚本默认路径改成
`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl`。

训练 step 指标：

```text
chunk_state_source/selected_original_acc_mean=0.906
chunk_state_source/prompt_original_mean=0.334
chunk_state_source/prompt_original_pass=0.906
chunk_state_diag/source_pseudo_acc_mean=0.844
chunk_state_diag/source_original_acc_mean=0.906
chunk_state_diag/state_all_negative_ratio=0.406
chunk_state_diag/state_mixed_ratio=0.594
chunk_state_diag/probe_mean_source_original_correct=0.237
chunk_state_diag/probe_mean_source_original_wrong=0.000
chunk_state_diag/jsonl_rows=32
chunk_state/num_states=32
chunk_state/num_candidates=256
chunk_state/num_actor_samples=152
chunk_state/positive_ratio=0.215
chunk_state/informative_ratio=0.594
chunk_state/kept_state_ratio=0.594
chunk_state_actor_span/mode_chunk=1.000
chunk_state_actor_span/response_len_mean=207.113
chunk_state_actor_span/truncated_ratio=0.000
actor/powerflow_loss=1.299
timing_s/gen=51.312
timing_s/chunk_state_chunks=1.567
timing_s/chunk_state_probe=13.861
timing_s/chunk_state_score=10.311
timing_s/chunk_state_ref=4.921
timing_s/update_actor=3.603
timing_s/testing=293.007
```

这次脚本虽然设置了低频 validation，但训练结束仍触发 final validation，所以端到端显示约 6 分 22 秒；训练本体约为：

```text
gen 51.312 + chunks 1.567 + probe 13.861 + score 10.311 + ref 4.921 + update 3.603
= 85.575s
```

final validation：

```text
val-core/math/acc/mean@16=0.4595
val-core/math/acc/maj@16/mean=0.58238
val-core/math/acc/best@16/mean=0.853788
```

离线 JSONL 统计：

```text
rows=32
probe_mean mean=0.21484375 min=0.0 max=0.625
probe_max mean=0.59375 min=0.0 max=1.0
probe_min mean=0.0 min=0.0 max=0.0
probe_positive_count mean=1.71875 min=0 max=5
source_original_correct=29/32 = 0.90625
all_negative=13/32 = 0.40625
all_positive=0/32 = 0.0
mixed=19/32 = 0.59375
probe_mean_correct=0.2370689655
probe_mean_wrong=0.0
flat_candidate_scores=256
flat_score_mean=0.21484375
flat_positive_ratio=0.21484375
```

按 boundary 分桶：

```text
boundary=0:
  states=8
  probe_mean=0.359375
  all_negative_ratio=0.125
  source_original_correct_ratio=1.0

boundary=1..256:
  states=14
  probe_mean=0.1785714286
  all_negative_ratio=0.4285714286
  source_original_correct_ratio=0.9285714286

boundary=257..768:
  states=10
  probe_mean=0.15
  all_negative_ratio=0.6
  source_original_correct_ratio=0.8
```

结论：

- JSONL 诊断确认当前 `state -> chunk -> probe` 信号不是完全随机：source 原始正确时 probe mean 为 0.237，source 原始错误时为 0。
- 但信号仍很稀疏：40.6% state 是 all-negative，且没有 all-positive；越往中后段 boundary，probe mean 越低、all-negative 越高。
- 这解释了之前连续 PowerFlow chunk actor update 失败：actor 确实用 PowerFlow loss 更新了 chunk，但 target distribution 大量来自稀疏/局部 probe，容易把完整回答分布拉坏。
- 目前不应该把这版直接放大到 20/80 step；下一步应该仍优先使用 PowerFlow loss 做 chunk actor update，但先改 state/probe 采样，让 PowerFlow target 更像 improved continuation distribution，而不是稀疏终局 reward 的局部投影。

下一步优先级：

1. 保留 PowerFlow actor loss 主线。
2. 把 JSONL 路径默认指向持久目录或在脚本里同步到持久目录，避免 worker `/tmp` 和 devbox `/tmp` 不一致。
3. 改 state 选择：减少 boundary=0，重点抽取 source correct 且 probe mixed 的中间 state；同时记录 source chunk 是否来自 correct final answer。
4. 改 probe 设计：对每个 chunk candidate 做更短但更宽的 probe，优先获得 ranking/distribution，而不是只靠 0/1 稀疏 score。
5. 再做 3-step smoke，只有 `mean@16` 不低于 0.60 且 `all_negative_ratio` 明显下降时才升 20-step。

## 2026-07-31 Mid-State PowerFlow 变体设计

动机：

- 上一轮 JSONL 显示 `boundary=0` 的 probe 信号最强，但它不是我们要的 chunk-level search-state training；它退化成 prompt-only next chunk。
- 真实创新点要求训练对象是 `query + generated prefix -> next chunk`，所以必须让 actor update 发生在中间推理状态。
- 直接把 boundary 列表设成 `[256,512,768]` 仍不够：如果选中的 source rollout response 太短，原代码会 fallback 到 `0`。

代码改动：

- 新增默认关闭配置：

```text
ttrl.chunk_state_min_boundary=0
```

- 当该值为正数时，state 构造会优先选择 response 长度足够覆盖最小 boundary 的 source rollout。
- 在 `source_mode=success` 下，优先级变成：

```text
correct and long enough source
-> long enough source
-> original fallback
```

- boundary 选择时，如果存在不小于 `min_boundary` 的合法 boundary，则过滤掉更短 boundary。
- 默认值为 0，旧脚本语义不变。

新 smoke 脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_midstate_b32_r32_v64_3step_20260731.sh
```

关键配置：

```text
ttrl.chunk_state_boundaries=[256,512,768]
ttrl.chunk_state_min_boundary=256
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_skip_all_negative=True
ttrl.chunk_state_teacher_anchor_enable=True
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
ttrl.chunk_state_actor_span=chunk
ttrl.chunk_state_diag_jsonl=/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/${RUN_ID}.jsonl
```

验收重点：

- `chunk_state_diag/boundary_zero_ratio` 必须接近 0。
- `chunk_state_diag/state_all_negative_ratio` 相比上一轮 0.406 不能更差太多，理想下降。
- `chunk_state/kept_state_ratio` 不能太低，否则训练样本不足。
- `actor/powerflow_loss` 必须真实出现，确认仍是 PowerFlow actor update。
- 3-step final val 如果仍在 0.45-0.46 附近，说明仅修正 mid-state 还不够，下一步转向 probe target 设计，而不是升 20-step。

## 2026-07-31 Mid-State PowerFlow 3-Step Smoke 结果

运行脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_midstate_b32_r32_v64_3step_20260731.sh
```

持久诊断产物：

```text
/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_midstate_b32_r32_v64_3step_20260731.jsonl
```

配置确认：

```text
total_training_steps=3
train_batch_size=32
rollout.n=32
ttrl.chunk_state_boundaries=[256,512,768]
ttrl.chunk_state_min_boundary=256
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_skip_all_negative=True
ttrl.chunk_state_teacher_anchor_enable=True
ttrl.chunk_state_actor_span=chunk
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.use_dynamic_bsz=False
rollout attention_config.backend=FLASH_ATTN
```

step-level 关键指标：

```text
step1:
  boundary_mean=456.000
  boundary_zero_ratio=0.000
  state_all_negative_ratio=0.000
  state_mixed_ratio=0.969
  positive_ratio=0.320
  kept_state_ratio=1.000
  actor/powerflow_loss=0.574
  update_actor=7.818s
  training step wall=94.51s

step2:
  boundary_mean=424.000
  boundary_zero_ratio=0.062
  state_all_negative_ratio=0.000
  state_mixed_ratio=1.000
  positive_ratio=0.273
  kept_state_ratio=1.000
  actor/powerflow_loss=1.377
  update_actor=7.046s
  training step wall average=74.70s

step3:
  boundary_mean=488.000
  boundary_zero_ratio=0.031
  state_all_negative_ratio=0.000
  state_mixed_ratio=0.938
  positive_ratio=0.449
  kept_state_ratio=1.000
  actor/powerflow_loss=0.763
  update_actor=7.501s
  testing=292.974s
```

final validation：

```text
val-core/math/acc/mean@16=0.49475
val-core/math/acc/maj@16/mean=0.622692
val-core/math/acc/best@16/mean=0.864892
val-aux/math/format_score/mean@16=0.900625
```

JSONL 离线统计（三步合计 96 个 state）：

```text
rows=96
steps=[1,2,3]

step1:
  boundary_zero_ratio=0.000
  source_correct=0.90625
  probe_mean=0.3203125
  all_negative=0.000
  mixed=0.96875
  all_positive=0.03125
  probe_positive_count_mean=2.5625

step2:
  boundary_zero_ratio=0.0625
  source_correct=0.96875
  probe_mean=0.2734375
  all_negative=0.000
  mixed=1.000
  all_positive=0.000
  probe_positive_count_mean=2.1875

step3:
  boundary_zero_ratio=0.03125
  source_correct=0.9375
  probe_mean=0.44921875
  all_negative=0.000
  mixed=0.9375
  all_positive=0.0625
  probe_positive_count_mean=3.59375

overall:
  boundary_mean=456.0
  boundary_zero_ratio=0.03125
  source_correct_ratio=0.9375
  all_negative_ratio=0.0
  mixed_ratio=0.96875
  all_positive_ratio=0.03125
  flat_score_mean=0.34765625
  flat_positive_ratio=0.34765625
```

结论：

- `chunk_state_min_boundary` 修复了上一轮最关键的监督退化问题：`boundary_zero_ratio` 从上一轮 0.25 降到整体 0.031，绝大多数训练样本是真正的中间 search state。
- target 分布明显健康：上一轮 `state_all_negative_ratio=0.406`，这轮三步都是 0；`mixed_ratio` 约 0.94-1.00，PowerFlow target 有真实区分度。
- actor 侧确认仍是 PowerFlow loss：`actor/powerflow_loss` 每步都有，且 `powerflow_use_chunk_weights=True` 后 `actor/powerflow_weight` 非恒等。
- 但是 final val 仍然只有 `mean@16=0.49475`，比前几版 0.45-0.46 有改善，但距离 20-step gate 的 0.60 下限仍很远。
- 这说明“mid-state 采样 + teacher anchor + weighted PowerFlow”修好了 target 统计，但还没修好训练语义对 full-answer accuracy 的破坏。

下一步：

1. 不升 20-step。
2. 保留 mid-state gate 和 PowerFlow loss。
3. 改 probe target：减少把单个 0/1 final reward 直接投影到 chunk 的噪声，优先做更宽/更短 probe 或 probe-majority target。
4. 增加一个 no-final-val smoke 模式，避免 3-step 诊断每次多花约 293s validation。

## 2026-07-31 Multi-Probe Target 设计

动机：

- Mid-state 版本已经把 `boundary_zero_ratio` 和 `all_negative_ratio` 修好，但 `mean@16=0.49475` 仍明显低于 gate。
- 当前每个 chunk candidate 只有一次 probe，`score_j` 是单次 0/1 final correctness，对 chunk credit assignment 噪声太大。
- 下一步保持 PowerFlow actor loss 不变，只把 `score_j` 从单次 Bernoulli 样本改成多 probe 的均值，更接近 `P(success | state, chunk)`。

代码改动：

```text
ttrl.chunk_state_probe_samples: 1
```

- 默认值 1，旧实验语义不变。
- 当 `probe_samples > 1` 时，对每个 `state+chunk` 采样多个 probe completion。
- reward 先对每个 probe completion 做 rule-based math correctness，再 reshape 为：

```text
num_states x candidates x probe_samples
```

- 最终 chunk score 是 probe 维度均值：

```text
score_j = mean_k final_correctness(state + chunk_j + probe_{j,k})
```

- PowerFlow loss、chunk actor span、teacher anchor、chunk weights 都保持不变。

同时新增：

```text
trainer.final_val_enable: True
```

- 默认开启，正式训练和历史脚本不变。
- 诊断脚本可设为 `False`，避免最后一步强制跑 500 x 16 validation。

新 smoke 脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_midstate_probe4_b32_r32_v64_1step_20260731.sh
```

关键配置：

```text
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
trainer.final_val_enable=False
trainer.total_training_steps=1
```

验收：

- 1-step 能跑通，无 shape mismatch。
- `chunk_state_probe/samples=4`。
- `chunk_state_probe/raw_positive_ratio` 和聚合后的 `chunk_state/positive_ratio` 同时出现。
- final validation 被跳过，节省约 293s。

运行结果：

```text
script:
  verl/run_records/ttrl_chunk_state_powerflow_midstate_probe4_b32_r32_v64_1step_20260731.sh

config:
  trainer.total_training_steps=1
  trainer.final_val_enable=False
  ttrl.chunk_state_probe_samples=4
  ttrl.chunk_state_probe_max_tokens=1024
  ttrl.chunk_state_boundaries=[256,512,768]
  ttrl.chunk_state_min_boundary=256
  ttrl.chunk_state_source_mode=success
  actor.powerflow_enable=True
  actor.powerflow_use_chunk_weights=True
  actor.use_dynamic_bsz=False
```

stdout 关键指标：

```text
chunk_state_probe/samples=4.000
chunk_state_probe/raw_positive_ratio=0.303
chunk_state_diag/boundary_zero_ratio=0.000
chunk_state_diag/state_all_negative_ratio=0.000
chunk_state_diag/state_all_positive_ratio=0.156
chunk_state_diag/state_mixed_ratio=0.844
chunk_state_diag/probe_score_std=0.416
chunk_state_diag/state_probe_mean_min=0.125
chunk_state_diag/state_probe_mean_max=0.875
chunk_state/num_states=32
chunk_state/num_candidates=256
chunk_state/positive_ratio=0.379
chunk_state/informative_ratio=1.000
chunk_state/target_entropy=1.037
chunk_state_actor_span/response_len_mean=225.680
actor/powerflow_loss=0.810
timing_s/gen=52.030
timing_s/chunk_state_chunks=1.685
timing_s/chunk_state_probe=7.019
timing_s/chunk_state_score=10.783
timing_s/chunk_state_ref=6.157
timing_s/update_actor=7.774
Final validation skipped
```

JSONL 离线统计：

```text
file:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_midstate_probe4_b32_r32_v64_1step_20260731.jsonl

rows=32
probe_scores_flat_count=256
probe_mean_avg=0.37890625
probe_mean_min=0.125
probe_mean_max=0.875
probe_mean_median=0.359375
probe_positive_count_avg=4.21875
probe_positive_count_min=1
probe_positive_count_max=8
flat_positive_ratio=0.52734375
all_negative=0
all_positive=5
mixed=27
boundary_counts={256: 14, 512: 11, 768: 7}
source_len_mean=1272.65625
source_len_min=296
source_len_max=3072
source_original_correct_mean=0.90625
probe_mean_source_correct=0.40086206896551724
probe_mean_source_wrong=0.16666666666666666
```

结论：

- Multi-probe target 按预期降低了单次 0/1 probe 的噪声：没有 all-negative state，所有 32 个 state 都 informative。
- `source_original_correct` 与后续 probe mean 有可见分离，说明这个 target 比单 probe 更像 `P(success | state, chunk)`，更适合接 PowerFlow distribution matching。
- 这一步只是 1-step 诊断，不看 final accuracy；后续如果扩展，应先做 3-step no-final-val gate，再决定是否升 20-step。
- 代价是额外 probe/score 开销：1-step 中 `chunk_state_probe=7.019s`、`chunk_state_score=10.783s`。正式实验需要控制 probe samples、probe max tokens 和 state 数量，避免 target 稳定性收益被吞吐吃掉。

## 2026-07-31 Probe4 Mid-State 3-Step Gate

目的：

- 验证 multi-probe target 在连续 PowerFlow actor update 后是否稳定。
- 不跑 final validation，避免把 gate 时间浪费在 500 x 16 validation。
- 仍然优先使用 PowerFlow loss 做 chunk actor update，不切到 GRPO 或 weighted NLL。

脚本：

```text
verl/run_records/ttrl_chunk_state_powerflow_midstate_probe4_b32_r32_v64_3step_20260731.sh
```

关键配置：

```text
train_batch_size=32
rollout.n=32
ttrl.chunk_state_candidates=8
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
ttrl.chunk_state_boundaries=[256,512,768]
ttrl.chunk_state_min_boundary=256
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_teacher_anchor_enable=True
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.use_dynamic_bsz=False
trainer.total_training_steps=3
trainer.final_val_enable=False
```

运行结果：

```text
log:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_midstate_probe4_b32_r32_v64_3step_20260731.log

jsonl:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_midstate_probe4_b32_r32_v64_3step_20260731.jsonl

jsonl_rows=96
Final validation skipped
```

逐步 stdout 指标：

```text
step 1:
  raw_positive_ratio=0.303
  positive_ratio=0.379
  all_negative_ratio=0.000
  all_positive_ratio=0.156
  mixed_ratio=0.844
  target_entropy=1.037
  actor/powerflow_loss=1.443
  actor/grad_norm=47.833
  timing_s/gen=52.014
  timing_s/chunk_state_probe=7.020
  timing_s/chunk_state_score=10.730
  timing_s/chunk_state_ref=6.130
  timing_s/update_actor=7.909

step 2:
  raw_positive_ratio=0.231
  positive_ratio=0.316
  all_negative_ratio=0.000
  all_positive_ratio=0.094
  mixed_ratio=0.906
  target_entropy=0.924
  actor/powerflow_loss=0.643
  actor/grad_norm=19.524
  timing_s/gen=24.890
  timing_s/chunk_state_probe=6.634
  timing_s/chunk_state_score=11.518
  timing_s/chunk_state_ref=2.262
  timing_s/update_actor=7.231

step 3:
  raw_positive_ratio=0.335
  positive_ratio=0.410
  all_negative_ratio=0.000
  all_positive_ratio=0.188
  mixed_ratio=0.812
  boundary_zero_ratio=0.031
  target_entropy=1.068
  actor/powerflow_loss=0.799
  actor/grad_norm=22.770
  timing_s/gen=22.365
  timing_s/chunk_state_probe=6.713
  timing_s/chunk_state_score=11.677
  timing_s/chunk_state_ref=2.368
  timing_s/update_actor=7.521
```

JSONL 离线统计：

```text
step 1:
  rows=32
  probe_mean_avg=0.37890625
  flat_positive_ratio=0.52734375
  all_negative=0
  all_positive=5
  mixed=27
  boundary_counts={256: 14, 512: 11, 768: 7}
  source_len_mean=1272.65625
  source_original_correct_mean=0.90625
  probe_mean_correct=0.40086206896551724
  probe_mean_wrong=0.16666666666666666

step 2:
  rows=32
  probe_mean_avg=0.31640625
  flat_positive_ratio=0.50390625
  all_negative=0
  all_positive=3
  mixed=29
  boundary_counts={256: 16, 512: 8, 768: 8}
  source_len_mean=1158.125
  source_original_correct_mean=0.90625
  probe_mean_correct=0.33620689655172414
  probe_mean_wrong=0.125

step 3:
  rows=32
  probe_mean_avg=0.41015625
  flat_positive_ratio=0.58203125
  all_negative=0
  all_positive=6
  mixed=26
  boundary_counts={0: 1, 256: 13, 512: 11, 768: 7}
  source_len_mean=1178.75
  source_original_correct_mean=0.96875
  probe_mean_correct=0.41935483870967744
  probe_mean_wrong=0.125
```

结论：

- Gate 通过训练稳定性检查：连续 3 步没有 shape mismatch、没有 NaN、没有 all-negative state，PowerFlow loss 和 grad norm 没有爆炸。
- Probe4 target 的信息密度稳定：三步 `flat_positive_ratio` 在 0.50 到 0.58，state 级 `all_negative=0/32`。
- 第 3 步出现 1 个 `boundary=0`，离线看是单个 source response 长度只有 95，说明 `chunk_state_min_boundary=256` 仍可能在极短成功轨迹上退回 0 boundary；不是整体退化，但后续 20-step 前最好把 source selection 改成严格过滤短于 256 的 source，而不是 fallback。
- 稳态速度比 1-step 更清楚：step2/3 的 `gen` 降到 24.9s/22.4s，`update_actor` 为 7.2s/7.5s。当前新增开销主要是 rule-based scoring，`chunk_state_score` 约 11.5s，probe generation 约 6.7s。
- 下一步不建议直接上 80-step；建议先修掉短 source fallback，再跑 20-step with validation gate，看 mean@16 是否能明显超过此前 chunk MVP 的 0.49 区间。

## 2026-07-31 Strict Source Boundary 修复

问题：

- 3-step gate 的第 3 步出现 1 个 `boundary=0`。
- 原因是 `_make_chunk_state_prompts` 在某个 prompt 下找不到长度达到 `chunk_state_min_boundary=256` 的 source 时，会 fallback 到任意 source；如果这个 source response 太短，allowed boundary 为空，最后退回 query-only state。
- 这会把训练单位从 chunk-level transition 退化成 query-level prefix，和当前方法定义不一致。

代码修复：

- 当 `chunk_state_min_boundary > 0` 时，source 选择必须满足最小 response 长度。
- 对于没有合法 source 的 prompt，跳过该 prompt 的 chunk state，不再 fallback 到 boundary 0。
- 若整个 batch 没有合法 chunk state，直接抛出错误，而不是静默训练 query-only state。
- 新增诊断字段：

```text
chunk_state_diag/skipped_short_sources
skipped_short_sources
```

Strict smoke：

```text
RUN_ID=ttrl_chunk_state_powerflow_midstate_probe4_strictsrc_b32_r32_v64_1step_20260731
trainer.total_training_steps=1
trainer.final_val_enable=False
ttrl.chunk_state_probe_samples=4
```

结果：

```text
jsonl_rows=32
boundary_counts={256: 14, 512: 11, 768: 7}
skipped_short_sources_values=[0]
boundary_zero_ratio=0.000
all_negative=0
all_positive=5
mixed=27
probe_mean_avg=0.37890625
source_len_min=296
source_len_mean=1272.65625
Final validation skipped
```

结论：

- 修复没有改变正常 batch 的样本数和 target 分布。
- 后续如果遇到极短 source，会跳过该 state，并通过 `skipped_short_sources` 记录，而不是混入 boundary 0。
- 现在可以进入更有意义的 20-step gate：仍然使用 PowerFlow loss 做 chunk actor update，先看 final val 是否摆脱此前 `mean@16≈0.49` 的失败区间。

## 2026-07-31 Strict Source + DP Padding 20-step Gate

Run:

```text
RUN_ID=ttrl_chunk_state_powerflow_midstate_probe4_strictsrc_b32_r32_v64_20step_rerun_20260731
train_batch_size=32
rollout.n=32
trainer.total_training_steps=20
trainer.test_freq=20
trainer.final_val_enable=True
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_boundaries=[256,512,768]
ttrl.chunk_state_min_boundary=256
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_teacher_anchor_enable=True
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.use_dynamic_bsz=False
```

这次 rerun 使用 `Pad strict chunk states for DP rollout` 修复：strict source 过滤后，如果合法 state 数不是 8 的倍数，就复制最后一个合法 state 补齐，并把补齐样本的 `chunk_state_loss_weight` 设为 0。actor batch 中 `chunk_weights` 和 `powerflow_chunk_weights` 都乘该 loss weight，因此补齐样本不贡献 PowerFlow loss。

运行稳定性：

```text
step1: real_states=32 pad_states=0 loss_weight_mean=1.000
step2: real_states=30 pad_states=2 loss_weight_mean=0.938
step3: real_states=31 pad_states=1 loss_weight_mean=0.969
step18: real_states=29 pad_states=3 loss_weight_mean=0.906
step19: real_states=30 pad_states=2 loss_weight_mean=0.938
step20: real_states=31 pad_states=1 loss_weight_mean=0.969
```

结论是工程链路已跑通：没有再出现 `DataProto 31 and chunk 8`，`boundary_zero_ratio=0`，且每步都有 `actor/powerflow_loss`。step20 中 `actor/pg_loss=0.556`、`actor/powerflow_loss=0.556`，说明 actor update 确实走的是 PowerFlow loss。

step20 耗时拆分：

```text
timing_s/gen=22.551
timing_s/chunk_state_chunks=1.488
timing_s/chunk_state_probe=6.988
timing_s/chunk_state_score=12.380
timing_s/chunk_state_ref=2.293
timing_s/update_actor=7.367
timing_s/testing=306.205
```

训练进度条显示不含 final validation 的稳态大约 58 到 62 秒/step。当前新增开销主要来自 probe4 搜索监督：`chunk_state_probe + chunk_state_score` 大约 19 秒/step。actor update 本身约 7.3 到 7.8 秒/step。

Final validation:

```text
val-core/math/acc/mean@16=0.399875
val-core/math/acc/maj@16/mean=0.500504
val-core/math/acc/best@16/mean=0.813222
val-aux/math/format_score/mean@16=0.874125
val-aux/math/format_score/maj@16/mean=0.833808
```

结论：

- 这个版本不通过 20-step gate，不能升 80-step。
- 指标低于此前失败的 chunk MVP 区间，也远低于 MV baseline 的 20-step `mean@16≈0.735`。
- `format_score` 仍高，但 acc 崩，说明模型仍会输出格式化答案，只是数学正确性/语义分布被破坏。
- 失败更像目标构造问题，而不是 infra 问题：PowerFlow loss、chunk span、DP padding、Flash attention/vLLM 路径都已生效。
- 需要停止沿着“strict success source + teacher anchor score floor=1 + probe4 PowerFlow target”直接扩展；下一版应重新设计 target，优先考虑降低 teacher-anchor 硬替换和 score floor 的强度，或改为从完整 rollout 随机截断后的 state 重新做 candidate/probe，避免每个 state 都被成功轨迹 teacher anchor 牵引到退化分布。

## 2026-07-31 Random Source + No Teacher Anchor PowerFlow 3-step Smoke

Run:

```text
RUN_ID=ttrl_chunk_state_powerflow_randomsrc_probe4_b32_r32_v64_3step_20260731
train_batch_size=32
rollout.n=32
trainer.total_training_steps=3
trainer.test_freq=2000000
trainer.final_val_enable=False
ttrl.chunk_state_source_mode=random
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_boundaries=[0,256,512,768,1024]
ttrl.chunk_state_min_boundary=0
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.use_dynamic_bsz=False
```

这次 smoke 的目的不是追指标，而是回到 `chunk_level_search_state_ttrl_24h_goal.md` 的主语义：先采完整 on-policy rollout，再随机截断成 search state，并用 PowerFlow distribution matching 做 next-chunk actor update。它去掉了 strict-source 版本里的 teacher-anchor 硬替换和 score floor。

运行稳定性：

```text
jsonl_rows=96
step1: real_states=32 pad_states=0 boundary_zero_ratio=0.188 actor/powerflow_loss=0.784
step2: real_states=32 pad_states=0 boundary_zero_ratio=0.188 actor/powerflow_loss=0.982
step3: real_states=32 pad_states=0 boundary_zero_ratio=0.188 actor/powerflow_loss=1.273
Final validation skipped
```

目标信号诊断：

```text
step1: selected_original_acc_mean=0.094 positive_ratio=0.205 informative_ratio=0.531 state_all_negative_ratio=0.469 state_mixed_ratio=0.500
step2: selected_original_acc_mean=0.000 positive_ratio=0.138 informative_ratio=0.469 state_all_negative_ratio=0.531 state_mixed_ratio=0.438
step3: selected_original_acc_mean=0.125 positive_ratio=0.085 informative_ratio=0.344 state_all_negative_ratio=0.656 state_mixed_ratio=0.344
```

耗时拆分：

```text
step1: gen=51.234s chunk_state_probe=6.790s chunk_state_score=10.809s update_actor=8.868s
step2: gen=23.427s chunk_state_probe=7.187s chunk_state_score=10.938s update_actor=8.456s
step3: gen=22.812s chunk_state_probe=6.977s chunk_state_score=11.023s update_actor=8.726s
```

结论：

- 工程链路正确：3 step 都完成，chunk actor update 走的是 PowerFlow loss，不是 GRPO 或 weighted NLL。
- 但完全 random source 的监督信号太稀疏：`state_all_negative_ratio` 从 0.469 升到 0.656，`positive_ratio` 从 0.205 降到 0.085。继续放大到 20 step 很可能只是用大量全负 state 做退化更新。
- 下一版应保持 no-teacher-anchor 和 PowerFlow loss，但把 source selection 改成 successful full rollout source：`ttrl.chunk_state_source_mode=success`、`ttrl.chunk_state_teacher_anchor_enable=False`。这样仍然从完整 rollout 构造 chunk state，不做 teacher 硬替换，但会提高 probe target 的正信号密度。

## 2026-07-31 Success Source + No Teacher Anchor PowerFlow 3-step Smoke

Run:

```text
RUN_ID=ttrl_chunk_state_powerflow_successsrc_noanchor_probe4_b32_r32_v64_3step_20260731
train_batch_size=32
rollout.n=32
trainer.total_training_steps=3
trainer.test_freq=2000000
trainer.final_val_enable=False
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_boundaries=[0,256,512,768,1024]
ttrl.chunk_state_min_boundary=0
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.use_dynamic_bsz=False
```

这次 smoke 只改 state source：从 random source 改成 successful full rollout source，但仍然不做 teacher-anchor hard replacement，也不做 score floor。训练更新继续是 PowerFlow chunk loss。

配置/infra 证据：

```text
model.path=/models/Qwen2.5-Math-7B
rollout backend=vLLM FLASH_ATTN
vLLM CUDA graph capture enabled
flashinfer autotune triggered
NCCL isAllDirectP2p=1
NCCL_NVLS_ENABLE=1, nvls channels available
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
PowerFlow proj_z added and wrapped by FSDP
```

目标信号诊断：

```text
step1: selected_original_acc_mean=0.906 positive_ratio=0.310 informative_ratio=0.750 state_all_negative_ratio=0.250 state_mixed_ratio=0.656
step2: selected_original_acc_mean=0.906 positive_ratio=0.215 informative_ratio=0.625 state_all_negative_ratio=0.375 state_mixed_ratio=0.625
step3: selected_original_acc_mean=0.906 positive_ratio=0.325 informative_ratio=0.781 state_all_negative_ratio=0.219 state_mixed_ratio=0.625
```

PowerFlow actor update：

```text
step1: actor/powerflow_loss=1.914 grad_norm=35.559 boxed_reward_mean=0.258
step2: actor/powerflow_loss=1.565 grad_norm=46.741 boxed_reward_mean=0.172
step3: actor/powerflow_loss=0.413 grad_norm=30.287 boxed_reward_mean=0.352
```

耗时拆分：

```text
step1: gen=51.138s chunk_state_probe=6.469s chunk_state_score=10.893s update_actor=8.828s
step2: gen=23.109s chunk_state_probe=6.992s chunk_state_score=11.860s update_actor=8.273s
step3: gen=30.547s chunk_state_probe=6.842s chunk_state_score=11.039s update_actor=8.432s
```

结论：

- 这个版本通过 3-step smoke gate。相比 random source，正信号和 mixed-state 比例明显更健康，`state_all_negative_ratio` 稳定在 0.219 到 0.375，而不是持续升高到 0.656。
- 它仍然符合当前主线：先完整 on-policy rollout，再构造 chunk search state，用 PowerFlow distribution matching 更新 next-chunk actor；没有切回 GRPO，也没有用 weighted NLL 替代。
- 可以进入 20-step gate，观察 final validation 是否至少摆脱 strict teacher-anchor 版本的 `mean@16=0.399875` 失败区间，并评估是否有机会接近/超过 20-step MV baseline。

## 2026-07-31 Success Source + No Teacher Anchor PowerFlow 20-step Gate

Run:

```text
RUN_ID=ttrl_chunk_state_powerflow_successsrc_noanchor_probe4_b32_r32_v64_20step_20260731
train_batch_size=32
rollout.n=32
trainer.total_training_steps=20
trainer.test_freq=20
trainer.final_val_enable=True
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_boundaries=[0,256,512,768,1024]
ttrl.chunk_state_min_boundary=0
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.chunk_weighted_nll_enable=False
actor.use_dynamic_bsz=False
```

运行命令：

```text
RUN_ID=ttrl_chunk_state_powerflow_successsrc_noanchor_probe4_b32_r32_v64_20step_20260731 TOTAL_TRAINING_STEPS=20 TEST_FREQ=20 FINAL_VAL_ENABLE=True DIAG_JSONL=/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_successsrc_noanchor_probe4_b32_r32_v64_20step_20260731.jsonl bash /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_randomsrc_probe4_b32_r32_v64_20step_20260731.sh ttrl.chunk_state_source_mode=success ttrl.chunk_state_teacher_anchor_enable=False ttrl.chunk_state_min_boundary=0 2>&1 | tee /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_successsrc_noanchor_probe4_b32_r32_v64_20step_20260731.log
```

最终 validation:

```text
val-core/math/acc/mean@16=0.4335
val-core/math/acc/maj@16/mean=0.55015
val-core/math/acc/best@16/mean=0.8437740000000001
val-aux/math/format_score/mean@16=0.88925
val-aux/math/format_score/maj@16/mean=0.86167
timing_s/testing=298.499
```

step20 诊断：

```text
chunk_state_source/selected_original_acc_mean=0.906
chunk_state_probe/raw_positive_ratio=0.276
chunk_state_diag/state_all_positive_ratio=0.219
chunk_state_diag/state_all_negative_ratio=0.438
chunk_state_diag/state_mixed_ratio=0.344
chunk_state/positive_ratio=0.276
chunk_state/informative_ratio=0.562
actor/pg_loss=0.755
actor/powerflow_loss=0.755
actor/grad_norm=9.603
jsonl_rows=640
```

耗时：

```text
step20: gen=31.038s chunk_state_probe=6.933s chunk_state_score=12.073s update_actor=7.783s testing=298.499s
steady step after warmup: roughly 58-62s/step
```

结论：

- 工程链路稳定，20 step 训练和 final validation 均完成；诊断 JSONL 为 640 行，符合 20 step x 32 state。
- actor update 明确走 PowerFlow chunk loss：`actor/powerflow_loss` 与 `actor/pg_loss` 对齐，且 `actor.chunk_weighted_nll_enable=False`。这不是 GRPO，也不是 weighted NLL。
- 这个 gate 指标失败：`mean@16=0.4335`、`maj@16=0.55015`，远低于 20-step MV baseline 和 PowerFlow 原 repo 轨迹。不能直接扩到 80 step。
- 主要问题不是 loss 没接上，而是 target construction 仍然太弱：即使 source 是 successful rollout，chunk probe 在 step20 仍有 `state_all_negative_ratio=0.438`，并且 positive ratio 只有 0.276。当前 chunk target 很容易把模型推向“局部看起来可行但终局不稳”的 next-chunk 分布。
- 下一轮必须继续以 PowerFlow distribution matching 为主 loss，但重做 target：优先考虑对每个 state 保留原 successful source chunk 作为 support candidate，并用 probe-improved distribution 做软重加权，而不是只依赖重采样 chunk；同时减少全负 state 的更新权重，避免错误地把无信息 state 当成有效训练信号。

## 2026-07-31 Source Chunk Support + PowerFlow 3-step Smoke

改动：

```text
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_source_chunk_candidate_index=0
ttrl.chunk_state_teacher_anchor_enable=False
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.chunk_weighted_nll_enable=False
```

这个版本不是 teacher anchor。它只把 successful source rollout 的原始 next chunk 注入候选 support 的第 0 个位置；后续仍然用同一套 probe rollout 给所有候选打分，再构造 PowerFlow soft target。也就是说，source chunk 进入分布支持集，但不被硬设满分，不做 score floor。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_sourcechunk_probe4_b32_r32_v64_3step_20260731
TOTAL_TRAINING_STEPS=3
TEST_FREQ=2000000
FINAL_VAL_ENABLE=False
DIAG_JSONL=/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_sourcechunk_probe4_b32_r32_v64_3step_20260731.jsonl
launcher=/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_sourcechunk_probe4_b32_r32_v64_20step_20260731.sh
```

稳定性和注入：

```text
jsonl_rows=96
step1: source_chunk_injected_ratio=1.000 source_chunk_mean_len=232.094
step2: source_chunk_injected_ratio=1.000 source_chunk_mean_len=221.438
step3: source_chunk_injected_ratio=1.000 source_chunk_mean_len=213.312
Final validation skipped
```

目标信号：

```text
step1: selected_original_acc_mean=0.906 positive_ratio=0.341 informative_ratio=0.750 state_all_negative_ratio=0.250 state_mixed_ratio=0.625
step2: selected_original_acc_mean=0.906 positive_ratio=0.279 informative_ratio=0.719 state_all_negative_ratio=0.281 state_mixed_ratio=0.562
step3: selected_original_acc_mean=0.969 positive_ratio=0.310 informative_ratio=0.812 state_all_negative_ratio=0.188 state_mixed_ratio=0.625
```

PowerFlow actor update：

```text
step1: actor/powerflow_loss=0.845 grad_norm=8.119 boxed_reward_mean=0.289
step2: actor/powerflow_loss=0.734 grad_norm=8.753 boxed_reward_mean=0.109
step3: actor/powerflow_loss=0.999 grad_norm=8.869 boxed_reward_mean=0.414
```

耗时：

```text
step1: gen=51.277s chunk_state_probe=6.356s chunk_state_score=11.247s update_actor=8.820s
step2: gen=23.343s chunk_state_probe=6.916s chunk_state_score=10.858s update_actor=8.432s
step3: gen=22.817s chunk_state_probe=6.673s chunk_state_score=11.130s update_actor=8.381s
```

结论：

- 这个 smoke 通过。source chunk 100% 注入，PowerFlow loss 正常，且没有启用 teacher-anchor score floor。
- 相比上一版 success-source/no-anchor 3-step，step3 的 `state_all_negative_ratio` 从 0.219 降到 0.188，`informative_ratio` 从 0.781 升到 0.812；step1 positive ratio 也从 0.310 升到 0.341。
- 这说明“把成功轨迹 next chunk 纳入 support，再用 probe-improved PowerFlow 分布软更新”是更合理的 target construction。下一步进入 20-step gate，看 final validation 能否摆脱前一版 `mean@16=0.4335` 的失败区间。

## 2026-07-31 Source Chunk Support + PowerFlow 20-step Gate

配置延续 3-step smoke：

```text
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
trainer.total_training_steps=20
trainer.test_freq=20
trainer.final_val_enable=True
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_source_chunk_candidate_index=0
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.chunk_weighted_nll_enable=False
actor.use_dynamic_bsz=False
```

运行产物：

```text
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_sourcechunk_probe4_b32_r32_v64_20step_20260731.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_sourcechunk_probe4_b32_r32_v64_20step_20260731.jsonl
diag_jsonl_rows=640
```

最终 validation：

```text
val-core/math/acc/mean@16=0.390125
val-core/math/acc/maj@16/mean=0.5003960000000001
val-core/math/acc/best@16/mean=0.81548
val-aux/math/format_score/mean@16=0.88225
val-aux/math/format_score/maj@16/mean=0.8471900000000001
timing_s/testing=301.412
```

step20 诊断：

```text
chunk_state_source/selected_original_acc_mean=0.906
chunk_state_source_chunk/injected_ratio=1.000
chunk_state_source_chunk/mean_len=227.750
chunk_state_probe/raw_positive_ratio=0.305
chunk_state_diag/state_all_positive_ratio=0.156
chunk_state_diag/state_all_negative_ratio=0.281
chunk_state_diag/state_mixed_ratio=0.562
chunk_state/positive_ratio=0.305
chunk_state/informative_ratio=0.719
actor/powerflow_loss=0.563
actor/grad_norm=12.067
timing_s/gen=22.747
timing_s/chunk_state_probe=6.846
timing_s/chunk_state_score=11.798
timing_s/update_actor=8.485
```

结论：

- 这个 20-step gate 失败，且比上一版 success-source/no-anchor 20-step 还差：`mean@16` 从 `0.4335` 降到 `0.390125`，`maj@16` 从 `0.55015` 降到 `0.500396`。
- PowerFlow loss 已经接在 chunk actor update 上；失败不是因为走成 GRPO 或 weighted NLL，而是 target construction 噪声仍然过大。
- 单纯把 successful source next chunk 加进 support 不够。虽然注入率是 1.0，但 step20 仍有 `state_all_negative_ratio=0.281`，且 positive ratio 只有 `0.305`。
- 下一步继续优先用 PowerFlow distribution matching，但必须先过滤或降权 all-negative / 弱信息 chunk state，避免无区分度的 probe 结果参与 actor update。

## 2026-08-01 Source Chunk + Skip All-negative PowerFlow 3-step Smoke

改动：

```text
ttrl.chunk_state_skip_all_negative=True
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=False
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.chunk_weighted_nll_enable=False
```

实现语义：

- 训练目标仍然是 PowerFlow distribution matching，不切 GRPO，也不切 weighted NLL。
- 对每个 chunk state 仍保留 8 个 candidate 的 batch 形状；如果该 state 的 probe scores 全部为 0，则把该 state 下所有 candidate 的 `powerflow_chunk_weights` 置为 0。
- 这样 all-negative state 不贡献 actor update，同时 FSDP/Ray batch shape 不变，避免极端 batch 回退到噪声更新。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_sourcechunk_skipneg_probe4_b32_r32_v64_3step_20260731
TOTAL_TRAINING_STEPS=3
TEST_FREQ=2000000
FINAL_VAL_ENABLE=False
CHUNK_STATE_SKIP_ALL_NEGATIVE=True
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_sourcechunk_skipneg_probe4_b32_r32_v64_3step_20260731.jsonl
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_sourcechunk_skipneg_probe4_b32_r32_v64_3step_20260731.log
diag_jsonl_rows=96
worker=1024321, 8x NVIDIA B200
```

三步关键信号：

```text
step1: state_all_negative_ratio=0.250 zeroed_state_ratio=0.250 kept_state_ratio=0.750 powerflow_loss=0.267 grad_norm=13.155
step2: state_all_negative_ratio=0.156 zeroed_state_ratio=0.156 kept_state_ratio=0.844 powerflow_loss=0.136 grad_norm=8.953
step3: state_all_negative_ratio=0.219 zeroed_state_ratio=0.219 kept_state_ratio=0.781 powerflow_loss=0.410 grad_norm=18.067
```

三步耗时：

```text
step1: gen=52.039s chunk_state_probe=6.285s chunk_state_score=11.319s update_actor=8.761s
step2: gen=22.295s chunk_state_probe=6.512s chunk_state_score=10.925s update_actor=7.953s
step3: gen=22.701s chunk_state_probe=6.473s chunk_state_score=11.578s update_actor=8.516s
```

结论：

- smoke 通过。PowerFlow chunk actor update 正常，且 all-negative state 的零权重逻辑生效。
- step1 慢主要来自 worker/vLLM/FSDP/JIT warmup；step2/3 回到当前 chunk-state 链路的稳定区间。
- 该版本比 sourcechunk 20-step gate 更合理，因为不会把无信息 state 当成有效 target 训练。下一步可做 20-step gate；如果 20-step 仍失败，应继续改 target construction，而不是回到 GRPO 或 weighted NLL。

## 2026-08-01 Source Chunk + Skip All-negative PowerFlow 20-step Gate

配置延续 3-step smoke：

```text
RUN_ID=ttrl_chunk_state_powerflow_sourcechunk_skipneg_probe4_b32_r32_v64_20step_20260801
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
trainer.total_training_steps=20
trainer.test_freq=20
trainer.final_val_enable=True
ttrl.chunk_state_source_mode=success
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_source_chunk_candidate_index=0
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
ttrl.chunk_state_skip_all_negative=True
ttrl.chunk_state_skip_uniform=False
ttrl.chunk_state_min_informative_gap=0.0
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.chunk_weighted_nll_enable=False
actor.use_dynamic_bsz=False
```

运行产物：

```text
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_sourcechunk_skipneg_probe4_b32_r32_v64_20step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_sourcechunk_skipneg_probe4_b32_r32_v64_20step_20260801.jsonl
diag_jsonl_rows=640
worker=1024321, 8x NVIDIA B200
```

最终 validation：

```text
val-core/math/acc/mean@16=0.401
val-core/math/acc/maj@16/mean=0.506
val-core/math/acc/best@16/mean=0.813
val-aux/math/format_score/mean@16=0.890
val-aux/math/format_score/maj@16/mean=0.853
timing_s/testing=301.707
```

step20 诊断：

```text
chunk_state_source/selected_original_acc_mean=0.875
chunk_state_source/prompt_original_pass=0.875
chunk_state_source/prompt_original_mean=0.309
chunk_state_source_chunk/injected_ratio=1.000
chunk_state_source_chunk/mean_len=233.156
chunk_state_probe/raw_positive_ratio=0.262
chunk_state_diag/state_all_positive_ratio=0.125
chunk_state_diag/state_all_negative_ratio=0.375
chunk_state_diag/state_mixed_ratio=0.500
chunk_state/zeroed_state_ratio=0.375
chunk_state/kept_state_ratio=0.625
chunk_state/target_entropy=1.650
actor/powerflow_loss=0.338
actor/grad_norm=9.563
timing_s/gen=23-25s steady range before validation
timing_s/chunk_state_probe=6.6-7.0s steady range
timing_s/chunk_state_score=11-12s steady range
timing_s/update_actor=7.5-8.3s steady range
```

结论：

- 20-step gate 仍失败：`mean@16=0.401`、`maj@16=0.506`，和 TTRL/MV 对齐基线的 20-step 目标相差很大；相对上一版 sourcechunk/no-anchor `mean@16=0.390125` 只小幅改善。
- PowerFlow chunk actor update 路径已经启用，失败不是因为误跑成 GRPO 或 weighted NLL；当前问题集中在 chunk target construction。
- `skip_all_negative` 过滤是必要但不充分的。step20 仍有 `state_all_negative_ratio=0.375`，有效训练 state 只剩 `0.625`；同时 `raw_positive_ratio=0.262`，说明 future probe 给出的局部分布依然稀疏且噪声大。
- validation 出现明显极长/重复输出，且 `timing_s/testing=301.707s`。这会放大 SymPy/parser timeout，也会使 chunk probe 的 target 更不稳；后续需要增加答案解析 fast path/cache，以及对异常长重复输出做监控。

## TTRL 2504.16084 对 Chunk 设计的约束

从 TTRL 原文方法段抽出的关键约束：

- 原始 TTRL 的 state 是 prompt `x`，action 是完整输出 `y ~ pi_theta(y|x)`。
- 每个 state 必须采多个 candidate outputs `{y_i}`，再由 majority/aggregation 得到 consensus `y*`，最后用 `r(y,y*)` 形成训练 reward。
- 原文强调有效性来自三件事：label estimation、reward calculation、online learning。
- 多输出 rollout 的价值不只是给一个伪标签，而是提供更稠密、更稳健的 reward signal；即使 label 不完全准，rollout 内的多个输出也能通过 negative reward / lucky hit 保留方向性。
- Qwen2.5-Math 设置里，原文使用 `64` responses 做 voting label estimation，再 downsample `32` responses per prompt 训练。

对 chunk-state 版本的含义：

- chunk state 应该是 `query + generated prefix`，但它仍然必须遵守“同一个 state 下多输出估计 reward/distribution”的 TTRL 语义。
- 当前做法把 successful source chunk 注入 support，但本质更像 hard source teacher + short probe，不足以复现 TTRL 原文的多输出稳健 reward 机制。
- 下一版不应回退到 GRPO；仍以 PowerFlow distribution matching 为 actor update 主 loss，但 target 要改成 chunk-level vote/reward accuracy gate：对同一个 state 采多个 next chunk，用这些 chunk 的后续 rollout 聚合出更稳健的 improved distribution，再做 PowerFlow。
- 具体可先做两个 gate：
  1. 只训练 vote/probe 有足够信息量的 state，例如 `max_score - mean_score` 或 positive count 达阈值。
  2. target 不是单条 successful source chunk，而是同一 state 下多 candidate 的 search-improved distribution。

## 2026-08-01 Voteinfo PowerFlow 3-step Smoke

本轮改动目标：

- 取消 `source_chunk` 注入，避免把单条 successful source chunk 当成 hard teacher。
- 取消 `teacher_anchor`，不做 score floor。
- `source_mode=random`，state 来自真实 on-policy full rollout。
- 对同一 state 下 `K=8` 个 next chunk 做 `probe_samples=4`，用 probe score 形成 PowerFlow target。
- 开启 `skip_uniform=True` 和 `skip_all_negative=True`，只训练有正负/强弱差异的 state。
- actor update 仍然是 PowerFlow distribution matching，不切 GRPO，不切 weighted NLL。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_voteinfo_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
TEST_FREQ=2000000
FINAL_VAL_ENABLE=False
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_voteinfo_probe4_b32_r32_v64_3step_20260801.jsonl
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_voteinfo_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl_rows=96
worker=1024321, 8x NVIDIA B200
```

三步关键信号：

```text
step1: all_negative=0.469 all_positive=0.031 mixed=0.500 kept_state_ratio=0.531 positive_ratio=0.205 powerflow_loss=0.059 grad_norm=10.337
step2: all_negative=0.562 all_positive=0.031 mixed=0.406 kept_state_ratio=0.438 positive_ratio=0.131 powerflow_loss=0.000 grad_norm=4.623
step3: all_negative=0.562 all_positive=0.000 mixed=0.438 kept_state_ratio=0.438 positive_ratio=0.108 powerflow_loss=0.053 grad_norm=5.293
```

三步耗时：

```text
step1: gen=51.330s chunk_state_chunks=1.593s chunk_state_probe=6.785s chunk_state_score=11.291s chunk_state_ref=6.484s update_actor=8.727s
step2: gen=23.371s chunk_state_chunks=1.486s chunk_state_probe=6.969s chunk_state_score=10.984s chunk_state_ref=2.684s update_actor=8.477s
step3: gen=23.389s chunk_state_chunks=1.479s chunk_state_probe=7.097s chunk_state_score=11.468s chunk_state_ref=2.694s update_actor=8.559s
```

实现观察：

- smoke 跑通，无 NaN，无 shape/Ray/FSDP 崩溃。
- 这版 target 更接近 TTRL 原文的多输出 reward/distribution 语义：每个 chunk state 的目标来自同一 state 下多个 candidate 的 probe distribution，而不是单条 source chunk。
- 有效 state 比例约 `0.438-0.531`，说明 strict information gate 生效，但仍保留了足够训练样本。
- `source_original_correct_ratio` 只有 `0.062-0.156`，这是随机 source state 的预期结果；这版不依赖 source rollout 正确性，而依赖同一 state 下 chunk candidates 的相对 probe 表现。
- step2 actor 聚合指标显示 `powerflow_loss=0`、`actor/powerflow_weight/mean=0`，但 chunk batch 侧 `powerflow_weight_mean=0.438` 非零；这可能是 actor 微批指标只保留了最后一个全零微批，需要后续核查指标聚合，不应直接解读为整步完全没训练。
- parser timeout 仍然明显，是后续 infra 必修项；但本 smoke 先验证训练语义链路。

结论：

- 可以启动 20-step gate。通过标准仍然是 step20 `mean@16` 接近 MV baseline；如果继续大幅低于 baseline，下一步应增强 target reliability，而不是回退到 source chunk teacher。

## 2026-08-01 Voteinfo PowerFlow 20-step Gate

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_voteinfo_probe4_b32_r32_v64_20step_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_voteinfo_probe4_b32_r32_v64_20step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_voteinfo_probe4_b32_r32_v64_20step_20260801.jsonl
diag_jsonl_rows=640
worker=1024321, 8x NVIDIA B200
```

最终 validation：

```text
val-core/math/acc/mean@16=0.4385
val-core/math/acc/maj@16/mean=0.560
val-core/math/acc/best@16/mean=0.828
val-aux/math/format_score/mean@16=0.894
val-aux/math/format_score/maj@16/mean=0.869392
timing_s/testing=296.079
```

step20 诊断：

```text
chunk_state_source/selected_original_acc_mean=0.125
chunk_state_source/prompt_original_pass=0.906
chunk_state_source/prompt_original_mean=0.362
chunk_state_probe/raw_positive_ratio=0.143
chunk_state_diag/state_all_positive_ratio=0.094
chunk_state_diag/state_all_negative_ratio=0.719
chunk_state_diag/state_mixed_ratio=0.188
chunk_state/kept_state_ratio=0.281
chunk_state/powerflow_weight_mean=0.281
actor/powerflow_loss=0.252
actor/grad_norm=14.964
timing_s/gen=22.132
timing_s/chunk_state_probe=6.560
timing_s/chunk_state_score=12.434
timing_s/update_actor=8.489
timing_s/testing=296.079
```

结论：

- 20-step gate 完整跑完，训练链路没有 Ray/FSDP/NaN 崩溃；此前 master shell 看不到进程是因为不在 worker namespace，worker `1024321` 内训练一直在跑。
- 这版 `mean@16=0.4385`、`maj@16=0.560`，仍然远低于 MV/TTRL 20-step 对齐目标，不可作为有效方法。
- 相比 sourcechunk + skip-negative 版本的 `mean@16=0.401` 有小幅改善，但幅度太小；说明取消 hard source teacher 是对的，但仅靠 `K=8` next chunk + `probe_samples=4` 的 short probe 仍然不能构造可靠 target。
- step20 `state_all_negative_ratio=0.719`、`kept_state_ratio=0.281`，有效 chunk states 太少；同时 `raw_positive_ratio=0.143`，probe 分布非常稀疏。PowerFlow loss 被正确启用，但大部分 update 信号来自少量高噪声局部转移。
- 下一轮不应继续扩大这种 short-probe gate 到 80 step。需要把 chunk target 改成更接近 TTRL 原文 2504.16084 的同 state 多输出 label estimation：对 `query + prefix` state 采多个 next chunk 后，继续 rollout 到完整答案，用 state-level answer majority / consistency 给 next chunk 分配 search-improved weight，而不是直接用短 probe 的稀疏正确率。

## 2026-08-01 Majority-completion PowerFlow 3-step Smoke

实现目标：

- 保留 chunk actor update 的 PowerFlow loss，不切 GRPO，不切 weighted NLL。
- 新增 `ttrl.chunk_state_score_mode=majority_completion`，默认不影响旧实验。
- 对每个 `query + prefix` state 采 `K=8` 个 next chunk，每个 chunk 再采 `probe_samples=4` 个 continuation。
- 在同一 state 的 `K * probe_samples = 32` 条完整 continuation 上做 answer majority，得到 state-local self-supervised label。
- 每个 next chunk 的 score = 该 chunk 的 4 条 continuation 中匹配 state majority answer 的比例。
- 这更贴近 TTRL 2504.16084 的同一 state 多输出 label estimation / reward calculation 语义，同时保持 PowerFlow distribution matching 作为 actor update。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_majority_completion_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
TEST_FREQ=2000000
FINAL_VAL_ENABLE=False
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_majority_completion_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_majority_completion_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
worker=1024321, 8x NVIDIA B200
```

三步关键信号：

```text
step1: majority_ratio=0.343 answer_coverage=0.758 raw_positive=0.343 all_positive=0.156 all_negative=0.000 mixed=0.844 kept_state=1.000 powerflow_weight=1.000 powerflow_loss=0.517 grad_norm=37.760
step2: majority_ratio=0.242 answer_coverage=0.657 raw_positive=0.242 all_positive=0.125 all_negative=0.031 mixed=0.844 kept_state=0.969 powerflow_weight=0.969 powerflow_loss=1.568 grad_norm=42.783
step3: majority_ratio=0.274 answer_coverage=0.698 raw_positive=0.274 all_positive=0.156 all_negative=0.062 mixed=0.781 kept_state=0.938 powerflow_weight=0.938 powerflow_loss=0.835 grad_norm=29.193
```

三步耗时：

```text
step1: gen=51.446s chunk_state_probe=6.800s chunk_state_score=30.783s update_actor=8.853s
step2: gen=27.382s chunk_state_probe=6.944s chunk_state_score=12.337s update_actor=8.578s
step3: gen=24.123s chunk_state_probe=6.954s chunk_state_score=9.907s update_actor=8.606s
```

结论：

- smoke 跑通，无 NaN/Ray/FSDP 崩溃。
- target density 明显优于 short-probe correct-rate：上一版 20-step gate step20 `state_all_negative_ratio=0.719`、`kept_state_ratio=0.281`；本版 3-step smoke 的 `state_all_negative_ratio=0.000/0.031/0.062`、`kept_state_ratio=1.000/0.969/0.938`。
- `answer_coverage=0.657-0.758`，说明多数 completion 能抽到答案；但 `majority_ratio=0.242-0.343`，state-local majority 本身还不够尖锐，后续可能需要 confidence sharpening / majority margin gate。
- 首步 `chunk_state_score=30.783s` 受 parser/JIT/冷启动影响，step2/3 降到 `12.337s/9.907s`，与旧 short-probe 版本接近；parser timeout 仍是 infra 优化重点。
- 可以启动 20-step gate。通过标准：至少不能出现 short-probe 版那种 `mean@16~0.44` 崩坏；若 20-step 指标仍差，下一步加 majority confidence gate，而不是回退 source chunk teacher。

## 2026-08-01 Majority-completion PowerFlow 20-step Gate

论文依据：

- 用户补充参考 arXiv 2504.16084。原文 TTRL 把 prompt `x` 视为 state，对同一 state 重复采样多个输出 `{y_i}`，用 majority voting 得到 consensus label `y*`，再按输出是否匹配 `y*` 构造 reward。
- 本轮实现是该语义的 chunk-state 版本：把 state 从原始 prompt 扩展为 `query + prefix`，对同一 chunk state 采多个 next chunk，并对 `state + chunk` 的后续 completions 做 state-local majority label estimation。
- actor update 仍使用 PowerFlow loss 做 distribution matching，不切 GRPO，不切 weighted NLL。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_majority_completion_probe4_b32_r32_v64_20step_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
TTRL_RUNTIME_DIR=/tmp/cmc20
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_majority_completion_probe4_b32_r32_v64_20step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_majority_completion_probe4_b32_r32_v64_20step_20260801.jsonl
diag_jsonl_rows=640
worker=1024321, 8x NVIDIA B200
```

最终 validation：

```text
val-core/math/acc/mean@16=0.521125
val-core/math/acc/maj@16/mean=0.662318
val-core/math/acc/best@16/mean=0.872676
val-aux/math/format_score/mean@16=0.904125
val-aux/math/format_score/maj@16/mean=0.893140
timing_s/testing=301.144
```

step20 诊断：

```text
chunk_state_source/selected_original_acc_mean=0.188
chunk_state_source/prompt_original_pass=0.875
chunk_state_source/prompt_original_mean=0.415
chunk_state_majority_completion/majority_ratio_mean=0.301
chunk_state_majority_completion/answer_coverage_mean=0.618
chunk_state_majority_completion/raw_positive_ratio=0.301
chunk_state_diag/state_all_positive_ratio=0.188
chunk_state_diag/state_all_negative_ratio=0.094
chunk_state_diag/state_mixed_ratio=0.719
chunk_state/kept_state_ratio=0.906
chunk_state/target_entropy=1.390
actor/powerflow_loss=0.345
actor/boxed_reward/mean=0.250
actor/grad_norm=8.056
timing_s/gen=21.999
timing_s/chunk_state_probe=6.878
timing_s/chunk_state_score=11.037
timing_s/update_actor=8.708
```

step2-20 平均耗时/信号：

```text
timing_s/gen=23.626
timing_s/chunk_state_probe=7.074
timing_s/chunk_state_score=11.120
timing_s/update_actor=8.639
chunk_state_majority_completion/majority_ratio_mean=0.263
chunk_state_diag/state_mixed_ratio=0.867
chunk_state/kept_state_ratio=0.972
```

结论：

- 20-step gate 完整跑完，训练和 final validation 均成功，无 NaN/Ray/FSDP 崩溃。
- 相比 sourcechunk/no-anchor、skip-all-negative、voteinfo short-probe 三个失败版本，这版明显改善：`mean@16` 从 `0.390/0.401/0.4385` 提升到 `0.521125`，`maj@16` 从 `0.500/0.506/0.560` 提升到 `0.662318`。
- 但它仍显著低于 MV/TTRL 20-step 对齐基线（`mean@16~0.76`、`maj@16~0.82`），不能扩到 80 step 当主结果。
- 主要问题不是链路崩溃，而是 target 仍然不够可靠：step20 `majority_ratio_mean=0.301`、`answer_coverage_mean=0.618`，说明 state-local majority 可以提供信号，但多数标签不够尖锐；如果直接 PowerFlow 蒸馏，会把相当多低置信局部偏好也学进去。
- 工程瓶颈仍集中在 full rollout、parser/scoring 和 actor update：稳态 `gen~23.6s`、`chunk_state_score~11.1s`、`chunk_state_probe~7.1s`、`update_actor~8.6s`。日志里大量 SymPy warning/timeout 和每步 parser subprocess shutdown，后续 80/160 step 前应做 parser cache / process pool 复用。

下一步：

- 不继续扩大当前 raw majority-completion 到 80 step。
- 做 confidence-gated majority-completion：只训练 `majority_ratio`、`answer_coverage`、`score margin` 足够的 state；低置信 state 要么跳过，要么降低 PowerFlow weight。
- 同时试 `chunk_size=128`，更贴近 chunked search / PowerFlow 的局部转移粒度，并降低 actor chunk response 长度。
- 保持 PowerFlow loss 为主路径，GRPO 只作为后续 ablation。

## 2026-08-01 Confidence-gated Majority-completion c128 3-step Smoke

实现目标：

- 保持 PowerFlow loss 为 chunk actor update 主路径。
- 在 `majority_completion` 中把 state-local `majority_ratio` 和 `answer_coverage` 写入 state batch。
- actor batch 构造阶段新增 confidence gate：
  - `chunk_state_min_majority_ratio=0.25`
  - `chunk_state_min_answer_coverage=0.60`
  - `chunk_state_confidence_power=0.5`
- 低置信 state 被跳过，高置信 state 按 `(majority_ratio * answer_coverage) ** 0.5` 降权。
- `chunk_state_chunk_size=128`，boundary 改为 `[0,128,256,384,512,640,768,896,1024]`，更贴近 chunk-level search 的局部转移粒度。
- 默认配置保持中性：阈值 0、power 0，不影响旧脚本和对照实验。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_majority_conf_gate_c128_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
TEST_FREQ=2000000
FINAL_VAL_ENABLE=False
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_majority_conf_gate_c128_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_majority_conf_gate_c128_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
worker=1024321, 8x NVIDIA B200
```

三步关键信号：

```text
step1: majority_ratio=0.364 answer_coverage=0.706 confidence_gate=0.500 kept_state=0.500 loss_weight=0.379 response_len=126.102 powerflow_loss=0.377 grad_norm=6.883
step2: majority_ratio=0.333 answer_coverage=0.777 confidence_gate=0.500 kept_state=0.500 loss_weight=0.354 response_len=124.938 powerflow_loss=0.228 grad_norm=5.797
step3: majority_ratio=0.264 answer_coverage=0.632 confidence_gate=0.438 kept_state=0.438 loss_weight=0.285 response_len=119.875 powerflow_loss=0.094 grad_norm=2.210
```

三步耗时：

```text
step1: gen=51.290s chunk_state_chunks=1.200s chunk_state_probe=6.953s chunk_state_score=11.262s update_actor=8.295s
step2: gen=23.990s chunk_state_chunks=0.956s chunk_state_probe=7.302s chunk_state_score=10.262s update_actor=7.460s
step3: gen=23.265s chunk_state_chunks=0.963s chunk_state_probe=7.109s chunk_state_score=11.518s update_actor=7.446s
```

diag 聚合：

```text
diag rows=96
majority_ratio mean=0.3203 min=0.0000 max=0.9688
answer_coverage mean=0.7051 min=0.0000 max=1.0000
```

结论：

- smoke 成功，无 NaN/Ray/FSDP 崩溃，confidence gate 按预期生效。
- 相比 raw majority-completion c256，c128 把 actor response span 从约 228-249 token 降到约 120-126 token，`update_actor` 从约 `8.6s` 降到约 `7.45s`，`chunk_state_chunks` 从约 `1.5s` 降到约 `1.0s`。
- gate 没有过紧：三步保留 `43.8%-50.0%` state，仍有可训练信号。
- 风险是信号可能过弱：step3 `powerflow_loss=0.094`、`grad_norm=2.210` 已明显偏小；20-step gate 需要关注是否因为降权过强导致学习不足。

下一步：

- 启动同配置 20-step gate，看 final `mean@16/maj@16/best@16` 是否优于 raw majority-completion `0.521/0.662/0.873`。
- 若 20-step 低于 raw majority-completion，优先降低 `confidence_power` 或 `min_answer_coverage`，而不是取消 confidence gate。

## 2026-08-01 Confidence-gated Majority-completion c128 20-step

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_majority_conf_gate_c128_probe4_b32_r32_v64_20step_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
chunk_state_chunk_size=128
chunk_state_min_majority_ratio=0.25
chunk_state_min_answer_coverage=0.60
chunk_state_confidence_power=0.5
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_majority_conf_gate_c128_probe4_b32_r32_v64_20step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_majority_conf_gate_c128_probe4_b32_r32_v64_20step_20260801.jsonl
diag_jsonl_rows=640
worker=1024321, 8x NVIDIA B200
```

Final validation：

```text
val-core/math/acc/mean@16=0.5325
val-core/math/acc/maj@16/mean=0.666974
val-core/math/acc/best@16/mean=0.877136
val-aux/math/format_score/mean@16=0.913625
val-aux/math/format_score/maj@16/mean=0.905506
timing_s/testing=291.097
```

step20 诊断：

```text
chunk_state_source/selected_original_acc_mean=0.156
chunk_state_source/prompt_original_pass=0.906
chunk_state_source/prompt_original_mean=0.406
chunk_state_majority_completion/majority_ratio_mean=0.403
chunk_state_majority_completion/answer_coverage_mean=0.784
chunk_state_majority_completion/raw_positive_ratio=0.404
chunk_state_diag/state_all_positive_ratio=0.344
chunk_state_diag/state_all_negative_ratio=0.031
chunk_state_diag/state_mixed_ratio=0.625
chunk_state/confidence_gate_ratio=0.625
chunk_state/kept_state_ratio=0.562
chunk_state/loss_weight_mean=0.388
chunk_state/target_entropy=1.348
actor/powerflow_loss=0.198
actor/boxed_reward/mean=0.258
actor/grad_norm=3.219
timing_s/gen=21.715
timing_s/chunk_state_probe=6.827
timing_s/chunk_state_score=12.285
timing_s/update_actor=7.448
```

对比：

```text
raw majority-completion c256 20-step:
  mean@16=0.521125 maj@16=0.662318 best@16=0.872676 format_mean@16=0.904125

confidence-gated c128 20-step:
  mean@16=0.532500 maj@16=0.666974 best@16=0.877136 format_mean@16=0.913625
```

结论：

- 这轮完整跑完，训练和 final validation 均成功。
- c128 + confidence gate 相比 raw majority-completion 有小幅提升：`mean@16 +0.0114`，`maj@16 +0.0047`，`best@16 +0.0045`，format 也略好。
- 但提升幅度太小，仍远低于 MV/TTRL 20-step 对齐目标（`mean@16~0.76`、`maj@16~0.82`），不能作为主线扩到 80 step。
- hard gate 的问题很明确：训练信号稀疏且波动大，step12 只保留 `18.8%` state，`loss_weight_mean=0.119`，`grad_norm=1.267`；后半段虽有恢复，但整体没有形成足够强的 policy improvement。
- 这说明当前版本不是“只要过滤低置信 state 就会好”，而是需要重新定义 state-local target。更合理的方向是参考 arXiv 2504.16084：同一个 state 下采多个输出，先估计 pseudo-label / majority reward，再以软分布或置信加权方式做 PowerFlow 蒸馏，而不是 hard skip 大量 state。

下一步：

- 不继续 hard-gated c128 80-step。
- 做 2504.16084 风格的 soft multi-output label estimation：保留全部 informative state，只用 `majority_ratio`、`answer_coverage`、候选得分分布调整 PowerFlow target sharpness/weight。
- 优先取消 hard `min_answer_coverage=0.60`，改成软置信权重；目标是恢复足够训练信号，同时避免 raw majority-completion 的低置信噪声。
- 继续使用 PowerFlow loss 作为 chunk actor update 主路径，GRPO 暂不作为主实验。

## 2026-08-01 Soft-confidence Majority-completion c128 3-step Smoke

动机：

- hard confidence gate 的 20-step 只小幅好于 raw majority-completion，核心问题是过早丢掉大量 state。
- 按 arXiv 2504.16084 的语义，同一个 state 下多输出应先用于 label/reward estimation，再通过置信度调节学习强度；不应把低 coverage state 全部 hard skip。
- 因此这轮保留 c128、majority-completion、probe4、PowerFlow loss，只取消 hard gate：
  - `chunk_state_min_majority_ratio=0.0`
  - `chunk_state_min_answer_coverage=0.0`
  - `chunk_state_confidence_power=0.5`

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_majority_soft_conf_c128_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
TEST_FREQ=2000000
FINAL_VAL_ENABLE=False
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_majority_soft_conf_c128_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_majority_soft_conf_c128_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
worker=1024321, 8x NVIDIA B200
```

三步关键信号：

```text
step1: kept_state=0.969 loss_weight=0.483 majority_ratio=0.364 answer_coverage=0.706 powerflow_loss=0.386 grad_norm=9.195 gen=52.106s chunk_state_score=11.604s update_actor=7.677s
step2: kept_state=0.938 loss_weight=0.338 majority_ratio=0.217 answer_coverage=0.608 powerflow_loss=0.104 grad_norm=6.089 gen=23.148s chunk_state_score=10.812s update_actor=7.438s
step3: kept_state=0.969 loss_weight=0.422 majority_ratio=0.321 answer_coverage=0.666 powerflow_loss=0.192 grad_norm=2.020 gen=26.209s chunk_state_score=10.743s update_actor=7.626s
```

结论：

- smoke 成功，无 NaN/Ray/FSDP 崩溃。
- 与 hard gate c128 相比，soft-confidence 保留 `93.8%-96.9%` state，而不是 `18.8%-62.5%`；训练信号明显更连续。
- `loss_weight_mean=0.338-0.483`，没有塌到 hard gate step12 的 `0.119`。
- `update_actor~7.4-7.7s`，仍保留 c128 对 actor update 的收益。
- 这版比 hard gate 更适合作为 2504.16084 风格 multi-output label estimation 的最小可行实验，下一步跑 20-step final validation。

## 2026-08-01 Soft-confidence Majority-completion c128 20-step

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_majority_soft_conf_c128_probe4_b32_r32_v64_20step_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
chunk_state_chunk_size=128
chunk_state_min_majority_ratio=0.0
chunk_state_min_answer_coverage=0.0
chunk_state_confidence_power=0.5
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_majority_soft_conf_c128_probe4_b32_r32_v64_20step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_majority_soft_conf_c128_probe4_b32_r32_v64_20step_20260801.jsonl
diag_jsonl_rows=640
worker=1024321, 8x NVIDIA B200
```

Final validation：

```text
val-core/math/acc/mean@16=0.370000
val-core/math/acc/maj@16/mean=0.475650
val-core/math/acc/best@16/mean=0.810974
val-aux/math/format_score/mean@16=0.871500
val-aux/math/format_score/maj@16/mean=0.822820
timing_s/testing=308.964
```

训练耗时统计（step2-step20 平均）：

```text
timing_s/gen=24.848
timing_s/chunk_state_score=12.343
timing_s/update_actor=7.569
chunk_state/kept_state_ratio=0.961
chunk_state/loss_weight_mean=0.411
chunk_state/majority_ratio_mean=0.287
chunk_state/answer_coverage_mean=0.658
actor/powerflow_loss=0.251
actor/grad_norm=5.124
```

step20 诊断：

```text
chunk_state_source/selected_original_acc_mean=0.094
chunk_state_source/prompt_original_pass=0.906
chunk_state_source/prompt_original_mean=0.303
chunk_state_majority_completion/majority_ratio_mean=0.327
chunk_state_majority_completion/answer_coverage_mean=0.686
chunk_state_majority_completion/raw_positive_ratio=0.328
chunk_state_diag/state_all_positive_ratio=0.188
chunk_state_diag/state_all_negative_ratio=0.000
chunk_state_diag/state_mixed_ratio=0.812
chunk_state/confidence_gate_ratio=1.000
chunk_state/kept_state_ratio=1.000
chunk_state/loss_weight_mean=0.446
chunk_state/target_entropy=1.249
actor/powerflow_loss=0.191
actor/boxed_reward/mean=0.297
actor/grad_norm=5.178
timing_s/gen=22.596
timing_s/chunk_state_probe=6.668
timing_s/chunk_state_score=9.989
timing_s/update_actor=7.541
```

对比：

```text
raw majority-completion c256 20-step:
  mean@16=0.521125 maj@16=0.662318 best@16=0.872676 format_mean@16=0.904125

confidence-gated c128 20-step:
  mean@16=0.532500 maj@16=0.666974 best@16=0.877136 format_mean@16=0.913625

soft-confidence c128 20-step:
  mean@16=0.370000 maj@16=0.475650 best@16=0.810974 format_mean@16=0.871500
```

结论：

- 这轮完整跑完，但结果明显失败；soft-confidence 20-step 反而低于 raw majority-completion 和 hard gate。
- smoke 阶段看到的“训练信号更连续”没有转化为最终 accuracy，说明只用连续置信权重会把大量低置信、弱对齐甚至错误的 chunk pseudo-label 一起蒸馏进 actor。
- `kept_state_ratio` 平均达到 `0.961`，但 `majority_ratio_mean` 只有 `0.287`，`answer_coverage_mean` 只有 `0.658`；这意味着保留了太多没有稳定 pseudo-label 的 state。
- actor update 仍然维持 `~7.6s`，证明 chunk actor update 的 infra 方向是成立的；失败主要是训练语义，不是 actor update 性能。
- validation format 分数也下降到 `0.8715`，说明错误 target 不只是没有提升推理正确性，还伤到了输出格式稳定性。

下一步：

- 不扩 soft-confidence 到 80-step。
- 转向更贴近 arXiv 2504.16084 的 state-level label estimation：同一 state 下先用多 completion 得到稳定 pseudo-label / majority reward，只对 label-consistent 的 continuation 构造 PowerFlow target。
- 保留 PowerFlow loss 作为 chunk actor update 主路径，但 target 要从“所有候选按弱 reward 连续加权”改成“先估计 state label，再蒸馏 search-improved distribution”。
- 优先做一个 3-step smoke：`state_label_estimation + label_consistent_powerflow`，检查 pseudo-label 覆盖率、label-consistent candidate ratio、target entropy、format 分数，再决定是否跑 20-step。

## 2026-08-01 Label-consistent Majority-completion c128 3-step Smoke

动机：

- soft-confidence 20-step 失败说明“保留所有 informative state + 连续置信权重”会把低置信/错误 chunk target 一起蒸馏进去。
- 按 arXiv 2504.16084 的语义，应该先对同一个 state 的多输出估计 pseudo-label，再把匹配 pseudo-label 的输出作为 reward/target。
- 因此这轮新增 `chunk_state_label_consistent_only=True`：仍用 state 下 `candidates * probe_samples = 32` 个 completion 做 majority label estimation，但 PowerFlow target 只从至少一个 probe completion 匹配该 label 的 chunk candidate 构造。

代码变更：

```text
verl/trainer/config/ppo_trainer_ttrl.yaml:
  ttrl.chunk_state_label_consistent_only=false  # 默认关闭，保持旧实验语义

verl/trainer/ppo/ray_trainer.py:
  _score_chunk_state_majority_completion 写入 chunk_state_label_consistent[state, candidate]
  _build_chunk_actor_batch 在开关启用时用 label_consistent mask 构造 PowerFlow target
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_label_consistent_c128_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
TEST_FREQ=2000000
FINAL_VAL_ENABLE=False
chunk_state_chunk_size=128
chunk_state_min_majority_ratio=0.25
chunk_state_min_answer_coverage=0.60
chunk_state_confidence_power=0.0
chunk_state_label_consistent_only=True
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_label_consistent_c128_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_label_consistent_c128_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
worker=1024321, 8x NVIDIA B200
```

三步关键信号：

```text
step1: kept_state=0.500 label_consistent=0.602 loss_weight=0.500 target_entropy=1.440 powerflow_loss=1.442 grad_norm=53.314 gen=51.293s chunk_state_score=11.424s update_actor=7.634s
step2: kept_state=0.531 label_consistent=0.590 loss_weight=0.531 target_entropy=1.314 powerflow_loss=0.661 grad_norm=44.746 gen=24.316s chunk_state_score=10.014s update_actor=7.356s
step3: kept_state=0.438 label_consistent=0.512 loss_weight=0.438 target_entropy=1.231 powerflow_loss=0.238 grad_norm=33.374 gen=22.876s chunk_state_score=10.589s update_actor=7.591s
```

结论：

- smoke 成功，无 NaN、Ray/FSDP/vLLM 崩溃。
- `label_consistent_ratio=0.512-0.602`，说明同一 state 的 majority label 能过滤掉约 40%-49% chunk candidate；这比 soft-confidence 的“全候选弱加权”更接近论文的 label-estimation reward 语义。
- `kept_state_ratio=0.438-0.531`，比 soft-confidence 稀疏，但不像 hard gate step12 那样塌到 `0.188`。
- `grad_norm=53.314 -> 44.746 -> 33.374` 偏高但下降，没有发散；需要 20-step final validation 判断是否带来 accuracy 改善。
- actor update 仍保持 `~7.3-7.6s`，chunk actor update 的 infra 收益保留。

下一步：

- 跑同配置 20-step final validation，若 `mean@16` 明显高于 hard gate `0.5325` 或接近 raw MV 20-step 基线，再考虑 80-step。
- 如果 20-step 仍差，下一轮不再调权重，而改 state construction：减少 prompt-only/early state，优先从 high-pass full rollout 中截取中后段 state，提升 pseudo-label 与最终正确性的相关性。

## 2026-08-01 Label-consistent Majority-completion c128 20-step Final Validation

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_label_consistent_c128_probe4_b32_r32_v64_20step_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
train_batch_size=32
rollout.n=32
chunk_state_candidates=8
chunk_state_chunk_size=128
chunk_state_probe_samples=4
chunk_state_probe_max_tokens=1024
chunk_state_min_majority_ratio=0.25
chunk_state_min_answer_coverage=0.60
chunk_state_confidence_power=0.0
chunk_state_label_consistent_only=True
chunk_state_skip_all_negative=True
chunk_state_skip_uniform=True
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.use_dynamic_bsz=False
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_label_consistent_c128_probe4_b32_r32_v64_20step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_label_consistent_c128_probe4_b32_r32_v64_20step_20260801.jsonl
diag_jsonl_rows=640
worker=1024321, 8x NVIDIA B200
```

final validation：

```text
val-core/math/acc/mean@16=0.517500
val-core/math/acc/maj@16/mean=0.658620
val-core/math/acc/best@16/mean=0.872798
val-aux/math/format_score/mean@16=0.911875
val-aux/math/format_score/maj@16/mean=0.896806
timing_s/testing=298.511
```

20-step 训练平均：

```text
timing_s/gen=25.199
timing_s/chunk_state_probe=6.908
timing_s/chunk_state_score=11.647
timing_s/chunk_state_ref=2.568
timing_s/update_actor=7.519
chunk_state/kept_state_ratio=0.488
chunk_state/label_consistent_ratio=0.552
actor/powerflow_loss=0.323
actor/grad_norm=11.706
```

step20 诊断：

```text
chunk_state_source/selected_original_acc_mean=0.250
chunk_state_source/prompt_original_pass=0.938
chunk_state_source/prompt_original_mean=0.419
chunk_state_majority_completion/majority_ratio_mean=0.358
chunk_state_majority_completion/answer_coverage_mean=0.706
chunk_state_majority_completion/raw_positive_ratio=0.359
chunk_state_majority_completion/label_consistent_ratio=0.613
chunk_state_diag/state_all_positive_ratio=0.375
chunk_state_diag/state_all_negative_ratio=0.062
chunk_state_diag/state_mixed_ratio=0.562
chunk_state/kept_state_ratio=0.531
chunk_state/target_entropy=1.448
actor/powerflow_loss=0.029
actor/boxed_reward/mean=0.172
actor/grad_norm=4.687
timing_s/gen=21.564
timing_s/chunk_state_probe=6.831
timing_s/chunk_state_score=10.773
timing_s/update_actor=7.374
```

与前几轮 20-step 对比：

```text
raw majority-completion c256:
  mean@16=0.521125 maj@16=0.662318 best@16=0.872676 format_mean@16=0.904125

confidence-gated c128:
  mean@16=0.532500 maj@16=0.666974 best@16=0.877136 format_mean@16=0.913625

soft-confidence c128:
  mean@16=0.370000 maj@16=0.475650 best@16=0.810974 format_mean@16=0.871500

label-consistent c128:
  mean@16=0.517500 maj@16=0.658620 best@16=0.872798 format_mean@16=0.911875
```

结论：

- label-consistent 版本完整跑完，但没有超过 hard confidence gate，也略低于 raw majority-completion。
- 这说明仅仅把候选过滤成“与 state-level majority label 一致”还不够；当前 chunk state 的 pseudo-label 与最终正确性相关性仍弱。
- `label_consistent_ratio` 平均 `0.552`，`kept_state_ratio` 平均 `0.488`，过滤强度是合理的；失败不是因为所有信号都被过滤掉，而是被保留的 target 仍然不够好。
- actor update 仍稳定在 `~7.5s`，chunk actor update 的 infra 方向继续成立；端到端主要成本在 rollout generation 和 `chunk_state_score` 的 parser/scoring。
- step19 的 `chunk_state_score=29.816s` 是 SymPy comparison timeout 拉高的典型例子；parser/scoring 是后续 infra 优化点。
- final validation 里出现大量重复 `boxed{}` 的超长输出，说明训练已经明显产生格式退化风险；虽然 format mean 仍有 `0.911875`，但长文本重复会拖慢 validation 和日志写入。

arXiv 2504.16084 对下一步的约束：

- 论文里的 TTRL 把 prompt `x` 视为 state，同一 state 下采多个 outputs `{y_i}`，用 majority/aggregation 得到 consensus `y*`，再用 `r(y, y*)` 作为 reward。
- 论文还强调 multiple outputs within a rollout 能提高 reward 对 pseudo-label 错误的鲁棒性。
- 我们的 chunk-state 版本要保持这个语义：把 state 从 prompt 扩展成 `query + partial reasoning`，但同一个 state 下必须先做稳定 label estimation，再训练 state-conditional next-chunk improver。
- 当前失败说明“随机 full rollout 截 prefix + 局部 completion majority”还没有形成足够可靠的 state label；下一轮应改 state construction，而不是继续调权重。

下一步：

- 不扩 label-consistent c128 到 80-step。
- 做 mid-state / high-pass state construction：先采 `32` 条完整 rollout，优先从 prompt-level pass 或高 majority-ratio 的轨迹中截取中后段 state，减少 prompt-only/early/noisy state。
- 对每个 state 仍按 2504.16084 语义采多 completion 做 label estimation，但 PowerFlow target 只蒸馏更稳定的 search-improved next-chunk distribution。
- 保持 `actor.use_dynamic_bsz=False` 和 PowerFlow loss 主路径不变，先跑 3-step smoke，再跑 20-step gate；如果 20-step 不能明显超过 hard gate `mean@16=0.5325`，需要引入更强 verifier/probe，而不是再调 chunk 权重。

## 2026-08-01 Majority-consistent Mid-state c128 3-step Smoke

动机：

- label-consistent c128 20-step 失败后，判断主要问题不在 PowerFlow loss，而在 state construction。
- 随机 source 经常从错误/早期 noisy rollout 截 prefix，导致局部 majority label 与最终正确性弱相关。
- 新增 `chunk_state_source_mode=majority_consistent`：先对同一个 prompt 的 32 条完整 rollout 做 majority pseudo-label，只优先选匹配该 prompt-level majority label 的 source rollout；这不使用真实 GT，符合 2504.16084 的无标签 majority label estimation 语义。
- 新增 `chunk_state_boundary_mode=mid`：从中段 prefix 截 state，减少 prompt-only/early state。

代码变更：

```text
verl/trainer/ppo/ray_trainer.py:
  _compute_full_rollout_majority_consistency 只用 full rollout outputs 计算 prompt-level majority consistency
  _make_chunk_state_prompts 支持 source_mode=majority_consistent
  _make_chunk_state_prompts 支持 boundary_mode=mid
  diag/jsonl 增加 source_majority_consistent 和 source_prompt_majority_ratio

verl/trainer/config/ppo_trainer_ttrl.yaml:
  chunk_state_boundary_mode=cycle
  chunk_state_mid_boundary_min_ratio=0.25
  chunk_state_mid_boundary_max_ratio=0.80
  默认 source_mode 仍为 random，旧实验语义不变
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_majority_consistent_mid_c128_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
chunk_state_source_mode=majority_consistent
chunk_state_boundary_mode=mid
chunk_state_source_chunk_enable=True
chunk_state_label_consistent_only=True
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_majority_consistent_mid_c128_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_majority_consistent_mid_c128_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
```

三步关键信号：

```text
step1: source_original=0.844 source_majority_consistent=1.000 prompt_majority_ratio=0.343 chunk_majority_ratio=0.482 label_consistent=0.668 kept=0.594 boundary_mean=624.0 loss=0.166 grad=15.238 gen=51.147s score=9.694s update=7.722s
step2: source_original=0.719 source_majority_consistent=1.000 prompt_majority_ratio=0.337 chunk_majority_ratio=0.489 label_consistent=0.719 kept=0.594 boundary_mean=512.0 loss=0.000 grad=14.832 gen=22.722s score=27.418s update=6.802s
step3: source_original=0.812 source_majority_consistent=1.000 prompt_majority_ratio=0.340 chunk_majority_ratio=0.486 label_consistent=0.711 kept=0.688 boundary_mean=460.0 loss=0.748 grad=5.177 gen=22.225s score=9.365s update=7.002s
```

三步平均：

```text
source_original_correct=0.792
source_majority_consistent=1.000
prompt_majority_ratio=0.340
chunk_majority_ratio=0.486
chunk_label_consistent_ratio=0.699
kept_state_ratio=0.625
timing_s/gen=32.031
timing_s/chunk_state_score=15.492
timing_s/update_actor=7.175
grad_norm=11.749
```

结论：

- smoke 成功，无 NaN/Ray/FSDP/vLLM 崩溃。
- 关键改进成立：source selection 没有用真实 GT，但事后看 `source_original_correct` 平均达到 `0.792`，远高于随机 source 的 `~0.1-0.25`。
- chunk-level majority ratio 提高到 `~0.486`，label-consistent ratio 提高到 `~0.699`，说明高质量 source + mid-state 让局部 completion majority 更稳定。
- actor update 仍是 `~7.2s`，保持 chunk actor update 的 infra 收益。
- step2 的 `chunk_state_score=27.418s` 仍受 SymPy/parser timeout 影响，这不是本轮语义问题。
- 下一步可以跑同配置 20-step final validation；gate 是必须超过 hard confidence gate `mean@16=0.5325`，否则需要继续增强 verifier/probe，而不是只扩到 80-step。

## 2026-08-01 Majority-consistent Mid-state c128 20-step Gate

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_majority_consistent_mid_c128_probe4_b32_r32_v64_20step_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
chunk_state_source_mode=majority_consistent
chunk_state_boundary_mode=mid
chunk_state_source_chunk_enable=True
chunk_state_label_consistent_only=True
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_majority_consistent_mid_c128_probe4_b32_r32_v64_20step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_majority_consistent_mid_c128_probe4_b32_r32_v64_20step_20260801.jsonl
diag_jsonl_rows=640
```

Final validation：

```text
mean@16=0.486625
maj@16=0.619360
best@16=0.863332
format_mean@16=0.898000
format_maj@16=0.879554
testing=295.888s
```

20-step 平均训练信号：

```text
source_original_correct=0.752
source_majority_consistent=1.000
source_pseudo_acc=0.853
chunk_majority_ratio=0.466
chunk_label_consistent_ratio=0.677
kept_state_ratio=0.619
positive_ratio=0.467
probe_mean_source_original_correct=0.531
probe_mean_source_original_wrong=0.262
timing_s/gen=26.320
timing_s/chunk_state_chunks=1.006
timing_s/chunk_state_probe=6.617
timing_s/chunk_state_score=10.832
timing_s/chunk_state_ref=2.511
timing_s/update_actor=7.538
```

与当前短程 gate 对比：

```text
hard confidence-gated c128:
  mean@16=0.532500 maj@16=0.666974 best@16=0.877136 format_mean@16=0.913625

majority-consistent mid-state c128:
  mean@16=0.486625 maj@16=0.619360 best@16=0.863332 format_mean@16=0.898000
```

结论：

- 这轮 20-step gate 明确失败，不能扩到 80-step。
- `majority_consistent` source construction 本身有效：没有使用真实 GT，但事后真实正确率平均 `0.752`，pseudo acc 平均 `0.853`。
- 失败点在局部 target：同一个 chunk state 下的 completion majority 平均只有 `0.466`，即便 source 更可靠，next-chunk search distribution 仍不够干净。
- 这说明 arXiv 2504.16084 的“同一 state 多输出聚合 label/reward”语义要继续保留，但 state 变成 `query + partial reasoning` 后，不能只用短 probe 的局部 completion majority 当训练目标。
- 下一步应改 label estimation / verifier，而不是继续调 PowerFlow 权重：例如对 chunk candidate 做更长 horizon probe、复用完整 rollout 的 future success、或引入 answer-level verifier 聚合，保证 improved distribution 真正指向最终正确性。
- infra 侧 actor update 已经稳定在 `~7.5s`；当前端到端主要由 full rollout `~26.3s`、chunk probe `~6.6s`、parser/scoring `~10.8s` 和 validation `~296s` 构成。parser timeout 仍是明显噪声源。

## 2026-08-01 Teacher-anchor Mid-state c128 3-step Smoke

动机：

- majority-consistent mid-state 20-step gate 失败后，判断失败点是局部 target 不够干净，而不是 source construction 或 PowerFlow actor update 崩掉。
- 按 24h goal 的下一步，应改 label estimation / verifier，而不是继续调 PowerFlow 权重。
- 本轮引入 teacher-anchor：把 source full rollout 在 boundary 之后的真实 next chunk 注入候选槽 0，并给它 `score_floor=1.0`。
- 这相当于先复用 full rollout 的 future-success path 作为一个 search anchor，再让 PowerFlow 在同一 state 的候选分布上蒸馏；仍然不使用真实 GT 选择 source。

代码变更：

```text
verl/trainer/ppo/ray_trainer.py:
  teacher_anchor score floor 生效时，同步把 anchor candidate 的 chunk_state_label_consistent 置 1
  避免 ttrl.chunk_state_label_consistent_only=True 时 anchor 被 mask 掉

run_records/ttrl_chunk_state_powerflow_teacher_anchor_mid_c128_probe4_b32_r32_v64_3step_20260801.sh:
  复用 majority-consistent mid-state c128 配置
  打开 chunk_state_teacher_anchor_enable=True
  关闭 source_chunk_enable，避免两个 anchor 机制重复
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_teacher_anchor_mid_c128_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
chunk_state_source_mode=majority_consistent
chunk_state_boundary_mode=mid
chunk_state_teacher_anchor_enable=True
chunk_state_teacher_anchor_candidate_index=0
chunk_state_teacher_anchor_score=1.0
chunk_state_source_chunk_enable=False
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_teacher_anchor_mid_c128_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_teacher_anchor_mid_c128_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
```

三步平均：

```text
teacher_anchor_replaced_ratio=1.000
teacher_anchor_label_consistent_forced=1.000
source_original_correct=0.812
source_majority_consistent=1.000
chunk_majority_ratio=0.473
chunk_majority_label_consistent_ratio=0.673
positive_ratio_after_anchor=0.535
label_consistent_ratio_after_anchor=0.698
kept_state_ratio=0.635
target_entropy=1.382
timing_s/gen=32.716
timing_s/chunk_state_probe=7.164
timing_s/chunk_state_score=9.771
timing_s/update_actor=7.390
actor/powerflow_loss=0.293
actor/grad_norm=16.443
```

结论：

- smoke 成功，无 NaN/Ray/FSDP/vLLM 崩溃。
- anchor 机制确实生效：`replaced_ratio=1.0`，`score_floor=1.0`，`label_consistent_forced=1.0`。
- 相比上一轮 majority-consistent 3-step smoke，source 质量略高：`source_original_correct` 从 `0.792` 到 `0.812`。
- target 更锐：`target_entropy` 从约 `1.59` 降到 `1.38`，`positive_ratio` 从约 `0.486` 提到 `0.535`。
- 这只是 smoke，不代表最终 acc 会提升；但它满足“改 label estimation / verifier，而不是调权重”的下一轮条件。
- 下一步可以跑 20-step gate。gate 仍然是超过 hard confidence-gated c128 的 `mean@16=0.5325`；如果 20-step 仍失败，就说明 naive future chunk anchor 不足，需要更强的 answer-level verifier / longer horizon probe，而不是直接扩到 80-step。

## 2026-08-01 Teacher-anchor Mid-state c128 20-step Gate

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_teacher_anchor_mid_c128_probe4_b32_r32_v64_20step_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
chunk_state_source_mode=majority_consistent
chunk_state_boundary_mode=mid
chunk_state_candidates=8
chunk_state_chunk_size=128
chunk_state_probe_samples=4
chunk_state_probe_max_tokens=1024
chunk_state_teacher_anchor_enable=True
chunk_state_teacher_anchor_candidate_index=0
chunk_state_teacher_anchor_score=1.0
chunk_state_source_chunk_enable=False
actor.powerflow_enable=True
actor.use_dynamic_bsz=False
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_teacher_anchor_mid_c128_probe4_b32_r32_v64_20step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_teacher_anchor_mid_c128_probe4_b32_r32_v64_20step_20260801.jsonl
diag_jsonl_rows=640
```

Final validation：

```text
mean@16=0.525875
maj@16=0.656294
best@16=0.867390
format_mean@16=0.899000
format_maj@16=0.883678
format_best@16=0.999000
testing=295.607s
```

20-step 平均训练信号：

```text
teacher_anchor_replaced_ratio=1.000
teacher_anchor_label_consistent_forced=1.000
source_original_correct=0.780
source_majority_consistent=1.000
positive_ratio_after_anchor=0.556
label_consistent_ratio_after_anchor=0.731
kept_state_ratio=0.678
target_entropy=1.450
timing_s/gen=25.185
timing_s/chunk_state_probe=6.670
timing_s/chunk_state_score=10.945
timing_s/update_actor=7.356
```

与短程 gate 对比：

```text
hard confidence-gated c128:
  mean@16=0.532500 maj@16=0.666974 best@16=0.877136 format_mean@16=0.913625

majority-consistent mid-state c128:
  mean@16=0.486625 maj@16=0.619360 best@16=0.863332 format_mean@16=0.898000

teacher-anchor mid-state c128:
  mean@16=0.525875 maj@16=0.656294 best@16=0.867390 format_mean@16=0.899000
```

结论：

- 这轮 teacher-anchor 20-step gate 没有超过 hard confidence-gated c128 的 `mean@16=0.5325`，因此不能扩到 80-step。
- 相比 majority-consistent mid-state，teacher-anchor 明显修复了局部训练信号：`mean@16` 从 `0.486625` 回到 `0.525875`，但仍略低于 hard gate。
- anchor 机制本身按预期生效：`replaced_ratio=1.0`，`label_consistent_forced=1.0`，`kept_state_ratio=0.678`，actor update 只训练约 128-token chunk，平均 `~7.36s`。
- 失败说明 naive future chunk anchor 只能减少局部 target 噪声，不能保证 chunk-level improved distribution 指向最终正确性。
- 下一步不应继续调 PowerFlow 权重或直接扩 80-step；应参考 arXiv 2504.16084 的 state-level 多输出 label estimation / reward calculation / online self-improvement 思路，把 `query + partial reasoning` state 下的 candidate scoring 改成更强的 answer-level verifier 或 longer-horizon probe 聚合。
- infra 侧当前端到端主要由 full rollout `~25.2s`、chunk probe `~6.7s`、math parser/scoring `~10.9s`、actor update `~7.4s` 构成；SymPy warning/timeout 仍是 chunk search 实验的主要工程噪声源。

## 2026-08-01 Answer-consensus Mid-state c128 3-step Smoke

动机：

- 用户补充参考 arXiv 2504.16084。该文的关键处理是：给定同一 state，采多个输出，先做 label estimation，再用 rule/verifier 对每个输出计算 reward；即便 label accuracy 不高，逐样本 reward 仍可能因为 scattered wrong answers / lucky hit 而保持可用。
- 当前 majority-completion chunk scoring 的问题是只用 `chunk + probe` 的局部 completion majority，teacher-anchor 只能把 source future chunk 放进候选，但仍不能保证 improved distribution 指向最终正确性。
- 本轮新增 `chunk_state_score_mode=answer_consensus`：复用 full rollout 的 prompt-level majority label，给每个 `state + next_chunk + probe` completion 做 answer-level rule reward。训练 loss 仍然是 PowerFlow，不切 GRPO。

代码变更：

```text
verl/trainer/ppo/ray_trainer.py:
  _compute_full_rollout_majority_consistency 额外返回 prompt-level majority labels
  _make_chunk_state_prompts 将对应 prompt consensus label 写入 chunk_state_prompt_majority_label
  新增 _score_chunk_state_answer_consensus
  在 chunk_state_score_mode 分支接入 answer_consensus

run_records/ttrl_chunk_state_powerflow_answer_consensus_mid_c128_probe4_b32_r32_v64_3step_20260801.sh:
  复用 majority-consistent mid-state c128 配置
  ttrl.chunk_state_score_mode=answer_consensus
  关闭 teacher_anchor 和 source_chunk 注入，先只验证 scoring 语义
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_consensus_mid_c128_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
chunk_state_source_mode=majority_consistent
chunk_state_boundary_mode=mid
chunk_state_score_mode=answer_consensus
chunk_state_candidates=8
chunk_state_chunk_size=128
chunk_state_probe_samples=4
chunk_state_min_majority_ratio=0.25
chunk_state_min_answer_coverage=0.60
chunk_state_label_consistent_only=True
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_answer_consensus_mid_c128_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_answer_consensus_mid_c128_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
```

三步平均：

```text
answer_consensus_majority_ratio=0.331
answer_consensus_answer_coverage=0.755
answer_consensus_raw_positive_ratio=0.396
answer_consensus_label_consistent_ratio=0.586
answer_consensus_state_positive_ratio=0.750
kept_state_ratio=0.375
target_entropy=1.756
actor/powerflow_loss=0.109
actor/grad_norm=4.252
timing_s/gen=32.730
timing_s/chunk_state_score=8.935
timing_s/update_actor=7.801
```

结论：

- smoke 成功，无 NaN/Ray/FSDP/vLLM 崩溃，`chunk_state_score/mode_answer_consensus=1.0`。
- 新 scoring 确实产生 answer-level reward：`raw_positive_ratio=0.396`，`state_positive_ratio=0.750`。
- 但沿用旧的 `min_majority_ratio=0.25` 后只保留 `37.5%` state，训练信号偏稀；这和 arXiv 2504.16084 的分析相冲突，因为它强调 majority label 低精度时逐样本 reward 仍可能有用，不应过度用 majority ratio gate 丢掉样本。
- 下一步跑 20-step gate 前应放宽 `chunk_state_min_majority_ratio`，保留 answer-consensus 的 dense reward；同时继续保持 `min_answer_coverage=0.60`，避免无可解析答案的状态进入训练。

## 2026-08-01 Answer-consensus Mid-state c128 Relaxed 20-step Gate

动机：

- 基于 3-step smoke 的诊断，旧的 `chunk_state_min_majority_ratio=0.25` 只保留约 `37.5%` state，和 2504.16084 中“state label estimation 不必过强过滤，逐样本 reward 仍可提供信号”的叙事不一致。
- 本轮保持 PowerFlow loss、full rollout majority-consistent source、answer-consensus scoring 不变，只把 `chunk_state_min_majority_ratio` 放宽到 `0.0`，继续保留 `chunk_state_min_answer_coverage=0.60`。
- 目标是验证更 dense 的 answer-level reward 能否超过 hard confidence-gated c128 的 20-step gate：`mean@16=0.532500`。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_consensus_mid_c128_probe4_b32_r32_v64_minmaj0_20step_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_boundary_mode=mid
ttrl.chunk_state_score_mode=answer_consensus
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
ttrl.chunk_state_min_majority_ratio=0.0
ttrl.chunk_state_min_answer_coverage=0.60
ttrl.chunk_state_label_consistent_only=True
actor_rollout_ref.actor.powerflow_enable=True
actor_rollout_ref.actor.use_dynamic_bsz=False
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_answer_consensus_mid_c128_probe4_b32_r32_v64_minmaj0_20step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_answer_consensus_mid_c128_probe4_b32_r32_v64_minmaj0_20step_20260801.jsonl
diag_jsonl_rows=640
```

Final validation：

```text
mean@16=0.521750
maj@16=0.652016
best@16=0.865020
format_mean@16=0.907875
format_maj@16=0.893966
format_best@16=0.999918
testing=291.520s
```

20-step 平均训练信号：

```text
answer_consensus_majority_ratio=0.371
answer_consensus_answer_coverage=0.793
answer_consensus_raw_positive_ratio=0.445
answer_consensus_label_consistent_ratio=0.618
answer_consensus_state_positive_ratio=0.780
kept_state_ratio=0.719
target_entropy=1.753
actor/powerflow_loss=0.866
actor/grad_norm=36.073
timing_s/gen=25.684
timing_s/chunk_state_score=9.669
timing_s/update_actor=7.354
```

与短程 gate 对比：

```text
hard confidence-gated c128:
  mean@16=0.532500 maj@16=0.666974 best@16=0.877136 format_mean@16=0.913625

teacher-anchor mid-state c128:
  mean@16=0.525875 maj@16=0.656294 best@16=0.867390 format_mean@16=0.899000

answer-consensus minmaj0 mid-state c128:
  mean@16=0.521750 maj@16=0.652016 best@16=0.865020 format_mean@16=0.907875
```

结论：

- 这轮 relaxed answer-consensus 20-step gate 没有超过 hard confidence-gated c128 的 `mean@16=0.5325`，因此不扩到 80-step。
- 放宽 majority gate 的工程效果明确：`kept_state_ratio` 从 smoke 的 `0.375` 提高到 `0.719`，每步 32 个 state 基本都有可训练信号；但 final accuracy 没有提升，说明问题不是训练信号密度不足，而是当前 chunk target 的方向仍不够可靠。
- actor update 不是主要瓶颈，平均 `update_actor=7.354s`；端到端训练步主要成本仍是 full rollout 生成 `gen=25.684s` 和 answer-consensus probe/scoring `chunk_state_score=9.669s`。
- 当前 answer-consensus 的 reward 只判断 `state + next_chunk + probe` 是否落到 full-rollout prompt-level consensus answer，仍可能奖励“局部看起来能走到多数答案”的 chunk，而不是奖励真正改善后续搜索分布的 chunk。
- 下一版不应继续只调 gate 或延长训练，应改 scoring 语义：考虑 state 内多 probe 的 answer distribution sharpening、chunk 后续 value margin、或把 source full rollout 的 mid-state 与候选 chunk 的 long-horizon success 做更直接的 distribution matching。

## 2026-08-01 Answer-distribution Mid-state c128 3-step Smoke

动机：

- relaxed answer-consensus 的失败说明：只把 `state + next_chunk + probe` 匹配到 full-rollout prompt-level consensus answer 不够，容易奖励“能走到多数答案”的局部 continuation，而不一定是真正改善当前 state 搜索分布的 chunk。
- 2504.16084 的关键语义是同一 state 下多输出先做 label estimation，再做 reward calculation；因此本轮新增 `chunk_state_score_mode=answer_distribution`，完全在同一 chunk state 内用所有 `candidate x probe` 的 answer 分布估计局部 label。
- 对每个 state，提取 8 个候选 chunk、每个 4 条 probe completion 的最终答案，得到 32 个 answer sample；每个 probe 的软分数是该 answer 在当前 state 中的频率 `count(answer) / valid_answer_count`，candidate score 取 4 条 probe 的均值。
- 该 target 比 `majority_completion` 更 soft，比 `answer_consensus` 更少依赖 full prompt consensus；actor update 仍然是 PowerFlow distribution matching，不切 GRPO、不切 weighted NLL。

代码变更：

```text
verl/trainer/ppo/ray_trainer.py:
  新增 _score_chunk_state_answer_distribution
  新增 ttrl.chunk_state_score_mode=answer_distribution 分支
  记录 top_mass/top2_margin/unique_answer/answer_coverage/raw_score 等统计

run_records/ttrl_chunk_state_powerflow_answer_distribution_mid_c128_probe4_b32_r32_v64_3step_20260801.sh:
  复用 majority-consistent mid-state c128 配置
  ttrl.chunk_state_score_mode=answer_distribution
  关闭 teacher_anchor 和 source_chunk 注入
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_distribution_mid_c128_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
chunk_state_source_mode=majority_consistent
chunk_state_boundary_mode=mid
chunk_state_score_mode=answer_distribution
chunk_state_candidates=8
chunk_state_chunk_size=128
chunk_state_probe_samples=4
chunk_state_min_majority_ratio=0.20
chunk_state_min_answer_coverage=0.60
chunk_state_label_consistent_only=True
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_answer_distribution_mid_c128_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_answer_distribution_mid_c128_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
```

三步平均：

```text
answer_distribution_top_mass=0.601
answer_distribution_top2_margin=0.501
answer_distribution_unique_answer=8.042
answer_distribution_answer_coverage=0.749
answer_distribution_raw_score=0.369
answer_distribution_label_consistent_ratio=0.865
answer_distribution_state_positive_ratio=0.979
kept_state_ratio=0.677
target_entropy=1.789
powerflow_weight_max=3.143
actor/powerflow_loss=0.219
actor/grad_norm=8.993
timing_s/gen=32.543
timing_s/chunk_state_score=11.384
timing_s/update_actor=8.224
```

结论：

- smoke 成功，无 NaN/Ray/FSDP/vLLM 崩溃，`chunk_state_score/mode_answer_distribution=1.0`。
- 新 target 的训练信号更 dense：`state_positive_ratio=0.979`、`label_consistent_ratio=0.865`、`kept_state_ratio=0.677`；同时不是完全 uniform，`powerflow_weight_max=3.143`、`target_entropy=1.789`。
- `top_mass_mean=0.601`、`top2_margin=0.501` 说明同一 state 内 answer 分布有明显头部答案；但 `unique_answer_mean=8.042` 也说明答案空间仍然分散，20-step gate 必须验证这种 state-local frequency target 是否真的改善最终 acc。
- 该 smoke 只验证链路和 target 统计，不含 final validation。下一步可以跑同配置 20-step gate；通过标准仍是超过 hard confidence-gated c128 的 `mean@16=0.5325`，否则继续增强 long-horizon verifier / value margin，而不是扩 80-step。

## 2026-08-01 Answer-distribution Mid-state c128 20-step Gate

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_distribution_mid_c128_probe4_b32_r32_v64_20step_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
chunk_state_source_mode=majority_consistent
chunk_state_boundary_mode=mid
chunk_state_score_mode=answer_distribution
chunk_state_candidates=8
chunk_state_chunk_size=128
chunk_state_probe_samples=4
chunk_state_probe_max_tokens=1024
chunk_state_min_majority_ratio=0.20
chunk_state_min_answer_coverage=0.60
chunk_state_label_consistent_only=True
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_answer_distribution_mid_c128_probe4_b32_r32_v64_20step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_answer_distribution_mid_c128_probe4_b32_r32_v64_20step_20260801.jsonl
diag_jsonl_rows=640
```

Final validation：

```text
mean@16=0.399875
maj@16=0.511854
best@16=0.828208
format_mean@16=0.888250
format_maj@16=0.852188
format_best@16=0.999942
testing=309.036s
```

20-step 平均训练信号：

```text
answer_distribution_top_mass=0.538
answer_distribution_top2_margin=0.429
answer_distribution_unique_answer=8.580
answer_distribution_answer_coverage=0.737
answer_distribution_raw_score=0.343
answer_distribution_label_consistent_ratio=0.865
answer_distribution_state_positive_ratio=0.966
kept_state_ratio=0.634
target_entropy=1.823
powerflow_weight_max=3.397
actor/powerflow_loss=0.262
actor/grad_norm=5.521
timing_s/gen=24.961
timing_s/chunk_state_score=10.096
timing_s/update_actor=7.392
```

与短程 gate 对比：

```text
hard confidence-gated c128:
  mean@16=0.532500 maj@16=0.666974 best@16=0.877136 format_mean@16=0.913625

teacher-anchor mid-state c128:
  mean@16=0.525875 maj@16=0.656294 best@16=0.867390 format_mean@16=0.899000

answer-consensus minmaj0 mid-state c128:
  mean@16=0.521750 maj@16=0.652016 best@16=0.865020 format_mean@16=0.907875

answer-distribution mid-state c128:
  mean@16=0.399875 maj@16=0.511854 best@16=0.828208 format_mean@16=0.888250
```

结论：

- 这轮 answer-distribution 20-step gate 明显失败，低于 hard confidence-gated c128 的 `mean@16=0.5325`，因此不扩到 80-step。
- 失败不是链路崩溃：20 step 全程无 NaN/Ray/FSDP/vLLM 崩溃，`diag_jsonl_rows=640`，PowerFlow loss 正常更新，actor update 平均 `7.392s`。
- 失败更像 target 语义错误：用同一 state 内 `candidate x probe` 的 answer frequency 直接做 soft score，会奖励“在短 probe 中落到高频答案”的 chunk，但这个高频答案不一定是更好的完整解题方向；`best@16=0.828208` 说明搜索空间仍有正确答案，`mean@16/maj@16` 大幅下降说明 policy 被推向了错误或格式化但不可靠的局部分布。
- 2504.16084 的启发应该继续保留，但要更严格实现：它支持同一 state 多输出 label estimation / reward calculation，而不是把 state-local frequency 本身当最终 chunk reward。下一版需要让 chunk label estimation 和后续 search improvement 绑定，例如用候选 chunk 后的 long-horizon pass/value margin、state value gain、或 search-improved distribution 的 KL/PowerFlow target，而不是只用答案频率。
- infra 观察：本 run 的额外成本主要是 `chunk_state_score=10.096s`，其中包含 probe generation 和答案解析；actor update 仍不是瓶颈。日志里仍有大量 SymPy warning/subprocess shutdown，说明 parser/scoring 长尾还需要优化，但这次指标失败首先是算法目标问题。

## 2026-08-01 Answer-value-gain Mid-state c128 3-step Smoke

动机：

- answer-distribution 20-step gate 失败后，不继续调频率 target。新的目标是把 chunk reward 从“state 内答案频率”改成“该 chunk 是否让后续 probe 相对 full-rollout baseline 更容易回到 prompt-level pseudo label”。
- 该设计继续遵循 2504.16084 的无 GT label estimation 思路：先用同一 prompt 的 32 条完整 rollout 做 prompt-level majority label 和 baseline ratio，再在 `state + chunk` 下做 probe，计算候选 chunk 的 value gain。
- 对每个 state，baseline 是该 prompt 原始 32 条 rollout 的 majority ratio；候选 chunk 的 hit rate 是 4 条 probe 命中 prompt pseudo label 的比例；score 为 `max(0, (hit_rate - baseline) / (1 - baseline))`。因此只有相对 baseline 改善的 chunk 才拿正分。
- actor update 仍然是 PowerFlow distribution matching，不切 GRPO、不切 weighted NLL；真实 GT 只用于 diag，不参与 chunk source/target。

代码变更：

```text
verl/trainer/ppo/ray_trainer.py:
  新增 _score_chunk_state_answer_value_gain
  新增 ttrl.chunk_state_score_mode=answer_value_gain 分支
  answer_value_gain 自动触发 full-rollout majority label/ratio 计算
  记录 baseline_ratio/hit_rate/gain/max_gain/improved_state 等统计

run_records/ttrl_chunk_state_powerflow_answer_value_gain_mid_c128_probe4_b32_r32_v64_3step_20260801.sh:
  继承 majority-consistent mid-state c128 配置
  ttrl.chunk_state_score_mode=answer_value_gain
  关闭 teacher_anchor 和 source_chunk 注入
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_value_gain_mid_c128_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
chunk_state_source_mode=majority_consistent
chunk_state_boundary_mode=mid
chunk_state_score_mode=answer_value_gain
chunk_state_candidates=8
chunk_state_chunk_size=128
chunk_state_probe_samples=4
chunk_state_min_majority_ratio=0.20
chunk_state_min_answer_coverage=0.60
chunk_state_label_consistent_only=True
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_answer_value_gain_mid_c128_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_answer_value_gain_mid_c128_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
```

三步平均：

```text
answer_value_gain_baseline_ratio=0.338
answer_value_gain_answer_coverage=0.737
answer_value_gain_hit_rate=0.446
answer_value_gain_gain=0.375
answer_value_gain_max_gain=0.615
answer_value_gain_improved_state_ratio=0.740
answer_value_gain_label_consistent_ratio=0.531
kept_state_ratio=0.510
target_entropy=1.687
powerflow_weight_max=6.838
actor/powerflow_loss=0.098
actor/grad_norm=7.097
timing_s/gen=32.062
timing_s/chunk_state_score=9.015
timing_s/update_actor=7.303
```

结论：

- smoke 成功，无 NaN/Ray/FSDP/vLLM 崩溃，`chunk_state_score/mode_answer_value_gain=1.0`。
- 训练信号不是空的：`gain_mean=0.375`、`improved_state_ratio=0.740`、`kept_state_ratio=0.510`；相对 answer-distribution 的 `kept_state_ratio=0.634` 更稀疏，但语义上更接近 chunk-level search-state improvement。
- `target_entropy=1.687`、`powerflow_weight_max=6.838` 说明目标分布比 answer-distribution 更尖，可能带来更强更新，也可能不稳定；20-step gate 必须观察 `mean@16/maj@16` 是否比 hard confidence-gated c128 更好。
- infra 仍然正常：8 卡 B200、vLLM `FLASH_ATTN`、FlashInfer autotune、CUDA graph capture、NCCL P2P/CUMEM/NVLS、Actor fused kernels/Triton backend 均在日志中出现；actor dynamic batch 关闭。
- 下一步可以跑同配置 20-step gate；通过标准仍是超过 hard confidence-gated c128 的 `mean@16=0.5325`，否则继续把 value gain 从 majority-label hit rate 扩展为 long-horizon pass/value margin。

## 2026-08-01 Answer-value-gain Mid-state c128 20-step Gate

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_value_gain_mid_c128_probe4_b32_r32_v64_20step_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
chunk_state_source_mode=majority_consistent
chunk_state_boundary_mode=mid
chunk_state_score_mode=answer_value_gain
chunk_state_candidates=8
chunk_state_chunk_size=128
chunk_state_probe_samples=4
chunk_state_probe_max_tokens=1024
actor.powerflow_enable=True
actor.powerflow_use_chunk_weights=True
actor.chunk_weighted_nll_enable=False
actor.use_dynamic_bsz=False
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_answer_value_gain_mid_c128_probe4_b32_r32_v64_20step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_answer_value_gain_mid_c128_probe4_b32_r32_v64_20step_20260801.jsonl
diag_jsonl_rows=640
```

final validation：

```text
mean@16=0.362500
maj@16=0.463248
best@16=0.784332
format_mean@16=0.873625
format_maj@16=0.827412
format_best@16=0.999916
testing=305.987s
```

20-step 平均训练信号：

```text
answer_value_gain_baseline_ratio=0.290
answer_value_gain_answer_coverage=0.744
answer_value_gain_hit_rate=0.384
answer_value_gain_gain=0.314
answer_value_gain_max_gain=0.576
answer_value_gain_improved_state_ratio=0.720
answer_value_gain_label_consistent_ratio=0.482
kept_state_ratio=0.398
target_entropy=1.626
powerflow_weight_max=6.270
actor/powerflow_loss=0.305
actor/grad_norm=12.182
timing_s/gen=26.440
timing_s/chunk_state_score=11.418
timing_s/update_actor=7.322
```

与短程 gate 对比：

```text
hard confidence-gated c128:
  mean@16=0.532500 maj@16=0.666974 best@16=0.877136 format_mean@16=0.913625

answer-distribution mid-state c128:
  mean@16=0.399875 maj@16=0.511854 best@16=0.828208 format_mean@16=0.888250

answer-value-gain mid-state c128:
  mean@16=0.362500 maj@16=0.463248 best@16=0.784332 format_mean@16=0.873625
```

结论：

- 这轮 answer-value-gain 20-step gate 失败，低于 hard confidence-gated c128，也低于 answer-distribution gate，因此不能扩到 80-step。
- 失败不是 infra 崩溃：20 step 完成，`diag_jsonl_rows=640`，无 NaN/Ray/FSDP/vLLM 崩溃，PowerFlow loss、grad norm、chunk weights 都有正常更新。
- 问题仍是 chunk target 语义：`improved_state_ratio=0.720` 和 `kept_state_ratio=0.398` 表明 proxy 能选出“相对 majority baseline 命中率提升”的 chunk，但 final `mean@16=0.3625` 和日志中的重复 `\boxed{}` 输出说明该 proxy 会鼓励局部 boxed/答案吸引子，不保证完整推理路径质量。
- 这个结果进一步说明不能只用短 probe 命中 prompt-level pseudo label 做 chunk reward。下一版应转为 long-horizon pass/value margin：候选 chunk 后继续更长 rollout 或小规模 search，score 直接衡量相对原 full rollout 的 pass/maj/best 改善，并加入 anti-degeneration/format guard；或者更贴近 2504.16084，把同一 state 的多候选先做可靠 label/value estimation，再把 search-improved distribution 蒸馏回 PowerFlow target。
- infra 观察：actor update 平均 `7.322s`，仍不是主要瓶颈；额外成本主要在 `gen=26.440s` 和 `chunk_state_score=11.418s`，其中 scoring 受 SymPy/parser timeout 长尾影响明显。后续若继续 chunk path，应优先降低 probe/search 评估成本，而不是扩大当前错误 target。

## 2026-08-01 2504.16084 Chunk-state 设计约束

论文要点：

- arXiv 2504.16084 本身是 TTRL，不是 chunk-level 方法。它的核心是：对同一个 state/prompt 采多个输出，先做 label estimation，再用估计 label 计算 rule-based reward，最后在线 RL 更新。
- 原文实现细节：Qwen2.5-Math 和 LRM rollout temperature 使用 `1.0`；先采 64 条用于 voting-based label estimation，再下采样 32 条用于训练；MATH-500 用 10 episodes；max generation length 为 3072。
- 对我们有用的是“label estimation”和“reward calculation”必须分开。不能把 state-local answer frequency 或短 probe hit-rate 本身当作最终 chunk target，否则容易把模型推向局部答案吸引子。
- 原文解释 TTRL 能工作的关键是 reward robustness：多输出比较给了更稠密的正/负 reward，即使 majority label 不准，错误答案分散时负样本仍然可靠。chunk 版本也要保留这个性质，即候选 chunk reward 应来自候选后续 search/value 相对 baseline 的改善，同时保留对坏 chunk 的过滤/负信息，而不是只拉高短程高频答案。

对下一版 chunk 方案的约束：

- 继续使用 PowerFlow loss 做 actor update，但 target 必须是 search-improved distribution，不再使用 answer-distribution frequency。
- chunk scoring 至少要包含：prompt-level pseudo label、candidate 后续 probe/search、相对 full-rollout baseline 的 margin、format/degeneration guard。
- 如果短 probe 太不可靠，应该改成更长 horizon 的 pass/value margin，或者按每个 state 的 top-k margin 做 normalized target，避免大量 state 得到 uniform/zero target。

## 2026-08-01 Answer-value-margin Mid-state c128 3-step Smoke

动机：

- answer-value-gain 20-step gate 失败后，先做一个更保守的非负 PowerFlow target：候选 chunk 只有在 `hit_rate - baseline >= margin` 时才给正分。
- 加入 format guard：probe 拼接后的回答必须能抽到非空 answer，答案字符串不超过 128 字符，`\boxed` 次数不超过 8；否则该 probe 视作无效。目标是压住上一轮 final validation 中出现的重复 `\boxed{}` 退化。
- 这个版本仍然不是完整 long-horizon search-state TTRL，只是验证 “margin + anti-degeneration guard” 是否能稳定跑通，并观察 target 稀疏度和 PowerFlow loss 是否正常。

代码变更：

```text
verl/trainer/ppo/ray_trainer.py:
  新增 _chunk_state_probe_is_well_formed
  新增 _score_chunk_state_answer_value_margin
  新增 ttrl.chunk_state_score_mode=answer_value_margin 分支

run_records/ttrl_chunk_state_powerflow_answer_value_margin_mid_c128_probe4_b32_r32_v64_3step_20260801.sh:
  继承 majority-consistent mid-state c128 配置
  ttrl.chunk_state_score_mode=answer_value_margin
  +ttrl.chunk_state_value_margin=0.25
  +ttrl.chunk_state_format_guard=True
  +ttrl.chunk_state_max_boxed_count=8
  +ttrl.chunk_state_max_answer_chars=128
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_value_margin_mid_c128_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_answer_value_margin_mid_c128_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_answer_value_margin_mid_c128_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
```

三步平均：

```text
answer_value_margin_baseline_ratio=0.336
answer_value_margin_answer_coverage=0.714
answer_value_margin_hit_rate=0.382
answer_value_margin_format_rate=0.714
answer_value_margin_margin=0.205
answer_value_margin_score=0.178
answer_value_margin_max_margin=0.356
answer_value_margin_improved_state_ratio=0.542
answer_value_margin_label_consistent_ratio=0.336
kept_state_ratio=0.385
target_entropy=1.739
powerflow_weight_max=7.035
actor/powerflow_loss=0.070
actor/grad_norm=9.785
timing_s/gen=35.786
timing_s/chunk_state_score=8.930
timing_s/update_actor=7.553
```

结论：

- smoke 成功，无 NaN/Ray/FSDP/vLLM 崩溃，`chunk_state_score/mode_answer_value_margin=1.0`。
- format guard 生效后信号更稀疏：`format_rate=0.714`、`score_mean=0.178`、`kept_state_ratio=0.385`，但不是全零。
- 当前不能直接扩 20-step：前两步 `actor/powerflow_loss=0.0`，第三步才到 `0.210`。这说明保守 margin target 虽然能跑，但和 PowerFlow 的 chunk weights/boxed_reward 交互不够稳定，可能导致有效更新不足。
- 下一版应优先修 target-to-loss 的有效性：可以把 margin target 改成 state 内 top-k normalized margin，降低 hard margin 到 0.125，或给每个 kept state 至少一个 top candidate 正权重；同时继续保留 format/degeneration guard。只有确认每步 PowerFlow loss 都非零且 format 不退化后，才跑 20-step validation gate。

## 2026-08-01 Answer-value-margin Top2 Mid-state c128 3-step Smoke

动机：

- 上一版 hard margin `0.25` 太稀疏，且前两步 `actor/powerflow_loss=0.0`。这轮只改 target-to-loss 有效性，不改变 chunk state 主语义：仍是 majority-consistent source、mid-state、c128、probe4、PowerFlow loss。
- 按 arXiv 2504.16084 给我们的约束，先在同一 state 下做多输出 label/value estimation，再把 improved distribution 变成训练 target。这里的近似实现是：候选 chunk 的 probe hit-rate 相对 full-rollout majority baseline 至少高 `0.125`，再在每个 state 内只保留 top2 正 margin candidate。
- 继续保留 format/degeneration guard：probe 必须能抽取非空 answer，答案字符串不超过 128，`\boxed` 次数不超过 8。

代码变更：

```text
verl/trainer/ppo/ray_trainer.py:
  answer_value_margin scoring 新增 ttrl.chunk_state_value_topk
  先构造 valid_matrix = margin >= min_margin，再按 state 保留 top-k valid positive margin
  输出 chunk_state_answer_value_margin/topk 指标

run_records/ttrl_chunk_state_powerflow_answer_value_margin_mid_c128_probe4_b32_r32_v64_3step_20260801.sh:
  将 margin/topk/format guard 参数改为 env 控制，默认保持旧 hard-margin 语义

run_records/ttrl_chunk_state_powerflow_answer_value_margin_top2_mid_c128_probe4_b32_r32_v64_3step_20260801.sh:
  CHUNK_STATE_VALUE_MARGIN=0.125
  CHUNK_STATE_VALUE_TOPK=2
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_value_margin_top2_mid_c128_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_answer_value_margin_top2_mid_c128_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_answer_value_margin_top2_mid_c128_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
```

三步平均：

```text
answer_value_margin_baseline_ratio=0.344
answer_value_margin_answer_coverage=0.759
answer_value_margin_hit_rate=0.418
answer_value_margin_format_rate=0.759
answer_value_margin_margin=0.223
answer_value_margin_score=0.082
answer_value_margin_max_margin=0.377
answer_value_margin_improved_state_ratio=0.635
answer_value_margin_label_consistent_ratio=0.146
answer_value_margin_min_margin=0.125
answer_value_margin_topk=2.000
kept_state_ratio=0.448
positive_ratio=0.082
target_entropy=1.255
powerflow_weight_mean=0.448
powerflow_weight_max=7.665
actor/powerflow_loss=0.153
actor/boxed_reward_mean=0.074
actor/powerflow_weight_mean=0.250
actor/powerflow_weight_max=3.801
actor/grad_norm=7.137
timing_s/gen=32.396
timing_s/chunk_state_chunks=1.041
timing_s/chunk_state_probe=6.720
timing_s/chunk_state_score=8.960
timing_s/chunk_state_ref=3.594
timing_s/update_actor=7.444
```

逐步 PowerFlow loss：

```text
actor/powerflow_loss: [0.000, 0.174, 0.285]
actor/powerflow_weight_mean: [0.000, 0.250, 0.500]
chunk_state/powerflow_weight_mean: [0.469, 0.469, 0.406]
chunk_state/kept_state_ratio: [0.469, 0.469, 0.406]
```

结论：

- top2 margin target 比 hard margin 更好：每步 chunk target 都有非零 state，`kept_state_ratio` 稳定在 0.4 以上，step2/step3 已经能产生非零 PowerFlow loss。
- 但仍不能扩 20-step：step1 的 `chunk_state/powerflow_weight_mean=0.469`，actor 侧却是 `actor/powerflow_weight_mean=0.000`、`actor/powerflow_loss=0.000`。这说明 target 已经产出，但进入 actor PowerFlow loss 后仍可能被二次权重/boxed_reward/CISPO 链路清空。
- 下一步不再调大 rollout 参数，也不先跑 20-step。应优先修 actor loss 语义：chunk-state PowerFlow 应直接蒸馏 search-improved continuation distribution，`powerflow_chunk_weights` 作为主权重；`boxed_reward` 只作为 target/value term 或可关闭的 ablation，避免把 margin score 同时作为权重和 reward 后再被 PowerFlow 内部公式压掉。
- 这轮保留为有效 smoke 证据：infra 没崩，B200/vLLM/FSDP 链路正常，问题集中在 chunk target 到 PowerFlow loss 的映射。

## 2026-08-01 Answer-value-margin Top2 Target-only Mid-state c128 3-step Smoke

动机：

- 上一轮 top2 margin smoke 中，trainer 侧 `chunk_state/powerflow_weight_mean` 每步都有非零信号，但 actor 侧 step1 `actor/powerflow_weight_mean=0.0`、`actor/powerflow_loss=0.0`。
- 为了隔离是不是 `boxed_reward` 进入 PowerFlow residual 后把局部 margin target 压掉，这轮新增 `actor_rollout_ref.actor.powerflow_chunk_loss_mode=target_only`。该模式保持 PowerFlow 的 `log_z + avg_log_prob - beta * avg_ref_log_prob` residual 和 chunk weights，但不把局部 margin score 作为 boxed reward 偏移项。
- 该分支默认不启用，`standard` 模式保持原 PowerFlow 语义。

代码变更：

```text
verl/workers/actor/dp_actor.py:
  compute_powerflow 新增 chunk_loss_mode
  chunk_loss_mode=target_only 时不使用 boxed_reward residual term
  输出 actor/powerflow_chunk_loss_target_only 指标

verl/trainer/config/ppo_trainer.yaml:
  新增 actor.powerflow_chunk_loss_mode，默认 standard

run_records/ttrl_chunk_state_powerflow_answer_value_margin_top2_targetonly_mid_c128_probe4_b32_r32_v64_3step_20260801.sh:
  继承 top2 margin smoke
  actor_rollout_ref.actor.powerflow_use_boxed_reward=False
  actor_rollout_ref.actor.powerflow_chunk_loss_mode=target_only
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_value_margin_top2_targetonly_mid_c128_probe4_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
raw_log=important_experiment_logs/ttrl_chunk_state_powerflow_answer_value_margin_top2_targetonly_mid_c128_probe4_b32_r32_v64_3step_20260801.log
diag_jsonl=important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_answer_value_margin_top2_targetonly_mid_c128_probe4_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows=96
```

三步平均：

```text
answer_value_margin_baseline_ratio=0.351
answer_value_margin_answer_coverage=0.771
answer_value_margin_hit_rate=0.430
answer_value_margin_format_rate=0.771
answer_value_margin_margin=0.230
answer_value_margin_score=0.089
answer_value_margin_max_margin=0.409
answer_value_margin_improved_state_ratio=0.719
answer_value_margin_label_consistent_ratio=0.164
answer_value_margin_topk=2.000
kept_state_ratio=0.458
positive_ratio=0.089
target_entropy=1.152
powerflow_weight_mean=0.458
powerflow_weight_max=7.336
actor/powerflow_loss=0.199
actor/powerflow_chunk_loss_target_only=1.000
actor/boxed_reward_mean=0.079
actor/powerflow_weight_mean=0.167
actor/powerflow_weight_max=3.251
actor/grad_norm=26.486
actor/cispo_mask_ratio=0.194
timing_s/gen=32.778
timing_s/chunk_state_chunks=1.028
timing_s/chunk_state_probe=6.703
timing_s/chunk_state_score=8.800
timing_s/chunk_state_ref=3.585
timing_s/update_actor=7.368
```

逐步 PowerFlow loss：

```text
actor/powerflow_loss: [0.000, 0.236, 0.360]
actor/powerflow_weight_mean: [0.000, 0.250, 0.250]
chunk_state/powerflow_weight_mean: [0.469, 0.500, 0.406]
chunk_state/kept_state_ratio: [0.469, 0.500, 0.406]
actor/powerflow_chunk_loss_target_only: [1.000, 1.000, 1.000]
```

结论：

- `target_only` 分支成功执行，且 step2/step3 loss 稳定非零；但 step1 仍然是 `actor/powerflow_weight_mean=0.0`、`actor/powerflow_loss=0.0`。
- 因此第一步清零不是 boxed_reward residual 造成的。更可能的问题在 actor batch 的 `powerflow_chunk_weights` 传递、keep_indices 后的排序/分片、或 actor update 内部 microbatch/mini-batch 选择上；trainer 侧已经有非零 `chunk_state/powerflow_weight_mean`。
- 暂不跑 20-step。下一步应该增加 actor-batch 权重诊断，例如在 `_build_chunk_state_actor_batch` 输出 `actor_batch_powerflow_weight_mean/max/nonzero_ratio`，并在 actor `update_policy` 入口输出实际收到的 `powerflow_chunk_weights` 统计，以确认权重是在 trainer->actor 传递前还是 actor 内部分片后变零。

## 2026-08-01 Chunk PowerFlow Weight Shard Balance Fix

问题定位：

- `target_only` 3-step smoke 中 step1 的 `actor/powerflow_loss=0.0` 不是 chunk target 没信号。1-step 诊断显示 trainer 全局 `chunk_state/powerflow_weight_mean=0.469`、`chunk_state/actor_batch_powerflow_weight_nonzero_ratio=0.469`。
- 真正问题是 actor batch 按连续顺序切给 8 个 FSDP/DP shard 时，正权重样本集中在部分 shard。修复前 shard0 完全没有正权重，导致 rank0 上报的 `actor/powerflow_weight/mean=0.0`、`actor/powerflow_loss=0.0`。

修复前诊断：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_value_margin_top2_targetonly_diag1_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=1
diag_jsonl_rows=32
chunk_state/powerflow_weight_mean=0.469
chunk_state/actor_batch_powerflow_weight_nonzero_ratio=0.469
chunk_state/actor_batch_powerflow_weight_shard0_mean=0.000
chunk_state/actor_batch_powerflow_weight_shard0_nonzero_ratio=0.000
chunk_state/actor_batch_powerflow_weight_shard_mean_low=0.000
chunk_state/actor_batch_powerflow_weight_shard_mean_high=0.750
chunk_state/actor_batch_powerflow_weight_zero_shard_ratio=0.250
actor/powerflow_loss=0.000
actor/powerflow_weight/mean=0.000
actor/powerflow_weight/nonzero_ratio=0.000
actor/grad_norm=60.661
timing_s/update_actor=7.705
```

代码变更：

```text
verl/trainer/ppo/ray_trainer.py:
  在 _build_chunk_state_actor_batch 中按 powerflow_flat_weights 对 actor batch 做 shard-balanced reorder
  positive samples 按权重降序 round-robin 分配到 n_gpus_per_node * nnodes 个 bucket
  zero-weight samples 也 round-robin 填充
  只改变 actor batch 顺序，不改变 score/weight/loss/训练样本集合

verl/workers/actor/dp_actor.py:
  增加 actor/powerflow_weight/nonzero_ratio
  增加 actor/powerflow_local_batch_size
```

修复后诊断：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_value_margin_top2_targetonly_balance1_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=1
diag_jsonl_rows=32
chunk_state/powerflow_weight_mean=0.469
chunk_state/actor_batch_powerflow_weight_nonzero_ratio=0.469
chunk_state/actor_batch_powerflow_weight_shard0_mean=0.615
chunk_state/actor_batch_powerflow_weight_shard0_nonzero_ratio=0.469
chunk_state/actor_batch_powerflow_weight_shard_mean_low=0.379
chunk_state/actor_batch_powerflow_weight_shard_mean_high=0.615
chunk_state/actor_batch_powerflow_weight_zero_shard_ratio=0.000
actor/powerflow_loss=0.177
actor/powerflow_weight/mean=0.615
actor/powerflow_weight/nonzero_ratio=0.469
actor/grad_norm=12.662
timing_s/update_actor=6.166
```

结论：

- 这是一个实现/infra 修复，不改变 chunk scoring 语义，也不改变 PowerFlow loss 公式。它只保证 search-improved positive chunks 不会因为 batch ordering 被集中分配到少数 DP shard。
- 修复后所有 shard 都有正权重，`zero_shard_ratio` 从 `0.250` 降到 `0.000`，rank0 actor loss 从 `0.000` 变为 `0.177`。
- 下一步应跑同配置 3-step balance smoke，确认三步 `actor/powerflow_loss` 都非零；若通过，再进入 20-step validation gate。

## 2026-08-01 Target-Only Chunk PowerFlow 3-Step Balance Smoke

运行配置：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_value_margin_top2_targetonly_balance3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
ttrl.chunk_state_score_mode=answer_value_margin
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_boundary_mode=mid
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
ttrl.chunk_state_value_margin=0.125
ttrl.chunk_state_value_topk=2
ttrl.chunk_state_format_guard=True
ttrl.chunk_state_label_consistent_only=True
actor_rollout_ref.actor.powerflow_enable=True
actor_rollout_ref.actor.powerflow_use_boxed_reward=False
actor_rollout_ref.actor.powerflow_use_chunk_weights=True
actor_rollout_ref.actor.powerflow_chunk_loss_mode=target_only
actor_rollout_ref.actor.use_dynamic_bsz=False
```

关键结果：

```text
diag_jsonl_rows=96
chunk_state/powerflow_weight_mean: [0.469, 0.437, 0.438], avg=0.448
chunk_state/actor_batch_powerflow_weight_nonzero_ratio: [0.469, 0.438, 0.438], avg=0.448
chunk_state/actor_batch_powerflow_weight_shard0_mean: [0.615, 0.605, 0.567], avg=0.596
chunk_state/actor_batch_powerflow_weight_shard0_nonzero_ratio: [0.469, 0.438, 0.438], avg=0.448
chunk_state/actor_batch_powerflow_weight_shard_mean_low: [0.379, 0.366, 0.374], avg=0.373
chunk_state/actor_batch_powerflow_weight_shard_mean_high: [0.615, 0.605, 0.567], avg=0.596
chunk_state/actor_batch_powerflow_weight_zero_shard_ratio: [0.000, 0.000, 0.000], avg=0.000
actor/powerflow_loss: [0.668, 0.654, 0.251], avg=0.524
actor/powerflow_weight/mean: [0.615, 0.605, 0.567], avg=0.596
actor/powerflow_weight/nonzero_ratio: [0.469, 0.438, 0.438], avg=0.448
actor/grad_norm: [36.075, 23.063, 17.323], avg=25.487
timing_s/gen: [51.940, 23.641, 22.254], avg=32.612
timing_s/chunk_state_score: [9.115, 10.959, 8.638], avg=9.571
timing_s/chunk_state_ref: [5.704, 1.818, 1.838], avg=3.120
timing_s/update_actor: [6.100, 5.582, 5.620], avg=5.767
```

结论：

- 3-step gate 通过。三步 `actor/powerflow_loss` 全部非零，且 `actor_batch_powerflow_weight_zero_shard_ratio=0.000`，说明 shard-balanced reorder 已经解决正权重样本集中到少数 DP shard 的问题。
- 第一步 `timing_s/gen=51.940` 包含 vLLM/Triton 首次 shape JIT；step2/step3 稳态 rollout gen 约 `22-24s`，chunk scoring 约 `8.6-11.0s`，chunk ref 约 `1.8s`，actor update 约 `5.6s`。
- 当前 20-step gate 的主要风险已经不是 actor loss 清零，而是 `answer_value_margin` 的局部 target 是否真的带来 validation 提升。下一步进入同配置 20-step validation gate。

arXiv 2504.16084 对当前设计的约束：

- 该文是 TTRL 原文，核心是无 GT 场景下先做 label estimation，再把估计出的 label/reward 用于 RL 更新。
- 对 chunk-level search-state TTRL 来说，不能把短 probe hit-rate 直接当成终极监督；更稳的表述应该是：对同一 `query + prefix` state 的 multiple next-chunk candidates 先估计局部 improved continuation distribution，再用 PowerFlow target 做 actor update。
- 当前 `answer_value_margin + top2 + target_only PowerFlow` 路径已经按这个拆分执行：probe/search 负责 label/distribution estimation，PowerFlow target 负责 reward/target calculation 和参数更新。

## 2026-08-01 Target-Only Chunk PowerFlow 20-Step Gate

误启动记录：

- `RUN_ID=ttrl_chunk_state_powerflow_answer_value_margin_top2_targetonly_balance20_mid_c128_probe4_b32_r32_v64_20260801`
- 这次从 devbox master 普通 shell 启动，不在 GPU worker 内，Ray 看到 `Total available GPUs 0 is less than total desired GPUs 8` 后退出。
- 该 run 没有训练，不计入有效实验结果；日志仍保留用于排查启动位置问题。

有效运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_answer_value_margin_top2_targetonly_balance20b_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
ttrl.chunk_state_score_mode=answer_value_margin
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_boundary_mode=mid
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
ttrl.chunk_state_value_margin=0.125
ttrl.chunk_state_value_topk=2
ttrl.chunk_state_format_guard=True
ttrl.chunk_state_label_consistent_only=True
actor_rollout_ref.actor.powerflow_use_boxed_reward=False
actor_rollout_ref.actor.powerflow_chunk_loss_mode=target_only
actor_rollout_ref.actor.use_dynamic_bsz=False
```

训练侧统计：

```text
diag_jsonl_rows=640
chunk_state/kept_state_ratio avg=0.392 first=0.469 last=0.500 min=0.219 max=0.500
chunk_state/powerflow_weight_mean avg=0.392 first=0.469 last=0.500 min=0.219 max=0.500
chunk_state/actor_batch_powerflow_weight_zero_shard_ratio avg=0.000 min=0.000 max=0.000
chunk_state/positive_ratio avg=0.0817 min=0.049 max=0.119
chunk_state/target_entropy avg=1.307 min=1.047 max=1.514
actor/powerflow_loss avg=0.186 first=0.354 last=0.074 min=0.008 max=0.473
actor/powerflow_weight/nonzero_ratio avg=0.392 first=0.469 last=0.500 min=0.219 max=0.500
actor/log_z avg=-0.491 first=-1.090 last=-0.231 min=-1.090 max=-0.147
actor/grad_norm avg=9.904 first=22.397 last=6.448 min=1.572 max=25.017
timing_s/gen avg=25.476 first=51.246 last=21.296 min=21.296 max=51.246
timing_s/chunk_state_score avg=9.784 min=8.535 max=11.749
timing_s/chunk_state_ref avg=1.916 first=5.742 last=1.590 min=1.462 max=5.742
timing_s/update_actor avg=5.283 first=6.138 last=4.832 min=4.334 max=6.138
timing_s/testing=301.278
```

final validation：

```text
val-core/math/acc/mean@16 = 0.427
val-core/math/acc/maj@16/mean = 0.552
val-core/math/acc/best@16/mean = 0.836
val-aux/math/format_score/mean@16 = 0.888
val-aux/math/format_score/maj@16/mean = 0.856
```

结论：

- infra gate 通过：20 个训练 step 中 `actor_batch_powerflow_weight_zero_shard_ratio` 始终为 `0.000`，说明 shard-balanced actor batch reorder 在长一点的 gate 中也稳定。
- 性能分解可接受但不够快：稳态 `gen` 约 `21-25s`，`chunk_state_score` 约 `9-12s`，`chunk_state_ref` 约 `1.5-1.9s`，`update_actor` 约 `4.3-5.8s`；第 20 步 validation 约 `301s`。
- 训练语义失败：20-step final `mean@16=0.427`、`maj@16=0.552`，明显低于 MV/TTRL baseline。虽然 `best@16=0.836` 还保留一部分搜索上限，但 mean/majority 已经严重塌陷。
- 失败形态很明确：`actor/powerflow_loss` 从 step1 的 `0.354` 衰减到 step20 的 `0.074`，最低到 `0.008`；`actor/log_z` 从 `-1.090` 往 `-0.231` 靠近。当前 target-only chunk PowerFlow 容易把局部分布学成弱更新/退化更新，不能扩 80-step。

下一步判断：

- 不继续扩 `answer_value_margin + top2 + target_only`。
- 保留已经验证的 infra 修复：shard-balanced reorder 是必要且正确的。
- 方法侧要回到 2504.16084 的拆分原则，重新设计 label estimation 和 target calculation：
  - 不能只依赖短 probe 的局部 positive hit-rate。
  - 需要把 full-rollout majority label / original answer distribution 作为全局 anchor，避免 chunk target 把模型推向空 boxed 或局部格式吸引子。
  - 下一版优先尝试 `anchored PowerFlow target`：局部 chunk target 只对能保持 source full-answer label 或提升 full-answer consistency 的 candidate 加权，同时加入原始 full rollout answer 分布的保守 anchor，而不是 target-only。

## 2026-08-01 Anchored Standard Chunk PowerFlow 3-Step Smoke

目的：

- 验证 target-only 失败后，回到更保守的 standard PowerFlow loss。
- 把 full rollout 里的 source chunk 注入为 candidate 0，并用 teacher anchor 赋 score floor，避免局部短 probe target 单独主导 actor update。
- 这版仍遵循 2504.16084/TTRL 的拆分：先在同一 `query + prefix` state 上做 label/distribution estimation，再用 PowerFlow target 做参数更新；真实 GT 只用于事后 diag，不用于选 source。

配置：

```text
RUN_ID=ttrl_chunk_state_powerflow_anchor_standard3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
TEST_FREQ=2000000
FINAL_VAL_ENABLE=False
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
ttrl.chunk_state_score_mode=answer_value_margin
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_boundary_mode=mid
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
ttrl.chunk_state_value_margin=0.125
ttrl.chunk_state_value_topk=2
ttrl.chunk_state_format_guard=True
ttrl.chunk_state_label_consistent_only=True
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_source_chunk_candidate_index=0
ttrl.chunk_state_teacher_anchor_enable=True
ttrl.chunk_state_teacher_anchor_candidate_index=0
ttrl.chunk_state_teacher_anchor_score=0.5
actor_rollout_ref.actor.powerflow_enable=True
actor_rollout_ref.actor.powerflow_use_chunk_weights=True
actor_rollout_ref.actor.powerflow_use_boxed_reward=True
actor_rollout_ref.actor.powerflow_chunk_loss_mode=standard
actor_rollout_ref.actor.use_dynamic_bsz=False
```

训练侧统计：

```text
diag_jsonl_rows=96
chunk_state_source_chunk/injected_ratio: [1.000, 1.000, 1.000], avg=1.000
chunk_state_teacher_anchor/replaced_ratio: [1.000, 1.000, 1.000], avg=1.000
chunk_state/kept_state_ratio: [0.531, 0.500, 0.500], avg=0.510
chunk_state/powerflow_weight_mean: [0.531, 0.500, 0.500], avg=0.510
chunk_state/actor_batch_powerflow_weight_nonzero_ratio: [0.531, 0.500, 0.500], avg=0.510
chunk_state/actor_batch_powerflow_weight_zero_shard_ratio: [0.000, 0.000, 0.000], avg=0.000
chunk_state/positive_ratio: [0.139, 0.112, 0.125], avg=0.125
chunk_state/target_entropy: [0.725, 0.640, 0.683], avg=0.683
actor/powerflow_loss: [0.369, 0.227, 0.084], avg=0.227
actor/boxed_reward/mean: [0.131, 0.031, 0.100], avg=0.087
actor/powerflow_weight/nonzero_ratio: [0.531, 0.500, 0.500], avg=0.510
actor/log_z: [-0.998, -1.202, -1.189], avg=-1.130
actor/grad_norm: [21.184, 8.865, 4.272], avg=11.440
timing_s/gen: [53.577, 32.351, 22.230], avg=36.053
timing_s/chunk_state_score: [9.167, 19.346, 11.090], avg=13.201
timing_s/chunk_state_ref: [5.693, 1.788, 1.906], avg=3.129
timing_s/update_actor: [6.215, 5.441, 6.047], avg=5.901
```

结论：

- 3-step smoke 通过。source chunk 和 teacher anchor 均为 `1.000`，说明 anchor 按预期进入 actor batch。
- `actor_batch_powerflow_weight_zero_shard_ratio=0.000`，确认 shard-balanced reorder 在 anchored/standard 路径下也稳定。
- `actor/powerflow_loss` 三步均非零，但从 `0.369 -> 0.227 -> 0.084` 明显下降；这不是立即失败，但 20-step gate 必须重点看 loss 是否继续衰减到弱更新，以及 final validation 是否恢复到 MV baseline 附近。
- 性能侧仍是 rollout + chunk scoring 主导。稳态 step3 中 `gen=22.230s`、`chunk_state_score=11.090s`、`chunk_state_ref=1.906s`、`update_actor=6.047s`。

下一步：

- 启动同配置 20-step validation gate。
- 若 20-step `mean@16/maj@16` 仍明显低于 MV baseline，则不要继续扩 80-step；优先调整 anchor target 的 score/floor 或 chunk label estimation，而不是再调 infra 参数。

## 2026-08-01 Anchored Standard Chunk PowerFlow 20-Step Gate

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_anchor_standard20_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
ttrl.chunk_state_score_mode=answer_value_margin
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_boundary_mode=mid
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
ttrl.chunk_state_value_margin=0.125
ttrl.chunk_state_value_topk=2
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=True
ttrl.chunk_state_teacher_anchor_score=0.5
actor_rollout_ref.actor.powerflow_use_boxed_reward=True
actor_rollout_ref.actor.powerflow_chunk_loss_mode=standard
actor_rollout_ref.actor.use_dynamic_bsz=False
```

final validation：

```text
val-core/math/acc/mean@16 = 0.501
val-core/math/acc/maj@16/mean = 0.624
val-core/math/acc/best@16/mean = 0.858
val-aux/math/format_score/mean@16 = 0.9045
val-aux/math/format_score/maj@16/mean = 0.882708
timing_s/testing = 299.221
diag_jsonl_rows = 640
```

训练侧摘要：

```text
chunk_state_source_chunk/injected_ratio: all 1.000
chunk_state_teacher_anchor/replaced_ratio: all 1.000
chunk_state/actor_batch_powerflow_weight_zero_shard_ratio: all 0.000
chunk_state/kept_state_ratio: roughly 0.34-0.69, step20=0.594
actor/powerflow_loss: nonzero throughout, examples step1=0.105, step10=0.264, step11=0.046, step15=0.045, step20=0.084
timing_s/gen steady: about 21-24s after warmup
timing_s/chunk_state_score steady: about 8-12s, with occasional SymPy timeout spikes
timing_s/update_actor steady: about 4.7-5.5s
```

对比 target-only 20-step：

```text
target-only:        mean@16=0.427, maj@16=0.552, best@16=0.836, format_mean=0.888
anchored standard:  mean@16=0.501, maj@16=0.624, best@16=0.858, format_mean=0.9045
```

结论：

- Anchored standard 明显好于 target-only，说明 source chunk / teacher anchor / boxed residual 确实缓解了 target-only 的局部 target 退化。
- 但该版本仍没有接近 MV/TTRL 20-step baseline，不能扩 80-step。按当前 20-step 指标，它仍是失败 gate。
- 失败不是 infra 问题：20 步 `zero_shard_ratio=0.000`，anchor 全程注入，final validation 正常完成。
- 主要问题仍在训练语义：chunk-level improved distribution 的估计太弱，且 teacher anchor score floor=0.5 只提供保守锚点，没有足够约束模型保持 full-answer majority distribution；validation 样本中已经出现重复 boxed/题面复读现象。

下一版方向：

- 不继续扩 `target-only` 或当前 `teacher_anchor_score=0.5` 的 anchored standard。
- 优先改 target construction，而不是调 infra：
  - 把 full-rollout answer distribution 作为显式 source/teacher 分布，而不只是 candidate 0 score floor。
  - 对 next-chunk candidates 的 PowerFlow target 做 answer-consistency residual：只允许局部 probe 目标在不破坏 source majority answer 的条件下增益。
  - 对重复 boxed / 空 boxed / prompt-copy 引入 hard negative 或 format-collapse penalty，避免 format_score 看似高但 mean/maj 低。
  - 评估是否将 chunk state 从随机 mid 边界改成 answer-prefix-aware 边界，减少在无意义位置更新。

## 2026-08-01 Boxed Reward Alignment Fix

问题：

- 在 `_build_chunk_actor_batch` 中，`responses`、`target_weights`、`powerflow_chunk_weights` 会先经过 `keep_indices`，再为了避免 FSDP shard 权重清零做 balanced reorder。
- 但 `boxed_reward` 仍然从原始 `score_matrix.reshape(-1)[keep_indices]` 直接写入，没有跟随 balanced reorder。
- 这会导致 standard PowerFlow 中的 reward residual 和实际 next-chunk 样本错位；`target-only` 不受影响，但 anchored standard 会被污染。

修复：

- 新增 `flat_boxed_rewards = score_matrix.reshape(-1)[keep_indices]`。
- 如果触发 balanced reorder，`flat_boxed_rewards` 与 `responses/weights/powerflow_chunk_weights` 使用同一个 `balanced_order` 重排。
- `actor_batch["boxed_reward"]` 改为使用重排后的 `flat_boxed_rewards`。
- 新增 trainer 侧诊断指标：
  - `chunk_state/boxed_reward_mean`
  - `chunk_state/boxed_reward_weighted_mean`

3-step smoke：

```text
RUN_ID=ttrl_chunk_state_powerflow_anchor_boxfix3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=True
ttrl.chunk_state_teacher_anchor_score=0.5
actor_rollout_ref.actor.powerflow_use_boxed_reward=True
actor_rollout_ref.actor.powerflow_chunk_loss_mode=standard
```

关键指标：

```text
step1:
  chunk_state/boxed_reward_mean=0.139
  chunk_state/boxed_reward_weighted_mean=0.561
  actor/boxed_reward/mean=0.301
  actor/powerflow_loss=0.199
  zero_shard_ratio=0.000
  timing_s/gen=51.099, chunk_state_score=9.116, chunk_state_ref=5.689, update_actor=6.156

step2:
  chunk_state/boxed_reward_mean=0.120
  chunk_state/boxed_reward_weighted_mean=0.492
  actor/boxed_reward/mean=0.300
  actor/powerflow_loss=0.074
  zero_shard_ratio=0.000
  timing_s/gen=22.520, chunk_state_score=9.087, chunk_state_ref=1.930, update_actor=5.859

step3:
  chunk_state/boxed_reward_mean=0.147
  chunk_state/boxed_reward_weighted_mean=0.593
  actor/boxed_reward/mean=0.282
  actor/powerflow_loss=0.034
  zero_shard_ratio=0.000
  timing_s/gen=23.298, chunk_state_score=8.802, chunk_state_ref=1.935, update_actor=6.008
```

结论：

- 修复后 actor 侧 `boxed_reward/mean` 不再像修复前 anchored 3-step 那样偏低到 `0.031-0.131` 的错位形态，step1 对比也从修复前 `actor/boxed_reward/mean=0.131` 变成 `0.301`。
- 三步 `actor/powerflow_loss` 均非零，`zero_shard_ratio=0.000`，说明 reward alignment 修复没有破坏 balanced shard 逻辑。
- 下一步必须重新跑 20-step validation gate。此前 anchored standard 20-step 的失败结论是在 reward/sample 错位条件下得到的，不能作为修复后版本的最终判断。
- 方法设计继续遵守 TTRL 原文 arXiv 2504.16084 的拆分：同一 state 下多样本先做 label/distribution estimation，再做 reward/target calculation；短 probe hit-rate 只能作为局部 target 的证据之一，不能直接替代最终 answer label。

## 2026-08-01 Boxfix Anchored Standard 20-Step Gate

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_anchor_boxfix20_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
ttrl.chunk_state_score_mode=answer_value_margin
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_boundary_mode=mid
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
ttrl.chunk_state_value_margin=0.125
ttrl.chunk_state_value_topk=2
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=True
ttrl.chunk_state_teacher_anchor_score=0.5
actor_rollout_ref.actor.powerflow_use_boxed_reward=True
actor_rollout_ref.actor.powerflow_chunk_loss_mode=standard
actor_rollout_ref.actor.use_dynamic_bsz=False
```

final validation：

```text
val-core/math/acc/mean@16 = 0.42675
val-core/math/acc/maj@16/mean = 0.54592
val-core/math/acc/best@16/mean = 0.845676
val-aux/math/format_score/mean@16 = 0.892625
val-aux/math/format_score/maj@16/mean = 0.86140
timing_s/testing = 303.088
diag_jsonl_rows = 640
```

训练侧摘要：

```text
step2+ avg chunk_state/kept_state_ratio = 0.447
step2+ avg chunk_state/boxed_reward_weighted_mean = 0.511
step2+ avg actor/powerflow_loss = 0.214
step2+ zero_shard_ratio = 0.000 for all steps
step2+ avg timing_s/gen = 24.417
step2+ avg timing_s/chunk_state_score = 9.750
step2+ avg timing_s/chunk_state_ref = 1.769
step2+ avg timing_s/update_actor = 5.496
```

对比：

```text
target-only 20-step:             mean@16=0.427, maj@16=0.552, best@16=0.836, format_mean=0.888
anchored standard 20-step:       mean@16=0.501, maj@16=0.624, best@16=0.858, format_mean=0.9045
boxfix anchored standard 20-step: mean@16=0.42675, maj@16=0.54592, best@16=0.845676, format_mean=0.892625
```

观察到的退化样本：

- validation generation 中出现多条重复 `\boxed{}` 的 prompt-copy / format-copy 输出。
- 也出现一个回答把多个题目串在一起继续解的样本，说明局部 chunk update 仍在破坏 instruction / answer boundary。
- `format_score` 仍有 0.893，但 `mean@16/maj@16` 很低，说明重复 boxed 可以通过部分格式检查但不带来正确性。

结论：

- Boxed reward alignment 修复是必要的工程修复：20 步中 actor 侧 reward 和 PowerFlow loss 都正常，`zero_shard_ratio=0.000`，不再是 actor batch 传递问题。
- 但 boxfix 后 20-step validation 明确失败，甚至低于修复前 anchored standard。不能扩到 80-step。
- 失败已经转化为方法语义问题：`teacher_anchor_score=0.5` 只把 source chunk 当成一个候选锚点，没有把 full-answer majority distribution 作为必须保持的 label distribution；局部 probe margin 可以奖励“看似有希望”的 chunk，但不能防止 repetition/prompt-copy。
- 后续不应继续调这个配置的 infra。下一版必须改 target construction：
  - 显式 full-answer distribution anchor：从 32 条 full rollout 估计 prompt-level answer distribution，然后 next-chunk target 只能在不偏离 anchor answer set 的条件下 sharpen。
  - 增加 hard negative：重复 boxed、空 boxed、prompt-copy、多题串联、超长 boxed spam 的 chunk candidate 权重置零或给负 residual。
  - 边界从随机 mid 改为 answer-prefix-aware / reasoning-boundary-aware，避免在题面或无语义位置做 chunk actor update。
  - 继续保持 TTRL 原文 label estimation 与 reward calculation 拆分：先从 full rollout 和 chunk probe 估计 state label/distribution，再计算 PowerFlow target。

## 2026-08-01 Full-Answer Target Guard 3-Step Smoke

代码改动：

- 新增 `_compute_full_rollout_answer_metadata`，在 chunk scoring 前从 32 条 full rollout 提取 prompt-level answer distribution 和每条 source answer 的 mass。
- 新增 `_apply_chunk_state_target_guard`，在 `answer_value_margin` score 和 teacher/source anchor 后执行：
  - 空答案、重复 boxed、超长答案、明显 prompt-copy / 多题串联 candidate 置零。
  - guard 按 probe bad ratio 过滤，默认 `bad_probe_ratio >= 0.5` 才清零 candidate，避免单个 probe 误杀。
  - 保留 TTRL arXiv 2504.16084 的顺序：先 full rollout label/distribution estimation，再计算 chunk target。
- `ppo_trainer_ttrl.yaml` 加入默认关闭的 guard 配置，不影响既有实验。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_anchor_guard3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
TEST_FREQ=2000000
FINAL_VAL_ENABLE=False
TTRL_RUNTIME_DIR=/tmp/csguard3b
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_max_boxed_count=2
ttrl.chunk_state_target_guard_bad_probe_ratio=0.5
ttrl.chunk_state_target_guard_min_answer_mass=0.0
ttrl.chunk_state_target_guard_anchor=True
```

关键指标：

```text
step1:
  target_guard kept_candidate_ratio=0.641, zeroed_candidate_ratio=0.359
  empty_answer_probe_ratio=0.279, repeated_boxed_probe_ratio=0.078
  score_mean_before=0.090, score_mean_after=0.085
  kept_state_ratio=0.438, zero_shard_ratio=0.000
  actor/powerflow_loss=0.200
  timing_s/gen=51.167, chunk_state_score=11.357, chunk_state_ref=5.740, update_actor=6.188

step2:
  target_guard kept_candidate_ratio=0.637, zeroed_candidate_ratio=0.363
  empty_answer_probe_ratio=0.305, repeated_boxed_probe_ratio=0.048
  score_mean_before=0.078, score_mean_after=0.076
  kept_state_ratio=0.562, zero_shard_ratio=0.000
  actor/powerflow_loss=0.321
  timing_s/gen=23.647, chunk_state_score=10.171, chunk_state_ref=1.779, update_actor=5.396

step3:
  target_guard kept_candidate_ratio=0.664, zeroed_candidate_ratio=0.336
  empty_answer_probe_ratio=0.212, repeated_boxed_probe_ratio=0.101
  score_mean_before=0.084, score_mean_after=0.077
  kept_state_ratio=0.312, zero_shard_ratio=0.000
  actor/powerflow_loss=0.125
  timing_s/gen=22.703, chunk_state_score=8.930, chunk_state_ref=1.794, update_actor=5.339
```

infra 观察：

- vLLM 配置仍为 `attention_config.backend=FLASH_ATTN`，运行中出现 FlashInfer autotune 和 CUDA graph capture。
- NCCL 日志显示 8 卡 P2P/NVLS 可用，`Check P2P Type isAllDirectP2p 1`，`NCCL_NVLS_ENABLE=1`。
- actor dynamic batch 仍关闭，PowerFlow loss 为 `standard`，`powerflow_use_boxed_reward=True`。

结论：

- 这是健康 smoke，不是有效算法结果：final validation 按计划跳过。
- guard 没有把训练打空：`actor/powerflow_loss` 三步非零，`actor_batch_powerflow_weight_zero_shard_ratio=0.000`。
- guard 主要清掉空答案和重复 boxed，未观察到 prompt-copy / multi-problem 触发；这说明之前 20-step validation 中的多题串联可能在更长训练后才显著出现。
- 下一步可跑 20-step validation gate。若 20-step 仍失败，优先尝试：
  - `chunk_state_target_guard_min_answer_mass > 0`，限制 chunk target 必须落在 full-rollout answer distribution 内。
  - 去掉 `teacher_anchor_score` 强行 floor，只保留 source chunk injection + distribution guard。
  - 引入 answer-boundary-aware state selection，减少 boundary=0 和无语义 mid-boundary。

## 2026-08-01 Full-Answer Target Guard 20-Step Gate

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_anchor_guard20_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
TTRL_RUNTIME_DIR=/tmp/csguard20
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
ttrl.chunk_state_score_mode=answer_value_margin
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_boundary_mode=mid
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
ttrl.chunk_state_value_margin=0.125
ttrl.chunk_state_value_topk=2
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=True
ttrl.chunk_state_teacher_anchor_score=0.5
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_max_boxed_count=2
ttrl.chunk_state_target_guard_bad_probe_ratio=0.5
ttrl.chunk_state_target_guard_min_answer_mass=0.0
ttrl.chunk_state_target_guard_anchor=True
actor_rollout_ref.actor.powerflow_use_boxed_reward=True
actor_rollout_ref.actor.powerflow_chunk_loss_mode=standard
actor_rollout_ref.actor.use_dynamic_bsz=False
```

final validation：

```text
val-core/math/acc/mean@16 = 0.455875
val-core/math/acc/maj@16/mean = 0.573016
val-core/math/acc/best@16/mean = 0.845844
val-aux/math/format_score/mean@16 = 0.893625
val-aux/math/format_score/maj@16/mean = 0.868344
timing_s/testing = 297.691
diag_jsonl_rows = 640
```

20 step 平均 timing：

```text
timing_s/gen avg = 25.363, min = 21.176, max = 51.145
timing_s/chunk_state_probe avg = 6.705, min = 6.269, max = 7.208
timing_s/chunk_state_score avg = 12.967, min = 8.758, max = 30.182
timing_s/chunk_state_ref avg = 1.906, min = 1.481, max = 5.694
timing_s/update_actor avg = 5.331, min = 4.550, max = 6.189
progress avg s/it ~= 63.9
progress avg s/it excluding warmup ~= 66.5
```

20 step 训练侧摘要：

```text
target_guard kept_candidate_ratio avg = 0.639, min = 0.480, max = 0.762
target_guard zeroed_candidate_ratio avg = 0.361, min = 0.238, max = 0.520
chunk_state/kept_state_ratio avg = 0.424, min = 0.188, max = 0.625
chunk_state/positive_ratio avg = 0.079, min = 0.047, max = 0.101
chunk_state_actor_span/response_len_mean avg = 123.644
actor/powerflow_loss avg = 0.222, min = 0.034, max = 0.612
actor_batch_powerflow_weight_zero_shard_ratio = 0.000 for all steps
```

尾部异常：

```text
RuntimeError: DataLoader worker (pid 1740812) is killed by signal: Killed.
```

这个异常发生在 final validation metrics 已经完整打印之后。结论上这次 gate 有可用数值，但不是干净退出的 run；后续长跑前需要降低 validation/日志内存压力，尤其是避免 validation 打印超长样本列表。

结论：

- guard20 训练链路稳定到 step20，PowerFlow loss 非零，`zero_shard_ratio=0.000`，说明 actor batch / FSDP shard 没有被 guard 打空。
- 结果仍然失败：`mean@16=0.4559`、`maj@16=0.5730` 明显低于 MV step20 gate；`best@16=0.8458` 说明 sampling 里仍有正确轨迹，但训练后主分布没有变好。
- guard 清掉了大量空答案 / 重复 boxed candidate，但只是在 candidate 层过滤坏样本，不能解决局部 chunk target 与 full-answer policy 的错配。
- 当前端到端速度约 64-66s/step，actor update 只有约 5.3s；慢点来自 full rollout generation、probe generation 和 SymPy/answer scoring，不是短 chunk actor update。
- 结合 arXiv 2504.16084 的处理方式，下一版不应再继续在这个 guard 上微调阈值，而应更彻底拆开两件事：
  - label / answer distribution estimation：先从 32 条完整 rollout 估计 prompt-level answer distribution 和 state-level candidate distribution。
  - reward / target calculation：再把 chunk probe 作为局部 evidence，用 full-answer distribution 约束 PowerFlow target，避免 probe 偶然成功、重复 boxed、prompt-copy 直接进入 actor target。

下一步：

- 不扩 guard20 到 80 step。
- 优先实现 `chunk_state_target_guard_min_answer_mass > 0` 或更强的 distribution-in-support target：candidate 的 probe answer 不在 full-rollout answer support 中时，权重置零或强降权。
- 去掉 `teacher_anchor_score=0.5` 的硬 floor 做对照；下一版先验证 support-only target，再显式重跑 source chunk injection + distribution guard，判断 anchor floor 是否在错误地保护局部坏 chunk。
- 设计 answer-boundary-aware state selection，尽量在 reasoning/answer 边界附近截 state，而不是随机 mid boundary。
- validation 侧需要减少超长样本 stdout，避免 metrics 后 DataLoader/Ray worker 清理阶段被 kill。

## 2026-08-01 Full-Answer Support Target 3-Step Smoke

目的：

- 接上 guard20 的失败结论，不继续微调普通 guard 阈值。
- 更接近 arXiv 2504.16084 的拆分：先从 32 条 full rollout 估计 answer distribution，再用该 distribution 约束 chunk target。
- 去掉 `teacher_anchor_score=0.5` 的硬 floor，先跑 support-only 版本；避免 teacher anchor floor 保护局部坏 chunk。
- 启用 full-answer support guard：probe answer 不在 full rollout answer support 中时置零；同时把 prompt answer mass 注入 score，形成 support-constrained target。
- 注意：本次实际配置中 `answer_value_margin` wrapper 最后覆盖了 `ttrl.chunk_state_source_chunk_enable=False`。因此这次 smoke 是 support-only target，不是 source-chunk-injection target。后续 20-step gate 必须显式补 `ttrl.chunk_state_source_chunk_enable=True` 并重新确认 Hydra final config。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_support3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
TEST_FREQ=2000000
FINAL_VAL_ENABLE=False
TTRL_RUNTIME_DIR=/tmp/cssup3
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
ttrl.chunk_state_score_mode=answer_value_margin
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_boundary_mode=mid
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
ttrl.chunk_state_source_chunk_enable=False
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_bad_probe_ratio=0.5
ttrl.chunk_state_target_guard_min_answer_mass=0.03125
ttrl.chunk_state_target_guard_use_distribution_score=True
actor_rollout_ref.actor.powerflow_use_boxed_reward=True
actor_rollout_ref.actor.powerflow_chunk_loss_mode=standard
actor_rollout_ref.actor.use_dynamic_bsz=False
```

关键 step 指标：

```text
step1:
  kept_candidate_ratio=0.422, zeroed_candidate_ratio=0.578
  distribution_oov_probe_ratio=0.528
  score_mean_before=0.090, score_mean_after=0.201
  kept_state_ratio=0.438, positive_ratio=0.088
  boxed_reward_weighted_mean=0.638
  actor/powerflow_loss=0.629, zero_shard_ratio=0.000
  timing_s/gen=51.530, chunk_state_probe=6.539, chunk_state_score=11.273, update_actor=5.909

step2:
  kept_candidate_ratio=0.480, zeroed_candidate_ratio=0.520
  distribution_oov_probe_ratio=0.442
  score_mean_before=0.090, score_mean_after=0.257
  kept_state_ratio=0.594, positive_ratio=0.110
  boxed_reward_weighted_mean=0.668
  actor/powerflow_loss=1.515, zero_shard_ratio=0.000
  timing_s/gen=22.825, chunk_state_probe=7.427, chunk_state_score=11.259, update_actor=5.081

step3:
  kept_candidate_ratio=0.520, zeroed_candidate_ratio=0.480
  distribution_oov_probe_ratio=0.439
  score_mean_before=0.099, score_mean_after=0.281
  kept_state_ratio=0.531, positive_ratio=0.113
  boxed_reward_weighted_mean=0.659
  actor/powerflow_loss=1.631, zero_shard_ratio=0.000
  timing_s/gen=22.548, chunk_state_probe=6.489, chunk_state_score=10.320, update_actor=5.162
```

3 step 平均：

```text
kept_candidate_ratio avg = 0.474
zeroed_candidate_ratio avg = 0.526
distribution_oov_probe_ratio avg = 0.470
score_mean_before avg = 0.093
score_mean_after avg = 0.246
kept_state_ratio avg = 0.521
positive_ratio avg = 0.104
boxed_reward_weighted_mean avg = 0.655
actor/powerflow_loss avg = 1.258
actor/grad_norm avg = 39.546
timing_s/update_actor avg = 5.384
diag_jsonl_rows = 96
```

结论：

- support-constrained target 路径真实生效：约 44%-53% probe 因 answer 不在 full-rollout support 中被 OOV guard 拦下，`score_mean_after` 从约 0.09 提升到 0.20-0.28。
- 去掉 `teacher_anchor_score` 后 actor batch 仍没有打空，三步 `zero_shard_ratio=0.000`，PowerFlow loss 正常非零。
- `boxed_reward_weighted_mean` 升到约 0.65，说明 distribution score 正在把权重集中到 full-answer support 内的 candidate。
- 风险是 actor loss/grad norm 明显高于 guard20：`actor/powerflow_loss` 到 1.5-1.6，`grad_norm` 到 32-52。20-step gate 必须观察是否过强更新导致 mean/maj 继续塌。
- 该 smoke 没有 final validation，不能判断效果；下一步先跑显式 source-chunk-injection 的 20-step gate，并必须以 step20 mean/maj 为硬判据。

## 2026-08-01 Source-Chunk + Support Target 20-Step Gate

背景：

- 前一次 `support3` smoke 实际是 support-only target，因为 launcher wrapper 最后覆盖了 `ttrl.chunk_state_source_chunk_enable=False`。
- 本次重新确认 Hydra final config 后运行，确保 `ttrl.chunk_state_source_chunk_enable=True`，并在 20 step 做 final validation。
- 用户补充的 arXiv 2504.16084 实际是 TTRL 原文。对当前 chunk 方案的约束是：reward / target 仍应来自 full rollout group 的无标签分布估计，probe 只能作为局部 evidence，不能让任意 probe answer 直接主导 actor target。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_support_src20b_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
TTRL_RUNTIME_DIR=/tmp/cssrc20b
OUTPUT_DIR=/tmp/ttrl_b200/checkpoints/ttrl_chunk_state_powerflow_support_src20b_mid_c128_probe4_b32_r32_v64_20260801
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
actor_rollout_ref.rollout.val_kwargs.n=16
actor_rollout_ref.actor.use_dynamic_bsz=False
actor_rollout_ref.actor.powerflow_enable=True
actor_rollout_ref.actor.powerflow_use_boxed_reward=True
actor_rollout_ref.actor.powerflow_use_chunk_weights=True
actor_rollout_ref.actor.powerflow_chunk_loss_mode=standard
ttrl.chunk_state_enable=True
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_boundary_mode=mid
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_probe_max_tokens=1024
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_max_boxed_count=2
ttrl.chunk_state_target_guard_bad_probe_ratio=0.5
ttrl.chunk_state_target_guard_min_answer_mass=0.03125
ttrl.chunk_state_target_guard_anchor=True
ttrl.chunk_state_target_guard_use_distribution_score=True
```

产物：

```text
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_src20b_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_src20b_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 640
```

最终 validation：

```text
val-core/math/acc/mean@16 = 0.410
val-core/math/acc/maj@16  = 0.523
val-core/math/acc/best@16 = 0.833
val-aux/math/format_score/mean@16 = 0.888
val-aux/math/format_score/maj@16  = 0.855
timing_s/testing = 301.386s
```

训练 timing，统计 step2-step20：

```text
timing_s/gen avg = 23.990s, min = 21.540s, max = 32.348s
timing_s/chunk_state_probe avg = 6.690s, min = 6.269s, max = 7.168s
timing_s/chunk_state_score avg = 11.913s, min = 9.288s, max = 16.524s
timing_s/chunk_state_ref avg = 1.745s, min = 1.548s, max = 1.953s
timing_s/update_actor avg = 5.331s, min = 4.692s, max = 5.909s
chunk_state_source_chunk/mean_len avg = 123.791 tokens
```

target / diagnostic 统计：

```text
chunk_state_source_chunk/injected_ratio = 1.000 throughout
chunk_state_target_guard/kept_candidate_ratio avg = 0.393
chunk_state_target_guard/distribution_oov_probe_ratio avg = 0.543
chunk_state/actor_batch_powerflow_weight_nonzero_ratio avg = 0.383
chunk_state/positive_ratio avg = 0.087
chunk_state/informative_ratio avg = 0.559
chunk_state/target_entropy avg = 1.316

diag source_original_correct avg = 0.720
diag source_majority_consistent avg = 1.000
diag answer_coverage avg = 0.734
diag probe_mean avg = 0.192
diag probe_max avg = 0.390
diag all_negative_ratio = 0.397
diag mixed_ratio = 0.481
diag boundary avg = 546.6 tokens
diag source_response_len avg = 1269.7 tokens
```

异常与观察：

- Run 正常打印 step20 和 final validation metrics，worker shell 正常退出。
- Validation stdout 中出现多条明显退化样本，典型是大量重复 `\boxed{}` 或重复 assistant solution，说明训练后主分布出现格式/重复退化。这个现象与 `format_score/mean@16=0.888` 一致，低于理想 gate。
- 训练时 source chunk 注入稳定生效，actor span 长度约 124 tokens，actor update 约 5.3s，证明 chunk actor update 本身不慢。
- 端到端 step 仍约 60s 量级，主要来自 full rollout generation、chunk probe、SymPy/answer scoring，而不是 actor update。
- target 仍然偏稀疏：guard 后候选保留率约 39%，actor 非零 PowerFlow 权重约 38%，probe answer OOV 约 54%。这说明当前 chunk candidate/probe 很多没有落在 full-rollout answer support 上。

结论：

- 这是失败 gate：`mean@16=0.410`、`maj@16=0.523` 明显低于 MV / PowerFlow baseline，不能扩到 80 step。
- Source chunk injection 没有解决 full-answer target 与局部 next-chunk target 的错配；它只是让 actor 更新在真实 source chunk 上发生。
- 失败形态不是 actor update 不够强，而是 target 语义有毒：局部 chunk 训练把模型推向重复 boxed / 格式退化，同时只有少量 candidate 有有效 support mass。
- 2504.16084/TTRL 语义要求 target 来自 group-level distribution estimation。下一版不能再让 probe raw correctness 直接决定局部 target，必须把 chunk candidate 的价值绑定到 full rollout answer support / answer distribution improvement 上。

下一步设计：

- 停止继续扩展 `support_src20b`。
- 保留 PowerFlow loss，但重做 target construction：
  - 先采样 32 条 full rollout，得到 prompt-level answer support、majority answer、answer mass 和 format/duplicate 统计。
  - 对每个 chunk state 采 next-chunk candidate 后，不直接用 probe raw score；先把 probe completion 映射回 full-answer support，计算 support mass gain / majority-answer mass gain。
  - 对 OOV、empty、repeated boxed、prompt-copy candidate 做强惩罚或置零；对 in-support 且提升 answer mass 的 candidate 才构造 PowerFlow sharpened target。
  - 增加 anti-repetition target guard：候选 chunk 内重复 `\boxed{}` 或 assistant marker 时直接 zero，不只看最终 probe。
  - 优先选择 reasoning 中后段 boundary，减少 boundary=0 或过早 state；这更接近 chunk-level search-state improvement，而不是从 prompt 开头制造局部捷径。
- 先做 3-step smoke 验证 target nonzero ratio、OOV ratio、重复 boxed 率，再跑 20-step gate；硬门槛仍是 step20 `mean@16` 不低于 MV 20-step 对齐线。

## 2026-08-01 Mass-Gain + Candidate Guard 3-Step Smoke

目的：

- 接上 `support_src20b` 的失败结论，先不扩 80 step。
- 增加两个默认关闭的 target 修正：
  - candidate-level anti-repetition guard：单独检查 next chunk 自身，提前拦截重复 `\boxed{}`、assistant/user/system marker、prompt-copy。
  - full-answer mass gain：distribution score 从 raw prompt answer mass 改成 `max(0, candidate_answer_mass - source_answer_mass)`，让 target 只奖励相对 source full answer 更高的 answer support。
- 保持 PowerFlow loss、source chunk injection、batch32/rollout32、dynamic batch off。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_support_massgain_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_min_answer_mass=0.03125
ttrl.chunk_state_target_guard_use_distribution_score=True
ttrl.chunk_state_target_guard_use_mass_gain=True
ttrl.chunk_state_target_guard_candidate_enable=True
ttrl.chunk_state_target_guard_candidate_max_boxed_count=1
ttrl.chunk_state_target_guard_candidate_assistant_marker=True
```

产物：

```text
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_massgain_src3_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_massgain_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 96
```

3 step 平均：

```text
kept_candidate_ratio = 0.405
score_mean_before = 0.082
score_mean_after = 0.074
prompt_mass_mean_after_mass_gain = 0.000
raw_prompt_mass_mean = 0.224
source_answer_mass_mean = 0.377
distribution_oov_probe_ratio = 0.541
candidate_repeated_boxed_ratio = 0.009
actor_batch_powerflow_weight_nonzero_ratio = 0.406
positive_ratio = 0.074
informative_ratio = 0.531
target_entropy = 1.380
boxed_reward_weighted_mean = 0.546
actor/powerflow_loss = 0.098
actor/grad_norm = 6.626
timing_s/update_actor = 5.690s
```

诊断：

```text
diag source_original_correct = 0.781
diag source_majority_consistent = 1.000
diag answer_coverage = 0.731
diag probe_mean = 0.0745
diag probe_max = 0.319
diag all_negative_ratio = 0.469
diag mixed_ratio = 0.458
```

结论：

- 工程链路通过：新开关真实生效，3 step 正常结束，final validation 按预期跳过。
- Candidate-level guard 有信号但占比不大：candidate 自身重复 boxed 约 0.9%。它适合作为低风险防退化 guard 保留。
- Mass-gain 过严：`raw_prompt_mass_mean=0.224`，但减去 source answer mass 后 `prompt_mass_mean_after_mass_gain=0.000`，说明当前 source chunk 已经来自多数一致轨迹，candidate probe 很少超过 source answer mass。
- 这版不能扩 20 step。它会把 distribution support signal 全部清掉，剩下主要是原始 `answer_value_margin` score，无法解决 `support_src20b` 的语义问题。

下一步：

- 保留 candidate anti-repetition guard。
- 关闭 `chunk_state_target_guard_use_mass_gain`，回到 raw full-answer support mass 作为 distribution score。
- 另做一个 3-step smoke：`source_chunk + support target + candidate anti-repetition`，看是否在不清零 distribution score 的情况下改善重复 boxed 风险。
- 如果 smoke 健康，再跑 20-step gate；硬门槛仍是 step20 `mean@16` 不能低于 MV 20-step 两点以上。

## 2026-08-01 Raw Support + Candidate Anti-Repeat 3-Step Smoke

目的：

- 验证上一个 mass-gain 版本的问题是否来自 `max(0, candidate_mass - source_mass)` 过严，而不是 candidate guard 本身。
- 保留 candidate-level anti-repetition guard，关闭 mass-gain，继续把 target 绑定到 full rollout answer support / distribution score。
- 仍然使用 PowerFlow loss、source chunk injection、batch32/rollout32、dynamic batch off。
- `2504.16084` 是 TTRL 原文，约束这里的 chunk target 不能用 GT，也不能让任意短 probe 直接当 teacher；target 必须尽量来自同组完整 rollout 的 answer distribution / majority prior。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_support_antirepeat_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_min_answer_mass=0.03125
ttrl.chunk_state_target_guard_use_distribution_score=True
ttrl.chunk_state_target_guard_use_mass_gain=False
ttrl.chunk_state_target_guard_candidate_enable=True
ttrl.chunk_state_target_guard_candidate_max_boxed_count=1
ttrl.chunk_state_target_guard_candidate_assistant_marker=True
```

产物：

```text
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_antirepeat_src3_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_antirepeat_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 96
```

Step 级指标：

```text
step  gen     chunks  probe  score   ref    update  total_est  kept  score_after  prompt_mass  oov    cand_repeat_boxed  pf_nonzero
1     51.321  1.077   6.517  11.485  5.598  5.839   81.837     0.395 0.190        0.236        0.530  0.012              0.406
2     33.069  0.973   6.376  10.668  1.577  4.713   57.376     0.531 0.279        0.296        0.434  0.000              0.531
3     22.835  1.111   6.760  9.530   1.807  5.546   47.589     0.391 0.176        0.231        0.548  0.004              0.375
```

3 step 平均：

```text
kept_candidate_ratio = 0.439
score_mean_after = 0.215
prompt_mass_mean = 0.254
distribution_oov_probe_ratio = 0.504
candidate_repeated_boxed_ratio = 0.005
actor_batch_powerflow_weight_nonzero_ratio = 0.437
boxed_reward_weighted_mean = 0.647
actor/powerflow_loss = 0.168
actor/grad_norm = 8.812
timing_s/update_actor = 5.366s
timing_total_est = 62.267s
```

Step 2-3 稳态近似：

```text
timing_s/gen = 27.952s
timing_s/chunk_state_chunks = 1.042s
timing_s/chunk_state_probe = 6.568s
timing_s/chunk_state_score = 10.099s
timing_s/chunk_state_ref = 1.692s
timing_s/update_actor = 5.130s
timing_total_est = 52.483s
```

诊断：

- 工程链路通过：3 step 正常结束，final validation 按预期跳过，diag 96 行完整。
- 配置确认：`source_chunk_enable=True`，`target_guard_use_mass_gain=False`，`candidate_guard_enable=True`，`use_distribution_score=True`，没有被 base wrapper 覆盖。
- Infra 确认：vLLM attention backend 打印为 `FLASH_ATTN`，FlashInfer autotune、CUDA graph capture、actor flash-attn monkey patch、Triton fused kernels、NCCL NVLS/P2P 均出现。
- 与 mass-gain 对比：raw support mass 没有被清零，`prompt_mass_mean=0.254`；mass-gain 版本对应值为 0。
- Candidate guard 没有误伤主信号：candidate 自身重复 boxed 只有约 0.5%，但它能拦住最明显的重复/assistant-marker 退化源。
- 仍有风险：distribution OOV probe ratio 约 50%，说明短 probe 回到 full rollout answer support 的比例还不高；这可能继续限制 20-step gate 的上限。

结论：

- 这是健康 smoke，可以扩 20-step gate。
- 当前方案比 mass-gain 更符合 TTRL 原文的 group-level answer distribution 语义：不使用 GT，source 来自 majority-consistent full rollout，target 使用 full rollout answer support 过滤/加权 probe。
- 当前端到端额外成本主要是 `chunk_state_score` 和 probe，不是 actor update；chunk actor update 约 5s，长度约 121-123 tokens，符合“chunk update 更轻”的预期。

下一步：

- 启动同配置 20-step gate，打开 final validation。
- Gate 目标：至少不能复现 `support_src20b` 的 repeated boxed 退化；step20 `mean@16` 不能低于 MV 20-step 两点以上。
- 若 20-step 指标健康，再扩 80-step pilot；若仍低，下一步优先减少 OOV：例如增大 full rollout support 样本数、用 answer-support matching 而不是 raw probe answer、或把 state boundary 限制到更稳定的中后段。


## 2026-08-01 Raw Support + Candidate Anti-Repeat 20-Step Gate

目的：

- 扩展上一节健康的 3-step smoke 到 20-step gate，确认 `source_chunk + raw full-answer support target + candidate anti-repeat guard` 是否能避免 `support_src20b` 的重复 boxed 退化。
- 继续保持 TTRL 原文 `2504.16084` 的约束：不使用 GT 选 source / target；source 和 target 语义绑定到同组 full rollout 的 answer distribution / majority prior；chunk state 只作为训练状态和局部 transition，不让短 probe 自己变成任意 teacher。
- 用户补充的 chunk 处理参考 `https://arxiv.org/pdf/2504.16084` 后，下一轮实现需要更严格区分：full rollout group 给全局 answer support，chunk/probe 只估计 state transition 是否把质量流向该 support。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_support_antirepeat_src20_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
VAL_BEFORE_TRAIN=False
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_min_answer_mass=0.03125
ttrl.chunk_state_target_guard_use_distribution_score=True
ttrl.chunk_state_target_guard_use_mass_gain=False
ttrl.chunk_state_target_guard_candidate_enable=True
ttrl.chunk_state_target_guard_candidate_max_boxed_count=1
ttrl.chunk_state_target_guard_candidate_assistant_marker=True
```

产物：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_antirepeat_src20_mid_c128_probe4_b32_r32_v64_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_antirepeat_src20_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_antirepeat_src20_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 640
```

训练侧结果：

```text
main_log_steps = 19   # stdout step 指标打印到 step 19；diag 640 行显示 20 个 batch 的 chunk 诊断已写完
avg_total_est_step2plus = 49.689s
last_total_est = 46.782s
avg timing_s/gen = 23.484s
avg timing_s/chunk_state_probe = 6.556s
avg timing_s/chunk_state_score = 11.667s
avg timing_s/chunk_state_ref = 1.694s
avg timing_s/update_actor = 5.259s
avg kept_candidate_ratio = 0.431
avg prompt_mass_mean = 0.271
avg source_answer_mass_mean = 0.419
avg distribution_oov_probe_ratio = 0.510
avg repeated_boxed_probe_ratio = 0.085
avg candidate_repeated_boxed_probe_ratio = 0.014
avg actor_batch_powerflow_weight_nonzero_ratio = 0.432
avg positive_ratio = 0.095
avg source_original_acc_mean = 0.778
avg prompt_original_pass = 0.919
```

Final validation 状态：

```text
validation_generation_started = True
validation_generation_end = True
final_metric_found = True
val-core/math/acc/mean@16 = 0.525125
val-core/math/acc/maj@16 = 0.657004
val-core/math/acc/best@16 = 0.870802
val-aux/math/format_score/mean@16 = 0.895796
checkpoint_dir_found = False
validation_tail_empty_boxed_count = 288383
validation_tail_assistant_marker_count = 375
```

结论：

- 20-step gate 失败，不能扩 80-step。
- 训练侧没有崩：diag 640 行完整，step2+ 约 50s，chunk actor update 约 5.3s，candidate guard 后 candidate 自身重复 boxed 只有约 1.4%。
- 失败发生在训练语义而不是 actor update 性能：final validation 已经生成结束并落出指标，但指标很差，`mean@16=0.525`、`maj@16=0.657`、`best@16=0.871`，显著低于 MV 20-step 基线；validation 输出尾部出现极大量空 `\boxed{}` 和 assistant marker 串扰，说明完整 policy 仍然发生退化。
- 这说明 candidate-level anti-repeat guard 只是局部清洗，并没有改变 target 的根本问题：短 probe 到 full rollout answer support 的 OOV 仍约 51%，正样本比例只有约 9.5%，PowerFlow 权重会把少量局部高分 chunk 过强蒸馏，导致完整答案分布被污染。
- 和 `2504.16084` 的 TTRL 目标相比，当前版本仍然过早把 chunk probe 结果当成局部监督；下一版需要让 chunk 的目标更像 full group answer distribution 的 state-conditioned improvement，而不是直接奖励一个短 continuation 的 boxed answer。

下一步：

- 停止这条路线继续扩步。
- 重新设计 chunk transition target：先完整 rollout 32 条形成 group answer distribution，再从正确/多数一致轨迹上截 state；对 next-chunk continuation 做后续 rollout/probe 时，只计算它把后续 completion 引向 full-group support/mass 的概率提升。
- 优先改成 PowerFlow-style distribution matching：target 权重来自 `future completion answer mass under original group support`，并加入 length-normalized chunk likelihood / answer-support transport，而不是 raw short-probe answer score。
- Anti-repeat guard 保留为安全 guard，但不能作为主要创新或主要监督。


## 2026-08-01 Answer-Support MassAvg 3-Step Smoke

目的：

- 修正上一轮 `raw support + candidate anti-repeat` 的核心问题：target 不再由短 probe 的 hard margin 或 max prompt mass 过强决定，而是用完整 32 条 rollout 得到的 answer support distribution，对每个 chunk candidate 的后续 probe completion 计算平均 support mass。
- 这更贴近 `2504.16084` 的 TTRL 语义：full rollout group 先形成 answer distribution / pseudo-label prior，chunk state 只学习把未来 completion 推向该 distribution 的局部 transition。
- 继续优先 PowerFlow loss；guard 只做 malformed / repeated boxed / prompt copy / OOV 清洗，不再用 `target_guard_use_distribution_score=True` 覆盖 scorer。

代码改动：

```text
新增 ttrl.chunk_state_score_mode=answer_support_mass
新增 ttrl.chunk_state_answer_support_score={mass,gain,relative_gain}
新增 ttrl.chunk_state_answer_support_min_mass
默认关闭，不影响旧实验。
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_support_massavg_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_score_mode=answer_support_mass
ttrl.chunk_state_answer_support_score=mass
ttrl.chunk_state_answer_support_min_mass=0.03125
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_use_distribution_score=False
ttrl.chunk_state_target_guard_candidate_enable=True
```

产物：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_massavg_src3_mid_c128_probe4_b32_r32_v64_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_massavg_src3_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_massavg_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 96
```

3 step 平均：

```text
source_mass_mean = 0.393
probe_support_coverage_mean = 0.509
mass_mean = 0.192
max_mass_mean = 0.258
score_mean = 0.192
label_consistent_ratio = 0.667
improved_state_ratio = 0.854
kept_candidate_ratio = 0.462
distribution_oov_probe_ratio = 0.491
repeated_boxed_probe_ratio = 0.064
candidate_repeated_boxed_probe_ratio = 0.008
actor_batch_powerflow_weight_nonzero_ratio = 0.406
powerflow_weight_max = 2.637
target_entropy = 1.882
actor_grad_norm = 8.148
```

Step 3：

```text
probe_support_coverage_mean = 0.547
score_mean = 0.206
kept_candidate_ratio = 0.508
distribution_oov_probe_ratio = 0.453
powerflow_weight_max = 2.218
actor_grad_norm = 7.897
timing_s/update_actor = 4.935s
```

Timing：

```text
step_total_est = [81.809s, 47.857s, 46.453s]
step2_3_steady_total_est = 47.155s
step2_3_update_actor ≈ 5s
```

结论：

- smoke 健康，可以作为下一轮 20-step gate 候选。
- 相比失败的 `support_antirepeat_src20`，这版 target 明显更温和：`powerflow_weight_max` 约 2.2-3.5，而失败 gate step19/20 常见 7.x；这降低了少数短 probe 命中被过度蒸馏的风险。
- OOV 仍偏高，约 49%，但不再通过 max mass 把单个 probe 命中放大成主监督；这更像 distribution matching。
- 下一步建议先跑 20-step gate，final val 目标至少恢复到 MV 20-step 附近；如果仍低，再试 `chunk_state_answer_support_score=relative_gain` 或提高 `probe_samples`，但不要回到 max-mass target。

## 2026-08-01 support-mass 20-step gate

目的：

- 验证 `answer_support_mass` target 是否能把 3-step smoke 的温和信号扩展到 20-step。
- 配置继续遵守 no-GT target、no dynamic batch、8x B200、PowerFlow chunk actor update。
- 该 gate 用 full rollout 的 answer support mass 作为 chunk probe continuation 的软 target，避免 raw max support 对单个短 probe 过尖。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_support_massavg_src20_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
ttrl.chunk_state_score_mode=answer_support_mass
ttrl.chunk_state_answer_support_score=mass
ttrl.chunk_state_answer_support_min_mass=0.03125
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_min_answer_mass=0.03125
ttrl.chunk_state_target_guard_use_distribution_score=False
ttrl.chunk_state_target_guard_use_mass_gain=False
ttrl.chunk_state_target_guard_candidate_enable=True
ttrl.chunk_state_target_guard_candidate_max_boxed_count=1
ttrl.chunk_state_target_guard_candidate_assistant_marker=True
actor_rollout_ref.actor.use_dynamic_bsz=False
actor_rollout_ref.actor.powerflow_enable=True
actor_rollout_ref.actor.powerflow_use_chunk_weights=True
```

产物：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_massavg_src20_mid_c128_probe4_b32_r32_v64_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_massavg_src20_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_massavg_src20_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 640
```

Final validation:

```text
val-core/math/acc/mean@16 = 0.453625
val-core/math/acc/maj@16 = 0.586292
val-core/math/acc/best@16 = 0.850910
val-aux/math/format_score/mean@16 = 0.899375
val-aux/math/format_score/maj@16 = 0.881332
val-aux/math/format_score/best@16 = 0.999992
```

20 step 平均诊断：

```text
timing_s/gen = 26.296s
timing_s/chunk_state_probe = 6.635s
timing_s/chunk_state_score = 11.777s
timing_s/chunk_state_ref = 1.911s
timing_s/update_actor = 5.269s
distribution_oov_probe_ratio = 0.518
probe_support_coverage_mean = 0.482
positive_ratio = 0.161
kept_state_ratio = 0.386
powerflow_weight_max = 3.122
```

Step 20:

```text
probe_support_coverage_mean = 0.488
score_mean = 0.223
kept_candidate_ratio = 0.453
distribution_oov_probe_ratio = 0.512
powerflow_weight_max = 2.617
timing_s/gen = 21.519s
timing_s/chunk_state_probe = 6.837s
timing_s/chunk_state_score = 18.715s
timing_s/update_actor = 5.382s
```

对比：

```text
MV/TTRL 20-step target baseline: mean@16 ≈ 0.735, maj@16 ≈ 0.818, best@16 ≈ 0.913
support_antirepeat_src20: mean@16 = 0.525125, maj@16 = 0.657004, best@16 = 0.870802
support_massavg_src20: mean@16 = 0.453625, maj@16 = 0.586292, best@16 = 0.850910
```

结论：

- gate 失败，不扩 80 step。`answer_support_mass` 比 anti-repeat 版本更温和，但 final acc 更低，说明单纯用 full rollout answer mass 做 chunk probe distribution matching 不能稳定迁移到局部 state。
- actor update 本身没有变慢，平均约 5.27s；额外开销主要来自 `chunk_state_score` 和 probe generation。当前链路不是 actor update bound。
- 主要训练信号问题是 support 覆盖不足：OOV probe 平均 0.518，positive ratio 平均 0.161，只有约 38.6% states 被保留；大量 probe continuation 落在原 full rollout answer support 之外，PowerFlow loss 在很稀的局部监督上更新，容易把 policy 推向格式尚可但答案错的区域。
- validation 样例中仍能看到重复 instruction、`\boxed{}` 空壳和 `asy` 片段串扰；format mean 仍有 0.899，但 acc 已经退化，说明 guard 没有解决语义级 drift。

下一步：

- 停止沿 `answer_support_mass=mass` 扩展。
- 若继续 chunk-level route，需要重新定义 target：不能只看 answer mass support，需要让 chunk target 和完整解答语义绑定得更强。
- 可选方向一：从 full 32 rollout 选高置信 majority-consistent source，然后只在 source 轨迹的中后段截取 state，target 使用 source answer 的 conditional likelihood / rank，而不是 probe answer 是否落入 global support。
- 可选方向二：参考 chunked search inference 的 beam/prune 语义，用同一 state 下的 continuation tree 做 listwise preference，先保证 chunk choice 能预测 full-answer correctness，再接 PowerFlow distribution matching。
- 可选方向三：先做离线诊断集，统计 chunk boundary、probe horizon、OOV、source correctness 到 final acc 的相关性，避免继续用 20-step training gate 盲试 target。

## 2026-08-01 Source-Answer Consistency 3-Step Smoke

目的：

- 回应上一节失败结论中的“source answer conditional target”方向：不再奖励任意 full-group support answer，而是把每个 chunk state 绑定到被选中的 majority-consistent source rollout 的最终 answer。
- 对 `state + next_chunk + probe` 抽取 final answer，并只在它与该 state 的 `chunk_state_source_answer` 等价时给分；source answer 来自同组 full rollout，不使用 GT 选择 target。
- 继续使用 PowerFlow chunk actor update、source chunk candidate 注入、candidate/target guard 和固定 batch；不启用 actor dynamic batch。

代码改动：

```text
新增 _score_chunk_state_answer_source_consistency
新增 ttrl.chunk_state_score_mode=answer_source_consistency 分支
该 mode 自动触发 full-rollout answer metadata 计算
新增 chunk_state_score/mode_answer_source_consistency 指标
默认关闭，不影响既有实验
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_source_consistency_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_score_mode=answer_source_consistency
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_min_answer_mass=0.03125
ttrl.chunk_state_target_guard_use_distribution_score=False
ttrl.chunk_state_target_guard_use_mass_gain=False
ttrl.chunk_state_target_guard_candidate_enable=True
ttrl.chunk_state_target_guard_candidate_max_boxed_count=1
ttrl.chunk_state_target_guard_candidate_assistant_marker=True
```

产物：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_source_consistency_src3_mid_c128_probe4_b32_r32_v64_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_source_consistency_src3_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_source_consistency_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 96
```

3 step 平均：

```text
source_answer_mass_mean = 0.396
valid_source_ratio = 0.938
answer_coverage_mean = 0.740
raw_positive_ratio = 0.419
label_consistent_ratio = 0.585
state_positive_ratio = 0.709
kept_candidate_ratio = 0.403
distribution_oov_probe_ratio = 0.524
repeated_boxed_probe_ratio = 0.103
candidate_repeated_boxed_probe_ratio = 0.038
positive_ratio_after_guard = 0.330
kept_state_ratio = 0.521
powerflow_weight_max = 7.787
boxed_reward_weighted_mean = 0.860
actor_powerflow_loss = 0.504
actor_grad_norm = 16.327
timing_s/gen = 32.439
timing_s/chunk_state_probe = 6.682
timing_s/chunk_state_score = 11.351
timing_s/chunk_state_ref = 3.143
timing_s/update_actor = 5.839
```

Step 3：

```text
raw_positive_ratio = 0.394
score_mean_after_guard = 0.304
kept_candidate_ratio = 0.387
distribution_oov_probe_ratio = 0.535
candidate_repeated_boxed_probe_ratio = 0.086
kept_state_ratio = 0.500
powerflow_weight_max = 7.787
actor_powerflow_loss = 0.426
actor_grad_norm = 17.472
timing_s/gen = 22.569
timing_s/chunk_state_score = 12.284
timing_s/update_actor = 5.587
```

结论：

- smoke 工程通过：8x B200、Ray/FSDP/vLLM、source metadata、source chunk injection、PowerFlow update 都正常；`chunk_state_score/mode_answer_source_consistency=1.0`，final validation 按预期跳过。
- 该 target 显著提高了 chunk supervision 密度：raw positive 从 support-mass 的约 0.19 提到约 0.42，guard 后 positive 约 0.33，state_positive_ratio 约 0.71；说明把 state 绑定到 source final answer 能提供更强的局部信号。
- 但不建议原样扩 20 step：`powerflow_weight_max=7.787` 三步固定偏尖，`distribution_oov_probe_ratio` 仍约 0.52，step3 candidate repeated boxed ratio 升到 0.086。按前面多次 gate 经验，这类尖权重 + OOV/重复风险很容易在 20 step 退化。
- 下一步若继续该方向，应先做温度/权重裁剪或 listwise smoothing：例如限制 `powerflow_weight_max`，或把 source-answer consistency 与 full answer mass 混合成 soft target，而不是 hard 0/1 consistency 直接进 PowerFlow。

## 2026-08-01 Future-Support-Gain 3-Step Smoke

背景：

- 用户明确指出当前多条 chunk gate 的主要问题是“chunk-local short probe 决定 target”，而不是 TTRL 原文 `2504.16084` 强调的 group-level label/value estimation。
- 本轮收敛到一个更窄的版本：full rollout group 先形成 prompt-level support distribution；chunk/probe 只估计 future completion 是否相对 source answer mass 更靠近该 support。
- source chunk 保留为 target prior，不再作为 hard teacher floor；target 是 per-state sharpened soft distribution，不是 binary correctness reward。

代码改动：

```text
新增 _score_chunk_state_future_support_gain
新增 ttrl.chunk_state_score_mode=future_support_gain
新增 chunk_state_target_prior，actor target raw_weights 会乘以 prior 后归一化
source chunk candidate 可作为 prior，默认不改变 candidate score / boxed reward
新增 chunk_state_score/mode_future_support_gain
```

score 定义：

```text
prompt support: full 32 rollout answer mass map
future_value_j = mean_probe_support_mass_j + max_mass_coef * max_probe_support_mass_j
baseline = source_answer_mass
score_j = clamp(future_value_j - baseline + gain_slack, 0, 1)
q_j ∝ (score_j + eps)^alpha * prior_j
```

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_future_support_gain_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_score_mode=future_support_gain
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_future_support_min_mass=0.03125
ttrl.chunk_state_future_support_max_mass_coef=0.25
ttrl.chunk_state_future_support_gain_slack=0.125
ttrl.chunk_state_future_support_baseline_scale=1.0
ttrl.chunk_state_future_support_source_prior_weight=2.0
ttrl.chunk_state_skip_all_negative=True
ttrl.chunk_state_min_answer_coverage=0.25
ttrl.chunk_state_label_consistent_only=True
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_candidate_enable=True
```

产物：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_gain_src3_mid_c128_probe4_b32_r32_v64_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_gain_src3_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_gain_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 96
```

3 step 平均：

```text
source_mass_mean = 0.397
prompt_top_mass_mean = 0.398
support_coverage_mean = 0.486
mean_mass_mean = 0.188
max_mass_mean = 0.316
raw_gain_mean = -0.143
score_mean = 0.084
label_consistent_ratio = 0.565
improved_state_ratio = 0.521
kept_candidate_ratio = 0.436
distribution_oov_probe_ratio = 0.514
repeated_boxed_probe_ratio = 0.078
candidate_repeated_boxed_probe_ratio = 0.005
probe_mean_source_original_correct = 0.092
probe_mean_source_original_wrong = 0.015
positive_ratio_after_guard = 0.076
kept_state_ratio = 0.531
target_entropy = 1.816
target_prior_mean = 1.125
weight_max = 0.664
powerflow_weight_max = 5.315
boxed_reward_weighted_mean = 0.190
actor_powerflow_loss = 0.097
actor_grad_norm = 3.743
```

Timing：

```text
timing_s/gen = 38.646   # step1 includes warmup/JIT, step2/3 about 32s
timing_s/chunk_state_probe = 6.756
timing_s/chunk_state_score = 10.943
timing_s/chunk_state_ref = 3.457
timing_s/update_actor = 5.960
```

结论：

- 工程 smoke 通过：3 step 完整，diag 96 行，final validation 按预期跳过，PowerFlow loss 非零，`chunk_state_score/mode_future_support_gain=1.0`。
- 相比 hard `answer_source_consistency`，这版更接近用户要求的 search-improved distribution distillation：score 不再是 probe 是否打中 source answer，而是 future support value 相对 source answer mass 的 soft gain。
- 好现象：source-correct state 的 future score 明显高于 wrong source state，`0.092 vs 0.015`，说明 target 与完整 rollout 质量有方向一致性；candidate repeated boxed 均值只有 `0.005`，比 hard source-consistency 的 `0.038` 稳。
- 风险仍然明显：support coverage 只有 `0.486`，OOV 仍有 `0.514`，raw gain 平均为负；这说明多数 chunk/probe 还没有把 future completion 推向 full support，当前只靠 `gain_slack=0.125` 保留了一些 soft signal。
- 不建议立刻扩 20 step。下一步应提高 state/probe 质量后再 gate：
  - 提高 `chunk_state_min_answer_coverage` 到 0.45 或 0.50，直接跳过低覆盖 state。
  - 降低 `gain_slack` 或改成 per-state top-k smoothing，避免负 gain 也被过多保留。
  - 尝试更长 probe horizon 或少量增加 probe samples，目标是把 OOV 降到 0.40 以下。
  - 保留 source prior，但进一步限制 `powerflow_weight_max`，避免尖权重重复早期退化。

## 2026-08-01 Future-Support-Gain Strict 3-Step Smoke

目的：

- 回应当前最重要的方向纠偏：chunk target 不能继续由 short-probe local correctness 主导，而要由 full rollout group support/value estimation 主导。
- 本轮不新增方法名，只做 strict ablation：提高 coverage/informative gate，降低 gain slack，降低 source prior，验证能否减少 OOV 和尖权重，同时保留非零训练信号。

运行：

```text
RUN_ID=ttrl_chunk_state_powerflow_future_support_gain_strict_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_score_mode=future_support_gain
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_teacher_anchor_enable=False
ttrl.chunk_state_future_support_gain_slack=0.05
ttrl.chunk_state_future_support_source_prior_weight=1.25
ttrl.chunk_state_min_answer_coverage=0.50
ttrl.chunk_state_min_informative_gap=0.02
ttrl.chunk_state_skip_uniform=True
ttrl.chunk_state_skip_all_negative=True
ttrl.chunk_state_label_consistent_only=True
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_candidate_enable=True
```

产物：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_gain_strict_src3_mid_c128_probe4_b32_r32_v64_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_gain_strict_src3_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_gain_strict_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 96
```

3 step 平均：

```text
support_coverage_mean = 0.485
distribution_oov_probe_ratio = 0.515
candidate_repeated_boxed_probe_ratio = 0.0067
kept_state_ratio = 0.417
zeroed_state_ratio = 0.583
powerflow_weight_max = 4.018
actor_powerflow_loss = 0.316
probe_mean_source_original_correct = 0.0497
probe_mean_source_original_wrong = 0.0110
raw_gain_mean = -0.145
score_mean = 0.045
improved_state_ratio = 0.563
state_all_negative_ratio = 0.364
state_mixed_ratio = 0.541
answer_coverage_mean = 0.485
label_consistent_ratio = 0.378
informative_ratio = 0.615
chunk_actor_response_len_mean = 124.125
```

Timing：

```text
timing_s/gen = 35.273
timing_s/chunk_state_probe = 6.702
timing_s/chunk_state_score = 11.169
timing_s/chunk_state_ref = 3.136
timing_s/update_actor = 5.779
```

结论：

- strict smoke 工程通过：3 step 完整结束，diag 96 行，PowerFlow chunk actor update 正常；final validation 按 smoke 配置跳过。
- 相比 baseline future-support-gain，strict gate 确实降低了尖权重：`powerflow_weight_max` 从约 `5.315` 降到约 `4.018`，`kept_state_ratio` 从约 `0.531` 降到约 `0.417`；candidate repeated boxed 仍很低，约 `0.0067`。
- 但这还没有达到可以扩 20 step 的质量门槛：`distribution_oov_probe_ratio` 仍约 `0.515`，`support_coverage_mean` 仍只有 `0.485`，`raw_gain_mean=-0.145`。也就是说，严格门控只是在减少坏样本和尖权重，还没有真正让大多数 candidate 的 future completion 靠近 full group support。
- 一个关键问题是当前 coverage gate 没有把 step-level `support_coverage_mean` 推到 0.50 以上，说明保留逻辑仍会留下低覆盖 state 或依赖 source prior/guard。下一步要在 state selection 层硬跳过低覆盖/OOV-heavy state，而不是只在 target guard 层做 candidate zeroing。
- 方向判断：future-support-gain 是当前最接近用户要求的版本，因为它已经把 full rollout support 作为 label estimation，把 source chunk 作为 prior 而非 hard floor；但下一轮必须改成更强的 high-support state/candidate selection，不能继续用 slack 保存大量负 gain。

下一步建议：

- 实现 per-state hard skip：如果 support coverage 低、top mass 太平、all-negative 或 OOV-heavy，整条 state 不进 actor batch。
- candidate score 从 `mean/max support mass - source_mass + slack` 改为更明确的 future support margin，例如 top-supported answer mass gain 或 KL/transport improvement，减少负 gain 被 slack 变成正 target。
- source prior 继续保留，但只做弱 prior；必要时加 `powerflow_weight_clip`，把 smoke 的 `powerflow_weight_max` 稳定压到 3-4 以下。
- 下一次 gate 仍先跑 3 step，只看四个量：target nonzero ratio、OOV ratio、repeated boxed ratio、source-correct vs wrong future score；这些过线后再跑 20 step。

## 2026-08-01 Future-Support-Gain Hardfilter 3-Step Smoke

目的：

- 把上轮 strict FSG 的“低质量 state 只是 loss weight 置零，但仍进入 ref/logprob/update”的问题改成真实 hard filtering。
- score 从 `slack_gain` 改为 `positive_gain`，不再用 slack 把负 gain 变成正 target。
- 打开 `zero_inconsistent_candidates` 和 `prune_zero_weight_samples`，验证是否能减少无效 actor batch 计算，同时保持 full-rollout support driven 的 target 语义。

代码改动：

```text
新增 ttrl.chunk_state_future_support_score_type:
  - slack_gain: 兼容旧行为
  - positive_gain: score = max(raw_gain, 0)
  - relative_positive_gain: score = max(raw_gain / (1 - source_mass), 0)

新增 ttrl.chunk_state_future_support_min_positive_margin
新增 per-state chunk_state_future_support_keep
新增 ttrl.chunk_state_zero_inconsistent_candidates
新增 ttrl.chunk_state_prune_zero_weight_samples
actor batch prune 后按 8 卡 shard 裁成可均分样本数，避免 DataProto.chunk 断言失败
```

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_gain_hardfilter_src3_mid_c128_probe4_b32_r32_v64_20260801.sh
RUN_ID=ttrl_chunk_state_powerflow_future_support_gain_hardfilter2_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_future_support_score_type=positive_gain
ttrl.chunk_state_future_support_gain_slack=0.0
ttrl.chunk_state_future_support_source_prior_weight=1.15
ttrl.chunk_state_future_support_min_positive_margin=0.001
ttrl.chunk_state_zero_inconsistent_candidates=True
ttrl.chunk_state_prune_zero_weight_samples=True
```

产物：

```text
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_gain_hardfilter2_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 96
raw_log = not tee'd by this wrapper; step metrics below are from live console output and diag_jsonl
```

工程问题与修复：

- 第一次 `hardfilter` run 在 step1 后失败：prune 后 actor batch 样本数为 70，不能被 8 卡均分，触发 `DataProto.chunk` 的 `only support equal chunk` 断言。
- 修复：prune 后只保留 nonzero 样本，并裁到 `n_gpus_per_node * nnodes` 的倍数；如果不足一个 shard，则回退不裁剪。
- `hardfilter2` 完整 3 step 通过，final validation 按 smoke 配置跳过。

3 step 平均：

```text
support_coverage_mean = 0.490
distribution_oov_probe_ratio = 0.510
candidate_repeated_boxed_probe_ratio = 0.005
num_actor_samples = 50.7        # strict/baseline 是 256
pruned_sample_ratio = 0.802
kept_state_ratio = 0.406
future_support_keep_ratio = 0.552
score_mean = 0.025
positive_margin_mean = -0.067
powerflow_weight_max = 5.558
actor_powerflow_loss = 2.657
actor_grad_norm = 27.764
probe_mean_source_original_correct = 0.029
probe_mean_source_original_wrong = 0.005
```

Timing：

```text
timing_s/gen = 35.483
timing_s/chunk_state_probe = 6.452
timing_s/chunk_state_score = 10.598
timing_s/chunk_state_ref = 1.662     # step1 4.187, step2/3 about 0.4
timing_s/update_actor = 1.287        # strict FSG was about 5.8
```

结论：

- 工程上 hard filtering 是有效的：actor samples 从 256 降到约 40-56，`update_actor` 从约 5.8s 降到约 1.3s，`chunk_state_ref` 稳态也从约 1.7-1.9s 降到约 0.4s。
- 语义上仍未达到 20-step gate：OOV 仍约 0.51，support coverage 约 0.49，positive margin 平均仍为负。也就是说，真正的问题仍然是 candidate/probe 没有稳定把 future completion 推向 full group support。
- prune 带来新的风险：target 变得更稀疏，`powerflow_weight_max` 均值约 5.56，step3 到 8.0；actor loss/grad 明显变大。这版不适合直接扩 20 step。
- 下一步应该保留 hard filtering 的工程收益，但必须做权重平滑：加 `powerflow_weight_clip` 或改为“保留低权重样本 + zero inconsistent target”的 soft pruning，把 `powerflow_weight_max` 控制在 3-4，同时继续提高 state/probe 的 support coverage。

## 2026-08-01 Future-Support-Gain Hardfilter + Clip4 3-Step Smoke

目的：

- 在 hardfilter2 已经把 actor batch 从 256 裁到约 40-56、actor update 降到约 1s 的基础上，验证 PowerFlow sample weight clipping 能否控制尖权重。
- 保持训练语义不变：full rollout group support 仍是 label/value estimation 来源；source chunk 仍只是 prior / drift guard；只在 actor batch 的 `powerflow_flat_weights` 上做 clip + optional renorm。
- 该实验只做 3-step gate，不做 validation。

代码改动：

```text
ttrl.chunk_state_powerflow_weight_clip: 0.0
ttrl.chunk_state_powerflow_weight_clip_renorm: true

actor batch 组装后：
  powerflow_weight_before_clip = powerflow_flat_weights.clone()
  if clip > 0:
    powerflow_flat_weights = clamp(max=clip)
    if renorm:
      divide by nonzero mean
      clamp(max=clip) again

新增日志：
  chunk_state/powerflow_weight_clip
  chunk_state/powerflow_weight_clip_renorm
  chunk_state/powerflow_weight_before_clip_max
  chunk_state/powerflow_weight_before_clip_mean
```

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_gain_hardfilter_clip4_src3_mid_c128_probe4_b32_r32_v64_20260801.sh
RUN_ID=ttrl_chunk_state_powerflow_future_support_gain_hardfilter_clip4_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_future_support_score_type=positive_gain
ttrl.chunk_state_future_support_gain_slack=0.0
ttrl.chunk_state_future_support_source_prior_weight=1.15
ttrl.chunk_state_future_support_min_positive_margin=0.001
ttrl.chunk_state_zero_inconsistent_candidates=True
ttrl.chunk_state_prune_zero_weight_samples=True
ttrl.chunk_state_powerflow_weight_clip=4.0
ttrl.chunk_state_powerflow_weight_clip_renorm=True
```

产物：

```text
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_gain_hardfilter_clip4_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 96
raw_log = not tee'd by this wrapper; step timing/weight metrics below are from live console output, diag metrics are from jsonl
```

diag jsonl 统计：

```text
step1: source_original_correct=0.844, coverage=0.470, majority_ratio=0.405, probe_mean=0.0230, all_negative=0.531, mixed=0.438
step2: source_original_correct=0.781, coverage=0.507, majority_ratio=0.376, probe_mean=0.0203, all_negative=0.406, mixed=0.594
step3: source_original_correct=0.719, coverage=0.481, majority_ratio=0.367, probe_mean=0.0269, all_negative=0.469, mixed=0.531

boundary_mean = 624 / 548 / 524
boundary_zero_ratio = 0.000 / 0.031 / 0.031
probe_mean_source_original_correct = 0.0273 / 0.0237 / 0.0374
probe_mean_source_original_wrong = 0.0000 / 0.0081 / 0.0000
```

live console 关键指标：

```text
step1:
  powerflow_weight_before_clip_max = 4.673
  powerflow_weight_before_clip_mean = 1.857
  powerflow_weight_max = 2.174
  actor/powerflow_weight/max = 2.174
  num_actor_samples = 56
  update_actor = 1.501s
  chunk_state_ref = 4.155s   # warmup

step2:
  distribution_oov_probe_ratio = 0.493
  support_coverage_mean = 0.507
  num_actor_samples = 48
  pruned_sample_ratio = 0.812
  powerflow_weight_before_clip_max = 8.000
  powerflow_weight_before_clip_mean = 2.134
  powerflow_weight_max = 1.956
  actor/powerflow_weight/max = 1.956
  actor_powerflow_loss = 1.161
  grad_norm = 15.915
  update_actor = 1.345s
  chunk_state_ref = 0.387s

step3:
  distribution_oov_probe_ratio = 0.519
  support_coverage_mean = 0.481
  num_actor_samples = 40
  pruned_sample_ratio = 0.844
  powerflow_weight_before_clip_max = 8.000
  powerflow_weight_before_clip_mean = 2.107
  powerflow_weight_max = 2.000
  actor/powerflow_weight/max = 2.000
  actor_powerflow_loss = 1.567
  grad_norm = 11.689
  update_actor = 0.975s
  chunk_state_ref = 0.358s
```

对比 hardfilter2：

```text
hardfilter2:
  powerflow_weight_max_avg ~= 5.56, step3 = 8.0
  actor_powerflow_loss_avg ~= 2.66
  grad_norm_avg ~= 27.8
  update_actor_avg ~= 1.29s
  support_coverage ~= 0.49
  OOV ~= 0.51

hardfilter_clip4:
  powerflow_weight_before_clip_max step2/3 = 8.0 / 8.0
  powerflow_weight_max after clip+renorm = 2.17 / 1.96 / 2.00
  actor_powerflow_loss step2/3 = 1.16 / 1.57
  grad_norm step2/3 = 15.9 / 11.7
  update_actor step2/3 = 1.35s / 0.98s
  support_coverage still about 0.48-0.51
  OOV still about 0.49-0.52
```

结论：

- `powerflow_weight_clip=4 + renorm` 工程上有效：hardfilter2 中 step3 到 8.0 的尖权重被压到约 2.0，actor loss/grad 明显变稳，同时保留了 hardfilter 的 actor update 提速。
- 这不是 target-quality fix：`support_coverage` 仍只有约 0.48-0.51，OOV 仍约 0.49-0.52。clip 只是让稀疏 target 不炸，不会让 chunk candidate 更贴近 full rollout support。
- 这版可以作为后续 chunk PowerFlow 的默认稳定器，但不应单独扩 20 step。下一步应该把精力放在 state/candidate 的 label estimation：从 full rollout answer support 中抽更高 coverage 的 state，过滤 OOV-heavy/malformed candidate，并把 score 改成更明确的 future support mass gain / transport improvement，而不是靠 clip 掩盖 target 噪声。

## 2026-08-01 Future-Support-Gain Candidate Filter 3-Step Smoke

目的：

- 验证一个更贴近 full-rollout support 语义的 candidate 级过滤：如果某个 chunk 的 probe 很少落入 prompt-level full rollout answer support，就不让它进入 PowerFlow target。
- 继续复用 `hardfilter + clip4` 的工程稳定器，只新增 `future_support_gain` scorer 内部的 candidate quality gate。
- 这轮仍是 3-step gate，不做 validation。

代码改动：

```text
新增默认关闭配置：
  ttrl.chunk_state_future_support_min_candidate_coverage: 0.0
  ttrl.chunk_state_future_support_min_candidate_mean_mass: 0.0

在 future_support_gain scorer 内：
  candidate_coverage = valid_probe_mass_ratio per candidate
  mean_mass = mean(prompt_answer_support_mass) per candidate
  candidate_quality_ok =
    candidate_coverage >= min_candidate_coverage
    and mean_mass >= min_candidate_mean_mass
  score_matrix *= candidate_quality_ok

新增日志：
  chunk_state_future_support_gain/min_candidate_coverage
  chunk_state_future_support_gain/min_candidate_mean_mass
  chunk_state_future_support_gain/candidate_coverage_mean
  chunk_state_future_support_gain/candidate_quality_keep_ratio
  chunk_state_future_support_gain/score_mean_before_candidate_filter
  chunk_state_future_support_gain/label_consistent_ratio_before_candidate_filter
```

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_gain_candfilter_src3_mid_c128_probe4_b32_r32_v64_20260801.sh
RUN_ID=ttrl_chunk_state_powerflow_future_support_gain_candfilter_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_future_support_min_candidate_coverage=0.25
ttrl.chunk_state_future_support_min_candidate_mean_mass=0.03125
ttrl.chunk_state_powerflow_weight_clip=4.0
ttrl.chunk_state_powerflow_weight_clip_renorm=True
```

产物：

```text
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_gain_candfilter_src3_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_gain_candfilter_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 96
```

3 step 关键指标：

```text
step1:
  support_coverage_mean = 0.470
  candidate_quality_keep_ratio = 0.602
  score_mean_before_candidate_filter = 0.026
  score_mean = 0.026
  label_consistent_before = 0.281
  label_consistent_after = 0.281
  OOV = 0.530
  num_actor_samples = 56
  powerflow_weight_max = 2.174
  update_actor = 1.523s

step2:
  support_coverage_mean = 0.522
  candidate_quality_keep_ratio = 0.656
  score_mean_before_candidate_filter = 0.034
  score_mean = 0.034
  label_consistent_before = 0.289
  label_consistent_after = 0.289
  OOV = 0.478
  num_actor_samples = 56
  powerflow_weight_max = 2.051
  update_actor = 1.283s

step3:
  support_coverage_mean = 0.438
  candidate_quality_keep_ratio = 0.559
  score_mean_before_candidate_filter = 0.018
  score_mean = 0.018
  label_consistent_before = 0.211
  label_consistent_after = 0.211
  OOV = 0.562
  num_actor_samples = 40
  powerflow_weight_max = 1.908
  update_actor = 1.174s
```

diag jsonl 统计：

```text
step1: boundary_mean=624, source_correct=0.844, coverage=0.470, probe_mean=0.0230, all_negative=0.531, mixed=0.438
step2: boundary_mean=564, source_correct=0.750, coverage=0.522, probe_mean=0.0328, all_negative=0.438, mixed=0.562
step3: boundary_mean=532, source_correct=0.719, coverage=0.438, probe_mean=0.0183, all_negative=0.469, mixed=0.531
```

结论：

- 工程上通过：3 step 完整结束，raw log 和 diag 都已落盘；actor update 仍保持约 1.2-1.5s，clip 后 `powerflow_weight_max` 约 1.9-2.2。
- 这是一个负结果：candidate quality gate 保留了约 56%-66% candidate，但 `score_mean` 和 `label_consistent_ratio` 在过滤前后完全一致。说明当前 positive-gain target 本来就只来自 support 内 probe，简单 candidate coverage / mean-mass gate 不会进一步改善 target。
- target 质量瓶颈仍在 state/source/probe 分布：step3 `support_coverage=0.438`、`OOV=0.562`，比 clip4 还差；这版不应扩 20 step。
- 下一步应从 state selection 和 probe horizon 改，而不是继续加 candidate gate：例如只选 prompt full-rollout support coverage 更高、majority mass 更强的 prompt/state；或者增加/拉长 probe，使 future support mass 不再被大量 OOV/empty answer 稀释。

## 2026-08-01 Future-Support-Gain Source Gate 3-Step Smoke

目的：

- 验证用户最新纠正里的一个关键假设：先用 full rollout group support 做 prompt/state label estimation，优先从 prompt top answer mass 和 source answer mass 更高的 majority-consistent rollout 里截中后段 state。
- source chunk 继续只作为 prior / drift guard，不直接决定 target；target 仍由 `future_support_gain + PowerFlow distribution matching` 产生。
- 这轮只做 3-step smoke，不做 validation；目标是看 state/source 支持过滤能不能降低 OOV、提高 support coverage，并保持有效 actor samples。

代码改动：

```text
新增默认关闭配置：
  ttrl.chunk_state_min_prompt_top_mass: 0.0
  ttrl.chunk_state_min_source_answer_mass: 0.0

在 majority-consistent source selection 内：
  prompt_top_mass = max(full_rollout_answer_support)
  source_answer_mass = full_rollout_answer_support[source_answer]
  当 prompt_top_mass 或 source_answer_mass 低于阈值时跳过该 source

新增日志：
  chunk_state_future_support_gain/source_mass_mean
  chunk_state_future_support_gain/prompt_top_mass_mean
  chunk_state_diag/skipped_support_sources
  chunk_state_diag/source_answer_mass_mean/min
  chunk_state_diag/source_prompt_top_mass_mean/min
```

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_gain_sourcegate_src3_mid_c128_probe4_b32_r32_v64_20260801.sh
RUN_ID=ttrl_chunk_state_powerflow_future_support_gain_sourcegate_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_min_prompt_top_mass=0.35
ttrl.chunk_state_min_source_answer_mass=0.35
ttrl.chunk_state_powerflow_weight_clip=4.0
ttrl.chunk_state_powerflow_weight_clip_renorm=True
```

产物：

```text
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_gain_sourcegate_src3_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_gain_sourcegate_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 40
```

3 step 关键指标：

```text
step1:
  selected_original_acc_mean = 1.000
  source_mass_mean = 0.548
  support_coverage_mean = 0.248
  distribution_oov_probe_ratio = 0.752
  skipped_support_sources = 18
  real_state_count = 14
  pad_state_count = 2
  num_actor_samples = 8
  powerflow_weight_max = 1.639
  update_actor = 0.733s

step2:
  selected_original_acc_mean = 0.875
  source_mass_mean = 0.594
  support_coverage_mean = 0.414
  distribution_oov_probe_ratio = 0.586
  skipped_support_sources = 25
  real_state_count = 7
  pad_state_count = 1
  num_actor_samples = 8
  powerflow_weight_max = 1.406
  update_actor = 0.390s

step3:
  selected_original_acc_mean = 1.000
  source_mass_mean = 0.502
  support_coverage_mean = 0.311
  distribution_oov_probe_ratio = 0.689
  skipped_support_sources = 20
  real_state_count = 12
  pad_state_count = 4
  num_actor_samples = 16
  powerflow_weight_max = 1.664
  update_actor = 0.562s
```

diag jsonl 聚合：

```text
rows = 40
source_answer_mass_mean = 0.539, min = 0.360, max = 0.793
source_prompt_top_mass_mean = 0.539, min = 0.360, max = 0.793
loss_weight_mean = 0.825, min = 0.0, max = 1.0
boundary_mean = 790.4, min = 384, max = 1024
```

结论：

- 这是一个负结果，不应扩 20 step。source gate 确实把 selected source 变得更“可信”：`selected_original_acc_mean` 在 step1/3 为 1.0，`source_mass_mean` 约 0.50-0.59，说明 prompt/source support gate 生效了。
- 但它没有解决 chunk target 的核心问题，反而明显削弱训练信号：`support_coverage_mean` 只有 0.25-0.41，`OOV` 高达 0.59-0.75，`num_actor_samples` 只有 8/8/16。actor update 变快主要是因为样本被裁掉了，不是有效 infra 提速。
- 这说明“full rollout support 更强的 source”不等价于“中间 state 更可学习”。强筛 source 会偏向高置信完整轨迹，但这些 state 的 chunk continuation 仍然大量落在 full group support 之外，short probe 仍不足以构造稳定的 search-improved distribution。
- 下一步不要继续加 source mass gate。更合理的方向是：先用 full group 建 prompt-level support，再对 state 本身做可学习性过滤，例如 future support coverage、candidate OOV/malformed、top answer mass margin、source continuation transport/KL improvement；同时考虑更长 horizon 或多阶段 probe，让 score 真正表示“靠近 full-rollout group 认为好的答案分布”的未来质量提升。

## 2026-08-01 Future-Support-Gain State Learnability Gate 3-Step Smoke

目的：

- 直接验证 state 级可学习性过滤，而不是继续筛 source：如果一个 state 的 chunk probes 大量落不回 full-rollout answer support，或者 candidate 间 target 太平，就不进入 actor update。
- 继续基于 `hardfilter + clip4`，只新增默认关闭的 state-level learnability gate。
- 这轮只做 3-step smoke，不做 validation。

代码改动：

```text
新增默认关闭配置：
  ttrl.chunk_state_future_support_min_state_coverage: 0.0
  ttrl.chunk_state_future_support_max_state_oov: 1.0
  ttrl.chunk_state_future_support_min_state_mean_mass: 0.0
  ttrl.chunk_state_future_support_min_state_max_mass: 0.0
  ttrl.chunk_state_future_support_min_state_top_margin: 0.0

future_support_gain 内新增 state 指标：
  state_coverage = valid_probe_mass_ratio over candidates x probe_samples
  state_oov = 1 - state_coverage
  state_mean_mass = mean(candidate mean support mass)
  state_max_mass = max(candidate max support mass)
  state_top_margin = top1(score_matrix) - top2(score_matrix)

future_support_keep = positive_margin_gate AND learnable_state_keep

同时修复 prune 边界：
  之前非零 actor samples 少于 shard_count=8 时不会 prune，导致 256 个几乎全零样本进入 ref/update。
  修复后非零样本不足 8 时重复 top nonzero 补齐到 8 个样本，并记录 chunk_state/prune_padded_to_shards。
```

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_gain_stategate_src3_mid_c128_probe4_b32_r32_v64_20260801.sh
RUN_ID=ttrl_chunk_state_powerflow_future_support_gain_stategate_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_future_support_min_state_coverage=0.50
ttrl.chunk_state_future_support_max_state_oov=0.50
ttrl.chunk_state_future_support_min_state_mean_mass=0.10
ttrl.chunk_state_future_support_min_state_max_mass=0.12
ttrl.chunk_state_future_support_min_state_top_margin=0.02
```

产物：

```text
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_gain_stategate_src3_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_gain_stategate_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 96
```

3 step 关键指标：

```text
step1:
  selected_original_acc_mean = 0.844
  support_coverage_mean = 0.470
  state_oov_mean = 0.530
  state_mean_mass_mean = 0.175
  state_max_mass_mean = 0.273
  state_top_margin_mean = 0.002
  learnable_state_keep_ratio = 0.031
  num_actor_samples = 256
  actor_batch_powerflow_weight_nonzero_ratio = 0.012
  update_actor = 6.823s

step2:
  selected_original_acc_mean = 0.719
  support_coverage_mean = 0.367
  state_oov_mean = 0.633
  state_mean_mass_mean = 0.125
  state_max_mass_mean = 0.208
  state_top_margin_mean = 0.022
  learnable_state_keep_ratio = 0.094
  num_actor_samples = 8
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  update_actor = 0.402s

step3:
  selected_original_acc_mean = 0.844
  support_coverage_mean = 0.489
  state_oov_mean = 0.511
  state_mean_mass_mean = 0.199
  state_max_mass_mean = 0.348
  state_top_margin_mean = 0.008
  learnable_state_keep_ratio = 0.031
  num_actor_samples = 256
  actor_batch_powerflow_weight_nonzero_ratio = 0.004
  update_actor = 5.086s
```

diag jsonl 聚合：

```text
step1:
  answer_coverage_mean = 0.470
  future_support_keep_mean = 0.031
  state_mean_mass = 0.175
  state_max_mass = 0.273
  state_top_margin = 0.0016

step2:
  answer_coverage_mean = 0.367
  future_support_keep_mean = 0.094
  state_mean_mass = 0.125
  state_max_mass = 0.208
  state_top_margin = 0.0221

step3:
  answer_coverage_mean = 0.489
  future_support_keep_mean = 0.031
  state_mean_mass = 0.199
  state_max_mass = 0.348
  state_top_margin = 0.0078
```

结论：

- 这是一个负结果，不应扩 20 step。state gate 能筛出少量“看起来可学习”的 state，但比例只有 3%-9%，训练信号过稀疏。
- `state_top_margin_mean` 在 step1/3 接近 0，说明即使 coverage/mass 看起来不差，candidate 间 target 仍然很平，PowerFlow 分布没有稳定的 search-improved direction。
- step1/3 暴露出一个工程 bug：非零权重样本少于 8 卡时，旧 prune 逻辑不会裁剪，导致 256 个几乎全零样本进入 ref/update，`update_actor` 到 5-7s。已修复为非零样本不足 8 时重复 top nonzero 补齐到 8 个样本，并记录 `chunk_state/prune_padded_to_shards`。
- 下一步不应继续用硬阈值筛 state。更有希望的方向是把 score 从 `mean_mass + max_mass_coef * max_mass - source_mass` 改成更直接的 transport / distribution distance improvement，例如 candidate probe answer distribution 到 prompt full-support distribution 的 KL/TV/Wasserstein-like improvement；或者把 probe horizon 加长，让 top margin 不再接近 0。

## 2026-08-01 Future-Support TV-Improvement 3-Step Smoke

目的：

- 验证一个更接近 search-improved distribution distillation 的 score：不再直接用 raw local answer hit，而是比较 candidate probe answer empirical distribution 和 prompt full-rollout answer support distribution 的 TV distance。
- source chunk 只作为 source answer one-hot baseline 和 target prior，不作为 score floor。
- 这轮仍然是 3-step smoke，不做 validation；目的是判定 TV transport score 是否解决“短 probe 局部命中噪声”问题。

代码改动：

```text
ttrl.chunk_state_future_support_score_type 新增：
  tv_positive_gain
  relative_tv_positive_gain

对每个 state：
  support_dist = prompt full-rollout answer support distribution
  source_tv = TV(one_hot(source_answer), support_dist) = 1 - support_dist[source_answer]

对每个 candidate：
  probe_dist = candidate 后续 probe_samples 条 completion 的 empirical answer distribution
  candidate_tv = TV(probe_dist, support_dist)
  tv_gain = source_tv - candidate_tv
  score = clamp(tv_gain, 0, 1)
```

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_tv_src3_mid_c128_probe4_b32_r32_v64_20260801.sh
RUN_ID=ttrl_chunk_state_powerflow_future_support_tv_src3_mid_c128_probe4_b32_r32_v64_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
ttrl.chunk_state_future_support_score_type=tv_positive_gain
ttrl.chunk_state_future_support_min_positive_margin=0.001
```

产物：

```text
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_tv_src3_mid_c128_probe4_b32_r32_v64_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_tv_src3_mid_c128_probe4_b32_r32_v64_20260801.jsonl
diag_jsonl_rows = 96
```

3 step 关键指标：

```text
step1:
  support_coverage_mean = 0.470
  state_oov_mean = 0.530
  source_tv_mean = 0.595
  candidate_tv_mean = 0.686
  tv_gain_mean = -0.090
  score_mean = 0.020
  score_after_guard = 0.002
  label_consistent_ratio = 0.133
  num_actor_samples = 8
  update_actor = 0.653s
  gen = 51.358s

step2:
  support_coverage_mean = 0.566
  state_oov_mean = 0.434
  source_tv_mean = 0.614
  candidate_tv_mean = 0.646
  tv_gain_mean = -0.032
  score_mean = 0.030
  score_after_guard = 0.010
  label_consistent_ratio = 0.191
  num_actor_samples = 16
  update_actor = 0.603s
  gen = 23.006s

step3:
  support_coverage_mean = 0.481
  state_oov_mean = 0.519
  source_tv_mean = 0.613
  candidate_tv_mean = 0.721
  tv_gain_mean = -0.107
  score_mean = 0.035
  score_after_guard = 0.011
  label_consistent_ratio = 0.164
  num_actor_samples = 8
  update_actor = 0.432s
  gen = 22.379s
```

结论：

- 这是一个负结果，不应扩 20 step。TV scoring 路径工程上跑通了，`NameError: answer_values` 已修复，3 step 均完成并打印指标；退出阶段的 `DataLoader worker killed` 出现在 torch dynamo atexit compile-times dump，主训练已完成并跳过 final validation。
- 核心失败点很清楚：即使 score 改成 TV transport gain，candidate 的短 probe empirical distribution 平均仍然比 source one-hot 更远离 full-rollout support distribution。3 个 step 的 `tv_gain_mean` 都是负数：-0.090、-0.032、-0.107。
- guard 之后训练信号仍然非常稀疏：`score_after_guard` 只有 0.002/0.010/0.011，`num_actor_samples` 只有 8/16/8。update_actor 很快不是正向证据，主要是有效 actor samples 被裁得太少。
- 这进一步支持当前方法学判断：最该放弃的不是 PowerFlow backbone、hardfilter+clip4、或 group-level label estimation，而是“局部短视可判定性”这个训练约束。只要 candidate target 仍主要由短 horizon probe 的 empirical answer distribution 决定，它就仍然是在学习 noisy local-answer reward，而不是 search-improvement reward。

下一步原则：

- full rollout group 先定义 prompt-level answer support / majority / pass / coverage / trajectory value。
- chunk/state target 不再要求在短 probe 局部 answer hit 里判清楚；probe 只作为 proposal 或弱 evidence，不能单独主导 teacher。
- state 选择继续偏中后段、偏高质量 rollout，但 source chunk 只保留为 prior / drift guard。
- 需要设计更长 horizon 或分阶段 future evaluation：让 score 真正回答“这个 local transition 会不会把后续 completion distribution 推向 full group 认为好的答案分布”，而不是“短 probe 是否碰巧抽中 boxed answer”。

## 2026-08-01 support_anchor：放弃短 probe teacher 的 3-step smoke

目的：

- 验证最新方法学修正：chunk target 不再主要由 short-horizon probe 的局部 answer hit / source consistency 定义。
- 用同 prompt 的 full rollout group 先估计 answer support，再把高 support 完整轨迹在当前 boundary 之后的 next chunk 注入为 anchor candidates。
- anchor chunk 的 score 直接来自 full-group answer support mass；probe generation 在该模式下跳过。

实现：

```text
score_mode = support_anchor
每个 prompt 先 full rollout n=32
每个 state 从同 prompt 的 full rollout 中选 support mass >= 1/32 且 boundary 后仍有 token 的轨迹
把最多 4 个 anchor next chunk 注入 candidate 0..3
score(candidate) = 对应完整轨迹 answer 在 full group 中的 support mass
非 anchor / 不一致 candidate 权重置零，再走 PowerFlow weighted actor update
```

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_anchor_mid_c128_b32_r32_v64_3step_20260801.sh
RUN_ID=ttrl_chunk_state_powerflow_support_anchor_mid_c128_b32_r32_v64_3step_20260801
TOTAL_TRAINING_STEPS=3
FINAL_VAL_ENABLE=False
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
ttrl.chunk_state_score_mode=support_anchor
ttrl.chunk_state_boundary_mode=mid
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_support_anchor_count=4
ttrl.chunk_state_support_anchor_min_mass=0.03125
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_label_consistent_only=True
ttrl.chunk_state_zero_inconsistent_candidates=True
ttrl.chunk_state_prune_zero_weight_samples=True
ttrl.chunk_state_powerflow_weight_clip=4.0
actor_rollout_ref.actor.use_dynamic_bsz=False
```

产物：

```text
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_anchor_mid_c128_b32_r32_v64_3step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_anchor_mid_c128_b32_r32_v64_3step_20260801.jsonl
diag_jsonl_rows = 96
```

3 step 关键指标：

```text
step1:
  support_anchor_injected_ratio = 1.000
  support_anchor_state_keep_ratio = 1.000
  skipped_no_anchor_ratio = 0.000
  support_anchor_score_mean = 0.080
  support_anchor_score_max_mean = 0.258
  label_consistent_ratio = 0.500
  answer_coverage_mean = 0.500
  positive_ratio = 0.080
  num_actor_samples = 128
  target_entropy = 1.121
  chunk_state_probe/skipped_for_support_anchor = 1.000
  gen = 51.851s
  chunk_state_chunks = 1.085s
  chunk_state_score = 5.850s
  chunk_state_ref = 4.849s
  update_actor = 3.671s

step2:
  support_anchor_injected_ratio = 1.000
  support_anchor_state_keep_ratio = 1.000
  skipped_no_anchor_ratio = 0.000
  support_anchor_score_mean = 0.084
  support_anchor_score_max_mean = 0.281
  label_consistent_ratio = 0.500
  answer_coverage_mean = 0.500
  positive_ratio = 0.084
  num_actor_samples = 128
  target_entropy = 1.101
  chunk_state_probe/skipped_for_support_anchor = 1.000
  gen = 23.432s
  chunk_state_chunks = 0.975s
  chunk_state_score = 6.170s
  chunk_state_ref = 0.965s
  update_actor = 3.013s

step3:
  support_anchor_injected_ratio = 0.984
  support_anchor_state_keep_ratio = 1.000
  skipped_no_anchor_ratio = 0.000
  support_anchor_score_mean = 0.079
  support_anchor_score_max_mean = 0.230
  label_consistent_ratio = 0.492
  answer_coverage_mean = 0.492
  positive_ratio = 0.079
  num_actor_samples = 120
  target_entropy = 1.188
  chunk_state_probe/skipped_for_support_anchor = 1.000
  gen = 32.391s
  chunk_state_chunks = 1.092s
  chunk_state_score = 5.687s
  chunk_state_ref = 1.032s
  update_actor = 3.242s
```

结论：

- 这次 smoke 完成 3 step，`Final validation skipped`，没有训练异常。
- 关键正向证据是 target 密度恢复：每步 32 个真实 state，`support_anchor_state_keep_ratio=1.0`，`skipped_no_anchor_ratio=0.0`，`num_actor_samples=128/128/120`。相比之前 TV / sourcegate 路线的 8/16 级别 actor samples，这说明“用 full rollout support anchor 定义 teacher”明显更健康。
- `chunk_state_probe/skipped_for_support_anchor=1.0`，日志中没有 `timing_s/chunk_state_probe`，说明这条路径已经真正跳过 short-horizon probe generation。这里 metric 里仍有 `chunk_state_probe/raw_positive_ratio`、diag 里仍有 `probe_scores` 等历史命名，但数值实际来自 support anchor score，不是 probe rollout。
- 训练更新不是主瓶颈：`update_actor` 稳态约 3.0-3.2s；chunk 生成约 1.0s；ref logprob 首步热启动后约 1.0s。当前额外慢点主要是 full rollout generation 和 `chunk_state_score` 里 Python/metadata 侧约 5.7-6.2s 的 support scoring/诊断开销。
- 这条路线比“短 probe 局部命中当 teacher”更符合当前理论叙事：full rollout group 定义好什么叫好的 answer support，chunk actor update 学习把局部 transition 推向这个 support，而不是在太短 horizon 上强行判断局部 answer 是否命中。

下一步：

- 把 `support_anchor` 的 metric 命名从 `probe_*` 历史字段中剥离，避免后续分析误读。
- 优化 `chunk_state_score` 的 Python 侧实现，目标把 support scoring 从约 6s 压到 1s 以内。
- 做 20-step 小跑并带 final val，判断这种 full-group support anchor teacher 是否能在早期指标上超过复现 MV baseline；如果 20-step 正向，再扩 80-step 轨迹。

## 2026-08-01 support_anchor diag-off：3-step timing smoke

目的：

- 确认上面 3-step 中 `timing_s/chunk_state_score ~= 5.7-6.2s` 是否来自 support_anchor scorer 本身。
- 关闭 `ttrl.chunk_state_diag_enable`，保留 support_anchor 训练语义不变。

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_anchor_diagoff_mid_c128_b32_r32_v64_3step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_anchor_diagoff_mid_c128_b32_r32_v64_3step_20260801.log
ttrl.chunk_state_diag_enable = False
ttrl.chunk_state_diag_jsonl = ""
```

结果：

```text
step1:
  gen = 51.334s
  chunk_state_chunks = 1.079s
  chunk_state_score = 0.001s
  chunk_state_ref = 4.920s
  update_actor = 3.666s
  num_actor_samples = 128

step2:
  gen = 33.408s
  chunk_state_chunks = 1.004s
  chunk_state_score = 0.001s
  chunk_state_ref = 0.843s
  update_actor = 2.612s
  num_actor_samples = 128

step3:
  gen = 22.206s
  chunk_state_chunks = 1.022s
  chunk_state_score = 0.001s
  chunk_state_ref = 0.895s
  update_actor = 2.828s
  num_actor_samples = 128
```

结论：

- `support_anchor` scorer 本身不是 6s 开销来源；之前的 `chunk_state_score` 主要是诊断路径在 timer 内重算 full rollout reward / GT reward。
- 关掉 heavy diag 后，chunk 侧新增开销很小：score 约 1ms，chunk generation 约 1s，ref 约 0.9-1.0s，actor update 约 2.6-3.3s。
- 当前 step time 主体仍是 full rollout generation，而不是 chunk actor update。

## 2026-08-01 support_anchor diag-off：20-step + final val

目的：

- 在不使用 short-horizon probe teacher 的新语义下，跑 batch32 / rollout32 / 20 step，拿 MATH-TTT final val。
- 保留 full rollout group support 定义 teacher；anchor 只来自同 prompt full rollout 的高 support completion chunk。

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_anchor_diagoff_mid_c128_b32_r32_v64_20step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_anchor_diagoff_mid_c128_b32_r32_v64_20step_20260801.log
RUN_ID = ttrl_chunk_state_powerflow_support_anchor_diagoff_mid_c128_b32_r32_v64_20step_20260801
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
data.train_batch_size = 32
actor_rollout_ref.rollout.n = 32
actor_rollout_ref.rollout.val_kwargs.n = 16
trainer.total_training_steps = 20
trainer.test_freq = 20
trainer.final_val_enable = True
ttrl.chunk_state_score_mode = support_anchor
ttrl.chunk_state_diag_enable = False
ttrl.chunk_state_boundary_mode = mid
ttrl.chunk_state_chunk_size = 128
ttrl.chunk_state_support_anchor_count = 4
ttrl.chunk_state_support_anchor_min_mass = 0.03125
actor_rollout_ref.actor.use_dynamic_bsz = False
```

补充说明：

- 第一次查看日志时，`validation generation end` 后样本表输出非常大，看起来像没有 final metrics；后续 shell flush 后确认 metrics 已完整落盘。
- 为避免以后 validation table 再污染主日志，20-step launcher 已加 `trainer.log_val_generations=0`。
- 曾尝试启动一个 `novallog` 重跑，但在发现原 run final metrics 已 flush 后立刻中断；该中断 run 不作为实验结果。

final val：

```text
val-core/math/acc/mean@16 = 0.43725
val-core/math/acc/maj@16  = 0.558596
val-core/math/acc/best@16 = 0.83514

val-aux/math/acc/maj@8   = 0.536776
val-aux/math/acc/best@8  = 0.780108
val-aux/math/acc/maj@4   = 0.497388
val-aux/math/acc/best@4  = 0.695802
```

20 step 聚合 timing：

```text
all 20 steps:
  gen mean = 24.949s, min = 21.837s, max = 51.442s
  chunk_state_chunks mean = 1.017s
  chunk_state_score mean = 0.001s
  chunk_state_ref mean = 1.158s
  update_actor mean = 3.088s
  testing final = 301.249s

stable steps 2-19:
  gen mean = 23.650s, min = 21.860s, max = 31.795s
  chunk_state_chunks mean = 1.011s
  chunk_state_ref mean = 0.965s
  update_actor mean = 3.057s
```

target / density：

```text
num_actor_samples = 128.0 on all 20 steps
support_anchor_state_keep_ratio = 1.0
support_anchor_skipped_no_anchor_ratio = 0.0
target_entropy mean = 1.187
support_anchor_score_mean mean = 0.094
positive_ratio mean = 0.094
```

结论：

- infra 侧是正结果：放弃 short probe teacher 后，support_anchor 仍能稳定给满 128 actor samples，score 开销约 1ms，chunk actor update 约 3s。
- 训练效果是负结果：20-step mean@16 只有 0.43725，maj@16 0.558596，明显低于我们已有 MV / PowerFlow 早期轨迹。当前 support_anchor 只把 full-support completion 的局部 chunk 注入并蒸馏，缺少真正的 search-improvement / distribution transport 信号。
- 方法判断：这证明“不要用短 probe 局部命中定义 teacher”方向是可实现且计算友好的，但 naive support-anchor distillation 不足以提升数学准确率。下一版需要让 chunk target 对齐 full rollout group 的未来分布改善，例如 support mass gain / value margin / answer support transport，而不是只学习高 support 完整轨迹里的局部 next chunk。

## 2026-08-01 N64 full-support future-support-gain smoke

目的：

- 验证 B200 大显存下把 full rollout support 从 32 放到 64，是否能直接改善 chunk target 的 support coverage / OOV。
- 保留 hardfilter + clip4 的高效 actor update 设置，只做 2-step target-quality smoke，不做 final val。
- 这是对“full group label estimation 是否因为 N 太小而噪”的排查，不作为正式方法扩展。

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_gain_n64_hardfilter_clip4_mid_c128_probe4_2step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_gain_n64_hardfilter_clip4_mid_c128_probe4_2step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_gain_n64_hardfilter_clip4_mid_c128_probe4_2step_20260801.jsonl
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
data.train_batch_size = 32
actor_rollout_ref.rollout.n = 64
trainer.total_training_steps = 2
trainer.final_val_enable = False
ttrl.chunk_state_score_mode = future_support_gain
ttrl.chunk_state_future_support_score_type = positive_gain
ttrl.chunk_state_chunk_size = 128
ttrl.chunk_state_probe_samples = 4
ttrl.chunk_state_probe_max_tokens = 1024
ttrl.chunk_state_powerflow_weight_clip = 4.0
ttrl.chunk_state_powerflow_weight_clip_renorm = True
actor_rollout_ref.actor.use_dynamic_bsz = False
```

target quality / timing：

```text
step 1:
  support_coverage_mean = 0.399
  state_oov_mean = 0.601
  raw_gain_mean = -0.209
  tv_gain_mean = -0.084
  label_consistent_ratio = 0.223
  num_actor_samples = 48
  gen = 45.274s
  chunk_state_probe = 6.690s
  chunk_state_score = 15.196s
  chunk_state_ref = 7.359s
  update_actor = 10.195s

step 2:
  support_coverage_mean = 0.454
  state_oov_mean = 0.546
  raw_gain_mean = -0.150
  tv_gain_mean = -0.082
  label_consistent_ratio = 0.188
  num_actor_samples = 40
  gen = 23.641s
  chunk_state_probe = 6.627s
  chunk_state_score = 13.843s
  chunk_state_ref = 0.820s
  update_actor = 9.031s
```

结论：

- 负结果。把 full rollout support 扩到 N=64 没有解决 target 质量，coverage 反而只有 0.399 / 0.454，OOV 仍是 0.601 / 0.546；相比之前 N32 hardfilter_clip4 的约 0.48-0.51 coverage / 0.49-0.52 OOV 没有改善。
- 成本明显变重。N64 的 generate / verifier / score 都更贵，actor samples 只有 48 / 40，update_actor 约 9-10s；这不是值得扩展到 20-step 的方向。
- 方法判断：主矛盾不是 full support N 太小，也不是 actor update 太慢，而是“局部短视可判定性”这个约束本身。后续不再要求 chunk target 主要由 short-horizon probe 的局部 answer hit、source consistency 或几条短 probe 的偶然命中定义。
- 下一版应改成：full rollout group 先定义 prompt-level support / majority / pass / coverage / value；chunk candidate 只学习哪个 local transition 会把 future completion distribution 推向这个 support。source chunk 只作为 prior / drift guard，不作为主要 teacher；probe 只能作为长程分布估计的一部分或诊断信号，不能单独决定 target。

## 2026-08-01 support_flow gain smoke

目的：

- 按新的方法约束，彻底跳过 short-horizon probe teacher；chunk target 不再由局部 answer hit 或 source consistency 定义。
- 复用 full rollout group 的 answer support，把同 prompt 的 full-support next chunk 作为候选；source chunk 只作为 prior / drift guard。
- 先用 2-step smoke 验证 TTRL 内 PowerFlow actor path、`support_flow` score path、8xB200 infra 是否能跑通。

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_flow_gain_mid_c128_b32_r32_v64_2step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_gain_mid_c128_b32_r32_v64_2step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_gain_mid_c128_b32_r32_v64_2step_20260801.jsonl
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
data.train_batch_size = 32
ttrl.n_votes_per_prompt = 64
actor_rollout_ref.rollout.n = 32
trainer.total_training_steps = 2
trainer.final_val_enable = False
ttrl.chunk_state_score_mode = support_flow
ttrl.chunk_state_support_flow_score_type = gain
ttrl.chunk_state_support_flow_gain_slack = 0.03125
ttrl.chunk_state_support_flow_baseline_scale = 1.0
ttrl.chunk_state_support_flow_source_prior_weight = 1.0
ttrl.chunk_state_chunk_size = 128
ttrl.chunk_state_candidates = 8
ttrl.chunk_state_support_anchor_count = 4
ttrl.chunk_state_label_consistent_only = True
ttrl.chunk_state_zero_inconsistent_candidates = True
ttrl.chunk_state_prune_zero_weight_samples = True
ttrl.chunk_state_powerflow_weight_clip = 4.0
actor_rollout_ref.actor.powerflow_enable = True
actor_rollout_ref.actor.use_kl_loss = False
actor_rollout_ref.actor.use_dynamic_bsz = False
```

结果：

```text
step 1:
  prompt_original_mean = 0.354
  prompt_original_pass = 0.906
  source_answer_mass_mean = 0.472
  anchor_mass_mean = 0.144
  positive_margin_mean = -0.104
  support_flow_score_mean = 0.007
  support_flow_score_max_mean = 0.021
  raw_positive_ratio = 0.007
  label_consistent_ratio = 0.238
  answer_coverage_mean = 0.492
  num_actor_samples = 56
  target_entropy = 1.329
  gen = 39.576s
  chunk_state_chunks = 1.154s
  chunk_state_score = 4.526s
  chunk_state_ref = 4.672s
  update_actor = 2.364s

step 2:
  prompt_original_mean = 0.301
  prompt_original_pass = 0.812
  source_answer_mass_mean = 0.446
  anchor_mass_mean = 0.148
  positive_margin_mean = -0.078
  support_flow_score_mean = 0.008
  support_flow_score_max_mean = 0.021
  raw_positive_ratio = 0.008
  label_consistent_ratio = 0.258
  answer_coverage_mean = 0.492
  num_actor_samples = 64
  target_entropy = 1.412
  gen = 10.520s
  chunk_state_chunks = 1.013s
  chunk_state_score = 4.510s
  chunk_state_ref = 0.922s
  update_actor = 2.245s
```

结论：

- 工程正结果：`support_flow` 成功绕开 short probe，日志中 `chunk_state_probe/skipped_for_support_flow=1`；PowerFlow actor path 正常，`ref_log_prob` 缺失问题已通过开启 `powerflow_enable` 并关闭 `use_kl_loss` 修掉。
- infra 正结果：chunk actor span 约 117-119 tokens，`update_actor` 只有 2.2-2.4s；这再次说明短 chunk actor update 不是当前主瓶颈。
- 方法负结果：`gain = anchor_mass - source_mass + slack` 太保守。source answer mass 约 0.45-0.47，但 injected anchor 的平均 support mass 只有 0.14-0.15，导致 raw positive ratio 只有 0.7%-0.8%，target 过稀，不能扩展成长训。
- 下一步不回到 short probe。应该把 `support_flow` 从 hard gain 改成 full-support soft target，例如直接用 support mass / relative value distribution 形成 `q_j ∝ exp(alpha * support_mass_j) * prior_j`，并继续跳过 all-negative、low coverage、flat support 的低信息 state。source chunk 只保留为 prior，不再作为主要 teacher 或 hard floor。

## 2026-08-01 support_flow soft_mass smoke

目的：

- 按用户修正后的原则，放弃“局部短视可判定性”：不要求 chunk 在 short-horizon probe / source consistency 层面被判清楚。
- 在 TTRL 内新增 `support_flow_score_type=soft_mass`，直接用 full rollout group 的 answer support mass 形成 PowerFlow soft target。
- 保留 source / anchor 作为候选 prior 和 drift guard，但不再用 `anchor_mass - source_mass` 这种 hard gain floor 定义主要 teacher。

代码改动：

```text
verl/trainer/ppo/ray_trainer.py:
  support_flow score_type 新增 soft_mass / soft_relative_mass
  soft_mass: score = anchor_mass
  soft_relative_mass: score = clamp(anchor_mass / source_mass, 0, 1)
  soft mode 的 keep_state 只要求该 state 有非零 support anchor，不再要求 positive_margin >= threshold

verl/trainer/config/ppo_trainer_ttrl.yaml:
  记录 support_flow 的 hard gain 与 soft_mass 语义差异
```

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_flow_softmass_mid_c128_b32_r32_v64_2step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_softmass_mid_c128_b32_r32_v64_2step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_softmass_mid_c128_b32_r32_v64_2step_20260801.jsonl
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
data.train_batch_size = 32
ttrl.n_votes_per_prompt = 64
actor_rollout_ref.rollout.n = 32
trainer.total_training_steps = 2
trainer.final_val_enable = False
ttrl.chunk_state_score_mode = support_flow
ttrl.chunk_state_support_flow_score_type = soft_mass
ttrl.chunk_state_chunk_size = 128
ttrl.chunk_state_candidates = 8
ttrl.chunk_state_support_anchor_count = 4
ttrl.chunk_state_label_consistent_only = True
ttrl.chunk_state_zero_inconsistent_candidates = True
ttrl.chunk_state_prune_zero_weight_samples = True
ttrl.chunk_state_powerflow_weight_clip = 4.0
actor_rollout_ref.actor.powerflow_enable = True
actor_rollout_ref.actor.use_kl_loss = False
actor_rollout_ref.actor.use_dynamic_bsz = False
```

结果：

```text
step 1:
  prompt_original_mean = 0.354
  prompt_original_pass = 0.906
  source_answer_mass_mean = 0.472
  support_flow_anchor_mass_mean = 0.144
  support_flow_score_mean = 0.144
  support_flow_score_max_mean = 0.368
  support_flow_raw_positive_ratio = 0.144
  label_consistent_ratio = 0.492
  answer_coverage_mean = 0.492
  state_all_negative_ratio = 0.000
  state_mixed_ratio = 1.000
  num_actor_samples = 120
  target_entropy = 1.192
  gen = 39.784s
  chunk_state_chunks = 1.085s
  chunk_state_score = 4.523s
  chunk_state_ref = 5.639s
  update_actor = 4.558s

step 2:
  prompt_original_mean = 0.306
  prompt_original_pass = 0.875
  source_answer_mass_mean = 0.435
  support_flow_anchor_mass_mean = 0.123
  support_flow_score_mean = 0.123
  support_flow_score_max_mean = 0.344
  support_flow_raw_positive_ratio = 0.123
  label_consistent_ratio = 0.492
  answer_coverage_mean = 0.492
  state_all_negative_ratio = 0.000
  state_mixed_ratio = 1.000
  num_actor_samples = 120
  target_entropy = 1.171
  gen = 11.100s
  chunk_state_chunks = 0.931s
  chunk_state_score = 4.652s
  chunk_state_ref = 1.675s
  update_actor = 4.171s
```

结论：

- target-quality 正结果：相比 hard gain smoke 的 `raw_positive_ratio=0.007/0.008`、`num_actor_samples=56/64`，soft_mass 提升到 `raw_positive_ratio=0.144/0.123`、`num_actor_samples=120/120`，并且没有 all-negative state。
- 语义上更贴合当前方法目标：teacher 由 full rollout support mass 定义，不依赖 short-horizon probe 的局部 answer hit，也不把 source answer mass 当 hard floor。
- infra 可接受：第二步 `gen=11.1s`、`chunk_state_score=4.65s`、`update_actor=4.17s`。相比 hard gain 的 update_actor 约 2.2s 更慢一些，是因为保留了更多 actor samples，但仍远低于 full-trajectory actor update。
- 下一步应该跑 20-step pilot 看 acc 轨迹，同时加低信息 state gate 的轻量 ablation：`min_answer_coverage` / `min_prompt_top_mass` / `min_source_answer_mass`，但不要回到 short-probe teacher。

## 2026-08-01 support_flow soft_mass 20-step pilot

目的：

- 按 2-step smoke 的正向信号，把 `support_flow + soft_mass` 扩展到 20-step pilot。
- 保持用户修正后的训练语义：不使用 short-horizon probe 局部命中 / source consistency 作为主要 teacher；chunk target 由 full-rollout group support mass 主导。
- 只观察 pilot 是否能在 20 step 后给出有效 MATH-TTT / math500 validation acc，以及 step time 是否仍处于快链路。

运行配置：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_flow_softmass_mid_c128_b32_r32_v64_20step_20260801.sh
rerun3_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_softmass_mid_c128_b32_r32_v64_20step_20260801_rerun3.log
rerun3_diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_softmass_mid_c128_b32_r32_v64_20step_20260801_rerun3.jsonl
rerun3_val_metrics = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_softmass_mid_c128_b32_r32_v64_20step_20260801_rerun3_val_metrics.json
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
data.train_batch_size = 32
ttrl.n_votes_per_prompt = 64
actor_rollout_ref.rollout.n = 32
actor_rollout_ref.rollout.val_kwargs.n = 16
trainer.total_training_steps = 20
trainer.total_epochs = 2
trainer.test_freq = 20
trainer.final_val_enable = True
ttrl.chunk_state_score_mode = support_flow
ttrl.chunk_state_support_flow_score_type = soft_mass
ttrl.chunk_state_chunk_size = 128
ttrl.chunk_state_candidates = 8
ttrl.chunk_state_support_anchor_count = 4
ttrl.chunk_state_label_consistent_only = True
ttrl.chunk_state_zero_inconsistent_candidates = True
ttrl.chunk_state_prune_zero_weight_samples = True
ttrl.chunk_state_powerflow_weight_clip = 4.0
actor_rollout_ref.actor.powerflow_enable = True
actor_rollout_ref.actor.use_kl_loss = False
actor_rollout_ref.actor.use_dynamic_bsz = False
```

过程问题和修复：

- 第一次 20-step run 和 rerun1 都只跑到 15/20 后自然结束。原因是 MATH-TTT train set 为 500 条，`train_batch_size=32` 时 dataloader size 为 15；如果 `trainer.total_epochs=1`，即使设置 `total_training_steps=20`，训练循环也会在 15 个 batch 后结束。已在 20-step launcher 中固化 `trainer.total_epochs=2`。
- rerun2 进入 final validation 后没有 acc 落盘。新增 `trainer.validation_metric_dump_path`，并在 `_validate` 中打印 `validation reward start/end` 和写出 scalar JSON。该改动只增强观测，不改变训练语义。
- 通过 `mlx worker login -- command` 后台启动会被 worker login 退出清理进程，`nohup/setsid` 都没有保住；最终用一个保持打开的 worker 交互 session 前台运行训练，并用 `tee` 写日志。

rerun3 结果：

```text
训练 scalar step: 20 / 20
final validation: completed
validation_metric_dump_path: 写出成功

steady timing, step >= 2:
  timing_s/gen avg = 10.910s, min = 9.856s, max = 20.265s
  timing_s/chunk_state_chunks avg = 1.029s, min = 0.939s, max = 1.376s
  timing_s/chunk_state_score avg = 4.880s, min = 4.376s, max = 7.037s
  timing_s/chunk_state_ref avg = 1.693s, min = 1.621s, max = 1.779s
  timing_s/update_actor avg = 4.357s, min = 4.173s, max = 4.658s

target diagnostics, step 1-20:
  raw_positive_ratio avg = 0.153, min = 0.107, max = 0.217
  label_consistent_ratio avg = 0.497, min = 0.488, max = 0.500
  answer_coverage_mean avg = 0.497, min = 0.488, max = 0.500
  num_actor_samples avg = 124.0, min = 120, max = 128
  target_entropy avg = 1.151, min = 1.014, max = 1.236
  source_mass_mean avg = 0.484, min = 0.422, max = 0.590

final val @ step 20:
  val-core/math/acc/mean@16 = 0.505375
  val-core/math/acc/maj@16/mean = 0.593922
  val-core/math/acc/best@16/mean = 0.847144
```

结论：

- infra 正结果：这条 chunk PowerFlow 链路的训练 step 是快链路，稳态墙钟约 30s/step；actor update 约 4.36s，不是主瓶颈。B200 NCCL 日志确认 `NVLS multicast support is available`、`isAllDirectP2p 1`，vLLM 配置确认 `attention_config.backend=FLASH_ATTN`。
- validation 观测结论：final validation 的 GPU generation 结束后，耗时主要在 `PrimeRewardManager` 的 rule-based reward 汇总。该路径对 500 prompts x 16 samples = 8000 条输出做 math reward，单条 async timeout 是 300s，畸形 / repeated boxed 输出会制造长尾。rerun3 的 validation reward 最终完成并写 JSON，但 final validation 把总墙钟从训练进度约 10 分钟拉到约 15 分钟。
- 方法负结果：20-step acc 明显不够好，`mean@16=0.5054`、`maj@16=0.5939`，低于原始 MV / PowerFlow 轨迹。说明 `soft_mass` 虽然解决了 hard gain 过稀的问题，但“只从 full-support anchors 形成局部 soft target”仍不足以产生有效 20-step improvement。
- 当前主要矛盾不是 short chunk actor update，也不是 FA/NCCL 这类 infra；主要矛盾仍是 chunk target 质量。`answer_coverage_mean` 约 0.50、`raw_positive_ratio` 约 0.15，说明 support anchor 只提供了温和软分布，缺少真正的 future distribution improvement 信号。
- 下一轮方法应继续放弃“局部短视可判定性”，但不能停在 `anchor support mass`。更合理的最小下一步是：full rollout group 先定义 prompt-level support/value，然后对同一 state 的候选 chunk 做更长 horizon / staged future support gain，score 直接衡量“未来 completion 分布向 full-support good answers 移动了多少”；source chunk 只作为 prior/drift guard，不作为 hard teacher。

下一轮约束更新：

- 不再要求 chunk target 主要由 short-horizon probe 的局部 answer hit、局部 source answer consistency、或“几条短 probe 是否碰巧 boxed 正确”来定义。这类局部短视可判定性是当前最像主矛盾的失败约束。
- 继续保留 PowerFlow-style distribution matching、`hardfilter + clip4`、不用 GT、依赖 group-level label estimation 的原则。不要因为 20-step soft_mass 失败就退回 full-trajectory MV 或 source hard teacher。
- full rollout group 必须先定义 prompt-level answer support、majority answer、coverage、pass/best/value；chunk 训练只学习“哪个局部 transition 会把未来 completion 分布推向这些 full-support good answers”。
- source chunk 只能作为 proposal prior / drift guard / 保守参考，不能作为主要 teacher，也不能把 source answer mass 当 hard score floor。
- score 应优先写成 per-state sharpened distribution，例如 `q_j ∝ exp(alpha * future_support_gain_j) * prior_j`。其中 `future_support_gain_j` 要来自 longer-horizon 或 staged rollout 后的 support mass / value margin / answer-support transport improvement，而不是 raw short-probe correctness。
- 低信息 state 要跳过或降权：all-negative、support coverage 低、OOV 高、top answer mass 太平、candidate malformed/repeated boxed/marker 污染严重。否则 PowerFlow 会稳定地拟合噪声分布。

## 2026-08-01 Future-Support Transport Affinity 2-Step Smoke 设计

目的：

- 继续沿着“放弃局部短视可判定性”的方向，把 chunk target 从局部 answer hit / source consistency 改成 full-rollout support distribution 的 transport affinity。
- 不使用 GT，不把 source answer 当 teacher；source 只保留弱 prior / drift guard。
- score 直接来自 candidate future answer distribution 到 prompt-level full-rollout answer support distribution 的距离，并把 `None` / OOV answer 显式当作 off-support mass 惩罚。

代码改动：

```text
ttrl.chunk_state_future_support_score_type 新增：
  transport_affinity:
    score = 1 - TV_OOV(candidate_future_answer_dist, full_rollout_support_dist)
  transport_positive_gain:
    score = max(TV_OOV(source_onehot, support_dist) - TV_OOV(candidate_dist, support_dist), 0)

新增指标：
  chunk_state_future_support_gain/source_oov_tv_mean
  chunk_state_future_support_gain/candidate_oov_tv_mean
  chunk_state_future_support_gain/transport_affinity_mean
  chunk_state_future_support_gain/transport_gain_mean
```

与旧 `tv_positive_gain` 的区别：

- 旧 TV 只在非空 answer counts 上归一化，`None` / malformed / OOV answer 的惩罚不够直接。
- 新 transport score 以 `probe_samples` 总数为分母，full-support 外的答案和空答案都会进入 OOV mass，因此更贴近“future completion 分布是否靠近 full rollout group 认为好的答案分布”。

运行计划：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_transport_affinity_mid_c128_probe4_long_b32_r32_v64_2step_20260801.sh
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
data.train_batch_size = 32
actor_rollout_ref.rollout.n = 32
trainer.total_training_steps = 2
trainer.final_val_enable = False
ttrl.chunk_state_score_mode = future_support_gain
ttrl.chunk_state_future_support_score_type = transport_affinity
ttrl.chunk_state_probe_samples = 4
ttrl.chunk_state_probe_max_tokens = 2048
ttrl.chunk_state_chunk_size = 128
ttrl.chunk_state_powerflow_weight_clip = 4.0
ttrl.chunk_state_prune_zero_weight_samples = True
actor_rollout_ref.actor.use_dynamic_bsz = False
```

Gate：

- 如果 `candidate_oov_tv_mean` 仍高、`transport_affinity_mean` 太低、`state_top_margin` 仍接近 0，则说明仅换 transport score 不够，需要改 state selection / staged long-horizon probe。
- 如果 target entropy、nonzero ratio、OOV-aware transport 指标明显健康，再扩 20-step；否则只作为负结果记录。

运行结果：

```text
run_id = ttrl_chunk_state_powerflow_future_support_transport_affinity_mid_c128_probe4_long_b32_r32_v64_2step_20260801
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_transport_affinity_mid_c128_probe4_long_b32_r32_v64_2step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_transport_affinity_mid_c128_probe4_long_b32_r32_v64_2step_20260801.jsonl
diag_jsonl_rows = 64
final validation = skipped
```

step 指标：

```text
step1:
  support_coverage_mean = 0.304
  candidate_oov_tv_mean = 0.832
  transport_affinity_mean = 0.168
  transport_gain_mean = -0.304
  state_top_margin_mean = 0.051
  label_consistent_ratio = 0.492
  future_support_keep_ratio = 0.719
  num_actor_samples = 112
  target_entropy = 1.543
  powerflow_weight_max = 2.753
  grad_norm = 19.438
  gen = 39.787s
  chunk_state_probe = 7.134s
  chunk_state_score = 9.137s
  chunk_state_ref = 5.504s
  update_actor = 4.166s

step2:
  support_coverage_mean = 0.429
  candidate_oov_tv_mean = 0.756
  transport_affinity_mean = 0.244
  transport_gain_mean = -0.212
  state_top_margin_mean = 0.049
  label_consistent_ratio = 0.602
  future_support_keep_ratio = 0.844
  num_actor_samples = 152
  target_entropy = 1.567
  powerflow_weight_max = 3.060
  grad_norm = 14.675
  gen = 10.733s
  chunk_state_probe = 8.999s
  chunk_state_score = 6.846s
  chunk_state_ref = 2.025s
  update_actor = 4.963s
```

diag 聚合：

```text
answer_coverage mean = 0.366, min = 0.000, max = 0.969
probe_mean mean = 0.206, min = 0.000, max = 0.804
probe_max mean = 0.312, min = 0.000, max = 0.893
source_answer_mass mean = 0.464, min = 0.091, max = 0.938
source_original_correct mean = 0.734
future_support_keep mean = 0.781
future_support_state_top_margin mean = 0.050
all_negative = 0.203
mixed = 0.500
all_positive = 0.297
```

结论：

- 工程通过：2 step 完整结束，8 卡 B200、`/models/Qwen2.5-Math-7B`、`FLASH_ATTN`、NVLS/P2P 路径均正常；dynamic batch 关闭；final validation 按 smoke 配置跳过。
- 相比 hard positive-gain / source-consistency 这类稀疏目标，`transport_affinity` 明显提高了 target 密度：step2 `label_consistent_ratio=0.602`、`future_support_keep_ratio=0.844`、`num_actor_samples=152`，PowerFlow 权重 clip 后 `powerflow_weight_max` 约 3.06，未出现尖权重失控。
- 但这不是可扩 20-step 的正结果：OOV-aware transport 仍然显示 candidate future distribution 离 full-rollout support 很远，step2 `candidate_oov_tv_mean=0.756`、`transport_affinity_mean=0.244`、`transport_gain_mean=-0.212`。也就是说，大多数 chunk 的后续 completion 仍没有比 source baseline 更接近 full support。
- `state_top_margin_mean` 约 0.05，比之前接近 0 的 state-gate 略好，但区分度仍弱；`answer_coverage` 均值只有 0.366，说明主要问题仍是 state/probe distribution，而不是 loss 或 actor update。
- 下一步不应扩 20 step。更合理的路线是把 transport affinity 作为 target 诊断指标，转向 staged / longer-horizon state selection：先从 full rollout group 选 coverage 高、top-mass margin 高的 prompt/state，再对 candidate 做分阶段 probe 或 beam-style continuation，使 candidate future distribution 真正进入 support 后再做 PowerFlow matching。

## 2026-08-01 Transport Affinity Long-Context Probe 1-Step 设计

目的：

- 修正上一轮 `transport_affinity_mid_c128_probe4_long` 的一个重要观测偏差：launcher 中 `MAX_RESPONSE_LENGTH=1024`，导致 vLLM `max_model_len=2048`，trainer 内部实际 probe tokens 被 `max_model_len - max_prompt_len` 裁到约 1024；所以 `probe_max_tokens=2048` 并没有真正形成 2k+ future horizon。
- 本轮只做 1-step smoke，把 `MAX_RESPONSE_LENGTH=3072`、`max_model_len=4096`、`probe_max_tokens=3072` 打开，验证真实 long-horizon probe 是否能把 candidate future distribution 拉进 full-rollout support。
- 继续使用 `transport_affinity`，不引入 GT，不回到 short-probe local correctness teacher。

运行计划：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_transport_affinity_longctx_c128_probe4_b32_r32_v64_1step_20260801.sh
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
data.train_batch_size = 32
actor_rollout_ref.rollout.n = 32
MAX_RESPONSE_LENGTH = 3072
actor_rollout_ref.rollout.max_model_len = 4096
ROLLOUT_MAX_NUM_BATCHED_TOKENS = 65536
GPU_MEMORY_UTILIZATION = 0.86
trainer.total_training_steps = 1
trainer.final_val_enable = False
ttrl.chunk_state_score_mode = future_support_gain
ttrl.chunk_state_future_support_score_type = transport_affinity
ttrl.chunk_state_probe_samples = 4
ttrl.chunk_state_probe_max_tokens = 3072
ttrl.chunk_state_chunk_size = 128
actor_rollout_ref.actor.use_dynamic_bsz = False
```

Gate：

- 如果真实 long-horizon 后 `support_coverage_mean` 仍低于约 0.50、`candidate_oov_tv_mean` 仍高于约 0.65、`transport_gain_mean` 仍为负，则说明问题不只是 probe horizon，而是 state/candidate distribution 本身没有走向 full support。
- 如果 coverage / transport affinity 明显改善，再考虑同配置跑 2-3 step；否则记录为负结果，不扩 20 step。

结果：

第一次启动失败：

```text
run = ttrl_chunk_state_powerflow_future_support_transport_affinity_longctx_c128_probe4_b32_r32_v64_1step_20260801
status = failed before training
error = validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes
bad path = /tmp/cfsg_transport_affinity_longctx_1step/ray/ray/session_.../sockets/plasma_store
resolution = 将 TTRL_RUNTIME_DIR 缩短为 /tmp/cfsglc1，并改为直接调用 8GPU launcher，避免父脚本重复 Hydra override
```

rerun1 完整跑完 1 step：

```text
run = ttrl_chunk_state_powerflow_future_support_transport_affinity_longctx_c128_probe4_b32_r32_v64_1step_20260801_rerun1
model = /models/Qwen2.5-Math-7B
data = MATH-TTT
batch = 32 prompts x 32 rollout
votes = 64
probe_samples = 4
probe_max_tokens = 3072
max_model_len = 4096
dynamic_bsz = False
final_validation = skipped

support_coverage_mean = 0.479
candidate_coverage_mean = 0.479
state_oov_mean = 0.521
candidate_oov_tv_mean = 0.738
source_oov_tv_mean = 0.579
transport_affinity_mean = 0.262
transport_gain_mean = -0.159
tv_gain_mean = -0.112
state_top_margin_mean = 0.029
label_consistent_ratio = 0.703
future_support_keep_ratio = 0.906
num_actor_samples = 176
target_entropy = 1.653
powerflow_weight_max = 3.150
grad_norm = 33.139

timing_s/gen = 43.381
timing_s/chunk_state_chunks = 1.063
timing_s/chunk_state_probe = 17.195
timing_s/chunk_state_score = 11.363
timing_s/chunk_state_ref = 6.182
timing_s/update_actor = 6.688
```

diag 聚合：

```text
jsonl_rows = 32
answer_coverage mean = 0.479, min = 0.000, max = 0.969
probe_mean mean = 0.262, min = 0.000, max = 0.668
probe_max mean = 0.364, min = 0.000, max = 0.750
source_answer_mass mean = 0.421, min = 0.048, max = 0.808
source_original_correct mean = 0.844
future_support_keep mean = 0.906
future_support_state_top_margin mean = 0.029
boundary mean = 552, min = 0, max = 1024
source_response_len mean = 1257.5, min = 227, max = 3072
state_all_positive_ratio = 0.531
state_all_negative_ratio = 0.094
state_mixed_ratio = 0.375
```

结论：

- 工程链路通过：8x B200、Qwen2.5-Math-7B、MATH-TTT、`FLASH_ATTN`、CUDA graph capture、FlashInfer autotune、NCCL 单机 P2P/CUMEM/NVLS channels 都正常；`use_dynamic_bsz=False`；Ray AF_UNIX path 问题由短 runtime dir 解决。
- 真实 long-horizon probe 没有通过方法 gate。`support_coverage_mean=0.479` 低于 0.50，`candidate_oov_tv_mean=0.738` 高于 0.65，`transport_gain_mean=-0.159` 仍为负。也就是说，单纯把 probe horizon 从约 1k 拉到 3k，并没有让 candidate future distribution 比 source baseline 更接近 full-rollout support。
- 这支持当前核心判断：最该放弃的是“局部短视可判定性”约束，而不是 PowerFlow loss、hardfilter/clip4 或 full-group label-estimation 原则。继续让 chunk target 主要由 short-horizon local answer hit / source consistency / 局部 probe 命中来定义，会把 noisy local-answer reward 当成 search-improvement target。
- 下一步目标应改为 full-rollout group 主导 target：先由完整 32/64 rollout 定义 prompt-level support、majority/pass/coverage/value；state 优先来自高 coverage、高 top-mass margin、majority-consistent 的中后段；candidate score 计算为对该 full support 的 future distribution transport improvement。source chunk 只保留为 prior / drift guard，不再作为主要 teacher 或 hard floor。
- 在 target 质量没有改善前，不扩这个 longctx transport 配置到 20 step。

## 2026-08-01 Transport Support Gain + Full-Support Gate 设计

背景：

- `transport_affinity` 的真实 3k long-horizon probe 证明：单纯拉长 probe 不能解决 target 质量。`support_coverage_mean=0.479`、`candidate_oov_tv_mean=0.738`、`transport_gain_mean=-0.159` 说明 candidate future distribution 仍没有比 source baseline 更靠近 full-rollout support。
- 因此下一步不再强化“局部 probe 命中 / source consistency / 局部短视可判定性”，而是把 full-rollout group support/value 作为主目标，低信息 state 直接跳过。

实现改动：

```text
score_type = transport_support_gain
score = relu(source_oov_tv - candidate_oov_tv) * (1 - candidate_oov_tv)
```

含义：

- `source_oov_tv - candidate_oov_tv` 要求 candidate future distribution 相对 source baseline 对 full support 有净 transport improvement。
- `(1 - candidate_oov_tv)` 是 absolute support affinity，避免“只是比很差的 source 稍好一点、但仍然离 full support 很远”的候选被当成强 teacher。
- source chunk 继续只作为 prior / drift guard：`source_prior_weight=1.05`，不做 hard score floor。

smoke 配置：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_transport_support_gain_gate_c128_probe4_b32_r32_v64_1step_20260801.sh
model = /models/Qwen2.5-Math-7B
data = MATH-TTT
batch = 32 prompts x 32 rollout
votes = 64
probe_samples = 4
probe_max_tokens = 3072
max_model_len = 4096
score_type = transport_support_gain
min_state_coverage = 0.50
max_state_oov = 0.50
min_state_top_margin = 0.02
min_candidate_coverage = 0.25
min_candidate_mean_mass = 0.02
dynamic_bsz = False
final_validation = skipped
```

Gate：

- 这是 target 质量 smoke，不看最终 acc。
- 通过标准不是 actor samples 越多越好，而是 retained state/candidate 的质量要明显高于 longctx affinity：`candidate_oov_tv_mean` 下降，`transport_gain_mean` 不再显著为负，`score_mean` 非零但不过密，`state_keep_ratio` 不能因 gate 全灭。
- 如果 `num_actor_samples` 接近 0 或 `state_keep_ratio` 过低，说明 full-support gate 太硬，但方向仍是 target 估计问题，不回退到局部短 probe teacher。

结果：

```text
run = ttrl_chunk_state_powerflow_future_support_transport_support_gain_gate_c128_probe4_b32_r32_v64_1step_20260801
model = /models/Qwen2.5-Math-7B
data = MATH-TTT
batch = 32 prompts x 32 rollout
votes = 64
probe_samples = 4
probe_max_tokens = 3072
max_model_len = 4096
score_type = transport_support_gain
dynamic_bsz = False
final_validation = skipped

support_coverage_mean = 0.479
candidate_coverage_mean = 0.479
candidate_quality_keep_ratio = 0.645
state_oov_mean = 0.521
candidate_oov_tv_mean = 0.738
source_oov_tv_mean = 0.579
transport_affinity_mean = 0.262
transport_gain_mean = -0.159
score_mean_before_candidate_filter = 0.003
score_mean = 0.003
label_consistent_ratio = 0.082
state_top_margin_mean = 0.012
learnable_state_keep_ratio = 0.156
state_keep_ratio = 0.156
num_actor_samples = 8
pruned_sample_ratio = 0.969
target_entropy = 1.446
powerflow_weight_max = 1.383
grad_norm = 8.922

timing_s/gen = 43.631
timing_s/chunk_state_chunks = 1.055
timing_s/chunk_state_probe = 17.178
timing_s/chunk_state_score = 11.356
timing_s/chunk_state_ref = 3.885
timing_s/update_actor = 0.722
```

diag 聚合：

```text
jsonl_rows = 32
answer_coverage mean = 0.479, min = 0.000, max = 0.969
probe_mean mean = 0.003, min = 0.000, max = 0.022
probe_max mean = 0.016, min = 0.000, max = 0.138
source_answer_mass mean = 0.421, min = 0.048, max = 0.808
source_original_correct mean = 0.844
future_support_keep mean = 0.156
future_support_state_top_margin mean = 0.012
state_all_positive_ratio = 0.000
state_all_negative_ratio = 0.625
state_mixed_ratio = 0.375
```

结论：

- 工程通过，配置确认为 `transport_support_gain`，full-support state/candidate gate 生效，8x B200 训练链路正常。
- 方法 gate 没过。`score_mean=0.003`、`label_consistent_ratio=0.082`、`state_keep_ratio=0.156`、`num_actor_samples=8`，说明“必须相对 source baseline 有正 transport gain”在当前 candidate/probe 分布下过于稀疏。
- 这轮不是简单 gate 太硬：核心信号仍是 `candidate_oov_tv_mean=0.738`、`transport_gain_mean=-0.159`，与 longctx affinity 一致，说明候选未来分布整体没有走向 full support。hard positive-gain 只会把 actor update 压到 0.7s，但训练信号几乎全灭。
- 下一步不应回到 short-horizon local teacher，也不应继续强化 source hard constraint。更合理的是 `full-support gated soft affinity`：先用 state-level full-support coverage/top-mass/margin 过滤低信息 state；保留 candidate 的 absolute support affinity 作为 soft distribution；但不要强制 positive transport gain。目标是先得到非稀疏、由 full support 主导、且 OOV 可控的 target，再考虑 20-step。

## 2026-08-01 Transport Affinity + Full-Support State Gate 设计

背景：

- `transport_support_gain` 验证了 hard positive transport gain 过稀疏：`score_mean=0.003`、`num_actor_samples=8`、`state_keep_ratio=0.156`。这能减少 actor update 时间，但几乎没有训练信号。
- 因此下一轮不再要求 candidate 必须相对 source 有正 gain，而是保留 `transport_affinity = 1 - candidate_oov_tv` 的 soft distribution；full-rollout support 只用于过滤低信息 state/candidate 和定义 affinity。

配置：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_transport_affinity_stategate_c128_probe4_b32_r32_v64_1step_20260801.sh
model = /models/Qwen2.5-Math-7B
data = MATH-TTT
batch = 32 prompts x 32 rollout
votes = 64
probe_samples = 4
probe_max_tokens = 3072
max_model_len = 4096
score_type = transport_affinity
min_state_coverage = 0.50
max_state_oov = 0.50
min_state_top_margin = 0.02
min_candidate_coverage = 0.25
min_candidate_mean_mass = 0.02
source_prior_weight = 1.05
dynamic_bsz = False
final_validation = skipped
```

Gate：

- 目标是介于原始 longctx affinity 和 hard support-gain 之间：`num_actor_samples` 不应低到个位数，`state_keep_ratio` 不应全灭，同时 `candidate_quality_keep_ratio` 和 `state_top_margin` 应体现 full-support gate 的筛选效果。
- 如果仍然只有很少样本，说明当前 probe/candidate distribution 本身不进入 full support，需要改 candidate 生成或 state selection，而不是继续调 loss。

结果：

```text
run = ttrl_chunk_state_powerflow_future_support_transport_affinity_stategate_c128_probe4_b32_r32_v64_1step_20260801
model = /models/Qwen2.5-Math-7B
data = MATH-TTT
batch = 32 prompts x 32 rollout
votes = 64
probe_samples = 4
probe_max_tokens = 3072
max_model_len = 4096
score_type = transport_affinity
dynamic_bsz = False
final_validation = skipped

support_coverage_mean = 0.479
candidate_coverage_mean = 0.479
candidate_quality_keep_ratio = 0.645
state_oov_mean = 0.521
candidate_oov_tv_mean = 0.738
source_oov_tv_mean = 0.579
transport_affinity_mean = 0.262
transport_gain_mean = -0.159
score_mean_before_candidate_filter = 0.262
score_mean = 0.260
label_consistent_ratio = 0.645
state_top_margin_mean = 0.025
learnable_state_keep_ratio = 0.250
state_keep_ratio = 0.250
num_actor_samples = 56
pruned_sample_ratio = 0.781
target_entropy = 1.777
powerflow_weight_max = 1.807
grad_norm = 8.338

timing_s/gen = 43.816
timing_s/chunk_state_chunks = 1.086
timing_s/chunk_state_probe = 17.145
timing_s/chunk_state_score = 11.306
timing_s/chunk_state_ref = 4.595
timing_s/update_actor = 2.641
```

diag 聚合：

```text
jsonl_rows = 32
answer_coverage mean = 0.479, min = 0.000, max = 0.969
probe_mean mean = 0.260, min = 0.000, max = 0.668
probe_max mean = 0.358, min = 0.000, max = 0.750
source_answer_mass mean = 0.421, min = 0.048, max = 0.808
source_original_correct mean = 0.844
future_support_keep mean = 0.250
future_support_state_top_margin mean = 0.025
state_all_positive_ratio = 0.438
state_all_negative_ratio = 0.188
state_mixed_ratio = 0.375
```

结论：

- 这轮比 hard `transport_support_gain` 明显更可训练：`num_actor_samples` 从 8 回升到 56，`score_mean` 从 0.003 回升到 0.260，`state_keep_ratio` 从 0.156 到 0.250，`update_actor=2.641s` 仍然很轻。
- 但它没有解决根因：`candidate_oov_tv_mean=0.738`、`transport_gain_mean=-0.159` 与前两轮一致，说明 candidate future distribution 仍整体不比 source 更接近 full support。这个版本只是把训练信号从“全灭”拉回“可训练”，不是一个应扩 20-step 的正结果。
- 当前最清楚的方向是：loss / gate 已经不是主矛盾，candidate 生成和 state selection 才是。下一步应该让 candidate proposal 更接近 full-rollout support，例如从 full support 内的高质量 rollout 后续 chunk 做 contrastive candidate、或在同一 state 下用 support-conditioned resampling / staged continuation 生成候选，再用 soft affinity 做 PowerFlow matching。
