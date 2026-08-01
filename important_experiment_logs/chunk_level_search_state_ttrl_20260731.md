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
