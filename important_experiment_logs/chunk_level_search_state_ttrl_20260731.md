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

## 2026-08-01 Mass-Ranked Source + State-Gated Affinity 设计

背景：

- `transport_affinity_stategate` 说明 soft target + full-support gate 是目前最稳的诊断基线，但 `candidate_oov_tv_mean=0.738` 和 `transport_gain_mean=-0.159` 没有改善。
- 下一步先改 state selection，而不是继续调 loss：在同一个 prompt 的 full rollout group 内，优先选择 answer mass 更高的 majority-consistent source rollout 来截取 state。这样 source prefix 更靠近 full support 主峰，但仍然不使用 GT。

实现：

```text
新增配置: chunk_state_source_select_by_mass
默认: false
行为: 对候选 source locals 按 source_answer_mass 降序排序，再按原有 source_offset cycling 选择
```

smoke 配置：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_transport_affinity_masssrc_stategate_c128_probe4_b32_r32_v64_1step_20260801.sh
model = /models/Qwen2.5-Math-7B
data = MATH-TTT
batch = 32 prompts x 32 rollout
votes = 64
probe_samples = 4
probe_max_tokens = 3072
score_type = transport_affinity
source_mode = majority_consistent
source_select_by_mass = True
min_source_answer_mass = 0.25
min_state_coverage = 0.50
max_state_oov = 0.50
min_state_top_margin = 0.02
min_candidate_coverage = 0.25
min_candidate_mean_mass = 0.02
dynamic_bsz = False
final_validation = skipped
```

Gate：

- 先看 source 是否真的改善：`chunk_state_diag/source_answer_mass_mean` 应高于上一轮约 0.421，且 `skipped_support_sources` 不应导致 state 不足。
- 再看 downstream：如果更好的 source prefix 有帮助，`candidate_oov_tv_mean` 应下降、`state_keep_ratio` 或 `num_actor_samples` 应不低于 state-gated affinity baseline。
- 如果 source mass 改善但 candidate OOV/transport gain 不动，说明只换 source prefix 还不够，需要真正改 candidate proposal，例如支持集内后续 chunk 对比或 staged continuation。

结果：

```text
run = ttrl_chunk_state_powerflow_future_support_transport_affinity_masssrc_stategate_c128_probe4_b32_r32_v64_1step_20260801
model = /models/Qwen2.5-Math-7B
data = MATH-TTT
batch = 32 prompts x 32 rollout
votes = 64
probe_samples = 4
probe_max_tokens = 3072
score_type = transport_affinity
source_mode = majority_consistent
source_select_by_mass = True
min_source_answer_mass = 0.25
dynamic_bsz = False
final_validation = skipped

source_mass_mean = 0.423
prompt_top_mass_mean = 0.423
support_coverage_mean = 0.564
candidate_coverage_mean = 0.564
candidate_quality_keep_ratio = 0.781
state_oov_mean = 0.436
candidate_oov_tv_mean = 0.704
source_oov_tv_mean = 0.577
transport_affinity_mean = 0.296
transport_gain_mean = -0.127
score_mean = 0.295
label_consistent_ratio = 0.781
state_top_margin_mean = 0.024
learnable_state_keep_ratio = 0.292
state_keep_ratio = 0.292
num_actor_samples = 40
pruned_sample_ratio = 0.792
target_entropy = 1.979
powerflow_weight_max = 1.683
grad_norm = 18.901

timing_s/gen = 43.523
timing_s/chunk_state_chunks = 1.110
timing_s/chunk_state_probe = 15.622
timing_s/chunk_state_score = 10.730
timing_s/chunk_state_ref = 4.463
timing_s/update_actor = 2.020
```

与上一轮 `transport_affinity_stategate` 对比：

```text
source_answer_mass_mean: 0.421 -> 0.423
support_coverage_mean: 0.479 -> 0.564
candidate_oov_tv_mean: 0.738 -> 0.704
transport_affinity_mean: 0.262 -> 0.296
transport_gain_mean: -0.159 -> -0.127
label_consistent_ratio: 0.645 -> 0.781
state_keep_ratio: 0.250 -> 0.292
num_actor_samples: 56 -> 40
update_actor: 2.641s -> 2.020s
```

diag 聚合：

```text
jsonl_rows = 24
real_state_count = 19
pad_state_count = 5
skipped_support_sources = 13
majority_consistent_fallbacks = 13
source_answer_mass mean = 0.423, min = 0.259, max = 0.808
source_original_correct_ratio = 0.917
answer_coverage mean = 0.564
probe_mean_source_original_correct = 0.301
probe_mean_source_original_wrong = 0.229
state_all_positive_ratio = 0.583
state_all_negative_ratio = 0.167
state_mixed_ratio = 0.250
```

结论：

- 这轮有小幅正向：candidate coverage、candidate OOV、transport affinity、label consistency 和 state keep 都比上一轮 soft-affinity state-gate 更好。
- 但 `source_answer_mass_mean` 基本没有提升，且 `skipped_support_sources=13` 触发 fallback，说明单纯把 source candidates 按 answer mass 排序不是主解。它改善了采样分布的一部分，但没有让 source selection 进入一个明显更高质量的 support regime。
- 更关键的是 `transport_gain_mean` 仍为负，`candidate_oov_tv_mean=0.704` 仍偏高。也就是说 candidate future distribution 仍整体没有比 source 更靠近 full-rollout support，当前 target 质量仍不足以扩成 20-step 主实验。
- 接下来不应继续要求 chunk target 主要由 short-horizon probe 的局部命中、局部 source consistency 或更强 source hard gate 判清楚。主方向应改为：先由 full rollout group 定义 prompt-level support/value，再让 candidate proposal 学习把未来分布推向该 support；source chunk 只作为 prior / drift guard。可落地的下一步是做 support-conditioned candidate：在同一 state 下混入 high-support rollout continuation chunk 或 staged continuation，再用 soft transport affinity 做 PowerFlow matching。

## 2026-08-01 Support-Proposal + Transport Affinity 1-step smoke

背景：

- `mass-ranked source` 说明只改 source selection 不够，candidate future distribution 仍然 OOV 高、transport gain 为负。
- 这轮做一个最小 candidate proposal 实验：保留 `future_support_gain + transport_affinity` 作为 target scorer，但在 candidate slots 中注入同 prompt full-rollout support 内的 high-mass continuation chunk。这样不把 anchor mass 直接当 teacher，只让它作为 proposal 进入后续 transport scoring。
- 目的不是回到 `support_anchor/support_flow`，而是验证“更靠近 full support 的 candidate proposal 是否能改善 transport target 质量”。

实现：

```text
新增配置: chunk_state_support_anchor_enable
默认: false
语义: 在 support_anchor/support_flow 以外也允许注入 support anchors；注入只改变 candidate proposal，active scorer 仍定义 target。
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_transport_affinity_supportprop_c128_probe4_b32_r32_v64_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_transport_affinity_supportprop_c128_probe4_b32_r32_v64_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_transport_affinity_supportprop_c128_probe4_b32_r32_v64_1step_20260801.jsonl
```

配置：

```text
score_mode = future_support_gain
score_type = transport_affinity
source_mode = majority_consistent
source_select_by_mass = True
source_chunk_candidate_index = 0
support_anchor_enable = True
support_anchor_count = 4
support_anchor_candidate_start = 1
support_anchor_min_mass = 0.03125
candidates = 8
probe_samples = 4
probe_max_tokens = 3072
dynamic_bsz = False
final_validation = skipped
```

结果：

```text
support_anchor_injected_ratio = 1.000
support_anchor_state_keep_ratio = 1.000
support_anchor_anchor_mass_mean = 0.187
support_anchor_anchor_mass_max = 0.808
support_anchor_anchor_len_mean = 116.052

source_mass_mean = 0.423
support_coverage_mean = 0.456
candidate_coverage_mean = 0.456
candidate_quality_keep_ratio = 0.641
state_oov_mean = 0.544
candidate_oov_tv_mean = 0.761
source_oov_tv_mean = 0.577
transport_affinity_mean = 0.239
transport_gain_mean = -0.184
score_mean = 0.236
label_consistent_ratio = 0.641
state_top_margin_mean = 0.019
state_keep_ratio = 0.125
num_actor_samples = 16
pruned_sample_ratio = 0.917
target_entropy = 1.615
update_actor = 1.079s

timing_s/gen = 43.408
timing_s/chunk_state_chunks = 0.984
timing_s/chunk_state_probe = 15.735
timing_s/chunk_state_score = 11.707
timing_s/chunk_state_ref = 3.955
timing_s/update_actor = 1.079
```

与 `mass-ranked source + state-gated affinity` 对比：

```text
support_coverage_mean: 0.564 -> 0.456
candidate_oov_tv_mean: 0.704 -> 0.761
transport_affinity_mean: 0.296 -> 0.239
transport_gain_mean: -0.127 -> -0.184
label_consistent_ratio: 0.781 -> 0.641
state_keep_ratio: 0.292 -> 0.125
num_actor_samples: 40 -> 16
update_actor: 2.020s -> 1.079s
```

diag 聚合：

```text
jsonl_rows = 24
source_answer_mass mean = 0.423, min = 0.259, max = 0.808
answer_coverage mean = 0.456, min = 0.031, max = 0.781
future_support_state_mean_mass mean = 0.168
future_support_state_max_mass mean = 0.365
future_support_state_top_margin mean = 0.019
probe_mean mean = 0.236
future_support_keep mean = 0.125
```

结论：

- 这是一个明确负结果。support anchor 注入本身成功，`injected_ratio=1.0`，但 candidate future distribution 没有靠近 full support，反而 `candidate_oov_tv_mean` 上升、`transport_gain_mean` 更负、`state_keep_ratio` 和 actor samples 明显下降。
- 直接把同 prompt 高 support rollout 的后续 chunk 拼到当前 state 后面并不等价于一个 state-compatible local action。原因大概率是这些 continuation chunk 依赖各自原始前文，和当前 state prefix 不匹配；它们虽然来自高 support 完整轨迹，但在当前 state 上不是自然下一步。
- 这个结果进一步收窄下一步方向：candidate proposal 不能是无条件 high-support continuation copy。需要做 state-compatible proposal，例如从当前 state 继续做 staged long-horizon resampling、用 high-support trajectory 只提供 answer/value target 而不是直接提供 chunk token，或者先做 prefix alignment / nearest-state matching 再注入 continuation。
- 因此下一步不扩 20-step。应设计 `state-compatible staged proposal`：同一 state 先生成 candidate chunk，再对这些 candidate 做更长 horizon future support estimation；或在 full rollout group 中只选择与当前 state prefix 语义/文本接近的 continuation 作为 anchor。

## 2026-08-01 Wide Current-State Candidate c16 smoke

背景：

- 上一轮 `support proposal` 说明无条件复制 high-support rollout continuation 是负方向，因为它不一定和当前 state prefix 兼容。
- 这轮不改 target/scorer，不复制外部 continuation，只把当前 state 的自采样 candidate 从 8 扩到 16，利用 B200 大显存做更宽的 state-compatible search。
- 目标是检查：更多当前 state 采样是否能提高进入 full-rollout support 的概率，并改善 `candidate_oov_tv` / `transport_gain`。

配置：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_transport_affinity_masssrc_stategate_c128_probe4_b32_r32_v64_c16_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_transport_affinity_masssrc_stategate_c128_probe4_b32_r32_v64_c16_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_transport_affinity_masssrc_stategate_c128_probe4_b32_r32_v64_c16_1step_20260801.jsonl

score_mode = future_support_gain
score_type = transport_affinity
source_mode = majority_consistent
source_select_by_mass = True
source_chunk_enable = True
candidates = 16
probe_samples = 4
probe_max_tokens = 3072
support_anchor_enable = False
dynamic_bsz = False
final_validation = skipped
```

结果：

```text
group_size = 64
source_mass_mean = 0.423
support_coverage_mean = 0.586
candidate_coverage_mean = 0.586
candidate_quality_keep_ratio = 0.773
state_oov_mean = 0.414
candidate_oov_tv_mean = 0.706
source_oov_tv_mean = 0.577
transport_affinity_mean = 0.294
transport_gain_mean = -0.129
score_mean = 0.293
label_consistent_ratio = 0.773
state_top_margin_mean = 0.024
state_keep_ratio = 0.250
num_actor_samples = 40
pruned_sample_ratio = 0.896
target_entropy = 2.654
update_actor = 1.905s

timing_s/gen = 43.488
timing_s/chunk_state_chunks = 1.074
timing_s/chunk_state_probe = 19.782
timing_s/chunk_state_score = 12.804
timing_s/chunk_state_ref = 4.438
timing_s/update_actor = 1.905
```

与 c8 mass-ranked source baseline 对比：

```text
support_coverage_mean: 0.564 -> 0.586
candidate_oov_tv_mean: 0.704 -> 0.706
transport_affinity_mean: 0.296 -> 0.294
transport_gain_mean: -0.127 -> -0.129
label_consistent_ratio: 0.781 -> 0.773
state_keep_ratio: 0.292 -> 0.250
num_actor_samples: 40 -> 40
chunk_state_probe: 15.622s -> 19.782s
update_actor: 2.020s -> 1.905s
```

diag 聚合：

```text
jsonl_rows = 24
source_answer_mass mean = 0.423, min = 0.259, max = 0.808
answer_coverage mean = 0.586, min = 0.000, max = 0.891
future_support_state_mean_mass mean = 0.216
future_support_state_max_mass mean = 0.327
future_support_state_top_margin mean = 0.024
probe_mean mean = 0.293
future_support_keep mean = 0.250
```

结论：

- c16 是轻微正向但不够强：`support_coverage_mean` 和 state-level mass 有改善，说明更宽的当前 state 自采样确实更容易碰到 full support。
- 但核心 transport 指标没有改善：`candidate_oov_tv_mean` 基本不变，`transport_gain_mean` 仍为负，`state_keep_ratio` 还略降。因此单纯把 candidate 数翻倍不是主解。
- infra 侧可以承受：probe 时间从 15.6s 增到 19.8s，actor update 仍约 2s，B200 大显存/算力可以支撑更宽 search；但方法收益不足，不应直接扩 20-step。
- 下一步应做真正 staged proposal：不是只增加 chunk candidate 数，而是对当前 state 的 candidate 做更长 horizon / 多阶段 future support estimation，或者把 probe 预算从每个 candidate 固定 4 条改成先宽后深的两阶段分配，优先把算力给已经接近 full support 的 candidate。

## 2026-08-01 staged future-support estimation smoke

背景：

- 用户明确要求放弃“局部短视可判定性”：chunk target 不应主要由 short-horizon probe 的局部命中 / source consistency 定义。
- 当前目标是保留 PowerFlow-style distribution matching 骨架，但把 probe 预算改成两阶段：先用 full-rollout support transport signal 宽筛 candidate，再只对 top-k candidate 做额外 future probe。
- 这个实验只验证 staged estimator 的工程闭环和 target 质量，不做 validation。

实现：

- 在 `ray_trainer.py` 新增默认关闭的 staged future-support path：
  - 第一阶段：所有 candidate 使用原 `chunk_state_probe_samples=4` probe，并复用 `future_support_gain + transport_affinity` scorer 做 full-support 粗排。
  - 第二阶段：每个 state 取 top2 candidate，再追加 `extra_samples=4` 的 deeper probe。
  - 最终仍调用同一个 full-rollout support transport scorer；source chunk 只作为 prior/drift guard，不作为 hard teacher。
- 新增配置默认关闭：
  - `chunk_state_staged_probe_enable`
  - `chunk_state_staged_probe_topk`
  - `chunk_state_staged_probe_extra_samples`
  - `chunk_state_staged_probe_extra_max_tokens`

### staged v1：zero padding merge 失败证据

```text
run_id = ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_masssrc_stategate_c128_probe4xextra4_top2_b32_r32_v64_1step_20260801
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_masssrc_stategate_c128_probe4xextra4_top2_b32_r32_v64_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_masssrc_stategate_c128_probe4xextra4_top2_b32_r32_v64_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_masssrc_stategate_c128_probe4xextra4_top2_b32_r32_v64_1step_20260801.jsonl
```

关键结果：

```text
base support_coverage_mean = 0.564
base candidate_oov_tv_mean = 0.704
base transport_gain_mean = -0.127
base state_keep_ratio = 0.292

final support_coverage_mean = 0.363
final candidate_oov_tv_mean = 0.750
final transport_gain_mean = -0.173
final state_keep_ratio = 0.042
num_actor_samples = 192
powerflow_weight_mean = 0.000
actor/powerflow_loss = 0.000

chunk_state_probe = 15.620s
chunk_state_staged_probe_extra = 13.318s
update_actor = 6.964s
diag_rows = 24
```

结论：

- 这是一个实现失败，不是方法失败。问题在于非 top-k candidate 的 extra probe slot 用空 response padding 补齐，scorer 把这些空 probe 解读成 OOV answer，直接污染 full-support transport target。
- 这个失败证据很重要：staged estimator 不能用“空 probe”去凑固定形状，否则会人为制造 OOV mass，让 state gate 和 PowerFlow weight 全部坍缩。

### staged v1 fixedmerge：有效更新恢复，但 target 质量未突破

修复：

- 非 top-k candidate 的 extra slot 不再用空 probe padding，而是循环复用该 candidate 的第一阶段 base probes。
- 这样非 top-k candidate 保持第一阶段证据不变，top-k candidate 才获得额外 future evidence。

```text
run_id = ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_masssrc_stategate_c128_probe4xextra4_top2_fixedmerge_b32_r32_v64_1step_20260801
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_masssrc_stategate_c128_probe4xextra4_top2_b32_r32_v64_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_masssrc_stategate_c128_probe4xextra4_top2_fixedmerge_b32_r32_v64_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_masssrc_stategate_c128_probe4xextra4_top2_fixedmerge_b32_r32_v64_1step_20260801.jsonl
```

关键结果：

```text
base support_coverage_mean = 0.564
base candidate_oov_tv_mean = 0.704
base transport_gain_mean = -0.127
base state_keep_ratio = 0.292

final support_coverage_mean = 0.557
final candidate_oov_tv_mean = 0.704
final transport_gain_mean = -0.127
final state_keep_ratio = 0.208
label_consistent_ratio = 0.781
num_actor_samples = 24
pruned_sample_ratio = 0.875
powerflow_weight_mean = 1.000
actor/powerflow_loss = 0.400
actor/grad_norm = 2.960

chunk_state_probe = 15.610s
chunk_state_staged_score = 2.939s
chunk_state_staged_probe_extra = 13.307s
chunk_state_score = 12.116s
chunk_state_ref = 4.213s
update_actor = 1.319s
diag_rows = 24
```

与 c8 masssrc baseline 对比：

```text
support_coverage_mean: 0.564 -> 0.557
candidate_oov_tv_mean: 0.704 -> 0.704
transport_gain_mean: -0.127 -> -0.127
state_keep_ratio: 0.292 -> 0.208
num_actor_samples: 40 -> 24
chunk_state_probe total: 15.622s -> 15.610s + 13.307s extra
update_actor: 2.020s -> 1.319s
```

结论：

- fixedmerge 证明 staged probe 工程闭环可跑通，并且不会再把 PowerFlow update 打成 0。
- 但当前 top2 x extra4 设计没有改善核心 target 质量：`candidate_oov_tv_mean` 和 `transport_gain_mean` 基本与 base 相同，`state_keep_ratio` 还下降。
- 这说明“先用同一批短 probe 的 transport_affinity 选 top-k，再对 top-k 加同长度 extra probe”不足以摆脱 short-horizon target 噪声；它更像是对已有 noisy signal 做重复确认。
- 下一步不应直接跑 20-step。更合理的改法是让第二阶段真正回答 future improvement：
  - 第二阶段 probe horizon / completion policy 要和第一阶段不同，例如更长 continuation 或 fewer but deeper completion。
  - top-k 选择不能只看当前 `transport_affinity`，还应引入 full rollout answer support margin / coverage uncertainty，优先给高不确定但可学习 state 加深。
  - scorer 需要区分 base evidence 与 extra evidence，允许 top-k 的 extra evidence 更新 target，而非简单平均后让 base evidence 稀释。

### staged v1 extra_override：短 probe 主导 target 的反证

动机：

- fixedmerge 会把 base probe 和 extra probe 混在一起，可能让第一阶段 short-horizon evidence 稀释第二阶段信号。
- 因此新增 `chunk_state_staged_probe_merge_mode=extra_override`：被 top-k 选中的 candidate 最终只用 extra probes 作为 scorer evidence，未选中的 candidate 保持 base probes。
- 这个 smoke 只检验“让 top-k 由 extra evidence 主导”是否能改善 target 质量。

```text
run_id = ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_extraoverride_masssrc_stategate_c128_probe4xextra4_top2_b32_r32_v64_1step_20260801
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_extraoverride_masssrc_stategate_c128_probe4xextra4_top2_b32_r32_v64_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_extraoverride_masssrc_stategate_c128_probe4xextra4_top2_b32_r32_v64_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_staged_transport_affinity_extraoverride_masssrc_stategate_c128_probe4xextra4_top2_b32_r32_v64_1step_20260801.jsonl
```

关键结果：

```text
base support_coverage_mean = 0.564
base candidate_oov_tv_mean = 0.704
base transport_gain_mean = -0.127
base state_keep_ratio = 0.292

final support_coverage_mean = 0.551
final candidate_oov_tv_mean = 0.711
final transport_gain_mean = -0.135
final state_top_margin_mean = 0.013
final state_keep_ratio = 0.167
label_consistent_ratio = 0.766
num_actor_samples = 16
pruned_sample_ratio = 0.917
powerflow_weight_mean = 1.000
actor/powerflow_loss = 0.132

chunk_state_probe = 15.793s
chunk_state_staged_probe_extra = 13.237s
chunk_state_score = 9.953s
chunk_state_ref = 3.955s
update_actor = 1.044s
gen = 43.567s
diag_rows = 24
```

与 fixedmerge 对比：

```text
support_coverage_mean: 0.557 -> 0.551
candidate_oov_tv_mean: 0.704 -> 0.711
transport_gain_mean: -0.127 -> -0.135
state_keep_ratio: 0.208 -> 0.167
num_actor_samples: 24 -> 16
update_actor: 1.319s -> 1.044s
```

结论：

- extra_override 没有解决 target 质量问题，反而让 support coverage、OOV、transport gain 和 state_keep_ratio 都变差。
- 它证明 actor update 已经不是主矛盾：在 hardfilter + clip4 + actor span chunk 下，`update_actor` 已经可以压到 1.0s 左右。
- 主矛盾仍然是 target 定义：当前 staged top-k 仍由同一套 short-horizon probe/local answer evidence 起步，后续 extra probe 只是对 noisy local signal 做再确认，不能把 chunk target 变成真正的 search-improvement target。
- 因此后续应明确放弃这个约束：chunk 的好坏不应要求在短 horizon、局部 answer hit、source consistency 这一级被判清楚。

下一步算法约束：

- full rollout group 先定义 prompt-level answer support distribution、majority/pass/coverage 和 state value；chunk target 必须服务于这个 group-level support/value。
- source chunk 只保留为 prior / drift guard，不能作为主要 teacher，也不能把 source consistency 变成硬监督。
- candidate score 要从 raw short-probe correctness 改成相对 full group support 的 future quality improvement，例如 support mass gain、top answer margin、transport/KL improvement 或 longer-horizon pass gain。
- 对低信息 state 直接跳过或降权：all-negative、support coverage 低、OOV 高、top mass 太平、malformed/repeated boxed 或 marker 污染。
- PowerFlow loss 骨架可以保留，但 target 应是 per-state sharpened distribution `q_j ∝ exp(alpha * score_j) * prior_j`，其中 `score_j` 来自 full-rollout support/value improvement，而不是局部短 probe 命中。

## 2026-08-01 state-compatible support proposal smoke

背景：

- 前一轮 `supportprop` 直接把同 prompt 的 high-mass full-rollout continuation 注入 candidate slot，结果 target 质量没有提升；原因是这些 continuation 虽然 answer support 高，但不一定兼容当前 state 的局部推理路径。
- 本轮新增一个默认关闭的 prefix compatibility gate：support anchor 必须来自同 prompt，且在当前 boundary 前的 tail tokens 与 selected source prefix 足够匹配，才允许作为 candidate proposal 注入。
- 注意：这仍然只是 proposal，不是 teacher。active scorer 仍是 `future_support_gain + transport_affinity`，PowerFlow target 仍由 probe 后的 full-support transport score 决定。

实现：

- 新增配置：
  - `chunk_state_support_anchor_prefix_compat_enable`
  - `chunk_state_support_anchor_prefix_compat_tokens`
  - `chunk_state_support_anchor_min_prefix_match`
  - `chunk_state_support_anchor_skip_source`
- 在 `_apply_chunk_state_support_anchors` 中，比较 candidate full rollout 与 selected source rollout 在 `boundary - compat_tokens : boundary` 的 token-level match ratio。
- 默认关闭，兼容已有实验；launcher 显式打开。

### strict compat：tail128 / min match 0.75

```text
run_id = ttrl_chunk_state_powerflow_future_support_transport_affinity_compatprop_c128_probe4_b32_r32_v64_1step_20260801
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_transport_affinity_compatprop_c128_probe4_b32_r32_v64_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_transport_affinity_compatprop_c128_probe4_b32_r32_v64_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_transport_affinity_compatprop_c128_probe4_b32_r32_v64_1step_20260801.jsonl
```

关键结果：

```text
support_anchor injected_ratio = 0.000
support_anchor skipped_no_anchor_ratio = 1.000
support_anchor skipped_by_compat_ratio = 0.409
support_anchor prefix_match_mean = 0.000

support_coverage_mean = 0.564
candidate_oov_tv_mean = 0.704
transport_gain_mean = -0.127
state_keep_ratio = 0.292
num_actor_samples = 40
actor/powerflow_loss = 0.196
chunk_state_probe = 15.551s
update_actor = 2.085s
gen = 43.701s
diag_rows = 24
```

### relaxed compat：tail32 / min match 0.25

```text
run_id = ttrl_chunk_state_powerflow_future_support_transport_affinity_compatprop_relaxed_c128_probe4_b32_r32_v64_1step_20260801
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_transport_affinity_compatprop_relaxed_c128_probe4_b32_r32_v64_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_transport_affinity_compatprop_relaxed_c128_probe4_b32_r32_v64_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_transport_affinity_compatprop_relaxed_c128_probe4_b32_r32_v64_1step_20260801.jsonl
```

关键结果：

```text
support_anchor injected_ratio = 0.000
support_anchor skipped_no_anchor_ratio = 1.000
support_anchor skipped_by_compat_ratio = 0.409
support_anchor prefix_match_mean = 0.000

support_coverage_mean = 0.564
candidate_oov_tv_mean = 0.704
transport_gain_mean = -0.127
state_keep_ratio = 0.292
num_actor_samples = 40
actor/powerflow_loss = 0.248
chunk_state_probe = 15.599s
update_actor = 1.911s
gen = 43.628s
diag_rows = 24
```

结论：

- token-level state-compatible support proposal 在当前独立 rollout group 中基本没有可用交集；即使放松到 tail32 / 0.25，非 source high-support trajectory 也无法通过 prefix compatibility。
- 因为注入率为 0，两个 smoke 的 target 指标完全退回 masssrc baseline：`support_coverage_mean=0.564`、`candidate_oov_tv_mean=0.704`、`transport_gain_mean=-0.127`、`state_keep_ratio=0.292`。
- 这说明“从其它 full rollout 截 high-support continuation”不是一个可用的 next-chunk proposal，除非引入语义级 state matching / edit-distance retrieval / tree-search shared-prefix 机制。
- 继续降低 token threshold 没意义，会退化成前一轮 non-compatible supportprop，把别的推理路径硬塞进当前 state。

下一步：

- 不再继续放松 token-prefix anchor。更合理的最小下一步是同一 source path 的 longer-horizon value estimation：保留当前 state 的 sampled chunk，但 probe 不再用短局部 answer hit 主导，而是更长 horizon / multi-stage rollout 后计算相对 full-support distribution 的 transport/value gain。
- 如果继续做 support proposal，需要先构造真正的 shared-prefix search tree 或语义相似检索，而不是从独立 full rollouts 按相同 token boundary 直接截 continuation。
- 因此当前最稳的后续实验是：`source-path staged long probe`，即对同一 state 的 candidate 先做 chunk，再 rollout 到更接近完整答案的 horizon，用 full-rollout group support/value 计算 `q_j ∝ exp(alpha * future_support_gain_j) * prior_j`。

## 2026-08-01 smoothed full-support transport smoke

背景：

- 用户明确纠偏：不要再要求 chunk target 主要由 short-horizon probe 的局部命中 / source consistency 来定义。
- 本轮实现一个更接近 full-rollout group support/value 的 score 变体：probe 只作为 candidate future distribution 的观测证据，full-rollout answer support 作为 posterior prior 平滑 candidate distribution，再计算 transport affinity / gain。
- 这不是让 source chunk 或短 probe hit 当 teacher。source chunk 只保留为 candidate slot 0 的弱 prior / drift guard，score 仍来自 prompt-level full-rollout support distribution。

实现：

- 新增配置 `chunk_state_future_support_prior_smoothing`，默认 0，保持旧实验兼容。
- 在 `future_support_gain` 中新增：
  - `smoothed_transport_affinity`
  - `smoothed_transport_positive_gain`
  - `smoothed_transport_support_gain`
- smoothed score 对每个 candidate 的 probe answer counts 加入 full-support prior：

```text
posterior(answer) = (count(answer) + smoothing * support_dist(answer)) / (probe_samples + smoothing)
```

然后用 posterior distribution 与 full support distribution 的 OOV-aware TV / transport gain 构造 PowerFlow target。

运行：

```text
run_id = ttrl_chunk_state_powerflow_future_support_smoothed_transport_support_gain_masssrc_stategate_c128_probe8_b32_r32_v64_1step_20260801
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_smoothed_transport_support_gain_masssrc_stategate_c128_probe8_b32_r32_v64_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_smoothed_transport_support_gain_masssrc_stategate_c128_probe8_b32_r32_v64_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_smoothed_transport_support_gain_masssrc_stategate_c128_probe8_b32_r32_v64_1step_20260801.jsonl
```

配置：

```text
score_type = smoothed_transport_support_gain
prior_smoothing = 16.0
probe_samples = 8
probe_max_tokens = 3072
staged_probe_enable = false
candidates = 8
chunk_size = 128
source_mode = majority_consistent
source_chunk_enable = true
source_prior_weight = 1.05
min_state_coverage = 0.50
max_state_oov = 0.50
min_state_top_margin = 0.02
min_candidate_coverage = 0.25
min_candidate_mean_mass = 0.02
```

关键结果：

```text
support_coverage_mean = 0.562
candidate_oov_tv_mean = 0.696
transport_gain_mean = -0.119

smoothed_candidate_oov_tv_mean = 0.232
smoothed_transport_affinity_mean = 0.768
smoothed_transport_gain_mean = 0.345

score_mean = 0.240
label_consistent_ratio = 0.766
state_keep_ratio = 0.167
num_actor_samples = 16
target_entropy = 2.043
actor/powerflow_loss = 0.237

timing_s/gen = 43.509
timing_s/chunk_state_probe = 19.844
timing_s/chunk_state_score = 12.391
timing_s/update_actor = 1.106
diag_rows = 24
```

环境确认：

```text
worker = trial-302172199-trialrun-302172199-worker-0
GPU = 8x NVIDIA B200
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
venv = /mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200
vLLM attention_config.backend = FLASH_ATTN
NCCL = 2.28.9+cuda13.0, NVLS enabled, isAllDirectP2p 1
```

结论：

- 这轮不是算法通过。raw target 质量没有本质改善：`support_coverage_mean=0.562`、`candidate_oov_tv_mean=0.696` 仍接近之前 masssrc baseline。
- smoothed 后的 `candidate_oov_tv` 从 0.696 降到 0.232，`smoothed_transport_gain` 变成 0.345，说明 full-support posterior prior 可以把 target 表面拉向 full support，但这主要来自 smoothing，不代表 sampled chunk 的真实 future support coverage 已经改善。
- `state_keep_ratio=0.167`、`num_actor_samples=16` 仍偏低，和 extra_override 类似，不能升级 20-step。
- `update_actor=1.106s` 再次确认训练更新不是主矛盾。更大的成本在 probe / score：`chunk_state_probe=19.844s`、`chunk_state_score=12.391s`。如果 target 质量不提升，继续扩大 probe 是低性价比。

下一步：

- 保留 smoothed support posterior 作为一个可用的 target construction building block，但不能把它当作通过 gate 的方案。
- 下一轮应改 state/candidate 生成本身，而不是继续在同样 noisy candidate set 上加平滑：
  - 优先选 high-support / high-pass full rollout 中更晚的 state，减少低信息 mid-state。
  - 对同一 source path 做 multi-boundary 或 suffix-aware candidate，不再让独立 short probe 决定 teacher。
  - 引入 state-level skip：all-negative、高 OOV、low coverage、top margin 太平的 state 直接不训。
  - 如果继续用 smoothing，应把 smoothing 强度作为 prior，不允许它掩盖 raw coverage/OOV 的失败。

## 2026-08-01 high-support late-state smoke

背景：

- 上一轮 smoothed posterior 说明 target construction 可以被 full-support prior 拉动，但 raw candidate quality 没改善。
- 本轮先不改代码，只用已有 source / boundary gate 验证一个假设：如果从 high-support source rollout 选更靠后的 state，是否能减少低信息 state，并让 raw future support coverage / transport gain 改善。
- 这是对“改 state/candidate 生成本身”的最小验证，不使用 smoothing。

运行：

```text
run_id = ttrl_chunk_state_powerflow_future_support_highmass_late_transport_support_gain_c128_probe4_b32_r32_v64_1step_20260801
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_highmass_late_transport_support_gain_c128_probe4_b32_r32_v64_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_highmass_late_transport_support_gain_c128_probe4_b32_r32_v64_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_highmass_late_transport_support_gain_c128_probe4_b32_r32_v64_1step_20260801.jsonl
```

配置：

```text
score_type = transport_support_gain
prior_smoothing = 0.0
source_mode = majority_consistent
source_select_by_mass = true
min_prompt_top_mass = 0.40
min_source_answer_mass = 0.50
boundary_mode = mid
mid_boundary_min_ratio = 0.50
mid_boundary_max_ratio = 0.90
probe_samples = 4
candidates = 8
chunk_size = 128
```

关键结果：

```text
source_mass_mean = 0.645
prompt_top_mass_mean = 0.645
source_original_acc_mean = 1.000
source_pseudo_acc_mean = 0.875

real_state_count = 7
pad_state_count = 1
skipped_support_sources = 25
majority_consistent_fallbacks = 9
boundary_mean = 928
boundary_min = 768
boundary_max = 1024

support_coverage_mean = 0.434
candidate_oov_tv_mean = 0.688
transport_gain_mean = -0.333
score_mean = 0.001
label_consistent_ratio = 0.016
state_keep_ratio = 0.000

num_actor_samples = 64
kept_state_ratio = 0.000
actor/powerflow_loss = 0.000
chunk_state_probe = 13.404s
chunk_state_score = 8.393s
update_actor = 2.880s
diag_rows = 8
```

结论：

- 这条路线失败，而且失败原因清楚：high-support source selection 本身有效，`source_mass_mean` 从 0.423 提到 0.645，`source_original_acc_mean=1.0`，但 late boundary 让 candidate 未来 probe 更难回到 full support。
- raw target 质量没有改善，反而更差：`support_coverage_mean=0.434`，低于 masssrc baseline 的约 0.564；`transport_gain_mean=-0.333`，比 baseline 的约 -0.127 更差。
- 因为 source baseline 太强，要求 `transport_support_gain` 正增益导致几乎所有 candidate 被打零：`label_consistent_ratio=0.016`、`state_keep_ratio=0`、`actor/powerflow_loss=0`。这不是可训练信号。
- 因此不能简单做“高 support source + late boundary + positive gain”。这会把训练目标变成“在一个已经很强的后段状态上找到更强 continuation”，实际 sampled candidate 很难超过 source future distribution。

下一步：

- 不再重复 strict highmass-late positive-gain 设置。
- 可以保留 high-support source 作为 state prior，但 boundary 不应推到 0.5-0.9；更合理的是 mid-state 0.25-0.70 或多 boundary mix。
- score 也不应只用 hard positive gain；更适合用 soft support affinity / smoothed posterior / rank target，同时显式报告 raw coverage 和 OOV，防止 smoothing 掩盖失败。
- 如果要继续验证 state selector，应尝试“high-support source + normal/mid boundary + soft transport affinity”，目标是先让 raw support coverage 不低于 baseline，再谈 20-step gate。

## 2026-08-01 high-support mid-state soft-affinity smoke

背景：

- 这轮按最新纠偏执行：不再把 short-horizon probe 的局部命中、source answer consistency 或 positive gain 当作主要 teacher。
- full rollout group support 仍是 score 的参照分布；source chunk 只作为候选 prior / drift guard。
- 相比 highmass-late，把 boundary 拉回 0.25-0.70，source mass gate 降到 0.40，score 改为 soft `transport_affinity`，并显式关闭 `label_consistent_only` / `zero_inconsistent_candidates`。

运行：

```text
run_id = ttrl_chunk_state_powerflow_future_support_highsupport_mid_transport_affinity_c128_probe4_b32_r32_v64_1step_20260801
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_highsupport_mid_transport_affinity_c128_probe4_b32_r32_v64_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_highsupport_mid_transport_affinity_c128_probe4_b32_r32_v64_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_highsupport_mid_transport_affinity_c128_probe4_b32_r32_v64_1step_20260801.jsonl
```

配置：

```text
score_type = transport_affinity
prior_smoothing = 0.0
source_mode = majority_consistent
source_select_by_mass = true
min_prompt_top_mass = 0.35
min_source_answer_mass = 0.40
boundary_mode = mid
mid_boundary_min_ratio = 0.25
mid_boundary_max_ratio = 0.70
probe_samples = 4
candidates = 8
chunk_size = 128
label_consistent_only = false
zero_inconsistent_candidates = false
skip_all_negative = false
min_state_coverage = 0.50
max_state_oov = 0.50
min_state_top_margin = 0.02
```

关键结果：

```text
source_mass_mean = 0.520
prompt_top_mass_mean = 0.520
source_original_acc_mean = 1.000
source_pseudo_acc_mean = 0.938

real_state_count = 11
pad_state_count = 5
skipped_support_sources = 21
majority_consistent_fallbacks = 11
boundary_mean = 592
boundary_min = 256
boundary_max = 1024

support_coverage_mean = 0.598
candidate_oov_tv_mean = 0.662
transport_affinity_mean = 0.338
transport_gain_mean = -0.182
state_top_margin_mean = 0.021
learnable_state_keep_ratio = 0.125
state_keep_ratio = 0.125

num_actor_samples = 8
zeroed_state_ratio = 0.938
target_entropy = 1.894
actor/powerflow_loss = 2.063
chunk_state_probe = 14.235s
chunk_state_score = 9.499s
chunk_state_ref = 3.904s
update_actor = 0.737s
diag_rows = 16
```

结论：

- 这轮比 highmass-late 健康：raw `support_coverage_mean=0.598`，高于 masssrc baseline 的约 0.564，也明显高于 highmass-late 的 0.434。把 boundary 拉回 mid，并放弃 hard positive-gain collapse 是正确方向。
- 但这轮仍不能升级 20-step：`candidate_oov_tv_mean=0.662` 仍高，`transport_gain_mean=-0.182` 仍为负，说明候选未来分布平均没有比 source 更接近 full-rollout support。
- 主要失败点从“无正样本”变成“可训练 state 太少”：`learnable_state_keep_ratio=0.125`、`num_actor_samples=8`、`zeroed_state_ratio=0.938`。只有 2/16 个 state 通过 gate，最后每卡约 1 个样本，训练信号太稀。
- 这轮也进一步支持最新判断：训练更新不是瓶颈，`update_actor=0.737s` 很快；主矛盾仍是 target/state/candidate 质量，不能继续靠短 probe 局部命中或 source 硬约束加码。

下一步：

- 保留 `transport_affinity` 作为 full-support soft target 方向，但需要改 candidate/state 生成，而不是继续调 PowerFlow 更新速度。
- 更合理的下一版应让 full rollout group support 更直接地参与 label estimation：
  - 从每个 prompt 的完整 rollout group 先构造 answer support / value support。
  - state 选择优先来自 support 内高质量 source，但不要只选 late boundary。
  - candidate target 使用 per-state softened full-support distribution，source chunk 只作为 prior，不作为 hard teacher。
  - 对低信息 state 直接跳过：all-negative、高 OOV、support coverage 低、top margin 太平。
- 下一轮 smoke gate 不看 smoothed 指标单独变好，仍以 raw `support_coverage_mean`、`candidate_oov_tv_mean`、`transport_gain_mean`、`state_keep_ratio`、`num_actor_samples` 为准。

## 2026-08-01 high-support mid-state soft-affinity c16 softgate smoke

背景：

- 上一轮 `highsupport_mid_transport_affinity` 的 raw coverage 有改善，但 state gate 只留下 2/16 个 state，最终只有 8 个 actor samples。
- 本轮只验证一个工程/统计假设：在不回到 local-hit teacher 的前提下，把候选宽度从 8 扩到 16，并去掉 top-margin hard gate，是否能解决可训练样本太少的问题。
- 仍使用 full-rollout support 的 `transport_affinity` 作为 soft target，不开启 `label_consistent_only` / `zero_inconsistent_candidates`。

运行：

```text
run_id = ttrl_chunk_state_powerflow_future_support_highsupport_mid_transport_affinity_c16_softgate_c128_probe4_b32_r32_v64_1step_20260801
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_future_support_highsupport_mid_transport_affinity_c16_softgate_c128_probe4_b32_r32_v64_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_future_support_highsupport_mid_transport_affinity_c16_softgate_c128_probe4_b32_r32_v64_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_future_support_highsupport_mid_transport_affinity_c16_softgate_c128_probe4_b32_r32_v64_1step_20260801.jsonl
```

配置：

```text
score_type = transport_affinity
prior_smoothing = 0.0
source_mode = majority_consistent
source_select_by_mass = true
min_prompt_top_mass = 0.35
min_source_answer_mass = 0.40
boundary_mode = mid
mid_boundary_min_ratio = 0.25
mid_boundary_max_ratio = 0.70
probe_samples = 4
candidates = 16
chunk_size = 128
label_consistent_only = false
zero_inconsistent_candidates = false
skip_all_negative = false
min_state_coverage = 0.50
max_state_oov = 0.50
min_state_top_margin = 0.0
```

关键结果：

```text
source_mass_mean = 0.520
prompt_top_mass_mean = 0.520
source_original_acc_mean = 1.000
source_pseudo_acc_mean = 0.938

real_state_count = 11
pad_state_count = 5
skipped_support_sources = 21
majority_consistent_fallbacks = 11
boundary_mean = 592
boundary_min = 256
boundary_max = 1024

support_coverage_mean = 0.581
candidate_oov_tv_mean = 0.659
transport_affinity_mean = 0.341
transport_gain_mean = -0.179
state_top_margin_mean = 0.005
learnable_state_keep_ratio = 0.750
state_keep_ratio = 0.750

num_actor_samples = 112
zeroed_state_ratio = 0.562
target_entropy = 2.652
actor/powerflow_loss = 0.417
chunk_state_probe = 16.534s
chunk_state_score = 12.078s
chunk_state_ref = 5.464s
update_actor = 4.475s
diag_rows = 16
```

结论：

- c16 + softgate 解决了上一轮最直接的样本稀疏问题：`state_keep_ratio` 从 0.125 提到 0.750，`num_actor_samples` 从 8 提到 112。
- 但它没有解决核心 target 质量问题：`support_coverage_mean=0.581` 低于上一轮 0.598，`candidate_oov_tv_mean=0.659` 仍很高，`transport_gain_mean=-0.179` 仍为负。
- 所以单纯扩大 candidate 宽度、放松 gate 只能让 PowerFlow 有东西可训，不能证明这些 chunk transition 是 search improvement。
- 成本也开始上升：`update_actor=4.475s` 可接受，但 `probe+score+ref` 已约 34s；如果 target 质量不改善，继续加宽度不是高性价比方向。

下一步：

- 不再把主要精力放在 c16/c32 或继续调 hard gate；这些只能改善样本数，不能改变 target 是否正确。
- 下一步应改 label estimation 机制：让 full rollout group support 直接生成 per-state target distribution，例如从完整 rollout 的 answer support / future suffix support 中构造 target，再把 source chunk 只作为 prior。
- 候选生成也需要更贴近 support：可以从 high-support rollout 的真实 suffix chunk 做 replay/proposal，或混合 on-policy resample 与 support-suffix proposal，而不是完全依赖 state-local short probe 判断候选好坏。

## 2026-08-01 support-flow suffix soft-mass smoke

背景：

- 这轮是对前面失败结论的直接修正：放弃“局部短视可判定性”，不再要求 chunk target 主要由 short-horizon probe 的局部命中、source answer consistency 或 source chunk hard teacher 决定。
- full rollout group 先定义 prompt-level answer support；同 prompt support 内的真实 suffix chunk 被注入为 candidate proposal，score 使用 support mass 的 soft target。
- source chunk 仍保留在 candidate 0，但只作为 prior / drift guard；support anchors 从 candidate 1 开始填充，不再让 source chunk 直接决定 target。
- 本轮 `score_mode=support_flow`，因此 short probe 被显式跳过：`chunk_state_probe/skipped_for_support_flow=1.000`。

运行：

```text
run_id = ttrl_chunk_state_powerflow_support_flow_suffix_softmass_mid_c128_b32_r32_v64_1step_20260801
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_flow_suffix_softmass_mid_c128_b32_r32_v64_1step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_suffix_softmass_mid_c128_b32_r32_v64_1step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_suffix_softmass_mid_c128_b32_r32_v64_1step_20260801.jsonl
```

配置：

```text
score_mode = support_flow
support_flow_score_type = soft_mass
source_mode = majority_consistent
source_select_by_mass = true
min_prompt_top_mass = 0.35
min_source_answer_mass = 0.40
source_chunk_enable = true
source_chunk_candidate_index = 0
support_anchor_enable = true
support_anchor_count = 7
support_anchor_candidate_start = 1
support_anchor_min_mass = 0.03125
support_anchor_prefix_compat_enable = false
support_anchor_skip_source = true
boundary_mode = mid
mid_boundary_min_ratio = 0.25
mid_boundary_max_ratio = 0.70
candidates = 8
chunk_size = 128
probe_samples = 1
probe_max_tokens = 1
label_consistent_only = false
zero_inconsistent_candidates = false
skip_all_negative = false
```

关键结果：

```text
source_mass_mean = 0.520
source_prompt_top_mass_mean = 0.520
source_original_acc_mean = 1.000
source_pseudo_acc_mean = 0.938

real_state_count = 11
pad_state_count = 5
skipped_support_sources = 21
majority_consistent_fallbacks = 11
boundary_mean = 592
boundary_min = 256
boundary_max = 1024

source_chunk_injected_ratio = 1.000
support_anchor_injected_ratio = 0.973
support_anchor_positive_candidate_ratio = 0.852
support_anchor_anchor_mass_mean = 0.246
support_anchor_anchor_mass_max = 0.808

support_flow_anchor_mass_mean = 0.210
support_flow_source_mass_mean = 0.520
support_flow_positive_margin_mean = -0.077
support_flow_state_keep_ratio = 1.000
support_flow_label_consistent_ratio = 0.852

answer_coverage_mean = 0.852
future_support_keep_ratio = 1.000
positive_ratio = 0.210
informative_ratio = 1.000
num_actor_samples = 88
zeroed_state_ratio = 0.312
target_entropy = 1.406
powerflow_weight_before_clip_max = 7.456
powerflow_weight_max = 4.000

actor/powerflow_loss = 0.960
actor/boxed_reward/mean = 0.248
actor/boxed_reward/max = 0.808
actor/grad_norm = 41.035
chunk_state_actor_span/response_len_mean = 119.406

timing_s/gen = 43.537
timing_s/chunk_state_chunks = 1.101
timing_s/chunk_state_score = 7.570
timing_s/chunk_state_ref = 5.316
timing_s/update_actor = 3.900
diag_rows = 16
```

diag JSONL 统计：

```text
rows = 16
answer_coverage_mean = 0.851562
answer_coverage_min = 0.500000
answer_coverage_max = 0.875000
future_support_keep_mean = 1.000000
loss_weight_mean = 0.687500
probe_positive_count_mean = 6.812500
boundary_mean = 592
source_answer_mass_mean = 0.520083
source_answer_mass_min = 0.416667
source_answer_mass_max = 0.807692
all_positive_mean = 0
all_negative_mean = 0
```

结论：

- 这是当前 chunk-state 方向里第一次真正避开 short-horizon probe local hit teacher 的 smoke。日志明确显示 `chunk_state_probe/skipped_for_support_flow=1.000`，target 来自 full-rollout support suffix soft mass，而不是局部短 probe 命中。
- target 质量出现方向性改善：`answer_coverage_mean=0.852`，显著高于前两轮 probe/local 候选的约 0.58-0.60；`support_anchor_positive_candidate_ratio=0.852`，说明候选空间确实被拉回 full support 内。
- 样本量也可训练：`num_actor_samples=88`，比 highsupport-mid 的 8 大幅改善，虽然低于 c16 softgate 的 112，但后者 target coverage/OOV 仍没有解决。
- 这不是 20-step gate 通过，只是确认“放弃局部短视可判定性、用 full support suffix 定义 target”是更正确的主线。下一步必须跑 20-step 看 validation trajectory，而不是继续在 short-probe 打分上调 gate。
- 仍有一个重要风险：`support_flow_positive_margin_mean=-0.077`，表示 support anchor soft mass 平均还低于 source mass；当前 soft target 更像 support distillation / replay proposal，不等价于已经证明每个 transition 都带来正向 support gain。20-step 必须同时看 validation 和 target drift。

下一步：

- 基于这一版启动 20-step gate：每 20 step 做 val，优先看 `mean@16/maj@16/best@16` 是否至少不低于复现 MV 的 step20。
- 20-step 过程中保留这些诊断：`answer_coverage_mean`、`support_anchor_injected_ratio`、`support_flow_positive_margin_mean`、`num_actor_samples`、`actor/grad_norm`、`timing_s/update_actor`。
- 若 20-step 不稳，下一版不回退到 short-probe teacher，而是在 support-flow 内改 score：从 `soft_mass` 升级为 `soft_relative_mass` 或 value-margin target，并加入低信息 state skip。

## 2026-08-01 support-flow suffix soft-mass 20-step gate

背景：

- 本轮把上一节通过 smoke 的 support-flow suffix soft-mass 配置扩到 20-step gate。
- 第一次启动使用 `trainer.total_epochs=1`，在 batch32 的 MATH-TTT dataloader 下自然只跑到 step15，没有触发 `test_freq=20` validation。这不是模型崩溃，而是 launcher 的 epoch 上限截断。
- 已修正 launcher：增加 `TOTAL_EPOCHS=2`，保持 `TOTAL_TRAINING_STEPS=20`、`TEST_FREQ=20`，用 `rerun1` 独立 run id 从头重跑。

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_flow_suffix_softmass_mid_c128_b32_r32_v64_20step_20260801.sh

first_run_id = ttrl_chunk_state_powerflow_support_flow_suffix_softmass_mid_c128_b32_r32_v64_20step_20260801
first_run_status = stopped_after_step15_due_to_total_epochs_1
first_raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_suffix_softmass_mid_c128_b32_r32_v64_20step_20260801.log
first_diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_suffix_softmass_mid_c128_b32_r32_v64_20step_20260801.jsonl

rerun_id = ttrl_chunk_state_powerflow_support_flow_suffix_softmass_mid_c128_b32_r32_v64_20step_20260801_rerun1
rerun_status = completed_step20_and_validation
rerun_raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_suffix_softmass_mid_c128_b32_r32_v64_20step_20260801_rerun1.log
rerun_diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_suffix_softmass_mid_c128_b32_r32_v64_20step_20260801_rerun1.jsonl
```

配置确认：

```text
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
train_batch_size = 32
rollout.n = 32
N_VOTES_PER_PROMPT = 64
VAL_N = 16
total_training_steps = 20
total_epochs = 2
test_freq = 20
dynamic_bsz = false
score_mode = support_flow
support_flow_score_type = soft_mass
support_anchor_count = 7
source_chunk_candidate_index = 0
chunk_size = 128
candidates = 8
PowerFlow loss = enabled
KL loss = disabled
```

first run 诊断：

```text
steps_seen = 15
validation = not_triggered
diag_rows = 160
Traceback = 0
OOM = 0
Timeout during comparison = 3

step15 answer_coverage_mean = 0.805
step15 num_actor_samples = 88
step15 support_flow_positive_margin_mean = -0.203
step15 update_actor = 3.424s
```

first run 结论：

- `trainer.total_epochs=1` 会让 20-step gate 在一个 epoch 结束时提前退出，必须设成 `TOTAL_EPOCHS=2` 或更高。
- 这个问题解释了为什么第一次没有 val；它不是算法崩溃，也不是 GPU/Ray 失败。

rerun1 训练统计：

```text
steps_seen = 20
diag_rows = 216
Timeout during comparison = 10
Traceback = 0
OOM = 0
progress_end = about 22min

answer_coverage_mean = 0.8085
answer_coverage_min = 0.617
answer_coverage_max = 0.875

support_anchor_injected_ratio_mean = 0.9242
support_anchor_positive_candidate_ratio_mean = 0.8085
support_flow_positive_margin_mean = -0.14285
support_flow_positive_margin_min = -0.324
support_flow_positive_margin_max = 0.000

num_actor_samples_mean = 58.8
num_actor_samples_min = 24
num_actor_samples_max = 96
positive_ratio_mean = 0.21555
target_entropy_mean = 1.46365

actor/powerflow_loss_mean = 0.61265
actor/grad_norm_mean = 12.3639
timing_s/gen_mean = 25.0949
timing_s/chunk_state_score_mean = 6.69815
timing_s/chunk_state_ref_mean = 1.07515
timing_s/update_actor_mean = 2.29475
```

rerun1 validation：

```text
val-core/math/acc/mean@16 = 0.423875
val-core/math/acc/maj@16/mean = 0.541522
val-core/math/acc/best@16/mean = 0.835492
val-aux/math/acc/maj@8/mean = 0.518274
val-aux/math/acc/best@8/mean = 0.775236
val-aux/math/format_score/maj@16/mean = 0.853730
```

20-step gate 对比：

```text
MV step20 target: mean@16 ~= 0.760, maj@16 ~= 0.820, best@16 ~= 0.901
support-flow suffix soft-mass step20: mean@16 = 0.423875, maj@16 = 0.541522, best@16 = 0.835492
```

结论：

- 20-step gate 明确失败，不能扩到 80-step。虽然 smoke 和训练过程里的 `answer_coverage_mean` 明显好于 short-probe/local-hit 方案，但 validation accuracy 大幅低于 MV gate。
- 这说明 `soft_mass` suffix replay 只解决了“candidate 是否落在 full support 内”的表层问题，没有解决“这个 chunk transition 是否让未来分布变好”。`support_flow_positive_margin_mean=-0.14285` 是关键证据：support anchor 平均仍低于 source mass，它更像 conservative distillation / replay，而不是 search-improvement target。
- 训练链路本身不是瓶颈：`update_actor_mean=2.29s`，`chunk_state_score_mean=6.70s`，20-step 训练和一次 val 可完整跑完。当前主矛盾仍是目标定义，不是 B200 infra。
- verifier 有 10 次 `Timeout during comparison` 和大量 SymPy deprecation warning，但没有 Traceback/OOM；这会影响耗时和个别 reward 噪声，但不足以解释 step20 accuracy 从 MV gate 掉到 0.42。

下一步：

- 不继续扩这个 `soft_mass` 版本到 80-step。
- 不回退到 short-horizon probe local-hit teacher。
- 下一版应在 support-flow 框架内把目标从 `soft_mass` 改成真正的 improvement target：
  - `soft_relative_mass` / relative support gain：让候选相对 source future support 有增益才高权重。
  - value-margin target：结合 prompt support top mass、source mass、candidate mass 的 margin，而不是只看 anchor mass。
  - skip 低信息 state：低 coverage、support 过平、positive margin 过负、source mass 过强但无可提升空间的 state 直接降权或跳过。
  - 保留 support suffix proposal 作为 candidate proposal / prior，但不要把 support mass 本身当最终 teacher。

方法纠偏：

- 当前最应该放弃的不是 PowerFlow loss、hardfilter/clip4、或“无 GT、依赖 group-level label estimation”的原则，而是“局部短视可判定性”这个约束。
- 不再要求 chunk 的好坏必须在 short horizon、局部 answer hit、source answer consistency 或几条短 probe 的偶然命中上被判清楚。
- `hardfilter + clip4` 能把 actor update 压到约 1.0-1.5s，说明更新速度不是主矛盾；但 coverage/OOV 仍长期在约 0.5 附近，说明 target 质量没有解决。
- `sourcegate` 这类更强 source 硬约束会把 support coverage 压低、OOV 拉高、actor samples 变少，说明继续强化 source hard teacher 不是正确方向。
- 后续 target 必须由 full-rollout group support/value 主导：先定义 prompt-level answer support、majority/value、coverage，再让 chunk 学习哪个局部 transition 会把未来分布推向这个 support。
- source chunk 只保留为 prior 或 drift guard，不能作为主要 teacher 或 hard floor。
- probe 如果继续使用，也应是 longer-horizon / staged future support gain 的估计器，而不是短 probe 局部命中本身。

## 2026-08-01 support-flow suffix soft-relative smoke

目的：

- 在 `soft_mass` 20-step gate 失败后，先用已有 `soft_relative_mass` 做 3-step smoke。
- 这版仍然跳过 short-horizon probe teacher，target 由 full-rollout support suffix mass 主导；区别是把 score 从 absolute support mass 改成 `anchor_mass / source_mass` 的软相对比例。
- 预期只验证 target density / actor batch / timing；如果仍没有正的 transition margin，就不扩到 20-step。

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_support_flow_suffix_softrel_mid_c128_b32_r32_v64_3step_20260801.sh
run_id = ttrl_chunk_state_powerflow_support_flow_suffix_softrel_mid_c128_b32_r32_v64_3step_20260801
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_suffix_softrel_mid_c128_b32_r32_v64_3step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_suffix_softrel_mid_c128_b32_r32_v64_3step_20260801.jsonl
status = completed_3_steps_no_val
```

配置：

```text
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
train_batch_size = 32
rollout.n = 32
N_VOTES_PER_PROMPT = 64
total_training_steps = 3
test_freq = -1
score_mode = support_flow
support_flow_score_type = soft_relative_mass
support_anchor_count = 7
support_anchor_candidate_start = 1
source_chunk_candidate_index = 0
chunk_size = 128
candidates = 8
dynamic_bsz = false
PowerFlow loss = enabled
KL loss = disabled
```

运行确认：

```text
chunk_state_probe/skipped_for_support_flow = 1.000
score_type_soft_relative_mass = 1.000
Traceback = 0
OOM = 0
Timeout during comparison = 3
Final validation skipped = expected
```

3-step 均值：

```text
chunk_state/answer_coverage_mean = 0.831
chunk_state/num_actor_samples = 72.000
chunk_state/positive_ratio = 0.439667
chunk_state/target_entropy = 1.506667
chunk_state/weight_max = 0.911

chunk_state_support_flow/anchor_mass_mean = 0.228667
chunk_state_support_flow/source_mass_mean = 0.521
chunk_state_support_flow/positive_margin_mean = -0.110
chunk_state_support_flow/score_mean = 0.439667
chunk_state_support_flow/score_max_mean = 0.779333
chunk_state_support_flow/label_consistent_ratio = 0.831

actor/powerflow_loss = 1.148667
actor/grad_norm = 42.403333
timing_s/gen = 29.476333
timing_s/chunk_state_score = 6.390
timing_s/chunk_state_ref = 2.392667
timing_s/update_actor = 3.086333
```

infra 证据：

- vLLM config 里 attention backend 为 `FLASH_ATTN`。
- 运行中出现 `flashinfer.jit` autotune 和 CUDA graph capture。
- NCCL 日志显示 `NCCL_NVLS_ENABLE=1`、NVLS multicast available、P2P/CUMEM、`isAllDirectP2p 1`。
- Qwen2.5-Math-7B config 显示 GQA：`num_attention_heads=28`、`num_key_value_heads=4`。

结论：

- 工程上这版是健康的：没有 short probe target，没有 OOM/Traceback，actor batch 没打空，update actor 约 3s。
- 方法上仍不能扩：`positive_margin_mean=-0.110`，说明 anchor 平均仍低于 source future mass。`soft_relative_mass` 只是把 support replay 权重拉尖，不等于“局部 transition 让未来分布变好”。
- 这再次支持当前主判断：问题不在 PowerFlow actor update，也不在 target density，而在 target 是否真实表达 search improvement。

下一步：

- 不扩 `soft_relative_mass` 到 20-step。
- 下一版不要继续在 absolute/relative support replay 之间调参。
- 需要实现真正的 future/value-gain target：对候选 chunk 后的 longer-horizon 或 staged continuation 估计其 future answer distribution，再用相对 full-rollout support 的 mass gain / top-mass margin / transport improvement 定义 `q_j`。
- short probe 如果使用，只能作为多阶段估计器的一部分，不能单独决定 teacher。

## 2026-08-01 future-gain smoothed transport smoke

目的：

- 按最新方法纠偏，停止在 `soft_mass` / `soft_relative_mass` 这种 support replay 上继续调参。
- 使用已有 `future_support_gain` 路径，让 target 来自 `state + candidate chunk + longer-horizon probe` 的 future answer distribution 相对 full-rollout support 的 transport improvement。
- probe 在这里不是局部 teacher，而是估计 future distribution 的工具；target 由 full-rollout support distribution 主导。

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_smoothed_transport_mid_c128_probe1024x4_b32_r32_v64_3step_20260801.sh
run_id = ttrl_chunk_state_powerflow_futuregain_smoothed_transport_mid_c128_probe1024x4_b32_r32_v64_3step_20260801
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_smoothed_transport_mid_c128_probe1024x4_b32_r32_v64_3step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_smoothed_transport_mid_c128_probe1024x4_b32_r32_v64_3step_20260801.jsonl
status = completed_3_steps_no_val
```

配置：

```text
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
train_batch_size = 32
rollout.n = 32
N_VOTES_PER_PROMPT = 64
total_training_steps = 3
test_freq = -1
score_mode = future_support_gain
future_support_score_type = smoothed_transport_support_gain
future_support_prior_smoothing = 4.0
future_support_min_state_coverage = 0.125
future_support_max_state_oov = 0.875
future_support_min_state_mean_mass = 0.015
future_support_min_state_max_mass = 0.03125
future_support_min_state_top_margin = 0.001
probe_samples = 4
probe_max_tokens = 1024
support_anchor_count = 3
support_anchor_candidate_start = 5
chunk_size = 128
candidates = 8
dynamic_bsz = false
PowerFlow loss = enabled
KL loss = disabled
```

运行确认：

```text
score_type_smoothed_transport_support_gain = 1.000
chunk_state_score/mode_future_support_gain = 1.000
chunk_state_probe/samples = 4.000
Traceback = 0
OOM = 0
Timeout during comparison = 3
Final validation skipped = expected
```

3-step 均值：

```text
chunk_state_future_support_gain/improved_state_ratio = 0.729333
chunk_state_future_support_gain/positive_margin_mean = 0.137333
chunk_state_future_support_gain/smoothed_transport_gain_mean = 0.073333
chunk_state_future_support_gain/smoothed_transport_affinity_mean = 0.630333
chunk_state_future_support_gain/transport_affinity_mean = 0.260000

chunk_state_future_support_gain/support_coverage_mean = 0.427667
chunk_state_future_support_gain/state_oov_mean = 0.572333
chunk_state_future_support_gain/state_keep_ratio = 0.292000
chunk_state_future_support_gain/learnable_state_keep_ratio = 0.292000
chunk_state_future_support_gain/label_consistent_ratio = 0.612000
chunk_state_future_support_gain/score_mean = 0.094000
chunk_state_future_support_gain/raw_gain_mean = -0.298667

chunk_state/num_actor_samples = 24.000000
chunk_state/answer_coverage_mean = 0.427667
chunk_state/positive_ratio = 0.094000
chunk_state/target_entropy = 1.928000
chunk_state/kept_state_ratio = 0.292000

actor/powerflow_loss = 0.285667
actor/grad_norm = 6.264000
timing_s/gen = 34.521333
timing_s/chunk_state_probe = 5.266333
timing_s/chunk_state_score = 7.815333
timing_s/chunk_state_ref = 1.645000
timing_s/update_actor = 1.145667
```

结论：

- 这是目前最符合方法目标的一版：3 个 step 都有正的 future-support improvement signal，`positive_margin_mean` 从 support replay 的负值转为正值。
- target 仍然依赖 full-rollout support distribution；probe 只是估计 candidate 后的 future distribution，不是 short-horizon local hit teacher。
- 主要问题是信号稀疏：`state_keep_ratio=0.292`、`num_actor_samples=24`，20-step 直接扩可能训练过弱。
- 但方向上优于 `soft_mass` / `soft_relative_mass`，后两者只是 support replay，不能证明 transition improvement。
- actor update 很快，均值约 1.15s；额外开销主要在 probe 和 score，均值合计约 13s，工程上可接受作为 pilot。

下一步：

- 不回退到 support replay 或 source hard teacher。
- 优先提高 candidate/proposal 的 support coverage，让 positive future-gain state 更多，而不是降低成局部短视 target。
- 最小下一版可以保持 `future_support_gain + smoothed_transport_support_gain`，把 support-suffix proposal 的覆盖从 3 个提高到 5-7 个，或放松 state gate 后跑 3-step/20-step 对比。
- 20-step gate 通过标准：`state_keep_ratio` 不低于约 0.4、`num_actor_samples` 不低于约 32-48、`positive_margin_mean` 保持正，再看 validation。

## 2026-08-01 support5 future-gain smoke

目的：

- 检查“增加 full-rollout support suffix proposal 覆盖”能否提高 `future_support_gain + smoothed_transport_support_gain` 的可学习 state 数。
- 该实验仍不把 source chunk 当 teacher；support suffix 只作为候选 proposal，最终评分仍由 candidate future answer distribution 相对 full-rollout support distribution 的 transport improvement 决定。

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_smoothed_transport_support5_mid_c128_probe1024x4_b32_r32_v64_3step_20260801.sh
run_id = ttrl_chunk_state_powerflow_futuregain_smoothed_transport_support5_mid_c128_probe1024x4_b32_r32_v64_3step_20260801
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_smoothed_transport_support5_mid_c128_probe1024x4_b32_r32_v64_3step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_smoothed_transport_support5_mid_c128_probe1024x4_b32_r32_v64_3step_20260801.jsonl
status = completed_3_steps_no_val
```

相对 support3 baseline 的改动：

```text
support_anchor_count = 5
support_anchor_candidate_start = 3
其他训练语义保持一致：batch32 / rollout32 / votes64 / chunk128 / probe1024x4 / PowerFlow loss / dynamic_bsz=false
```

对比结果：

```text
metric                                            support3      support5
improved_state_ratio                              0.729333      0.708333
positive_margin_mean                              0.137333      0.117000
smoothed_transport_gain_mean                      0.073333      0.016667
smoothed_transport_affinity_mean                  0.630333      0.598667
transport_affinity_mean                           0.260000      0.197000
support_coverage_mean                             0.427667      0.347000
state_oov_mean                                    0.572333      0.653000
state_keep_ratio                                  0.292000      0.229000
label_consistent_ratio                            0.612000      0.604333
score_mean                                        0.094000      0.073333
raw_gain_mean                                    -0.298667     -0.383000
num_actor_samples                                 24.000000     16.000000
target_entropy                                    1.928000      1.921333
timing_s/chunk_state_score                        7.815333      8.682667
timing_s/update_actor                             1.145667      0.869667
Timeout during comparison                         3             5
Timeout during parsing                            0             94
Traceback / OOM                                   0 / 0         0 / 0
```

结论：

- support5 不通过 20-step gate。它没有提高 full-support future improvement，反而降低了 support coverage、state keep ratio、actor samples 和 transport gain。
- 这说明“继续增加 support/source 侧候选约束”不是当前主线，尤其在 parsing timeout 增多时会让 future distribution estimation 更脏。
- 该结果和最新方法判断一致：最该放弃的是“局部短视可判定性”和更强 source hard constraint，不是 PowerFlow loss 或 chunk actor update。

下一步：

- 回到 support3 future-gain 作为当前 baseline。
- 设计 staged full-support target：先用 full-rollout support transport 对候选 chunk 粗排序，再对 top-k chunk 做更长/更多 future probe；probe 只用于估计 future distribution，最终 target 仍由 full-rollout group support/value 主导。
- source chunk 继续只作为 neutral prior / drift guard，不作为主要 teacher，不再扩大 support anchor 数量。

## 2026-08-01 staged full-support smoke

目的：

- 验证 staged full-support target 是否能减少“短 horizon 局部判定”噪声。
- 设计为先用便宜 base probe 对所有 chunk 候选做 full-support transport 粗排序，再只对 top-k 候选追加更长 future probe。
- 目标仍是 `smoothed_transport_support_gain`；probe 只估计 candidate future answer distribution，不直接提供 local answer teacher。

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_staged_fullsupport_mid_c128_probe512x2_extra1536x6_b32_r32_v64_3step_20260801.sh
failed_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_staged_fullsupport_mid_c128_probe512x2_extra1536x6_b32_r32_v64_3step_20260801.log
rerun_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_staged_fullsupport_mid_c128_probe512x2_extra1536x6_b32_r32_v64_3step_20260801_rerun1.log
rerun_diag = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_staged_fullsupport_mid_c128_probe512x2_extra1536x6_b32_r32_v64_3step_20260801_rerun1.jsonl
status = rerun1_completed_3_steps_no_val
```

工程修复：

```text
first failure = RuntimeError: stack expects each tensor to be equal size, but got [512] and [1536]
root cause = staged merge assumed base probe and extra probe response tensors have identical sequence length
fix = _merge_chunk_state_staged_probe_output now pads base/extra batch tensors to the merged max shape before torch.stack
```

配置：

```text
base_probe = 512 tokens x 2 samples
staged_topk = 2 chunks per state
extra_probe = 1536 tokens x 6 samples
score_probe_samples = 8 after repeat_base merge
support_anchor_count = 3
source_prior_weight = 0.5
prior_smoothing = 6.0
dynamic_bsz = false
```

对比结果：

```text
metric                                            support3      staged_rerun1
improved_state_ratio                              0.729333      0.875000
positive_margin_mean                              0.137333      0.104333
smoothed_transport_gain_mean                      0.073333      0.015000
smoothed_transport_affinity_mean                  0.630333      0.560667
transport_affinity_mean                           0.260000      0.231333
support_coverage_mean                             0.427667      0.372667
state_oov_mean                                    0.572333      0.627333
state_keep_ratio                                  0.292000      0.208333
label_consistent_ratio                            0.612000      0.598667
score_mean                                        0.094000      0.055667
raw_gain_mean                                    -0.298667     -0.310667
num_actor_samples                                 24.000000     18.666667
target_entropy                                    1.928000      1.832667
timing_s/chunk_state_probe                        5.266333      2.646667
timing_s/chunk_state_staged_probe_extra           n/a           6.856333
timing_s/chunk_state_score                        7.815333      10.912000
timing_s/update_actor                             1.145667      0.958333
Timeout during comparison                         3             3
Timeout during parsing                            0             0
Traceback / OOM                                   0 / 0         0 / 0
```

额外诊断：

```text
staged_base/positive_margin_mean = 0.277667
staged_base/smoothed_transport_gain_mean = 0.264667
staged_base/state_keep_ratio = 0.291667
staged_base/support_coverage_mean = 0.412667
```

结论：

- staged 机制工程上已打通，但方法上不扩 20-step。
- base 粗筛指标显著好于最终 long-probe 指标，说明 `512x2 + smoothing=6` 产生了过乐观的 first-stage signal；它不能作为可靠 target。
- 最终 target 的关键指标低于 support3：`smoothed_transport_gain_mean` 从 0.073 降到 0.015，`state_keep_ratio` 从 0.292 降到 0.208，actor samples 从 24 降到 18.7。
- 这再次说明不能让短 horizon 粗筛主导 target，即便它名义上使用了 full-support transport。

下一步：

- 不扩 staged 到 20-step。
- 不再让 short base probe 先决定 teacher 候选。
- 下一版改为全候选 longer-horizon future distribution estimation：保持所有 candidates 都用同一 probe horizon，先验证 `probe1536x4` 或 `probe2048x4` 是否提高 support coverage / state_keep_ratio / transport gain。
- 如果全候选长 probe 仍不提升，再考虑改 target 公式，而不是继续堆 staged gate。

## 2026-08-01 full-candidate long-probe future-gain smoke

目的：

- 验证最新方法判断：不要让 short-horizon probe/local hit/source consistency 主导 chunk target。
- 取消 staged top-k 粗筛，对同一 state 的全部 8 个 candidates 使用同一较长 horizon probe 来估计 future answer distribution。
- target 仍由 full-rollout group support distribution 定义，probe 只作为 candidate future distribution estimator。

运行：

```text
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_3step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_3step_20260801.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_3step_20260801.jsonl
status = completed_3_steps_no_val
```

配置：

```text
batch = 32 prompts
rollout.n = 32
votes_per_prompt = 64
chunk_size = 128
candidates = 8
probe = 1536 tokens x 4 samples
staged_probe_enable = false
support_anchor_count = 3
support_anchor_candidate_start = 5
score = future_support_gain / smoothed_transport_support_gain
PowerFlow loss = enabled
dynamic_bsz = false
```

运行确认：

```text
model = /models/Qwen2.5-Math-7B
venv = /mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200
vLLM attention_config.backend = FLASH_ATTN
FlashInfer autotune = observed
NCCL NVLS = enabled
NCCL P2P/CUMEM = observed
Traceback / OOM = 0 / 0
Timeout during comparison = 3
Timeout during parsing = 0
Final validation skipped = expected
```

3-step 均值：

```text
chunk_state_future_support_gain/improved_state_ratio = 0.979333
chunk_state_future_support_gain/positive_margin_mean = 0.173333
chunk_state_future_support_gain/smoothed_transport_gain_mean = 0.119333
chunk_state_future_support_gain/smoothed_transport_affinity_mean = 0.674667
chunk_state_future_support_gain/transport_affinity_mean = 0.349667

chunk_state_future_support_gain/support_coverage_mean = 0.543000
chunk_state_future_support_gain/state_oov_mean = 0.457000
chunk_state_future_support_gain/state_keep_ratio = 0.437333
chunk_state_future_support_gain/label_consistent_ratio = 0.854000
chunk_state_future_support_gain/score_mean = 0.104333
chunk_state_future_support_gain/raw_gain_mean = -0.196000

chunk_state/num_actor_samples = 32.000000
chunk_state/target_entropy = 1.817000

timing_s/generate_sequences = 22.003000
timing_s/gen = 30.231000
timing_s/chunk_state_probe = 7.444000
timing_s/chunk_state_score = 8.176333
timing_s/chunk_state_ref = 1.801333
timing_s/update_actor = 1.426667
```

与 support3 baseline 对比：

```text
metric                                            support3      fullcand1536x4
improved_state_ratio                              0.729333      0.979333
positive_margin_mean                              0.137333      0.173333
smoothed_transport_gain_mean                      0.073333      0.119333
smoothed_transport_affinity_mean                  0.630333      0.674667
transport_affinity_mean                           0.260000      0.349667
support_coverage_mean                             0.427667      0.543000
state_oov_mean                                    0.572333      0.457000
state_keep_ratio                                  0.292000      0.437333
label_consistent_ratio                            0.612000      0.854000
score_mean                                        0.094000      0.104333
raw_gain_mean                                    -0.298667     -0.196000
num_actor_samples                                 24.000000     32.000000
target_entropy                                    1.928000      1.817000
timing_s/chunk_state_probe                        5.266333      7.444000
timing_s/chunk_state_score                        7.815333      8.176333
timing_s/update_actor                             1.145667      1.426667
Timeout during comparison                         3             3
Timeout during parsing                            0             0
```

结论：

- 这是目前 chunk-state future-support 系列里最好的 3-step smoke，且方向和最新方法约束一致。
- 全候选长 probe 明显改善 target 质量：coverage 从 0.428 到 0.543，OOV 从 0.572 到 0.457，state keep 从 0.292 到 0.437，transport gain 从 0.073 到 0.119。
- 该结果支持“放弃局部短视可判定性”：不要用短 probe/top-k/source hard gate 过早决定 teacher，而是让所有候选用较长 horizon 估计 future distribution。
- 代价是 probe 从约 5.27s 增至 7.44s，score 约 8.18s，update_actor 仍只有约 1.43s；瓶颈仍不是 chunk actor update。
- step 1 端到端约 111s，受 CUDA graph/JIT/首轮生成长度影响；step 2/3 约 48-51s，稳定后可接受作为 20-step pilot。

下一步：

- 可以扩到 20-step validation gate，优先验证 mean/maj/best trajectory 是否真正优于 MV 复现线。
- 20-step 前建议保持语义不变，只做工程层面的日志解析和 timeout 统计；不要重新引入 staged short-probe top-k。
- 如果 20-step 训练稳定但速度偏慢，再考虑用 B200 大显存做并行 candidate probe batching 或更长 max_num_batched_tokens，而不是改变 target 语义。

## 2026-08-02 20-step gate 修正与方法约束更新

本轮方法约束更新：

- 明确放弃“局部短视可判定性”作为主监督假设。
- 不再要求 chunk target 主要由 short-horizon probe 的局部命中、局部 answer hit 或 source consistency 决定。
- full-rollout group support/value 必须先定义 prompt-level 的好答案分布；chunk 只学习哪个 local transition 会把未来 completion distribution 推向该 support。
- source chunk / teacher anchor 只能作为 prior 或 drift guard，不能作为主要 teacher，也不能用更强 source hard gate 继续收紧 target。
- probe 的角色降级为 future distribution estimator：可以拉长、分阶段或用于估计 candidate future support，但不能单独把短 probe local hit 当成最终 label。
- 低信息 state 仍应跳过或降权：all-negative、高 OOV、support coverage 低、support mass 太平、malformed/repeated boxed/marker 污染等。

20-step gate 第一次运行状态：

```text
run_id = ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_20step_20260801
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_20step_20260801.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_20step_20260801.log
status = incomplete_15_of_20_no_final_val
```

根因：

- 配置里 `trainer.total_training_steps=20` 生效，但训练循环仍由 `for epoch in total_epochs` 和 dataloader 驱动。
- MATH-TTT train set 长度 500，batch 32 时 `Size of train dataloader: 15`。
- 第一次 20-step launcher 继承 `TOTAL_EPOCHS=1`，所以第 15 个 batch 后 dataloader 耗尽，训练自然退出，没有进入 `is_last_step`，因此没有 final validation。
- 另外 final validation 分支要求 `trainer.test_freq > 0`；之前 `TEST_FREQ=-1` 即使 `final_val_enable=True` 也不会触发最后验证。

修正：

```text
TOTAL_EPOCHS=2
TOTAL_TRAINING_STEPS=20
TEST_FREQ=20
FINAL_VAL_ENABLE=True
SAVE_FREQ=-1
```

rerun：

```text
run_id = ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_20step_20260801_rerun1
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_20step_20260801_rerun1.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_20step_20260801_rerun1.jsonl
output_dir = /tmp/ttrl_b200/checkpoints/ttrl_chunk_state_powerflow_futuregain_fullcand_mid_c128_probe1536x4_b32_r32_v64_20step_20260801_rerun1
status = completed_failed_gate
```

当前工程确认：

```text
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
venv = /mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200
dynamic_bsz = false
vLLM attention_config.backend = FLASH_ATTN
NCCL NVLS = enabled
NCCL P2P/CUMEM = observed
```

rerun1 final validation：

```text
val-core/math/acc/mean@16 = 0.468875
val-core/math/acc/best@16/mean = 0.826982
val-core/math/acc/maj@16/mean = 0.593118
val-aux/math/format_score/mean@16 = 0.895750
val-aux/math/format_score/maj@16/mean = 0.875520
timing_s/testing = 301.784s
```

20-step mean diagnostics：

```text
chunk_state_future_support_gain/support_coverage_mean = 0.419250
chunk_state_future_support_gain/state_oov_mean = 0.580750
chunk_state_future_support_gain/state_keep_ratio = 0.224950
chunk_state_future_support_gain/smoothed_transport_gain_mean = 0.051350
chunk_state_future_support_gain/improved_state_ratio = 0.803100
chunk_state_future_support_gain/positive_margin_mean = 0.130500
chunk_state_future_support_gain/label_consistent_ratio = 0.633550

chunk_state/num_actor_samples = 26.400000
chunk_state/zeroed_state_ratio = 0.806200
chunk_state/positive_ratio = 0.078400
actor/pg_loss = 0.260050

timing_s/gen = 25.980400
timing_s/chunk_state_probe = 7.391450
timing_s/chunk_state_score = 7.605300
timing_s/chunk_state_ref = 0.630900
timing_s/update_actor = 1.127300
training_progress_s_per_it_all = 64.141
training_progress_s_per_it_step2_to_19 = 61.668
```

逐步诊断要点：

```text
state_keep_ratio == 0 的 step = 9, 15, 17
support_coverage_mean range = 0.203 - 0.662
state_oov_mean range = 0.338 - 0.797
num_actor_samples range = 8 - 64
```

结论：

- 这次 corrected 20-step gate 没有通过，final mean@16 只有 0.468875，明显低于 MV/PowerFlow 对齐目标。
- 失败不是 actor update 慢导致的：update_actor 均值约 1.13s，端到端主要仍在 full rollout generation、probe 和 score。
- 真正主矛盾是 target 质量和可学习 state 覆盖：20-step 的 support coverage 均值只有 0.419，OOV 均值 0.581，state keep 均值 0.225，且 9/15/17 三个 step 的 keep 直接为 0。
- 3-step smoke 的高 coverage/高 keep 没能在 20-step 上稳定复现，说明 full-candidate long-probe 虽然比 support/source hard gate 好，但仍没有解决 full-rollout group support/value 到 chunk target 的稳定投影问题。
- 不能回退到 short-horizon local answer hit / source consistency 当主 teacher，也不能继续加 source hard gate；下一轮应把 target 改成更明确的 full-rollout support/value 主导形式，让 probe 只估计 future distribution，低 coverage / 高 OOV / flat support / malformed state 直接跳过或强降权。

下一轮建议：

- 先实现 per-prompt full rollout support/value cache：answer support distribution、top mass、coverage、majority answer、source rollout future distribution。
- state 采样优先来自 support 清晰且 majority-consistent 的中后段，低信息 state 直接不进 actor batch。
- candidate score 改为相对 full group support 的 future distribution improvement，例如 support mass gain、transport/KL improvement、value margin；source chunk 只作为 prior/drift guard。
- 保留 PowerFlow-style distribution matching 和 hardfilter+clip4 的工程骨架，但不让 short probe local hit 或 source answer consistency 决定 teacher。

## 2026-08-02 support-strict 3-step smoke

目的：

- 验证“继续提高 full-rollout support/source/candidate hard gate”是否能解决 20-step gate 暴露的 target 质量问题。
- 该实验不改变 loss 主体，仍用 future_support_gain + PowerFlow-style chunk update；只把 prompt/source/support/candidate coverage 门槛调严。
- 如果严格门控能同时提高 coverage 并保持足够 actor samples，再考虑 20-step；如果只提高 source 质量但压低 keep ratio，则说明 hard gate 不是主方向。

配置：

```text
run_id = ttrl_chunk_state_powerflow_futuregain_supportstrict_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportstrict_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_supportstrict_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportstrict_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
status = completed_failed_smoke

chunk_state_min_prompt_top_mass = 0.50
chunk_state_min_source_answer_mass = 0.50
chunk_state_support_anchor_min_mass = 0.0625
chunk_state_future_support_min_mass = 0.0625
chunk_state_future_support_min_positive_margin = 0.02
chunk_state_future_support_min_state_coverage = 0.35
chunk_state_future_support_max_state_oov = 0.65
chunk_state_future_support_min_state_mean_mass = 0.08
chunk_state_future_support_min_state_max_mass = 0.25
chunk_state_future_support_min_state_top_margin = 0.01
chunk_state_future_support_min_candidate_coverage = 0.25
chunk_state_future_support_min_candidate_mean_mass = 0.05
chunk_state_min_answer_coverage = 0.35
chunk_state_min_informative_gap = 0.01
```

3-step mean diagnostics：

```text
chunk_state_source/selected_original_acc_mean = 1.000000
chunk_state_diag/source_answer_mass_mean = 0.676667
chunk_state_support_anchor/anchor_mass_mean = 0.676333

chunk_state_future_support_gain/support_coverage_mean = 0.484333
chunk_state_future_support_gain/state_oov_mean = 0.515667
chunk_state_future_support_gain/state_keep_ratio = 0.166667
chunk_state_future_support_gain/learnable_state_keep_ratio = 0.166667
chunk_state_future_support_gain/smoothed_transport_gain_mean = 0.011333
chunk_state_future_support_gain/positive_margin_mean = 0.127667
chunk_state_future_support_gain/label_consistent_ratio = 0.541667

chunk_state/num_actor_samples = 32.000000
chunk_state/zeroed_state_ratio = 0.833333
chunk_state/actor_batch_powerflow_weight_nonzero_ratio = 0.666667
actor/pg_loss = 0.127667

timing_s/gen = 34.165667
timing_s/chunk_state_probe = 7.049333
timing_s/chunk_state_score = 7.305667
timing_s/update_actor = 1.450333
```

逐步诊断：

```text
step 1: state_keep_ratio = 0.125, num_actor_samples = 8, source_answer_mass_mean = 0.645
step 2: state_keep_ratio = 0.000, actor_batch_powerflow_weight_nonzero_ratio = 0.000, pg_loss = 0
step 3: state_keep_ratio = 0.375, num_actor_samples = 24, source_answer_mass_mean = 0.704
```

结论：

- support-strict 提高了 source/support 质量：source answer mass 均值约 0.677，明显高于 20-step failed gate 里的普通状态。
- 但它没有解决可学习 state 稀疏，反而把 state_keep_ratio 压到 0.167，step 2 直接全零更新。
- 这证明“继续加 source/support/candidate hard gate”不是下一步主方向；它只是在减少样本，并没有把 candidate future distribution 与 full-rollout support 更稳定地对齐。
- 不扩 20-step。
- 下一步应进入代码层面的 target 重构：把 per-prompt full-rollout answer support/value 显式作为目标分布，候选 chunk 的 probe 只估计 future distribution，再用 transport/KL/value-improvement 构造 soft target；source chunk 仍只作为 prior/drift guard，不再继续硬收 source 侧约束。

## 2026-08-02 support-value-affinity 3-step smoke

目的：

- 放弃“候选 chunk 必须相对 source 产生正 gain 才能当 teacher”的强约束，转向更软的 full-rollout support/value affinity。
- `score_type=support_value_affinity` 定义为 `future_value * smoothed_transport_affinity`，让 target 直接偏向“未来分布有价值且更贴近 full-rollout support”的 candidate。
- source chunk 仍保留为 prior / drift guard，但不作为主要 teacher，也不再继续加 source 侧 hard gate。

配置：

```text
run_id = ttrl_chunk_state_powerflow_futuregain_supportvalue_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportvalue_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_supportvalue_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportvalue_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
status = completed_failed_smoke

ttrl.chunk_state_future_support_score_type = support_value_affinity
ttrl.chunk_state_min_prompt_top_mass = 0.40
ttrl.chunk_state_min_source_answer_mass = 0.35
ttrl.chunk_state_support_anchor_min_mass = 0.03125
ttrl.chunk_state_future_support_min_mass = 0.03125
ttrl.chunk_state_future_support_min_positive_margin = 0.08
ttrl.chunk_state_future_support_min_state_coverage = 0.25
ttrl.chunk_state_future_support_max_state_oov = 0.75
ttrl.chunk_state_future_support_min_state_mean_mass = 0.04
ttrl.chunk_state_future_support_min_state_max_mass = 0.12
ttrl.chunk_state_future_support_min_state_top_margin = 0.005
ttrl.chunk_state_future_support_min_candidate_coverage = 0.0
ttrl.chunk_state_future_support_min_candidate_mean_mass = 0.0
```

代码增量：

```text
score_matrix = future_value.clamp(0, 1) * smoothed_transport_affinity
```

3-step mean diagnostics：

```text
chunk_state_future_support_gain/support_coverage_mean = 0.401000
chunk_state_future_support_gain/state_oov_mean = 0.599000
chunk_state_future_support_gain/state_keep_ratio = 0.270667
chunk_state_future_support_gain/learnable_state_keep_ratio = 0.270667
chunk_state_future_support_gain/smoothed_transport_affinity_mean = 0.632667
chunk_state_future_support_gain/smoothed_transport_gain_mean = 0.076333
chunk_state_future_support_gain/positive_margin_mean = 0.369333
chunk_state_future_support_gain/label_consistent_ratio = 0.617333

chunk_state/num_actor_samples = 16.000000
chunk_state/zeroed_state_ratio = 0.854000
chunk_state/actor_batch_powerflow_weight_nonzero_ratio = 1.000000
actor/pg_loss = 1.094000

timing_s/gen = 29.957667
timing_s/chunk_state_probe = 7.573333
timing_s/chunk_state_score = 7.938000
timing_s/chunk_state_ref = 1.538000
timing_s/update_actor = 0.884333
```

逐步诊断：

```text
step 1: support_coverage = 0.541, state_oov = 0.459, state_keep = 0.312, num_actor_samples = 24
step 2: support_coverage = 0.418, state_oov = 0.582, state_keep = 0.375, num_actor_samples = 8
step 3: support_coverage = 0.244, state_oov = 0.756, state_keep = 0.125, num_actor_samples = 16
```

结论：

- support-value-affinity 路径能跑通，且 actor update 很快，均值约 0.88s；训练更新速度不是主矛盾。
- 相比 support-strict，state_keep_ratio 从 0.167 提到 0.271，step 2 不再全零更新，说明放松 source/support hard gate 是正确方向。
- 但 target 质量仍不达标：support coverage 均值只有 0.401，OOV 均值 0.599，step 3 甚至变成 coverage 0.244 / OOV 0.756 / keep 0.125。
- 这进一步支持最新判断：不能让 short-horizon probe 的局部 answer hit / source consistency 主导 target，也不能通过继续加硬门控解决；probe 只能作为 future distribution estimator。
- 下一步要把 label estimation 前移到 full rollout group：先建立 prompt-level answer support/value，再对 chunk candidate 的后续分布做 transport/KL/value-improvement 匹配；低 support coverage、高 OOV、support 太平或 malformed 的 state 直接跳过或强降权。

## 2026-08-02 support-distribution-match 3-step smoke

目的：

- 进一步把 chunk target 从 scalar local hit / source consistency 转成 full-rollout support distribution matching。
- 新增 `score_type=support_distribution_match`，候选 chunk 的 probe 只用来估计 future answer distribution。
- score 由 full support distribution 主导：

```text
support_expected_value = E_{future answer distribution}[full_support_mass(answer)]
support_overlap = overlap(future answer distribution, full_support_distribution)
score = 0.5 * (support_expected_value + support_overlap) * smoothed_transport_affinity
```

配置：

```text
run_id = ttrl_chunk_state_powerflow_futuregain_supportdist_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportdist_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_supportdist_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportdist_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
status = completed_failed_smoke

ttrl.chunk_state_future_support_score_type = support_distribution_match
ttrl.chunk_state_future_support_min_positive_margin = 0.04
ttrl.chunk_state_future_support_min_state_coverage = 0.25
ttrl.chunk_state_future_support_max_state_oov = 0.75
ttrl.chunk_state_future_support_min_state_mean_mass = 0.04
ttrl.chunk_state_future_support_min_state_max_mass = 0.12
ttrl.chunk_state_future_support_min_state_top_margin = 0.005
ttrl.chunk_state_min_answer_coverage = 0.25
ttrl.chunk_state_min_informative_gap = 0.005
```

3-step mean diagnostics：

```text
chunk_state_future_support_gain/support_coverage_mean = 0.390667
chunk_state_future_support_gain/state_oov_mean = 0.609333
chunk_state_future_support_gain/state_keep_ratio = 0.291667
chunk_state_future_support_gain/learnable_state_keep_ratio = 0.291667
chunk_state_future_support_gain/smoothed_transport_affinity_mean = 0.628667
chunk_state_future_support_gain/support_expected_value_mean = 0.184000
chunk_state_future_support_gain/support_overlap_mean = 0.257667
chunk_state_future_support_gain/candidate_entropy_mean = 0.608000
chunk_state_future_support_gain/smoothed_transport_gain_mean = 0.032333
chunk_state_future_support_gain/positive_margin_mean = 0.293667
chunk_state_future_support_gain/label_consistent_ratio = 0.591000

chunk_state/num_actor_samples = 26.666667
chunk_state/zeroed_state_ratio = 0.750000
chunk_state/actor_batch_powerflow_weight_nonzero_ratio = 1.000000
actor/pg_loss = 0.943333
actor/grad_norm = 6.903333

timing_s/gen = 29.688333
timing_s/chunk_state_probe = 7.758000
timing_s/chunk_state_score = 8.227667
timing_s/chunk_state_ref = 1.680000
timing_s/update_actor = 1.300000
```

逐步诊断：

```text
step 1: support_coverage = 0.541, state_oov = 0.459, state_keep = 0.375, num_actor_samples = 32
step 2: support_coverage = 0.334, state_oov = 0.666, state_keep = 0.250, num_actor_samples = 32
step 3: support_coverage = 0.297, state_oov = 0.703, state_keep = 0.250, num_actor_samples = 16
```

结论：

- support-distribution-match 代码路径跑通，且新增指标能直接观测 full support distribution matching：`support_expected_value`、`support_overlap`、`candidate_entropy`。
- 相比 support-value-affinity，`state_keep_ratio` 略升到 0.292，`zeroed_state_ratio` 从 0.854 降到 0.750，step 2/3 没有全零更新；这说明显式分布匹配比 source/gain 硬约束更可训练。
- 但它仍未通过扩展条件：coverage 均值 0.391、OOV 均值 0.609，step 2/3 仍然大量 future answers 不在 full-rollout support 内。
- 不扩 20-step。主问题已经从 loss 公式转为 label estimation / state-probe distribution 质量：需要让 probe 更可靠地估计“未来分布是否靠近 full support”，而不是继续调 PowerFlow 权重或 gate。
- 下一步优先做 state/probe 侧：减少低支持 prompt，改用 support 清晰且 coverage 高的 state；或者提高 full rollout support 样本数 / probe reuse，使 full support distribution 对 chunk state 的监督更稳定。

## 2026-08-02 support-distribution-match probe3072 3-step smoke

目的：

- 检验上一组 support-distribution-match 的高 OOV / 低 coverage 是否主要来自 probe horizon 太短。
- 只把 `ttrl.chunk_state_probe_max_tokens` 从 1536 提到 3072，其余 target 语义保持一致。
- 这个实验不是为了继续强化 short-horizon local hit，而是验证“拉长 probe 是否足够让 future answer distribution 更贴近 full-rollout support”。

配置：

```text
run_id = ttrl_chunk_state_powerflow_futuregain_supportdist_mid_c128_probe3072x4_b32_r32_v64_3step_20260802
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportdist_mid_c128_probe3072x4_b32_r32_v64_3step_20260802.sh
raw_log = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_supportdist_mid_c128_probe3072x4_b32_r32_v64_3step_20260802.log
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportdist_mid_c128_probe3072x4_b32_r32_v64_3step_20260802.jsonl
status = completed_failed_smoke

ttrl.chunk_state_future_support_score_type = support_distribution_match
ttrl.chunk_state_probe_max_tokens = 3072
ttrl.chunk_state_probe_samples = 4
ttrl.chunk_state_future_support_min_positive_margin = 0.04
ttrl.chunk_state_future_support_min_state_coverage = 0.25
ttrl.chunk_state_future_support_max_state_oov = 0.75
```

3-step mean diagnostics：

```text
chunk_state_future_support_gain/support_coverage_mean = 0.403000
chunk_state_future_support_gain/state_oov_mean = 0.597000
chunk_state_future_support_gain/state_keep_ratio = 0.291667
chunk_state_future_support_gain/learnable_state_keep_ratio = 0.291667
chunk_state_future_support_gain/smoothed_transport_affinity_mean = 0.635667
chunk_state_future_support_gain/support_expected_value_mean = 0.193667
chunk_state_future_support_gain/support_overlap_mean = 0.271333
chunk_state_future_support_gain/candidate_entropy_mean = 0.669333
chunk_state_future_support_gain/smoothed_transport_gain_mean = 0.070667
chunk_state_future_support_gain/positive_margin_mean = 0.337000
chunk_state_future_support_gain/label_consistent_ratio = 0.653667

chunk_state/num_actor_samples = 18.666667
chunk_state/zeroed_state_ratio = 0.833000

timing_s/gen = 29.585667
timing_s/chunk_state_probe = 14.763667
timing_s/chunk_state_score = 8.460333
timing_s/chunk_state_ref = 1.549000
timing_s/update_actor = 0.949000
```

逐步诊断：

```text
step 1: support_coverage = 0.521, state_oov = 0.479, state_keep = 0.312, num_actor_samples = 24
step 2: support_coverage = 0.461, state_oov = 0.539, state_keep = 0.375, num_actor_samples = 8
step 3: support_coverage = 0.227, state_oov = 0.773, state_keep = 0.188, num_actor_samples = 24
```

对 probe1536 的直接对比：

```text
probe1536 support_coverage = 0.390667, state_oov = 0.609333, state_keep = 0.291667, probe_time = 7.758000s
probe3072 support_coverage = 0.403000, state_oov = 0.597000, state_keep = 0.291667, probe_time = 14.763667s
```

结论：

- probe horizon 从 1536 拉到 3072 只把 coverage 从 0.391 小幅提高到 0.403，OOV 从 0.609 小幅降到 0.597，state_keep 完全没有提高。
- probe 成本从约 7.76s 增加到约 14.76s，几乎翻倍；这不值得扩到 20-step。
- source 质量不是主问题：diag 里 `source_majority_consistent` 均值 1.0，`source_original_correct` 均值约 0.975，`source_answer_mass` 均值约 0.553。
- 失败仍然是 candidate/probe future distribution 与 full-rollout support 的重叠不够；继续依赖更长的 local probe 或更强 source gate 不是主线。
- 下一步应按最新约束改设计：full rollout group 先定义 prompt-level support/value；chunk candidate 只学习是否把 future distribution 推向该 support；source chunk 只做 prior/drift guard；低信息 state 直接跳过或降权。不要再让 short-horizon local hit / source consistency 主导 teacher。

## 2026-08-02 prompt-level support-quality supportq2 3-step smoke

目的：

- 验证最新方法约束：不要再用 short-horizon local hit / source consistency / source hard gate 主导 chunk target。
- 在 support-distribution-match 基础上，只做 prompt-level full-rollout support quality 过滤：full rollout group 先定义 support/value；candidate probe 只估计 future answer distribution。
- 修正第一版 supportq 的问题：第一版仍继承 `chunk_state_min_source_answer_mass=0.4` / `chunk_state_source_select_by_mass=True`，且 top-margin/entropy 曾用 raw count 而非 mass。supportq2 已显式关闭 source mass hard gate，并修复 mass 计算。

配置：

```text
run_id = ttrl_chunk_state_powerflow_futuregain_supportdist_supportq2_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
launcher = /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportdist_supportq2_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
diag_jsonl = /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportdist_supportq2_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
status = completed_positive_smoke_not_yet_20step

ttrl.chunk_state_future_support_score_type = support_distribution_match
ttrl.chunk_state_min_source_answer_mass = 0.0
ttrl.chunk_state_source_select_by_mass = False
ttrl.chunk_state_min_prompt_valid_answer_coverage = 0.75
ttrl.chunk_state_min_prompt_top_mass = 0.45
ttrl.chunk_state_min_prompt_top_margin = 0.05
ttrl.chunk_state_max_prompt_answer_entropy = 1.4
ttrl.chunk_state_probe_max_tokens = 1536
ttrl.chunk_state_probe_samples = 4
```

3-step mean diagnostics：

```text
answer_coverage_mean = 0.609375
state_oov_mean = 0.390625
future_support_keep = 0.291667
future_support_learnable_keep = 0.291667
future_support_state_mean_mass = 0.375190
future_support_state_max_mass = 0.616268
future_support_state_top_margin = 0.051783

prompt_valid_answer_coverage = 0.933594
prompt_answer_top_margin = 0.527915
prompt_answer_entropy = 1.073852
source_answer_mass = 0.673708
source_prompt_top_mass = 0.673708
source_original_correct = 0.958333
source_majority_consistent = 1.000000

probe_mean = 0.358529
probe_max = 0.540050
probe_positive_count = 6.291667

real_state_count = 5.333333
pad_state_count = 2.666667
skipped_support_sources = 26.666667
loss_weight = 0.666667
```

逐步诊断：

```text
step 1:
  answer_coverage = 0.554688
  state_oov = 0.445312
  future_support_keep = 0.250000
  prompt_valid_answer_coverage = 0.906250
  prompt_answer_top_margin = 0.670177
  prompt_answer_entropy = 1.170663
  probe_mean = 0.340326
  timing_s/gen = 43.397
  timing_s/chunk_state_probe = 7.114
  timing_s/chunk_state_score = 8.362
  timing_s/chunk_state_ref = 4.062
  timing_s/update_actor = 0.990

step 2:
  answer_coverage = 0.535156
  state_oov = 0.464844
  future_support_keep = 0.250000
  prompt_valid_answer_coverage = 0.953125
  prompt_answer_top_margin = 0.664928
  prompt_answer_entropy = 0.936970
  probe_mean = 0.355401
  timing_s/gen = 23.099
  timing_s/chunk_state_probe = 7.120
  timing_s/chunk_state_score = 8.049
  timing_s/chunk_state_ref = 0.275
  timing_s/update_actor = 0.771

step 3:
  answer_coverage = 0.738281
  state_oov = 0.261719
  future_support_keep = 0.375000
  prompt_valid_answer_coverage = 0.941406
  prompt_answer_top_margin = 0.248642
  prompt_answer_entropy = 1.113922
  probe_mean = 0.379859
  timing_s/gen = 22.369
  timing_s/chunk_state_probe = 6.911
  timing_s/chunk_state_score = 6.529
  timing_s/chunk_state_ref = 0.273
  timing_s/update_actor = 0.814
```

对 supportdist/probe3072 的直接对比：

```text
supportdist probe1536:
  support_coverage = 0.390667
  state_oov = 0.609333
  state_keep = 0.291667
  probe_time = 7.758000s
  update_actor = 1.300000s

supportdist probe3072:
  support_coverage = 0.403000
  state_oov = 0.597000
  state_keep = 0.291667
  probe_time = 14.763667s
  update_actor = 0.949000s

supportq2 prompt-quality:
  support_coverage = 0.609375
  state_oov = 0.390625
  state_keep = 0.291667
  probe_time ~= 7.048333s
  update_actor ~= 0.858333s
```

结论：

- 这是一个正向 smoke：prompt-level full-rollout support quality 过滤把 support coverage 从约 0.39/0.40 提到 0.61，OOV 从约 0.60 降到 0.39，且没有增加 probe 成本。
- 这个改善来自 full-rollout group support/value 的 prompt-level label estimation，而不是 source answer hard gate。`chunk_state_min_source_answer_mass=0`、`chunk_state_source_select_by_mass=False` 已生效。
- 代价是 state 供给变稀：每 step 8 个 state 里只保留 2-3 个 learnable state，`skipped_support_sources` 均值 26.7。它更像“高置信 target 入口”，不是最终完整训练方案。
- 不应回退到更强 source gate。source chunk 只保留为 prior / drift guard；继续加 source 侧 hard constraint 已被 sourcegate/supportq 第一版证明会降低 coverage、抬高 OOV。
- 下一步可以进入 20-step pilot，但目标应明确：验证高质量 prompt support filter 是否能改善 validation；同时设计 candidate/proposal 侧补强，例如混入 high-support rollout suffix/replay proposal，使保留下来的 state 数量增加，而不是降低 target 质量去追求样本量。

## 2026-08-02 supportq2 20-step pilot 结果

运行：

```text
run_id = ttrl_chunk_state_powerflow_futuregain_supportdist_supportq2_mid_c128_probe1536x4_b32_r32_v64_20step_20260802
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
train_batch_size = 32
rollout.n = 32
val.n = 16
total_training_steps = 20
dynamic_bsz = false
chunk_score = future_support_gain / support_distribution_match
probe = 4 samples, max_tokens 1536
source hard gate = off
prompt support quality gate = on
```

产物：

```text
launcher:
  /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportdist_supportq2_mid_c128_probe1536x4_b32_r32_v64_20step_20260802.sh
diag:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportdist_supportq2_mid_c128_probe1536x4_b32_r32_v64_20step_20260802.jsonl
val:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_supportdist_supportq2_mid_c128_probe1536x4_b32_r32_v64_20step_20260802_val_metrics.json
raw log:
  未按预期落盘到 important_experiment_logs，最终 val 可从 val json 和 worker session 输出恢复。
```

最终 validation：

```text
mean@16 = 0.466250
maj@16  = 0.594092
best@16 = 0.852098
format_mean@16 = 0.897875
format_maj@16  = 0.872378
```

state-level 诊断汇总：

```text
diag rows = 176
steps = 20
answer_coverage_mean = 0.505859
state_oov_mean = 0.494141
future_support_keep_mean = 0.136364
future_support_learnable_keep_mean = 0.136364
prompt_valid_answer_coverage_mean = 0.905717
prompt_answer_top_margin_mean = 0.673565
prompt_answer_entropy_mean = 1.010018
source_answer_mass_mean = 0.741878
zero-update steps = 9 / 20
zero-update step ids = 2, 3, 5, 7, 10, 14, 15, 18, 20
```

逐 step keep / coverage：

```text
step  keep   coverage  oov
1     0.250  0.555     0.445
2     0.000  0.828     0.172
3     0.000  0.082     0.918
4     0.125  0.520     0.480
5     0.000  0.746     0.254
6     0.375  0.621     0.379
7     0.000  0.109     0.891
8     0.125  0.531     0.469
9     0.125  0.688     0.312
10    0.000  0.668     0.332
11    0.125  0.543     0.457
12    0.125  0.309     0.691
13    0.250  0.463     0.537
14    0.000  0.289     0.711
15    0.000  0.262     0.738
16    0.438  0.697     0.303
17    0.250  0.672     0.328
18    0.000  0.160     0.840
19    0.125  0.652     0.348
20    0.000  0.574     0.426
```

结论：

- 这次 20-step pilot 失败，validation 明显低于 MV / PowerFlow 参考轨迹，不应继续按该配置扩 80-step。
- prompt-level full-rollout support quality filter 是有效的：`prompt_valid_answer_coverage_mean=0.906`、`prompt_answer_top_margin_mean=0.674`、`source_answer_mass_mean=0.742`，说明 full group 能筛出高质量 prompt/source。
- 失败点在 chunk candidate / target：candidate 侧 `answer_coverage_mean=0.506`，`future_support_learnable_keep_mean=0.136`，且 45% 的 step 没有有效 actor update。即便 full prompt support 很强，当前 chunk proposal 仍经常离开 full-rollout answer support。
- update_actor 不是主矛盾。有效更新步里 update_actor 常在 0.4-1s 量级；20-step 慢和失败主要来自 full rollout + probe/score 成本，以及 target keep 太稀。
- 日志中出现 repeated boxed 污染样例，说明 malformed / repeated boxed 过滤还需要进入 state/candidate quality gate。

设计纠偏：

- 放弃“局部短视可判定性”这个训练约束。不能再要求 chunk target 主要由 short-horizon probe 的局部命中、source answer consistency 或少量 probe 是否碰巧答对来定义。
- probe 只能作为 future answer distribution estimator，不能直接当 teacher。它的作用是估计 candidate chunk 把未来 completion 推向 full-rollout group support 的程度。
- full rollout group 必须先定义 prompt-level support / value / answer mass；chunk 学的是哪个 local transition 会让未来分布更接近这个 support。
- source chunk 保留为 prior / drift guard，不再作为 hard floor 或主要 teacher。
- hard keep 需要改成 soft distribution matching / soft weighting。现在大量 step 被 hard keep 清零，直接导致训练信号稀疏；下一版应对低质量 state 降权或跳过，但不能让可学习状态被 `top_margin` 等局部硬阈值过度清零。
- 下一版优先做 support-conditioned proposal：对同一 state 的 candidate 采样不只从 base policy 采 next chunk，还混入 high-support rollout suffix/replay chunk 或 full-support-conditioned anchor proposal，使 candidate distribution 自身更靠近 full-rollout support，再用 PowerFlow-style q_j ∝ exp(alpha * support_score_j) * prior_j 训练。

## 2026-08-02 soft keep + support-anchor proposal 3-step smoke

目的：

- 验证上一节的核心诊断：`future_support_keep` hard gate 是 20-step 中大量零更新的直接工程原因。
- 不改变 full-rollout support/value 主导 target 的原则，只把 state-level keep 从 hard gate 改成 soft weight。
- 同时打开 support-anchor proposal，把同 prompt 的 high-support rollout suffix/replay chunk 注入 candidate 集合，验证 support-conditioned proposal 能否保持非零 actor update。

代码改动：

```text
ray_trainer.py:
  新增 chunk_state_future_support_soft_weight = state_coverage * state_max_mass
  新增 ttrl.chunk_state_future_support_keep_mode = hard | soft | off
  hard: 保持旧行为，future_support_keep=false 时整 state 清零
  soft: 不用 future_support_keep 清零整 state，而是用 soft_weight 乘到 state loss weight
  off: 完全不使用 future_support_keep

ppo_trainer_ttrl.yaml:
  默认 chunk_state_future_support_keep_mode: hard
  默认 chunk_state_future_support_soft_weight_floor: 0.0
```

运行：

```text
run_id = ttrl_chunk_state_powerflow_futuregain_supportdist_softkeep_anchorprop_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
train_batch_size = 32
rollout.n = 32
total_training_steps = 3
final_val_enable = false
score = future_support_gain / support_distribution_match
probe = 4 samples, max_tokens 1536
support_anchor_enable = true
support_anchor_count = 3
support_anchor_candidate_start = 5
future_support_keep_mode = soft
future_support_soft_weight_floor = 0.05
```

产物：

```text
launcher:
  /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportdist_softkeep_anchorprop_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
diag:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportdist_softkeep_anchorprop_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
raw worker output:
  /home/tiger/.trae/cli/sessions/2026/07/12/rollout-2026-07-12T04-54-36-019f54ad-60d1-7981-938e-2c5f116e8fe3.artifacts/tool-results/exec_command-call_IydwJfsZXxInavqZl5yd9iVD-2.txt
```

diag 汇总：

```text
diag rows = 32
steps = 1,2,3
answer_coverage_mean = 0.481445
state_oov_mean = 0.518555
future_support_keep_mean = 0.062500
future_support_learnable_keep_mean = 0.062500
future_support_state_mean_mass = 0.354567
future_support_state_max_mass = 0.601146
future_support_state_top_margin = 0.036441
probe_mean = 0.323084
prompt_valid_answer_coverage_mean = 0.891602
prompt_answer_top_margin_mean = 0.687526
prompt_answer_entropy_mean = 1.093173
loss_weight_mean = 0.562500
source_answer_mass_mean = 0.734520
```

逐 step：

```text
step  coverage  hard_keep  learnable_keep  diag_loss_weight
1     0.554688  0.250      0.250           0.625
2     0.583984  0.000      0.000           0.5625
3     0.203125  0.000      0.000           0.500
```

关键 step 证据：

```text
step 2:
  future_support_keep_ratio = 0.000
  future_support_keep_mode_soft = 1.000
  future_support_soft_weight_mean = 0.455
  future_support_state_weight_mean = 0.465
  kept_state_ratio = 1.000
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  grad_norm = 23.428
  update_actor = 2.723s

step 3:
  future_support_keep_ratio = 0.000
  future_support_keep_mode_soft = 1.000
  future_support_soft_weight_mean = 0.139
  future_support_state_weight_mean = 0.154
  kept_state_ratio = 1.000
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  grad_norm = 5.390
  update_actor = 1.255s
```

结论：

- soft keep smoke 跑通，证明 20-step 失败中的“零更新”不是不可避免的算法现象，而是 actor batch 里的 hard keep 直接造成的工程/目标构造问题。
- 在 step 2 和 step 3，旧 hard keep 会因为 `future_support_keep_ratio=0` 把整步清零；soft keep 下 `kept_state_ratio=1.0`、`actor_batch_powerflow_weight_nonzero_ratio=1.0`，actor 有真实梯度。
- support-anchor proposal 没有破坏 full-support target 语义：它只是把 high-support rollout suffix 注入 candidate 集合，最终 score 仍由 `support_distribution_match` 和 full-rollout support distribution 定义。
- 这版不应该直接扩 80-step。虽然零更新问题被解决，但 step 3 的 `answer_coverage=0.203`，candidate support 仍明显不稳。下一步应继续改 proposal：增加 replay/suffix anchor 覆盖、降低无效 base-policy chunk 比例，或切到 `support_flow soft_mass` 先验证纯 full-support proposal 的训练曲线。
- 后续 20-step 应使用 soft keep，但必须同时提高 candidate support coverage，否则只是“有梯度地学噪声”。

## 2026-08-02 full-support anchor prior + soft keep 3-step smoke

目的：

- 按最新纠偏，放弃“短 horizon 局部命中必须直接判定 teacher”的约束。
- 保留 PowerFlow-style distribution matching，但把 full-rollout support anchor 只作为 `target_prior`，不作为 score floor。
- 目标形式变成：

```text
q_j ∝ exp(alpha * future_support_score_j) * prior_j

future_support_score_j:
  由 probe future answer distribution 和 prompt-level full-rollout support distribution 的 match 定义。
prior_j:
  对注入的 high-support suffix/replay anchor 乘以 1 + anchor_prior_weight * anchor_mass。
```

代码改动：

```text
ray_trainer.py:
  新增 ttrl.chunk_state_future_support_anchor_prior_weight
  新增 ttrl.chunk_state_future_support_anchor_prior_power
  当 future_support_gain 路径存在 chunk_state_support_anchor_scores 时，
  用 support-anchor mass 构造 target_prior 乘子。

ppo_trainer_ttrl.yaml:
  默认 anchor_prior_weight = 0.0
  默认 anchor_prior_power = 1.0
```

运行：

```text
run_id = ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
model = /models/Qwen2.5-Math-7B
data = /mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT
train_batch_size = 32
rollout.n = 32
total_training_steps = 3
final_val_enable = false
chunk_state_score_mode = future_support_gain
future_support_score_type = support_distribution_match
support_anchor_count = 4
support_anchor_candidate_start = 4
future_support_anchor_prior_weight = 4.0
future_support_prior_smoothing = 8.0
future_support_keep_mode = soft
future_support_soft_weight_floor = 0.05
probe = 4 samples, max_tokens 1536
```

产物：

```text
launcher:
  /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
raw log:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log
diag:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
```

启动问题：

- 第一次启动失败在 Ray socket path，原因是 `TTRL_RUNTIME_DIR=/tmp/ttrl_b200/runtime_${RUN_ID}` 太长。
- 修复为短路径 `/tmp/cfsp82` 后正常启动。
- 这不是训练语义问题，属于 Ray UNIX socket 路径长度限制。

diag 汇总：

```text
diag rows = 40
steps = 1,2,3
answer_coverage_mean = 0.525000
state_oov_mean = 0.475000
future_support_state_mean_mass = 0.268649
future_support_state_max_mass = 0.492381
future_support_state_top_margin = 0.033298
probe_mean = 0.263889
source_answer_mass_mean = 0.559468
source_prompt_top_mass_mean = 0.559468
prompt_valid_answer_coverage_mean = 0.845313
prompt_answer_top_margin_mean = 0.487966
prompt_answer_entropy_mean = 1.683339
loss_weight_mean = 0.675000
future_support_keep_mean = 1.000000
future_support_learnable_keep_mean = 1.000000
```

逐 step：

```text
step  rows  real_state  pad_state  skipped_support  coverage  probe_mean  loss_weight
1     16    11          5          21               0.515625  0.226647    0.6875
2     16    9           7          23               0.568359  0.332460    0.5625
3     8     7           1          25               0.457031  0.201231    0.8750
```

关键训练指标：

```text
step 1:
  target_prior_mean = 1.504
  answer_coverage_mean = 0.516
  future_support_soft_weight_mean = 0.254
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  grad_norm = 21.467
  update_actor = 3.643s

step 2:
  target_prior_mean = 1.929
  answer_coverage_mean = 0.568
  future_support_soft_weight_mean = 0.356
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  grad_norm = 15.974
  update_actor = 2.859s

step 3:
  target_prior_mean = 1.671
  answer_coverage_mean = 0.457
  future_support_soft_weight_mean = 0.218
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  grad_norm = 7.000
  update_actor = 2.431s
```

时延：

```text
step 1:
  gen = 43.645s
  chunk_probe = 7.675s
  chunk_score = 10.365s
  ref = 5.089s
  update_actor = 3.643s

step 2:
  gen = 22.560s
  chunk_probe = 7.655s
  chunk_score = 8.373s
  ref = 1.100s
  update_actor = 2.859s

step 3:
  gen = 22.403s
  chunk_probe = 7.150s
  chunk_score = 6.752s
  ref = 0.813s
  update_actor = 2.431s
```

结论：

- 这版比 supportq2 / softkeep-anchor smoke 更接近正确语义：full-rollout support 通过 score 和 prior 共同定义目标，anchor 不是 hard teacher，短 probe 不是局部 hit teacher，而是 future answer distribution estimator。
- `actor_batch_powerflow_weight_nonzero_ratio=1.0` 连续稳定，说明 soft keep 解决了 hard gate 清零问题。
- `answer_coverage_mean=0.525` 比上一版 softkeep smoke 的整体 `0.481` 好，step 2 达到 `0.568`，说明 full-support anchor prior 有帮助。
- 不能直接扩 20-step。主要问题是 source selection 太窄：`min_source_answer_mass=0.40` 导致每步只剩 `7-11` 个 real states，大量 prompt 被 `skipped_support_sources` 跳过。这个会让训练分布过窄，并且 pad state 比例过高。
- 下一轮应保持同样 target 语义，只放宽 source gate：把 `chunk_state_min_source_answer_mass` 降到 `0.20-0.25`，保留 prompt-level support quality gate 和 mid boundary，目标是每步恢复到接近 32 个 real states，同时观察 coverage/OOV 是否仍维持在 `0.50+`。
- 如果放宽后 coverage 下降明显，再考虑 state-level soft weight 或 prompt support quality 的连续权重，不回到 source hard gate 或 short-horizon local teacher。

## 2026-08-02 chunk-state PowerFlow: full-support prior + soft keep + relaxed source gate smoke

目的：

- 验证上一版 full-support anchor prior 的主要瓶颈是不是 `min_source_answer_mass=0.40` 太窄。
- 只放宽 source hard gate，不改变 target 语义：full-rollout support distribution 仍通过 `future_support_gain/support_distribution_match` 定义 score，anchor 仍只做 prior，不做 hard teacher。

配置差异：

```text
base = ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
chunk_state_min_source_answer_mass = 0.25
chunk_state_min_prompt_top_mass = 0.30
TTRL_RUNTIME_DIR = /tmp/cfsp82b
```

产物：

```text
launcher:
  /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_relaxedsrc025_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
raw log:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_relaxedsrc025_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log
diag:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_relaxedsrc025_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
```

逐 step 结果：

```text
step  real_state  pad_state  skipped_support  coverage  source_mass  target_prior  nonzero  update_actor
1     15          1          17               0.488     0.490        1.387         1.000    4.831s
2     10          6          22               0.414     0.628        1.259         1.000    3.143s
3     11          5          21               0.334     0.417        1.389         1.000    3.411s
```

结论：

- 这是一个负结果。放宽 source mass hard gate 没有把 real states 恢复到接近 32，只从上一版的 `7-11` 提到 `10-15`，仍然有大量 prompt/source 被跳过。
- target 质量反而变差：coverage 从上一版整体约 `0.525` 掉到 step 级 `0.488/0.414/0.334`，OOV 对应上升。
- `actor_batch_powerflow_weight_nonzero_ratio=1.0` 仍稳定，说明 soft keep / clip 后 update 不是主矛盾。
- 当前最该放弃的是“chunk 的好坏必须由短 horizon、局部 answer hit、source consistency 这一级判清楚”的约束。短 probe 只能作为 future answer distribution estimator，不能作为 teacher；source chunk 只能作为 prior/drift guard，不能作为 hard floor 或主要监督。
- 下一版不继续扩 source-side hard constraints：保持 full-support target + anchor prior + soft keep，取消 `min_source_answer_mass` hard gate，使用 prompt-level support quality 选择可学习 prompt，并把 source answer mass 只作为连续 loss weight / prior。目标是恢复状态覆盖，同时让 q_j 仍由 full-rollout support/value 主导。

## 2026-08-02 chunk-state PowerFlow: no source hard gate + source continuous weight smoke

目的：

- 执行上一节结论：取消 `min_source_answer_mass` 对 source rollout 的 hard gate。
- source answer mass 只作为连续 state loss weight，不再决定 state 是否存在；full-rollout support distribution 仍主导 `q_j`。
- 保留 prompt-level support gate，避免低信息 prompt 直接进入训练。

配置：

```text
base = ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
chunk_state_min_source_answer_mass = 0.0
chunk_state_min_prompt_top_mass = 0.30
chunk_state_source_select_by_mass = True
chunk_state_source_quality_weight_mode = source_mass
chunk_state_source_quality_weight_floor = 0.05
chunk_state_source_quality_weight_power = 0.5
```

启动说明：

- 新 wrapper 文件已落盘：`verl/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_sourceweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`。
- worker 文件视图当时未看到这个新 wrapper，所以实际运行用已存在 base launcher 加同等 CLI overrides。
- 由于 shell 展开顺序问题，worker 侧原始 diag 先写到了 `chunk_state_diag/.jsonl`；已归档为标准 run id 文件。

产物：

```text
launcher:
  /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_sourceweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
raw log:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_sourceweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log
diag:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_sourceweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
```

逐 step 结果：

```text
step  rows  real_state  pad_state  skipped_support  coverage  oov     source_mass  probe_mean  loss_weight
1     24    24          0          8                0.509115  0.4909  0.491254     0.203030    1.000
2     16    16          0          16               0.527344  0.4727  0.602963     0.295374    1.000
3     24    18          6          14               0.454427  0.5456  0.483652     0.210829    0.750
```

训练侧关键指标：

```text
step 1:
  source_quality_weight_mean = 0.694
  answer_coverage_mean = 0.509
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  update_actor = 7.303s

step 2:
  source_quality_weight_mean = 0.769
  answer_coverage_mean = 0.527
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  update_actor = 4.674s

step 3:
  source_quality_weight_mean = 0.685
  answer_coverage_mean = 0.454
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  update_actor = 5.350s
```

结论：

- 这是正向 smoke。相比 relaxed-source025 的 `real_state=15/10/11`、coverage `0.488/0.414/0.334`，取消 source hard gate + 连续 source weight 提升到 `real_state=24/16/18`、coverage `0.509/0.527/0.454`。
- 说明主矛盾确实不是 source 侧 hard constraint 不够强，而是 hard constraint 把训练状态分布压窄了。
- `source_quality_weight_mean=0.685-0.769`，说明 source mass 作为连续权重在起作用，但没有把样本清零；`actor_batch_powerflow_weight_nonzero_ratio=1.0` 继续稳定。
- 仍不能直接宣称已解决：step 2/3 仍只有 16/18 real states，主要剩余瓶颈是 `chunk_state_min_prompt_top_mass=0.30` 仍是 prompt-level hard gate。下一版应把 prompt support 也从 hard gate 改成连续权重，或把 `min_prompt_top_mass` 降到 `0.0-0.2` 后用 `prompt_top_mass` / coverage / entropy 组成 state quality weight。
- 方法方向保持：full-rollout support/value 定义 target，probe 仅估计 future distribution，source/anchor 只做 prior 和连续 drift guard。

## 2026-08-02 chunk-state PowerFlow: no source/prompt hard gates + product weight smoke

目的：

- 进一步测试是否可以完全取消 source/prompt hard gate，用连续质量权重保持样本覆盖。
- 这版将 `source_answer_mass * prompt_top_mass` 作为 state loss weight 的基础，仍保持 full-support target + anchor prior + soft keep。

配置：

```text
base = ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
chunk_state_min_source_answer_mass = 0.0
chunk_state_min_prompt_top_mass = 0.0
chunk_state_source_select_by_mass = True
chunk_state_source_quality_weight_mode = product
chunk_state_source_quality_weight_floor = 0.05
chunk_state_source_quality_weight_power = 0.5
```

产物：

```text
launcher:
  /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_sourcepromptweight_nogates_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
raw log:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_sourcepromptweight_nogates_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log
diag:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_sourcepromptweight_nogates_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
```

注意：

- 训练主体完成并打印 `Final validation skipped`。
- 退出阶段有 `DataLoader worker ... killed by signal: Killed` 的 weakref cleanup traceback；发生在训练完成后，指标和 diag 已落盘。
- 由于临时 diag 文件混入上一轮 source-soft 的 64 行，标准归档文件已修正为只保留本轮后 96 行。

逐 step 结果：

```text
step  rows  real_state  pad_state  skipped_support  coverage  oov     source_correct  source_mass  probe_mean  loss_weight
1     32    32          0          0                0.464844  0.5352  0.843750        0.416713     0.169265    1.000
2     32    32          0          0                0.395508  0.6045  0.687500        0.402151     0.145902    1.000
3     32    32          0          0                0.444336  0.5557  0.750000        0.340878     0.154012    1.000
```

训练侧关键指标：

```text
step 1:
  source_quality_weight_mean = 0.418
  answer_coverage_mean = 0.465
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  update_actor = 9.519s

step 2:
  source_quality_weight_mean = 0.403
  answer_coverage_mean = 0.396
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  update_actor = 9.388s

step 3:
  source_quality_weight_mean = 0.343
  answer_coverage_mean = 0.444
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  update_actor = 9.183s
```

结论：

- 这版实现了覆盖目标：`real_state=32/32/32`、`skipped_support=0`，说明 prompt/source hard gate 都不是必须的。
- 但 target 质量明显变脏：coverage 只有 `0.465/0.396/0.444`，source original correct 降到 `0.844/0.688/0.750`，比 source-soft 的 `0.509/0.527/0.454` 和 source correct `1.0/0.938/0.958` 更差。
- 不建议直接扩 20-step。更合理的下一版是折中：保留 source hard gate off，但 prompt 侧不要完全放开；用 `min_prompt_top_mass=0.20` 或连续 weight 加强低质量 prompt 降权，同时保持 real states 尽量接近 24-32。
- 如果要扩 20-step，当前优先级仍是 source-soft no-source-gate 版，而不是 no-gates product 版。

## 2026-08-02 chunk-state PowerFlow: prompt top-mass 0.20 + product weight smoke

目的：

- 验证 no-gates product 与 source-soft 之间的折中点。
- 保持 source hard gate off，避免回到 source-side hard constraint；prompt 侧只保留弱 top-mass gate。
- 继续遵守新的 target 语义：short probe 只估计 future answer distribution，`q_j` 由 full-rollout support/value、support-distribution match、anchor prior、OOV/coverage 质量共同定义；source chunk 只做 prior/drift guard，不做 teacher 或 hard floor。

配置：

```text
base = ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
chunk_state_min_source_answer_mass = 0.0
chunk_state_min_prompt_top_mass = 0.20
chunk_state_source_select_by_mass = True
chunk_state_source_quality_weight_mode = product
chunk_state_source_quality_weight_floor = 0.05
chunk_state_source_quality_weight_power = 0.5
```

产物：

```text
launcher:
  /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
raw log:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log
diag:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
```

运行状态：

- 训练主体完成，raw log 1149 行，diag 96 行，正好是 3 step x 32 states。
- 无 RuntimeError / Traceback；最后按 smoke 配置打印 `Final validation skipped`。
- 确认模型为 `/models/Qwen2.5-Math-7B`，数据为 `data/MATH-TTT`，venv 为 `/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200`。
- vLLM config 中 `attention_config.backend=FLASH_ATTN`，启动阶段有 flashinfer autotune 和 CUDA graph capture；NCCL 日志显示 NVLS 可用、P2P direct 通过。

逐 step 结果：

```text
step  rows  real_state  pad_state  skipped_support  coverage  oov     source_correct  source_mass  prompt_top  probe_mean  loss_weight
1     32    29          3          3                0.482422  0.5176  0.906250        0.450254     0.454719    0.189564    0.906250
2     32    25          7          7                0.502930  0.4971  0.781250        0.546111     0.546111    0.252860    0.781250
3     32    25          7          7                0.360352  0.6396  0.718750        0.413163     0.413163    0.150081    0.781250
```

训练侧关键指标：

```text
step 1:
  source_quality_weight_mean = 0.452
  answer_coverage_mean = 0.482
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  num_actor_samples = 232
  gen = 43.501s
  chunk_probe = 9.096s
  chunk_score = 10.479s
  chunk_ref = 7.259s
  update_actor = 8.841s
  step wall = 122.74s

step 2:
  source_quality_weight_mean = 0.546
  answer_coverage_mean = 0.503
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  num_actor_samples = 200
  gen = 23.064s
  chunk_probe = 9.151s
  chunk_score = 8.442s
  chunk_ref = 2.858s
  update_actor = 7.279s
  step wall = 86.91s

step 3:
  source_quality_weight_mean = 0.413
  answer_coverage_mean = 0.360
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  num_actor_samples = 200
  gen = 22.248s
  chunk_probe = 9.214s
  chunk_score = 20.005s
  chunk_ref = 2.886s
  update_actor = 7.441s
  step wall ~= 127.74s
```

结论：

- 这是一个有价值但还不够稳定的折中 smoke。
- 相比 no-gates product，prompt top-mass 0.20 明显改善了 step 2 target 质量：coverage 从 no-gates step 2 的 `0.396` 提到 `0.503`，并且 real states 仍有 `25/32`。
- 相比 source-soft，prompt020 的状态覆盖更稳定：`29/25/25` 好于 source-soft 的 `24/16/18`；但 coverage 不稳定，step 3 掉到 `0.360`，说明低信息 state / prompt 仍会混入。
- 不支持直接扩 20-step。下一版应该继续保留 full-support target + soft keep + source hard gate off，但增加 full-support 质量降权/过滤：例如基于 `prompt_valid_answer_coverage`、`prompt_answer_top_margin`、candidate coverage/OOV、support entropy 做 state weight 或 early skip。
- 不能回到“source consistency / short-probe local hit 当 teacher”的路线。当前失败点仍是 target 质量，不是 actor update；稳态 `update_actor` 约 `7.3-7.4s`，主要时间在 full rollout、chunk probe 和 `chunk_state_score` 的 answer parsing / support scoring。

## 2026-08-02 chunk-state PowerFlow: prompt020 + full-support coverage gate cov040 smoke

目的：

- 在 prompt top-mass 0.20 + product continuous weight 的基础上，加入真正影响 actor batch 的 full-support coverage gate。
- 验证 `chunk_state_min_answer_coverage=0.40` 是否能过滤低信息 state，避免 prompt020 第 3 step coverage 掉到 `0.360`。
- 继续遵守最新约束：不能让 short-horizon probe/local answer hit/source consistency 主导 target；coverage gate 只能看 full-rollout support/value 质量，不回退到 source hard teacher。

配置：

```text
base = ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
chunk_state_min_answer_coverage = 0.40
chunk_state_confidence_power = 0.5
chunk_state_min_source_answer_mass = 0.0
chunk_state_min_prompt_top_mass = 0.20
chunk_state_source_select_by_mass = True
chunk_state_source_quality_weight_mode = product
chunk_state_source_quality_weight_floor = 0.05
chunk_state_source_quality_weight_power = 0.5
```

产物：

```text
launcher:
  /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_prompt020_cov040_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
raw log:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_prompt020_cov040_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log
diag:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_prompt020_cov040_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
```

运行状态：

- 3-step smoke 完成，无 RuntimeError / Traceback。
- diag 88 行，不是 96 行：step 3 gate 后只有 24 个真实 state 进入本轮 actor batch/diag。
- 模型仍为 `/models/Qwen2.5-Math-7B`，数据为 `data/MATH-TTT`，venv 为 `/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200`。

逐 step 结果：

```text
step  rows  real_state  pad_state  skipped_support  coverage  oov     source_correct  source_mass  prompt_top  probe_mean  loss_weight  state_mean_mass  state_max_mass  top_margin
1     32    29          3          3                0.482422  0.5176  0.906250        0.450254     0.454719    0.189564    0.906250    0.194856         0.430132        0.052269
2     32    27          5          5                0.370117  0.6299  0.812500        0.506799     0.506799    0.136728    0.843750    0.141400         0.318099        0.035094
3     24    24          0          8                0.510417  0.4896  0.916667        0.476603     0.477948    0.221492    1.000000    0.224135         0.453056        0.018830
```

训练侧关键指标：

```text
step 1:
  confidence_gate_ratio = 0.625
  confidence_weight_mean = 0.442
  num_actor_samples = 144
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  gen = 43.529s
  chunk_probe = 9.079s
  chunk_score = 12.043s
  chunk_ref = 5.942s
  update_actor = 5.846s
  step wall = 120.12s

step 2:
  confidence_gate_ratio = 0.500
  confidence_weight_mean = 0.375
  num_actor_samples = 128
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  gen = 23.302s
  chunk_probe = 9.553s
  chunk_score = 10.574s
  chunk_ref = 1.884s
  update_actor = 4.851s

step 3:
  confidence_gate_ratio = 0.792
  confidence_weight_mean = 0.467
  num_actor_samples = 152
  actor_batch_powerflow_weight_nonzero_ratio = 1.000
  gen = 22.510s
  chunk_probe = 8.386s
  chunk_score = 9.346s
  chunk_ref = 2.138s
  update_actor = 5.438s
```

对 prompt020 的对比：

```text
prompt020 coverage: 0.482 / 0.503 / 0.360
cov040   coverage: 0.482 / 0.370 / 0.510

prompt020 num_actor_samples: 232 / 200 / 200
cov040   num_actor_samples: 144 / 128 / 152

prompt020 update_actor: 8.841s / 7.279s / 7.441s
cov040   update_actor: 5.846s / 4.851s / 5.438s
```

结论：

- cov040 没有打空 batch，且能明显压低 actor update：`num_actor_samples` 从约 `200-232` 降到 `128-152`，`update_actor` 从约 `7-9s` 降到 `5s` 左右。
- 但它没有稳定改善 target 质量。step 2 的 support coverage 直接掉到 `0.370`，OOV 到 `0.630`；step 3 虽回到 `0.510`，但真实 state 只有 24 个，状态数不稳定。
- 因此 cov040 不是可扩 20-step 的正结果。它证明“full-support gate 能提速”，但没有解决主矛盾：target/proposal 仍没有可靠地把 local transition 指向 full-rollout group support。
- 下一步不要继续沿 source hard gate、short-probe local hit、source consistency 的方向加码，也不要把 cov040 当成主线。更合理的最小改动是：改 candidate/proposal 生成和 target 定义，让 score 直接来自 longer-horizon / staged future support distribution match；low-information state 做 soft skip/降权，但不能让短视局部命中重新成为 teacher。

## 2026-08-02 chunk-state PowerFlow: prompt020 + staged top2 extra probe smoke

目的：

- 验证 staged future-support estimator：先对所有 candidate 做 base probe，再对每个 state 的 top-2 candidate 补 4 条更长 probe。
- 目标不是把短 probe 局部命中当 teacher，而是用更多 future distribution evidence 改善 support-distribution match。
- 不使用 cov040 actor batch gate，避免把“提速 gate”与“target 质量”混在一起判断。

配置：

```text
base = ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802
chunk_state_staged_probe_enable = True
chunk_state_staged_probe_topk = 2
chunk_state_staged_probe_extra_samples = 4
chunk_state_staged_probe_extra_max_tokens = 2048
chunk_state_staged_probe_merge_mode = repeat_base
```

产物：

```text
launcher:
  /mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_prompt020_stagedtop2x4_2048_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh
raw log:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_prompt020_stagedtop2x4_2048_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log
diag:
  /mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_fullsupport_prior_softkeep_prompt020_stagedtop2x4_2048_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl
```

运行状态：

- 3-step smoke 完成，无 RuntimeError / Traceback，最后按 smoke 配置打印 `Final validation skipped`。
- diag 88 行：`32 + 32 + 24`，与 step 3 真实 state 数一致。
- 模型、数据、venv 仍为 `/models/Qwen2.5-Math-7B`、`data/MATH-TTT`、`/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200`。
- 启动日志再次确认 vLLM `attention_config.backend=FLASH_ATTN`、flashinfer autotune、CUDA graph capture、NCCL NVLS/P2P direct；actor 侧 `use_fused_kernels=True`。

逐 step 结果：

```text
step  rows  real_state  pad_state  skipped_support  coverage  oov     source_correct  source_mass  prompt_top  probe_mean  loss_weight  state_mean_mass  state_max_mass  top_margin
1     32    29          3          3                0.458008  0.5420  0.906250        0.450254     0.454719    0.164783    0.906250    0.185299         0.438374        0.052486
2     32    25          7          7                0.464355  0.5356  0.906250        0.494799     0.494799    0.177234    0.781250    0.198526         0.391767        0.021041
3     24    24          0          8                0.468099  0.5319  0.916667        0.473780     0.483039    0.183976    1.000000    0.199172         0.452195        0.043440
```

base probe 与 staged merge 后对比：

```text
step  base_coverage  merged_coverage  base_score  merged_score  selected_chunks  num_actor_samples
1     0.482          0.458            0.190       0.165         64               232
2     0.482          0.464            0.200       0.177         64               200
3     0.486          0.468            0.206       0.184         48               192
```

训练侧关键耗时：

```text
step 1:
  chunk_probe = 9.088s
  staged_extra = 9.038s
  chunk_score = 12.941s
  update_actor = 8.866s
  gen = 43.500s

step 2:
  chunk_probe = 9.343s
  staged_extra = 9.117s
  chunk_score = 11.813s
  update_actor = 7.418s
  gen = 22.478s

step 3:
  chunk_probe = 8.458s
  staged_extra = 8.972s
  chunk_score = 9.195s
  update_actor = 7.026s
  gen = 22.073s
```

结论：

- staged top2 extra probe 是负结果，不扩 20-step。
- 它没有改善 target 质量：base coverage 约 `0.482/0.482/0.486`，merge 后反而变成 `0.458/0.464/0.468`；OOV 仍约 `0.53-0.54`。
- 它还每步额外增加约 `9s` 的 staged probe 开销，actor update 仍在 `7-9s`，整体没有性价比。
- 这进一步说明问题不在“对已有 top-k candidate 多 probe 几次”，而在 candidate/proposal 本身没有可靠进入 full-rollout support 分布。下一步应该改 proposal/target：构造 support-conditioned candidate，或者在 score 中直接使用 candidate future distribution 的 transport affinity / support overlap，而不是继续给 top-k 加深 probe。

## 2026-08-02 soft prefix support-anchor smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softprefix_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 目标：验证 full-rollout support anchor 作为 proposal/prior 时，prefix compatibility 不再用 hard filter 打空 anchor set，而是 soft downweight。
- 关键配置：
  - `ttrl.chunk_state_score_mode=future_support_gain`
  - `ttrl.chunk_state_future_support_score_type=support_distribution_match`
  - `ttrl.chunk_state_support_anchor_enable=True`
  - `ttrl.chunk_state_support_anchor_prefix_compat_enable=True`
  - `ttrl.chunk_state_support_anchor_prefix_compat_mode=soft`
  - `ttrl.chunk_state_support_anchor_prefix_score_power=1.0`
  - `actor_rollout_ref.actor.powerflow_enable=True`
  - `actor_rollout_ref.actor.use_dynamic_bsz=False`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_fullsupport_prior_softprefix_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`
- raw console log: `important_experiment_logs/ttrl_chunk_state_powerflow_fullsupport_prior_softprefix_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`

注意：这次 worker 后台 nohup 会在 `mlx worker login` 退出时被清掉，所以改成前台执行。console log 从 Trae exec artifact 归档，部分长输出被截断；完整 state 级质量指标以 diag jsonl 为准，step 级 summary 来自前台会话输出。

三步质量汇总：

```text
step  support_coverage  OOV     source_mass  prompt_top_mass  raw_positive  label_consistent  num_actor_samples
1     0.482             0.518   0.450        0.455            0.199         0.699             232
2     0.558             0.442   0.554        0.554            0.308         0.766             200
3     0.453             0.547   0.375        0.383            0.143         0.660             208
mean  0.498             0.502   0.460        0.464            0.217         0.708             213
```

support-anchor 相关现象：

```text
step  injected_ratio  state_keep  prefix_match_mean  anchor_mass_mean  positive_anchor_candidate_ratio
1     0.977           1.000       0.264              0.254             0.488
2     0.945           1.000       0.139              0.394             0.473
3     0.992           1.000       0.418              0.196             0.496
```

耗时：

```text
step  gen      chunk_probe  chunk_score  chunk_ref  update_actor  progress_step
1     43.461   9.129        10.269       7.175      8.775         122.4s
2     22.504   9.399        8.852        2.860      7.488         ~63s incremental
3     22.769   9.398        9.353        2.945      7.601         ~64s incremental
```

结论：

- soft prefix compatibility 解决了 hard compatibility 的工程问题：anchor set 没有被打空，`injected_ratio=0.945-0.992`，`state_keep=1.0`。
- 但它没有解决 target 质量主矛盾：coverage 仍只有 `0.45-0.56`，均值约 `0.50`；OOV 均值约 `0.50`。这和用户纠偏一致，问题不是 source hard gate 太少，而是 target 仍然没有被 full-rollout group distribution 足够强地定义。
- 这轮不扩 20-step。下一步不再继续加 source/local short-probe 约束，应转向：先用 full rollout group 定义 prompt-level support/value，再选择高 support 中后段 state，并让 candidate score 直接反映 future distribution 是否向 full support 靠拢；source chunk 只保留为 prior/drift guard。

## 2026-08-02 high-support mid/late state selection smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_fullsupport_prior_softprefix_highsupport_midlate_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_highsupport_midlate_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_fullsupport_prior_softprefix_highsupport_midlate_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_fullsupport_prior_softprefix_highsupport_midlate_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

目的：

- 在 soft-prefix full-support prior 的基础上，只保留 prompt-level support 更强、boundary 更偏中后段的 state。
- 仍然遵守最新 target 语义：full-rollout group support/value 主导 `q_j`；probe 只是估计 future answer distribution；source chunk / support anchor 只做 prior 和 drift guard，不做 hard teacher / score floor。
- 检查“选更高质量的 source/prompt + mid/late boundary”能不能提升 support coverage、降低 OOV。

关键配置：

```text
ttrl.chunk_state_min_prompt_top_mass=0.35
ttrl.chunk_state_min_prompt_valid_answer_coverage=0.75
ttrl.chunk_state_min_prompt_top_margin=0.05
ttrl.chunk_state_max_prompt_answer_entropy=2.2
ttrl.chunk_state_min_boundary=128
ttrl.chunk_state_mid_boundary_min_ratio=0.45
ttrl.chunk_state_mid_boundary_max_ratio=0.85
ttrl.chunk_state_future_support_min_state_coverage=0.50
ttrl.chunk_state_future_support_max_state_oov=0.50
ttrl.chunk_state_future_support_min_state_mean_mass=0.12
ttrl.chunk_state_future_support_min_state_top_margin=0.02
```

运行状态：

- 3-step smoke 完成，无 RuntimeError / Traceback，最后按 smoke 配置打印 `Final validation skipped`。
- 模型、数据、venv 仍为 `/models/Qwen2.5-Math-7B`、`data/MATH-TTT`、`/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200`。
- 本轮继续用前台 `mlx worker login` 跑并 tee 到 raw log，避免后台进程在 login 退出后被清掉。

三步质量汇总：

```text
step  real_state  pad_state  skipped_support  skipped_short  boundary_mean  coverage  oov    source_mass  prompt_top  learnable_keep  raw_positive  label_consistent  num_actor_samples
1     15          1          17               0              592            0.492     0.508  0.546        0.546       0.312           0.231         0.742             120
2     15          1          17               0              528            0.400     0.600  0.610        0.610       0.062           0.218         0.656             120
3     11          5          20               1              544            0.416     0.584  0.658        0.658       0.062           0.224         0.727             88
mean  13.7        2.3        18.0             0.3            554.7          0.436     0.564  0.605        0.605       0.145           0.224         0.708             109.3
```

耗时：

```text
step  gen      chunk_probe  chunk_score  chunk_ref  update_actor
1     43.471   7.693        9.918        5.548      4.884
2     23.428   8.091        7.509        1.616      4.429
3     31.682   8.162        7.612        1.217      3.253
```

结论：

- 这是负结果，不扩 20-step。高 support / mid-late state selection 提高了 source/prompt 质量：`source_mass` 均值从 soft-prefix prompt020 的约 `0.46` 提到 `0.61`，boundary 也确实避开了 prompt-only / early prefix。
- 但它没有提升 target 质量：support coverage 均值只有 `0.436`，比 soft-prefix prompt020 的约 `0.498` 更低；OOV 均值 `0.564`，仍然太高。
- 它还把训练分布压得太窄：每步真实 state 只有 `11-15` 个，`skipped_support_sources=17/17/20`，step 2/3 的 `learnable_keep` 只有 `0.062`。actor update 变快主要来自样本数下降，不是 target 变好。
- 这进一步支持最新判断：主矛盾不是 actor update 慢，也不是 source 侧硬约束不够强，而是“局部短视可判定性”这个约束本身。不能再要求 chunk target 主要由 short-horizon probe 的局部命中、source consistency 或 anchor floor 来定义。

下一步：

- 放弃“短 horizon 局部 answer hit 能判清 chunk 好坏”的训练约束。
- 保留 PowerFlow-style distribution matching、hardfilter/clip4 这类有效工程骨架，但重构 target：
  - full rollout group 先定义 prompt-level answer support / value / coverage；
  - 同一 state 下的 candidate chunk 只通过 future distribution 是否向 full support 靠拢来得分；
  - source chunk / support anchor 只作为 prior / drift guard；
  - low-information state 直接 skip 或 soft downweight：all-negative、高 OOV、低 coverage、support mass 太平、malformed/repeated boxed。
- 下一轮不再继续加 sourcegate / source consistency / short-probe teacher，优先实现 full-rollout-support 主导的 per-state sharpened target，形式仍然是：

```text
q_j ∝ exp(alpha * score_j) * prior_j
score_j = future distribution match / support value gain from full-rollout group
prior_j = source/support-anchor drift guard, not teacher floor
```

## 2026-08-02 posterior support match smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_posterior_support_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_posterior_support_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_posterior_support_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_posterior_support_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

代码改动：

- 新增 `ttrl.chunk_state_future_support_score_type=posterior_support_match`。
- 它不是 short-probe teacher，也不是 source/anchor score floor。做法是先用 full-rollout answer support 对 sparse probe evidence 做 posterior shrinkage，再计算：

```text
score_j = 0.5 * smoothed_support_expected_value
        + 0.5 * smoothed_support_overlap
score_j *= smoothed_transport_affinity
```

- 目标是减少 4 条 probe 偶然 OOV / local hit 对 target 的支配，让 target 更接近 full-rollout group label estimation。

关键配置：

```text
ttrl.chunk_state_future_support_score_type=posterior_support_match
ttrl.chunk_state_future_support_prior_smoothing=16.0
ttrl.chunk_state_future_support_anchor_prior_weight=2.0
ttrl.chunk_state_future_support_source_prior_weight=1.0
ttrl.chunk_state_future_support_keep_mode=soft
ttrl.chunk_state_future_support_soft_weight_floor=0.05
ttrl.chunk_state_powerflow_weight_clip=4.0
ttrl.chunk_state_powerflow_weight_clip_renorm=True
```

运行状态：

- 3-step smoke 跑到 step 3 并打印 `Final validation skipped`。
- raw log 末尾出现一次 Ray/DataLoader worker killed traceback：

```text
RuntimeError: DataLoader worker (...) is killed by signal: Killed.
```

- traceback 出现在 step3 指标输出之后，`mlx worker login` 进程返回 0。记录为尾部清理/worker 异常信号，不作为可扩 20-step 的绿灯。
- 模型、数据、venv 仍为 `/models/Qwen2.5-Math-7B`、`data/MATH-TTT`、`/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200`。
- 启动日志确认 vLLM `attention_config.backend=FLASH_ATTN`、flashinfer autotune、CUDA graph capture、NCCL P2P/NVLS；actor 侧 `use_dynamic_bsz=False`、`use_fused_kernels=True`。

三步质量汇总：

```text
step  real_state  pad_state  skipped_support  boundary_mean  boundary_zero  coverage  oov    score  smooth_E  smooth_overlap  top_margin  target_entropy  weight_max  weight_min  actor_samples
1     29          3          3                456.000        0.188          0.482     0.518  0.474  0.247     0.856           0.014       2.060           0.241       0.076       232
2     24          0          8                554.667        0.042          0.461     0.539  0.478  0.268     0.849           0.013       2.065           0.210       0.072       192
3     25          7          7                488.000        0.031          0.521     0.479  0.468  0.234     0.855           0.011       2.065           0.184       0.069       200
mean  26.0        3.3        6.0              499.556        0.087          0.488     0.512  0.473  0.250     0.853           0.013       2.063           0.212       0.072       208
```

耗时：

```text
step  gen      chunk_probe  chunk_score  chunk_ref  update_actor
1     43.484   9.052        10.407       7.147      8.605
2     23.246   8.922        8.268        2.739      6.920
3     23.855   9.283        9.103        2.837      7.299
```

结论：

- 这是一个有用但不能扩 20-step 的 smoke。
- 有用的部分：`posterior_support_match` 确实解决了“局部短 probe 把 target 打空”的问题。三步都是 `label_consistent_ratio=1.0`、`future_support_keep_ratio=1.0`，没有 all-negative state；`smoothed_support_overlap` 稳定在约 `0.85`。
- 不足的部分：`prior_smoothing=16` 明显过强，target 过平。`target_entropy=2.06` 接近 8 candidates 均匀分布的 `ln(8)=2.079`，`state_top_margin` 只有 `0.011-0.014`，`weight_max` 只有 `0.18-0.24`。这说明模型收到的 chunk preference 太弱，容易退化成“几乎均匀的 conservative distillation”。
- 因此这版不扩 20-step。它验证了新方向：target 应该由 full-rollout support posterior 主导，而不是局部短视命中；但下一步要恢复区分度。

下一步：

- 保留 `posterior_support_match`，把 `prior_smoothing` 从 `16` 降到 `4-8`。
- 加一个 sharpen / margin 机制：例如对 posterior score 做 per-state centering 或温度放大，目标是把 `state_top_margin` 拉到 `0.03+`、`weight_max` 拉到 `0.30+`，同时不让 `label_consistent_ratio` 回到 `0.6-0.7`。
- 不回到 sourcegate / source consistency / short-probe teacher；仍然让 full-rollout support/value 定义 target，probe 只提供 future distribution evidence。

## 2026-08-02 posterior support match sharp a6/s8 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_posterior_support_sharp_a6s8_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_posterior_support_sharp_a6s8_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_posterior_support_sharp_a6s8_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_posterior_support_sharp_a6s8_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

关键配置：

```text
ttrl.chunk_state_future_support_score_type=posterior_support_match
ttrl.chunk_state_future_support_prior_smoothing=8.0
ttrl.chunk_state_alpha=6.0
ttrl.chunk_state_eps=0.01
ttrl.chunk_state_future_support_anchor_prior_weight=2.0
ttrl.chunk_state_future_support_source_prior_weight=1.0
ttrl.chunk_state_future_support_keep_mode=soft
ttrl.chunk_state_future_support_soft_weight_floor=0.05
ttrl.chunk_state_powerflow_weight_clip=4.0
ttrl.chunk_state_powerflow_weight_clip_renorm=True
```

三步质量汇总：

```text
step  real_state  pad_state  boundary_zero  coverage  oov    top_margin  label_consistent  target_entropy  weight_max  actor_samples
1     29          3          0.188          0.482     0.518  0.022       1.000             1.753           0.881       232
2     25          7          0.281          0.489     0.511  0.018       1.000             1.743           0.886       200
3     26          6          0.031          0.561     0.439  0.008       1.000             1.873           0.509       208
mean  26.7        5.3        0.167          0.511     0.489  0.016       1.000             1.790           0.759       213.3
```

耗时：

```text
step  gen      chunk_probe  chunk_score  update_actor
1     43.453   9.223        10.495       8.746
2     23.231   9.360        8.167        7.416
3     22.144   9.497        8.265        7.816
mean  29.609   9.360        8.976        7.993
```

结论：

- 这是正向 smoke，但还不是 20-step 扩展绿灯。
- 正向部分：相对 `prior_smoothing=16`，sharp a6/s8 明显恢复了 per-state target 区分度。`target_entropy` 从约 `2.063` 降到 `1.790`，`weight_max` 从约 `0.212` 提到 `0.759`，同时 `label_consistent_ratio` 仍为 `1.0`，没有退回短 probe/local hit 噪声主导。
- 仍然不过关的部分：`state_top_margin` 均值只有 `0.016`，低于 `0.03+` gate；coverage/OOV 仍在约 `0.51/0.49`，说明 candidate future distribution 对 full-rollout support 的贴合度没有本质改善。sharp 只是把已有 posterior signal 放大了，没有产生更可靠的 support-aligned candidate。
- actor update 约 `8.0s`，训练更新本身不是当前主矛盾；主要耗时仍来自 full rollout、probe、score 和 ref/logprob 链路。更重要的是 target 质量仍未达到可长跑标准。

下一步：

- 不再继续沿 source hard gate、source consistency 或 short-probe teacher 加约束。
- 保留 `posterior_support_match + PowerFlow weighted distillation` 作为骨架，但要改 candidate/state 侧：优先做更可靠的 full-support future estimator，例如 state-compatible staged longer-horizon proposal、按 full-rollout support 高 coverage / top margin 选择 state、以及对低 support/OOV candidate 做 soft downweight 或跳过。
- 20-step 前的最小 gate 仍然是：`state_top_margin > 0.03`，`coverage` 明显高于 `0.55` 或 `OOV` 明显低于 `0.45`，同时 `label_consistent_ratio` 不坍缩。

## 2026-08-02 posterior support quality-gate a6/s8 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_posterior_support_qualitygate_a6s8_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_posterior_support_qualitygate_a6s8_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_posterior_support_qualitygate_a6s8_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_posterior_support_qualitygate_a6s8_prompt020_productweight_nosrcgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

设计动机：

- 上一轮 `posterior_support_match sharp a6/s8` 证明 target 不再过平，但 support coverage / OOV 未改善。
- 本轮只加 full-support future distribution 质量门控，不引入 sourcegate、source consistency 或 short-probe local teacher。
- 目标是验证：事后过滤 low-support candidate/state，能否在不改变 target 语义的情况下同时提高 `state_top_margin` 与 support alignment。

关键配置：

```text
ttrl.chunk_state_future_support_score_type=posterior_support_match
ttrl.chunk_state_future_support_prior_smoothing=8.0
ttrl.chunk_state_alpha=6.0
ttrl.chunk_state_eps=0.01
ttrl.chunk_state_future_support_min_candidate_coverage=0.25
ttrl.chunk_state_future_support_min_state_coverage=0.50
ttrl.chunk_state_future_support_max_state_oov=0.50
ttrl.chunk_state_future_support_min_state_max_mass=0.35
ttrl.chunk_state_future_support_keep_mode=soft
ttrl.chunk_state_future_support_soft_weight_floor=0.05
```

三步质量汇总：

```text
step  real_state  actor_samples  coverage  oov    cand_keep  learnable_keep  label_consistent  top_margin  target_entropy  weight_max  all_pos  all_neg
1     29          232            0.482     0.518  0.699      0.531           0.699             0.037       1.477           1.000       0.188    0.000
2     26          208            0.352     0.648  0.531      0.281           0.531             0.078       1.376           1.000       0.156    0.156
3     24          192            0.376     0.624  0.646      0.208           0.646             0.035       1.466           1.000       0.292    0.042
mean  26.3        210.7          0.403     0.597  0.625      0.340           0.625             0.050       1.440           1.000       0.212    0.066
```

耗时：

```text
step  gen      chunk_probe  chunk_score  update_actor
1     43.491   9.365        10.283       9.123
2     23.347   10.862       8.670        7.567
3     22.844   8.913        7.860        6.957
mean  29.894   9.713        8.938        7.882
```

结论：

- 这是负结果，不扩 20-step。
- 正向部分：事后 quality gate 确实把 `state_top_margin` 拉到了 `0.035-0.078`，三步都超过 `0.03` gate；target entropy 进一步降到约 `1.44`。
- 关键问题：它没有提升 support alignment，反而让 coverage/OOV 从上一轮 sharp 的约 `0.511/0.489` 恶化到 `0.403/0.597`。这说明 target 变尖主要来自把低 coverage candidate 置零，而不是 candidate future distribution 真正靠近 full-rollout support。
- 训练信号变窄：`label_consistent_ratio` 只有 `0.53-0.70`，`learnable_state_keep_ratio` 掉到 `0.21-0.53`，step2 还出现 `state_all_negative_ratio=0.156`。这已经接近“用硬过滤制造局部 teacher”，不符合最新约束。
- 因此不能把这版作为 20-step 绿灯。它是一个有用反证：post-hoc quality gate 可以制造 margin，但不能解决 support coverage / OOV 主矛盾。

下一步：

- 不再继续单纯加事后 gate 或调 sharpness。
- 需要改 candidate/probe 生成本身，让 candidate future distribution 更容易进入 full-rollout support：
  - 方案 A：state-compatible longer-horizon estimator。对同一 state 的全部 candidates 做更长 horizon probe，减少 4 条 probe 的 OOV 偶然性。
  - 方案 B：support-aware proposal prior。不是直接 copy high-support continuation，而是在当前 state 下用 high-support answer support 引导 resampling / continuation probe，再做 posterior support match。
  - 方案 C：两阶段宽后深，但第一阶段不能用 short-probe local hit 决定 teacher；只能用 full-support posterior 的 coarse transport score 分配更多 probe budget。
- 下一轮 gate 应同时要求：`top_margin > 0.03`，`coverage >= 0.50`，`OOV <= 0.50`，`label_consistent_ratio >= 0.75`，否则不扩 20-step。

## 2026-08-02 posterior support longprobe2560 a6/s8 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_posterior_support_longprobe2560_a6s8_prompt020_productweight_nosrcgate_mid_c128_probe2560x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_posterior_support_longprobe2560_a6s8_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_posterior_support_longprobe2560_a6s8_prompt020_productweight_nosrcgate_mid_c128_probe2560x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_posterior_support_longprobe2560_a6s8_prompt020_productweight_nosrcgate_mid_c128_probe2560x4_b32_r32_v64_3step_20260802.jsonl`

设计动机：

- quality-gate 版本证明 post-hoc 过滤只能制造 margin，不能解决 support coverage / OOV。
- 本轮取消 staged probe 和 quality gate，对全部 candidate 使用更长 horizon 的 future distribution estimator，检查更长 probe 能否让 candidate future distribution 自然靠近 full-rollout support。
- 这轮仍保留 PowerFlow weighted distillation、posterior support match、source/anchor 作为 prior / drift guard，但不让 source hard gate 或 local answer hit 主导 target。

关键配置：

```text
ttrl.chunk_state_future_support_score_type=posterior_support_match
ttrl.chunk_state_probe_max_tokens=2560
ttrl.chunk_state_probe_samples=4
ttrl.chunk_state_staged_probe_enable=False
ttrl.chunk_state_future_support_prior_smoothing=8.0
ttrl.chunk_state_alpha=6.0
ttrl.chunk_state_eps=0.01
ttrl.chunk_state_future_support_min_candidate_coverage=0.0
ttrl.chunk_state_future_support_min_state_coverage=0.0
ttrl.chunk_state_future_support_max_state_oov=1.0
ttrl.chunk_state_future_support_keep_mode=soft
ttrl.chunk_state_future_support_soft_weight_floor=0.05
```

三步质量汇总：

```text
step  real_state  actor_samples  coverage  oov    top_margin  label_consistent  target_entropy  weight_max
1     29          232            0.466     0.534  0.020       1.000             1.744           0.826
2     27          216            0.399     0.601  0.012       1.000             1.903           0.417
3     23          184            0.384     0.616  0.028       1.000             1.844           0.720
mean  26.3        210.7          0.416     0.584  0.020       1.000             1.830           0.654
```

耗时：

```text
step  gen      chunk_probe  chunk_score  chunk_ref  update_actor
1     43.475   14.178       10.703       7.141      8.654
2     22.609   14.117       10.783       3.034      7.800
3     22.838   14.548       8.418        2.664      6.932
mean  29.641   14.281       9.968        4.280      7.795
```

结论：

- 这是负结果，不扩 20-step。
- 更长 horizon probe 没有改善 full-support alignment：三步均值 coverage 只有 `0.416`，OOV `0.584`，比 sharp a6/s8 的 `0.511/0.489` 明显更差，也没有达到 `coverage >= 0.50` / `OOV <= 0.50` gate。
- `label_consistent_ratio=1.0` 说明它没有像 quality-gate 那样直接坍缩到硬过滤，但 `state_top_margin` 均值只有 `0.020`，target 仍然不够可分。
- 耗时也更重：`chunk_state_probe` 从 sharp 版约 `9.36s` 增到 `14.28s`，而 target 质量下降。说明单纯拉长 probe 不是有效解。
- 退出期出现 `RuntimeError: DataLoader worker ... killed by signal: Killed`，发生在三步指标全部打印、`Final validation skipped` 后，不影响这次 smoke 的三步诊断。

更新后的判断：

- 需要优先放弃“局部短视可判定性”这个约束，而不只是继续拉长 probe 或加 gate。
- full rollout group 要先定义 prompt-level answer support / value；chunk candidate 的目标应是相对这个 support 的 future distribution improvement。
- 下一步不再把 short-horizon local hit、source consistency 或 post-hoc hard gate 当 teacher；更合理的方向是先提高 state/candidate proposal 的 support compatibility，再用 posterior support / transport-style target 做软分布蒸馏。

## 2026-08-02 support_flow softplus_gain smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_support_flow_suffix_softplusgain_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_support_flow_suffix_softplusgain_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_suffix_softplusgain_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_suffix_softplusgain_mid_c128_b32_r32_v64_3step_20260802.jsonl`

设计动机：

- `support_flow soft_mass` 已经证明可以完全绕开 short probe，但 20-step 失败，核心问题是它更像 full-support replay，不是 transition improvement。
- 本轮新增 `support_flow_score_type=softplus_gain`：用 `sigmoid((anchor_mass - source_mass * baseline_scale + gain_slack) / temperature)` 形成软 improvement target。
- 目标是保留 full-rollout support 作为 teacher，同时避免 hard gain 过稀疏，也避免 soft_mass 无条件 replay。

代码与配置变更：

```text
ray_trainer.py:
  新增 ttrl.chunk_state_support_flow_score_type=softplus_gain
  新增 ttrl.chunk_state_support_flow_softplus_temperature

ppo_trainer_ttrl.yaml:
  新增 chunk_state_support_flow_softplus_temperature: 0.125
```

三步质量汇总：

```text
step  real_state  actor_samples  anchor_mass  source_mass  pos_margin  score_mean  score_max  ans_cov  entropy  weight_max  grad_norm
1     11          88             0.210        0.520        -0.077      0.195       0.415      0.852    1.399    0.935       66.301
2     6           48             0.385        0.646        -0.135      0.288       0.381      0.828    1.788    0.933       44.807
3     13          104            0.198        0.528        -0.218      0.179       0.266      0.789    1.848    0.486       87.637
mean  10.0        80.0           0.264        0.565        -0.143      0.221       0.354      0.823    1.678    0.785       66.248
```

耗时：

```text
step  gen      chunk_score  chunk_ref  update_actor
1     43.561   7.626        5.193      3.591
2     22.112   6.025        0.695      1.897
3     22.720   5.749        1.480      4.056
mean  29.464   6.467        2.456      3.181
```

结论：

- 工程正结果：`chunk_state_probe/skipped_for_support_flow=1.0`，`score_type_softplus_gain=1.0`，PowerFlow actor path 正常。actor update 三步均值约 `3.18s`，明显快于 posterior-support probe 系列的 `7-8s`。
- 方法信号仍不够，暂不扩 20-step。`positive_margin_mean` 三步全负，均值约 `-0.143`，说明 anchor mass 平均仍低于 selected source mass；这仍然更像 conservative full-support distillation，而不是 search-state improvement。
- `answer_coverage_mean=0.823` 比 probe 系列高，说明 full-support anchor proposal 覆盖是健康的；但 `real_state` 均值只有 `10/32` 左右，`num_actor_samples` 均值只有 `80`，训练信号偏窄。这主要来自当前 `source_select_by_mass + min_source_answer_mass=0.40 + majority_consistent` 过强。
- 不应继续只调 temperature。温度可以改变 target sharpness，但不能把负 margin 变成真正的 improvement。

下一步：

- 保留 `support_flow` / no-probe / PowerFlow chunk update 这条快链路。
- 下一版不要继续把 source 选成高 mass 成功轨迹再要求 anchor 超过 source；这会天然让 positive margin 为负。应改成 paired state 或 mixed-source state：
  - 同 prompt 同 boundary 下，同时采 high-support source 和 lower-support/fail source；
  - full-rollout support anchor 仍定义 target；
  - 对低 source-mass state 做 improvement distillation，对高 source-mass state 只做 drift guard 或降权。
- 3-step gate 改为：`probe skipped=1`，`answer_coverage>=0.75`，`positive_margin_mean` 接近 0 或为正，`real_state>=20/32`，`actor update <5s`。不过 gate 才扩 20-step。

## 2026-08-02 support_flow mixed-source softplus_gain smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_support_flow_mixedsource_softplusgain_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_support_flow_mixedsource_softplusgain_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_mixedsource_softplusgain_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_mixedsource_softplusgain_mid_c128_b32_r32_v64_3step_20260802.jsonl`

设计动机：

- 上一版 `softplus_gain + majority_consistent/high-source` 的 `positive_margin_mean=-0.143`，不是因为 support_flow loss 不可用，而是因为 state source 被强行选成高 answer-mass 轨迹，baseline 天然过强。
- 本轮新增 `chunk_state_source_mode=support_mixed`，让 state source 来自同 prompt 内 lower-source-mass / mixed source；full-rollout answer support anchors 仍然定义 target，source chunk 只作为 candidate prior / drift guard。
- 这轮完全跳过 short-horizon probe，验证“full group 定义好 support/value，低 support state 学向高 support future distribution 转移”的快链路是否成立。

代码与配置变更：

```text
ray_trainer.py:
  新增 chunk_state_source_mode=support_low / support_mixed
  新增 lower-source state selection:
    chunk_state_source_mixed_low_ratio
    chunk_state_source_low_max_answer_mass
    chunk_state_source_min_valid_answer_mass
  新增 chunk_state_diag/support_low_fallbacks

ppo_trainer_ttrl.yaml:
  新增上述 source selection 默认配置，默认只在显式 opt-in source_mode 下生效。
```

关键配置：

```text
ttrl.chunk_state_score_mode=support_flow
ttrl.chunk_state_source_mode=support_mixed
ttrl.chunk_state_source_mixed_low_ratio=0.75
ttrl.chunk_state_source_low_max_answer_mass=0.35
ttrl.chunk_state_source_min_valid_answer_mass=0.03125
ttrl.chunk_state_min_prompt_top_mass=0.35
ttrl.chunk_state_min_source_answer_mass=0.0
ttrl.chunk_state_support_flow_score_type=softplus_gain
ttrl.chunk_state_support_flow_softplus_temperature=0.125
ttrl.chunk_state_support_anchor_count=7
ttrl.chunk_state_support_anchor_candidate_start=1
ttrl.chunk_state_source_chunk_enable=True
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_prune_zero_weight_samples=True
ttrl.chunk_state_powerflow_weight_clip=4.0
actor_rollout_ref.actor.use_dynamic_bsz=False
```

三步质量汇总：

```text
step  real_state  actor_samples  source_mass  pos_margin  ans_cov  anchor_inject  target_entropy
1     22          176            0.090        0.206       0.729    0.833          1.975
2     17          136            0.039        0.191       0.573    0.655          2.026
3     15          120            0.081        0.201       0.719    0.821          1.995
mean  18.0        144.0          0.070        0.199       0.674    0.770          1.999
```

耗时：

```text
step  gen      chunk_score  chunk_ref  update_actor
1     43.476   7.368        6.577      6.874
2     23.012   5.788        2.015      5.041
3     32.349   5.792        1.799      4.621
mean  32.946   6.316        3.464      5.512
```

结论：

- 这是方向上的正结果，但暂不扩 20-step。`source_mass_mean` 从上一版 `0.565` 降到 `0.070`，`positive_margin_mean` 从 `-0.143` 变成 `+0.199`，直接验证了当前主矛盾是 high-source baseline / 局部短视 teacher 约束，而不是 actor update。
- `chunk_state_probe/skipped_for_support_flow=1.0`，说明这轮没有再用 short-horizon probe/local answer hit 定义 target；teacher 来自 full-rollout support anchors。
- 工程链路仍可接受：`update_actor` 三步均值 `5.51s`，第 2/3 step 已降到 `5.04s/4.62s`。B200 上这条 no-probe PowerFlow chunk update 是可继续优化的快链路。
- 仍未过 gate：`real_state` 均值只有 `18/32`，`answer_coverage_mean=0.674`，第 2 step coverage 只有 `0.573`。原因主要是 `min_prompt_top_mass=0.35` 跳过了 10/15/17 个 prompt，且 `candidates=8` 中一个 slot 用 source chunk，support anchors 只有 7 个。

下一步：

- 不回退到 short probe teacher，也不继续加 source-side hard gate。
- 下一轮做更温和的 mixed-source：降低 prompt top-mass gate 到 `0.30` 或 `0.25` 以提高 real states；同时把 `candidates` 增到 `12`、support anchors 增到 `11`，用 B200 显存换更高 full-support anchor coverage。
- 目标是保持 `positive_margin_mean > 0`，同时把 `real_state >= 20/32`、`answer_coverage >= 0.75`、`update_actor < 6s` 稳住；过这个 3-step gate 才扩 20-step。

## 2026-08-02 support_flow mixed-source dense-anchor smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_support_flow_mixedsource_denseanchor_softplusgain_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_support_flow_mixedsource_denseanchor_softplusgain_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_mixedsource_denseanchor_softplusgain_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_mixedsource_denseanchor_softplusgain_mid_c128_b32_r32_v64_3step_20260802.jsonl`

设计动机：

- 上一版 mixed-source 已经把 `positive_margin_mean` 拉正，但 `real_state` 和 `answer_coverage` 不稳。
- 本轮不改变 target 语义，仍然使用 full-rollout support anchors + `softplus_gain`，不使用 short probe teacher。
- 只做一个密集 anchor ablation：`min_prompt_top_mass` 从 `0.35` 降到 `0.30`，`candidates` 从 `8` 增到 `12`，support anchors 从 `7` 增到 `11`，检查 B200 大显存能否直接换来更高 support coverage。

关键配置差异：

```text
ttrl.chunk_state_min_prompt_top_mass=0.30
ttrl.chunk_state_candidates=12
ttrl.chunk_state_support_anchor_count=11
ttrl.chunk_state_support_anchor_candidate_start=1
```

三步质量汇总：

```text
step  real_state  actor_samples  source_mass  pos_margin  ans_cov  anchor_inject  target_entropy
1     24          288            0.092        0.216       0.663    0.723          2.349
2     21          248            0.086        0.304       0.653    0.712          2.335
3     19          224            0.078        0.315       0.663    0.723          2.320
mean  21.3        253.3          0.085        0.278       0.660    0.719          2.335
```

耗时：

```text
step  gen      chunk_score  chunk_ref  update_actor
1     43.432   7.455        8.202      11.015
2     22.806   5.902        3.642      9.035
3     23.468   5.917        3.341      8.257
mean  29.902   6.425        5.062      9.436
```

结论：

- 这是负向 ablation，不扩 20-step。
- 降低 prompt gate 后 `real_state` 从上一版均值 `18.0` 提到 `21.3`，说明有效 state 数可以通过 prompt gate 调整改善。
- 但增加 candidate / support anchor 没有提高 support coverage：`answer_coverage_mean` 从上一版 `0.674` 降到 `0.660`，`anchor_injected_ratio` 也从 `0.770` 降到 `0.719`。原因是同 prompt 的 valid support answer 数有限，增加 slots 只会放入更多低 mass anchors 或空位。
- 代价明显变大：`num_actor_samples` 从 `144` 增到 `253`，`update_actor` 从 `5.51s` 增到 `9.44s`。这违背当前 fast-chain 目标。

下一步：

- 保留上一版 `candidates=8 / anchors=7` 快链路，不继续靠更多 anchors 硬堆 coverage。
- 可以只降低 prompt gate 到 `0.30`，保持 8 candidates，验证能否在不显著拖慢 actor update 的情况下把 real states 提到 20+。
- 另一条更关键的设计方向是改 support anchor 选择/score，而不是数量：例如对 anchors 做 answer-level 去重、按 top answer support 分布分层采样，或者把 prompt-level support mass 转成 per-answer target distribution 后再投到 chunk anchors，避免 slots 被同答案/同质低增益 anchor 浪费。

## 2026-08-02 support_flow mixed-source gate0.30 fast smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_support_flow_mixedsource_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_support_flow_mixedsource_gate030_softplusgain_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_mixedsource_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_mixedsource_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.jsonl`

设计动机：

- dense-anchor 证明靠增加 candidate / anchor 数量会显著拖慢 actor update，且没有提高 full-support answer coverage。
- 本轮回到 `candidates=8 / support_anchor_count=7` 快链路，只把 `min_prompt_top_mass` 从 `0.35` 降到 `0.30`，验证能否在不引入更多 actor samples 的情况下提高有效 state 数。
- 仍然坚持 full-rollout support/value 定义 target：`chunk_state_probe/skipped_for_support_flow=1.0`，不使用 short-horizon probe/local answer hit/source consistency 作为 teacher。

关键配置差异：

```text
ttrl.chunk_state_min_prompt_top_mass=0.30
ttrl.chunk_state_candidates=8
ttrl.chunk_state_support_anchor_count=7
ttrl.chunk_state_support_anchor_candidate_start=1
```

三步质量汇总：

```text
step  real_state  actor_samples  source_mass  pos_margin  ans_cov  anchor_inject  target_entropy
1     24          192            0.092        0.216       0.750    0.857          1.962
2     17          136            0.062        0.215       0.656    0.750          1.976
3     20          160            0.058        0.355       0.693    0.792          1.949
mean  20.3        162.7          0.071        0.262       0.700    0.800          1.962
```

耗时：

```text
step  gen      update_actor
1     43.575   7.405
2     23.176   5.110
3     23.537   6.003
mean  30.096   6.173
```

结论：

- 不扩 20-step。`positive_margin_mean=0.262` 是正的，说明 mixed lower-source + support-flow gain 的方向仍然成立；但 `answer_coverage_mean=0.700`，低于进入 20-step 的 `0.75` 门槛。
- `real_state` 均值从上一版 `18.0` 提到 `20.3`，说明降低 prompt gate 有帮助；但第 2 step 仍只有 `17/32`，state selection 还不稳定。
- `update_actor` 均值 `6.17s`，比 dense-anchor 的 `9.44s` 明显好，但仍略高于当前希望的 `<6s` fast-chain 目标。第 2/3 step 已接近可接受区间。
- 这次结果进一步支持最新判断：主矛盾不是 actor update 本身，而是 target support 质量。继续用 short-horizon probe/local answer hit 去判定 chunk 好坏会回到噪声 teacher；继续加 source hard gate 也会恶化 coverage/OOV。

下一步：

- 不再要求 chunk target 主要由短 probe 的局部命中信号定义；probe 只能作为 future answer distribution estimator。
- full rollout group 先定义 prompt-level support/value；chunk 学的是哪个 local transition 会把未来分布推向这个 support/value。
- 保留 source chunk 作为 prior / drift guard，但不作为 hard floor 或主要 teacher。
- 优先改 support anchor selection/score：answer-level 去重、按 prompt answer support 分层采样、把 prompt-level answer support distribution 投影到 anchors，减少同答案/低增益 anchor 占用 slot。

## 2026-08-02 support_flow answer-projection gate0.30 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_support_flow_answerproj_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_support_flow_answerproj_gate030_softplusgain_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_answerproj_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_answerproj_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.jsonl`

代码改动：

- 新增 opt-in 配置 `ttrl.chunk_state_support_anchor_selection_mode=answer_stratified`。
  - `mass_ranked` 保持原行为。
  - `answer_stratified` 先覆盖不同 full-rollout support answer，再用 mass-ranked anchors 补满剩余 slot。
- 新增 opt-in 配置 `ttrl.chunk_state_support_flow_split_mass_by_answer=True`。
  - 同一个 answer 的多个 anchor 共享该 answer 的 prompt-level support mass。
  - 目的不是做短 probe teacher，而是把 full-rollout answer support distribution 投影到 chunk anchors，避免重复 chunk 重复计权。

关键配置差异：

```text
ttrl.chunk_state_support_anchor_selection_mode=answer_stratified
ttrl.chunk_state_support_flow_split_mass_by_answer=True
ttrl.chunk_state_min_prompt_top_mass=0.30
ttrl.chunk_state_candidates=8
ttrl.chunk_state_support_anchor_count=7
```

三步质量汇总：

```text
step  real_state  actor_samples  source_mass  pos_margin  ans_cov  anchor_inject  uniq_ans  dup_ans  target_entropy
1     24          192            0.092        0.183       0.729    0.857          0.978     0.023    1.933
2     17          136            0.065        0.272       0.682    0.798          0.856     0.130    1.893
3     19          152            0.063        0.303       0.557    0.649          0.975     0.025    1.879
mean  20.0        160.0          0.073        0.253       0.656    0.768          0.936     0.059    1.902
```

耗时：

```text
step  gen      update_actor
1     43.541   7.344
2     32.693   5.128
3     23.048   5.491
mean  33.094   5.988
```

结论：

- 不扩 20-step。`answer-projection` 达到了预期的去重效果：`unique_answer_ratio=0.936`，`answer_duplicate_ratio=0.059`，说明 target 不再由同答案重复 anchors 隐式放大。
- 但质量 gate 没过：`answer_coverage_mean=0.656`，低于 gate0.30 fast smoke 的 `0.700`，更低于扩 20-step 的 `0.75` 门槛。第 3 step coverage 只有 `0.557`。
- `positive_margin_mean=0.253` 仍为正，说明 lower-source search-improvement 方向还成立；`update_actor=5.99s` 已回到 fast-chain 边界。
- 这次 smoke 说明“重复答案计权”确实是一个语义问题，但不是当前最大瓶颈。只做 answer-level 去重会让 target 更干净，同时也会暴露 full group support answer 覆盖不足的问题。

下一步：

- 不回到 short-horizon probe/local answer hit teacher，也不把 source chunk 变成 hard teacher。
- 需要提高 support coverage 的稳定性，而不是继续增加 anchor 数量：优先考虑 state selection 对 `prompt_valid_answer_coverage`、`prompt_answer_entropy`、`prompt_top_margin` 加软权重或轻量 gate，跳过 full group support 本身信息不足的 state。
- 另一个方向是保留 answer projection，但不要硬去重 anchor proposal：proposal 可 mass-ranked 保持 coverage，score 侧按 answer split mass，验证是不是能同时保留 coverage 和正确的 answer-level target semantics。

## 2026-08-02 support_flow mass-proposal answer-split gate0.30 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_support_flow_massprop_answersplit_gate030_softplusgain_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.jsonl`

设计动机：

- 上一轮 `answer_stratified + split_mass_by_answer` 让 target 语义更干净，但 `answer_coverage_mean` 降到 `0.656`。
- 本轮只保留 score 侧 `split_mass_by_answer=True`，anchor proposal 回到 `mass_ranked`，验证是否能同时保持 coverage 和 answer-level target semantics。
- 仍然不使用 short-horizon probe/local answer hit/source consistency 作为 teacher：`chunk_state_probe/skipped_for_support_flow=1.0`。

关键配置差异：

```text
ttrl.chunk_state_support_anchor_selection_mode=mass_ranked
ttrl.chunk_state_support_flow_split_mass_by_answer=True
ttrl.chunk_state_min_prompt_top_mass=0.30
ttrl.chunk_state_candidates=8
ttrl.chunk_state_support_anchor_count=7
```

三步质量汇总：

```text
step  real_state  actor_samples  source_mass  pos_margin  ans_cov  anchor_inject  uniq_ans  dup_ans  target_entropy
1     24          192            0.092        0.068       0.734    0.857          0.805     0.196    2.019
2     17          136            0.059        0.176       0.719    0.845          0.741     0.261    1.995
3     20          160            0.103        0.125       0.667    0.774          0.845     0.156    1.999
mean  20.3        162.7          0.085        0.123       0.707    0.825          0.797     0.204    2.004
```

耗时：

```text
step  gen      update_actor
1     43.559   7.584
2     32.810   5.182
3     22.738   5.890
mean  33.036   6.219
```

结论：

- 不扩 20-step。`answer_coverage_mean=0.707`，比 answer-stratified 的 `0.656` 高，也略高于 gate0.30 fast smoke 的 `0.700`；但仍低于扩展门槛 `0.75`。
- `positive_margin_mean=0.123` 明显弱于 gate0.30 fast smoke 的 `0.262` 和 answer-stratified 的 `0.253`。这说明只在 score 侧按 answer split mass 会把 duplicated high-mass answer 的虚高增益压掉，但也削弱了 search-improvement signal。
- `unique_answer_ratio=0.797`、`answer_duplicate_ratio=0.204`，介于 gate0.30 原始 proposal 和 answer-stratified 之间。proposal coverage 确实回来了一些，但 target margin 不够。
- `weight_max_mean=0.395`，比 answer-stratified 的 `0.811` 更平，target entropy 也更高。这更像 conservative answer-distribution replay，不像强 policy improvement。

下一步：

- 这轮说明 target 设计要同时满足两个条件：保留足够 support coverage，并保留足够 improvement margin。单纯 answer split 会过度抹平 gain。
- 更合理的下一个变量是对 split 后的 answer-level mass 加温度/幂次重新锐化，或只对同答案 duplicates 做部分分摊，例如 `projected_mass = answer_mass / count^gamma`，`gamma < 1`，在不重复计权的前提下保留 high-support transition 的 margin。
- state selection 也需要更直接地过滤低信息 prompt：把 `prompt_valid_answer_coverage`、`prompt_answer_top_margin`、`prompt_answer_entropy` 纳入轻量 gate/soft weight，避免 support 本身太平或太散的 state 主导更新。

## 2026-08-02 support_flow mass-proposal fractional answer-split gate0.30 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_support_flow_massprop_answersplit05_gate030_softplusgain_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.jsonl`

代码改动：

- 新增 opt-in 配置 `ttrl.chunk_state_support_flow_answer_split_power`，默认 `1.0` 保持 full answer split。
- 当 `split_mass_by_answer=True` 时，同答案 anchors 的质量由 `answer_mass / count^power` 得到：
  - `power=1.0`: 完全按答案去重分摊。
  - `power=0.5`: 对重复答案做部分惩罚，保留一部分 high-support transition margin。
  - `power=0.0`: 不分摊，接近原始 mass-ranked anchor mass。

关键配置差异：

```text
ttrl.chunk_state_support_anchor_selection_mode=mass_ranked
ttrl.chunk_state_support_flow_split_mass_by_answer=True
ttrl.chunk_state_support_flow_answer_split_power=0.5
ttrl.chunk_state_min_prompt_top_mass=0.30
ttrl.chunk_state_candidates=8
ttrl.chunk_state_support_anchor_count=7
```

三步质量汇总：

```text
step  real_state  actor_samples  source_mass  pos_margin  ans_cov  anchor_inject  uniq_ans  dup_ans  target_entropy
1     24          192            0.092        0.116       0.734    0.857          0.805     0.196    1.993
2     19          152            0.093        0.137       0.599    0.685          0.882     0.118    1.917
3     17          136            0.076        0.219       0.776    0.946          0.725     0.292    1.929
mean  20.0        160.0          0.087        0.157       0.703    0.829          0.804     0.202    1.946
```

耗时：

```text
step  gen      chunks  score   ref     update_actor
1     43.471   0.974   7.354   6.827   7.642
2     23.182   0.978   5.732   2.280   5.915
3     21.955   1.092   5.705   1.984   4.965
mean  29.536   1.015   6.264   3.697   6.174
```

结论：

- 不扩 20-step。均值 `answer_coverage_mean=0.703`、`positive_margin_mean=0.157`、`real_states=20.0`、`update_actor=6.17s`，没有达到扩展 gate。
- 但这轮比 full split 更健康：`positive_margin_mean` 从 `0.123` 回到 `0.157`，第 3 step 达到 `answer_coverage=0.776`、`positive_margin=0.219`、`update_actor=4.97s`，说明 `power=0.5` 能部分修复 full split 过度抹平 improvement signal 的问题。
- 仍然不能把这条线直接扩 20-step，因为均值 coverage 还只有 `0.703`，第 2 step coverage 掉到 `0.599`。当前主要问题不是 actor update 速度，而是 full group support 在部分 state 上信息不足或过散，导致 target 不稳定。
- 这轮再次支持最新判断：不要再要求 chunk target 由 short-horizon probe/local answer hit/source consistency 在局部短视条件下判清楚。当前版本 `chunk_state_probe/skipped_for_support_flow=1.0`，probe 只作为 future distribution estimator，方向是对的；下一步应该加强 full-rollout group distribution 的 state quality control。

下一步：

- 不继续加 source-side hard gate，不回到 source chunk hard teacher，也不让短 probe 命中率主导 target。
- 优先实现基于 full rollout group 的 state 质量 gate/soft weight：
  - `prompt_valid_answer_coverage` 过低的 state 跳过或降权。
  - `prompt_answer_top_margin` 太低、support 太平的 state 跳过或降权。
  - `prompt_answer_entropy` 过高的 state 跳过或降权。
- 目标是保留 `power=0.5` 的 partial answer split，同时提高可训练 state 的 support coverage 稳定性，再跑 3-step smoke；只有 `answer_coverage >= 0.75` 且 `positive_margin > 0.20` 稳定后再扩 20-step。

## 2026-08-02 support_flow fractional answer-split + group-quality weight smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_groupq_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_support_flow_massprop_answersplit05_groupq_gate030_softplusgain_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_groupq_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_groupq_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.jsonl`

代码改动：

- 扩展 `ttrl.chunk_state_source_quality_weight_mode`，新增基于 full-rollout group metadata 的 opt-in 权重：
  - `prompt_valid_answer_coverage`
  - `prompt_top_margin`
  - `coverage_margin_product`
  - `group_quality`
- 本轮使用 `coverage_margin_product`，不是 hard filter：低 coverage / 低 top-margin 的 state 降权，但不直接丢弃，避免再次把 actor batch 打得太碎。
- 仍保持 `chunk_state_probe/skipped_for_support_flow=1.0`，不让 short-horizon probe/local answer hit/source consistency 主导 target。

关键配置差异：

```text
ttrl.chunk_state_support_flow_answer_split_power=0.5
ttrl.chunk_state_source_quality_weight_mode=coverage_margin_product
ttrl.chunk_state_source_quality_weight_floor=0.25
ttrl.chunk_state_source_quality_weight_power=0.5
```

三步质量汇总：

```text
step  real_state  actor_samples  src_q_w  source_mass  pos_margin  ans_cov  anchor_inject  uniq_ans  dup_ans  target_entropy
1     24          192            0.583    0.092        0.116       0.734    0.857          0.805     0.196    1.993
2     16          128            0.593    0.067        0.120       0.703    0.821          0.922     0.081    2.011
3     18          144            0.527    0.077        0.204       0.781    0.917          0.809     0.186    1.938
mean  19.3        154.7          0.568    0.079        0.147       0.739    0.865          0.845     0.154    1.981
```

耗时：

```text
step  gen      chunks  score   ref     update_actor
1     43.516   0.968   7.379   6.798   7.509
2     32.528   0.941   5.860   1.875   4.836
3     22.425   0.976   5.966   2.099   5.327
mean  32.823   0.962   6.402   3.591   5.891
```

结论：

- 不扩 20-step。`answer_coverage_mean=0.739`，已经明显好于上一轮 `0.703`，但还没有稳定超过 `0.75`；`positive_margin_mean=0.147` 仍低于希望的 `0.20`。
- 这轮是正向信号：第 2 step 的 coverage 从上一轮 `0.599` 拉到 `0.703`，第 3 step 达到 `answer_coverage=0.781`、`positive_margin=0.204`，说明 full-group quality soft weight 在减少低信息 state 干扰。
- `update_actor_mean=5.89s`，训练更新不是主矛盾；target 质量仍是主矛盾。
- `source_quality_weight_mean=0.568`，说明当前 weight 主要是降权而不是过滤。这个符合“full rollout group 先定义 support/value，chunk 只学习把未来分布推向好答案”的方向。

下一步：

- 需要补充 quality-weighted target 诊断，而不是只看未加权的 `answer_coverage_mean/positive_margin_mean`。当前 actor 实际看到的是 `source_quality_weight * target weight`，但日志里还没有加权 coverage/margin。
- 如果加权后的 coverage/margin 已经达标，可以用这版扩 20-step；如果仍不够，下一版优先尝试：
  - `group_quality`，加入 entropy penalty。
  - 或轻量 hard gate：`chunk_state_min_prompt_top_margin` / `chunk_state_min_prompt_valid_answer_coverage`，只跳过 full group support 明显太散的 prompt。
  - 不回到 short-horizon probe teacher，不增加 source-side hard teacher。

## 2026-08-02 support_flow fractional answer-split + state-quality gate smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_stateq_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_support_flow_massprop_answersplit05_stateq_gate030_softplusgain_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_stateq_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_stateq_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.jsonl`

代码改动：

- `support_flow` scorer 额外写出 `chunk_state_positive_margin`。
- actor batch builder 新增 `chunk_state_min_state_positive_margin`，默认 `-1.0` 关闭。
- 新增诊断指标：
  - `chunk_state/answer_coverage_weighted_mean`
  - `chunk_state/positive_margin_mean`
  - `chunk_state/positive_margin_weighted_mean`

关键配置差异：

```text
ttrl.chunk_state_source_quality_weight_mode=off
ttrl.chunk_state_min_answer_coverage=0.75
ttrl.chunk_state_min_state_positive_margin=0.10
ttrl.chunk_state_support_flow_answer_split_power=0.5
```

三步质量汇总：

```text
step  real_state  actor_samples  keep_ratio  raw_cov  w_cov  raw_margin  w_margin  pos_margin_flow  dup_ans
1     24          64             0.333       0.734    0.875  0.116       0.289     0.116            0.196
2     17          88             0.667       0.797    0.830  0.274       0.343     0.274            0.248
3     18          88             0.708       0.797    0.852  0.229       0.360     0.229            0.274
mean  19.7        80             0.569       0.776    0.852  0.206       0.331     0.206            0.239
```

耗时：

```text
step  gen      score   ref     update_actor
1     43.604   7.396   4.822   2.783
2     23.134   5.824   1.224   3.220
3     23.627   5.822   1.306   3.452
mean  30.122   6.347   2.451   3.152
```

结论：

- 质量达标，但样本量偏小，暂不直接扩 20-step。
- 这是目前 target 质量最干净的一版：`answer_coverage_mean=0.776`、`answer_coverage_weighted_mean=0.852`、`positive_margin_mean=0.206`、`positive_margin_weighted_mean=0.331`。
- 训练速度也明显改善：`update_actor_mean=3.15s`，但这是因为 gate 后 `num_actor_samples_mean=80`，不是单纯 infra 提升。
- 样本量风险很明显：第 1 step 只剩 64 个 actor samples，`zeroed_state_ratio=0.667`。如果直接扩 20-step，可能更新过窄、方差偏大。
- 这个实验支持一个更明确的方向：target 质量控制应该基于 full-rollout support-flow 的 state-level coverage/margin，而不是 prompt-level soft metadata，也不是 short-horizon probe/local answer hit。

下一步：

- 先不扩 20-step，做一版放宽 gate 的 tradeoff smoke：
  - 候选 A: `chunk_state_min_answer_coverage=0.70`，`chunk_state_min_state_positive_margin=0.10`。
  - 候选 B: `chunk_state_min_answer_coverage=0.75`，`chunk_state_min_state_positive_margin=0.05`。
- 扩 20-step 的最低门槛建议改为：`num_actor_samples_mean >= 120`，`answer_coverage_weighted_mean >= 0.82`，`positive_margin_weighted_mean >= 0.25`，`update_actor < 5s`。
- 仍然不使用短 probe 局部命中作为 teacher；probe 只做 future distribution estimator，full rollout group support 定义 target。

## 2026-08-02 support_flow fractional answer-split + relaxed state-quality gate smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_stateq70_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_support_flow_massprop_answersplit05_stateq70_gate030_softplusgain_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_stateq70_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_support_flow_massprop_answersplit05_stateq70_gate030_softplusgain_mid_c128_b32_r32_v64_3step_20260802.jsonl`

关键配置差异：

```text
ttrl.chunk_state_source_quality_weight_mode=off
ttrl.chunk_state_min_answer_coverage=0.70
ttrl.chunk_state_min_state_positive_margin=0.10
ttrl.chunk_state_support_flow_answer_split_power=0.5
```

三步质量汇总：

```text
step  real_state  actor_samples  keep_ratio  raw_cov  w_cov  raw_margin  w_margin  dup_ans  target_entropy
1     24          64             0.333       0.734    0.875  0.116       0.289     0.196    1.993
2     17          40             0.500       0.646    0.825  0.335       0.335     0.158    1.903
3     20          80             0.583       0.760    0.837  0.186       0.311     0.257    1.922
mean  20.3        61.3           0.472       0.713    0.846  0.212       0.312     0.204    1.939
```

耗时：

```text
step  gen      score   ref     update_actor
1     43.426   7.355   4.854   2.857
2     24.068   5.838   0.588   1.662
3     22.527   5.921   1.192   3.144
mean  30.007   6.371   2.211   2.554
```

结论：

- 不扩 20-step。虽然 `answer_coverage_weighted_mean=0.846`、`positive_margin_weighted_mean=0.312`、`update_actor_mean=2.55s` 达标，但 `num_actor_samples_mean=61.3` 明显低于最低门槛 `120`。
- 放宽 coverage gate 从 `0.75` 到 `0.70` 没有增加 actor 样本，反而从上一轮均值 `80` 降到 `61.3`，说明瓶颈不是单个 coverage 阈值，而是当前 target 仍只覆盖少数高置信 state。
- 这个结果再次说明：训练更新速度不是主矛盾。hard gate + clip4 能把 actor update 压到 1-3s，但如果 target 只剩很窄的状态集合，就不适合拉长训练。
- 当前主矛盾是 target 语义：不能继续要求 chunk target 由 short-horizon probe/local answer hit/source consistency 这一级短视信号判清楚。short probe 只能作为 future distribution estimator，不能作为 teacher；source chunk 只能作为 prior/drift guard，不能作为主要 teacher/floor。
- run 已产出 3 个 step 的完整指标；尾部 `DataLoader worker ... is killed by signal: Killed` 出现在 Python weakref/退出清理阶段，随后打印 `'Final validation skipped'`，按退出清理 warning 记录，不按训练中断处理。

下一步：

- 放弃“局部短视可判定性”这个约束，回到 full-rollout group support/value 主导 target。
- 先用 32/64 条 full rollout 建 prompt-level answer support、top answer mass、coverage、margin 和 trajectory 中间 state value。
- 从 majority-consistent 或高 support/high pass 的完整轨迹中选中后段 state；低信息 state 直接 skip/downweight，包括 all-negative、高 OOV、support coverage 低、top mass 太平、malformed/repeated boxed/marker 污染。
- candidate chunk 的 score 改成相对 full-group support 的 future distribution improvement，例如 future support mass gain、top answer mass gain、value margin、transport/KL improvement，而不是短 probe 局部答对。
- PowerFlow target 仍保持 per-state sharpened distribution：`q_j ∝ exp(alpha * score_j) * prior_j`，其中 prior 可以来自 source chunk anchor/proposal prior，但不再由 source consistency 决定 teacher。

## 2026-08-02 future_support_gain support-distribution softkeep smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportdist_softkeep_fullsupport_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_supportdist_softkeep_fullsupport_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_supportdist_softkeep_fullsupport_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportdist_softkeep_fullsupport_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

关键配置差异：

```text
ttrl.chunk_state_score_mode=future_support_gain
ttrl.chunk_state_future_support_score_type=support_distribution_match
ttrl.chunk_state_future_support_keep_mode=soft
ttrl.chunk_state_future_support_soft_weight_floor=0.05
ttrl.chunk_state_future_support_min_state_coverage=0.0
ttrl.chunk_state_future_support_max_state_oov=1.0
ttrl.chunk_state_min_answer_coverage=0.0
ttrl.chunk_state_min_informative_gap=0.0
ttrl.chunk_state_prune_zero_weight_samples=False
ttrl.chunk_state_source_quality_weight_mode=group_quality
```

这轮的目的不是继续堆 hard gate，而是验证“full-rollout support distribution 定义 target，probe 只估计 chunk 后续分布，低信息 state 用 soft weight 而不是直接删掉”的可行性。

三步质量汇总：

```text
step  real_state  actor_samples  keep_ratio  support_cov  w_cov  fs_pos_margin  score_mean  label_cons  soft_w  nonzero_w  target_entropy
1     11          128            1.000       0.541        0.580  0.366          0.217       0.742       0.266   0.688      1.701
2     9           128            1.000       0.143        0.349  0.121          0.049       0.250       0.065   0.562      1.876
3     10          128            1.000       0.311        0.606  0.254          0.131       0.430       0.162   0.625      1.655
mean  10          128            1.000       0.332        0.512  0.247          0.132       0.474       0.164   0.625      1.744
```

耗时：

```text
step  gen      probe   score   ref     update_actor
1     43.475   7.684   10.010  5.618   4.720
2     22.950   8.488   7.195   1.947   4.862
3     23.211   7.804   8.804   1.872   4.700
mean  29.879   7.992   8.670   3.146   4.761
```

结论：

- 不扩 20-step。`num_actor_samples_mean=128`、`update_actor_mean=4.76s` 达到样本量和 actor 更新速度门槛，但 target 质量明显不足：`answer_coverage_weighted_mean=0.512`，`support_coverage_mean=0.332`，`label_consistent_ratio=0.474`。
- softkeep 证明了一件事：hardfilter 不是必须的，actor batch 可以保持全量，训练更新也没有炸。但 probe 估计出来的 future distribution 仍然大量落在 full-rollout support 之外，`state_oov_mean=0.668`，所以这版只是解决了“样本太少”，没有解决“future distribution estimator 质量差”。
- 这版符合新约束：不让 short-horizon local hit/source consistency 做 teacher；target 来自 full-rollout answer support distribution。失败点也因此更清楚：当前 `support_distribution_match` 的 4 条 1536-token probe 对数学 answer support 的覆盖太低，不能可靠地产生 improved distribution。
- 注意日志缺陷：通用 `chunk_state/positive_margin_mean` 仍为 `0`，因为 `future_support_gain` 写的是 `chunk_state_future_support_gain/positive_margin_mean`，尚未映射到 `chunk_state_positive_margin`。分析这类 run 时必须看 `chunk_state_future_support_gain/positive_margin_mean`。

下一步：

- 不再扩大这版。下一步要优化的是 future distribution estimator，而不是 actor update。
- 候选 A：把 source 选择从 `majority_consistent + min_source_answer_mass=0.40` 改成 support-mixed/high-quality full rollout source，增加中后段 state 覆盖，减少 majority-consistent fallback。
- 候选 B：score 从 `support_distribution_match` 改成 `posterior_support_match` 或 `support_value_affinity`，利用 smoothing 后的 posterior support overlap，避免 raw probe OOV 把目标压得过稀。
- 候选 C：probe 改成 staged：先用 1024-token probe 过滤/排序 candidate，再只对 top candidates 加长 probe，目标是减少 1536-token 全量 probe 成本，同时提高有效 support hit 率。
- 仍保持原则：full rollout group support/value 主导 target；probe 是未来分布估计器；source/anchor 只做 prior/drift guard。

## 2026-08-02 future_support_gain posterior-support softkeep smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_posteriormatch_softkeep_fullsupport_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_posteriormatch_softkeep_fullsupport_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_posteriormatch_softkeep_fullsupport_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_posteriormatch_softkeep_fullsupport_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

关键配置差异：

```text
ttrl.chunk_state_score_mode=future_support_gain
ttrl.chunk_state_future_support_score_type=posterior_support_match
ttrl.chunk_state_future_support_keep_mode=soft
ttrl.chunk_state_future_support_soft_weight_floor=0.05
ttrl.chunk_state_future_support_min_state_coverage=0.0
ttrl.chunk_state_future_support_max_state_oov=1.0
ttrl.chunk_state_min_answer_coverage=0.0
ttrl.chunk_state_min_informative_gap=0.0
ttrl.chunk_state_prune_zero_weight_samples=False
ttrl.chunk_state_source_quality_weight_mode=group_quality
```

这轮只改 score type，从 `support_distribution_match` 切到 `posterior_support_match`。目的不是让短 probe 判定 teacher，而是看 smoothing/posterior overlap 能不能降低 raw probe OOV 对 full-rollout support target 的破坏。

三步质量汇总：

```text
step  real_state  actor_samples  support_cov  w_cov  raw_margin  w_margin  state_oov  score_mean  label_cons  soft_w  nonzero_w  target_entropy
1     11          128            0.541        0.580  0.430       0.508     0.459      0.326       1.000       0.266   0.688      1.953
2     6           64             0.406        0.652  0.409       0.561     0.594      0.322       1.000       0.223   0.750      2.006
3     10          128            0.389        0.671  0.378       0.518     0.611      0.294       1.000       0.201   0.625      1.991
mean  9           106.7          0.445        0.634  0.406       0.529     0.555      0.314       1.000       0.230   0.688      1.983
```

耗时：

```text
step  gen      probe   score   ref     update_actor
1     43.498   7.645   9.948   5.735   4.960
2     22.859   6.988   6.274   0.980   2.678
3     23.022   8.219   9.913   1.874   4.654
mean  29.793   7.617   8.712   2.863   4.097
```

对比上一轮 `support_distribution_match` softkeep：

```text
metric                          support_distribution_match   posterior_support_match
answer_coverage_mean            0.332                        0.445
answer_coverage_weighted_mean   0.512                        0.634
positive_margin_mean            0.000*                       0.406
positive_margin_weighted_mean   0.000*                       0.529
support_coverage_mean           0.332                        0.445
state_oov_mean                  0.668                        0.555
score_mean                      0.132                        0.314
label_consistent_ratio          0.474                        1.000
num_actor_samples_mean          128.0                        106.7
update_actor_mean               4.761s                       4.097s
```

`*` 上一轮通用 positive_margin 指标为 0 是日志映射缺陷；`future_support_gain/positive_margin_mean` 当时是 0.247。本轮已在 `ray_trainer.py` 中把 `chunk_state_future_support_positive_margin` 同步写入通用 `chunk_state_positive_margin`，后续 `chunk_state/positive_margin_*` 可以直接看。

结论：

- 不扩 20-step。posterior match 明显改善了 full-support target 的加权质量：`answer_coverage_weighted_mean=0.634`、`positive_margin_weighted_mean=0.529`、`label_consistent_ratio=1.0`，但仍低于 20-step gate 的 `answer_coverage_weighted_mean >= 0.82`，并且 `num_actor_samples_mean=106.7` 低于 120。
- 这轮证明方向比 `support_distribution_match` 健康：posterior/smoothing 能把 raw probe OOV 从 `0.668` 降到 `0.555`，score/margin 都上升，actor update 仍约 4s，没有成为主瓶颈。
- 但它也再次证明主矛盾不是 update_actor，而是 target 质量。raw support coverage 只有 `0.445`，OOV 仍超过一半，说明当前 4 条 1536-token probe 仍然不能独立承担 teacher/label 的角色。
- 因此最该放弃的约束不是 PowerFlow、softkeep、full rollout group label estimation，而是“局部短视可判定性”：不能要求 chunk 的好坏在 short-horizon local answer hit/source consistency 这一级被判清楚。
- 后续 target 必须由 full rollout group 的 answer support/value/posterior 定义；source chunk 只保留为 prior/drift guard；probe 只作为 future distribution estimator，并且要拉长、分阶段或改成更贴近 group posterior 的估计。

下一步：

- 不继续做 sourcegate/source consistency hard constraint；已有证据显示这类约束会把 support coverage 和 actor sample volume 打坏。
- 设计下一版 `posterior_support_match_v2`：先由 full rollout group 建 prompt-level support/value，再从高 support/high pass 或 majority-consistent 完整轨迹选中后段 state；低信息 state 直接 skip/downweight。
- candidate score 改为“相对 full-group posterior 的 future distribution improvement”，例如 posterior expected value gain、top answer mass gain、support value margin、transport/KL improvement，而不是 local boxed hit。
- probe 可以分阶段：较短 probe 只用于初筛或估计粗分布，top candidates 再拉长 probe；所有 probe 信号必须回到 full-rollout support/value posterior 上计算 target。

## 2026-08-02 posterior-value-improvement staged smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_postvalue_staged_softkeep_fullsupport_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_postvalue_staged_softkeep_fullsupport_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_postvalue_staged_softkeep_fullsupport_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_postvalue_staged_softkeep_fullsupport_mid_c128_b32_r32_v64_3step_20260802.jsonl`

关键实现/配置差异：

```text
ttrl.chunk_state_future_support_score_type=posterior_value_improvement
ttrl.chunk_state_staged_probe_enable=True
ttrl.chunk_state_staged_probe_topk=2
ttrl.chunk_state_staged_probe_extra_samples=4
ttrl.chunk_state_staged_probe_extra_max_tokens=2048
ttrl.chunk_state_staged_probe_merge_mode=repeat_base
```

`posterior_value_improvement` 的 score 定义为：先用 `0.7 * smoothed_support_expected_value + 0.3 * smoothed_support_overlap` 乘以 `smoothed_transport_affinity` 得到 posterior value，再减去同一 state 内 candidate posterior value 的均值，只保留正向 advantage。这版的目的不是让 source answer 当 teacher，而是测试“相对 full-group posterior/value 的 state-local improvement”能不能比直接 posterior matching 更尖锐。

三步质量汇总：

```text
step  real_state  actor_samples  support_cov  w_cov  raw_margin  w_margin  state_oov  score_mean  label_cons  staged_extra  nonzero_w  target_entropy
1     11          128            0.521        0.580  0.115       0.133     0.479      0.034       0.531       8.757         0.688      1.696
2     4           64             0.691        0.612  0.077       0.108     0.309      0.025       0.578       8.566         0.500      1.790
3     11          128            0.265        0.537  0.123       0.152     0.735      0.026       0.336       9.055         0.688      1.545
mean  8.7         106.7          0.492        0.576  0.105       0.131     0.508      0.028       0.482       8.793         0.625      1.677
```

耗时：

```text
step  gen      probe   staged_extra  score   ref     update_actor
1     43.478   7.687   8.757         11.280  5.627   4.887
2     22.995   6.982   8.566         6.074   0.947   2.389
3     22.482   8.284   9.055         9.121   1.876   4.710
mean  29.652   7.651   8.793         8.825   2.817   3.995
```

对比上一轮 `posterior_support_match`：

```text
metric                          posterior_support_match   posterior_value_improvement_staged
answer_coverage_mean            0.445                     0.492
answer_coverage_weighted_mean   0.634                     0.576
positive_margin_mean            0.406                     0.105
positive_margin_weighted_mean   0.529                     0.131
state_oov_mean                  0.555                     0.508
score_mean                      0.314                     0.028
label_consistent_ratio          1.000                     0.482
num_actor_samples_mean          106.7                     106.7
update_actor_mean               4.097s                    3.995s
extra_probe_cost_mean           0.000s                    8.793s
```

结论：

- 不扩 20-step。虽然 raw support coverage 从 `0.445` 到 `0.492` 稍有改善，OOV 从 `0.555` 到 `0.508` 稍降，但加权 target 质量变差：`answer_coverage_weighted_mean=0.576`，`positive_margin_weighted_mean=0.131`，明显低于上一轮 `0.634/0.529`。
- 主要失败原因不是 probe 不够长，而是 state-local mean baseline 把 PowerFlow target 变得过稀。`score_mean=0.028`、`label_consistent_ratio=0.482` 表明大部分 candidate 被压成接近 0，最后训练信号弱且 target entropy 降低。
- staged probe 的额外成本约 `8.8s/step`，但没有换来更好的加权 target；因此不应把“更长 probe + advantage 截断”作为当前主线。
- 这版再次支持最新原则：不要要求 chunk 在局部短视视角里被硬判定为好/坏。PowerFlow 更适合吃一个 soft posterior target，而不是稀疏 binary/advantage target。

下一步：

- 回到 `posterior_support_match` 作为当前较优 scorer，不使用 state-local mean clipping 作为主 target。
- 优先优化 full rollout group posterior/value 的状态选择和 state 权重，而不是把 candidate score 做得更稀疏：例如 support-mixed/high-quality source、更多中后段 state、低信息 state soft skip。
- 如果继续 staged probe，只让它改善 posterior estimator 的置信度/方差，不直接把 top-k advantage 当 teacher。更合适的做法是 top-k 加深后重新估计 soft posterior distribution，再做 distribution matching。

## 2026-08-02 posterior-support-match support-mixed smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_posteriormatch_supportmixed_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_posteriormatch_supportmixed_softkeep_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_posteriormatch_supportmixed_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_posteriormatch_supportmixed_softkeep_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

关键配置：

```text
ttrl.chunk_state_future_support_score_type=posterior_support_match
ttrl.chunk_state_source_mode=support_mixed
ttrl.chunk_state_source_mixed_low_ratio=0.50
ttrl.chunk_state_source_low_max_answer_mass=0.35
ttrl.chunk_state_source_min_valid_answer_mass=0.03125
ttrl.chunk_state_min_source_answer_mass=0.0
ttrl.chunk_state_source_quality_weight_mode=group_quality
ttrl.chunk_state_future_support_keep_mode=soft
ttrl.chunk_state_future_support_soft_weight_floor=0.05
```

三步质量汇总：

```text
step  real_state  actor_samples  source_acc  source_mass  support_cov  w_cov  raw_margin  w_margin  state_oov  score_mean  nonzero_w  target_entropy
1     24          192            0.125       0.117        0.293        0.451  0.297       0.406     0.707      0.220       1.000      2.000
2     20          192            0.083       0.070        0.199        0.341  0.287       0.387     0.801      0.209       0.833      2.011
3     19          192            0.042       0.075        0.297        0.528  0.323       0.503     0.703      0.244       0.792      2.006
mean  21          192            0.083       0.087        0.263        0.440  0.302       0.432     0.737      0.224       0.875      2.006
```

耗时：

```text
step  gen      chunks  probe   score   ref     update_actor
1     43.580   0.968   9.008   11.083  6.658   7.455
2     23.082   0.979   9.220   9.648   2.872   7.187
3     22.799   1.275   9.607   8.347   2.847   7.263
mean  29.820   1.074   9.278   9.693   4.126   7.302
```

对比前两轮 smoke：

```text
metric                          posterior_support_match   posterior_value_staged   support_mixed_0.50
answer_coverage_weighted_mean   0.634                     0.576                    0.440
positive_margin_weighted_mean   0.529                     0.131                    0.432
support_coverage_mean           0.445                     0.492                    0.263
state_oov_mean                  0.555                     0.508                    0.737
source_original_acc_mean        n/a                       n/a                      0.083
source_answer_mass_mean         n/a                       n/a                      0.087
num_actor_samples_mean          106.7                     106.7                    192.0
update_actor_mean               4.097s                    3.995s                   7.302s
```

结论：

- 不扩 20-step。`support_mixed_low_ratio=0.50` 虽然把 actor samples 提到 192，但 source 质量明显过低：`selected_original_acc_mean=0.083`、`source_answer_mass_mean=0.087`，导致 `support_coverage_mean=0.263`、`state_oov_mean=0.737`。
- 这不是 PowerFlow actor update 的问题，而是 state/source/target 估计质量问题。更多低质量 state 只会让 batch 变大、`update_actor` 变慢到 `7.3s`，但不会改善 target。
- 这轮直接反证“继续加 source 侧 hard/low contrastive 约束”的方向。低质量 source 可以做诊断或少量 contrastive prior，但不能主导训练 batch。
- 当前最重要的方法约束更新：放弃“局部短视可判定性”。不要再要求 chunk target 主要由 short-horizon probe 的局部 answer hit、source answer consistency 或 source floor 来定义。
- 替代原则是：先由 full rollout group 建 prompt-level support/value/posterior；chunk candidate 只学习哪个局部 transition 会把未来分布推向这个 full-group posterior 认为好的区域；source chunk 只作为 prior/drift guard，不作为 teacher/floor。
- probe 后续只能作为 future distribution estimator。短 probe 可以做粗估、筛选或方差控制，但 target 必须回到 full-rollout support/value/posterior 上计算，不能单独由局部 boxed hit 决定。

下一步：

- 回退 `support_mixed_low_ratio=0.50`。如果需要 contrastive state，比例应降到 `0.10-0.15`，并且只参与 prior/diagnostic，不直接稀释 high-support state batch。
- 主线转向 `posterior_support_match_v2`：高 support / majority-consistent 中后段 state，低信息 state skip/downweight，candidate target 用 full-group posterior expected value / support mass gain / transport or KL improvement。
- 先做 3-step smoke 验证四个量：`support_coverage_mean`、`state_oov_mean`、`answer_coverage_weighted_mean`、`positive_margin_weighted_mean`。达不到 gate 不跑 20-step。

## 2026-08-02 posterior-support-match v2 high-support softgate smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_posteriormatch_v2_highsupport_softgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_posteriormatch_v2_highsupport_softgate_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_posteriormatch_v2_highsupport_softgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_posteriormatch_v2_highsupport_softgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

关键配置：

```text
ttrl.chunk_state_future_support_score_type=posterior_support_match
ttrl.chunk_state_source_mode=majority_consistent
ttrl.chunk_state_source_select_by_mass=True
ttrl.chunk_state_min_prompt_top_mass=0.45
ttrl.chunk_state_min_prompt_valid_answer_coverage=0.65
ttrl.chunk_state_min_prompt_top_margin=0.20
ttrl.chunk_state_min_source_answer_mass=0.40
ttrl.chunk_state_future_support_min_state_coverage=0.35
ttrl.chunk_state_future_support_max_state_oov=0.65
ttrl.chunk_state_future_support_min_state_mean_mass=0.06
ttrl.chunk_state_future_support_min_state_max_mass=0.12
ttrl.chunk_state_future_support_min_state_top_margin=0.02
ttrl.chunk_state_future_support_keep_mode=soft
ttrl.chunk_state_prune_zero_weight_samples=True
```

三步质量汇总：

```text
step  real_state  actor_samples  source_acc  source_mass  support_cov  w_cov  raw_margin  w_margin  state_oov  score_mean  keep_ratio  target_entropy
1     8           64             1.000       0.614        0.418        0.604  0.454       0.542     0.582      0.329       0.125       1.936
2     10          80             0.875       0.609        0.441        0.707  0.391       0.463     0.559      0.340       0.062       2.031
3     4           32             1.000       0.670        0.590        0.745  0.541       0.550     0.410      0.415       0.125       1.946
mean  7.3         58.7           0.958       0.631        0.483        0.685  0.462       0.518     0.517      0.361       0.104       1.971
```

耗时：

```text
step  gen      chunks  probe   score   ref     update_actor
1     43.691   1.037   7.050   8.444   4.795   2.744
2     23.328   0.989   8.169   8.141   1.286   5.181
3     22.502   0.937   7.066   6.548   0.493   1.279
mean  29.840   0.988   7.428   7.711   2.191   3.068
```

对比最近三轮：

```text
metric                          posterior_support   support_mixed_0.50   v2_highsupport_softgate
selected_original_acc_mean      n/a                 0.083                0.958
source_answer_mass_mean         n/a                 0.087                0.631
answer_coverage_weighted_mean   0.634               0.440                0.685
positive_margin_weighted_mean   0.529               0.432                0.518
support_coverage_mean           0.445               0.263                0.483
state_oov_mean                  0.555               0.737                0.517
num_actor_samples_mean          106.7               192.0                58.7
future_support_keep_ratio       1.000               1.000                0.104
update_actor_mean               4.097s              7.302s               3.068s
```

结论：

- 不扩 20-step。v2 验证了最新方向是对的：full-group high-support / majority-consistent source 能显著修复 source 质量，`selected_original_acc_mean=0.958`、`source_answer_mass_mean=0.631`，并把 `answer_coverage_weighted_mean` 提到 `0.685`，明显优于 support-mixed 的 `0.440`。
- 但这版 gate 太严，训练信号被收窄：`future_support_keep_ratio=0.104`、`num_actor_samples_mean=58.7`，第三步只有 32 个 actor samples。这个规模不足以支撑 20-step/80-step 训练。
- 关键失败点不是 target 方向，而是 state gate 过度硬化：`min_state_top_margin=0.02` 和 `min_state_coverage=0.35/max_oov=0.65` 让多数 state 只作为低权重或被 prune 掉。
- 这轮支持用户纠偏：不应该回到 short-horizon local hit teacher；应该保留 full-group posterior target，同时放松 state gate，让更多 high-support source 的中后段 state 进入 soft distribution matching。

下一步：

- 做 v3 relaxed-gate：保留 `majority_consistent + source_select_by_mass + posterior_support_match`，但放松到 `min_state_coverage=0.25`、`max_state_oov=0.75`、`min_state_mean_mass=0.03`、`min_state_max_mass=0.06`、`min_state_top_margin=0.0`。
- 保持 source 侧 high-support gate，不再使用 `support_mixed_low_ratio=0.50`；最多在后续用 0.10-0.15 contrastive prior 做诊断。
- v3 gate 目标：`num_actor_samples_mean >= 96`、`answer_coverage_weighted_mean >= 0.65`、`positive_margin_weighted_mean >= 0.50`、`state_oov_mean <= 0.55`。达到后再考虑 20-step pilot。

## 2026-08-02 posterior-support-match v3 high-support relaxed-gate smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_posteriormatch_v3_highsupport_relaxedgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_posteriormatch_v3_highsupport_relaxedgate_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_posteriormatch_v3_highsupport_relaxedgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_posteriormatch_v3_highsupport_relaxedgate_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

关键配置是在 v2 基础上只放松 future-support state gate：

```text
ttrl.chunk_state_future_support_min_state_coverage=0.25
ttrl.chunk_state_future_support_max_state_oov=0.75
ttrl.chunk_state_future_support_min_state_mean_mass=0.03
ttrl.chunk_state_future_support_min_state_max_mass=0.06
ttrl.chunk_state_future_support_min_state_top_margin=0.0
ttrl.chunk_state_future_support_soft_weight_floor=0.03
```

三步质量汇总：

```text
step  real_state  actor_samples  source_acc  source_mass  support_cov  w_cov  raw_margin  w_margin  state_oov  score_mean  keep_ratio  target_entropy
1     8           64             1.000       0.614        0.418        0.598  0.454       0.539     0.582      0.329       0.625       1.936
2     7           56             1.000       0.602        0.375        0.539  0.382       0.521     0.625      0.293       0.500       1.987
3     7           56             1.000       0.606        0.340        0.623  0.344       0.424     0.660      0.284       0.500       2.026
mean  7.3         58.7           1.000       0.607        0.378        0.587  0.393       0.495     0.622      0.302       0.542       1.983
```

耗时：

```text
step  gen      chunks  probe   score   ref     update_actor
1     43.505   1.049   7.038   8.400   4.745   2.726
2     23.276   1.046   7.168   8.529   0.901   2.294
3     32.081   0.935   7.108   8.785   0.846   2.242
mean  32.954   1.010   7.105   8.571   2.164   2.421
```

对比 v2：

```text
metric                          v2_highsupport_softgate   v3_highsupport_relaxedgate
selected_original_acc_mean      0.958                     1.000
source_answer_mass_mean         0.631                     0.607
answer_coverage_weighted_mean   0.685                     0.587
positive_margin_weighted_mean   0.518                     0.495
support_coverage_mean           0.483                     0.378
state_oov_mean                  0.517                     0.622
num_actor_samples_mean          58.7                      58.7
future_support_keep_ratio       0.104                     0.542
update_actor_mean               3.068s                    2.421s
```

结论：

- 不扩 20-step。v3 验证了只放松 future-support gate 并不能解决主问题：`future_support_keep_ratio` 从 `0.104` 提到 `0.542`，但 `num_actor_samples_mean` 仍是 `58.7`，因为真正限制已经变成 source/prompt high-support gate 下的 `real_states` 数量。
- target 质量从 v2 退化：`answer_coverage_weighted_mean` 从 `0.685` 降到 `0.587`，`state_oov_mean` 从 `0.517` 升到 `0.622`。说明继续放宽 future gate 会引入更多低信息 state，不能作为下一步。
- v3 仍然支持 high-support source 的必要性：`selected_original_acc_mean=1.0`、`source_answer_mass_mean=0.607` 很健康，但每步只留下 7-8 个 real states，训练覆盖太窄。
- 下一步不应该再靠 short-horizon probe/local hit 当 teacher，也不应该继续放松 future gate；应该在 full-group posterior 语义下扩大高质量 state 数量。

下一步：

- 尝试 `states_per_prompt=2` 或降低 prompt/source gate 的跳过率，同时保留 `majority_consistent + source_select_by_mass + posterior_support_match`。
- 更优先的 v4 方向：`states_per_prompt=2`，保留 v2 的 stricter future gate 或介于 v2/v3 之间的 gate，让每个合格 prompt 贡献两个不同中后段 state，而不是放入低质量 prompt。
- v4 gate 目标仍是：`num_actor_samples_mean >= 96`、`answer_coverage_weighted_mean >= 0.65`、`positive_margin_weighted_mean >= 0.50`、`state_oov_mean <= 0.55`。

## 2026-08-02 posterior-support-match v4 high-support states-per-prompt=2 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_posteriormatch_v4_highsupport_spp2_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_posteriormatch_v4_highsupport_spp2_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_posteriormatch_v4_highsupport_spp2_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_posteriormatch_v4_highsupport_spp2_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

关键配置：

- 继承 v2 high-support strict softgate。
- 只新增 `ttrl.chunk_state_states_per_prompt=2`。
- 保持 `source_mode=majority_consistent`、`source_select_by_mass=True`、`future_support_score_type=posterior_support_match`。
- 保持 source chunk / support anchor 只作为 proposal/prior/drift guard，不作为主要 teacher。

三步质量汇总：

```text
step  real_state  actor_samples  source_acc  source_mass  support_cov  w_cov  raw_margin  w_margin  state_oov  score_mean  keep_ratio  target_entropy
1     16          128            1.000       0.614        0.387        0.544  0.411       0.498     0.613      0.309       0.125       1.963
2     12          96             0.875       0.578        0.461        0.641  0.382       0.486     0.539      0.322       0.250       2.026
3     14          112            1.000       0.555        0.305        0.566  0.350       0.488     0.695      0.264       0.062       2.000
mean  14.0        112.0          0.958       0.582        0.384        0.584  0.381       0.491     0.616      0.298       0.146       1.996
```

耗时：

```text
step  gen      chunks  probe   score   ref     update_actor
1     43.410   0.946   7.900   9.066   5.791   5.102
2     23.140   0.938   8.536   6.718   1.452   3.701
3     22.081   1.087   8.170   7.398   1.655   4.258
mean  29.544   0.990   8.202   7.727   2.966   4.354
```

对比 v2/v3：

```text
metric                          v2_highsupport_softgate   v3_relaxedgate   v4_spp2
selected_original_acc_mean      0.958                     1.000            0.958
source_answer_mass_mean         0.631                     0.607            0.582
answer_coverage_weighted_mean   0.685                     0.587            0.584
positive_margin_weighted_mean   0.518                     0.495            0.491
support_coverage_mean           0.483                     0.378            0.384
state_oov_mean                  0.517                     0.622            0.616
num_actor_samples_mean          58.7                      58.7             112.0
future_support_keep_ratio       0.104                     0.542            0.146
update_actor_mean               3.068s                    2.421s           4.354s
```

结论：

- 不扩 20-step。v4 证明 `states_per_prompt=2` 可以解决样本量问题：`num_actor_samples_mean=112.0`，超过 smoke gate 的 96。
- 但 target 质量没有达标：`answer_coverage_weighted_mean=0.584 < 0.65`，`positive_margin_weighted_mean=0.491 < 0.50`，`state_oov_mean=0.616 > 0.55`。第三步 `support_coverage=0.305`、`state_oov=0.695` 尤其差。
- 这轮支持最新判断：主矛盾不是 actor update 速度，也不是单纯 state 数量，而是 target 仍然被 state-local candidate/probe 的噪声主导。虽然配置名是 posterior support match，但实际 candidate future distribution 的 coverage/OOV 仍决定了很多权重，导致 target 不是稳定的 search-improved full-group posterior。
- 不应该继续要求 chunk target 在 short-horizon/local answer hit/source consistency 里被判清楚。short probe 只能作为候选未来分布的一个辅助估计，不能作为 teacher/floor。

下一步约束：

- 放弃“局部短视可判定性”作为核心训练约束。
- full rollout group 先定义 prompt-level answer support / majority posterior / value；chunk 只学习哪个局部 transition 会把未来分布推向这个 group-level good posterior。
- source chunk / support anchor 保留为 prior 或 drift guard，但不能作为主要 teacher，也不能作为硬 floor。
- 对 low-information state 直接跳过或低权重：all-negative、low support coverage、high OOV、flat support mass、malformed/repeated boxed/marker contamination。
- 下一版应该把 target 写成 `q_j proportional exp(alpha * score_j) * prior_j`，其中 `score_j` 主要来自 full-rollout support/value/posterior improvement，而不是 raw short-probe correctness 或 source-answer consistency。

## 2026-08-02 support-flow full-posterior anchor v5 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_fullposterior_spp2_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_fullposterior_spp2_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_fullposterior_spp2_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_fullposterior_spp2_mid_c128_b32_r32_v64_3step_20260802.jsonl`

关键配置：

- `ttrl.chunk_state_score_mode=support_flow`。
- `ttrl.chunk_state_support_flow_score_type=soft_mass`。
- `ttrl.chunk_state_probe/skipped_for_support_flow=1.0`，确认这一版没有 short-horizon probe teacher。
- 继承 high-support source gate，并设置 `states_per_prompt=2`。
- support anchors: `count=4`、`candidate_start=4`、`selection_mode=answer_stratified`、`skip_source=True`。

三步质量汇总：

```text
step  real_state  actor_samples  source_acc  source_mass  anchor_mass  label_ratio  coverage  w_cov  pos_margin  w_margin  score_mean  target_entropy
1     16          128            1.000       0.614        0.059        0.492        0.492     0.487 -0.255      -0.273    0.059       1.075
2     20          160            1.000       0.696        0.070        0.479        0.479     0.458 -0.288      -0.385    0.070       1.037
3     10          80             1.000       0.693        0.081        0.500        0.500     0.500 -0.178      -0.260    0.081       0.788
mean  15.3        122.7          1.000       0.668        0.070        0.490        0.490     0.482 -0.240      -0.306    0.070       0.967
```

耗时：

```text
step  gen      chunks  probe  score   ref     update_actor
1     43.577   0.976   0.000  7.698   5.759   5.103
2     22.676   1.115   0.000  6.010   2.407   5.991
3     25.575   0.954   0.000  6.680   1.216   3.081
mean  30.609   1.015   0.000  6.796   3.127   4.725
```

结论：

- 不扩 20-step。v5 成功验证了“去掉 short probe teacher”这件事工程上可行：三步 `chunk_state_probe/skipped_for_support_flow=1.0`，没有用 short-horizon 局部命中定义 target。
- 但这版 target 退化成少数 support-anchor trajectory 的质量蒸馏，而不是完整 answer posterior matching。`source_mass_mean=0.668`，但 `anchor_mass_mean=0.070`，说明 answer-stratified anchors 把高质量 full-group support 稀释到了低 mass alternative answers 上。
- `positive_margin_mean=-0.240`，代表 anchor candidate 大多并没有超过 source answer mass；`target_entropy=0.967`，target 过尖，容易变成“学某几条 anchor chunk”，而不是学 full posterior distribution。
- 这不是回到 short probe 的理由；相反，它说明下一版不能只用 trajectory-level anchor mass，而要按 answer-level posterior mass 给 candidate/anchor 分配目标概率。

下一步：

- 做 v6 answer-level posterior target：对每个 state 先用 full rollout group 得到 `P_good(answer)`，再把候选 chunk/anchor 映射到 answer support；同一 answer 的多个 chunks 共享该 answer posterior mass，避免 answer-stratified 后把主答案质量稀释成低 trajectory mass。
- source chunk 只作为 drift prior；如果 source answer 是 top posterior，可给 source chunk 一个 prior multiplier，但不作为 score floor。
- 保留 no-short-probe 主约束。probe 最多用于补全 candidate answer identity 或 future distribution 的辅助估计，不再直接决定 teacher。

## 2026-08-02 support-flow answer-posterior v6 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_answerposterior_spp2_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_answerposterior_spp2_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_answerposterior_spp2_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_answerposterior_spp2_mid_c128_b32_r32_v64_3step_20260802.jsonl`

代码变化：

- 在 `support_flow` score path 增加 `posterior_mass` / `posterior_gain`。
- `posterior_mass` 从 full rollout group 的 `chunk_state_prompt_answer_mass` 读取 prompt-level answer posterior，再映射到 support anchor candidate 的 answer；同一 answer 的多个 anchor 平分该 answer posterior mass。
- `posterior_gain` 预留为 `posterior_mass - source_mass * baseline_scale + gain_slack`。
- `positive_margin` 对 posterior score type 改用 `posterior_mass - source_mass`，不再用 trajectory-level anchor mass。

关键配置：

- `ttrl.chunk_state_score_mode=support_flow`。
- `ttrl.chunk_state_support_flow_score_type=posterior_mass`。
- `ttrl.chunk_state_support_flow_split_mass_by_answer=False`。
- `chunk_state_probe/skipped_for_support_flow=1.0`，确认没有 short-horizon probe teacher。
- batch / rollout 仍是 `data.train_batch_size=32`、`rollout.n=32`、`n_votes_per_prompt=64`、`states_per_prompt=2`。

三步质量汇总：

```text
step  real_state  actor_samples  source_acc  source_mass  posterior_mean  posterior_max  label_ratio  coverage  w_cov  pos_margin  w_margin  positive_ratio  target_entropy
1     16          128            1.000       0.614        0.059           0.359          0.492        0.492     0.487 -0.255      -0.273    0.059           1.075
2     14          112            1.000       0.678        0.052           0.298          0.500        0.500     0.500 -0.380      -0.315    0.052           1.355
3     14          112            1.000       0.595        0.081           0.525          0.484        0.500     0.500 -0.070      -0.066    0.081           0.631
mean  14.7        117.3          1.000       0.629        0.064           0.394          0.492        0.497     0.496 -0.235      -0.218    0.064           1.020
```

耗时：

```text
step  gen      chunks  probe  score   ref     update_actor
1     43.445   0.945   0.000  7.653   5.683   4.948
2     23.377   0.984   0.000  6.651   1.552   4.233
3     22.923   0.976   0.000  5.750   1.531   4.643
mean  29.915   0.968   0.000  6.685   2.922   4.608
```

结论：

- 不扩 20-step。v6 确认 answer-level posterior target 工程路径可以跑通，且全程没有 short probe teacher：`skipped_for_support_flow=1.0`。
- 但质量仍不够：`answer_coverage_weighted_mean=0.496`，`positive_margin_weighted_mean=-0.218`，`positive_ratio=0.064`。这说明 full posterior 已经接入，但当前 support anchors 很少提供比 source posterior 更强的局部 transition。
- v6 比 v5 方向更正确，但在这个 smoke 里没有实质改善，因为 selected anchors 的 answer 基本唯一：`posterior_answer_duplicate_ratio=0.0`，answer-level mass 和 trajectory-level anchor mass 在多数 state 上几乎等价。
- 这个结果进一步支持最新约束：不能回到“短 probe 局部命中就是 teacher”的做法，也不该继续加 source-side hard gate。主问题是 candidate/anchor 生成和选择还没有形成真正的 search-improved transition distribution。

下一步：

- target 仍保持由 full rollout group posterior/value 主导，short probe 只能作为辅助 future identity/value estimate。
- 改 candidate 侧而不是加 source gate：让候选 chunk 来自 full group 高 posterior answer 的多条 trajectory，或从同一 state 重新采样后用 longer-horizon support gain 映射到 full posterior。
- 对 low-information state 做硬跳过或低权重：flat support、低 valid answer coverage、高 OOV、malformed/repeated boxed、posterior max 过低、candidate 无法映射到 group support。
- 下一版优先验证 `q_j proportional exp(alpha * full_group_value_gain_j) * prior_j`，其中 prior 只做 drift guard，不能重新变成 source answer floor。

## 2026-08-02 support-flow answer-posterior massrank nosplit v7 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_answerposterior_massrank_nosplit_spp2_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_answerposterior_massrank_nosplit_spp2_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_answerposterior_massrank_nosplit_spp2_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_answerposterior_massrank_nosplit_spp2_mid_c128_b32_r32_v64_3step_20260802.jsonl`

代码变化：

- 新增 `ttrl.chunk_state_support_flow_posterior_split_duplicates`，默认 `true` 保持 v6 语义。
- v7 显式设置 `posterior_split_duplicates=False`：同一 answer 的多个 support anchors 不再平分 answer posterior mass。
- v7 同时设置 `chunk_state_support_anchor_selection_mode=mass_ranked`，把 support anchors 从 answer-stratified 改成高 mass answer 优先。

关键配置：

- `ttrl.chunk_state_score_mode=support_flow`。
- `ttrl.chunk_state_support_flow_score_type=posterior_mass`。
- `ttrl.chunk_state_support_anchor_selection_mode=mass_ranked`。
- `ttrl.chunk_state_support_flow_posterior_split_duplicates=False`。
- `chunk_state_probe/skipped_for_support_flow=1.0`，确认仍然没有 short-horizon probe teacher。
- batch / rollout 仍是 `data.train_batch_size=32`、`rollout.n=32`、`n_votes_per_prompt=64`、`states_per_prompt=2`。

三步质量汇总：

```text
step  real_state  actor_samples  source_acc  source_mass  posterior_mean  posterior_max  dup_ratio  label_ratio  coverage  w_cov  pos_margin  w_margin  positive_ratio  target_entropy
1     16          128            1.000       0.614        0.093           0.359          0.125      0.492        0.492     0.487 -0.255      -0.273    0.093           1.249
2     10          80             1.000       0.659        0.077           0.235          0.109      0.492        0.492     0.478 -0.424      -0.351    0.077           1.611
3     12          96             1.000       0.666        0.076           0.191          0.156      0.469        0.492     0.487 -0.475      -0.500    0.076           1.694
mean  12.7        101.3          1.000       0.646        0.082           0.262          0.130      0.484        0.492     0.484 -0.385      -0.375    0.082           1.518
```

耗时：

```text
step  gen      chunks  probe  score   ref     update_actor
1     43.559   1.125   0.000  7.625   5.828   4.978
2     22.669   0.962   0.000  6.370   1.219   3.113
3     22.402   0.970   0.000  5.696   1.419   3.569
mean  29.543   1.019   0.000  6.564   2.822   3.887
```

结论：

- 不扩 20-step。v7 证明“高 posterior answer 多 chunk + 不平分 posterior mass”不是当前主解。
- 相比 v6，`posterior_mass_mean` 从 `0.064` 提到 `0.082`，但 `posterior_mass_max_mean` 从 `0.394` 降到 `0.262`，`positive_margin_weighted_mean` 从 `-0.218` 变差到 `-0.375`。这说明只是增加高 mass answer 的重复 chunk 密度，并没有得到比 source 更强的 search-improved transition。
- `target_entropy=1.518` 明显更高，target 更分散；`num_actor_samples=101.3` 低于 v6 的 117.3。actor update 更快（`3.887s`），但这是样本减少和 target 变散的副作用，不是有效优化。
- 这轮进一步支持“不能把 full group 高 posterior trajectory replay 直接当 improvement target”。如果 source 本身已经是 majority-consistent/high-mass，候选来自同一分布的高 mass chunk 很难在 `posterior_mass - source_mass` 上产生正 margin。
- 末尾出现 `DataLoader worker ... killed by signal: Killed` 的 Ray worker traceback，但日志包含 3 个训练 step、48 行 diag，且进程最终返回 0；本轮按完整 smoke 记录。后续若扩长实验，需要关注这个退出阶段 worker 清理问题。

下一步：

- 不继续调 `mass_ranked` / `answer_stratified` / duplicate split 这类 anchor replay 细节。
- score 要从“候选 answer posterior mass”转为“相对 source 的 future value gain / support improvement”。也就是 candidate 必须回答：这个 chunk 会不会把后续 completion 分布推向 full group 认可的好答案，而不是它自己属于哪个 high-mass answer。
- 可以保留 no-short-probe 主约束，但需要引入 longer-horizon future distribution estimator 或从完整 rollout group 构造 state-level value label；short probe 只统计 `p_j(a)`，不能直接当 teacher。

## 2026-08-02 future-gain support-distribution-match v8 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportdist_v8_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_supportdist_v8_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_supportdist_v8_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportdist_v8_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

关键配置：

- `ttrl.chunk_state_score_mode=future_support_gain`。
- `ttrl.chunk_state_future_support_score_type=support_distribution_match`。
- `ttrl.chunk_state_probe_samples=4`，`ttrl.chunk_state_probe_max_tokens=1536`。
- `ttrl.chunk_state_future_support_min_positive_margin=0.04`。
- `ttrl.chunk_state_future_support_min_state_coverage=0.25`，`max_state_oov=0.75`。
- `data.train_batch_size=32`、`rollout.n=32`、`n_votes_per_prompt=64`、`states_per_prompt=2`。
- `actor_rollout_ref.actor.use_dynamic_bsz=False`，保持训练语义不引入 dynamic batch。

设计意图：

- 这版不再让 short-horizon probe 的局部命中直接定义 teacher。
- full rollout group 先定义 prompt-level answer support distribution。
- 对每个 candidate chunk 只用 longer probe 估计它的 future answer distribution `p_j(a)`，再用 `support_distribution_match` 判断它是否把未来分布推向 full group support。
- source chunk / support anchor 只保留为 prior 和 drift guard，不作为主要 teacher/floor。

三步质量汇总：

```text
step  coverage  oov    keep  score  label  improved  margin  raw_gain  transport_gain  real_state  actor_samples  zeroed  w_cov  w_margin  positive_ratio  target_entropy  actor_w_nonzero
1     0.152     0.848  0.125 0.082  0.188  0.375     0.234   -0.586    -0.601          3           8              0.875   0.250  0.730     0.082           1.674           1.000
2     0.070     0.930  0.125 0.032  0.141  0.750     0.156   -0.530    -0.533          4           8              0.875   0.250  0.711     0.032           1.558           1.000
3     0.484     0.516  0.125 0.317  0.719  1.000     0.618   -0.252    -0.319          3           64             1.000   0.000  0.000     0.317           1.520           0.000
mean  0.235     0.765  0.125 0.144  0.349  0.708     0.336   -0.456    -0.484          3.3         26.7           0.917   0.167  0.480     0.144           1.584           0.667
```

耗时：

```text
step  gen      probe  score   update_actor
1     43.547   7.022  8.665   0.707
2     23.345   7.170  7.047   0.479
3     32.088   6.917  11.332  2.466
mean  32.993   7.036  9.015   1.217
```

结论：

- 不扩 20-step。v8 方向比 v6/v7 更接近目标语义，但当前配置不是有效训练。
- 这版证明 `support_distribution_match` 链路能跑，且 target 语义已经从局部 answer hit 转向 full-rollout support distribution。
- 失败点非常明确：prompt 级 full group 质量不差，`prompt_valid_answer_coverage_mean=0.922`；但 candidate/probe future distribution 和 full support 对不上，`support_coverage_mean=0.235`、`state_oov_mean=0.765`。
- hard keep 太窄：每步 `learnable_state_keep_ratio=0.125`，平均只有 `3.3` 个 real states，且 `skipped_support_sources=28.7/32`。
- 第 3 step 虽然 coverage 回到 `0.484`，但 `zeroed_state_ratio=1.0`、`actor/powerflow_weight/nonzero_ratio=0.0`、`actor/powerflow_loss=0.0`，说明 hard gate / actor batch weight 组合会把一次看似有信号的 step 清成无效更新。
- update_actor 很快，均值 `1.217s`，但这是强剪枝和局部全零权重的副作用，不能当作 infra 优化成果。

下一步：

- 保留 `future_support_gain + support_distribution_match` 作为主线，不回退到 short-probe local hit/source consistency。
- 放松 hard keep 为 soft weighting：低 coverage/high OOV state 可以低权重，但不能让整步 actor weight 归零；需要设置最小有效 state 或 fallback 到 soft target。
- 提高 candidate/probe 和 full support 的重合，而不是加 source-side 硬约束：优先从 full rollout group 的 mid/late state 构造同 boundary candidate，或增加 probe 分支数/候选质量，再计算 support match。
- 下一版 smoke 的通过标准：`actor/powerflow_weight/nonzero_ratio` 每步非零，`support_coverage_mean` 至少接近 v6/v7 的 `0.48-0.50`，`state_oov_mean` 明显低于 `0.5`，且不能靠 source answer floor 获得这些数。

## 2026-08-02 future-gain support-distribution-match softkeep v9 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportdist_softkeep_v9_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_supportdist_softkeep_v9_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_supportdist_softkeep_v9_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportdist_softkeep_v9_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

相对 v8 的变化：

- score 仍是 `ttrl.chunk_state_future_support_score_type=support_distribution_match`，不回退到 short-probe local hit/source consistency。
- `ttrl.chunk_state_future_support_keep_mode=soft`，`soft_weight_floor=0.02`。
- support quality gate 从 v8 的 `coverage>=0.25/oov<=0.75/min_margin=0.04` 放宽到 `coverage>=0.125/oov<=0.875/min_margin=0.0`。
- 目的只是避免 v8 第 3 step 的 actor weight 全零，验证 soft weighting 能否让 full-support target 持续产生更新。

三步质量汇总：

```text
step  coverage  oov    keep   score  label  improved  margin  raw_gain  transport_gain  real_state  actor_samples  zeroed  state_w  w_cov  w_margin  positive_ratio  target_entropy  actor_w_nonzero
1     0.152     0.848  0.250  0.082  0.188  0.375     0.234   -0.586    -0.601          3           24             0.625   0.118    0.627  0.593     0.082           1.674           1.000
2     0.086     0.914  0.125  0.019  0.203  0.875     0.064   -0.772    -0.781          3           8              0.875   0.047    0.312  0.310     0.019           1.909           1.000
3     0.137     0.863  0.125  0.083  0.234  0.500     0.118   -0.676    -0.693          3           16             0.750   0.110    0.554  0.484     0.083           2.016           1.000
mean  0.125     0.875  0.167  0.061  0.208  0.583     0.139   -0.678    -0.692          3.0         16.0           0.750   0.092    0.498  0.462     0.061           1.866           1.000
```

耗时：

```text
step  gen      probe  score   update_actor
1     43.477   7.043  8.501   1.348
2     23.399   7.088  6.286   0.427
3     23.471   7.197  6.263   0.805
mean  30.116   7.109  7.017   0.860
```

结论：

- 不扩 20-step。v9 只修复了 v8 的 hard-gate 零更新问题，没有修复 target 质量问题。
- soft keep 按预期生效：`future_support_keep_mode_soft=1.0`，三步 `actor/powerflow_weight/nonzero_ratio=1.0`，没有再出现 v8 第 3 step 的 `powerflow_loss=0`。
- 但 full-support match 的 candidate 分布仍然很差：`support_coverage_mean=0.125`，`state_oov_mean=0.875`，甚至比 v8 均值 `0.235/0.765` 更差。
- prompt-level full group 仍有足够信息：`prompt_valid_answer_coverage_mean=0.931`。问题不是 full group label 不存在，而是当前 mid-state candidate/probe 很少落回 full group support。
- update_actor 均值 `0.860s`，说明 chunk actor update 本身很轻；瓶颈和主矛盾继续是 label/candidate 构造，不是 actor update 吞吐。
- 退出阶段又出现 `DataLoader worker ... killed by signal: Killed`，但 3 个 step、24 行 diag 和 raw log 均完整，进程最终返回 0。扩长实验前需要继续关注 worker 清理/内存峰值。

下一步：

- 不再继续只调 gate/floor；soft keep 可作为默认安全机制，避免整步无效更新。
- 主改 candidate/state：从 full rollout group 的 high-support trajectory 在同 boundary 上构造 candidates，或让 chunk candidate 继承 full support answer path 的 state alignment，减少 OOV。
- 需要把“full rollout group 定义目标”落实到 state/candidate 构造层，而不是只在 score 层做 support match；否则 probe 采样即使更长，也仍主要采到 support 外答案。
- 下一条 smoke 应该比较两种 candidate source：当前 resample-from-state vs full-group aligned candidate，并用同一 `support_distribution_match + soft keep` 判断 coverage/OOV 是否显著改善。

## 2026-08-02 future-gain support-distribution-match alignedanchors v10 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportdist_alignedanchors_v10_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_supportdist_alignedanchors_v10_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_supportdist_alignedanchors_v10_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportdist_alignedanchors_v10_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

相对 v9 的变化：

- score 仍是 `support_distribution_match`，keep 仍是 soft keep，不回退到局部命中或 source consistency。
- candidate 结构改为：`candidate0=source continuation`，`candidate1-7=full rollout support anchors`。
- `support_anchor_count=7`，`candidate_start=1`，尽量用 full-group high-support continuation 覆盖 candidate space。
- 打开 `support_anchor_prefix_compat_enable=True`，`prefix_compat_mode=soft`，`prefix_compat_tokens=64`，`prefix_score_power=0.5`，用 prefix match 作为 soft prior 而不是 hard teacher。

三步质量汇总：

```text
step  anchor_inj  prefix  anchor_mass  coverage  oov    keep   score  label  improved  margin  raw_gain  transport_gain  real_state  actor_samples  zeroed  state_w  w_cov  w_margin  positive_ratio  target_entropy  actor_w_nonzero
1     0.500       0.006   0.515        0.250     0.750  0.625  0.151  0.297  1.000     0.588   -0.488    -0.518          3           24             0.625   0.174    0.358  0.567     0.151           0.863           1.000
2     0.482       0.015   0.099        0.324     0.676  0.375  0.213  0.594  1.000     0.526   -0.406    -0.465          4           16             0.750   0.257    0.299  0.593     0.213           1.402           1.000
3     0.518       0.016   0.036        0.051     0.949  0.000  0.002  0.156  0.750     0.010   -0.693    -0.691          2           64             1.000   0.020    0.000  0.000     0.002           2.066           0.000
mean  0.500       0.012   0.217        0.208     0.792  0.333  0.122  0.349  0.917     0.375   -0.529    -0.558          3.0         34.7           0.792   0.150    0.219  0.387     0.122           1.444           0.667
```

耗时：

```text
step  gen      probe  score   update_actor
1     43.547   7.257  8.408   1.290
2     22.510   7.154  6.544   0.695
3     23.166   7.210  6.625   2.450
mean  29.741   7.207  7.192   1.478
```

结论：

- 不扩 20-step，但 v10 给出了比 v9 更有价值的方向证据。
- step 1/2 明显改善：v9 的 coverage/OOV 是 `0.152/0.848`、`0.086/0.914`；v10 变成 `0.250/0.750`、`0.324/0.676`。这说明把 candidate space 更多交给 full-rollout support anchors 是有效方向。
- step 3 失败也很清楚：`anchor_mass_mean=0.036`，`support_coverage=0.051`，`state_keep=0.0`，actor weight 全零。也就是说 aligned anchors 只在 prompt/full-group support 本身强的时候有效；低质量 prompt/state 仍会把整步冲掉。
- prefix compatibility 没有提供真正 state alignment：`prefix_match_mean=0.012`，几乎为零。v10 的提升主要来自 anchor 覆盖率和 high-support continuation，而不是 prefix 连续性。
- 当前最重要的新结论：candidate 侧改动确实能改善 full-support target，但必须同时收紧 state/prompt selection 或构造同源 boundary anchors，不能把低 anchor-mass 的 prompt/state 送进 actor update。
- update_actor 仍然很轻，均值 `1.478s`；这个方向的 infra 成本可以接受。

下一步：

- 保留 `support_distribution_match + soft keep + full-group anchor-heavy candidate`。
- 不能依赖 token prefix match 解决 state alignment；应直接构造同源/同 trajectory 的 state-candidate pairs，或选择 source state 本身来自 high-support answer family，再从同 family 的 sibling rollouts 取 continuation。
- 对 state selection 加更硬的 full-support质量门：例如 `anchor_mass_mean`、prompt answer entropy、real state count、source answer mass 的组合，避免 v10 step 3 这种 `anchor_mass_mean=0.036` 的 batch 更新。
- 下一条 v11 应测试 `source_quality_weight_mode` 或更严格 source/prompt gate，而不是继续加 anchor 数；验收看三步都保持 `actor_w_nonzero=1` 且 coverage 不低于 `0.25`。

## 2026-08-02 future-gain support-distribution-match highmassanchors v11 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_supportdist_highmassanchors_v11_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_supportdist_highmassanchors_v11_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_supportdist_highmassanchors_v11_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_supportdist_highmassanchors_v11_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

相对 v10 的变化：

- score 仍是 `support_distribution_match`，keep 仍是 `soft`，不回退到 short-horizon local answer hit 或 source consistency。
- candidate 结构仍是 `candidate0=source continuation`，`candidate1-7=full rollout support anchors`。
- `support_anchor_min_mass` 从 v10 的低门槛提高到 `0.10`，只注入 full group 中 answer mass 更高的 anchors。
- 关闭 prefix compatibility：v10 已证明 `prefix_match_mean=0.012`，prefix prior 没有实际提供 state alignment。

三步质量汇总：

```text
step  anchor_inj  anchor_mass  anchor_pos  coverage  oov    keep   score  label  improved  margin  real_state  actor_samples  zeroed  w_cov  w_margin  positive_ratio  target_entropy  actor_w_nonzero
1     0.625       0.704        0.547       0.328     0.672  0.500  0.200  0.438  1.000     0.579   3           24             0.625   0.402  0.550     0.200           1.160           1.000
2     0.375       0.871        0.328       0.090     0.910  0.125  0.005  0.250  0.875     0.027   2           8              0.875   0.156  0.012     0.005           1.976           1.000
3     0.250       0.604        0.219       0.660     0.340  0.500  0.346  0.938  1.000     0.576   3           24             0.625   0.599  0.644     0.346           1.795           1.000
mean  0.417       0.726        0.365       0.359     0.641  0.375  0.184  0.542  0.958     0.394   2.7         18.7           0.708   0.386  0.402     0.184           1.644           1.000
```

耗时：

```text
step  gen      probe  score   ref    update_actor
1     43.579   7.042  8.114   4.153  1.433
2     24.218   7.107  8.532   0.134  0.444
3     22.910   7.086  7.047   0.438  1.194
mean  30.236   7.078  7.898   1.575  1.024
```

结论：

- 不扩 20-step。v11 是当前 support_distribution_match 系列里均值最好的一版，但仍不稳定。
- 相比 v9/v10，v11 均值提升明显：`support_coverage_mean=0.359`、`state_oov_mean=0.641`，优于 v9 的 `0.125/0.875` 和 v10 的 `0.208/0.792`。
- 单步上限有价值：step 3 达到 `coverage=0.660`、`OOV=0.340`、`label_consistent_ratio=0.938`，说明 full-group support anchors 能把 candidate distribution 拉回目标 support。
- 但 step 2 是关键反例：即使 `anchor_mass_mean=0.871`、source/prompt original 指标很高，candidate support coverage 仍只有 `0.090`，OOV `0.910`。这说明继续提高 source/anchor mass 门槛不能从根上解决 target 质量。
- `actor/powerflow_weight/nonzero_ratio=1.0` 每步都保持非零，`update_actor` 均值 `1.024s`，再次确认 actor update 不是主矛盾。当前主矛盾是 state/candidate target 的 label estimation 语义。
- v11 也支持最新策略纠偏：不能再要求 chunk target 主要由短 probe 的局部命中或 source answer consistency 判定。要放弃“局部短视可判定性”这个训练约束。

下一步：

- 保留 PowerFlow-style actor update、soft keep、clip4 和 full rollout group support label estimation。
- 放弃把 short-horizon probe local hit/source consistency 当 teacher 的设计；probe 只能作为 future distribution 的采样估计，不应单独定义 target。
- target 改为由 full rollout group posterior/value/support improvement 主导：先对 prompt 采完整 rollout group，形成 answer support distribution、majority/support mass、value/posterior，再在 chunk state 上学习哪些 local transition 会把未来分布推向这个 full-group posterior。
- source chunk/anchor 只作为 prior 或 drift guard，不能作为主要 teacher/floor；继续加 source-side 硬约束已经被 v10/v11 反例否定。
- state selection 必须优先解决低信息样本：跳过或强降权 all-negative、support coverage 低、OOV 高、top mass 过平、malformed/repeated boxed/marker 污染的 states。
- 下一轮应实现/测试 `posterior_support_match` 或 `posterior_value_improvement` 作为主 target，而不是继续调 anchor mass；smoke 先看 `support_coverage`、`OOV`、`label_consistent_ratio`、`actor_w_nonzero`，再决定是否跑 20-step。

## 2026-08-02 posterior-support-match highmassanchors v12 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_posteriormatch_highmassanchors_v12_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_posteriormatch_highmassanchors_v12_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_posteriormatch_highmassanchors_v12_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_posteriormatch_highmassanchors_v12_mid_c128_probe1536x4_b32_r32_v64_3step_20260802.jsonl`

相对 v11 的变化：

- 保持 v11 的 high-mass anchor / state gate / soft keep / clip4 配置不变。
- 唯一核心变化：`ttrl.chunk_state_future_support_score_type=posterior_support_match`。
- 目标是让 full-rollout support posterior 对 sparse probe evidence 做 smoothing，再由 posterior expected value / overlap / affinity 定义 PowerFlow target。
- 这版更符合最新策略纠偏：probe 只是 future distribution 的采样估计，不再由 short-horizon local hit 单独定义 teacher。

三步质量汇总：

```text
step  anchor_inj  anchor_mass  anchor_pos  coverage  oov    keep   score  label  improved  margin  real_state  actor_samples  zeroed  w_cov  w_margin  positive_ratio  target_entropy  actor_w_nonzero
1     0.625       0.704        0.547       0.328     0.672  0.500  0.338  1.000  1.000     0.611   3           24             0.625   0.402  0.595     0.338           1.706           1.000
2     0.750       0.837        0.656       0.578     0.422  0.000  0.543  1.000  1.000     0.635   3           8              0.875   0.781  0.779     0.543           2.027           1.000
3     0.875       0.534        0.766       0.633     0.367  0.625  0.412  1.000  1.000     0.577   3           24             0.625   0.621  0.593     0.412           1.921           1.000
mean  0.750       0.692        0.656       0.513     0.487  0.375  0.431  1.000  1.000     0.608   3.0         18.7           0.708   0.601  0.656     0.431           1.885           1.000
```

耗时：

```text
step  gen      probe  score   ref    update_actor
1     43.659   7.019  9.797   4.249  1.314
2     22.867   6.988  6.341   0.118  0.438
3     32.537   7.133  8.266   0.417  1.170
mean  33.021   7.047  8.135   1.595  0.974
```

与 v11 对比：

```text
metric                  v11 supportdist  v12 posterior
support_coverage_mean   0.359            0.513
state_oov_mean          0.641            0.487
score_mean              0.184            0.431
label_consistent_ratio  0.542            1.000
positive_ratio          0.184            0.431
weighted_coverage       0.386            0.601
weighted_margin         0.402            0.656
update_actor            1.024s           0.974s
```

结论：

- v12 是目前最符合目标语义的 smoke，建议进入下一阶段 20-step pilot。
- posterior smoothing 明显修复 v11 的主失败点：v11 step 2 `coverage=0.090/OOV=0.910`，v12 step 2 在相近 high-mass anchor 配置下达到 `coverage=0.578/OOV=0.422`。
- `label_consistent_ratio=1.0` 三步全满，说明 target 不再被短 probe 的局部稀疏命中打散；full-group posterior 开始主导 chunk target。
- `actor/powerflow_weight/nonzero_ratio=1.0` 每步保持非零，`update_actor` 仍约 `1s`，PowerFlow chunk actor update 的 infra 成本可接受。
- 仍有一个需要修的信号：step 2 `state_keep_ratio=0.0` 但 soft keep 仍产出 actor samples。说明 hard keep 指标对 posterior 模式过严，后续应把 learnable gate 改成 posterior-aware，不要让 `state_top_margin` 这类 old-score gate 误判。
- 第 2 步 `boundary_zero_ratio=0.75`，说明 state selection 仍可能抽到过早状态；下一轮 20-step 前应强制中后段 boundary 或统计单独分层。

下一步：

- 以 v12 为主线，不再继续调 `support_distribution_match` anchor/source gate。
- 先跑 20-step pilot：`posterior_support_match + highmassanchors + soft keep`，每 20 step 做一次 val 或先 final val，观察 mean@16/maj@16 是否开始超过 MV baseline。
- 同时准备 v13 小改：posterior-aware learnable gate，去掉对 `state_top_margin` 的硬依赖，强制 nonzero mid/late boundary，保留 low-information skip/downweight。
- 若 20-step target 指标稳定，再扩到 80-step pilot；若 20-step 仍差，优先改 state selection，不回退到 short-probe local teacher。

## 2026-08-02 posterior-support-match highmassanchors v12 20-step pilot partial

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_futuregain_posteriormatch_highmassanchors_v12_mid_c128_probe1536x4_b32_r32_v64_20step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_futuregain_posteriormatch_highmassanchors_v12_20step_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_futuregain_posteriormatch_highmassanchors_v12_mid_c128_probe1536x4_b32_r32_v64_20step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_futuregain_posteriormatch_highmassanchors_v12_mid_c128_probe1536x4_b32_r32_v64_20step_20260802.jsonl`

状态：

- 计划 20 step，但在 step 9 后人工中止。
- 没有产生 step 20 validation；本次只作为 target 质量诊断，不作为最终 acc 实验。
- 中止原因不是 actor update 慢，而是 target 质量反复塌陷，继续跑会浪费 GPU 时间。

1-9 step 聚合：

```text
step  coverage  oov    score  keep   actor_samples  actor_w_nonzero  update_actor  gen     probe  score_t
1     0.328     0.672  0.338  0.500  24             1.000            1.361         43.377  7.093  9.689
2     0.664     0.336  0.467  0.375  8              1.000            0.470         33.230  7.105  8.141
3     0.785     0.215  0.490  0.625  24             1.000            1.069         22.690  6.774  7.339
4     0.402     0.598  0.411  0.750  32             1.000            1.378         23.830  6.987  6.546
5     0.012     0.988  0.201  0.000  64             0.000            2.395         21.644  9.472  5.895
6     0.043     0.957  0.194  0.000  64             0.000            2.555         23.144  7.104  7.777
7     0.277     0.723  0.312  0.750  24             1.000            1.099         23.387  6.940  8.676
8     0.688     0.312  0.502  0.125  16             1.000            0.888         22.608  7.028  6.235
9     0.070     0.930  0.272  0.000  8              1.000            0.477         22.786  7.210  6.797
mean  0.363     0.637  0.354  0.347  29.3           0.778            1.299         26.300  7.301  7.455
```

关键结论：

- v12 smoke 的三步好转没有延续到 20-step pilot；step 5/6 连续出现 `actor_batch_powerflow_weight_nonzero_ratio=0`，实际是零权重 update。
- `support_coverage_mean` 均值只有 `0.363`，`state_oov_mean` 均值 `0.637`；step 5/6/9 分别掉到 `coverage=0.012/0.043/0.070`。
- `label_consistent_ratio=1.0` 在这里不再足以说明 target 好，因为 posterior smoothing 能把 label 做成 consistent，但 candidate future distribution 仍可能大面积 OOV。
- `update_actor` 均值 `1.299s`，即使塌陷 step 也只是 2.4-2.6s；所以 infra/actor update 不是当前主矛盾。
- 更准确的失败归因：`posterior_support_match` 仍然从 candidate probe answers/counts 出发，再用 full-rollout posterior 做 smoothing。它比 raw short-probe teacher 好，但结构上还没有真正放弃“局部 probe 分布主导 target”的约束。
- 这和最新策略纠偏一致：应放弃“局部短视可判定性”，不再让 short-horizon probe/local answer hit/source consistency 直接定义 chunk teacher。

下一步 v13：

- 不继续加 source gate 或 anchor mass gate。
- 切到 `chunk_state_score_mode=support_flow`，使用 full-rollout support anchors 和 `chunk_state_support_flow_score_type=posterior_mass`。
- target 直接由 prompt-level full-rollout answer posterior mass 映射到 support anchors；source chunk/anchor 只作为 proposal/prior/drift guard，不作为 teacher floor。
- probe 不再参与 teacher 定义；先跑 3-step smoke 看 `support_flow/score_mean`、`nonzero_state_ratio`、`actor_w_nonzero`、`target_entropy` 和 `update_actor`。

## 2026-08-02 support-flow posterior-mass v13 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_posteriormass_v13_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_posteriormass_v13_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_v13_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_posteriormass_v13_mid_c128_b32_r32_v64_3step_20260802.jsonl`

相对 v12 的关键变化：

- `ttrl.chunk_state_score_mode=support_flow`
- `ttrl.chunk_state_support_flow_score_type=posterior_mass`
- `chunk_state_probe/skipped_for_support_flow=1.0`，不再让 probe answers/counts 定义 teacher。
- target 由 full-rollout group answer posterior 映射到 injected support anchors；source/anchor 是 proposal/prior，不是 short-probe teacher。

三步质量汇总：

```text
step  anchor_inj  anchor_mass  uniq_answer  post_mean  post_max  source_mass  margin   score  label  answer_cov  pos_ratio  entropy  samples  actor_w_nz  update_actor
1     1.000       0.298        0.732        0.096      0.311     0.708        -0.397   0.096  0.875  0.875       0.096      1.360    24       1.000       1.318
2     0.286       0.031        1.000        0.008      0.031     0.531        -0.500   0.008  0.250  0.250       0.008      1.969    8        1.000       0.481
3     0.964       0.504        0.357        0.089      0.258     0.642        -0.384   0.089  0.844  0.844       0.089      1.588    32       1.000       1.419
mean  0.750       0.278        0.696        0.064      0.200     0.627        -0.427   0.064  0.656  0.656       0.064      1.639    21.3     1.000       1.073
```

耗时：

```text
step  gen      chunks  score  ref    update_actor
1     43.477   1.075   7.555  4.291  1.318
2     22.366   1.040   6.076  0.148  0.481
3     22.686   1.045   6.427  0.526  1.419
mean  29.510   1.053   6.686  1.655  1.073
```

结论：

- v13 完成了最重要的语义修正：teacher 不再由 short-horizon probe/local answer hit 定义。
- 三步 `actor_batch_powerflow_weight_nonzero_ratio=1.0`，没有 v12 step 5/6 那种零权重 update。
- 但 target 强度偏弱：`posterior_mass_mean=0.064`、`positive_ratio=0.064`，step 2 只有 `score_mean=0.008`。
- 主要原因是 `posterior_split_duplicates=True`：如果同一个 answer 下有多个 support anchors，answer posterior mass 被拆分到多个 candidate，导致单个 chunk target mass 过小；step 3 的 `posterior_answer_duplicate_ratio=0.643` 说明重复 answer 很常见。
- 这不是回到 source gate 的问题，而是 full-posterior target 在 chunk-candidate 分布上的 mass allocation 问题。

下一步：

- 跑 v14 smoke：保持 `support_flow + posterior_mass`，但设置 `chunk_state_support_flow_posterior_split_duplicates=False`。
- 目标是测试“每个来自高 posterior answer 的 chunk 都可作为有效 local transition”时，target score/positive ratio/grad 是否恢复，同时仍不使用 short-probe teacher。
- 如果 v14 稳定，再扩 20-step；如果 v14 过尖或过强，再考虑 partial split（`answer_split_power`）或按 chunk diversity 分配 mass。

## 2026-08-02 support-flow posterior-mass no-split v14 smoke

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v14_mid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_posteriormass_nosplit_v14_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v14_mid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v14_mid_c128_b32_r32_v64_3step_20260802.jsonl`

相对 v13 的唯一核心变化：

- `ttrl.chunk_state_support_flow_posterior_split_duplicates=False`
- 同一 full-rollout support answer 下的多个 anchor chunk 不再平分 answer posterior mass；每个来自高 posterior answer 的 chunk 都可作为有效 local transition target。
- 仍然保持 `chunk_state_probe/skipped_for_support_flow=1.0`，不回到 short-probe teacher。

三步质量汇总：

```text
step  split  anchor_inj  anchor_mass  uniq_answer  post_mean  post_max  source_mass  margin  score  label  answer_cov  pos_ratio  entropy  samples  actor_w_nz  update_actor
1     0      1.000       0.298        0.732        0.261      0.625     0.708        -0.083  0.261  0.875  0.875       0.261      1.217    24       1.000       1.350
2     0      0.982       0.388        0.571        0.334      0.613     0.698        -0.085  0.334  0.859  0.859       0.334      1.502    24       1.000       0.995
3     0      1.000       0.351        0.661        0.307      0.706     0.706         0.000  0.307  0.875  0.875       0.307      1.282    16       1.000       0.767
mean  0      0.994       0.346        0.655        0.301      0.648     0.704        -0.056  0.301  0.870  0.870       0.301      1.334    21.3     1.000       1.037
```

耗时：

```text
step  gen      chunks  score  ref    update_actor
1     43.480   1.140   7.634  4.262  1.350
2     23.011   0.939   5.995  0.373  0.995
3     22.359   1.045   5.712  0.289  0.767
mean  29.617   1.041   6.447  1.641  1.037
```

与 v13 对比：

```text
metric                       v13 split  v14 no-split
posterior_mass_mean          0.064      0.301
posterior_mass_max_mean      0.200      0.648
label_consistent_ratio       0.656      0.870
answer_coverage_mean         0.656      0.870
positive_ratio               0.064      0.301
target_entropy               1.639      1.334
actor_w_nonzero              1.000      1.000
update_actor                 1.073s     1.037s
```

结论：

- v14 是当前最好的 chunk-level search-state TTRL target 版本。
- 它真正放弃了 short-probe teacher，同时比 v13 修复了 target 过弱问题。
- 三步都没有 v12 的零权重 update，`answer_coverage_mean=0.870`，`positive_ratio=0.301`，target 强度足够进入 20-step pilot。
- no-split 的解释是合理的：训练目标是 local transition improvement，不是 prompt-level answer probability conservation；同一个高 posterior answer 下的多个不同 chunk 都可能是有效的下一段推理转移。
- 注意：末尾有 `Exception ignored in atexit callback ... DataLoader worker ... killed`，但主进程 exit code 为 0，且 `Final validation skipped` 是预期 smoke 配置。先记录为非阻塞退出清理噪声。

下一步：

- 扩 v14 到 20-step pilot，保留 batch32/rollout32/votes64、8 卡、dynamic bsz off。
- step 20 做 final validation，观察 mean@16/maj@16 是否优于 MV 对齐链路。
- 若 20-step 中 `posterior_mass_mean` 保持 0.25-0.35、`actor_w_nonzero=1.0`，再扩 80-step。

## 2026-08-02 v14 20-step partial 与 v15 nonzero-mid 修复

背景：

- 用户明确要求放弃“局部短视可判定性”：chunk target 不应主要由 short-horizon probe 的局部命中、source answer consistency 或短 probe answer distribution 来定义。
- 因此 v13/v14 后续都保持 `chunk_state_score_mode=support_flow` + `posterior_mass`，并且 `chunk_state_probe/skipped_for_support_flow=1.0`。
- source chunk / support anchor 只作为 proposal/prior/drift guard；teacher target 来自 prompt-level full-rollout answer posterior/support，而不是 short-probe hit。

### v14 no-split 20-step partial

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v14_mid_c128_b32_r32_v64_20step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_posteriormass_nosplit_v14_20step_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v14_mid_c128_b32_r32_v64_20step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v14_mid_c128_b32_r32_v64_20step_20260802.jsonl`

结果：

- 运行到 step 18 后失败，未产出 final validation。
- 失败点：

```text
RuntimeError: chunk-state source selection produced no valid states;
min_required_response_len=1024, boundaries=[0, 256, 512, 768, 1024]
```

v14 partial 聚合：

```text
steps                                      18
posterior_mass_mean                       0.2710
posterior_mass_max_mean                   0.5793
source_mass_mean                          0.7359
label_consistent_ratio                    0.7301
boundary_zero_ratio                       0.0486
real_state_count                          3.2222
answer_coverage_mean                      0.7396
positive_ratio                            0.2710
num_actor_samples                         25.7778
actor_batch_powerflow_weight_nonzero      1.0000
gen                                       25.5223s
chunk_state_chunks                         0.9713s
chunk_state_score                          7.4726s
chunk_state_ref                            0.6088s
update_actor                               1.1842s
```

结论：

- v14 的 full-posterior no-split target 语义仍然正确：全程 `skipped_for_support_flow=1.0`，没有回到 short-probe teacher。
- 但工程实现有一个不合理硬约束：`mid` boundary 预筛用了最大 boundary 1024 作为 `min_required_response_len`，导致某些 batch 明明有 256/512 这类可用中间 state，却被认为没有 valid state。
- 另一个问题是 `mid` fallback 仍可能退回 boundary 0，和“中后段 search state”的实验设计不一致。
- 低信息 / 无有效 state batch 不应该杀掉训练；它应该被跳过并记录。

### v15 nonzero-mid + skip-empty 20-step

代码修复：

- `verl/trainer/ppo/ray_trainer.py`
  - 新增 `EmptyChunkStateBatchError`，将“没有可训练 chunk state”从普通 `RuntimeError` 区分出来。
  - 新增 `ttrl.chunk_state_mid_require_nonzero_boundary`：mid mode 下要求使用非零 chunk boundary；如果没有非零 mid boundary，该 source 被跳过。
  - 修正 `min_required_response_len`：从“最大 boundary”改为“可用最小 boundary”，避免要求所有 source 都长到 1024。
  - 新增 `ttrl.chunk_state_skip_empty_state_batch`：当整个 batch 没有有效 state 时，记录 `chunk_state/empty_state_batch_skipped=1.0` 并跳过 actor update，而不是中断实验。
- `verl/trainer/config/ppo_trainer_ttrl.yaml`
  - 新增上述两个配置项，默认关闭，保持历史兼容。

实验：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v15_nonzeromid_skipempty_c128_b32_r32_v64_20step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_posteriormass_nosplit_v15_nonzeromid_skipempty_20step_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v15_nonzeromid_skipempty_c128_b32_r32_v64_20step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v15_nonzeromid_skipempty_c128_b32_r32_v64_20step_20260802.jsonl`
- val json: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v15_nonzeromid_skipempty_c128_b32_r32_v64_20step_20260802_val_metrics.json`

关键配置：

```text
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
ttrl.n_votes_per_prompt=64
ttrl.n_samples_per_prompt=32
ttrl.chunk_state_score_mode=support_flow
ttrl.chunk_state_support_flow_score_type=posterior_mass
ttrl.chunk_state_support_flow_posterior_split_duplicates=False
ttrl.chunk_state_mid_require_nonzero_boundary=True
ttrl.chunk_state_skip_empty_state_batch=True
actor_rollout_ref.actor.powerflow_enable=True
actor_rollout_ref.actor.powerflow_use_chunk_weights=True
actor_rollout_ref.actor.use_dynamic_bsz=False
```

v15 训练聚合：

```text
steps                                      20
trainable chunk steps                      19
empty_state_batch_skipped                   1
posterior_mass_mean                       0.4724
posterior_mass_max_mean                   0.6928
source_mass_mean                          0.7222
label_consistent_ratio                    0.8692
boundary_zero_ratio                       0.0000
real_state_count                          3.8947
answer_coverage_mean                      0.8733
positive_ratio                            0.4724
num_actor_samples                         31.1579
actor_batch_powerflow_weight_nonzero      1.0000
gen                                       25.0278s
chunk_state_chunks                         1.0574s
chunk_state_score                          6.4636s
chunk_state_ref                            0.6486s
update_actor                               1.2274s
final_validation                         301.4090s
```

v15 validation：

```text
mean@16        0.398375
maj@16         0.502288
best@16        0.804986
format mean@16 0.877750
format maj@16  0.837458
```

与 v14 partial 对比：

```text
metric                       v14 partial  v15 nonzero-mid
posterior_mass_mean          0.2710       0.4724
posterior_mass_max_mean      0.5793       0.6928
label_consistent_ratio       0.7301       0.8692
answer_coverage_mean         0.7396       0.8733
boundary_zero_ratio          0.0486       0.0000
real_state_count             3.2222       3.8947
num_actor_samples            25.7778      31.1579
update_actor                 1.1842s      1.2274s
```

结论：

- v15 工程修复是有效的：真正消除了 boundary 0，避免了 no-valid-state crash，并且 target 质量指标显著高于 v14。
- 但 v15 的最终 acc 很差，20-step 后 `mean@16=0.398`、`maj@16=0.502`、`best@16=0.805`，明显不接近 MV / PowerFlow 参考轨迹。
- final validation 日志出现大量重复 `\boxed{}`，说明当前 PowerFlow chunk actor update 虽然 target 更干净，但训练行为破坏了输出分布或格式稳定性。
- 因此，下一步不应简单扩 80-step。当前版本适合作为“full-posterior target + nonzero-mid infra”正交组件保留，但 loss/weighting 需要重新设计。

下一步判断：

- 保留：`support_flow + posterior_mass`、`posterior_split_duplicates=False`、nonzero mid boundary、empty-state skip、dynamic bsz off。
- 暂停扩展：当前 `powerflow_use_boxed_reward=True` + chunk weights 的直接 PowerFlow 更新。
- 优先排查：
  - 为什么 PowerFlow chunk update 诱发重复 boxed 污染。
  - 是否需要改成 target-only KL/distillation 或 supervised distribution matching，而不是当前 `boxed_reward` 注入式 PowerFlow loss。
  - 是否要对 chunk update 加格式/重复 marker guard，或让 source chunk 仅作为 prior 而不是 response span 的高权重训练样本。
  - 20-step smoke 的准入门槛不能只看 target 质量，还必须看 validation format collapse。

### 当前策略纠偏

这条线下一步要明确放弃的不是 PowerFlow 骨架、hardfilter/clip4、full-rollout label estimation，也不是 chunk-level search-state training 本身，而是“局部短视可判定性”这个训练约束：

```text
不要再要求 chunk target 主要由 short-horizon probe 的局部 answer hit、
source answer consistency、或短 probe answer distribution 来定义。
```

原因：

- v12/v13/v14/v15 已经反复显示，训练更新速度不是主矛盾。hardfilter/clip4/nonzero-mid 后 `update_actor` 可以稳定在约 `1.0-1.5s`。
- 真正没有解决的是 target 质量和训练行为：support/OOV/format collapse 指向同一个问题，即局部 probe 信号不是可靠的 search-improvement target。
- sourcegate/source consistency 方向更强时会把 support coverage 拉低、OOV 拉高，说明继续加 source 侧硬约束不是主解。
- v13-v15 的 `support_flow + posterior_mass` 方向之所以更合理，是因为它让 full-rollout group posterior/support 主导 target，probe 被跳过或降级为 future distribution estimator，而不是 teacher。

下一版 target 原则：

- full rollout group 先定义 prompt-level answer support posterior/value。
- chunk state 只学习哪个 local transition 会把未来分布推向这个 full-group posterior/value。
- source chunk / support anchor 只作为 proposal、prior 或 drift guard，不作为 teacher floor。
- low-information state 直接跳过或强降权，包括 all-negative、low support coverage、high OOV、flat posterior、malformed/repeated boxed/marker contamination。
- probe 如继续使用，只能用于估计 future distribution 或 longer-horizon value，不再单独决定 teacher。

因此 v16 不应简单扩展 v15，也不应回退到 short-probe local hit。优先方向是把当前 `boxed_reward` 注入式 PowerFlow 更新替换为更稳定的 full-posterior distribution matching / target-only KL / guarded PowerFlow variant，并把 validation format collapse 作为 smoke 准入指标。

## 2026-08-02 support-flow posterior-mass no-split v16 target-only smoke

目的：

- 保留 v15 的 `support_flow + posterior_mass`、`posterior_split_duplicates=False`、nonzero mid boundary、skip-empty。
- 只切 actor loss：`actor_rollout_ref.actor.powerflow_chunk_loss_mode=target_only`，并设置 `actor_rollout_ref.actor.powerflow_use_boxed_reward=False`。
- 目标是验证 full-posterior chunk target 可以进入 PowerFlow 权重，但不再通过 `boxed_reward` 注入 residual，避免 v15 20-step 后出现的重复 `\boxed{}` 污染。

文件：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v16_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_posteriormass_nosplit_v16_targetonly_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v16_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v16_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802.jsonl`

配置确认：

```text
actor_rollout_ref.actor.powerflow_chunk_loss_mode=target_only
actor_rollout_ref.actor.powerflow_use_boxed_reward=False
ttrl.chunk_state_score_mode=support_flow
ttrl.chunk_state_support_flow_score_type=posterior_mass
ttrl.chunk_state_support_flow_posterior_split_duplicates=False
ttrl.chunk_state_mid_require_nonzero_boundary=True
ttrl.chunk_state_skip_empty_state_batch=True
actor_rollout_ref.actor.use_dynamic_bsz=False
```

3-step 聚合：

```text
steps                                      3
posterior_mass_mean                       0.6050
posterior_mass_max_mean                   0.7230
source_mass_mean                          0.7230
label_consistent_ratio                    0.8750
answer_coverage_mean                      0.8750
positive_ratio                            0.6050
num_actor_samples                         45.3333
actor_powerflow_chunk_loss_target_only    1.0000
actor_powerflow_loss                      0.5200
actor_grad_norm                          26.6520
gen                                      31.1797s
chunk_state_chunks                        0.9267s
chunk_state_score                         6.5837s
chunk_state_ref                           1.9170s
update_actor                              1.7457s
```

观察：

- `actor/powerflow_chunk_loss_target_only=1.0` 三步全生效，说明 actor 端确实走 target-only 路径。
- `chunk_state_probe/skipped_for_support_flow=1.0`，仍没有回到 short-probe teacher。
- 日志中的 `boxed_reward_mean` 仍非零，是 trainer 为兼容现有 actor batch/metrics 继续写入 support-flow score；在 `target_only` 模式下它不进入 PowerFlow residual。
- target 质量指标健康：`posterior_mass_mean=0.605`、`label_consistent_ratio=0.875`、`answer_coverage=0.875`。
- 3-step 没有 final validation，不能判断 v15 的格式崩坏是否解决。

下一步：

- 跑同配置 20-step + final val，准入指标不只看 target 质量，还要看 `mean@16/maj@16/best@16` 与 validation sample 是否仍出现重复 `\boxed{}` 污染。

## 2026-08-02 support-flow posterior-mass no-split v16 target-only 20-step

目的：

- 在 v16 smoke 通过后，跑同配置 20-step + final validation。
- 核心约束保持不变：chunk target 由 full-rollout group posterior/support 主导，`chunk_state_probe/skipped_for_support_flow=1.0`，不回退到 short-horizon probe answer hit / source consistency teacher。
- 只验证 `target_only + powerflow_use_boxed_reward=False` 是否能缓解 v15 的重复 `\boxed{}` 污染。

文件：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v16_targetonly_nonzeromid_c128_b32_r32_v64_20step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_posteriormass_nosplit_v16_targetonly_20step_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v16_targetonly_nonzeromid_c128_b32_r32_v64_20step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v16_targetonly_nonzeromid_c128_b32_r32_v64_20step_20260802.jsonl`
- final val metrics: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v16_targetonly_nonzeromid_c128_b32_r32_v64_20step_20260802_val_metrics.json`

20-step 聚合：

```text
train metric                                mean       last
gen                                      24.9589s   22.6930s
chunk_state_chunks                       0.9611s    0.9870s
chunk_state_score                        6.3705s    6.8710s
chunk_state_ref                          0.6512s    0.5620s
update_actor                             1.2432s    1.7930s
posterior_mass_mean                      0.4485     0.4630
posterior_mass_max_mean                  0.7184     0.7640
source_mass_mean                         0.7346     0.7640
positive_ratio                           0.4485     0.4630
label_consistent_ratio                   0.8717     0.8750
answer_coverage_mean                     0.8734     0.8750
num_actor_samples                       32.0000    40.0000
real_states                              4.0000     5.0000
target_entropy                           1.5395     1.5140
actor_powerflow_loss                     0.7099     0.2140
actor_grad_norm                         21.3349    16.6950
actor_powerflow_chunk_loss_target_only   1.0000     1.0000
testing                                300.1980s  300.1980s
```

final validation:

```text
val-core/math/acc/mean@16           0.440250
val-core/math/acc/maj@16/mean       0.560144
val-core/math/acc/best@16/mean      0.840186
format mean@16                      0.890125
format maj@16                       0.855556
format worst@16                     0.444788
```

与 v15 对比：

```text
metric                       v15 boxed    v16 target-only
mean@16                      0.398375     0.440250
maj@16                       0.502288     0.560144
best@16                      0.804986     0.840186
format mean@16               0.877750     0.890125
format maj@16                0.837458     0.855556
update_actor mean            1.2274s      1.2432s
posterior_mass_mean          0.4724       0.4485
label_consistent_ratio       0.8692       0.8717
answer_coverage_mean         0.8733       0.8734
```

结论：

- v16 相比 v15 有小幅恢复，说明去掉 `boxed_reward` residual、改成 target-only 是正确方向，但效果仍远低于可接受的 MV / PowerFlow 参考轨迹。
- 训练侧速度不是主矛盾：`update_actor` 平均约 `1.24s`，chunk update 只有 128 token span，B200 上 actor 更新已经足够快。
- target 语义已经基本符合当前纠偏：`support_flow + posterior_mass`、`posterior_split_duplicates=False`、`chunk_state_probe/skipped_for_support_flow=1.0`，没有让 short probe 局部命中信号做 teacher。
- 但 final validation 仍出现严重重复 `\boxed{}` 和题面污染，raw log 中连续空 `\boxed{}` 片段约 4 万处；`format worst@16=0.4448` 也说明尾部样本退化很严重。
- 因此，v16 失败点不再是“target 由 short probe 定义”，而是 PowerFlow-style chunk update 即便 target-only，也会在当前权重/目标形态下造成局部 continuation 分布漂移，模型学到重复 marker / prompt echo 的坏模式。

下一步方向：

- 不回退到 short-horizon probe teacher，也不继续加强 sourcegate/source consistency 硬约束。
- 保留 full-rollout posterior/support target，但 actor update 要加 distribution-level guard：例如 KL-to-ref / entropy floor / repetition penalty mask / marker contamination filter / no-repeat boxed 负样本过滤。
- target 不应只按 posterior mass sharpen，还要显式惩罚 malformed、repeated boxed、prompt echo、长重复片段，把这些作为 low-information 或 negative transition 过滤掉。
- 20-step smoke 的准入门槛改成两类同时通过：target 质量指标健康，且 validation sample 无明显 repeated `\boxed{}`/prompt echo collapse。

## 2026-08-02 support-flow posterior-mass no-split v17 guarded target-only 3-step smoke

目的：

- v16 已证明 `target_only + powerflow_use_boxed_reward=False` 比 v15 有小幅恢复，但 final validation 仍有严重重复 `\boxed{}` / prompt echo。
- 本次 v17 只做 3-step smoke，不做 accuracy 判断；3 step 的作用是跨过首步 Ray/vLLM/FSDP/compile warmup，确认 no-probe support-flow 路径下 target/candidate guard 连续 step 可运行，并检查 guard 是否真的过滤候选。
- 核心原则不变：target 仍由 full-rollout group posterior/support 定义，`chunk_state_probe/skipped_for_support_flow=1.0`，不让 short-horizon local hit / source consistency 重新成为 teacher。

文件：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v17_guardcand_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_posteriormass_nosplit_v17_guardcand_targetonly_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v17_guardcand_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v17_guardcand_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802.jsonl`

代码改动：

- `_apply_chunk_state_target_guard(...)` 支持 `probe_output is None`，这样 support-flow / support-anchor 这种跳过 short probe 的路径也可以执行 candidate-level guard。
- 配置注释补充：support-flow 无 probe 时，candidate guard 可以作为唯一 target guard；这不是恢复 short-probe teacher。

关键配置：

```text
ttrl.chunk_state_score_mode=support_flow
ttrl.chunk_state_support_flow_score_type=posterior_mass
ttrl.chunk_state_support_flow_posterior_split_duplicates=False
ttrl.chunk_state_target_guard_enable=True
ttrl.chunk_state_target_guard_candidate_enable=True
ttrl.chunk_state_target_guard_candidate_max_boxed_count=1
ttrl.chunk_state_target_guard_candidate_assistant_marker=True
ttrl.chunk_state_zero_inconsistent_candidates=True
ttrl.chunk_state_prune_zero_weight_samples=True
actor_rollout_ref.actor.powerflow_chunk_loss_mode=target_only
actor_rollout_ref.actor.powerflow_use_boxed_reward=False
actor_rollout_ref.actor.use_dynamic_bsz=False
trainer.total_training_steps=3
trainer.final_val_enable=False
```

3-step 聚合：

```text
metric                                      mean      last
gen                                      33.0453s  22.9620s
chunk_state_chunks                       0.9360s   0.9330s
chunk_state_score                        6.5283s   5.6110s
chunk_state_ref                          1.6627s   0.2900s
update_actor                             1.1930s   0.8410s
real_states                              4.3333    3.0000
num_actor_samples                       26.6667   16.0000
pruned_sample_ratio                      0.5833    0.7500
posterior_mass_mean                      0.5473    0.5490
posterior_mass_max_mean                  0.7223    0.7080
source_mass_mean                         0.7223    0.7080
label_consistent_ratio                   0.8697    0.8590
answer_coverage_mean                     0.8697    0.8590
target_entropy                           1.7140    1.7380
powerflow_weight_max                     1.9250    3.1870
actor_powerflow_loss                     1.1827    0.7720
actor_grad_norm                         28.1563   17.9860
target_guard_kept_candidate_ratio        1.0000    1.0000
target_guard_zeroed_candidate_ratio      0.0000    0.0000
candidate_repeated_boxed_probe_ratio     0.0000    0.0000
candidate_assistant_marker_probe_ratio   0.0000    0.0000
candidate_prompt_copy_probe_ratio        0.0000    0.0000
```

观察：

- v17 smoke 跑通，3 个 step 都确认 `actor/powerflow_chunk_loss_target_only=1.0`，且 `chunk_state_probe/skipped_for_support_flow=1.0`，没有回到 short-probe teacher。
- target/candidate guard 在 no-probe support-flow 路径能正常打点，不再因为 `probe_output=None` 被跳过。
- actor update 不是瓶颈：三步均值约 `1.19s`，第 2/3 步约 `1.03s`；chunk span 短时 B200 actor update 已足够快。
- 主要端到端时间仍在 full rollout generation 与 chunk score：三步 `gen` 均值约 `33.0s`，第 2/3 步约 `27.8s`；`chunk_state_score` 约 `6.5s`。
- guard 现在基本是 no-op：`zeroed_candidate_ratio=0`，repeated boxed / assistant marker / prompt copy 三类 candidate 指标也全是 0。
- 这说明 v17 只证明了“guard wiring 正常”，没有证明它能挡住 v16 final validation 中真正出现的重复 `\boxed{}` / prompt echo 退化。

结论：

- 不直接扩 v17 到 20-step。当前 gate 未通过，原因不是链路崩溃，而是 guard 太弱，没有覆盖最终生成阶段暴露的坏模式。
- 下一版应改检测对象：不能只在 candidate chunk 里看局部 marker；需要在 actor batch 和/或定期 validation sample 中统计连续 `\boxed{}`、空 boxed、题面复制、assistant/user marker、长 n-gram 重复，并把这些信号作为 target transition 的 hard filter 或 weight penalty。
- 仍然保留 full-rollout posterior/support target 和 PowerFlow target-only loss；不回退到 short-horizon probe answer hit，也不继续加强 source-side hard gate。

## 2026-08-02 support-flow posterior-mass no-split v18 pollution-guard target-only 3-step smoke

目的：

- 继续 v17 的结论：guard wiring 已经可运行，但过弱、基本 no-op。
- v18 只加强 candidate pollution guard，检测对象对齐 v16 final validation 暴露的坏模式：空 `\boxed{}`、重复 `\boxed`、Human/Assistant/User/System marker、`\end{document}`、重复 `[asy]`、长 n-gram 重复、从 state prompt 复制题面。
- 不改变 target 语义：仍是 `support_flow + posterior_mass`，`chunk_state_probe/skipped_for_support_flow=1.0`，不回退到 short-probe answer hit / source consistency teacher。

文件：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v18_pollutionguard_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_posteriormass_nosplit_v18_pollutionguard_targetonly_smoke_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v18_pollutionguard_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v18_pollutionguard_targetonly_nonzeromid_c128_b32_r32_v64_3step_20260802.jsonl`

代码改动：

- `chunk_state_target_guard_candidate_*` 增加污染检测配置：
  - `candidate_empty_boxed`
  - `candidate_human_marker`
  - `candidate_document_marker`
  - `candidate_asy_repeat`
  - `candidate_ngram_repeat`
  - `candidate_prompt_copy` with prompt/chunk n-gram overlap
- 修正 candidate marker 正则，使 `Assistant:` / `Human:` 这类真实污染能被匹配。
- `_apply_chunk_state_target_guard(...)` 在 candidate guard 中解码 state prompt 文本，用于 prompt-copy overlap 检测。

3-step 聚合：

```text
metric                                      mean      last
gen                                      29.6520s  22.2130s
chunk_state_chunks                       0.9537s   0.9440s
chunk_state_score                        6.4907s   5.6960s
chunk_state_ref                          1.7950s   0.1240s
update_actor                             1.2690s   0.4630s
real_states                              5.3333    2.0000
num_actor_samples                       32.0000    8.0000
pruned_sample_ratio                      0.6457    0.8750
posterior_mass_mean                      0.5980    0.6950
posterior_mass_max_mean                  0.7233    0.7940
source_mass_mean                         0.7367    0.7940
label_consistent_ratio                   0.8723    0.8750
answer_coverage_mean                     0.8750    0.8750
target_entropy                           1.8400    1.9270
actor_powerflow_loss                     1.5620    1.2140
actor_grad_norm                         38.2310   13.2720
target_guard_zeroed_candidate_ratio      0.0107    0.0160
candidate_prompt_copy_ratio              0.0053    0.0000
candidate_repeated_boxed_ratio           0.0053    0.0160
candidate_empty_boxed_ratio              0.0000    0.0000
candidate_marker_ratio                   0.0000    0.0000
candidate_ngram_repeat_ratio             0.0000    0.0000
```

逐步 guard 触发：

```text
step1 zeroed=0/64   all candidate pollution metrics 0
step2 zeroed=2/128  candidate_prompt_copy_ratio=0.016
step3 zeroed=1/64   candidate_repeated_boxed_ratio=0.016
```

观察：

- v18 smoke 跑通，三步均保持 `actor/powerflow_chunk_loss_target_only=1.0` 和 `chunk_state_probe/skipped_for_support_flow=1.0`。
- guard 不再是 no-op：step2/step3 分别过滤 prompt-copy 和 repeated-boxed 候选。
- 过滤强度很轻，三步总共过滤 3 个候选；actor batch 没有被剪空，`num_actor_samples` 分别为 32 / 56 / 8，非零权重比例仍为 1。
- 触发率低说明 v18 只是接通了污染检测，并不能单独证明能解决 v16 的 20-step validation collapse；真正坏模式可能主要在训练后完整生成尾部出现，而不是每步 128-token candidate 中高频出现。
- v18 启动日志显示 vLLM 路径启用 `attention_config.backend=FLASH_ATTN`，并进行 CUDA graph capture；本次不是为了 infra speed，但 smoke 使用的是当前 B200 快链路。

结论：

- v18 可以作为下一次 20-step 候选，但准入条件必须包含 final validation pollution audit，不能只看 train-time guard 触发率。
- 如果 20-step 仍出现重复 `\boxed{}` / prompt echo，下一步应加 validation/rollout-level pollution monitor，并考虑把完整 rollout 中的污染轨迹降权或从 source/support pool 中移除；只在 128-token candidate 上做 hard filter 可能覆盖不够。
- 仍然不回退到 short-horizon probe teacher，也不引入 source consistency 硬 teacher。

## 2026-08-02 support-flow posterior-mass no-split v18 pollution-guard target-only 20-step final validation

目的：

- 把 v18 从 3-step smoke 扩到 20-step，验证 candidate pollution guard 是否能阻止 v16 看到的重复 `\boxed{}` / prompt echo collapse。
- 仍坚持 24h goal 的核心语义：full-rollout group posterior/support 定义 chunk target，PowerFlow target-only loss 更新 chunk span；不使用 short-horizon probe hit 或 source consistency 当 teacher。
- 只做 20-step + final validation，不保存 ckpt，输出路径放在 `/tmp/ttrl_b200/checkpoints/...`，日志和指标放在 `important_experiment_logs/`。

文件：

- launcher: `verl/run_records/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v18_pollutionguard_targetonly_nonzeromid_c128_b32_r32_v64_20step_20260802.sh`
- 前台 worker helper: `verl/run_records/run_front_supportflow_posteriormass_nosplit_v18_pollutionguard_targetonly_20step_20260802.sh`
- raw log: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v18_pollutionguard_targetonly_nonzeromid_c128_b32_r32_v64_20step_20260802.log`
- diag: `important_experiment_logs/chunk_state_diag/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v18_pollutionguard_targetonly_nonzeromid_c128_b32_r32_v64_20step_20260802.jsonl`
- final val metrics: `important_experiment_logs/ttrl_chunk_state_powerflow_supportflow_posteriormass_nosplit_v18_pollutionguard_targetonly_nonzeromid_c128_b32_r32_v64_20step_20260802_val_metrics.json`

关键配置：

```text
data.train_batch_size=32
actor_rollout_ref.rollout.n=32
ttrl.n_votes_per_prompt=64
ttrl.n_samples_per_prompt=32
ttrl.chunk_state_score_mode=support_flow
ttrl.chunk_state_support_flow_score_type=posterior_mass
ttrl.chunk_state_support_flow_posterior_split_duplicates=False
ttrl.chunk_state_mid_require_nonzero_boundary=True
ttrl.chunk_state_candidates=8
ttrl.chunk_state_chunk_size=128
ttrl.chunk_state_target_guard_candidate_enable=True
ttrl.chunk_state_zero_inconsistent_candidates=True
ttrl.chunk_state_prune_zero_weight_samples=True
actor_rollout_ref.actor.powerflow_enable=True
actor_rollout_ref.actor.powerflow_chunk_loss_mode=target_only
actor_rollout_ref.actor.use_dynamic_bsz=False
trainer.total_training_steps=20
trainer.test_freq=20
trainer.final_val_enable=True
```

20-step final validation：

```text
val-core/math/acc/mean@16      0.469750
val-core/math/acc/maj@16/mean  0.587000
val-core/math/acc/best@16/mean 0.852938
format_score/mean@16           0.903375
format_score/maj@16/mean       0.885464
format_score/worst@16/mean     0.473668
testing                        298.723s
```

对比 v16 target-only 20-step：

```text
metric                         v16        v18
mean@16                       0.440250   0.469750
maj@16                        0.560144   0.587000
best@16                       0.840186   0.852938
format mean@16                0.890125   0.903375
format maj@16                 0.855556   0.885464
format worst@16               0.444788   0.473668
```

训练段聚合（step 1-19，不含 final validation step 20 的 `testing`）：

```text
metric                                      mean       min       max      last
timing_s/gen                              25.474s   21.553s   43.397s  22.229s
timing_s/generate_sequences               18.532s   17.393s   29.899s  17.393s
timing_s/chunk_state_score                 6.412s    5.633s    7.788s   6.165s
timing_s/chunk_state_chunks                0.984s    0.917s    1.116s   0.918s
timing_s/chunk_state_ref                   0.614s    0.132s    4.372s   0.547s
timing_s/update_actor                      1.114s    0.413s    1.832s   1.455s
chunk_state/num_actor_samples             27.789     8.000    48.000   40.000
chunk_state/real_states                    4.579     2.000     7.000    6.000
chunk_state/pruned_sample_ratio            0.566     0.250     0.875    0.375
posterior_answer_duplicate_ratio           0.612     0.146     0.857    0.571
chunk_state/target_entropy                 1.598     0.863     1.946    1.554
target_guard_zeroed_candidate_ratio        0.023     0.000     0.109    0.000
```

guard 汇总：

```text
zeroed_candidates = 29 / 1280
zeroed_candidate_ratio ~= 2.27%
```

观察：

- v18 20-step 跑通并完成 final validation，Ray/runtime/env 没有失败。
- 当前 B200 快链路正常启用 vLLM CUDA graph，训练启动日志显示 `attention_config.backend=FLASH_ATTN`，actor 侧也有 flash attention monkey patch 和 fused kernel 日志。
- actor update 已经不是瓶颈：训练段 `update_actor` 均值约 `1.11s`，chunk span actor update 能吃到短序列收益。
- 主要耗时仍在 full rollout generation 与 group posterior scoring：`gen` 均值约 `25.47s`，`chunk_state_score` 均值约 `6.41s`；端到端进度受 math verifier timeout 和长生成拖动。
- v18 相对 v16 有小幅改善：mean@16 +2.95pt，maj@16 +2.69pt，best@16 +1.28pt，format worst@16 +2.89pt。
- 但 validation 日志仍然出现长段重复提示语和重复 `\boxed{}` 风格输出；raw log 中 `Please provide a step-by-step explanation` 重复出现 534 次。说明 v18 的 128-token candidate guard 只挡住了一部分污染，不能解决完整 rollout 尾部的退化。
- guard 触发率偏低：只 zero 29/1280 candidates。它更像一个轻量污染拦截器，不是足够强的 target-quality 修复。

结论：

- v18 不是最终方向，只能说明“candidate pollution guard 有效但覆盖不足”。最终指标仍远低于 major-vote / paper-style 20-step 目标，也低于我们希望的 85+ mean@16。
- 下一步不应继续堆 source hard gate，也不应回到 short-horizon local hit teacher。
- 更合理的 v19 方向：把 pollution/value filter 前移到 full rollout group 和 source/support pool 层面。完整 rollout 一旦出现 prompt echo、重复 boxed、marker 污染、过长复制，应从 group posterior / anchor pool 中剔除或强降权；否则污染轨迹仍会通过 source/support anchor 进入 chunk target。
- 同时应把 target 从纯 `posterior_mass` 改到更接近 search improvement 的 `posterior_gain` 或 support-value margin：保留 full-group posterior，但要求 candidate chunk 提升未来分布相对 source 的质量，而不是只匹配已有高 mass answer。
