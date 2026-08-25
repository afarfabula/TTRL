<!--
This section documents the local experiment branch maintained at
experiment/ttrl-workspace-history-20260807. The original upstream TTRL README
is kept below for paper/project context.
-->

# 本分支 Claims：Qwen2.5-Math TTRL 主闭环与 Qwen3 扩展探索

本分支围绕 TTRL（Test-Time Reinforcement Learning）在数学推理上的复现、算法改造和大模型工程化展开。主线不是只跑通论文脚本，而是在 Qwen2.5-Math-7B + MATH-TTT/MATH500 上建立一条可复跑、可诊断、可对照的无 GT Test-Time RL 实验闭环；Qwen3-30B-A3B 相关内容作为后续 MoE 扩展探索单独记录。

## Claim 1：构建 Qwen2.5-Math-7B / MATH500 无 GT TTRL 实验闭环，并验证 sharpened MV-anchor reward

本分支在 8xB200 上围绕 Qwen2.5-Math-7B 建立了完整 TTRL 实验闭环：MATH-TTT/MATH500 数据、vLLM rollout、majority pseudo-label、reference logprob、GRPO actor update、validation@16、step-time 分解和诊断日志均可追溯。实验不是单次跑通，而是围绕无 GT reward 设计做了持续迭代：paper-style majority-vote baseline、sharpened MV-anchor、full-rollout posterior target、weighted NLL / PowerFlow objective、chunk-level search-state 和 suffix-level counterfactual 都保留了脚本、日志和阶段性结论。

### 算法设计：sharpened MV-anchor reward

原始 TTRL 依赖同一 prompt 下多条 rollout 的 majority answer 作为 pseudo-label。这个 hard majority reward 稳定，但信息利用率低：非 majority 的候选即使有一定 self-consistency posterior support，也不会贡献训练信号。

本分支实现的 sharpened MV-anchor reward 保持 TTRL/GRPO 主训练路径不变，只替换 reward construction：

- 对每个 prompt 的 `n=32` 条 rollout 抽取答案并做规范化，按答案聚合成 answer posterior。
- 当存在 `rollout_log_probs` 时，用 response 平均 logprob 作为 confidence，对同一答案的 rollout 做加权聚合。
- 用 `top_prob >= tau_pos` 和 `margin >= tau_marg` 过滤低置信 prompt，避免把分歧过大的 answer posterior 注入 actor update。
- 对 accepted posterior 做 alpha sharpening，放大高置信答案和低置信答案之间的差异。
- `mv_anchor` 模式下，majority-vote 命中的样本保持 reward=1；非 majority 但 posterior 支持的样本只获得 `soft_coef` 量级的小正 reward。
- 最佳 run 关闭 negative reward，避免过早压制探索 diversity。

这条设计的核心取舍是：用 majority-vote answer 保住 TTRL 的稳定 anchor，同时把 answer posterior 中的软支持作为额外学习信号注入 GRPO。它比纯 hard pseudo-label 更充分利用同组 rollout，也比直接用完整 posterior 替代 majority reward 更保守。

实现位置：

- reward 构造：`verl/verl/trainer/ppo/ttrl_utils.py`
- 训练接入：`verl/verl/trainer/ppo/ray_trainer.py`
- 配置项：`verl/verl/trainer/config/ppo_trainer_ttrl.yaml`
- 最优 run 脚本：`verl/run_records/ttrl_sharpened_grpo_b32_r32_v64_150step_paperstyle_seedfix_mvanchor_soft002_20260731.sh`

### 主结果

| Run | Setting | Step | mean@16 | maj@16 | best@16 |
| --- | --- | ---: | ---: | ---: | ---: |
| Paper-style MV baseline | Qwen2.5-Math-7B, B32/R32/V64, val_n=16 | 80 | `0.8240` | `0.852` | `0.889` |
| Paper-style MV baseline | Qwen2.5-Math-7B, B32/R32/V64, val_n=16 | 150 | `0.8275` | `0.853` | `0.885` |
| Sharpened MV-anchor soft0.02 | Qwen2.5-Math-7B, B32/R32/V64, val_n=16 | 80 | `0.836` | `0.864` | `0.892` |
| Sharpened MV-anchor soft0.02 | Qwen2.5-Math-7B, B32/R32/V64, val_n=16 | 140 | `0.844` | `0.864` | `0.888` |
| Sharpened MV-anchor soft0.02 | Qwen2.5-Math-7B, B32/R32/V64, val_n=16 | 150 | `0.842125` | `0.866578` | `0.881822` |

结论：

- `sharpened MV-anchor soft0.02` 在同一 Qwen2.5-Math-7B / MATH500 设置下稳定超过当前 paper-style MV baseline 的 mean@16 和 maj@16。
- step80 相比 historical MV step80 提升约 `+0.012 mean@16 / +0.012 maj@16 / +0.003 best@16`。
- step140 mean@16 peak 到 `0.844`，step150 保持 `mean@16=0.842125`、`maj@16=0.866578`，是当前 TTRL-native 线中最强的稳定结果之一。
- 边界是明确的：没有达到预设 `mean@16 >= 0.85`，且 best@16 后期下降，说明该 reward 更偏向提升 mean/majority consistency，可能牺牲部分探索多样性。

### Ablation evidence：full-rollout target 与 actor objective 诊断

full-rollout 方向用于验证“能否从完整 rollout 的 answer posterior 构造无 GT 训练 target”。实验记录在 `important_experiment_logs/full_rollout_ttrl_12h_progress_20260802.md`，其中每轮都保留了算法假设、配置、结果、target health、性能分解和下一步决策。

| Run | Target | Actor objective | mean@16 | maj@16 | best@16 |
| --- | --- | --- | ---: | ---: | ---: |
| FR-B0 | posterior sharpen | PowerFlow squared-delta | `0.45325` | `0.57668` | `0.83677` |
| FR-D0 | oracle correctness | PowerFlow squared-delta | `0.398875` | `0.503472` | `0.814348` |
| FR-D0-NLL | oracle correctness | weighted NLL | `0.69275` | `0.797464` | `0.904338` |
| FR-B0-NLL | posterior sharpen, no GT | weighted NLL | `0.68800` | `0.793828` | `0.908718` |
| FR-B2-NLL | margin-aware posterior, no GT | weighted NLL | `0.692875` | `0.794484` | `0.899186` |

诊断结论：

- full-rollout target 构造链路本身可用，oracle-NLL 和无 GT posterior-NLL 能恢复到接近的指标区间。
- 失败点主要在 full-response PowerFlow squared-delta objective，而不是 reward estimation 或 rollout 数据链路完全不可用。
- 后续算法设计应优先围绕 weighted NLL / distribution matching / 更可靠的 target construction，而不是继续扩大这版 full-response squared-delta。
- Oracle correctness 只用于定位问题，不属于正式无 GT 结果。

### Ablation evidence：chunk-level / suffix-level search-state 诊断

chunk-level 和 suffix-level 方向用于探索“能否把完整 rollout 的 future improvement 分解到中间推理状态”。这一方向实现了 mid-state 构造、candidate generation、support mass / future gain / TV transport / support-anchor、suffix-to-EOS counterfactual 等 scorer，并把 chunk target 接入 actor update。

| Direction | Evidence | Result |
| --- | --- | --- |
| chunk-state pipeline | `_make_chunk_state_prompts`、boundary/source metadata、diag JSONL、chunk actor batch 写入 `powerflow_flat_weights` | 多轮 3-step smoke 和 20-step pilot 可运行 |
| support-anchor no-probe | 不依赖 short-probe teacher，score 来自 full rollout support mass | `chunk_state_probe/skipped_for_support_anchor=1.0` |
| support-anchor 20-step | B32/R32/V64, val_n=16 | `mean@16=0.43725` / `maj@16=0.558596` / `best@16=0.83514` |
| suffix-level v28 | suffix-to-EOS + answer-dedup target | `mean@16=0.529875` / `maj@16=0.667382` / `best@16=0.882996` |
| infra overhead | stable chunk-state steps | chunk score 约 `0.001s`，chunk ref 约 `0.965s`，update_actor 约 `3.057s` |

诊断结论：

- 工程链路是正结果：状态构造、candidate、scoring、actor batch、diag JSONL 和多轮 smoke/pilot 已闭环。
- 方法效果是负结果：local-probe / support-anchor / suffix-target 目前没有形成足够可靠的 search-improvement signal，不能声称 chunk-level TTRL 已提升 MATH500。
- 这些负结果仍有价值：它们把下一步问题收敛到 long-horizon target quality、answer duplicate、candidate/anchor selection 和 distribution matching，而不是简单归因于 infra 没跑通。

### AI infra 闭环

为了让算法比较可解释，本分支把 infra 约束显式写入实验语义：

- 固定 B32/R32/V64、`VAL_N=16`、150-step scheduler 和 validation every 20 steps，避免用不同 scheduler phase 的 20-step run 对比 baseline。
- 默认关闭 actor dynamic batch，因为代码审计发现其 microbatch scalar loss 按 sample count 缩放，在 response length 不同时可能改变 token-level GRPO loss 权重。
- 通过 rollout old-logprob diff monitor 验证复用 rollout logprob 的语义安全性，日志中 `training/rollout_probs_diff_*` 保持 `0.000` 时才作为 guardrail。
- 保留 `timing_s/gen`、`timing_s/reward`、`timing_s/ref`、`timing_s/update_actor`、`timing_s/testing` 和 throughput，用于区分算法退化、target 质量问题和系统瓶颈。

### 已同步证据

- 主实验进展：`important_experiment_logs/full_rollout_ttrl_12h_progress_20260802.md`
- chunk/suffix 长实验记录：`important_experiment_logs/chunk_level_search_state_ttrl_20260731.md`
- chunk 24h 审计：`important_experiment_logs/chunk_level_search_state_ttrl_24h_audit_20260801.md`
- MV baseline infra 语义审计：`important_experiment_logs/mv_b32_r32_v64_infra_semantics_20260731.md`
- sharpened MV-anchor 最优 run：`important_experiment_logs/sharpened_mvanchor_soft002_seedfix_20260731.md`
- PowerFlow 对比：`important_experiment_logs/powerflow_80step_trajectory_comparison_20260719.md`
- 原始日志与可复跑脚本：`important_experiment_logs/`、`verl/run_records/`

## Claim 2：Qwen3-30B-A3B MoE 的本地评测和 Expert-Sample 复现完成了可追问的负结果验证

本分支在本地可用的 `/tmp/Qwen3-30B-A3B-Base` 上完成了 MATH500 和 GPQA-Diamond 的 n=16 评测，并实现了 vLLM router 侧 Expert-Sample 环境变量开关。

Evidence：

| Run | Dataset | n | mean_acc | pass@n | elapsed |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B-Base | MATH500 | 16 | `37.25%` | `72.00%` | `3062s` |
| Expert-Sample keep7/pool16/tau0.5 | MATH500 | 16 | `34.59%` | `71.20%` | `4214s` |
| Expert-Sample keep5/pool32/tau0.5 stochastic | MATH500 | 16 | `32.45%` | `71.00%` | `4199s` |
| Qwen3-30B-A3B-Base | GPQA-Diamond | 16 | `28.38%` | `90.40%` | `1054s` |
| Expert-Sample keep5/pool32/tau0.5 stochastic | GPQA-Diamond | 16 | `28.88%` | `89.39%` | `1417s` |

结论：

- vLLM router patch 确认命中过 Expert-Sample 路由逻辑，支持 deterministic 和 stochastic tail sampling。
- 在本地 Base 模型、n=16、无 verifier Best-of-N 的设置下，没有复现论文 headline gain。
- 这是一条有价值的 reproduction audit：早期 subset gain 没有在 full MATH500 / GPQA-Diamond 上保持。

边界：

- 论文 headline 使用 `Qwen3-30B-A3B-Instruct`、`GPQA-Diamond pass@32` 和 verifier Best-of-N；本地只有 Base checkpoint，因此不能声称严格复现失败。
- 该部分应写成“本地可用条件下的复现审计和负结果”，不是论文结论否定。

## Claim 3：Qwen3-30B-A3B TTRL 大模型训练链路完成 smoke，但全参高吞吐训练仍受 sharding/offload 约束

本分支把 Qwen3-30B-A3B MoE 接入 TTRL/verl 训练链路，完成权重、vLLM、rollout、ref logprob 和 actor update 的 2xB200 smoke。

Evidence：

| Setting | Result |
| --- | --- |
| Model | `/tmp/Qwen3-30B-A3B-Base` |
| Hardware | 2 x B200, about 183GB per GPU |
| Working strategy | FSDP2 + `offload_policy=True` |
| Completed training | 1 TTRL training step |
| GPU memory after FSDP2 offload | about `36GB allocated` / `45GB reserved` |
| CPU memory used | about `1374GB` |
| Step time | about `325s`, with `update_actor=256.548s` |

结论：

- 两卡 B200 能跑通 Qwen3-30B-A3B 的 TTRL smoke 闭环。
- FSDP1 `optimizer_offload=True` 不能解决 30B 全参 Adam 首次 state 初始化峰值；FSDP2 offload 才是当前可行 smoke 路线。
- 两卡 B200 更适合兼容性验证，不适合高吞吐全参 RL 训练。

边界：

- 当前 Qwen3-30B-A3B 只是 smoke，不是长训练结果。
- 若要形成最终训练 claim，需要 8 卡 H100/B200 更稳定的 Megatron/FSDP2/ZeRO 路线和完整 validation 曲线。

## 简历写法

围绕 Claim 1 的项目文段：

```text
Qwen2.5-Math-7B Test-Time RL 数学推理优化：基于 TTRL/verl 在 8xB200 上构建 MATH-TTT/MATH500 无 GT 强化学习实验闭环，打通 vLLM rollout、self-consistency majority pseudo-label、reference logprob、GRPO actor update、val@16 评测、step-time/target-health 日志与可复跑 run records。针对原始 TTRL majority-vote reward 信号过硬、非 majority 但高置信候选利用不足的问题，设计 sharpened MV-anchor reward：对同 prompt 的 32 条 rollout 抽取答案并构建 answer posterior，引入 rollout logprob 作为 confidence weight，通过 top-prob/margin gate 过滤低置信 prompt，再做 alpha sharpening；训练时保持 majority answer reward=1 作为稳定 anchor，同时给 posterior 支持的非 majority 答案小权重 soft reward，兼顾 self-consistency 稳定性和软分布学习信号。该方法在 Qwen2.5-Math-7B + MATH500、B32/R32/V64、val_n=16 的 150-step pilot 中达到 mean@16=0.8421、maj@16=0.8666；相较 paper-style MV baseline step80 的 0.8240/0.852/0.889，sharpened MV-anchor step80 达到 0.836/0.864/0.892，mean@16 和 maj@16 均提升约 +1.2pt，并保留 best@16 后期下降、未达 0.85 mean@16 目标等边界分析。围绕该主线进一步完成 full-rollout posterior target、weighted NLL、PowerFlow squared-delta、chunk/suffix search-state target 等 ablation，定位 full-response squared-delta 和 search-state target quality 是主要瓶颈，沉淀为后续 distribution matching / long-horizon target 设计依据。
```

更短的 Claim 1 bullet：

```text
基于 TTRL/verl 构建 Qwen2.5-Math-7B 在 MATH500 上的无 GT Test-Time RL 闭环，设计 sharpened MV-anchor reward，将 majority-vote hard pseudo-label 扩展为 majority anchor + answer-posterior soft support；在 8xB200、B32/R32/V64、val_n=16 设置下，150-step pilot 达到 mean@16=0.8421、maj@16=0.8666，step80 相比 paper-style MV baseline 的 mean@16/maj@16 均提升约 +1.2pt，并通过 full-rollout、PowerFlow、weighted NLL、chunk/suffix search-state ablation 定位后续优化瓶颈。
```

可以直接使用的中文项目经历：

```text
Test-Time RL / GRPO 数学推理后训练：基于 TTRL/verl 构建无 GT 测试时强化学习实验链路，在 8xB200 上复现 Qwen2.5-Math-7B + MATH-TTT/MATH500 的 rollout、majority-vote reward、reference logprob、actor update 和 val@16 评估流程；稳定运行 B32/R32/V64 配置，并通过日志化脚本记录 step time、target health、rollout diff 和验证指标。

设计并验证 sharpened MV-anchor reward：基于同 prompt 多 rollout 的 answer posterior 和 rollout logprob confidence 做 selective sharpening，以 majority-vote answer 作为 reward=1 的稳定 anchor，同时给非 majority 但 posterior 支持的答案小权重软奖励；150-step pilot 达到 mean@16=0.8421、maj@16=0.8666，相比 paper-style MV baseline 在 mean/majority 指标上取得稳定增益。

系统排查 full-rollout PowerFlow 与 chunk-level / suffix-level search-state TTRL：通过 oracle target、posterior target、weighted NLL、PowerFlow squared-delta、support-anchor、future-support gain、suffix-to-EOS counterfactual 等 ablation，定位 full-response squared-delta 和 search-state target quality 是主要瓶颈；将失败路径转化为后续 distribution matching / long-horizon target 设计依据。

扩展到 Qwen3-30B-A3B MoE：完成 Base checkpoint 的 MATH500/GPQA n=16 评测与 vLLM Expert-Sample router patch，验证本地 Base 模型下 Expert-Sample 未复现 headline gain；同时打通 2xB200 Qwen3-30B-A3B TTRL smoke，定位 FSDP1 optimizer offload 不足与 FSDP2 offload 的可行边界。
```

## 训练链路吞吐与稳定性优化

这部分可以作为简历里 Claim 1 的 infra 追问展开：目标不是盲目调快，而是在不改变 TTRL/GRPO 训练语义的前提下，把 Qwen2.5-Math-7B + MATH500 的 8xB200 实验变成可重复迭代的高吞吐闭环。

### 关键优化与取舍

| 方向 | 做法 | 结果 / 结论 |
| --- | --- | --- |
| B200 单机 8 卡环境固化 | 固定 Python venv、CUDA/Ray/vLLM/verl 路径，使用短 `/tmp` runtime 目录规避 Ray Unix socket 路径过长问题 | 环境可复用，避免每次实验重新排查依赖和运行目录 |
| vLLM rollout 快路径 | 保留 vLLM `FLASH_ATTN` backend、CUDA graph capture、fused Triton kernels；不用 `import flash_attn` 作为唯一判断标准 | Qwen2.5-Math-7B rollout generation 稳定在可迭代范围内，日志中保留 backend 和 step timing |
| NCCL / Ray 稳定化 | 单机 B200 使用 `TTRL_FORCE_LOCAL_NCCL=1`、`NCCL_NVLS_ENABLE=1`、`NCCL_P2P_DISABLE=0`、`NCCL_IB_DISABLE=1`、`NCCL_SOCKET_IFNAME=\"=eth0\"`、`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` | 降低多进程/Ray/vLLM 混合训练中的通信和设备可见性问题 |
| 80-step 快速迭代预算 | 从 150-step paper-style run 中抽取 80-step gate：step80 已接近最终主趋势 | 80-step 约 `2.5-2.6h`，适合作为新 reward / target 的中期筛选点 |
| validation 成本拆分 | 单独记录 `timing_s/testing`，避免把 validation step 当成普通训练 step | step80 validation 约 `289s`，普通非 validation step 约 `109-113s`，便于判断真实训练吞吐 |
| actor dynamic batch 审计 | 代码审计发现 dynamic batch 先算 microbatch scalar loss 再按 sample count 缩放，response length 不同时会改变 token-level loss 权重 | 不把 actor dynamic batch 作为默认吞吐优化；报告实验保持 `actor.use_dynamic_bsz=False` |
| old-logprob 复用 guardrail | 加 `training/rollout_probs_diff_max/mean/std` 监控，只有 diff 持续为 `0.000` 时才把 rollout logprob 复用视为语义安全 | 在 sharpened MV-anchor run 中 diff monitor 保持 0，减少重复 logprob 路径的不确定性 |
| fixed microbatch / ref microbatch 探索 | 只允许固定 microbatch 变大这类低风险优化，并要求用 fixed-batch loss/gradient/parameter-delta 等价性验证 | 明确区分“语义安全优化”和“可能改变训练结果的吞吐优化” |
| chunk-state 短 span actor update | 把 chunk target 写入 `powerflow_flat_weights`，用 hardfilter/clip4/nonzero-mid 控制 actor batch 密度和权重尖峰 | 多轮 chunk-state smoke 中 `update_actor` 可稳定到约 `1.0-3.3s`，说明 actor update 不是主要瓶颈 |
| bottleneck attribution | 对 `gen/reward/ref/update_actor/testing/throughput` 做结构化记录 | 能判断瓶颈主要在 full rollout generation、math verifier / support scoring、validation，而不是单纯 actor update |

### 可写进简历的 infra 表述

```text
训练链路吞吐与稳定性优化：在 8xB200 上搭建 Qwen2.5-Math-7B TTRL/MATH500 高吞吐实验环境，固化 Python/CUDA/Ray/vLLM/verl 运行栈和短路径 Ray runtime，配置 vLLM FLASH_ATTN、CUDA graph、Triton fused kernels、NCCL NVLS/P2P 与本地通信参数；将训练日志拆分为 rollout generation、reward/verifier、reference logprob、actor update、validation 和 throughput 指标，建立 80-step gate 作为约 2.5-2.6 小时的算法迭代预算。针对 actor dynamic batch、fixed microbatch、old-logprob reuse、ref/rollout logprob microbatch 等吞吐优化做语义审计，发现 dynamic batch 会因 sample-count 缩放改变 token-level GRPO loss 权重，因此默认禁用；通过 rollout_probs_diff monitor 验证 old-logprob 复用语义安全。chunk-state 方向通过 hardfilter/clip4/nonzero-mid 等机制将短 span actor update 控制到约 1-3s，最终定位系统瓶颈主要在 full rollout generation、verifier/scoring 和 validation，而非 actor update 本身。
```

### 证据文档

- 环境与运行手册：`important_experiment_logs/B200_ENV_USAGE.md`
- MV baseline 与 dynamic batch 语义审计：`important_experiment_logs/mv_b32_r32_v64_infra_semantics_20260731.md`
- chunk-state 24h 吞吐/稳定性审计：`important_experiment_logs/chunk_level_search_state_ttrl_24h_audit_20260801.md`
- chunk/suffix 长进展记录：`important_experiment_logs/chunk_level_search_state_ttrl_20260731.md`
- sharpened MV-anchor timing 与 old-logprob diff：`important_experiment_logs/sharpened_mvanchor_soft002_seedfix_20260731.md`
- full-rollout step timing 与 target health：`important_experiment_logs/full_rollout_ttrl_12h_progress_20260802.md`

更短的英文 bullet：

```text
Built a TTRL/verl-based test-time RL pipeline for mathematical reasoning,
covering rollout generation, majority-vote reward construction, reference
log-prob computation, actor updates, and val@16 evaluation on 8xB200; developed
a sharpened MV-anchor GRPO reward that combines majority-vote anchoring with
answer-posterior soft support, reaching 84.2% mean@16 and 86.7% maj@16 on
MATH500; preserved scripts/logs for full-rollout PowerFlow, chunk/suffix
search-state ablations, and Qwen3-30B-A3B MoE scaling audits.
```

## 代码与证据索引

- 训练脚本：`verl/run_records/`
- 中文实验记录：`important_experiment_logs/`
- Qwen3-30B-A3B 评测 summary：`important_experiment_logs/qwen3_30b_a3b_*_summary.json`
- B200 环境与稳定运行配置：`important_experiment_logs/B200_ENV_USAGE.md`
- Qwen2.5-Math MV 语义审计：`important_experiment_logs/mv_b32_r32_v64_infra_semantics_20260731.md`
- sharpened MV-anchor 最优结果：`important_experiment_logs/sharpened_mvanchor_soft002_seedfix_20260731.md`
- chunk-level 审计：`important_experiment_logs/chunk_level_search_state_ttrl_24h_audit_20260801.md`
- chunk/suffix 长实验记录：`important_experiment_logs/chunk_level_search_state_ttrl_20260731.md`
- full-rollout TTRL 审计：`important_experiment_logs/full_rollout_ttrl_12h_progress_20260802.md`

---

下面保留原始 TTRL 项目 README，作为论文背景和官方使用说明。

<div align="center">

# TTRL: Test-Time Reinforcement Learning

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.16084)  [![Github](https://img.shields.io/badge/TTRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/PRIME-RL/TTRL)
[![Wandb Log of AIME](https://img.shields.io/badge/Wandb%20Log%20of%20AIME-%2300B4AB?style=for-the-badge&logo=weightsandbiases&logoColor=white&labelColor=000000)](https://wandb.ai/truman-yx-zuo-nlp/TTRL/workspace?nw=nwusertrumanyxzuo) [![HF Papers](https://img.shields.io/badge/HF--Paper-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/papers/2504.16084)  [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/zuo_yuxin/status/1915406839669572036)

</div>

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a> •
    <a href="#main-results" style="text-decoration: none; font-weight: bold;">📊 Main Results</a>
  </p>
  <p>
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#contact" style="text-decoration: none; font-weight: bold;">📨 Contact</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a> •
    <a href="#star-history" style="text-decoration: none; font-weight: bold;">🌟 Star History</a>
  </p>
</div>

> Welcome to the Era of Experience.  --David Silver, Richard S. Sutton

# 🎉News
- **[2026-03-10]** We investigate the mechanisms and potential applications of [Unsupervised RLVR (URLVR)](https://arxiv.org/pdf/2603.08660), and find that it is particularly well suited for test-time training and quantifying model priors. Here is [code](https://github.com/PRIME-RL/TTRL/tree/urlvr-dev). URLVR paper is accepted to [ICLR 2026](https://iclr.cc/Conferences/2026)!
- **[2025-09-18]** TTRL paper is accepted to [NeurIPS 2025](https://neurips.cc/Conferences/2025)!
- **[2025-08-17]** We bump into [verl v0.4.1](https://github.com/volcengine/verl/releases/tag/v0.4.1), and now you can enable TTRL by simply setting `+ttrl.enable=True`!
- **[2025-05-23]** We update both the paper and the code, with the implementation based on the [verl](https://github.com/volcengine/verl).
- **[2025-04-24]** We release the code and experimental logs. Check it out: [Getting Started](#getting-started).
- **[2025-04-23]** We present **TTRL** (Test-Time Reinforcement Learning), an open-source solution for online RL on data without ground-truth labels, especially test data.

# 📖Introduction

**We investigate Reinforcement Learning (RL) on data without explicit labels for reasoning tasks in Large Language Models (LLMs).**
The core challenge of the problem is reward estimation during inference while not having access to ground-truth information. While this setting appears elusive, we find that common practices in Test-Time Scaling (TTS), such as majority voting, yield surprisingly effective rewards suitable for driving RL training.

<p align="center">
   <img src="figs/teaser.png" alt="Performance and settings of TTRL." style="width: 80%;">
</p>


<p align="center">
   <img src="figs/overview.png" alt="Overview of TTRL." style="width: 80%;">
</p>


# 📊Main Results

Our experiments demonstrate that TTRL consistently improves performance across a variety of tasks and models. Notably, TTRL boosts the `pass@1` performance of Qwen-2.5-Math-7B by approximately 211% on `AIME 2024` with only unlabeled test data.

Furthermore, although TTRL is only supervised by the `maj@n` metric, TTRL has demonstrated performance to consistently surpass this upper limit of the initial model, and approach the performance of models trained directly on test data with ground-truth labels.

<p align="center">
   <img src="figs/results.png" alt="Main results of TTRL." style="width: 60%;">
</p>


# ✨Getting Started

## Env Setup

```bash
git clone https://github.com/PRIME-RL/TTRL.git

cd TTRL/verl

conda create -n ttrl python==3.10
conda activate ttrl
bash scripts/install_ttrl_deps.sh
pip install -e .
```

## Reproduce TTRL
You can reproduce the results on `AIME 2024` with the following commands:

```bash
bash examples/ttrl/Qwen2.5/aime.sh
```

> [!NOTE]
> - You can use the script [verl/data/preprocess.py](https://github.com/PRIME-RL/TTRL/blob/main/verl/data/preprocess.py) to convert data from the `JSON` format to the `Parquet` format for training with verl.
> - We provide scripts in the [verl/examples/ttrl](https://github.com/PRIME-RL/TTRL/tree/main/verl/examples/ttrl) directory for running TTRL on multiple models across various benchmarks.
> - For further details regarding the code, please refer to the [verl documentation](https://verl.readthedocs.io/en/latest/index.html).

We additionally conducted three independent runs using the preview version of our code. Two of the runs achieved a pass@1 (greedy) of 43.3, while one run reached 46.7. Please refer to the [Weights & Biases logs](https://wandb.ai/truman-yx-zuo-nlp/TTRL/workspace).

*All experiments were conducted on 8 x NVIDIA A100 80GB GPUs.*

<details>
<summary>
  Pseudo-Code
</summary>

The implementation of TTRL can be achieved rapidly by simply modifying the reward function. Please refer to the following code snippet for details:

<p align="center">
   <img src="figs/ttrl_reward.png" alt="The pseudo-code of the majority voting reward function." style="width: 60%;">
</p>
</details>

# 📨Contact

- Kaiyan Zhang: zhang-ky22@mails.tsinghua.edu.cn
- Ning Ding: dingning@mail.tsinghua.edu.cn

# 🎈Citation
If you find TTRL helpful, please cite us.

```bibtex
@article{zuo2025ttrl,
  title={Ttrl: Test-time reinforcement learning},
  author={Zuo, Yuxin and Zhang, Kaiyan and Sheng, Li and Qu, Shang and Cui, Ganqu and Zhu, Xuekai and Li, Haozhan and Zhang, Yuchen and Long, Xinwei and Hua, Ermo and others},
  journal={arXiv preprint arXiv:2504.16084},
  year={2025}
}
```

# 🌟Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PRIME-RL/TTRL&type=Date)](https://www.star-history.com/#PRIME-RL/TTRL&Date)
