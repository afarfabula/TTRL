# Chunk-Level Search-State TTRL：24 小时实验目标

## 背景

当前 TTRL / majority-vote 路线的训练单位是完整 response：模型先生成整条推理轨迹，再根据终局答案、majority vote 或规则 reward 更新整条 rollout。这种做法的主要限制是 credit assignment 太粗，模型无法直接学习“在中间推理状态下，下一段应该怎么走”。

我们希望把 TTRL 从 full-trajectory terminal-reward training，改造成 chunk-level search-state improvement：把任意时刻的 `query + generated prefix` 当成一个 state，对该 state 重采样多个 next-chunk continuation，用后续搜索 / probe / verifier 估计哪些 chunk 更有前途，再把 search-improved chunk distribution 蒸馏回 actor。

一句话目标：

> 将 TTRL 的训练对象从完整答案级伪标签，改为搜索过程中的局部状态转移；让模型学会在每个中间推理状态上生成更好的下一段。

## 核心 Idea

第一版不直接做复杂的逐 chunk 在线 RL，而采用更稳的半在线流程：

1. 对每个 prompt 先采样 32 条完整 rollout，得到当前 policy 的真实 on-policy 推理分布。
2. 从完整 rollout 中随机截取 chunk boundary，构造 state：`state = query + response[:t]`。
3. 对同一个 state 重采样多个 next chunk：`chunk_1 ... chunk_K`。
4. 对每个 `state + chunk_i` 做 future probe，但 probe 只用来估计后续 answer distribution。
5. 由 full-rollout group 先定义 prompt-level answer support / value，再判断哪个 chunk 把 future distribution 推向这个 support。
6. 用 PowerFlow-style distribution matching 或 preference loss 更新 actor 的 `pi(next_chunk | state)`。

第一阶段主推 chunk-level PowerFlow-style distribution matching，而不是 chunk-GRPO。原因是这里的核心监督信号不是 scalar local reward，而是 full-rollout support/value 诱导出的 search-improved continuation distribution；weighted NLL / preference loss 只作为 fallback 或 ablation。

## 第一版算法定义

给定 prompt `x`，当前 policy 先生成完整 rollout：

```text
y_1, ..., y_32 ~ pi_theta(. | x)
```

从某条 rollout 中随机选择 chunk boundary `t`：

```text
s = concat(x, y_i[:t])
```

对 state `s` 采样 K 个 next chunk：

```text
c_1, ..., c_K ~ pi_theta(. | s), len(c_j) <= chunk_size
```

对每个候选 chunk 做 future probe，得到候选的后续 answer distribution：

```text
p_j(a) = ProbeAnswerDistribution(s + c_j)
```

full rollout group 先定义 support/value：

```text
q_full(a | x) = empirical answer support from 32 full rollouts
V_full(a) = support mass / value under q_full
```

chunk target 由 future distribution 与 full support 的 transport / overlap / value gain 决定：

```text
score_j = SupportImprovement(p_j, q_full, V_full)
q_j = normalize(prior_j * exp(alpha * score_j))
```

source chunk 可以作为 prior / drift guard，但不能作为主要 teacher；不能用 short-horizon local answer hit 或 source consistency 直接决定 target。

更强约束：

- 放弃“局部短视可判定性”作为主监督假设。
- chunk 的好坏不要求在 short horizon、局部 answer hit、source answer consistency 或几条短 probe 的偶然命中上被判清楚。
- full rollout group 必须先定义 prompt-level support/value；probe 只是估计 candidate 后的 future answer distribution。
- short-horizon probe 不再承担“谁是 teacher”的职责；它只能为 `p_j(a)` 提供采样证据，最终 `q_j` 必须由 full-rollout support/value、transport overlap、support mass gain、OOV/coverage 质量共同定义。
- prompt / state 的低信息过滤优先看 full-rollout valid answer coverage、top mass、top margin、entropy、candidate OOV 和 support overlap。
- source chunk 只能作为 prior / drift guard，不能作为 hard floor 或主要 teacher；source 侧 hard gate 已作为负方向处理，后续优先用连续质量权重。

优先用 PowerFlow-style weighted chunk NLL / distribution matching 更新：

```text
L = - sum_j stopgrad(q_j) * log pi_theta(c_j | s)
```

可选 fallback / ablation 是 pairwise/listwise preference：

```text
L_pref = -log sigmoid(beta * (log pi_theta(c_good | s) - log pi_theta(c_bad | s)))
```

## 与现有工作的区别

- 区别于 TTRL / majority vote：不只在完整答案结束后给终局伪标签，而是在中间 search state 上训练下一段 continuation。
- 区别于简单 reward shaping：不是给完整 response 改 reward，而是改变训练单位和 policy-improvement 对象。
- 接近 PowerFlow / inference-time scaling：训练目标来自 search-improved distribution，而不是手工构造单一 reward。
- 接近 chunked search：训练时学习的局部转移和测试时 chunked expand / resample / prune 的运行方式一致。

## 24 小时实验计划

### 0-2 小时：设计冻结与代码入口确认

目标：

- 固定第一版 MVP，不扩散到完整在线 chunk RL。
- 明确复用当前 8 卡 B200 环境、Qwen2.5-Math-7B、MATH-TTT、VAL_N=16 评测链路。
- 明确禁止使用会改变训练语义的 actor dynamic batch。
- 确认当前 TTRL 代码中可复用的数据结构：prompt batch、rollout response、attention mask、position ids、old logprob、ref logprob、reward tensor。

产物：

- 一个实验脚本：`run_records/ttrl_chunk_state_powerflow_20step_*.sh`
- 一个配置开关组：`ttrl.chunk_state_enable=True`
- 日志文件记录所有关键配置、step time、token throughput、validation metrics。

### 2-6 小时：实现最小 chunk-state 数据构造

实现内容：

- 在完整 rollout 之后，从每个 prompt 的 32 条 response 中采样 state。
- state boundary 初版只取固定网格：`t in {0, 256, 512, 768, 1024}`，且不超过实际 response 长度。
- 每个 prompt 暂取 1 个 state，避免第一版样本数爆炸。
- chunk size 初版设为 256。
- 记录 state 长度、chunk 长度、有效样本数、跳过比例。

关键约束：

- 不打乱 prompt 内 group 关系。
- 不改变原始 full rollout 的生成语义。
- chunk training batch 必须显式记录来源 prompt、来源 rollout id、boundary t。

验收：

- 能在一个小 batch 上打印出 state/chunk 样本。
- 能确认 state 是从真实 on-policy full rollout 中截取，而不是人工 prefix。

### 6-10 小时：实现 chunk resampling 与 support-distribution scoring

实现内容：

- 对每个 state 采样 `K=8` 个 next chunk。
- 对每个 `state + chunk` 做后续 completion，尽量复用 vLLM rollout。
- 从 probe completion 中抽取 answer distribution。
- 用 full-rollout group 的 answer support / value / transport overlap 聚合得到 `score_j`。

第一版 scoring 不再是 local correctness，而是 support distribution matching：

```text
support_expected_value = E_{a ~ p_j}[q_full(a)]
support_overlap = overlap(p_j, q_full)
score_j = 0.5 * (support_expected_value + support_overlap) * transport_affinity(p_j, q_full)
```

state / prompt 过滤：

- full-rollout valid answer coverage 太低，跳过。
- full-rollout top mass 太平或 top1/top2 margin 太小，跳过或降权。
- candidate future distribution 大量 OOV，跳过或降权。
- malformed / repeated boxed / marker contamination，跳过或降权。
- 不再用 source answer mass hard gate 作为主路径；如果使用 source chunk，只用于 prior。
- 不再把 source consistency / short-probe local hit 当作 target 的主排序依据；任何局部 probe 命中只能进入 future distribution 的统计，而不能绕过 full support label estimation。

验收：

- 日志中输出 support coverage、OOV、support expected value、support overlap、candidate entropy。
- 确认 probe 只是 future distribution estimator，target 由 full-rollout support/value 主导。

### 10-14 小时：实现 chunk-level PowerFlow loss

实现内容：

- 对同一个 state 的 K 个 chunk，根据 full-support distribution score 构造 chunk-level target。
- 只在 next chunk token 上计算 PowerFlow loss，不训练 state prefix token。
- 主路径使用 PowerFlow-style weighted chunk NLL / distribution matching：

```text
q_j = normalize(prior_j * exp(alpha * score_j))
L = - sum_j stopgrad(q_j) * log pi_theta(chunk_j | state)
```

初始超参：

```text
chunk_size = 256
K = 8
probe_n = 1
alpha = 2.0
eps = 0.05
max_state_response_prefix = 1024
states_per_prompt = 1
powerflow_beta_coef = 4.0
powerflow_use_boxed_reward = true
```

验收：

- 单 step 可跑通 forward/backward/update。
- loss 不为 NaN。
- actor update 的 token 数、step time、显存峰值被记录。

### 14-18 小时：20-step pilot

运行配置：

- 8 卡 B200。
- Qwen2.5-Math-7B。
- MATH-TTT。
- prompt batch 与现有 MV baseline 尽量对齐。
- validation every 20 steps。
- 不保存频繁 ckpt，避免系统盘问题。

必须记录：

- step time。
- rollout time。
- chunk resampling time。
- probe time。
- actor update time。
- total tokens / throughput。
- `val-core/math/acc/mean@16`
- `val-core/math/acc/maj@16/mean`
- `val-core/math/acc/best@16/mean`
- informative state ratio。
- positive chunk ratio。
- target entropy。

20-step gate：

- 如果 step20 `mean@16` 明显低于 MV baseline 2 个点以上，先停，检查 scoring 和 loss。
- 如果 step20 接近 MV baseline，继续到 80 step。
- 如果 step20 超过 MV baseline 或 target statistics 明显健康，优先扩到 80 step。

当前参考 baseline：

```text
MV step20: mean@16 ~= 0.760, maj@16 ~= 0.820, best@16 ~= 0.901
soft0.02 step20: mean@16 ~= 0.757, maj@16 ~= 0.821, best@16 ~= 0.911
```

### 18-24 小时：80-step 扩展与快速 ablation

如果 20-step pilot 通过 gate，扩展到 80 step，validation every 20 step。

目标比较：

```text
MV step80: mean@16 ~= 0.828, maj@16 ~= 0.854, best@16 ~= 0.897
soft0.02 step80: mean@16 ~= 0.836, maj@16 ~= 0.864, best@16 ~= 0.892
strong external target step80: mean@16 ~= 0.843, maj@16 ~= 0.880, best@16 ~= 0.928
```

优先 ablation 顺序：

1. `K=8` vs `K=16`：看 target 区分度和 probe 成本。
2. `chunk_size=128` vs `256`：看 actor update 是否显著变快，以及指标是否变差。
3. `max_state_response_prefix=512` vs `1024`：看长 state 是否拖慢 update。
4. weighted NLL vs pairwise preference：如果 weighted NLL 学不动，再切 preference。

推进条件：

- 如果 80-step mean@16 >= soft0.02 step80，并且方法统计健康，继续做 150-step。
- 如果 mean@16 没超过 soft0.02，但 target statistics 明显说明 chunk supervision 有效，可以做 loss/scoring ablation。
- 如果 informative state ratio 很低，优先改 probe / state sampling，不继续盲跑。

## Infra 预期与约束

预期收益：

- actor update 只监督 next chunk，单次 backward token 数少于完整 response 训练。
- chunk-level loss 更接近局部 continuation，理论上 actor update 应该快于 full-response GRPO update。

主要风险：

- 如果 state prefix 太长，forward 仍需处理完整 state，上下文成本不会低。
- 如果每个 state 的 K 和 probe 太大，rollout/probe 会成为新的瓶颈。
- 如果每个 chunk 都单独触发 update，Ray/vLLM/FSDP 调度开销会很高。

第一版 infra 原则：

- 采集一批 chunk states 后批量 update，不做每个 chunk 立即 update。
- 禁用 actor dynamic batch，除非证明 loss、grad、one-step parameter delta 与 fixed batch 等价。
- old-logprob reuse 只有在 diff 监控为 0 时才允许。
- 依赖环境保留在 `/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200`，不得放到 `/tmp`。

## 补充方向：借鉴 Scalable Power Sampling

参考论文：`Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening` (`arXiv:2601.21590`)

这篇工作的价值，不在于直接替换当前 chunk-level PowerFlow loss，而在于为
`next-chunk proposal` 和 `target distribution` 提供一个更便宜、更稳的先验。

当前版本的主要问题是：

- `q_j` 几乎完全由 `state + chunk + probe` 的 verifier 后验构造，噪声较大。
- 单次或少量 probe 的 `score_j` 很容易追逐局部偶然成功，而不是稳定对应最终 correctness。
- probe / scoring 开销较大，容易吞掉 chunk-level update 的效率收益。

Scalable Power Sampling 的启发是：

- sequence-level 的 improved distribution，不一定非要靠昂贵的全局 iterative search 才能近似。
- 可以用局部的 distribution sharpening / low-temperature scaling，给 continuation proposal 一个更强的先验偏置。

对本项目，最自然的借法不是改 actor loss，而是改 `chunk proposal + q_j construction`。

### 建议的最小借法

保持当前主路径不变：

- `actor.powerflow_enable=True`
- `PowerFlow-style distribution matching` 仍是 actor update 主 loss
- `probe score` 仍保留，作为最终 correctness proxy

新增一条 SPS-style proposal / prior 支路：

1. 对每个 state，不只用当前默认采样温度重采 chunk，也额外尝试一组低温 / sharpened proposal。
2. 对每个 candidate chunk，构造一个 `sps_prior_j`，表示该 chunk 在局部分布修正下的优先级。
3. 用混合分数而不是纯 probe 后验构造 target：

```text
mixed_score_j = lambda_probe * probe_score_j + lambda_sps * sps_prior_j
q_j = normalize((mixed_score_j + eps) ** alpha)
powerflow_weight_j = q_j * K
```

这样 actor 仍然学习 search-improved chunk distribution，但 target 不再完全依赖高方差 verifier。

### 为什么这条借法更合适

- 它保留了当前方案最重要的语义：训练对象仍然是 `pi(next_chunk | state)`。
- 它不需要把当前 PowerFlow 实现整体推倒重来。
- 它能自然兼容已有的 multi-probe / source-chunk-support / skip-all-negative 机制。
- 它把论文中的核心价值，落在当前最痛的地方：proposal 质量和 `q_j` 的稳定性。

### 具体实验建议

如果要在当前 24 小时目标内插入一个最小 SPS 借鉴版，优先顺序应是：

1. **SPS-style proposal only**
   - 保持原有 PowerFlow loss 和 probe score 不变
   - 只比较默认 proposal vs 低温 proposal 的 `positive_ratio`、`informative_state_ratio`

2. **SPS prior + probe mixed target**
   - 在 `q_j` 中引入 `sps_prior_j`
   - 看 step-level target entropy、all-negative ratio 是否改善

3. **multi-probe + SPS prior**
   - 用 multi-probe mean 降低 verifier 噪声
   - 用 SPS prior 提供 model-based 先验
   - 两者共同构造更稳的 `q_j`

### 边界

- 不应把这篇工作简单理解为“再换一个新 loss”。
- 第一阶段不建议直接把 SPS 写成 actor objective；应先把它当成 proposal / target prior。
- 如果 mixed target 仍然没有改善 validation，再考虑更激进的 preference / critic / full-answer regularization 路线。

一句话补充目标：

> 在保持 chunk-level PowerFlow 主路径不变的前提下，引入 Scalable Power Sampling 风格的局部 proposal / prior，降低 probe-only target 的噪声与成本，让 `q_j` 更像真正的 improved continuation distribution。
- ckpt 和大日志可以放 `/tmp/ttrl_b200`，但重要配置和实验总结必须落盘到 playground。

## 成功标准

24 小时内的最低成功标准：

- 跑通 chunk-state data construction、chunk resampling、probe scoring、weighted chunk distribution matching。
- 完成至少一个 20-step pilot，并记录完整指标。
- 能判断该方向的主要瓶颈是 scoring、probe 成本、actor update 还是 validation。

理想成功标准：

- 20-step 不低于当前 MV baseline。
- 80-step 达到或超过 soft0.02 step80。
- actor update 单项耗时明显低于 full-response GRPO。
- 产出一条清晰故事线：chunked search 产生 improved local continuation distribution，actor 学习 state-conditional continuation improver。

强成功标准：

- 80-step mean@16 超过 0.843。
- 后续 150-step 有机会冲击 mean@16 >= 0.85。
- 方法指标显示不是偶然波动：informative state ratio、positive chunk ratio、target entropy 与 validation 提升一致。

## 当前判断

这个方向比继续调 `mv_anchor soft_coef` 更值得投入。它不是在已有 TTRL 上做小幅 reward shaping，而是把训练单位从完整 response 改成 chunk-level search transition。

最新实验判断：

- `supportdist probe1536/3072` 说明单纯拉长 probe 不解决 target 质量，probe 成本还会显著增加。
- `sourcegate/supportq` 说明继续强化 source 侧 hard constraint 会降低 support coverage、提高 OOV。
- `supportq2 prompt-quality` 是当前正向 smoke：用 prompt-level full-rollout support quality 过滤后，support coverage 从约 0.39/0.40 提到约 0.61，OOV 从约 0.60 降到约 0.39，且 probe 成本没有增加。
- 代价是 state 供给变稀，每 step 8 个 state 大约只保留 2-3 个 learnable state。
- `supportq2 20-step pilot` 已失败：mean@16=0.466、maj@16=0.594、best@16=0.852，20 个 step 中 9 个 step 没有有效 actor update。prompt support 很强，但 chunk candidate support 仍不稳，说明 target/proposal 是主矛盾。

下一阶段应围绕两件事推进：

1. 保持 full support/value 主导 target，但放弃“局部短视可判定性”：short-horizon probe 只能作为 future distribution estimator，不能作为 teacher 的主要判据。
2. 把 hard keep 改成 soft distribution matching / soft weighting，避免大量 step 因局部阈值被清零。
3. 提高 candidate/proposal 的 support coverage，例如混入 high-support rollout suffix/replay proposal 或 support-conditioned next-chunk proposal，而不是回退到 short-horizon local hit teacher。

只有当这个最小闭环成立后，再考虑 chunk-GRPO、pairwise preference、critic/value model 或更复杂的 online chunk search。
