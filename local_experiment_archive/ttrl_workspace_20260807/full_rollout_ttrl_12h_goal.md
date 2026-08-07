# Full-Rollout TTRL 12 小时广度优先实验方案

## 背景结论

彻底放弃 chunk-level search-state 路线。最近的 suffix-to-EOS v29 20-step pilot 虽然 target health proxy 明显好于 v28，但最终 validation 很差：

- mean@16 = 0.408875
- maj@16 = 0.526426
- best@16 = 0.81171

这说明 chunk / mid-state / suffix resampling 这条线当前没有有效转化为完整答案能力提升。后续实验回到完整 rollout 粒度：每条训练样本都是 prompt 到完整 response，不再截取中间 state，不再做 suffix resampling，不再做 short-horizon probe。

## 12 小时目标

围绕完整 rollout group 构造 search-improved target distribution，用 PowerFlow-style distribution matching 更新 actor。12 小时内完成广度优先探索，快速筛选能超过当前 MV / PowerFlow 历史曲线的 full-rollout 训练语义。

目标产出：

- 至少完成 5-7 个 20-step pilot。
- 每个 pilot 做最终 validation，记录 mean@16 / maj@16 / best@16。
- 选 1-2 个最有希望方案扩展到 40-step。
- 如果 40-step 明显强，再继续 80-step。
- 所有实验记录到中文日志，保留 launcher、raw log、metrics、关键 commit。

## 固定设置

- model: `/models/Qwen2.5-Math-7B`
- data: `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT`
- workspace: `/mlx_devbox/users/quyanyi/playground/TTRL/verl`
- env: `/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200`
- GPUs: 8x B200
- train batch size: 32
- rollout samples: `n=32`，必要时对候选方案提升到 `n=64`
- validation: 每个 pilot 只在最后做一次，优先看 mean@16 / maj@16 / best@16
- actor dynamic batch: 关闭，保持 `actor_rollout_ref.actor.use_dynamic_bsz=False`
- training target: 不用 GT 作为正式训练监督；GT 只用于 validation 和 oracle upper-bound 诊断

## 核心方法

完整回答粒度的训练流程：

```text
prompt
  -> sample complete rollout group
  -> extract final answers
  -> estimate answer-level posterior
  -> sharpen / reweight into search-improved rollout target distribution
  -> train actor with PowerFlow-style distribution matching
```

关键区别：

- 不再 hard majority vote。
- 不再把单个 majority answer 当唯一 teacher。
- 不做 chunk / suffix / middle state。
- 同一个 prompt 的完整 rollout group 是 label estimation 的基本单位。
- actor 学的是完整回答分布如何向 search-improved distribution 移动。

## 需要新增的通用指标

所有 full-rollout target mode 都应记录：

- `full_rollout_target/support_coverage`
- `full_rollout_target/oov_ratio`
- `full_rollout_target/top1_mass`
- `full_rollout_target/top2_mass`
- `full_rollout_target/margin`
- `full_rollout_target/target_entropy`
- `full_rollout_target/valid_answer_coverage`
- `full_rollout_target/repeated_boxed_ratio`
- `full_rollout_target/prompt_echo_ratio`
- `full_rollout_target/marker_ratio`
- `full_rollout_target/weight_mean`
- `full_rollout_target/weight_max`
- `full_rollout_target/nonzero_ratio`
- `full_rollout_target/num_actor_samples`

同时保留性能指标：

- `timing_s/gen`
- `timing_s/score`
- `timing_s/ref`
- `timing_s/update_actor`
- `timing_s/testing`

## 实验矩阵

### FR-B0: Answer Posterior Sharpening

最基础的新主线。

对每个 prompt 的完整 rollout group 抽取 final answer，按 answer 聚合 mass：

```text
p(a) = count(answer = a) / valid_answer_count
```

每条 rollout 的 target weight：

```text
q_i ∝ exp(alpha * p(answer_i))
```

invalid / empty / repeated boxed / prompt echo / marker pollution 降权。

优先配置：

- `alpha = 4`
- 20-step + final validation

扩展扫描：

- `alpha = 2`
- `alpha = 8`

预期：比 hard MV 更软，保留 best@16，同时提升 mean@16 / maj@16。

### FR-B1: Posterior Gain Target

不直接强化高 support answer，而是强化相对 prompt baseline 有增益的 answer。

```text
score_i = p(answer_i) - baseline
q_i ∝ exp(alpha * relu(score_i + slack))
```

baseline 候选：

- valid answer uniform mass
- top2 average mass
- entropy-adjusted baseline

优先配置：

- `baseline = valid_uniform`
- `alpha = 6`
- `slack = 0.02`
- 20-step + final validation

目的：避免低置信 prompt 被偶然 majority 过度牵引。

### FR-B2: Margin-Aware Posterior Sharpening

利用 answer posterior 的 top margin 做 prompt-level confidence weighting。

```text
margin = p(top1) - p(top2)
prompt_weight = clip(margin / tau, floor, 1)
q_i ∝ exp(alpha * p(answer_i))
```

优先配置：

- `alpha = 6`
- `tau = 0.25`
- `floor = 0.2`
- 20-step + final validation

目的：高置信 prompt 强更新，低置信 prompt 轻更新，但不直接丢掉所有低置信样本。

### FR-B3: Top-k Representative Distribution Matching

解决 duplicate answer domination 和污染轨迹参与训练的问题。

流程：

1. 按 answer 聚合完整 rollout。
2. 保留 top-k answer clusters。
3. 每个 answer cluster 内只保留少量 representative rollout。
4. cluster mass 按 `p(a)^alpha` 分配。
5. 非 representative duplicates 降权或置零。

优先配置：

- `top_k_answers = 2`
- `representatives_per_answer = 2`
- `alpha = 4`
- 20-step + final validation

目的：保留多答案分布结构，同时避免重复答案把 target 压扁成 imitation。

### FR-B4: Self-Consistency Advantage PowerFlow

把完整 rollout 当 action，做 relative search-improvement weighting。

```text
adv_i = p(answer_i) - mean_j p(answer_j)
weight_i = exp(beta * adv_i)
```

invalid answer 给极低权重或置零。

优先配置：

- `beta = 4`
- `weight_clip = 4`
- per-prompt renormalization
- 20-step + final validation

目的：比 B0 更接近 relative improvement，但仍然保持完整 rollout 粒度，不回到 GRPO。

### FR-C0: Quality-Gated Full Rollout

这是过滤/加权实验，不是主创新。

只训练或强加权满足以下条件的 prompt / rollout：

- valid answer coverage >= 0.6
- top answer mass >= 0.25
- repeated boxed ratio <= 0.2
- prompt echo / marker pollution 低

target 使用 B2 或 B0。

目的：判断主要瓶颈是否是 label estimation noise。如果 C0 显著涨，后续要重点设计 label quality estimator。

### FR-C1: Anti-Collapse Soft Target

防止 posterior sharpening 过早坍缩。

target 混合：

```text
q = (1 - eps) * sharpened_answer_target + eps * policy_prior
```

或：

```text
q = (1 - eps) * sharpened_answer_target + eps * uniform_valid_rollout
```

优先扫描：

- `eps = 0.05`
- `eps = 0.10`

适用场景：如果 B0 / B2 出现 maj 上升但 best@16 或 mean@16 下降，用 C1 救 collapse。

### FR-D0: Oracle Upper Bound Diagnostic

诊断实验，不作为正式方法。

用 GT correctness 构造完整 rollout target：

```text
q_i ∝ exp(alpha * correctness_i)
```

只跑 5-step。

目的：

- 如果 oracle 都不涨，说明 full-rollout PowerFlow loss / actor update / optimization 有问题。
- 如果 oracle 明显涨，而 B 组不涨，说明主要问题在 label estimation。

## 执行顺序

不再复现 A 组，直接跑新方法：

1. FR-B0 alpha=4，20-step。
2. FR-B2 margin-aware，20-step。
3. FR-B1 posterior-gain，20-step。
4. FR-B3 top-k representative，20-step。
5. FR-C0 quality-gated B2，20-step。
6. FR-C1 anti-collapse，按前面结果决定是否跑。
7. FR-D0 oracle 5-step，穿插在第 6-8 小时做诊断。
8. 选 top 1-2 个方案跑 40-step。
9. 如果 40-step 明显超过 MV / PowerFlow 历史曲线，再继续 80-step。

## 12 小时时间安排

### 第 0-1 小时：实现通用 full-rollout target 框架

- 关闭 chunk-state 分支。
- 复用现有完整 rollout、answer extraction、reward/format/pollution filter。
- 新增 full-rollout target mode。
- 新增 full_rollout_target 指标。
- 创建统一 launcher 模板和实验日志模板。
- 做 1 个 1-3 step smoke，只验证不崩和指标正常，不看 acc。

### 第 1-6 小时：广度优先 20-step pilots

优先依次跑：

1. FR-B0
2. FR-B2
3. FR-B1
4. FR-B3
5. FR-C0

每个 pilot 最多 20-step。指标差就立刻换，不做长跑。

### 第 6-8 小时：诊断与修正

- 跑 FR-D0 oracle 5-step。
- 如果 B 组全差但 oracle 好，说明 label estimation 不够强，优先改 target。
- 如果 oracle 也差，优先查 PowerFlow full-rollout loss 实现和 actor update。
- 如果出现 collapse，跑 FR-C1。

### 第 8-12 小时：放大赢家

- 选当前最好的 1-2 个方案跑 40-step。
- 如果 40-step mean@16 明显超过 0.80，继续 80-step。
- 如果没有赢家，整理 negative results，下一轮转向 answer posterior estimator / verifier quality 设计。

## 晋级标准

20-step 进入 40-step 的条件，满足其一即可：

- `mean@16 >= 0.76`
- `maj@16 >= 0.83` 且 `best@16 >= 0.91`
- 相对当前同链路 pilot 有明显上升趋势，且 target health 好

40-step 进入 80-step 的条件：

- `mean@16 >= 0.81`
- 或 `maj@16 >= 0.86` 且 best@16 不掉

最终希望目标：

- 80-step `mean@16 > 0.85`
- `maj@16 >= 0.88`
- `best@16` 不低于 MV baseline

## 淘汰标准

任何 20-step 出现以下情况，直接停掉该分支：

- `mean@16 < 0.65`
- `maj@16 < 0.75`
- `best@16 < 0.88`
- target entropy 快速接近 0 且 best@16 下降
- OOV > 0.35
- response pollution 明显上升
- actor update 正常但 validation 暴跌，说明 target 有毒

## 当前最优先押注方向

### 1. FR-B2 Margin-Aware Posterior Sharpening

这是最值得优先跑的方向。它不是 hard majority，而是 confidence-weighted posterior improvement。理论叙事比较稳：

```text
posterior concentration gives confidence-weighted policy improvement
```

### 2. FR-B3 Top-k Representative Distribution Matching

这个方向解决 duplicate answer domination 和污染 rollout 两个实际问题。它比直接 majority 更像 search distribution distillation，也更适合写成方法。

## 预期方法叙事

一句话：

```text
We replace majority-vote pseudo-labeling with full-rollout posterior sharpening: sampled complete solutions define an answer-level posterior, and the actor is trained by PowerFlow-style distribution matching toward a confidence-weighted, search-improved rollout distribution.
```

中文叙事：

```text
我们不再把 majority vote 当成硬伪标签，而是把同一 prompt 下完整采样解的 answer posterior 看作 test-time search 得到的软分布估计，再通过置信度加权和分布锐化构造 search-improved target，用 PowerFlow-style loss 将 actor 推向这个改进后的完整回答分布。
```

## 实验记录要求

每个实验必须记录：

- run id
- git commit
- launcher path
- raw log path
- diag / metrics path
- 关键配置
- 20-step 或 40-step wall time
- step time 分解
- target health
- validation mean@16 / maj@16 / best@16 / worst@16
- 是否晋级或淘汰
- 简短结论

每完成一组有意义结果，做一次 scoped commit。不要 stage unrelated dirty files。
