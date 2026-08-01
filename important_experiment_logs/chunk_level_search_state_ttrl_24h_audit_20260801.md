# Chunk-Level Search-State TTRL 24h Audit - 2026-08-01

## 目标复述

本轮目标来自 `/mlx_devbox/users/quyanyi/playground/chunk_level_search_state_ttrl_24h_goal.md`：

- 按 chunk-level search-state TTRL 方案启动并推进 24 小时实验。
- 使用 8 卡 B200、Qwen2.5-Math-7B、MATH-TTT/MATH500 链路。
- 记录中文实验日志、关键配置、step time、validation metrics 和失败原因。
- 保持有效 commit 保存，避免只留下临时改动。
- 合理设计实验节奏，优先用 PowerFlow-style loss，不把 actor dynamic batch 作为默认优化。
- 尽可能利用 B200 大显存和算力，同时不改变训练语义。

## Prompt-to-Artifact Checklist

| 要求 | 证据 | 状态 |
| --- | --- | --- |
| 使用 goal 文档指导实验 | `/mlx_devbox/users/quyanyi/playground/chunk_level_search_state_ttrl_24h_goal.md` 已定义 0-24h 计划、20-step gate、80-step 扩展条件、infra 约束 | 已满足 |
| 8 卡 B200 链路 | run scripts 继承 `math500_7b_8gpu.sh`，训练日志显示 `trainer.n_gpus_per_node=8`，worker 通过 `mlx worker login` 进入 B200 环境 | 已满足 |
| 固定模型和数据 | 多个 run 记录使用 `/models/Qwen2.5-Math-7B` 和 `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT` | 已满足 |
| 禁用 actor dynamic batch | 关键脚本均显式 `actor_rollout_ref.actor.use_dynamic_bsz=False` | 已满足 |
| 先实现最小 chunk-state 构造 | `ray_trainer.py` 中 `_make_chunk_state_prompts`、boundary/source metadata、diag jsonl 已实现并多轮运行 | 已满足 |
| chunk resampling / scoring | 已实现 majority completion、answer distribution、answer support mass、future support gain、TV gain、support anchor 等 scorer | 已满足 |
| PowerFlow-style chunk actor update | `chunk_state` actor batch 写入 `powerflow_flat_weights`，主线使用 PowerFlow weighted actor update；GRPO 不是主路径 | 已满足 |
| 3-step smoke gate | 多个 3-step smoke 已落盘，包括 future support、hardfilter、clip4、candidate/source/state gate、TV、support_anchor | 已满足 |
| 20-step pilot + final val | 已跑多条 20-step；最新 support_anchor diag-off 20-step final val 为 mean@16 0.43725 / maj@16 0.558596 / best@16 0.83514 | 已满足但结果为负 |
| 20-step gate 决策 | support_anchor 20-step 低于 MV baseline 远超 2 点；future-support 系列 target quality 未过 gate，因此不应扩 80-step | 已满足 |
| 中文日志记录 | 主文档 `/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/chunk_level_search_state_ttrl_20260731.md` 已持续追加每轮实验目的、配置、指标、结论 | 已满足 |
| 有效 commit 保存 | 最近 commit 覆盖每轮关键代码和证据：`14adc9f`、`17ec423`、`23195e0`、`e8eedee`、`c57a4b3`、`2509b06`、`8b4d84c`、`0e2e8fe`、`95210bf`、`9e8394c` | 已满足 |
| 发挥 B200 算力 | batch32 / rollout32 / val@16 运行，稳定 full rollout generation 约 22-32s，chunk actor update 在 hardfilter/clip4 后可到约 1s，support_anchor diag-off update 约 3s | 部分满足 |
| 不把 short-probe local hit 当主 teacher | 最新方向已转向 future-support / TV / support-anchor；support_anchor 完全跳过 probe teacher，score 来自 full rollout support mass | 已满足 |

## 关键实验链路

### 1. Local-probe / majority-style chunk target

早期版本包括：

- `majority_completion`
- `majority_conf_gate`
- `majority_consistent`
- `answer_consensus`
- `answer_distribution`
- `answer_value_gain`
- `answer_value_margin`
- `teacher_anchor`
- `sourcechunk`

结论：

- 工程可跑，但语义仍靠短 horizon probe / local answer hit 定义 target。
- 20-step 指标没有接近 MV baseline；部分版本出现 repeated boxed、OOV、target 稀疏。
- 该路线被用户明确否定，不再作为主方向。

### 2. Full-rollout support driven target

核心实现：

- `_score_chunk_state_answer_support_mass`
- `_score_chunk_state_future_support_gain`
- `chunk_state_target_prior`
- `source_chunk` 作为 prior / drift guard，而不是 hard teacher floor。

关键结论：

- future-support-gain 方向更符合当前理论，因为 score 来自 full rollout answer support/value。
- 但 3-step smoke 中 `support_coverage` 通常只有约 0.48-0.51，`OOV` 约 0.49-0.52，raw gain 均值经常为负。
- `hardfilter + clip4` 能把 actor update 降到约 1s，并压住尖权重，但不能修复 target quality。

代表证据：

```text
future_support_gain hardfilter_clip4:
  support_coverage ~= 0.48-0.51
  OOV ~= 0.49-0.52
  update_actor step2/3 ~= 1.35s / 0.98s
  powerflow_weight_max after clip ~= 2.0
```

### 3. Candidate / source / state gates

已验证：

- candidate quality gate
- source support gate
- state learnability gate

结论：

- candidate gate 基本不改变 positive-gain target，过滤前后 `score_mean` 和 `label_consistent_ratio` 近似一致。
- source gate 让 source 更正确，但有效 actor samples 掉到 8/8/16，OOV 反而更高。
- state gate 能筛出少量可学习 state，但比例只有 3%-9%，训练信号过稀疏。
- 继续堆 gate 不是主解法。

### 4. TV / transport-style score

已实现：

```text
source_tv = TV(one_hot(source_answer), prompt_full_support_dist)
candidate_tv = TV(candidate_probe_answer_dist, prompt_full_support_dist)
score = clamp(source_tv - candidate_tv, 0, 1)
```

结果：

```text
tv_gain_mean = -0.090 / -0.032 / -0.107
num_actor_samples = 8 / 16 / 8
```

结论：

- TV transport 叙事正确，但短 probe empirical distribution 平均比 source one-hot 更远离 full support。
- 这进一步证明“局部短视可判定性”是主矛盾。

### 5. Support-anchor no-probe target

最新实现：

- `_apply_chunk_state_support_anchors`
- `_score_chunk_state_support_anchor`
- 同 prompt full rollout 中高 support 的 next chunk 注入 candidates。
- anchor chunk score 直接来自 full-group answer support mass。
- `chunk_state_probe/skipped_for_support_anchor=1.0`。

3-step diag-off：

```text
chunk_state_score = 0.001s
chunk_state_ref ~= 0.84-1.02s after warmup
update_actor ~= 2.6-3.3s
num_actor_samples = 128
```

20-step final val：

```text
val-core/math/acc/mean@16 = 0.43725
val-core/math/acc/maj@16  = 0.558596
val-core/math/acc/best@16 = 0.83514
testing final = 301.249s
stable steps 2-19:
  gen mean = 23.650s
  chunk_state_chunks mean = 1.011s
  chunk_state_ref mean = 0.965s
  update_actor mean = 3.057s
```

结论：

- infra 是正结果：不靠 probe teacher 也能稳定给满 target density，chunk score 几乎无开销。
- 训练效果是负结果：naive support-anchor distillation 远低于 MV baseline。
- 它缺少真正的 search-improvement / future-distribution transport，只是在学高 support 完整轨迹里的局部 next chunk。

## Completion Audit

### 已完成

- 代码入口和配置开关已经建立。
- 8 卡 B200 训练/验证链路已跑通。
- 多轮 3-step smoke 和 20-step pilot 已执行。
- 中文实验记录持续落盘。
- 关键代码、脚本、日志已分批 commit。
- `dynamic batch` 未作为默认 infra 优化引入。
- PowerFlow-style chunk actor update 是主路径。

### 未完成 / 不应声称完成

- 没有达到 20-step gate。当前最干净的 support_anchor no-probe 20-step 指标显著低于 MV baseline。
- 没有扩到 80-step，因为 gate 没过；按 goal 文档，这种情况应该先停下来检查 scoring/loss，而不是盲跑。
- 还没有找到能让 chunk target 稳定表示 search-improvement 的 score。
- B200 的算力利用在 rollout generation 上较好，但 chunk target 质量不足，不能通过继续加算力解决。

## 当前总判断

这 24h 实验完成了“工程可行性 + 方法负结果定位”：

1. Chunk-level PowerFlow actor update 在 TTRL/verl 内可跑，且可以很快。
2. `hardfilter + clip4` 是有效 infra 稳定器。
3. 单纯 local-probe / source-consistency / candidate gate / source gate / state gate 都不能解决 target quality。
4. support_anchor 证明“不用 short probe teacher”工程上可行，但 naive anchor distillation 训练效果很差。
5. 主矛盾不是 actor update 慢，而是 chunk target 还没有真正表达 full-rollout group 的 future distribution improvement。

## 下一轮建议

下一轮不应再横向堆 gate 或重复 3-step smoke。建议收敛到一个新 target：

```text
先 full rollout 32/64 条，得到 prompt answer support distribution 和 trajectory value。
从高质量 rollout 的中后段抽 state。
candidate 不靠短 probe 单独定 teacher。
对 candidate 做更长 horizon / multi-stage future evaluation，估计：
  future support mass gain
  top answer value margin
  distribution transport improvement
  与 source continuation 的 KL/TV improvement
再用 PowerFlow weighted distribution matching 更新 next chunk。
```

更具体的 v2 gate：

```text
3-step only:
  target_nonzero_ratio > 0.25
  OOV ratio < 0.40
  repeated boxed ratio < 0.02
  source-correct state future value > wrong state future value by clear margin
  num_actor_samples >= 64
  update_actor <= 3s

只有这些过线，才跑 20-step final val。
```

## 相关 commit

```text
14adc9f Add future support gain chunk target
17ec423 Record strict future support gain chunk smoke
23195e0 Add hard-filtered future support chunk smoke
e8eedee Stabilize chunk PowerFlow weights
c57a4b3 Add chunk candidate support filter
2509b06 Add chunk source support gate
8b4d84c Add chunk state learnability gate
0e2e8fe Add chunk future-support TV smoke
95210bf Add full-support chunk anchor smoke
9e8394c Record support-anchor diagoff run
```
