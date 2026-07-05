# TTRL/SPS 中文简版总结

## 设计思想

目标是在不使用 Math500 标注做训练奖励的前提下，用模型内部信号做短步数 test-time RL，并把采样分布锐化到更可靠的答案簇。真实答案只用于 final validation 和诊断。

当前最有效的路线不是把 SPS/base-logprob 直接当 dense reward，而是保留 majority pseudo label 作为正确性代理，再用 SPS/PowerFlow 风格的内部概率信号做置信度估计、答案簇锐化和训练容量控制。

训练侧采用 continuous confidence capacity：低置信、高截断、不可解析的 prompt 降低更新强度；高置信答案簇获得更大权重。推理侧采用无监督 answer-cluster selection：final validation 生成多条候选，只按解析出的 `pred` 做答案簇多数选择，不看 ground truth，然后把选中的答案回填到 `mean@4` 评估路径。

## 实验结果

| 场景 | 最好版本 | 训练步数 | 核心设计 | Math500/MATH-TTT 结果 |
|---|---|---:|---|---:|
| 早期 direct SPS reward | direct SPS | 185 | 直接用 SPS 替代 majority reward | `mean@4=69.9%` |
| 原版 majority TTRL | baseline | 185 | majority pseudo label | `mean@4=88.8%` |
| Qwen3-4B 50-step | v14 | 50 | majority + confidence capacity + answer sharpening | `mean@4=84.10%`, `best@4=88.29%` |
| Qwen3-8B 长跑 | v26 | 310 | hard-bucket answer sharpening + support projection | `mean@4=89.39%`, `maj@4=90.07%` |
| Qwen3-8B 复验 | v26 repeat | 620 | v26 + throughput 优化路径 | `mean@4=88.13%`, `best@4=92.20%` |
| Qwen3-4B 20-step | v31 | 20 | 直接压缩 v26 hard support projection | `mean@4=57.14%` |
| Qwen3-4B 20-step | v32 | 20 | 回到 v14 confidence capacity 主干 | `mean@4=61.07%`, `best@4=69.15%` |
| Qwen3-4B 20-step | v33 | 20 | v32 训练 + validation-time answer-cluster selection | `mean@4=79.07%` |

v33 达成当前 Efficient Test-Time RL 目标：20 个训练 step 后，Qwen3-4B 在 Math500/MATH-TTT 上 `val-core/MATH-TTT/acc/mean@4=0.7907444668008048`，超过 75% 目标。对应 raw uncollapsed validation 为 `mean@32=61.11%`、`best@32=78.21%`、`maj@32=61.82%`，说明提升主要来自无监督答案簇选择把候选分布锐化到了更可靠的答案。

当前结论：SPS 信号更适合作为置信度、容量和推理选择信号，而不是独立 reward。短预算下只靠 20-step GRPO 更新不够，必须把 majority/SPS 启发的 test-time answer selection 接入 final inference。
