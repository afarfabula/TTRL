# TTRL/SPS 中文简版总结

## 设计思想

目标是在不使用 Math500 标注做训练奖励的前提下，用模型内部信号做短步数 test-time RL，并把采样分布锐化到更可靠的答案簇。真实答案只用于 final validation 和诊断。

当前最有效的路线不是把 SPS/base-logprob 直接当 dense reward，而是保留 majority pseudo label 作为正确性代理，再用 SPS/PowerFlow 风格的内部概率信号做置信度估计、答案簇锐化和训练容量控制。

训练侧采用 continuous confidence capacity：低置信、高截断、不可解析的 prompt 降低更新强度；高置信答案簇获得更大权重。旧的 v33/v34 还在推理侧采用无监督 answer-cluster selection：final validation 生成多条候选，只按解析出的 `pred` 做答案簇多数选择，不看 ground truth，然后把选中的答案回填到 `mean@4` 评估路径。

当前新 goal 已收紧口径：最终成功指标必须是 strict `val_kwargs.n=4`，不能使用 validation n=32、best-of、major vote 或 answer selection。v33/v34 的 83% 只能作为诊断，不算达标。

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
| Qwen3-4B 20-step seed1 | v33 | 20 | 同上，seed sweep 的第一个有效 seed | `mean@4=80.08%` |
| Qwen2.5-Math-7B 20-step | v33 | 20 | 同一套 v33 训练 + validation-time answer-cluster selection | `mean@4=83.10%` |
| Qwen2.5-Math-7B 20-step strict | v35 | 20 | v33/v14 训练路径 + validation `n=4` + 禁用 answer selection | `mean@4=68.96%`, `best@4=82.20%`, `maj@4=71.37%` |
| Qwen2.5-Math-7B 20-step strict | v36 | 20 | v35 + sharpened answer-cluster capacity | `mean@4=70.37%`, `best@4=83.00%`, `maj@4=72.27%` |

v33 达成 Efficient Test-Time RL 目标：20 个训练 step 后，Qwen3-4B 在 Math500/MATH-TTT 上 `val-core/MATH-TTT/acc/mean@4=0.7907444668008048`，超过 75% 目标。Qwen3-4B seed sweep 的第一个有效 seed 达到 `0.8008048289738431`，随后因目标模型切换暂停。

同一套方案换到 Qwen2.5-Math-7B 后达到 `val-core/MATH-TTT/acc/mean@4=0.8309859154929577`。对应 raw uncollapsed validation 为 `mean@32=70.57%`、`best@32=93.19%`、`maj@32=81.90%`，说明 Qwen2.5-Math 的候选 support 明显更强，answer-cluster selection 能把一部分 high-best 候选质量转成 selected `mean@4`。但这依赖 validation `n=32`，不符合当前 strict n=4 goal。

v35 重新建立 strict n=4 baseline：最终 validation 样本数 `1988=497*4`，没有 validation answer selection，`mean@4=0.6896378269617707`，`best@4=0.8220402414486921`，`maj@4=0.7137223340040241`。这证明当前差距不是评估聚合问题，而是 4 条样本本身的正确率还不够。

v36 启用 `sps_answer_sharpen_capacity=True`，把 sharpened answer-cluster confidence 直接用于训练容量。它在 strict n=4 下小幅提升到 `mean@4=0.7037223340040242`，说明这个内部锐化信号方向有效，但仍远低于 85%。v36 跑完后 worker 出现 `/proc` 损坏和 `cuInit=304`，后续 GPU 实验必须先换健康 worker 或重启修复。

当前结论：SPS 信号更适合作为训练期置信度、容量和分布锐化信号，而不是独立 dense reward；当前 goal 下不能再靠推理时多采样选择。下一步必须提升 4 条 rollout 自身的候选质量，因为 v36 的 strict `best@4=83.00%` 仍低于 85%。
