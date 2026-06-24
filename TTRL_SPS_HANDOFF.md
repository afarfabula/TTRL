# TTRL × SPS 实验交接文档

最后更新：2026-06-25

本文档记录在 `/opt/tiger/TTRL`（内嵌 verl 0.4.1 + TTRL 补丁）上，把「SPS（base-model 序列 logprob）reward」接入 TTRL 替代 majority voting 的实现、实验、结论与后续优化方向。

---

## 1. 背景与动机

- TTRL（Test-Time RL，arXiv:2504.16084）= 无标注测试时强化学习。原版用 **majority voting** 产生伪标签：对每个 prompt 采 N 条 rollout，多数投票出的答案当作 ground truth，再用规则判分给每条 rollout 0/1 reward，喂 GRPO。
- 我们之前在 `reasoning-with-sampling` 里发现 **SPS（K=32 低温采样 + base model 序列 logprob 打分）** 在 Math500 推理选择上很有效。
- 本次目标：**用 SPS 的 base-logprob 信号替代 majority voting 作为 TTRL 的 reward**，看是否能更好。
- 模型 Qwen3-4B，数据 MATH-TTT（=Math500，500 题），GRPO，5 epoch。

---

## 2. 代码改动（均在 /opt/tiger/TTRL/verl）

### 2.1 新增：SPS reward 计算
- `verl/trainer/ppo/sps_utils.py`（新文件）
  - `compute_sps_reward(ref_log_prob, rollout_log_probs, response_mask, n, alpha, ...)`
  - 两种 reward 模式：
    - `group_norm_base`（默认，RL 友好）：`score = (alpha*logp_base - logq)/len`，再在每个 prompt 的 K 条 rollout 内做 **z-score 标准化** → 稠密、零均值，匹配 GRPO 的 group advantage。
    - `softmax_weight`（消融用）：`softmax_K(logw)`，会塌缩到单条 rollout（effective_K≈1），不适合当 RL reward。

### 2.2 编排：训练主循环
- `verl/trainer/ppo/ray_trainer.py`
  - 在 `fit()` 的生成阶段加 SPS 分支（`ttrl.sps_enable=True` 时）：
    1. 低温采样 K=32 条 rollout（`rollout_log_probs` 即 proposal logq，需开 `calculate_log_probs`）。
    2. 用 base/ref model 在 **温度=1.0** 下打分得 `logp_base`（通过新增的 `ref_temperature_override`）。
    3. `compute_sps_reward` 算每条 rollout 的 reward，写入 `token_level_scores`（last valid token）。
  - reward 覆盖、SPS 监控指标（effective_K / pick_accuracy / pass@K）均加了 `sps_enable` 的分支保护，不影响原 majority-voting 路径。
  - 关键修复：vLLM 返回的 TensorDict 是 locked 的，不能 `batch["x"]=...`，改用 `DataProto.from_dict + union`。

### 2.3 worker：base 模型温度可覆盖
- `verl/workers/fsdp_workers.py`：`compute_ref_log_prob` 支持 `meta_info["ref_temperature_override"]`。
  - 原因：verl 的 logprob 计算会把 logits 除以温度（dp_actor.py:184），默认用 `rollout.temperature`。SPS 需要 base 在 **温度=1.0** 下的真实序列 logprob，故必须可覆盖。

### 2.4 配置与脚本
- `verl/trainer/config/ppo_trainer_ttrl.yaml`：新增 `ttrl.sps_enable / sps_proposal_temperature / sps_reward_mode / sps_length_normalize / sps_weight_temperature`。
- `verl/examples/ttrl/run_sps_math_qwen3_4b.sh`：SPS-TTRL 训练脚本（4 卡 0-3，T=0.4，K=32，5 epoch，test_freq=5）。
- `verl/examples/ttrl/run_majvote_math_qwen3_4b.sh`：majority-voting baseline（4 卡 4-7，同口径对比，仅 `sps_enable=False`、采样 T=1.0）。
- `verl/probe.sh`：一键探测两条 run 的进度/GPU/val 曲线/健康度/报错。
  - 注意 majvote 用了 `RAY_TMPDIR=/tmp/ray_majvote`，其 session 真实路径是 `/tmp/ray_majvote/ray/session_latest`（多一层 `ray/`）。

---

## 3. 实验设置（两条对比 run，同 baseline 0.537）

| 维度 | SPS-TTRL | MajVote-TTRL |
|---|---|---|
| GPU | 0-3 | 4-7 |
| reward | SPS group_norm_base | majority voting 0/1 |
| 采样温度 | **0.4**（低温） | **1.0** |
| K / n_votes | 32 | 64 投票 → 32 训练 |
| 其余 | Qwen3-4B / Math500 / GRPO / lr 5e-7 / 5 epoch / test_freq=5 | 同左 |

> 踩坑记录：
> 1. 初版用 `softmax_weight` reward → effective_K 塌缩到 1，梯度稀疏，val 原地踏步。改 `group_norm_base` 后 effective_K≈32，reward 稠密。
> 2. 单步耗时一度 285s，定位到 81% 花在「每步都 val」。`test_freq=1→5` 后纯训练步降到 ~53s。
> 3. val 慢的根因：500题×n4×3072token 长解码长尾。已用 TP=1 + gpu_mem=0.8。

---

## 4. 结果（关键负结果）

Math500 验证集 `acc/mean@4`（主指标）：

| step | SPS mean@4 | SPS best@4 | MajVote mean@4 | MajVote best@4 |
|---:|---:|---:|---:|---:|
| 0 (baseline) | 0.537 | 0.622 | 0.537 | 0.622 |
| 15 | 0.584 | 0.661 | 0.620 | 0.691 |
| 30 | 0.651 | 0.720 | 0.755 | 0.811 |
| 50 | 0.678 | 0.738 | 0.813 | 0.862 |
| 70 | 0.695 | 0.750 | 0.845 | 0.889 |
| 95 | 0.695 | 0.755 | 0.865 | 0.912 |
| 120 | 0.719 | 0.769 | 0.870 | 0.913 |
| 150 | 0.704 | 0.767 | 0.873 | 0.912 |
| 185 | 0.699 | 0.746 | **0.888** | **0.923** |
| 250 | 0.698 | 0.758 | （run 仍在进行） | |

**结论：majority voting 大幅胜出。**
- MajVote：0.537 → **0.888**（+35pp），单调上升、仍在涨，best@4 到 0.923。
- SPS：step120 触顶 ~0.719 后进入平台并轻微回落，**全程未超过 0.72**（+18pp 后停滞）。
- **本次实验证伪了「SPS reward 替代 majority voting 会更好」的假设。**

---

## 5. 原因分析（背后的启发）

### 5.1 探索坍缩：低温采样让 SPS 先天近视（核心）
- policy entropy：SPS step1=0.095（一路降到 step95=0.042）；MajVote step1=0.265。SPS 探索性只有 MajVote 的 ~1/3。
- 低温 T=0.4 让 32 条 rollout 高度同质化，SPS reward 是在「一堆几乎相同的样本里抠微小差异」，信噪比低、梯度方向缺乏多样性 → 平滑慢爬、早早平台。
- **启发：TTRL/test-time RL 里 rollout 多样性（探索）比 reward 的精细度更重要。推理时低温为了选最优，训练时低温却扼杀探索——场景错配。**

### 5.2 reward 目标偏了：SPS 优化「base 偏好」而非「正确性」
- MajVote reward 直接代理正确性（实测 label_accuracy 0.875~1.0，伪标签质量高）。
- SPS reward = `α·logp_base − logq`，本质是「哪条更像 base model 高概率输出」。但 **base 高概率 ≠ 答案正确**，数学题上流畅常见的路径未必对。SPS 在优化一个与正确性只弱相关的目标。
- **启发：base-logprob 信号适合「从一批里挑一个」（推理时选择），但作为 RL 奖励，优化目标偏离了正确性。选择任务 ≠ 奖励任务。**

### 5.3 reward 形态过软
- 组内 z-score 抹掉了「题目难易」的绝对信息，且连续零均值奖励比 0/1 稀疏奖励对 GRPO 的推进更温和 → 又慢一档。

---

## 6. 后续优化方向（按性价比排序）

### 🥇 A. 解耦温度：训练高温探索 + SPS 照常打分
- rollout 采样温度提到 1.0（保证探索），SPS 打分公式不变（α 仍可用 1/0.25）。
- 当前代码里采样温度与 SPS α 绑死，需要拆成两个独立参数。
- 最小改动、最可能立刻见效，用来回答「SPS 是被低温拖累，还是奖励目标本身偏了」。

### 🥈 B. SPS 当「软加权」增强 majority voting，而非替代
- 主信号仍是正确性（投票 / self-consistency 的 0/1）。
- 用 **answer-marginalized SPS** 给伪标签纠偏：按 `S(a)=logsumexp({logw_i | answer=a})` 聚合每个答案的证据，选 top 答案当伪标签（= base-logprob 加权的多数票）。
- 在多数票不确定时（majority_ratio 低，如实测 step5=0.31）能纠偏。这是最有前途的结合点。

### 🥉 C. reward 改成更「硬」的排序奖励
- SPS top-k rollout 给 +1、其余 0，或 rank-based reward，让信号强度接近 0/1。配合 A 的高温采样。

### 方法论
- **防熵坍缩**：SPS entropy 已降到 0.04，长跑有崩风险；TTRL 这类自奖励应加 entropy 监控/正则。
- **看 best@4 不只看 mean@4**：若 best@4（能力上界）不涨，只是 mean@4 靠「变确定」涨，是在压缩分布而非提升能力。

---

## 7. 复现命令

SPS-TTRL：
```bash
bash /opt/tiger/TTRL/verl/examples/ttrl/run_sps_math_qwen3_4b.sh
```
majority-voting baseline：
```bash
bash /opt/tiger/TTRL/verl/examples/ttrl/run_majvote_math_qwen3_4b.sh
```
监控两条 run：
```bash
bash /opt/tiger/TTRL/verl/probe.sh
```

## 8. 关联文档
- SPS 推理实验交接：`/opt/tiger/reasoning-with-sampling/llm_experiments/SPS_EXPERIMENT_HANDOFF.md`

---

## 9. 2026-06-25 新 goal：优化 SPS-TTRL 内部反馈方案

目标：停止 GPU 0-3 上旧 SPS-TTRL run，用 GPU 0-3 先跑一个 **中途不带 validation 的 50 step** 训练实验；等 GPU 4-7 的 MajVote run 结束后，可扩到 8 卡。目标指标：最终 Math500 `acc/mean@4 >= 0.85`。约束：训练反馈必须是无监督的模型内部信号，不使用真实答案作为训练 reward。

### 9.1 已执行的资源操作
- 旧 SPS-TTRL run：`math-qwen3_4b-sps-5ep`，GPU 0-3，停止前约 `step 259/310`。
- 已对旧 SPS run 的进程组发送 `TERM` 并确认 GPU 0-3 释放。
- GPU 4-7 上 MajVote run 继续保留；用户说明约 3 小时后可用 8 卡。

### 9.2 实验 v1：SPS-weighted self-consistency

动机：上一版 SPS 直接 dense reward 失败，主要问题是低温探索坍缩和 reward 目标偏软。新方案不再让 SPS 直接替代训练 reward，而是把 SPS 用在 **答案簇级别的伪标签选择** 上：

1. 高温采样 `N_VOTES=64` 条 rollout，`rollout_temperature=1.0`，恢复探索。
2. 对每条 rollout 计算 `logw = alpha * logp_base - logq`，其中 base/ref logprob 固定温度 1.0。
3. 按最终答案聚类，计算答案级分数：
   `S(answer)=logsumexp(logw_i / weight_temperature | answer_i=answer)`。
4. 选择 `S(answer)` 最大的答案作为无监督伪标签。
5. 下采样前 32 条 rollout 训练，reward 使用普通 math reward 对这个伪标签打 `0/1`。
6. 真实答案只用于已有诊断指标，不参与训练信号。

代码改动：
- `verl/trainer/ppo/ttrl_utils.py`
  - 新增 `apply_sps_weighted_ttrl_gt(...)`。
  - 负责 decode rollout、抽取 boxed answer、答案聚类、答案级 SPS logsumexp 加权、写入伪标签。
- `verl/trainer/ppo/ray_trainer.py`
  - `ttrl.sps_reward_mode=answer_weighted_vote` 时：
    - 生成数量使用 `ttrl.n_votes_per_prompt`。
    - 调用 `apply_sps_weighted_ttrl_gt` 生成伪标签。
    - 再 `select_top_k_per_prompt(..., n_samples_per_prompt)` 下采样训练。
    - 不写入 `sps_reward`，因此不会覆盖普通 0/1 reward。
  - 旧 `group_norm_base` / `softmax_weight` 路径保持不变。
- `verl/trainer/config/ppo_trainer_ttrl.yaml`
  - 新增 `answer_weighted_vote` 模式说明。
  - 新增 `ttrl.sps_weight_temperature_base`，用于把训练采样温度和 SPS alpha 解耦。
- `examples/ttrl/run_sps_weighted_vote_math_qwen3_4b_50step.sh`
  - 新 4 卡实验脚本。
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`
  - `RAY_TMPDIR=/tmp/ray_sps_weighted_vote`
  - `trainer.val_before_train=False`
  - `trainer.test_freq=50`
  - `trainer.total_training_steps=50`
  - 设计意图：无启动 validation、无中途 validation，只在最后 step 触发一次 validation。

启动命令：
```bash
cd /opt/tiger/TTRL/verl
nohup bash examples/ttrl/run_sps_weighted_vote_math_qwen3_4b_50step.sh \
  > /opt/tiger/TTRL/verl/sps_weighted_vote_50step.log 2>&1 &
```

结果记录：
- 已启动，Ray session：
  `/tmp/ray_sps_weighted_vote/ray/session_2026-06-25_00-49-04_007116_371528`
- step 1 已完成，新分支跑通到 actor update：
  - `train/sps/reward_mode=2.0`（`answer_weighted_vote`）
  - `train/sps/effective_K=63.884`，高温探索没有 softmax 塌缩。
  - `train/sps/weighted_label_confidence=0.750`
  - `train/sps/unique_answer_count=0.750`
  - `train/label_accuracy=0.750`（只用于诊断，不参与训练）
  - `train/reward_accuracy=1.000`
  - `train/majority_ratio=0.535`
  - `actor/entropy=0.263`
  - `response_length/mean=2480.480`，`clip_ratio=0.605`
  - `timing_s/step=73.180`
  - 未触发 step 0 / 中途 validation，符合 `val_before_train=False` 和 `test_freq=50` 设计。
- step 2 已完成，趋势良好：
  - `train/sps/effective_K=63.875`
  - `train/sps/weighted_label_confidence=0.868`
  - `train/sps/unique_answer_count=1.000`
  - `train/label_accuracy=0.875`
  - `train/pass@32=0.875`
  - `train/majority_ratio=0.631`
  - `actor/entropy=0.275`
  - `response_length/clip_ratio=0.512`
  - `timing_s/step=69.949`
- 2026-06-25 00:58 CST 进度：run 仍在 GPU 0-3 正常训练，已到 `step 5/50`，未出现中途 validation，符合实验设置。
- step 3：
  - `train/sps/effective_K=63.856`
  - `train/sps/weighted_label_confidence=0.875`
  - `train/sps/unique_answer_count=0.875`
  - `train/label_accuracy=0.875`
  - `train/pass@32=0.875`
  - `train/majority_ratio=0.621`
  - `actor/entropy=0.238`
  - `response_length/clip_ratio=0.535`
  - `timing_s/step=70.697`
- step 4：
  - `train/sps/effective_K=63.877`
  - `train/sps/weighted_label_confidence=0.372`
  - `train/sps/unique_answer_count=0.500`
  - `train/label_accuracy=0.375`
  - `train/pass@32=0.375`
  - `train/majority_ratio=0.338`
  - `actor/entropy=0.282`
  - `response_length/clip_ratio=0.738`
  - `timing_s/step=71.791`
- step 5：
  - `train/sps/effective_K=63.892`
  - `train/sps/weighted_label_confidence=0.643`
  - `train/sps/unique_answer_count=1.125`
  - `train/label_accuracy=0.750`
  - `train/pass@32=0.750`
  - `train/majority_ratio=0.299`
  - `actor/entropy=0.243`
  - `response_length/clip_ratio=0.750`
  - `timing_s/step=69.473`
- 观察：SPS 加权分布本身没有塌缩（`effective_K` 稳定在 63.8+），但 response clipping 在 step 4-5 明显升高到约 0.74-0.75，可能限制最终 Math500 表现；继续等 50 step 最终验证后再决定是否改 `max_response_length`/rollout 温度/采样过滤。
- 2026-06-25 01:05 CST 进度：run 仍在 GPU 0-3 正常训练，已到 `step 11/50`，仍未出现中途 validation。
- step 6-11 摘要：
  - step 6：`effective_K=63.871`，`weighted_label_confidence=0.826`，`label_accuracy=0.500`，`pass@32=0.750`，`majority_ratio=0.516`，`entropy=0.209`，`clip_ratio=0.699`。
  - step 7：`effective_K=63.877`，`weighted_label_confidence=0.616`，`label_accuracy=0.375`，`pass@32=0.625`，`majority_ratio=0.570`，`entropy=0.259`，`clip_ratio=0.566`。
  - step 8：`effective_K=63.891`，`weighted_label_confidence=0.625`，`label_accuracy=0.625`，`pass@32=0.625`，`majority_ratio=0.562`，`entropy=0.268`，`clip_ratio=0.508`。
  - step 9：`effective_K=63.902`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.750`，`majority_ratio=0.672`，`entropy=0.220`，`clip_ratio=0.477`。
  - step 10：`effective_K=63.902`，`weighted_label_confidence=1.000`，`label_accuracy=1.000`，`pass@32=1.000`，`majority_ratio=0.742`，`entropy=0.238`，`clip_ratio=0.559`。
  - step 11：`effective_K=63.878`，`weighted_label_confidence=0.375`，`label_accuracy=0.375`，`pass@32=0.375`，`majority_ratio=0.252`，`entropy=0.286`，`clip_ratio=0.766`。
- 中间判断：`effective_K` 持续接近 64，说明 SPS answer-weighted vote 没有退化成单样本选择；小批次质量波动较大，且难批次通常伴随高 `clip_ratio`。若最终指标不达标，优先尝试降低长输出截断（更短 prompt/限制思考长度/调低 rollout 温度或增大可用 response length），其次再调 SPS 权重温度。
- 2026-06-25 01:16 CST 进度：run 仍在 GPU 0-3 正常训练，已到 `step 21/50`，仍未出现中途 validation。
- step 12-21 摘要：
  - step 12：`effective_K=63.902`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.521`，`entropy=0.256`，`clip_ratio=0.504`。
  - step 13：`effective_K=63.891`，`weighted_label_confidence=0.875`，`label_accuracy=0.750`，`pass@32=0.875`，`majority_ratio=0.570`，`entropy=0.273`，`clip_ratio=0.512`。
  - step 14：`effective_K=63.898`，`weighted_label_confidence=1.000`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.740`，`entropy=0.239`，`clip_ratio=0.488`。
  - step 15：`effective_K=63.925`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.779`，`entropy=0.199`，`clip_ratio=0.281`。
  - step 16：`effective_K=63.896`，`weighted_label_confidence=0.625`，`label_accuracy=0.625`，`pass@32=0.500`，`majority_ratio=0.500`，`entropy=0.235`，`clip_ratio=0.500`。
  - step 17：`effective_K=63.932`，`weighted_label_confidence=1.000`，`label_accuracy=1.000`，`pass@32=1.000`，`majority_ratio=0.863`，`entropy=0.207`，`clip_ratio=0.316`。
  - step 18：`effective_K=63.937`，`weighted_label_confidence=0.891`，`label_accuracy=0.750`，`pass@32=0.875`，`majority_ratio=0.785`，`entropy=0.187`，`clip_ratio=0.352`。
  - step 19：`effective_K=63.904`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.604`，`entropy=0.213`，`clip_ratio=0.520`。
  - step 20：`effective_K=63.862`，`weighted_label_confidence=0.745`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.594`，`entropy=0.241`，`clip_ratio=0.496`。
  - step 21：`effective_K=63.909`，`weighted_label_confidence=1.000`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.531`，`entropy=0.210`，`clip_ratio=0.641`。
- 中间判断：step 12-21 的小批次训练质量大多恢复到 `pass@32=0.75-1.0`，但 `clip_ratio` 仍在 0.28-0.64 波动。当前不提前修改 run，等待 step 50 最终 Math500 `mean@4` 决策。
- 2026-06-25 01:27 CST 进度：run 仍在 GPU 0-3 正常训练，已到 `step 30/50`，仍未出现中途 validation。
- step 22-30 摘要：
  - step 22：`effective_K=63.896`，`weighted_label_confidence=0.875`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.723`，`entropy=0.207`，`clip_ratio=0.402`。
  - step 23：`effective_K=63.908`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.498`，`entropy=0.195`，`clip_ratio=0.727`。
  - step 24：`effective_K=63.936`，`weighted_label_confidence=1.000`，`label_accuracy=1.000`，`pass@32=1.000`，`majority_ratio=0.928`，`entropy=0.192`，`clip_ratio=0.234`。
  - step 25：`effective_K=63.912`，`weighted_label_confidence=1.000`，`label_accuracy=1.000`，`pass@32=1.000`，`majority_ratio=0.891`，`entropy=0.162`，`clip_ratio=0.188`。
  - step 26：`effective_K=63.932`，`weighted_label_confidence=0.998`，`label_accuracy=0.875`，`pass@32=1.000`，`majority_ratio=0.873`，`entropy=0.170`，`clip_ratio=0.246`。
  - step 27：`effective_K=63.919`，`weighted_label_confidence=0.866`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.703`，`entropy=0.217`，`clip_ratio=0.410`。
  - step 28：`effective_K=63.941`，`weighted_label_confidence=1.000`，`label_accuracy=0.875`，`pass@32=1.000`，`majority_ratio=0.777`，`entropy=0.175`，`clip_ratio=0.410`。
  - step 29：`effective_K=63.959`，`weighted_label_confidence=0.974`，`label_accuracy=1.000`，`pass@32=0.875`，`majority_ratio=0.863`，`entropy=0.160`，`clip_ratio=0.172`。
  - step 30：`effective_K=63.921`，`weighted_label_confidence=1.000`，`label_accuracy=1.000`，`pass@32=1.000`，`majority_ratio=0.811`，`entropy=0.184`，`clip_ratio=0.320`。
- 中间判断：step 24-30 明显优于前段，训练小批次 `majority_ratio` 上升且 `clip_ratio` 多数低于 0.45；继续等待 final-only validation。
- 2026-06-25 01:38 CST 进度：run 仍在 GPU 0-3 正常训练，已到 `step 40/50`，仍未出现中途 validation。
- step 31-40 摘要：
  - step 31：`effective_K=63.904`，`weighted_label_confidence=0.824`，`label_accuracy=0.625`，`pass@32=0.875`，`majority_ratio=0.570`，`entropy=0.199`，`clip_ratio=0.523`。
  - step 32：`effective_K=63.933`，`weighted_label_confidence=0.748`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.744`，`entropy=0.188`，`clip_ratio=0.262`。
  - step 33：`effective_K=63.947`，`weighted_label_confidence=0.750`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.750`，`entropy=0.159`，`clip_ratio=0.250`。
  - step 34：`effective_K=63.925`，`weighted_label_confidence=1.000`，`label_accuracy=0.875`，`pass@32=1.000`，`majority_ratio=0.711`，`entropy=0.210`，`clip_ratio=0.449`。
  - step 35：`effective_K=63.918`，`weighted_label_confidence=0.623`，`label_accuracy=0.625`，`pass@32=0.625`，`majority_ratio=0.506`，`entropy=0.210`，`clip_ratio=0.660`。
  - step 36：`effective_K=63.647`，`weighted_label_confidence=0.891`，`label_accuracy=0.875`，`pass@32=1.000`，`majority_ratio=0.850`，`entropy=0.193`，`clip_ratio=0.234`。
  - step 37：`effective_K=63.919`，`weighted_label_confidence=0.750`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.734`，`entropy=0.200`，`clip_ratio=0.367`。
  - step 38：`effective_K=63.925`，`weighted_label_confidence=0.871`，`label_accuracy=0.750`，`pass@32=0.875`，`majority_ratio=0.645`，`entropy=0.195`，`clip_ratio=0.379`。
  - step 39：`effective_K=63.917`，`weighted_label_confidence=0.941`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.717`，`entropy=0.151`，`clip_ratio=0.398`。
  - step 40：`effective_K=63.932`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.863`，`entropy=0.219`，`clip_ratio=0.156`。
- 中间判断：step 31-40 仍有单步难批次（step 35），但最近几步 `clip_ratio` 已明显低于前段，`effective_K` 除 step 36 外都稳定在 63.9 左右。继续等 step 50 final validation。
- 2026-06-25 01:53 CST final：run 完整结束，GPU 0-3 已释放，最终验证只在 step 50 触发。
- step 41-50 摘要：
  - step 41：`effective_K=63.897`，`weighted_label_confidence=0.998`，`label_accuracy=0.750`，`pass@32=1.000`，`majority_ratio=0.896`，`entropy=0.164`，`clip_ratio=0.426`。
  - step 42：`effective_K=63.928`，`weighted_label_confidence=0.998`，`label_accuracy=0.875`，`pass@32=1.000`，`majority_ratio=0.775`，`entropy=0.184`，`clip_ratio=0.336`。
  - step 43：`effective_K=63.904`，`weighted_label_confidence=1.000`，`label_accuracy=0.875`，`pass@32=1.000`，`majority_ratio=0.496`，`entropy=0.199`，`clip_ratio=0.676`。
  - step 44：`effective_K=63.914`，`weighted_label_confidence=0.875`，`label_accuracy=0.750`，`pass@32=0.875`，`majority_ratio=0.590`，`entropy=0.182`，`clip_ratio=0.441`。
  - step 45：`effective_K=63.942`，`weighted_label_confidence=1.000`，`label_accuracy=0.750`，`pass@32=1.000`，`majority_ratio=0.617`，`entropy=0.178`，`clip_ratio=0.547`。
  - step 46：`effective_K=63.936`，`weighted_label_confidence=0.869`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.721`，`entropy=0.214`，`clip_ratio=0.383`。
  - step 47：`effective_K=63.930`，`weighted_label_confidence=0.875`，`label_accuracy=0.750`，`pass@32=0.875`，`majority_ratio=0.768`，`entropy=0.189`，`clip_ratio=0.340`。
  - step 48：`effective_K=63.933`，`weighted_label_confidence=0.750`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.703`，`entropy=0.188`，`clip_ratio=0.418`。
  - step 49：`effective_K=63.904`，`weighted_label_confidence=0.750`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.723`，`entropy=0.171`，`clip_ratio=0.371`。
  - step 50：`effective_K=63.943`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.803`，`entropy=0.164`，`clip_ratio=0.309`。
- Final Math500 validation:
  - `val-core/MATH-TTT/acc/mean@4=0.738430583501006`
  - `val-core/MATH-TTT/acc/best@4/mean=0.7992957746478874`
  - `val-core/MATH-TTT/acc/maj@4/mean=0.741702213279678`
  - `val-aux/MATH-TTT/acc/worst@4/mean=0.6725975855130784`
  - `val-aux/MATH-TTT/format_score/mean@4=0.7484909456740443`
- 结论：
  - 相比旧 direct-SPS 50 step `mean@4=0.678`，v1 提升到 `0.7384`，绝对提升约 `+6.04pp`，满足“有提升的改动要本地 commit 记录”。
  - 未达到 goal 要求的 `mean@4>=0.85`，不能标记 goal complete。
  - 该方案恢复了高温探索并避免 SPS softmax 坍缩，但最终 `best@4=0.7993`，说明 50 step 后能力上界仍不够；下一轮需要更强的伪标签质量或更贴近 MajVote 的稳定性。

### 9.3 下一轮候选

目标是在保留无监督内部反馈的前提下向 MajVote 50-step/长跑表现靠近，优先使用已释放的 GPU 0-3；GPU 4-7 上 MajVote 仍在跑，暂不可用。

候选 v2：
1. **Hybrid pseudo-label fallback**：先做普通 majority vote；仅当 answer-level SPS 与 majority 冲突且 SPS confidence 足够高时替换伪标签。动机：v1 完全用 SPS 选伪标签后 `best@4` 仍偏低，说明 SPS 单独主导会覆盖掉部分稳定多数票。
2. **Clip-aware rollout**：降低 rollout 温度到 0.8 或把 response 长度/过滤策略调到减少截断，避免高 `clip_ratio` 批次污染伪标签。
3. **Confidence gate**：当 weighted confidence 低或 unique answer 太少时不训练该 prompt 或回退 majority label，减少 step 11/35 这类难批次的错误强化。

### 9.4 实验 v2：SPS-gated majority fallback

启动前判断：v1 的 `mean@4=0.7384` 虽比旧 SPS 50 step 的 `0.678` 提升，但 `best@4=0.7993` 不够，说明 SPS 全量替换 majority 伪标签仍会降低候选答案质量。v2 改为 majority-first：只有在 majority 不够强且 SPS 答案级置信度足够高时，才允许 SPS 覆盖伪标签。

代码改动：
- `verl/trainer/ppo/ttrl_utils.py`
  - `apply_sps_weighted_ttrl_gt(...)` 新增 gate 参数：
    - `use_majority_fallback`
    - `gate_confidence_threshold`
    - `gate_majority_ratio_threshold`
  - 同时记录 `raw_majority_gt`、`sps_weighted_gt`、`sps_override_list`、`sps_agreement_list`。
- `verl/trainer/ppo/ray_trainer.py`
  - 新增 `ttrl.sps_reward_mode=answer_weighted_gate`。
  - 该模式 `train/sps/reward_mode=3.0`，额外输出：
    - `train/sps/override_rate`
    - `train/sps/agreement_rate`
- `verl/trainer/config/ppo_trainer_ttrl.yaml`
  - 新增 gate 模式说明和阈值：
    - `ttrl.sps_gate_confidence_threshold=0.8`
    - `ttrl.sps_gate_majority_ratio_threshold=0.75`
- `examples/ttrl/run_sps_weighted_gate_math_qwen3_4b_50step.sh`
  - 新 4 卡 v2 实验脚本。
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`
  - `RAY_TMPDIR=/tmp/ray_sps_weighted_gate`
  - `trainer.val_before_train=False`
  - `trainer.test_freq=50`
  - `trainer.total_training_steps=50`

启动前验证：
- `py_compile` 通过：
  - `verl/trainer/ppo/ttrl_utils.py`
  - `verl/trainer/ppo/ray_trainer.py`
- `bash -n examples/ttrl/run_sps_weighted_gate_math_qwen3_4b_50step.sh` 通过。
- Hydra 展开确认：
  - `trainer.total_training_steps=50`
  - `trainer.val_before_train=false`
  - `trainer.test_freq=50`
  - `ttrl.n_votes_per_prompt=64`
  - `ttrl.n_samples_per_prompt=32`
  - `ttrl.sps_reward_mode=answer_weighted_gate`
  - `ttrl.sps_gate_confidence_threshold=0.8`
  - `ttrl.sps_gate_majority_ratio_threshold=0.75`

计划启动命令：
```bash
cd /opt/tiger/TTRL/verl
PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 bash examples/ttrl/run_sps_weighted_gate_math_qwen3_4b_50step.sh \
  > /opt/tiger/TTRL/verl/sps_weighted_gate_50step.log 2>&1 &
```

启动记录：
- 2026-06-25 01:58 CST 已改用长会话前台方式启动，日志仍写入：
  `/opt/tiger/TTRL/verl/sps_weighted_gate_50step.log`
- Ray session：
  `/tmp/ray_sps_weighted_gate/ray/session_latest`
- GPU：0-3。
- 4-7 上 MajVote 仍在运行，暂未使用。

step 1：
- `train/sps/reward_mode=3.000`（`answer_weighted_gate` 生效）
- `train/sps/effective_K=31.949`
- `train/sps/weighted_label_confidence=0.863`
- `train/sps/override_rate=0.000`
- `train/sps/agreement_rate=0.875`
- `train/label_accuracy=0.875`
- `train/pass@32=0.875`
- `train/majority_ratio=0.473`
- `actor/entropy=0.267`
- `response_length/clip_ratio=0.609`
- `timing_s/step=55.960`
- 观察：首批 SPS 与 majority 多数一致，gate 没有触发覆盖；这符合“majority-first，SPS 只在高置信冲突时纠偏”的设计。`effective_K` 比 v1 的 63.9 低，需继续观察是否持续。

2026-06-25 02:13 CST 进度：v2 已到 `step 13/50`，仍未出现中途 validation。
- step 2-13 摘要：
  - step 2：`override_rate=0.000`，`agreement_rate=0.875`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.605`，`clip_ratio=0.496`。
  - step 3：`override_rate=0.000`，`agreement_rate=0.875`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.566`，`clip_ratio=0.582`。
  - step 4：`override_rate=0.000`，`agreement_rate=0.625`，`weighted_label_confidence=0.625`，`label_accuracy=0.625`，`pass@32=0.625`，`majority_ratio=0.340`，`clip_ratio=0.738`。
  - step 5：`override_rate=0.000`，`agreement_rate=0.750`，`weighted_label_confidence=0.699`，`label_accuracy=0.625`，`pass@32=0.750`，`majority_ratio=0.305`，`clip_ratio=0.750`。
  - step 6：`override_rate=0.000`，`agreement_rate=0.875`，`weighted_label_confidence=0.847`，`label_accuracy=0.500`，`pass@32=0.750`，`majority_ratio=0.520`，`clip_ratio=0.684`。
  - step 7：`override_rate=0.000`，`agreement_rate=0.750`，`weighted_label_confidence=0.750`，`label_accuracy=0.500`，`pass@32=0.750`，`majority_ratio=0.590`，`clip_ratio=0.578`。
  - step 8：`override_rate=0.000`，`agreement_rate=0.625`，`weighted_label_confidence=0.625`，`label_accuracy=0.625`，`pass@32=0.625`，`majority_ratio=0.574`，`clip_ratio=0.512`。
  - step 9：`override_rate=0.000`，`agreement_rate=0.875`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.684`，`clip_ratio=0.473`。
  - step 10：`override_rate=0.000`，`agreement_rate=1.000`，`weighted_label_confidence=1.000`，`label_accuracy=1.000`，`pass@32=1.000`，`majority_ratio=0.770`，`clip_ratio=0.547`。
  - step 11：`override_rate=0.000`，`agreement_rate=0.250`，`weighted_label_confidence=0.250`，`label_accuracy=0.250`，`pass@32=0.250`，`majority_ratio=0.246`，`clip_ratio=0.773`。
  - step 12：`override_rate=0.000`，`agreement_rate=0.750`，`weighted_label_confidence=0.750`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.508`，`clip_ratio=0.504`。
  - step 13：`override_rate=0.000`，`agreement_rate=0.875`，`weighted_label_confidence=0.875`，`label_accuracy=0.750`，`pass@32=0.875`，`majority_ratio=0.602`，`clip_ratio=0.551`。
- 中间判断：当前阈值下 SPS 尚未覆盖 majority，v2 实际接近 majority-first 训练；安全性较好，但纠偏力度偏弱。继续跑完 50 step final validation，再决定是否需要更激进的 gate 阈值或 clip-aware 设置。

2026-06-25 02:24 CST 进度：v2 已到 `step 26/50`，仍未出现中途 validation。
- step 14-26 摘要：
  - step 14：`override_rate=0.000`，`agreement_rate=1.000`，`weighted_label_confidence=0.990`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.723`，`clip_ratio=0.480`。
  - step 15：`override_rate=0.000`，`agreement_rate=0.875`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.754`，`clip_ratio=0.281`。
  - step 16：`override_rate=0.000`，`agreement_rate=0.625`，`weighted_label_confidence=0.625`，`label_accuracy=0.625`，`pass@32=0.625`，`majority_ratio=0.504`，`clip_ratio=0.500`。
  - step 17：`override_rate=0.000`，`agreement_rate=1.000`，`weighted_label_confidence=1.000`，`label_accuracy=1.000`，`pass@32=1.000`，`majority_ratio=0.848`，`clip_ratio=0.336`。
  - step 18：`override_rate=0.000`，`agreement_rate=1.000`，`weighted_label_confidence=0.964`，`label_accuracy=0.875`，`pass@32=1.000`，`majority_ratio=0.789`，`clip_ratio=0.375`。
  - step 19：`override_rate=0.000`，`agreement_rate=0.750`，`weighted_label_confidence=0.750`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.648`，`clip_ratio=0.504`。
  - step 20：`override_rate=0.000`，`agreement_rate=0.750`，`weighted_label_confidence=0.750`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.633`，`clip_ratio=0.477`。
  - step 21：`override_rate=0.000`，`agreement_rate=0.750`，`weighted_label_confidence=0.750`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.508`，`clip_ratio=0.625`。
  - step 22：`override_rate=0.000`，`agreement_rate=0.875`，`weighted_label_confidence=0.875`，`label_accuracy=0.750`，`pass@32=0.750`，`majority_ratio=0.699`，`clip_ratio=0.383`。
  - step 23：`override_rate=0.000`，`agreement_rate=0.875`，`weighted_label_confidence=0.875`，`label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.441`，`clip_ratio=0.797`。
  - step 24：`override_rate=0.000`，`agreement_rate=1.000`，`weighted_label_confidence=1.000`，`label_accuracy=1.000`，`pass@32=1.000`，`majority_ratio=0.918`，`clip_ratio=0.324`。
  - step 25：`override_rate=0.000`，`agreement_rate=1.000`，`weighted_label_confidence=1.000`，`label_accuracy=1.000`，`pass@32=1.000`，`majority_ratio=0.898`，`clip_ratio=0.191`。
  - step 26：`override_rate=0.000`，`agreement_rate=1.000`，`weighted_label_confidence=1.000`，`label_accuracy=0.875`，`pass@32=1.000`，`majority_ratio=0.863`，`clip_ratio=0.312`。
- 中间判断：step 24-26 质量较高，但 gate 仍没有任何覆盖；如果 final 不达标，下一轮需要降低阈值或直接设计“低 majority 时启用 SPS top-answer”的更激进版本。

2026-06-25 02:47 CST final：v2 完整结束，GPU 0-3 已释放，最终验证只在 step 50 触发。
- Final Math500 validation:
  - `val-core/MATH-TTT/acc/mean@4=0.7278672032193159`
  - `val-core/MATH-TTT/acc/best@4/mean=0.7892354124748491`
  - `val-core/MATH-TTT/acc/maj@4/mean=0.7293762575452716`
  - `val-aux/MATH-TTT/acc/worst@4/mean=0.6654929577464789`
  - `val-aux/MATH-TTT/format_score/mean@4=0.7364185110663984`
- 结论：
  - 低于 v1 `mean@4=0.738430583501006`，不满足“有提升则 commit”的条件，暂不提交。
  - 低于目标 `0.85`，goal 仍未完成。
  - 诊断发现 v2 实现存在 K 选择问题：`answer_weighted_gate` 没有被纳入 `K=n_votes_per_prompt` 分支，实际只采了 `K=32`，因此 `effective_K≈31.9`，而非 v1 的 `≈63.9`；这也解释了 gate 实验覆盖力度不足。

修复：
- `verl/trainer/ppo/ray_trainer.py`
  - 将 `K=n_votes_per_prompt` 条件从只识别 `answer_weighted_vote` 改成识别 `answer_weighted_vote` 和 `answer_weighted_gate`。
- 验证：
  - `py_compile` 通过。
  - `bash -n examples/ttrl/run_sps_weighted_gate_math_qwen3_4b_50step.sh` 通过。

下一步：重跑 v2b（同一 gate 逻辑，但正确使用 `N_VOTES=64`），仍然使用 GPU 0-3，final-only validation。

### 9.5 实验 v2b：修复 K 后重跑 SPS-gated majority fallback

启动记录：
- 2026-06-25 02:48 CST 启动，日志：
  `/opt/tiger/TTRL/verl/sps_weighted_gate64_50step.log`
- Ray session：
  `/tmp/ray_sps_weighted_gate64/ray/session_latest`
- GPU：0-3。
- 仍为 final-only validation：`val_before_train=false`，`test_freq=50`，`total_training_steps=50`。

step 1：
- `train/sps/reward_mode=3.000`
- `train/sps/effective_K=63.884`，确认已恢复 64-vote 口径。
- `train/sps/weighted_label_confidence=0.750`
- `train/sps/override_rate=0.000`
- `train/sps/agreement_rate=0.750`
- `train/label_accuracy=0.750`
- `train/pass@32=0.750`
- `train/majority_ratio=0.535`
- `actor/entropy=0.263`
- `response_length/clip_ratio=0.605`
- `timing_s/step=73.083`
- 观察：K 修复生效；gate 首步没有覆盖，继续观察后续是否出现高置信冲突覆盖。

2026-06-25 03:21 CST 进度：v2b 已到 `step 25/50`，仍未出现中途 validation。
- step 2-25 摘要：
  - `effective_K` 持续约 `63.8-63.9`，确认 64-vote 口径稳定。
  - `override_rate` 截至 step 25 全部为 `0.000`。
  - 高置信批次通常 `agreement_rate=1.000`，说明 SPS answer label 大多只是在确认 majority，而非提供冲突纠偏。
  - 低共识/坏批次通常 SPS confidence 也低，例如 step 11：`weighted_label_confidence=0.375`，`label_accuracy=0.250`，`pass@32=0.250`，`clip_ratio=0.762`，gate 正确没有覆盖。
  - 好批次集中在 step 17/24/25：`label_accuracy=1.000`，`pass@32=1.000`，`majority_ratio=0.889/0.945/0.904`。
- 中间判断：当前 gate 作为安全策略有效，但基本没有引入新训练信号；final 结果若不达标，下一轮应考虑把 SPS 用作 prompt 过滤/样本权重，而不是伪标签覆盖。

2026-06-25 03:51 CST final：v2b 完整结束，GPU 0-3 已释放，最终验证只在 step 50 触发。
- Final Math500 validation:
  - `val-core/MATH-TTT/acc/mean@4=0.7489939637826962`
  - `val-core/MATH-TTT/acc/best@4/mean=0.8069496981891349`
  - `val-core/MATH-TTT/acc/maj@4/mean=0.7469597585513079`
  - `val-aux/MATH-TTT/acc/worst@4/mean=0.6948551307847082`
  - `val-aux/MATH-TTT/format_score/mean@4=0.7565392354124748`
- 对比：
  - 高于 v1 `mean@4=0.738430583501006`，绝对提升约 `+1.06pp`，满足“有提升的改动要本地 commit 记录”。
  - 仍低于目标 `mean@4>=0.85`，goal 不能完成。
  - 低于同机 MajVote 长任务的 step50 `mean@4=0.813`；该 MajVote 任务最终 step310 达到 `mean@4=0.8832997987927566`，说明训练链路和模型容量足够，但当前 50-step SPS 方案的信号效率不足。
- 诊断：
  - v2b 的 `override_rate` 全程为 0，说明 gate 过于保守，实际退化为 majority fallback；SPS 只提供确认信号，没有产生纠偏。
  - 50-step SPS 脚本设置 `trainer.total_training_steps=50` 且 actor 使用 `warmup_style=cosine`、`lr_warmup_steps_ratio=0.03`，优化器学习率会在 50 step 内衰减完；而 MajVote 长任务的 step50 是按 310 总步数 cosine schedule 训练，学习率仍较高。这可能解释了 SPS 50-step 方案低于 MajVote step50 的一部分差距。

下一步候选：
1. 保留 50-step final-only 约束，改短跑学习率策略为 constant 或等效的较长 horizon cosine，避免 final 前学习率过早衰减。
2. 保留 majority pseudo-label 主干，把 SPS 作为 prompt/rollout 的内部置信权重或过滤信号，而不是只做伪标签 override；目标是减少低共识、高截断批次的负更新，同时不破坏 MajVote 的高效学习动态。
3. 现在 GPU 0-7 均已空闲，下一轮可用 8 卡跑 50-step final-only 实验。

### 9.6 实验 v3：SPS confidence-filtered majority，8 卡，constant LR

启动前设计：
- 主干仍使用 majority vote pseudo-label，避免 v1/v2 里 SPS pseudo-label 覆盖带来的不稳定。
- SPS 不再负责替换标签，而是作为内部置信过滤信号：
  - 若 `majority_ratio >= sps_filter_majority_ratio_threshold`，训练该 prompt。
  - 或者 SPS answer label 与 majority label 一致，且 `weighted_label_confidence >= sps_filter_confidence_threshold`，训练该 prompt。
  - 否则把该 prompt 的 reward 置零，避免低共识/高截断批次产生负更新。
- 训练反馈仍是无监督内部信号：pseudo-label、SPS confidence、majority ratio 都来自模型 rollouts 和 base/proposal logprob；真实 Math500 answer 只用于日志诊断和最终 validation。
- 修正短跑 LR：actor `warmup_style=constant`、`lr_warmup_steps_ratio=0.0`，避免 50 step 内 cosine 衰减到 0。
- 使用 GPU 0-7，因为 2026-06-25 03:52 CST 检查 GPU 0-7 均已空闲。

代码改动：
- `verl/trainer/ppo/ttrl_utils.py`
  - `apply_sps_weighted_ttrl_gt(...)` 新增 confidence filter 参数：
    - `confidence_filter`
    - `filter_confidence_threshold`
    - `filter_majority_ratio_threshold`
  - 新增 `sps_train_weight_list`，按 prompt 记录是否参与训练。
- `verl/trainer/ppo/ray_trainer.py`
  - 新增 `ttrl.sps_reward_mode=answer_conf_filter`。
  - 对该模式，仍走 64 vote 生成和 majority pseudo-label；在 reward 后按 `sps_train_weight_list` 把低置信 prompt reward 置零。
  - 新增日志 `train/sps/train_weight`。
- `verl/trainer/config/ppo_trainer_ttrl.yaml`
  - 新增模式说明和 filter 阈值。
- `examples/ttrl/run_sps_conf_filter_math_qwen3_4b_50step_8gpu.sh`
  - 新 8 卡 v3 脚本。
  - `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
  - `RAY_TMPDIR=/tmp/ray_sps_conf_filter8`
  - `trainer.val_before_train=False`
  - `trainer.test_freq=50`
  - `trainer.total_training_steps=50`
  - `trainer.n_gpus_per_node=8`

启动与验证：
- 2026-06-25 03:58 CST 启动，日志：
  `/opt/tiger/TTRL/verl/sps_conf_filter8_50step.log`
- Ray session：
  `/tmp/ray_sps_conf_filter8/ray/session_latest`
- 启动前静态检查：
  - `py_compile` 通过：
    `python -m py_compile verl/verl/trainer/ppo/ttrl_utils.py verl/verl/trainer/ppo/ray_trainer.py`
  - `bash -n examples/ttrl/run_sps_conf_filter_math_qwen3_4b_50step_8gpu.sh` 通过。
  - Hydra 展开确认 `sps_reward_mode=answer_conf_filter`，`total_training_steps=50`，
    `val_before_train=False`，`test_freq=50`，`warmup_style=constant`，
    `trainer.n_gpus_per_node=8`。

2026-06-25 04:45 CST final：v3 完整结束，最终验证只在 step 50 触发。
- 训练诊断：
  - `train/sps/reward_mode=4.000`
  - `train/sps/effective_K` 全程基本维持在 `63.8-64.0`，确认 64-vote 路径生效。
  - early steps 的 `train_weight` 有低谷，例如 step 4 为 `0.375`，符合 confidence filter 预期。
  - 后半段多数 batch `train_weight=0.875-1.000`，step 50 为 `0.875`。
  - step 50：`weighted_label_confidence=0.936`，`agreement_rate=1.000`，
    `label_accuracy=0.875`，`pass@32=0.875`，`majority_ratio=0.863`，
    `response_length/clip_ratio=0.203`。
- Final Math500 validation:
  - `val-core/MATH-TTT/acc/mean@4=0.81841046277666`
  - `val-core/MATH-TTT/acc/best@4/mean=0.8619939637826962`
  - `val-core/MATH-TTT/acc/maj@4/mean=0.8218933601609658`
  - `val-aux/MATH-TTT/acc/worst@4/mean=0.7694144869215291`
  - `val-aux/MATH-TTT/format_score/mean@4=0.8340040241448692`
- 对比：
  - 高于 v2b `mean@4=0.7489939637826962`，绝对提升约 `+6.94pp`，满足“有提升的改动要本地 commit 记录”。
  - 也高于同机 MajVote 长任务的 step50 `mean@4=0.813`，说明 confidence filter + constant LR 对 50-step 短跑有效。
  - 仍低于目标 `mean@4>=0.85`，goal 不能完成。
- 诊断：
  - `best@4=0.86199` 已超过 0.85，说明候选答案容量已经够；当前差距主要在 aggregation/训练后采样分布上，`mean@4=0.8184` 和 `maj@4=0.8219` 仍低于目标。
  - v3 的主要有效增益来自 constant LR 与低置信 prompt 过滤；SPS answer override 仍没有提供纠偏，因为该版本只把 SPS 用作过滤信号。
  - 下一轮应保留 constant LR 与 8 卡配置，优先尝试提高训练信号覆盖和输出分布集中度，而不是只提高候选多样性。

下一步候选：
1. 在 v3 基础上降低过滤阈值或改成连续样本权重，让 `weighted_label_confidence` 以软权重进入 reward，避免 hard filter 丢掉中置信但正确的样本。
2. 加入内部长度/截断惩罚，只依赖 rollout 长度与格式信号，抑制 clip-heavy 更新；v3 final 的 `format_score/mean@4=0.834` 高于 acc，但仍有截断波动。
3. 参考 `best@4` 已达标的事实，尝试 answer-level rank/self-consistency 蒸馏，让训练奖励更偏向多数票中的高置信短答案，提高 `mean@4` 而不是只提高 `best@4`。
