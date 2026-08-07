# Full-Rollout TTRL 12h 实验进展记录

## 固定实验语义

- 模型：`/models/Qwen2.5-Math-7B`
- 数据：`/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT`
- 环境：`/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200`
- 机器：8x B200
- 训练粒度：完整 rollout，不做 chunk / suffix / mid-state
- batch：32 prompts/step
- rollout：32 completions/prompt，总计 1024 complete responses/step
- validation：`val_n=16`，每个 20-step pilot 只做 final val
- dynamic batch：关闭，`trainer.balance_batch=False`
- 正式方法不使用 GT；GT 仅用于 validation 和 oracle diagnostic

## 2026-08-02 FR-B0

### 配置

- run id：`ttrl_full_rollout_powerflow_b0_posterior_alpha4_b32_r32_20step_20260802`
- launcher：`run_records/run_front_full_rollout_powerflow_b0_posterior_alpha4_b32_r32_20step_20260802.sh`
- raw log：`important_experiment_logs/ttrl_full_rollout_powerflow_b0_posterior_alpha4_b32_r32_20step_20260802.log`
- target mode：`posterior_sharpen`
- alpha：4.0
- eps：0.0
- weight clip：4.0，clip 后 renorm
- pollution guard：开启
- quality gate：关闭

### 结果

20-step final validation：

- mean@16：0.45325
- maj@16：0.57668
- best@16：0.83677

这组结果明显低于当前需要对齐/超过的 MV/PF 20-step 曲线，不适合作为主线继续扩展。

### 性能

训练总耗时约 40 分 9 秒，包含 final validation。

稳态 step 分解大致为：

- `timing_s/gen`：15-17s
- `timing_s/reward`：3-4s，偶有 timeout 噪声
- `timing_s/ref`：20-22s
- `timing_s/update_actor`：57-62s
- 普通训练 step：约 99-107s
- final validation：约 296s

当前慢点不是 rollout generation，而是完整 1024 条长 response 进入 actor update 和 ref logprob。

### Target Health

典型范围：

- support coverage：1.0
- OOV ratio：0.0
- valid answer coverage：约 0.51-0.69
- top1 mass：约 0.41-0.54
- margin：约 0.29-0.45
- nonzero ratio：约 0.51-0.69
- clean rollout ratio：约 0.66-0.82
- assistant/human marker pollution：约 0.09-0.19

结论：target 可以稳定构造，但单纯 answer posterior sharpening 把完整 response 分布推向了较差方向，至少在 20 step 上没有形成有效学习。

## 下一步

优先跑 FR-D0 oracle correctness diagnostic。

目的不是作为正式方法，而是检查 full-response PowerFlow actor update 是否有足够学习能力：

- 如果 oracle correctness 20-step 也不能明显提升 mean/maj，说明当前 full-response PowerFlow loss 或 actor update 链路有问题。
- 如果 oracle correctness 能明显提升，说明 B0 的主要问题是无 GT label estimation / target construction，而不是训练链路本身。

FR-D0 后再按结果选择：

- oracle 有效：继续跑 B2 margin-aware、B3 top-k representative、C0 quality-gated。
- oracle 无效：先回到 actor loss / weighting / response mask / PowerFlow objective 对齐排查。

## 2026-08-02 FR-D0

### 配置

- run id：`ttrl_full_rollout_powerflow_d0_oracle_alpha4_b32_r32_20step_20260802`
- launcher：`run_records/run_front_full_rollout_powerflow_d0_oracle_alpha4_b32_r32_20step_20260802.sh`
- raw log：`important_experiment_logs/ttrl_full_rollout_powerflow_d0_oracle_alpha4_b32_r32_20step_20260802.log`
- target mode：`oracle_correctness`
- alpha：4.0
- eps：0.0
- weight clip：4.0，clip 后 renorm
- pollution guard：开启
- quality gate：关闭
- actor objective：PowerFlow squared-delta，sequence-level full-response weight

### 结果

20-step final validation：

- mean@16：0.398875
- maj@16：0.503472
- best@16：0.814348

这组结果比 FR-B0 更差，否定了“只要 full-rollout target 用 oracle correctness，当前 PowerFlow actor update 就能学好”的假设。

### 性能与健康指标

final validation step：

- `timing_s/testing`：300.032s
- `timing_s/step`：405.820s
- `timing_s/update_actor`：59.734s
- `timing_s/ref`：21.139s
- `timing_s/gen`：15.982s
- `timing_s/reward`：5.428s
- `perf/throughput`：372.240

target / rollout health：

- `full_rollout_target/clean_rollout_ratio`：0.740
- `full_rollout_target/repeated_boxed_ratio`：0.070
- `full_rollout_target/empty_boxed_ratio`：0.086
- `full_rollout_target/assistant_marker_ratio`：0.141
- `full_rollout_target/human_marker_ratio`：0.131
- `full_rollout_target/nonzero_ratio`：0.628
- `full_rollout_target/observed_reward_mean`：0.339
- `actor/powerflow_weight/nonzero_ratio`：0.719
- `response_length/mean`：1084.771
- `response_length/clip_ratio`：0.091

### 诊断结论

当前失败更像 actor objective / full-response mask 的问题，而不是单纯 target construction 没有 GT 的问题。

`compute_powerflow` 当前对整条 response 做 sequence-level weighted squared-delta：

- `avg_log_prob = masked_mean(log_prob, combined_mask, axis=1)`
- `avg_ref_log_prob = masked_mean(ref_log_prob, combined_mask, axis=1)`
- `delta = log_z + avg_log_prob - beta * avg_ref_log_prob`
- `loss = weight * delta^2`

这不是 weighted imitation。一个高权重正确答案会把整条 response 的格式污染、重复 boxed、assistant/human marker、尾部乱码一起纳入平均 logprob/ref_logprob 目标，容易放大坏 token 形态。

下一步先做 FR-D0-NLL：保持同一套 oracle full-rollout weights，但关闭 `powerflow_enable`，打开 `chunk_weighted_nll_enable`，跑 20 step final val。

- 如果 weighted NLL 也失败：主问题在样本选择、污染过滤、完整 response target 质量。
- 如果 weighted NLL 明显改善：主问题在当前 PowerFlow squared-delta objective，不应继续把它作为 full-response 主 objective。

## 2026-08-02 FR-D0-NLL

### 配置

- run id：`ttrl_full_rollout_powerflow_d0_oracle_nll_b32_r32_20step_20260802`
- launcher：`run_records/run_front_full_rollout_powerflow_d0_oracle_nll_b32_r32_20step_20260802.sh`
- raw log：`important_experiment_logs/ttrl_full_rollout_powerflow_d0_oracle_nll_b32_r32_20step_20260802.log`
- target mode：`oracle_correctness`
- alpha：4.0
- eps：0.0
- weight clip：4.0，clip 后 renorm
- pollution guard：开启
- quality gate：关闭
- actor objective：weighted NLL，使用同一套 full-response sequence-level weights

### 结果

20-step final validation：

- mean@16：0.69275
- maj@16：0.797464
- best@16：0.904338

对比：

- FR-B0 posterior + PowerFlow squared-delta：mean@16 0.45325 / maj@16 0.57668 / best@16 0.83677
- FR-D0 oracle + PowerFlow squared-delta：mean@16 0.398875 / maj@16 0.503472 / best@16 0.814348
- FR-D0-NLL oracle + weighted NLL：mean@16 0.69275 / maj@16 0.797464 / best@16 0.904338

这说明 full-rollout target 的传递链路和完整 response 加权训练本身可以工作；当前 full-response PowerFlow squared-delta objective 是主要问题。后续正式方法不要继续用这版 squared-delta 做 full-response actor objective。

### 性能

稳态 step：

- 普通训练 step：约 73-80s，首步约 92s
- `timing_s/gen`：约 15-16s
- `timing_s/reward`：约 3-5s
- `timing_s/update_actor`：约 50-57s
- `perf/throughput`：约 1450-1800 tokens/s

final validation step：

- `timing_s/testing`：280.183s
- `timing_s/step`：357.727s
- `timing_s/update_actor`：55.397s
- `timing_s/gen`：15.081s
- `timing_s/reward`：3.464s
- `perf/throughput`：355.953

训练总 wall time 约 30 分 26 秒，包含 final validation。

### Target Health

final step：

- `full_rollout_target/observed_reward_mean`：0.519
- `full_rollout_target/clean_rollout_ratio`：0.933
- `full_rollout_target/repeated_boxed_ratio`：0.023
- `full_rollout_target/empty_boxed_ratio`：0.015
- `full_rollout_target/assistant_marker_ratio`：0.022
- `full_rollout_target/human_marker_ratio`：0.020
- `full_rollout_target/nonzero_ratio`：0.868
- `actor/chunk_weight_mean`：0.921
- `response_length/mean`：899.390
- `response_length/clip_ratio`：0.063

### 诊断结论

这组是 oracle diagnostic，不是正式无 GT 方法结果。但它给出清晰工程结论：

- full-rollout weights 必须和 actor objective 解耦，`full_rollout_powerflow_enable=True` 时应始终构造 weights，不能依赖 `actor.powerflow_enable=True`。
- weighted NLL 是当前可用的 full-response actor update 诊断/替代 objective。
- 当前 full-response PowerFlow squared-delta 会严重破坏训练，下一步先用正式无 GT target + weighted NLL 验证是否能恢复 B0。

下一步跑 FR-B0-NLL：`posterior_sharpen alpha=4` + weighted NLL，20-step final val。

## 2026-08-02 FR-B0-NLL

### 配置

- run id：`ttrl_full_rollout_powerflow_b0_posterior_nll_b32_r32_20step_20260802`
- launcher：`run_records/run_front_full_rollout_powerflow_b0_posterior_nll_b32_r32_20step_20260802.sh`
- raw log：`important_experiment_logs/ttrl_full_rollout_powerflow_b0_posterior_nll_b32_r32_20step_20260802.log`
- target mode：`posterior_sharpen`
- alpha：4.0
- eps：0.0
- weight clip：4.0，clip 后 renorm
- pollution guard：开启
- actor objective：weighted NLL，关闭 full-response PowerFlow squared-delta
- train batch / rollout：32 prompts x 32 samples
- validation：final only，`val_n=16`

### 结果

20-step final validation：

- mean@16：0.688000
- maj@16：0.793828
- best@16：0.908718

对比：

- FR-B0 posterior + PowerFlow squared-delta：mean@16 0.45325 / maj@16 0.57668 / best@16 0.83677
- FR-D0 oracle + weighted NLL：mean@16 0.69275 / maj@16 0.797464 / best@16 0.904338
- FR-B0 posterior + weighted NLL：mean@16 0.68800 / maj@16 0.793828 / best@16 0.908718

这说明正式无 GT posterior target + weighted NLL 能恢复到 oracle-NLL 附近，远好于 squared-delta，但 20-step mean@16 仍然低于历史 MV / PowerFlow 目标曲线，不足以作为主方案。

### 性能

普通训练 step，统计 step 1-19：

- 平均 `timing_s/step`：77.643s，min 73.372s，max 92.097s
- 平均 `timing_s/update_actor`：54.402s，min 52.129s，max 58.247s
- 平均 `timing_s/gen`：16.195s
- 平均 `timing_s/reward`：3.504s
- 平均 `perf/throughput`：1568.378 tokens/s

final validation step：

- `timing_s/testing`：277.983s
- `timing_s/step`：351.903s
- `timing_s/update_actor`：52.505s
- `timing_s/gen`：14.955s
- `timing_s/reward`：3.519s
- `perf/throughput`：343.617 tokens/s

训练总 wall time 约 30 分 28 秒，包含 final validation。

### Target Health

final step：

- `full_rollout_target/support_coverage`：1.000
- `full_rollout_target/oov_ratio`：0.000
- `full_rollout_target/top1_mass`：0.599
- `full_rollout_target/top2_mass`：0.079
- `full_rollout_target/margin`：0.520
- `full_rollout_target/target_entropy`：3.176
- `full_rollout_target/valid_answer_coverage`：0.872
- `full_rollout_target/nonzero_ratio`：0.872
- `full_rollout_target/weight_mean`：0.872
- `full_rollout_target/weight_max`：2.096
- `full_rollout_target/observed_reward_mean`：0.536
- `full_rollout_target/clean_rollout_ratio`：0.941
- `full_rollout_target/repeated_boxed_ratio`：0.017
- `full_rollout_target/empty_boxed_ratio`：0.010
- `full_rollout_target/assistant_marker_ratio`：0.020
- `full_rollout_target/human_marker_ratio`：0.016
- `response_length/mean`：849.278
- `response_length/clip_ratio`：0.044

### 诊断结论

B0-NLL 的 target health 很干净，说明污染过滤和 full-rollout answer posterior 传递链路可用。当前主要问题不是崩坏，而是 B0 的 posterior sharpening 太平均、没有区分 prompt-level label confidence，导致有效学习信号偏弱。

下一步跑 FR-B2-NLL：`margin_aware alpha=6 tau=0.25 floor=0.2` + weighted NLL，仍然是完整 rollout、无 GT，目标是降低低置信 prompt 的噪声更新。

## 2026-08-02 FR-B2-NLL

### 配置

- run id：`ttrl_full_rollout_powerflow_b2_margin_nll_b32_r32_20step_20260802`
- launcher：`run_records/run_front_full_rollout_powerflow_b2_margin_nll_b32_r32_20step_20260802.sh`
- raw log：`important_experiment_logs/ttrl_full_rollout_powerflow_b2_margin_nll_b32_r32_20step_20260802.log`
- target mode：`margin_aware`
- alpha：6.0
- margin tau：0.25
- prompt weight floor：0.2
- weight clip：4.0，clip 后 renorm
- pollution guard：开启
- actor objective：weighted NLL，关闭 full-response PowerFlow squared-delta
- train batch / rollout：32 prompts x 32 samples
- validation：final only，`val_n=16`

### 结果

20-step final validation：

- mean@16：0.692875
- maj@16：0.794484
- best@16：0.899186

对比：

- FR-B0 posterior + weighted NLL：mean@16 0.688000 / maj@16 0.793828 / best@16 0.908718
- FR-B2 margin-aware + weighted NLL：mean@16 0.692875 / maj@16 0.794484 / best@16 0.899186

B2 对 mean@16 有轻微改善，但 best@16 明显下降，说明 prompt-level margin weighting 只是弱增益，不是突破方向。

### 性能

普通训练 step，统计 step 1-19：

- 平均 `timing_s/step`：77.900s，min 72.474s，max 92.154s
- 平均 `timing_s/update_actor`：54.428s，min 50.879s，max 58.434s
- 平均 `timing_s/gen`：16.219s
- 平均 `timing_s/reward`：3.517s
- 平均 `perf/throughput`：1581.975 tokens/s
- 平均 `full_rollout_target/prompt_weight_mean`：0.858，min 0.756，max 0.933

final validation step：

- `timing_s/testing`：276.518s
- `timing_s/step`：352.317s
- `timing_s/update_actor`：54.208s
- `timing_s/gen`：15.372s
- `timing_s/reward`：3.366s
- `perf/throughput`：362.698 tokens/s

训练总 wall time 约 30 分 33 秒，包含 final validation。

### Target Health

final step：

- `full_rollout_target/support_coverage`：1.000
- `full_rollout_target/oov_ratio`：0.000
- `full_rollout_target/top1_mass`：0.604
- `full_rollout_target/top2_mass`：0.080
- `full_rollout_target/margin`：0.524
- `full_rollout_target/target_entropy`：3.099
- `full_rollout_target/valid_answer_coverage`：0.852
- `full_rollout_target/prompt_weight_mean`：0.826
- `full_rollout_target/nonzero_ratio`：0.852
- `full_rollout_target/weight_mean`：0.852
- `full_rollout_target/weight_max`：2.500
- `full_rollout_target/observed_reward_mean`：0.531
- `full_rollout_target/clean_rollout_ratio`：0.930
- `full_rollout_target/repeated_boxed_ratio`：0.020
- `full_rollout_target/empty_boxed_ratio`：0.011
- `full_rollout_target/assistant_marker_ratio`：0.029
- `full_rollout_target/human_marker_ratio`：0.025
- `response_length/mean`：902.912
- `response_length/clip_ratio`：0.069

### 诊断结论

B2 的 target health 正常，训练也稳定，但效果基本贴着 B0-NLL。下一步跑 FR-B3-NLL：`topk_representative` + weighted NLL，验证重复答案/重复轨迹支配 target 是否是限制 mean/best 的主因。

注意：启动 B3 前修正了 `topk_representative` 内部 representative 排序，去掉按 `sequence_reward` 排序的 GT 泄漏；现在同 answer cluster 内按 response length 和 index 做 deterministic selection。

## 2026-08-02 FR-B3-NLL

### 配置

- run id：`ttrl_full_rollout_powerflow_b3_topkrep_nll_b32_r32_20step_20260802`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_b3_topkrep_nll_b32_r32_20step_20260802.log`
- target mode：`topk_representative`
- `top_k_answers=2`
- `representatives_per_answer=2`
- `alpha=4`
- actor objective：weighted NLL
- train batch / rollout：32 prompts x 32 samples
- validation：final only，`val_n=16`

### 结果

20-step final validation：

- mean@16：0.622625
- maj@16：0.754308
- best@16：0.903838

对比：

- FR-B0 posterior + weighted NLL：mean@16 0.688000 / maj@16 0.793828 / best@16 0.908718
- FR-B2 margin-aware + weighted NLL：mean@16 0.692875 / maj@16 0.794484 / best@16 0.899186
- FR-B3 top-k representative + weighted NLL：mean@16 0.622625 / maj@16 0.754308 / best@16 0.903838

B3 明显拉低 mean@16 和 maj@16，说明当前 top-k representative 过稀疏，不能解决 full-rollout posterior target 的瓶颈。

### 性能

普通训练 step，统计 step 1-19：

- 平均 `timing_s/step`：79.211s，min 74.791s，max 91.305s
- 平均 `timing_s/update_actor`：55.263s，min 53.344s，max 57.959s

final validation step：

- `timing_s/testing`：288.478s
- `timing_s/step`：365.486s
- `timing_s/update_actor`：55.016s

训练总 wall time 约 31 分 11 秒，包含 final validation。

### Target Health

20 个 step 均值：

- `full_rollout_target/support_coverage`：0.165
- `full_rollout_target/oov_ratio`：0.835
- `full_rollout_target/nonzero_ratio`：0.108
- `full_rollout_target/representative_ratio`：0.108
- `full_rollout_target/observed_reward_mean`：0.420

final step：

- `full_rollout_target/support_coverage`：0.134
- `full_rollout_target/oov_ratio`：0.866
- `full_rollout_target/nonzero_ratio`：0.105
- `full_rollout_target/representative_ratio`：0.105
- `full_rollout_target/weight_max`：1.184
- `full_rollout_target/observed_reward_mean`：0.458

### 诊断结论

B3 的训练耗时没有显著恶化，但 target 只覆盖约 10% rollout，导致训练信号从 full-rollout distribution matching 退化为少量代表样本蒸馏。这个结果否定了“duplicate answer domination 是主要问题，少量 representative 就能改善”的假设。

下一步已启动 FR-B4-NLL：`self_consistency_advantage beta=4` + weighted NLL，保持完整 rollout、无 GT、batch32 / rollout32 / 20-step final val。B4 用稠密权重验证 relative posterior advantage 是否比 B0/B2 更有效。

## 2026-08-02 FR-B4-NLL

### 配置

- run id：`ttrl_full_rollout_powerflow_b4_selfcons_nll_b32_r32_20step_20260802`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_b4_selfcons_nll_b32_r32_20step_20260802.log`
- target mode：`self_consistency_advantage`
- `alpha=4`
- `beta=4`
- `slack=0.02`
- actor objective：weighted NLL
- train batch / rollout：32 prompts x 32 samples
- validation：final only，`val_n=16`

### 结果

20-step final validation：

- mean@16：0.683000
- maj@16：0.795648
- best@16：0.905824

对比：

- FR-B0 posterior + weighted NLL：mean@16 0.688000 / maj@16 0.793828 / best@16 0.908718
- FR-B2 margin-aware + weighted NLL：mean@16 0.692875 / maj@16 0.794484 / best@16 0.899186
- FR-B4 self-consistency advantage + weighted NLL：mean@16 0.683000 / maj@16 0.795648 / best@16 0.905824

B4 的 maj@16 略高，但 mean@16 低于 B0/B2，best@16 也没有超过 B0。说明稠密 advantage 权重本身没有解决当前 full-rollout target 质量问题。

### 性能

普通训练 step，统计 step 1-19：

- 平均 `timing_s/step`：77.816s，min 72.428s，max 91.583s
- 平均 `timing_s/update_actor`：54.651s，min 50.716s，max 57.941s

final validation step：

- `timing_s/testing`：282.410s
- `timing_s/step`：358.903s
- `timing_s/update_actor`：54.684s

训练总 wall time 约 30 分 38 秒，包含 final validation。

### Target Health

20 个 step 均值：

- `full_rollout_target/support_coverage`：1.000
- `full_rollout_target/oov_ratio`：0.000
- `full_rollout_target/nonzero_ratio`：0.794
- `full_rollout_target/observed_reward_mean`：0.477

final step：

- `full_rollout_target/support_coverage`：1.000
- `full_rollout_target/oov_ratio`：0.000
- `full_rollout_target/nonzero_ratio`：0.840
- `full_rollout_target/weight_max`：2.812
- `full_rollout_target/observed_reward_mean`：0.521

### 诊断结论

B4 的 target health 很干净，说明不是 OOV、污染过滤或稀疏覆盖导致失败；问题更像是当前无 GT answer posterior 的 target 排序不够强，稠密 reweighting 只是在同一噪声信号上换形状。

下一步跑 FR-B1-NLL：`posterior_gain` + weighted NLL，保持完整 rollout、无 GT、batch32 / rollout32 / 20-step final val。B1 用相对均匀 baseline 的 posterior gain 来弱化低信息 answer cluster，验证“相对增益”是否比 raw posterior / margin / self-consistency advantage 更有效。

## 2026-08-02 FR-B1-NLL

### 启动信息

- run id：`ttrl_full_rollout_powerflow_b1_postgain_nll_b32_r32_20step_20260802`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_b1_postgain_nll_b32_r32_20step_20260802.log`
- launcher：`/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/run_front_full_rollout_powerflow_b1_postgain_nll_b32_r32_20step_20260802.sh`
- target mode：`posterior_gain`
- `alpha=6`
- `slack=0.02`
- actor objective：weighted NLL
- train batch / rollout：32 prompts x 32 samples
- validation：final only，`val_n=16`
- output dir：`/tmp/ttrl_b200/checkpoints/ttrl_full_rollout_powerflow_b1_postgain_nll_b32_r32_20step_20260802`

### 启动确认

- 使用模型：`/models/Qwen2.5-Math-7B`
- 使用数据：`/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT`
- 使用环境：`/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python`
- `trainer.balance_batch=False`
- `ttrl.chunk_state_enable=False`
- `actor_rollout_ref.actor.powerflow_enable=False`
- `actor_rollout_ref.actor.chunk_weighted_nll_enable=True`
- vLLM config 中 `attention_config.backend=FLASH_ATTN`
- vLLM 已完成 FlashInfer autotune 和 CUDA graph capture
- NCCL 日志显示 `isAllDirectP2p 1`、`NCCL_NVLS_ENABLE=1`、`NVLS multicast support is available`

### 当前进展

- 21:03 左右完成 step 2/20。
- 进度条显示前 2 step 平均约 85.41s/it；首步含 Triton JIT，不作为稳态判断。
- reward 侧出现过一次 `Timeout during comparison`，属于规则打分的单样本比较超时，训练仍在继续。

### 结果

20-step final validation：

- mean@16：0.695000
- maj@16：0.801050
- best@16：0.904460

对比：

- FR-B0 posterior + weighted NLL：mean@16 0.688000 / maj@16 0.793828 / best@16 0.908718
- FR-B2 margin-aware + weighted NLL：mean@16 0.692875 / maj@16 0.794484 / best@16 0.899186
- FR-B4 self-consistency advantage + weighted NLL：mean@16 0.683000 / maj@16 0.795648 / best@16 0.905824
- FR-B1 posterior gain + weighted NLL：mean@16 0.695000 / maj@16 0.801050 / best@16 0.904460

B1 是目前 full-rollout no-GT weighted NLL 组里 mean@16 和 maj@16 最好的版本，但 best@16 低于 B0，整体仍未接近历史 MV / PowerFlow 目标曲线。

### 性能

普通训练 step，统计 step 1-19：

- 平均 `timing_s/step`：77.995s，min 73.811s，max 91.893s
- 平均 `timing_s/update_actor`：54.593s，min 52.308s，max 57.540s
- 平均 `perf/throughput`：1581.946 tokens/s，min 1425.076，max 1767.447

final validation step：

- `timing_s/testing`：281.067s
- `timing_s/step`：356.654s
- `timing_s/update_actor`：53.437s

训练总 wall time 约 30 分 39 秒，包含 final validation。

### Target Health

20 个 step 均值：

- `full_rollout_target/support_coverage`：1.000
- `full_rollout_target/oov_ratio`：0.000
- `full_rollout_target/nonzero_ratio`：0.802
- `full_rollout_target/observed_reward_mean`：0.489
- `full_rollout_target/clean_rollout_ratio`：0.883

final step：

- `full_rollout_target/support_coverage`：1.000
- `full_rollout_target/oov_ratio`：0.000
- `full_rollout_target/nonzero_ratio`：0.875
- `full_rollout_target/weight_max`：3.060
- `full_rollout_target/observed_reward_mean`：0.556
- `full_rollout_target/clean_rollout_ratio`：0.933

### 诊断结论

B1 相比 B0/B2/B4 有小幅提升，说明“相对 posterior gain”比 raw posterior / margin scalar / self-consistency advantage 更合理。但提升幅度很小，核心瓶颈仍是 full-rollout answer posterior 的 target 质量，而不是 target 覆盖率或污染过滤。

下一步跑 FR-C0-NLL：在 B1 posterior gain 基础上打开 prompt-level quality gate，先用保守阈值过滤低信息 prompt，验证低 coverage / 低 top mass / 低 margin prompt 是否在拖累无 GT target。

## 2026-08-02 FR-C0-NLL

### 实验设置

- run_id：`ttrl_full_rollout_powerflow_c0_qualitygate_nll_b32_r32_20step_20260802`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_c0_qualitygate_nll_b32_r32_20step_20260802.log`
- 训练：batch32 / rollout32 / 20 step / final-only validation / val_n=16
- target：B1 `posterior_gain` + weighted NLL
- quality gate：
  - `min_valid_answer_coverage=0.50`
  - `min_top_mass=0.35`
  - `min_margin=0.15`
- 不用 GT target；`powerflow_enable=False`，`chunk_weighted_nll_enable=True`

### 结果

- mean@16：0.682000
- maj@16：0.790000
- best@16：0.900000

### Timing

- 普通 step 数：19
- ordinary step avg：78.414s
- update_actor avg：55.019s
- throughput avg：1603.244 tokens/s
- step 20 含 final validation：351.942s
- final testing：276.577s

### Target Health

- prompt_weight_mean avg：0.716
- support_coverage avg：0.716
- oov_ratio avg：0.284
- nonzero_ratio avg：0.602
- observed_reward_mean avg：0.481
- step 20 target：prompt_weight_mean 0.719 / nonzero_ratio 0.622 / observed_reward_mean 0.529

### 判断

C0 明显低于 B1：

- B1：mean@16 0.695000 / maj@16 0.801050 / best@16 0.904460
- C0：mean@16 0.682000 / maj@16 0.790000 / best@16 0.900000

quality gate 没有解决 full-rollout posterior target 的质量问题，反而减少了有效训练信号。它把 `prompt_weight_mean` 从 B1 的 1.0 降到约 0.716，`nonzero_ratio` 也低于 B1，而最终 mean/maj/best 全部下降。后续不应沿着更强 hard gate 方向扩展；更合理的下一步是 anti-collapse / prior mixing，让 target 保持 B1 的 posterior gain 方向，但不要过度依赖尖锐 answer posterior。

下一步跑 FR-C1-NLL：B1 posterior gain + weighted NLL + `eps=0.05` uniform-valid prior mixing，验证 soft prior 是否比 hard quality gate 更稳。

## 2026-08-02 FR-C1-NLL

### 实验设置

- run_id：`ttrl_full_rollout_powerflow_c1_postgain_eps005_nll_b32_r32_20step_20260802`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_c1_postgain_eps005_nll_b32_r32_20step_20260802.log`
- 训练：batch32 / rollout32 / 20 step / final-only validation / val_n=16
- target：B1 `posterior_gain` + weighted NLL
- anti-collapse prior：`full_rollout_powerflow_eps=0.05`
- hard gate：关闭
- 不用 GT target；`powerflow_enable=False`，`chunk_weighted_nll_enable=True`

### 结果

- mean@16：0.694000
- maj@16：0.797000
- best@16：0.915000

### Timing

- 普通 step 数：19
- ordinary step avg：76.756s
- update_actor avg：54.156s
- throughput avg：1593.995 tokens/s
- step 20 含 final validation：360.857s
- final testing：284.005s

### Target Health

- prompt_weight_mean avg：1.000
- support_coverage avg：1.000
- oov_ratio avg：0.000
- nonzero_ratio avg：0.793
- observed_reward_mean avg：0.477
- step 20 target：nonzero_ratio 0.867 / observed_reward_mean 0.524

### 判断

C1 没有超过 B1 的 mean/maj，但 best@16 明显更高：

- B1：mean@16 0.695000 / maj@16 0.801050 / best@16 0.904460
- C1：mean@16 0.694000 / maj@16 0.797000 / best@16 0.915000
- C0：mean@16 0.682000 / maj@16 0.790000 / best@16 0.900000

这说明 soft prior mixing 比 hard quality gate 更健康：它保持 support coverage 和 prompt weight，不破坏 best@16，多样性更好；但它没有解决 mean/maj 的主问题。当前 evidence 指向：继续改 target shape 的边际收益很小，主要瓶颈更像 answer posterior label estimation 不够可靠或训练 horizon 不够。

下一步不再扩展 hard gate；优先跑 FR-E0-NLL：B1/C1 系列的 `n=64` 完整 rollout label-estimation pilot。目标是用更宽的完整 rollout group 提高 answer posterior 质量，再看 20-step mean/maj 是否能明显越过 B1。

## 2026-08-02 FR-E0-NLL

### 实验设置

- run_id：`ttrl_full_rollout_powerflow_e0_postgain_eps005_n64_nll_b32_20step_20260802`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_e0_postgain_eps005_n64_nll_b32_20step_20260802.log`
- 训练：batch32 / rollout64 / 20 step / final-only validation / val_n=16
- target：C1 `posterior_gain + eps=0.05` + weighted NLL
- hard gate：关闭
- 不用 GT target；`powerflow_enable=False`，`chunk_weighted_nll_enable=True`

### 当前状态

这条实验早停，没有 final validation，因此不作为效果结果。

已落盘的完整 step 只有 3 个：

- step 1：`timing_s/step=172.001`，`timing_s/update_actor=115.493`，`timing_s/gen=31.993`，`perf/throughput=1607.456`
- step 2：`timing_s/step=153.001`，`timing_s/update_actor=117.292`，`timing_s/gen=21.307`，`perf/throughput=1886.798`
- step 3：`timing_s/step=144.641`，`timing_s/update_actor=115.115`，`timing_s/gen=20.312`，`perf/throughput=1935.097`

日志中没有 Python traceback、CUDA OOM 或 RuntimeError；训练进程已不在，最后日志停在 step 3 之后的 SymPy warning。更像外层 worker / 会话中断或日志管道断掉，不是算法在明确报错处失败。

### 判断

`n=64` 的 label-estimation 分支把 `num_actor_samples` 提到 2048，actor update 从 B1/C1 的约 54s 上升到约 115s，普通 step 达到 145-172s。即使能够跑完，12 小时广度优先阶段的性价比也明显低于 `n=32`。当前先不继续扩展 `n=64`，改为把当前最强无 GT 分支 B1 扩到 40 step，判断训练 horizon 是否能把 mean/maj 继续推高。

## 2026-08-02 FR-B1-NLL-40

### 启动计划

- run_id：`ttrl_full_rollout_powerflow_b1_postgain_nll_b32_r32_40step_20260802`
- launcher：`/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/run_front_full_rollout_powerflow_b1_postgain_nll_b32_r32_40step_20260802.sh`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_b1_postgain_nll_b32_r32_40step_20260802.log`
- 训练：batch32 / rollout32 / 40 step / final-only validation / val_n=16
- target：B1 `posterior_gain alpha=6 slack=0.02` + weighted NLL
- hard gate：关闭
- anti-collapse eps：0.0
- 不用 GT target；`powerflow_enable=False`，`chunk_weighted_nll_enable=True`

### 目的

B1 是当前 20-step full-rollout no-GT weighted NLL 的最强 mean/maj 分支：

- B1 20-step：mean@16 0.695000 / maj@16 0.801050 / best@16 0.904460
- C1 20-step：mean@16 0.694000 / maj@16 0.797000 / best@16 0.915000

40-step 扩展用于判断当前方法是否只是训练 horizon 不够，还是 20-step 后已经接近 plateau。若 40-step 仍无法显著接近 85%，下一步应回到完整 rollout target 的 label-estimation 设计，而不是继续堆 hard gate 或 `n=64`。

### 结果

- mean@16：0.695750
- maj@16：0.797174
- best@16：0.906596
- val samples：8000 = 500 prompts x 16 samples

### Timing

- 总 wall time：55 分 28 秒，包含 final validation
- progress 平均：83.22s/step
- step 40 含 final validation：345.315s
- step 40 final testing：272.084s
- step 40 gen：15.251s
- step 40 reward：4.446s
- step 40 update_actor：50.171s

分段普通 step：

- step 1-4：step avg 88.683s / update_actor avg 57.273s / gen avg 18.932s / observed_reward_mean avg 0.349 / response_length mean 999.423
- step 5-12：step avg 76.923s / update_actor avg 54.751s / gen avg 15.633s / observed_reward_mean avg 0.467 / response_length mean 881.104
- step 13-20：step avg 75.579s / update_actor avg 52.535s / gen avg 15.256s / observed_reward_mean avg 0.548 / response_length mean 777.056
- step 21-28：step avg 73.299s / update_actor avg 51.540s / gen avg 15.190s / observed_reward_mean avg 0.515 / response_length mean 752.068

### Target Health

step 40：

- support_coverage：1.000
- oov_ratio：0.000
- top1_mass：0.619
- top2_mass：0.075
- margin：0.545
- target_entropy：3.123
- valid_answer_coverage：0.881
- nonzero_ratio：0.881
- observed_reward_mean：0.556
- clean_rollout_ratio：0.941
- repeated_boxed_ratio：0.013
- empty_boxed_ratio：0.008
- assistant_marker_ratio：0.025
- response_length/mean：735.182
- response_length/clip_ratio：0.034

### 判断

B1-40 没有超过 B1-20：

- B1 20-step：mean@16 0.695000 / maj@16 0.801050 / best@16 0.904460
- B1 40-step：mean@16 0.695750 / maj@16 0.797174 / best@16 0.906596
- C1 20-step：mean@16 0.694000 / maj@16 0.797000 / best@16 0.915000

40 step 的 target health 很干净，valid/nonzero/clean 都在高位，但 final mean/maj 基本没有随训练步数提升。这说明 B1 的主要问题不是 rollout 污染、support 覆盖或训练 horizon，而是 target 本身太弱：posterior gain 仍然给大量非主答案 cluster 分配正权重，weighted NLL 会把同一 prompt 下很多低质量完整 response 一起强化。

下一步跑 FR-B5-NLL：完整 rollout 仍然 batch32 / rollout32，但 target 改为 top-1 answer cluster only，即只强化 full-rollout group 内 posterior 最大的答案簇里的完整 response，其它 answer cluster 权重为 0。这个实验用于验证“低质量 answer cluster 的正权重”是否是 B1 平台期的主因；它不是回到 chunk，也不使用 GT。

## 2026-08-02 FR-B5-NLL

### 启动计划

- run_id：`ttrl_full_rollout_powerflow_b5_top1cluster_nll_b32_r32_20step_20260802`
- launcher：`/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/run_front_full_rollout_powerflow_b5_top1cluster_nll_b32_r32_20step_20260802.sh`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_b5_top1cluster_nll_b32_r32_20step_20260802.log`
- 训练：batch32 / rollout32 / 20 step / final-only validation / val_n=16
- target：`top1_answer_cluster alpha=6` + weighted NLL
- hard gate：关闭
- anti-collapse eps：0.0
- 不用 GT target；`powerflow_enable=False`，`chunk_weighted_nll_enable=True`

### 目的

B1-40 说明单纯延长 posterior_gain 训练不会解决平台期。B5 把同一 prompt 的完整 rollout group 仍作为 label estimation 来源，但只强化 posterior 最大的 answer cluster，避免把低 posterior answer cluster 也作为正样本更新。若 B5 提升 mean/maj，说明 B1 的主要噪声来自过宽的 positive support；若 B5 下降，则说明需要保留多样性或引入更可靠的 answer posterior estimator，而不是 hard cluster selection。

### 结果

- mean@16：0.701250
- maj@16：0.805874
- best@16：0.909430
- val samples：8000 = 500 prompts x 16 samples

### Timing

- 总 wall time：30 分 31 秒，包含 final validation
- progress 平均：91.58s/step，普通 step 约 71-80s
- step 20 含 final validation：355.658s
- step 20 final testing：280.166s
- step 20 gen：14.813s
- step 20 reward：3.284s
- step 20 update_actor：54.010s

### Target Health

B5 的 top-1 answer cluster 截断生效，positive support 明显小于 B1：

- step 1：support_coverage 0.538 / nonzero_ratio 0.341 / observed_reward_mean 0.367
- step 10：support_coverage 0.662 / nonzero_ratio 0.590 / observed_reward_mean 0.550
- step 16：support_coverage 0.757 / nonzero_ratio 0.676 / observed_reward_mean 0.665
- step 20：support_coverage 0.611 / nonzero_ratio 0.551 / observed_reward_mean 0.548 / clean_rollout_ratio 0.941

### 判断

B5 是当前 full-rollout no-GT weighted NLL 组里最好的 mean/maj：

- B1 20-step：mean@16 0.695000 / maj@16 0.801050 / best@16 0.904460
- B1 40-step：mean@16 0.695750 / maj@16 0.797174 / best@16 0.906596
- C1 20-step：mean@16 0.694000 / maj@16 0.797000 / best@16 0.915000
- B5 20-step：mean@16 0.701250 / maj@16 0.805874 / best@16 0.909430

这支持一个中间结论：B1 的全 support posterior-gain target 确实太宽，会把同一 prompt 内很多低 posterior answer cluster 一起作为正样本强化。top-1 cluster 截断能带来小幅提升，但仍远低于 85% 目标，说明只靠 hard top1 还不够。B5 也牺牲了一部分 best@16，相比 C1 的 0.915 低，因此下一步应该测试 top-k cluster 的软截断，试图在去噪和多样性之间找平衡。

下一步跑 FR-B6-NLL：`topk_answer_cluster`，保留 top-2 answer clusters 的全部完整 response，其它 answer cluster 权重为 0。B6 位于 B1 全 support 和 B5 top1 hard mask 之间。

## 2026-08-03 FR-B6-NLL

### 启动计划

- run_id：`ttrl_full_rollout_powerflow_b6_top2cluster_nll_b32_r32_20step_20260803`
- launcher：`/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/run_front_full_rollout_powerflow_b6_top2cluster_nll_b32_r32_20step_20260803.sh`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_b6_top2cluster_nll_b32_r32_20step_20260803.log`
- 训练：batch32 / rollout32 / 20 step / final-only validation / val_n=16
- target：`topk_answer_cluster top_k_answers=2 alpha=6` + weighted NLL
- hard gate：关闭
- anti-collapse eps：0.0
- 不用 GT target；`powerflow_enable=False`，`chunk_weighted_nll_enable=True`

### 目的

B6 保留 top-2 answer clusters 的完整 response，测试 B5 top-1 hard mask 是否过硬。预期 target support 应介于 B1 全 support 和 B5 top1 cluster 之间；如果 B6 超过 B5，说明需要保留少量次优 answer cluster 来维持多样性；如果 B6 不如 B5，说明目前主要收益来自更强的正样本去噪。

### 当前状态

已启动并确认配置：

- `full_rollout_powerflow_mode=topk_answer_cluster`
- `full_rollout_powerflow_top_k_answers=2`
- step 1：support_coverage 0.632 / nonzero_ratio 0.396 / observed_reward_mean 0.367 / step 91.250s
- step 2：support_coverage 0.598 / nonzero_ratio 0.417 / observed_reward_mean 0.352 / step 79.093s
- step 3：support_coverage 0.639 / nonzero_ratio 0.439 / observed_reward_mean 0.390 / step 77.715s

### 失败记录

B6 原始 run 没有得到可用 final acc。日志显示训练正常推进到 step 19，并进入 final validation：

- step 16：support_coverage 0.790 / nonzero_ratio 0.716 / observed_reward_mean 0.635 / step 78.014s
- step 17：support_coverage 0.697 / nonzero_ratio 0.614 / observed_reward_mean 0.534 / step 76.814s
- step 18：support_coverage 0.703 / nonzero_ratio 0.601 / observed_reward_mean 0.501 / step 79.540s
- step 19：support_coverage 0.802 / nonzero_ratio 0.723 / observed_reward_mean 0.620 / step 69.676s
- validation 已打印 `validation generation end` 和 `validation reward start`

但日志随后停在 reward manager 的完整 decoded response 样例打印，没有 `validation reward end`、`val-core/math/acc/*` 或 step 20 metrics。当前没有残留训练进程，GPU 已释放，因此这次 B6 只能作为 step 19 health/timing 参考，不能作为 final 指标。

根因判断：`main_ppo.py` 中 validation reward manager 使用 `num_examine=1`，PrimeRewardManager 会打印一个完整 validation batch 的 decoded responses。val_n=16、500 prompts 时该输出非常大，污染日志并让 final validation 阶段不稳定。已把 validation `num_examine` 改为 0；该改动只关闭 debug 样例打印，不改变 reward 或 metric 计算语义。

### Retry 计划

启动 FR-B6R-NLL，配置与 B6 相同，只换 run_id/runtime/log，并使用 validation 静默化修复：

- run_id：`ttrl_full_rollout_powerflow_b6r_top2cluster_nll_b32_r32_20step_20260803`
- launcher：`/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/run_front_full_rollout_powerflow_b6r_top2cluster_nll_b32_r32_20step_20260803.sh`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_b6r_top2cluster_nll_b32_r32_20step_20260803.log`
- target：`topk_answer_cluster top_k_answers=2 alpha=6` + weighted NLL
- 训练：batch32 / rollout32 / 20 step / final-only validation / val_n=16

### B6R 结果

B6R 正常完成 final validation；validation examine 打印关闭后没有再出现完整 decoded response 大行阻塞。

- mean@16：0.692000
- maj@16：0.804118
- best@16：0.901780
- final val samples：8000 = 500 prompts x 16 samples

### Timing

- 总 wall time：30 分 27 秒，包含 final validation
- progress 平均：91.36s/step
- step 20 含 final validation：351.616s
- step 20 final testing：279.437s
- step 20 update_actor：50.878s
- step 20 throughput：339.351 tokens/s（包含 final validation）

### Target Health

step 20：

- support_coverage：0.701
- oov_ratio：0.299
- top1_mass：0.627
- top2_mass：0.075
- margin：0.552
- target_entropy：2.791
- valid_answer_coverage：0.878
- nonzero_ratio：0.624
- observed_reward_mean：0.553

### 判断

B6R 明显低于 B5：

- B5 top1 cluster：mean@16 0.701250 / maj@16 0.805874 / best@16 0.909430
- B6R top2 cluster：mean@16 0.692000 / maj@16 0.804118 / best@16 0.901780

结论：保留第二答案簇没有带来有效多样性，反而把 top1 hard cluster 已经去掉的噪声重新加回来了。B5 的提升主要来自更强正样本去噪，而不是 support 多样性不足。下一步不扩 B6 到 40 step，改跑 FR-B7-NLL：在 B5 top1 cluster 基础上打开 prompt-level confidence gate，只训练 answer posterior 更可靠的 prompt group，验证 B5 仍然受低置信 prompt 噪声拖累的假设。

## 2026-08-03 FR-B7-NLL

### 启动计划

- run_id：`ttrl_full_rollout_powerflow_b7_top1cluster_gate_nll_b32_r32_20step_20260803`
- launcher：`/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/run_front_full_rollout_powerflow_b7_top1cluster_gate_nll_b32_r32_20step_20260803.sh`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_b7_top1cluster_gate_nll_b32_r32_20step_20260803.log`
- 训练：batch32 / rollout32 / 20 step / final-only validation / val_n=16
- target：`top1_answer_cluster alpha=6` + weighted NLL
- confidence gate：`valid_answer_coverage >= 0.75`、`top1_mass >= 0.55`、`margin >= 0.25`
- 不用 GT target；`powerflow_enable=False`，`chunk_weighted_nll_enable=True`

### 目的

B6R 证明扩大到 top2 answer cluster 会引入噪声；B5 仍然只有 70.1 mean，说明 top1 cluster 内也可能有低置信 prompt 噪声。B7 在 B5 基础上只更新 answer posterior 更可靠的 prompt group。如果 B7 超过 B5，说明 prompt-level confidence 是关键；如果 B7 下降，说明 hard gate 过稀疏，后续应改成软 prompt weighting 而不是归零。

### B7 结果

B7 正常完成 final validation。

- mean@16：0.665125
- maj@16：0.775326
- best@16：0.910594
- final val samples：8000 = 500 prompts x 16 samples

### Timing

- 总 wall time：30 分 38 秒，包含 final validation
- progress 平均：91.94s/step
- step 19 训练步：73.238s
- step 20 含 final validation：354.374s
- step 20 final testing：277.678s
- step 20 update_actor：55.069s
- step 20 throughput：356.719 tokens/s（包含 final validation）

### Target Health

step 20：

- support_coverage：0.425
- oov_ratio：0.575
- top1_mass：0.624
- top2_mass：0.080
- margin：0.544
- target_entropy：1.598
- valid_answer_coverage：0.854
- prompt_weight_mean：0.500
- nonzero_ratio：0.387
- observed_reward_mean：0.548
- clean_rollout_ratio：0.926

### 判断

B7 显著低于 B5：

- B5 top1 cluster：mean@16 0.701250 / maj@16 0.805874 / best@16 0.909430
- B7 top1 cluster + hard confidence gate：mean@16 0.665125 / maj@16 0.775326 / best@16 0.910594

结论：hard prompt confidence gate 过稀疏。早期 step 的 support_coverage 只有 0.091/0.130，nonzero_ratio 只有 0.074/0.116，甚至 step 2 的 actor weighted NLL 为 0；后期 support 虽回升到 0.4-0.5，但已经损失了大量早期学习信号。best@16 仍有 0.910594，说明探索覆盖没有完全坏，主要问题是训练 target 被 hard gate 砍得太窄。

下一步不再做 hard gate，回到 B5 top1 cluster，并加入 soft prompt weighting：低 margin prompt 不归零，只按连续置信度降权。

## 2026-08-03 FR-B8-NLL

### 启动计划

- run_id：`ttrl_full_rollout_powerflow_b8_top1cluster_softprompt_nll_b32_r32_20step_20260803`
- launcher：`/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/run_front_full_rollout_powerflow_b8_top1cluster_softprompt_nll_b32_r32_20step_20260803.sh`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_b8_top1cluster_softprompt_nll_b32_r32_20step_20260803.log`
- 训练：batch32 / rollout32 / 20 step / final-only validation / val_n=16
- target：`top1_answer_cluster alpha=6` + weighted NLL
- hard confidence gate：关闭
- soft prompt weighting：打开，`prompt_weight=max(0.35, min(1, margin/0.45))`
- 不用 GT target；`powerflow_enable=False`，`chunk_weighted_nll_enable=True`

### 目的

B8 保留 B5 的 top1 answer cluster 去噪，同时避免 B7 的 hard gate 稀疏问题。预期 target support 应接近 B5，但低 margin prompt 的有效权重降低。如果 B8 超过 B5，说明 prompt confidence 应作为连续权重进入 full-rollout target；如果仍低于 B5，说明 top1 hard cluster 已经是当前 no-GT target 的较优点，后续应转向更强的 answer posterior/value 估计，而不是继续调 prompt gate。

### B8 结果

B8 正常完成 final validation。中途日志出现 `Timeout during comparison`，但训练没有挂死，最终给出完整 step20 validation。

- mean@16：0.703000
- maj@16：0.806954
- best@16：0.906454
- worst@16：0.337214
- final val samples：8000 = 500 prompts x 16 samples

### Timing

- 总 wall time：30 分 25 秒，包含 final validation
- progress 平均：91.26s/step
- step 19 训练步：70.449s
- step 20 含 final validation：359.751s
- step 20 final testing：283.365s
- step 20 update_actor：54.509s
- step 20 throughput：349.645 tokens/s（包含 final validation）

### Target Health

step 20：

- support_coverage：0.628
- oov_ratio：0.372
- top1_mass：0.628
- top2_mass：0.077
- margin：0.551
- target_entropy：2.597
- valid_answer_coverage：0.865
- prompt_weight_mean：0.815
- nonzero_ratio：0.558
- observed_reward_mean：0.575
- clean_rollout_ratio：0.933

### 判断

B8 相比 B5 只有小幅提升 mean/maj，但 best 略降：

- B5 top1 cluster：mean@16 0.701250 / maj@16 0.805874 / best@16 0.909430
- B8 top1 cluster + soft prompt weighting：mean@16 0.703000 / maj@16 0.806954 / best@16 0.906454

结论：soft prompt weighting 修复了 B7 hard gate 的稀疏问题，也略微改善了 mean/maj，但幅度太小，不足以作为主要突破方向。当前 no-GT full-rollout 系列里，B5/B8 是同一档最优，说明 top1 answer cluster 去噪是有效核心，prompt confidence 只能带来边际收益。

下一步按 12 小时方案进入“放大赢家”：先跑 B5 top1 cluster 的 40-step，验证完整 rollout top1 cluster 是否随 step 数继续爬升。如果 40-step 仍只在 0.70 附近，则当前 target estimator 已到平台期，需要转向更强的 full-rollout answer posterior/value estimator，而不是继续 gate/soft weight 微调。

## 2026-08-03 FR-B5-NLL 40-step 放大

### 启动计划

- run_id：`ttrl_full_rollout_powerflow_b5_top1cluster_nll_b32_r32_40step_20260803`
- launcher：`/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/run_front_full_rollout_powerflow_b5_top1cluster_nll_b32_r32_40step_20260803.sh`
- raw log：`/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_full_rollout_powerflow_b5_top1cluster_nll_b32_r32_40step_20260803.log`
- 训练：batch32 / rollout32 / 40 step / final-only validation / val_n=16
- target：`top1_answer_cluster alpha=6` + weighted NLL
- hard confidence gate：关闭
- soft prompt weighting：关闭，回到 B5 winner 配置
- 不用 GT target；`powerflow_enable=False`，`chunk_weighted_nll_enable=True`
- 配置确认：`trainer.total_training_steps=40`，`trainer.test_freq=40`，`trainer.balance_batch=False`，`ttrl.chunk_state_enable=False`，vLLM attention backend `FLASH_ATTN`

### 目的

B5/B8 是当前 no-GT full-rollout 系列最优档。B8 只带来很小的 mean/maj 增益且 best 略降，因此先放大更干净的 B5 top1-cluster 配置到 40 step，判断 top1 answer-cluster target 是否随训练步数继续提高。如果 40-step 没有明显爬升，说明当前 target estimator 本身已到平台期。
