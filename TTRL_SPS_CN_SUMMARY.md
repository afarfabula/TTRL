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
| Qwen2.5-Math-7B 20-step strict | v38 | 20 | v36 + sharpened-cluster 训练期样本选择 | `mean@4=45.07%`, `best@4=66.33%`, `maj@4=46.85%` |
| Qwen2.5-Math-7B 20-step strict | v39 | 20 | v38 + majority-guarded sharpened selection | `mean@4=58.80%`, `best@4=77.15%`, `maj@4=61.43%` |
| Qwen2.5-Math-7B 20-step strict | v40 | 20 | 回到 v36 样本路径 + prompt-level consistency capacity | `mean@4=69.67%`, `best@4=83.13%`, `maj@4=71.69%` |
| Qwen2.5-Math-7B 20-step strict | v41 | 20 | v40 + 训练 rollout/proposal 温度降到 0.7 + CUDA/cuBLAS infra 固化 | `mean@4=69.57%`, `best@4=82.92%`, `maj@4=72.02%` |
| Qwen2.5-Math-7B 50-step strict | v42 | 50 | v36/v40 + low-budget first4 capacity | `mean@4=71.93%`, `best@4=82.22%`, `maj@4=73.37%` |
| Qwen2.5-Math-7B 50-step strict | v43 | 50 | v42 + base/ref answer-cluster support capacity | `mean@4=73.59%`, `best@4=83.81%`, `maj@4=75.06%` |

v33 达成 Efficient Test-Time RL 目标：20 个训练 step 后，Qwen3-4B 在 Math500/MATH-TTT 上 `val-core/MATH-TTT/acc/mean@4=0.7907444668008048`，超过 75% 目标。Qwen3-4B seed sweep 的第一个有效 seed 达到 `0.8008048289738431`，随后因目标模型切换暂停。

同一套方案换到 Qwen2.5-Math-7B 后达到 `val-core/MATH-TTT/acc/mean@4=0.8309859154929577`。对应 raw uncollapsed validation 为 `mean@32=70.57%`、`best@32=93.19%`、`maj@32=81.90%`，说明 Qwen2.5-Math 的候选 support 明显更强，answer-cluster selection 能把一部分 high-best 候选质量转成 selected `mean@4`。但这依赖 validation `n=32`，不符合当前 strict n=4 goal。

v35 重新建立 strict n=4 baseline：最终 validation 样本数 `1988=497*4`，没有 validation answer selection，`mean@4=0.6896378269617707`，`best@4=0.8220402414486921`，`maj@4=0.7137223340040241`。这证明当前差距不是评估聚合问题，而是 4 条样本本身的正确率还不够。

v36 启用 `sps_answer_sharpen_capacity=True`，把 sharpened answer-cluster confidence 直接用于训练容量。它在 strict n=4 下小幅提升到 `mean@4=0.7037223340040242`，说明这个内部锐化信号方向有效，但仍远低于 85%。v36 跑完后 worker 出现 `/proc` 损坏和 `cuInit=304`，后续 GPU 实验必须先换健康 worker 或重启修复。

v37 计划在 v36 基础上只改训练期样本选择：启用 `sps_rollout_selection=sharpened_cluster` 和 `sps_selection_priority=nonclip_parseable_bucket`，把更新样本投影到可解析、非截断、答案簇一致的内部 support。最终 validation 仍保持 `n=4` 且禁用 answer selection。判断重点是 `selected_parseable_rate`、`selected_clip_rate`、`selected_cluster_rate` 这些内部指标，以及 strict `best@4` 和 `mean@4` 是否一起提升。

v37 尚未启动：worker `984279` 登录后 `/proc/self` 和 `/proc/meminfo` 缺失，`PROC_COUNT=0`，`cuInit=304`。虽然 `nvidia-smi -L` 能看到 8 张 B200，但这个状态不能安全跑 Ray/CUDA 训练。已从 master 终端 kill `984279`，确认列表清空后申请新 worker `985081`；截至 2026-07-05 16:45 CST 已 pending 约 59 分钟，`mlx worker list` 仍显示 `podIP` 为空、端口 `9000`，不能登录也不能启动 v37；下一步需要取消重试或换资源/队列，但不能并行申请第二个 worker。

2026-07-05 16:46 CST 已 kill 卡住的 `985081`，确认列表清空后用同一 8x B200 命令重新申请；新 worker id 为 `985114`。截至 17:30 CST，`985114` 已 pending 约 42 分钟，`mlx worker list` 仍显示 `podIP` 为空、端口 `9000`；短超时 `mlx worker login 985114` 返回 `worker has not been ready yet.`，后续两轮各 6 次 30 秒轮询和一轮 10 次 30 秒轮询仍未 ready，无法进入 worker 或做 GPU 健康检查。

2026-07-05 17:36 CST 用户提供新的 8x H100 worker `985145`。已成功登录，host 为 `trial-301562365-trialrun-301562365-worker-0`；初始健康检查通过：`/proc/self` 和 `/proc/meminfo` 存在，`PROC_COUNT=71`，`nvidia-smi -L` 看到 8 张 H100 80GB，CUDA compat 按 driver `535.129.03` 启用 cuda-12.9 compat，`cuInit: 0`。17:38 CST 已 kill 旧的 pending B200 worker `985114`，确认列表只剩 `985145`，恢复单 worker 约束。

18:09-18:18 CST 在 H100 上用原版恢复后的 Ray/verl 代码跑 v37，多次在训练前失败：Ray actor 创建时 `RuntimeEnvSetupError`，runtime-env agent HTTP 被 Compliance Gateway 403 拦截，错误为 `Missing Destination-Service header`。清标准代理变量、扩展 `NO_PROXY`、设置 `NO_PROXY=*`、清空所有变量名包含 `proxy` 的环境变量都不能解决。最小 Ray 复现确认：默认 node IP `172.18.0.16` 时带 `runtime_env.env_vars` 的 actor 会 403；强制 `ray.init(..., _node_ip_address="127.0.0.1")` 的最小 actor 能正常返回，说明问题是 H100 worker 上 Ray runtime-env agent 通过容器私网 IP 通信被平台网关拦截，不是 TTRL 算法问题。

随后尝试用预启动 loopback Ray 作为无代码 workaround，但 Ray CLI 仍选择 `Local node IP: 172.18.0.16`，并触发已知 `/proc` 损坏状态：`/proc/self` 和 `/proc/meminfo` 缺失，`PROC_COUNT=0`，`ps` 提示 `mount -t proc proc /proc`。虽然 `nvidia-smi` 仍能看到 8 张 H100 空闲，但这个 worker 已不能可靠跑 Ray/verl。v37 尚未得到 strict n=4 训练/验证结果，不能做算法结论，也不应为 v37 做提升 commit。

2026-07-05 18:40 后用户提供 8x A100 worker `985168`。A100 初始健康，`/proc/self` 和 `/proc/meminfo` 存在，CUDA `cuInit: 0`，Ray 自动使用 `127.0.0.1`，没有复现 H100 的 runtime-env 403。v37 strict n=4 原路径成功跑完第 1 个 step：`timing_s/step=76.792`，其中生成 `59.812s`，actor update `14.491s`，`perf/total_num_tokens=239358`，verl 日志吞吐 `389.621 tok/s/GPU`，折算整机约 `3.12k tok/s`。随后进入第 2 个 step 前在 vLLM `wake_up(kv_cache)` 处 CUDA OOM，未跑到 final validation，因此不能作为 v37 算法结果。

本轮修复了 Ray/训练失败时的退出路径：`verl/verl/trainer/main_ppo.py` 在 driver 侧加 `finally: ray.shutdown()`；v37 wrapper 改为训练命令失败后仍继续抓 task log、写 throughput/proc/GPU 状态，并用 `EXIT` trap 兜底记录最终状态。验证：wrapper `bash -n` 通过，`main_ppo.py` 编译通过；最小 Ray teardown 脚本在同一 A100 上显式 `ray.shutdown()` 后 `/proc` 仍健康，8 张 A100 显存均为 `0 MiB`。当前 worker 没有残留 CUDA 进程。

随后做 10 轮完整链路启动/退出压力测试，脚本为 `verl/examples/ttrl/worker_stress_v37_teardown_10x.sh`，每轮用 v37 路径跑 1 个训练 step、关闭 validation、换新的短 `RAY_TMPDIR`。结果文件 `verl/v37_teardown_stress_10x_summary.tsv` 显示前 7 轮均 `status=0` 且 `proc_ok=OK`，每步约 `76.3-77.3s`，整轮冷启动+训练+退出约 `190-196s`。第 8 轮训练也 `status=0`，但退出后 `/proc` 损坏：`proc_ok=BAD`、`proc_count=0`、`/proc/self=False`、`/proc/meminfo=False`，shell 出现 `Error, do this: mount -t proc proc /proc`。因此 `ray.shutdown()` 修复只能改善退出记录和普通 Ray driver teardown，不能根治 MLX worker/container 的 procfs 损坏；当前 A100 worker `985168` 已不可继续跑 Ray/psutil/CUDA 训练，必须换 worker 或重启。

用户随后提供新 A100 worker `985218`，明确要求不要做 CUDA compat preflight、不要做驱动/compat 修复。新 worker 初始健康：`PROC_COUNT=72`，8 张 A100 空闲。本轮按设想测试精简 Ray init：`ray_init.no_runtime_env=True`，不传 Ray `runtime_env`；`ray_init.include_dashboard=False`；`ray_init.node_ip_address=127.0.0.1`。stress 脚本默认 `RUN_CUDA_COMPAT_PREFLIGHT=0`，日志明确记录 `CUDA_COMPAT_PREFLIGHT_SKIPPED`。结果文件 `verl/v37_teardown_stress_noenv_10x_summary.tsv` 显示前 8 轮 `proc_ok=OK`，越过了上次第 8 轮失败点，但第 9 轮仍然 `status=0` 后 `/proc` 损坏：`proc_ok=BAD`、`proc_count=0`、`/proc/self=False`、`/proc/meminfo=False`。因此 no-runtime-env、关 dashboard、loopback 只能延后失败，不能根治；日志里仍能看到 `ray::WorkerDict <defunct>` 和 compute app `[Not Found]`，问题更像 worker 进程 teardown / 平台 namespace 清理。当前 worker `985218` 也不可继续跑 Ray/psutil/CUDA 训练。后续可靠策略是避免同一 worker 多次完整启停：一个新 worker 跑一个长实验，或改成单个 Ray 生命周期内串行多配置。

`mlx worker` 没有 `status/logs` 子命令，只能用 `list/login/kill/quota` 做诊断。`mlx worker quota` 的 public resource 表里没有显示当前指定的 `cloudnative-useast1b` B200 可用量，这和 `985081` 长时间 pending 一致。

当前结论：SPS 信号更适合作为训练期置信度、容量、样本选择和分布锐化信号，而不是独立 dense reward；当前 goal 下不能再靠推理时多采样选择。下一步必须提升 4 条 rollout 自身的候选质量，因为 v36 的 strict `best@4=83.00%` 仍低于 85%。

2026-07-05 晚间 H100 v38 首次启动没有进入训练。虽然 Ray GCS 已经连到 `127.0.0.1`，但 wrapper 没有覆盖 MLX 环境里的 `MY_HOST_IP=10.*`，verl 的 WorkerDict 会优先用 `MY_HOST_IP` 生成 c10d `MASTER_ADDR`，导致 `WorkerDict.__init__` 阶段卡在 `TCPStore`，GPU 一直空闲。已修正 v38 runner：强制 `MY_HOST_IP=127.0.0.1`、`MASTER_ADDR=127.0.0.1`，并把 `GLOO/NCCL/TP_SOCKET_IFNAME` 设为 `lo`，同时禁用 `NCCL_SOCKET_FAMILY`。这说明 Ray 自己走 loopback 不够，WorkerDict/c10d 的地址也必须走 loopback。

修正后在 H100 worker `985239` 重跑 v38，loopback 设置已确认生效，Ray GCS 和 WorkerDict 报错里的 IP 都是 `127.0.0.1`。任务越过配置和数据集校验，随后在 `trainer.init_workers()` 的 `ref_policy_wg.init_model()` 阶段 WorkerDict actor 系统级死亡，同时 `/proc` 立即损坏为 `PROC_COUNT=0`，没有产生任何 training step 或 strict n=4 validation 结果。因此 v38 仍是 infra/procfs 失败，不是算法结果；`985239` 不能继续跑 Ray/psutil/CUDA 实验。

2026-07-05 晚间新的 B200 worker `985258` 可用，健康检查通过：`/proc/self`、`/proc/meminfo` 正常，8 张 B200 空闲，driver `580.105.08`，compat conf 为空，`cuInit: 0`。用 v38 strict n=4 runner 完整跑完 20 step，训练和最终 validation 均成功，退出状态 0，跑完后 `/proc` 仍健康。

v38 B200 strict n=4 最终结果很差：`mean@4=0.4507042253521127`，`best@4=0.6632555331991953`，`maj@4=0.46845472837022134`。内部指标显示训练期选择看起来很干净：`selected_parseable_rate=1.0`、`selected_clip_rate=0.0`、`answer_sharp_confidence=0.757`、`answer_effective_K=1.921`，但 strict validation 的候选质量反而塌了。这说明单纯强化 sharpened cluster 会把模型推向“自洽但错误”的答案簇，v38 不是提升方案，不能 commit 作为算法改进。

下一步算法方向：不要继续加强 raw sharpened-cluster selection，而是在训练期加入保守 gating，要求 majority pseudo label、weighted-label confidence、answer entropy/effective_K、base support 等内部信号互相一致后才放大更新；对高置信但与 majority/weighted label 冲突的答案簇降权，避免错误簇被 20 step 快速锐化。

v39 在同一 B200 worker `985258` 上完整跑完 20 step strict n=4，唯一算法改动是把 v38 的 `sps_selection_require_majority=False` 改成 `True`，即训练期 sharpened selection 必须优先落在 raw majority pseudo label 的答案簇内。结果从 v38 的 `mean@4=45.07%` 恢复到 `58.80%`，`best@4` 从 `66.33%` 恢复到 `77.15%`，说明 v38 的确有错误簇放大问题。但 v39 仍低于 v35/v36 的 strict baseline，不能作为提升方案。

v39 的内部指标：`selected_parseable_rate=1.0`，`selected_clip_rate=0.008`，`selected_cluster_rate=0.699`，`selection_fallback_rate=0.0`，`answer_sharp_confidence=0.777`。这说明 majority guard 生效且没有退化成 fallback，但 rollout-level support projection 仍然太激进。下一步应回到接近 v36 的样本选择，只在 prompt-level 做更保守的 capacity / do-no-harm gate：当 `majority_ratio`、`weighted_label_confidence`、`majority_sharp_confidence` 互相冲突时降低整题更新强度，而不是强行挑 32 条 rollout 锐化。

v39 退出状态为 0，但训练结束后 worker `985258` 的 `/proc` 再次损坏：`PROC_SELF_BAD_AFTER`、`PROC_MEMINFO_BAD_AFTER`、`PROC_COUNT_AFTER=0`。这个 worker 不能再继续跑 Ray/psutil/CUDA 训练，后续实验必须换健康 worker 或重启。

v40 在新 B200 worker `985302` 上完成 20 step strict n=4。设计是回到 v36 的 `sps_rollout_selection=first`，不再做 v38/v39 的 rollout-level projection，只新增 prompt-level consistency capacity：当 majority answer 在 sharpened answer distribution 里的置信度低时，降低整题训练权重；如果 SPS weighted answer 和 raw majority answer 不一致，再乘 `sps_consistency_disagreement_penalty=0.35`。这样把“分布锐化”用于保守控制更新容量，而不是强行挑 rollout。

v40 结果：`mean@4=0.6966800804828974`，`best@4=0.8312957746478873`，`maj@4=0.7168913480885312`。它略高于 v35 strict baseline 的 `68.96%`，但低于当前最好 v36 的 `70.37%`，因此不是算法提升，也不能作为达标方案。内部指标说明 consistency capacity 生效：step20 `answer_sharp_confidence=0.849`，`majority_sharp_confidence=0.849`，`consistency_capacity=0.849`，`majority_ratio=0.574`，`pass@32=0.875`，clip ratio 降到 `0.074`；但 strict `best@4` 仍只有 `83.13%`，说明 4 条 validation rollout 的候选质量仍不足。

v40 跑完后 worker `985302` 的 `/proc` 再次损坏：`PROC_SELF_BAD_AFTER`、`PROC_MEMINFO_BAD_AFTER`、`PROC_COUNT_AFTER=0`。这个 worker 不能继续跑 Ray/psutil/CUDA 训练。

当前最好 strict n=4 20-step 结果仍是 v36：`mean@4=70.37%`，相比 v35 strict baseline `68.96%` 提升 `+1.41pp`。下一步算法不能再靠 validation-time selection；应利用 v40 暴露的现象，把高 `pass@32` 但低 `best@4/mean@4` 的差距转化为训练侧低预算候选质量目标，例如提高前几条样本对 majority/parseable answer 的概率质量，同时保留保守 capacity gate。

v41 已在 B200 worker `986493` 上完成。infra 侧结论是正向的：训练前后 `/proc/self` 和 `/proc/meminfo` 都正常，`cuInit=0`，8 张 GPU 退出后显存回到 `0 MiB`，没有 Ray/TTRL 残留进程。新 helper `verl/examples/ttrl/setup_ttrl_cuda_env.sh` 会把 venv 内 cu12.9 的 cuBLAS、cuDNN、NCCL、cuDART、NVRTC、cuSolver、cuSparse、nvJitLink 等库放到 `LD_LIBRARY_PATH` 最前面，并把 Ray/HF/torch/vLLM/Triton/TorchInductor 缓存放到 `/tmp/ttrl_cache/<exp>`。配合 `check_cuda_compat_preflight.sh` 的 driver-version compat 规则后，GEMM smoke、1-step TTRL smoke、20-step TTRL+validation 都正常完成。这是目前支持“混合 CUDA/cuBLAS 依赖会污染启动环境”假设的最强缓解证据，但还不能证明全局根因。

v41 算法侧不是提升。它把训练 rollout 温度和 `sps_proposal_temperature` 从 `1.0` 降到 `0.7`，同时由于低温语义不允许复用温度 1.0 下的 logprob，实际配置关闭了 `sps_reuse_rollout_log_probs_as_old` 和 `sps_reuse_base_log_probs_as_ref`。最终 strict validation 仍保持 `n=4` 且禁用 answer selection，结果为 `mean@4=69.57%`、`best@4=82.92%`、`maj@4=72.02%`，低于当前最好 v36 的 `mean@4=70.37%`。低温训练提高了速度和内部分布锐化，但没有提升最终 4 条低预算样本的正确率。

v41 性能记录：1-step smoke 为 `34.271s/step`、约 `7913.8 tokens/s`；20-step 完整训练中非 validation 步骤 2-19 平均 `26.518s/step`、约 `9456.1 tokens/s`，步骤 11-19 平均 `24.706s/step`、约 `9524.3 tokens/s`。根盘只有约 `14G` 空余，但当前约束是不作为训练 gate，只记录；缓存和临时目录均落到 `/tmp`。

## v42 计划：低预算容量信号

本轮先读了 10 篇相关 arXiv/最新论文，覆盖 TTRL、自训练、verifier-free RL、entropy/confidence reward、confidence-weighted self-consistency 和 latent self-consistency。共同启发是：内部 confidence/entropy/majority 信号可以作为无监督训练信号，但容易把错误答案簇锐化；需要把它用作保守 capacity，而不是单独 reward 或 validation-time selection。

v42 的设计只针对当前 strict `mean@4` 的失败模式：v40/v41 说明 32/64 条 rollout 的 wide support 里有较强候选，但最终 4 条低预算样本仍不够好。因此 v42 不再做 rollout-level 强选，也不降低训练温度，而是在训练期加入 `sps_low_budget_capacity`：每个 prompt 只看训练生成组里前 `4` 条 rollout，计算它们的可解析率、截断率、局部 majority、以及它们落在全局 raw-majority answer 上的比例。如果前 4 条和全局 majority 不一致，就降低这个 prompt 的训练权重；如果前 4 条可解析、非截断、且已经有质量落在全局 majority 上，才允许更强更新。

这个信号不使用 Math500 标注，不改变 validation，不使用 `n=32` selection，不算 best-of 或 major vote。最终仍以 strict `val_kwargs.n=4`、禁用 `validation_answer_selection_enable` 的 `mean@4` 为唯一达标口径。v42 runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v42_lowbudget_capacity_strict_n4.sh`，沿用 v41 的 CUDA/cuBLAS infra recipe，并跑 `50` step。

v42 已完成 50 step，但没有达标：strict `val-core/MATH-TTT/acc/mean@4=71.93%`，`best@4=82.22%`，`maj@4=73.37%`。这里仍然只认 `mean@4`，`best@4/maj@4` 只是诊断，不能算成功。相比 v36 的 20-step strict `70.37%` 有提升，但预算不同，不能当成明确算法胜利；距离 85% 还很远。

内部信号显示 v42 的 low-budget gate 确实把前 4 条 rollout 的一致性拉高了：step 50 的 `low_budget_majority_mass=0.750`、`low_budget_agreement=0.875`、`low_budget_capacity=0.727`，`answer_sharp_confidence=0.965`，`majority_ratio=0.760`。问题是这种一致性没有足够转化成正确率，说明模型更自洽了，但错误簇仍会被锐化。下一步不能继续单纯强化 majority/sharpening，需要加入能拒绝高置信错误答案的内部 verifier/self-check 或多视角一致性 gate。

infra 正常：runner 退出状态 0，最终 `/proc/self` 和 `/proc/meminfo` 都正常，`PROC_COUNT_FINAL=96`。非 validation 训练步 2-49 平均 `24.254s/step`，整机约 `10.50k token/s`；runner 自带 steps 41-50 汇总包含 final validation step，所以显示 `47.258s/step`、`5.01k token/s`，不能拿它代表纯训练吞吐。

## v43 计划：base-support 内部 verifier capacity

v42 的内部一致性已经很高，但 strict `best@4` 仍只有 `82.22%`，所以继续提高 majority/sharpening 没有充分理由。v43 改用 base/ref support 做一个内部 verifier：对每个 prompt，把 64 条 rollout 按最终答案分簇，然后用 ref/base logprob 对每个答案簇做 logsumexp 支持度。如果采样 majority 也是 base/ref 支持的答案，就允许较强更新；如果采样 majority 很强但 base/ref 支持的 top answer 不同，就降低这个 prompt 的训练容量。

这仍然是无监督信号，不用 Math500 标注，也不改 validation。v43 新增 `sps_base_support_capacity=True`，默认配置里开关保持 off；runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v43_base_support_capacity_strict_n4.sh`。判断重点不是 answer 更锐，而是 `base_support_agreement/base_support_capacity` 能否识别高风险 prompt，并让 strict `mean@4`、尤其 `best@4` 上升。

v43 已完成 50 step，strict `val_kwargs.n=4` 且禁用 validation answer selection。结果是 `mean@4=73.59%`、`best@4=83.81%`、`maj@4=75.06%`。相比 v42 的 `mean@4=71.93%` 提升 `+1.66pp`，是当前 Qwen2.5-Math-7B strict 50-step 最好结果，但仍没有达到 85%。这里的 `best@4` 也还低于 85%，所以瓶颈仍然是 4 条低预算样本里的正确候选不够，而不是验证阶段选择方法。

v43 的 step 50 内部指标：`answer_sharp_confidence=0.969`、`low_budget_majority_mass=0.719`、`base_support_capacity=0.781`、`base_support_agreement=1.000`、`train_weight=0.577`、`ground_truth_reward=0.781`。这说明 base/ref support 作为保守 capacity 有帮助，但晚期大多和 majority 同向，独立纠错能力不够强。下一版应保留 v42/v43 的保守容量栈，但加入更有区分度的内部 correctness 信号，例如低温自检/改写一致性、base support 和 first4 冲突惩罚、answer-cluster margin，而不是继续单纯提高锐化强度。

infra 正常：runner 退出状态 0，最终 `/proc/self` 和 `/proc/meminfo` 正常，`PROC_COUNT_FINAL=102`。非 validation 训练步 2-49 平均 `24.802s/step`，整机约 `10.22k token/s`；step 50 包含 final validation，`testing=225.862s`、整步 `247.066s`。

## v44 计划：first4 和 base/ref 的 cross-view capacity

v44 不增加生成次数，也不改变 validation。它把 v42 的 first4 低预算视角和 v43 的 base/ref 支持视角合成一个训练期容量：`sqrt(base_support_majority_confidence * low_budget_majority_mass)`，再乘 first4 可解析率和非截断率；如果 base/ref top answer 或 first4 local majority 不同意 raw majority，就乘 `0.35` 惩罚。

这个设计来自 v43 的现象：base/ref support 有提升，但晚期多数时候和 majority 同向，单独 hard cap 不够像 verifier。v44 要求两个内部视角同时支持同一个 majority 答案簇，才允许较强更新；任一视角冲突就降低更新。它仍然不用 Math500 标注，不做 validation-time selection，不使用 best-of/major vote/n=32 作为达标指标。runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v44_cross_view_capacity_strict_n4.sh`，最终仍只看 strict `mean@4` 是否达到 85%。

v44 已完成 50 step strict n=4，结果为 `mean@4=73.84%`、`best@4=84.26%`、`maj@4=75.27%`。相比 v43 的 `mean@4=73.59%` 和 `best@4=83.81%` 是小幅提升，但仍远低于 85% 目标。这里仍然只认 `mean@4`，`best@4/maj@4` 只是诊断。

v44 的 step 50 内部指标显示容量信号确实很强：`cross_view_capacity=0.844`、`base_support_capacity=0.791`、`base_support_agreement=1.000`、`low_budget_majority_mass=0.906`、`low_budget_agreement=1.000`、`answer_sharp_confidence=0.979`、`weighted_label_confidence=0.793`、`pass@32=1.000`。问题是这些内部一致性仍没有足够转化成 4 条低预算样本的正确率，说明 base/ref 和 first4 多数时候只是确认同一个 majority 簇，独立纠错能力不够。

v44 infra 正常：退出状态 0，最终 `/proc/self` 和 `/proc/meminfo` 正常，`PROC_COUNT_FINAL=104`，8 张 B200 显存均释放到 `0 MiB`。非 validation steps 2-49 平均约 `24.77s/step`、整机约 `10.23k tokens/s`；包含最终 validation 的 steps 41-50 汇总为 `47.596s/step`、`4923.941 tokens/s`。下一步不要再单纯提高锐化或容量，而应加入更能区分正确性的内部信号，例如 answer-cluster margin / ambiguity control：只有当 majority 答案在 sharpened distribution、first4、base/ref 三个视角里都相对第二簇有明确 margin 时才强更新，对高置信但低 margin 的题降权。

## v45 计划：answer-cluster margin capacity

v45 继续保持 strict validation：`n=4`，禁用 validation answer selection，不使用 best-of、major vote 或 n=32 选择。算法动机来自 v44：即使 `low_budget_agreement=1.000`、`base_support_agreement=1.000`、`answer_sharp_confidence=0.979`，`mean@4` 仍只有 `73.84%`，说明“高一致性”还不足以判断正确，可能只是把错误簇锐化。

v45 新增默认关闭的 `sps_margin_capacity`。它计算 raw majority 答案相对第二答案簇的三个 margin：sharpened answer 分布 margin、base/ref support margin、first4 low-budget margin。三个 margin 截断到非负后取几何平均，再用 `floor=0.35` 转成训练容量，并乘 first4 可解析和非截断率。开启后这个容量继续作为 prompt train weight 的上限。这样做的目的不是调参碰运气，而是用“多数簇是否和第二簇拉开距离”来识别高置信但仍歧义的题。

runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v45_margin_capacity_strict_n4.sh`，沿用 v44 的 infra recipe 和所有 strict validation 约束。新增日志指标包括 `margin_capacity`、`sharp_majority_margin`、`base_support_majority_margin`、`low_budget_majority_margin`。

v45 已完成 50 step strict n=4，结果退步：`mean@4=73.09%`、`best@4=83.13%`、`maj@4=74.61%`，低于 v44 的 `mean@4=73.84%`、`best@4=84.26%`、`maj@4=75.27%`。因此 v45 不是提升方案，不做 improvement commit。

内部指标显示 margin gate 确实生效，但它只是更保守地控制容量，没有提高低预算候选正确率。step 50：`train_weight=0.648`、`margin_capacity=0.822`、`sharp_majority_margin=0.965`、`base_support_majority_margin=0.760`、`low_budget_majority_margin=0.750`、`cross_view_capacity=0.768`、`low_budget_majority_mass=0.844`、`answer_sharp_confidence=0.973`、`ground_truth_reward=0.793`。结论是：继续加纯 capacity cap 不够，下一版需要能改变候选质量的 train-time correctness signal，例如低成本 self-check / rephrase consistency / process consistency，而不是再只给同一个 pseudo-label 簇调权重。

v45 性能：非 validation steps 2-49 平均 `24.494s/step`、整机 `10213 tokens/s`；steps 41-49 平均 `23.759s/step`、整机 `9993 tokens/s`；最终 validation step `testing=233.424s`、整步 `254.095s`。infra 方面，preflight 健康且训练 status 为 0，GPU 释放到 `0 MiB`，但退出后 `/proc` 再次损坏：`PROC_SELF_BAD_FINAL`、`PROC_MEMINFO_BAD_FINAL`、`PROC_COUNT_FINAL=0`。当前 worker `986493` 不能继续用于 Ray/CUDA 训练，必须换 worker 或重启后再做下一轮。

## v46 计划：low-budget train rollout repair

v46 不再继续加纯 capacity cap。它回到当前最好 v44 的配置，只把训练阶段的 downsampling 改成局部修复：先保留原来的 first 32 条训练 rollout，只检查前 4 条低预算位置。如果前 4 条里有不可解析、截断、或者不等于 raw majority pseudo label 的样本，就从同一题后面的 64 条训练 rollout 里找“可解析、非截断、等于 raw majority”的样本替换。最多替换 4 条，后面 28 条保持原始顺序。

这个设计针对 v44/v45 的现象：strict `best@4` 仍不到 85%，说明前 4 条候选质量不够；v45 说明只降低 prompt weight 不够；v38/v39 又说明全局投影到 majority cluster 会放大错误簇。所以 v46 只修前 4 个训练位置，尽量让训练更关注低预算候选质量，同时避免把 32 条训练样本全部改成 majority cluster。

v46 仍然不用 Math500 标注，不改 validation，不使用 best-of、major vote 或 n=32 选择。runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v46_low_budget_repair_strict_n4.sh`，开启 `ttrl.sps_rollout_selection=low_budget_repair` 和 `ttrl.sps_low_budget_repair_max_replacements=4`。新增日志指标包括 `low_budget_repair_rate`、`selected_low_budget_parseable_rate`、`selected_low_budget_clip_rate`、`selected_low_budget_cluster_rate`、`available_low_budget_repair_rate`。

注意：v45 后 worker `986493` 的 `/proc` 已坏，不能继续跑 Ray/CUDA。v46 只能在新的或重启后的健康 worker 上跑。

v46 已在健康 B200 worker `986717` 上完成 50 step strict n=4，结果退步：`mean@4=72.33%`、`best@4=83.28%`、`maj@4=73.61%`，低于当前最好 v44 的 `mean@4=73.84%`、`best@4=84.26%`。因此 v46 不是提升方案，不做 improvement commit。

v46 的机制在内部指标上确实生效：step 50 `low_budget_repair_rate=0.250`、`available_low_budget_repair_rate=0.828`、`selected_low_budget_parseable_rate=1.000`、`selected_low_budget_clip_rate=0.000`、`selected_low_budget_cluster_rate=1.000`、`answer_sharp_confidence=0.983`、`answer_effective_K=1.036`、`ground_truth_reward=0.805`。但 validation 下降说明局部修复只是把前 4 个训练位置推向 raw majority 答案簇，并没有增加独立正确性证据，反而会强化自洽但错误的 majority cluster。

v46 infra 正常：训练退出状态 0，preflight 健康，driver `580.105.08` 下清空 compat，`cuInit=0`，cuBLAS/cuDNN/NCCL/nvJitLink 来自 venv cu12.9；训练结束后 `/proc/self` 和 `/proc/meminfo` 仍正常，`PROC_COUNT_FINAL=76`，8 张 B200 GPU 释放到 `0 MiB`。包含 final validation 的 steps 41-50 汇总为 `47.759s/step`、`4687 tokens/s`，最终 validation step `testing=232.951s`、整步 `254.550s`。

结论：当前最好 strict 50-step 仍是 v44 `mean@4=73.84%`。下一版不能再做单纯 majority/capacity-only 变体，应加入更独立的训练期正确性信号，例如 base/ref-supported rollout selection、self-check/rephrase/process consistency，或者能拒绝高置信错误簇的冲突惩罚，同时 validation 继续保持 `n=4` 且禁用 answer selection。

## v47 计划：base-supported low-budget repair

v47 保留 v46 “只修前 4 个低预算训练位置”的目标，但不再只看 raw majority。新的选择模式是 `sps_rollout_selection=base_supported_repair`：如果前 4 个训练 slot 里有不可解析、截断、或不等于 raw majority 的样本，才考虑从同题后续 rollout 里找可解析、非截断、等于 raw majority 的候选替换；候选按长度归一化 base/ref logprob 排序，并且只有当候选的 base/ref support 不低于被替换 slot 时才接受。v47 先用 `sps_base_supported_repair_min_gain=0.0`，理由是“不能降低 base/ref 支持”，不是后验调阈值。

这个设计来自 v46 的失败：v46 把 first4 训练槽位干净地修到了 majority cluster，但 validation 降低，说明 raw majority 会强化自洽错误簇。v43/v44 说明 base/ref support 有一定独立信息，所以 v47 把它从 prompt-level capacity 移到 rollout-level repair guard。它仍然不用 Math500 标注，不改变 strict validation，不使用 best-of、major vote、n=32 selection 或 validation answer selection。runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v47_base_supported_repair_strict_n4.sh`。

v47 已在 B200 worker `986717` 上完成 50 step strict n=4，结果为 `mean@4=73.29%`、`best@4=84.42%`、`maj@4=74.74%`。它比 v46 的 `mean@4=72.33%` 有恢复，但低于当前最好 v44 的 `mean@4=73.84%`，所以不是提升方案，不做 improvement commit。

内部指标说明 base-supported guard 生效：step 50 `base_supported_repair_rate=0.250`、`base_supported_available_repair_rate=0.777`、`base_supported_low_budget_parseable_rate=1.000`、`base_supported_low_budget_clip_rate=0.000`、`base_supported_low_budget_cluster_rate=0.969`、`base_supported_replacement_base_gain=0.053`、`base_supported_skipped_base_guard_rate=0.031`。同时 `answer_sharp_confidence=0.966`、`base_support_capacity=0.772`、`cross_view_capacity=0.727`、`ground_truth_reward=0.777`。这说明 v47 确实比 v46 更保守，避免了完全无约束地修向 majority，但仍没有提供足够独立的正确性证据来提高 strict mean@4。

v47 infra 正常：训练退出状态 0，preflight 中 `/proc` 正常、driver `580.105.08`、compat action `clear_compat`、`cuInit=0`、GEMM smoke 通过，cuBLAS/cuDNN/NCCL/nvJitLink 来自 venv cu12.9。runner 结束时 `/proc` 仍健康；结束瞬间 GPU2 有短暂 32GB 残留，但 `2026-07-07 03:16:25` 复查显示 8 张 B200 全部 `0 MiB`，无 Ray/vLLM/main_ppo/TaskRunner 残留。训练后段非 validation steps 40-49 约 `24.71s/step`、整机约 `9332 tokens/s`；包含 final validation 的 steps 41-50 汇总为 `47.357s/step`、`4857 tokens/s`。

结论：当前最好 strict 50-step 仍是 v44 `mean@4=73.84%`。v47 的经验是 base/ref support 能缓解 v46 的 majority repair 伤害，但 answer-cluster repair 这一类方法已经接近上限。下一步应转向更独立的训练期信号，比如 self-check、rephrase consistency 或 process consistency，用作 correctness filter 或 contrastive penalty；validation 继续保持 `n=4` 且禁用 answer selection。

## v48 计划：rollout 过程一致性容量

v48 不再修复或重排 first4 到 majority 簇，而是在 v44 的 cross-view capacity 上新增一个默认关闭的 `sps_process_consistency_capacity`。它只看训练 rollout 自身文本，不额外生成、不调用外部 verifier：最终答案必须可解析、不能截断；同一条解里出现的多个 `\boxed{}` 答案必须一致；最后一个 boxed answer 应该在回答尾部，避免“先给答案后继续修改”；如果 final answer 后还有明显纠错/不确定表达则降权。

这个信号的动机来自 v46/v47：把训练样本推向 majority cluster 会让内部一致性更干净，但 strict validation 反而下降，说明 majority/base/ref 多数时候只是在确认同一个可能错误的答案簇。v48 改为问“这个答案簇里的解题过程自己是否支持最终答案”，把 majority-cluster process support 聚合成 prompt-level 容量上限。它仍然不用 Math500 标注、不改变 validation、不使用 best-of/major vote/n=32/answer selection；成功仍只看 50 step strict `mean@4 >= 85%`。

v48 已在 B200 worker `986717` 上完成 50 step strict n=4，训练退出状态 0。最终 strict `val-core/MATH-TTT/acc/mean@4=74.55%`，诊断项 `best@4=86.02%`、`maj@4=76.25%`。这里仍然只认 `mean@4`，`best@4` 不能作为达标指标；目标还没有完成。

v48 是目前 50 step strict 最好结果：相比 v44 的 `mean@4=73.84%` 提升约 `+0.70pp`，`best@4` 从 `84.26%` 提升到 `86.02%`。说明过程一致性信号确实带来候选存在性的提升，但还没有把概率质量稳定压到前 4 条平均样本上。

step 50 内部指标：`train_weight=0.684`、`answer_sharp_confidence=0.975`、`answer_effective_K=1.058`、`low_budget_capacity=0.750`、`low_budget_majority_mass=0.750`、`base_support_capacity=0.799`、`cross_view_capacity=0.752`、`process_consistency_capacity=0.988`、`process_majority_support=0.988`、`process_consistent_rate=0.953`、`process_majority_consistent_rate=0.777`、`process_box_conflict_rate=0.031`、`ground_truth_reward=0.797`。关键现象是 process capacity 到后期几乎饱和，所以它更像早期过滤器，不足以作为后期强 verifier。

infra 正常：preflight 健康，driver `580.105.08` 下清空 compat，`cuInit=0`，cuBLAS/cuDNN/NCCL/nvJitLink 来自 venv cu12.9；runner final `/proc/self`、`/proc/meminfo` 正常，`PROC_COUNT_FINAL=84`。最终 GPU 快照有 teardown 窗口残留显存，后续 live check 显示 `/proc` 仍健康；下一轮实验启动前仍需重新检查 GPU 占用。

结论：当前最好 strict 50-step 是 v48 `mean@4=74.55%`，但距离 85% 仍有明显差距。下一步重点不是继续验证“有没有正确候选”，因为 `best@4` 已超过 85%；而是训练期把候选存在性转成低预算平均正确率，避免 validation-time selection，同时避免 v46/v47 那种盲目修向 majority cluster 的错误簇放大。
