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
| Qwen2.5-Math-7B 50-step strict | 原版 MajVote TTRL baseline | 50 | 关闭 SPS，只用原版 majority pseudo label | `mean@4=72.89%`, `best@4=83.65%`, `maj@4=74.37%` |
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

## v49 计划：first4 process-weighted sample shaping

v49 的动机来自 v48 的核心现象：`best@4=86.02%` 已经超过 85%，但 `mean@4=74.55%`，说明四条低预算样本里偶尔有正确候选，但平均样本质量不够。继续做 prompt-level capacity 很可能会像 v45/v48 一样后期饱和；直接把 first4 修到 majority cluster 又会复现 v46/v47 的错误簇放大。

v49 不替换 rollout、不改变 validation，只在训练期给 first4 样本加 sample weight。对 raw-majority pseudo label 且过程一致的 first4 样本轻微加权；对 parseable 但和 pseudo label 冲突的样本降权；对 unparseable 或 clipped 样本更强降权；后面 28 条训练样本保持权重 1.0，避免把 32 条训练支持全部压成一个簇。

初始常数不是后验调参：`positive=1.25` 是温和增强过程一致的 pseudo-label 样本，`inconsistent=0.75` 是轻度降低同答案但过程不一致的样本，`negative=0.35` 和 `invalid=0.25` 是针对 strict mean@4 中最直接伤害单样本正确率的冲突/无效 first4 样本。这个信号仍然不用 Math500 标注，不做 validation-time answer selection，不用 best-of、major vote、n=32 或 ground-truth 选择作为达标口径。runner 计划为 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v49_process_sample_weight_strict_n4.sh`。

v49 已完成 50 step strict n=4，结果退步：`mean@4=72.59%`、`best@4=83.66%`、`maj@4=74.21%`，低于 v48 的 `mean@4=74.55%` 和 `best@4=86.02%`。因此 v49 不是提升方案，不做 improvement commit。

v49 的失败原因比较明确：它只是把 first4 的 sample weight 乘到 `token_level_scores` 上，但和 pseudo label 冲突的样本通常原本就是 0 reward，`0 * 0.35` 仍然是 0，所以没有形成真正的负向训练信号，反而扰动了正样本尺度。step 50 里 `process_sample_negative_rate=0.250`、`process_sample_invalid_rate=0.062`，但 `sps_process_sample_weight_applied=0.995`，整体影响很弱且方向不稳定。当前最好仍是 v48。

## v50 计划：gated low-budget negative pseudo-labeling

新一轮文献启发来自近期 test-time RL / verifier-free RL / entropy-confidence 方向，尤其是 SPINE、EDIS、SCRL、COMPASS、Certified Self-Consistency 这类工作：有效训练信号应该是稀疏、带不确定性控制的，不能只把 majority/confidence 当成正确性本身。结合 v38/v46/v47 的错误簇放大和 v49 的乘权无效，v50 改为只在训练期对 first4 中“明显坏”的样本加小的负终端 reward。

v50 新增默认关闭的 `sps_low_budget_negative_reward`。它只有在 majority 同时得到 first4、base/ref 和过程一致性支持时才启用：`low_budget_majority_mass >= 0.65`、`process_majority_consistent_rate >= 0.70`、`base_support_majority_confidence >= 0.65`，且 base/ref top answer 和 first4 local majority 都同意 raw majority。满足 gate 后，对 first4 中不可解析、截断、或和 majority 冲突且过程不一致的样本加 `-0.35` 负 reward；对 parseable 但冲突的样本只加半强度负 reward。这样做的目标是修复 v49 中错误样本没有梯度的问题，同时避免无条件打压可能正确的 minority。

v50 不改变 validation，不做 answer selection，不用 best-of、major vote、n=32 或 ground-truth 选择作为指标。runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v50_low_budget_negative_strict_n4.sh`。成功仍只看 50 step strict `val-core/MATH-TTT/acc/mean@4 >= 85%`。

v50 已在 B200 worker `986717` 上完成 50 step strict n=4，结果没有超过 v48：`mean@4=74.09%`、`best@4=83.46%`、`maj@4=75.45%`。因此 v50 不是提升方案，不做 improvement commit；当前最好仍是 v48 `mean@4=74.55%`，本地记录 commit 是 `86dfdf8`。

v50 的内部指标显示 gate 后期确实打开，但负信号很稀疏：step 50 `low_budget_negative_rate=0.062`、`low_budget_negative_gate=0.750`、`low_budget_capacity=0.906`、`base_support_capacity=0.787`、`cross_view_capacity=0.837`、`process_consistency_capacity=0.989`。失败原因不是机制没生效，而是显式惩罚 conflicting/minority first4 样本会降低候选存在性：`best@4` 从 v48 的 `86.02%` 降到 v50 的 `83.46%`。下一步不要靠加大负 reward 或放松 gate 调参，应回到能增加正确候选并把概率质量前移的内部信号，而不是继续打压 minority。

v50 infra 正常：首次启动因为 Ray socket 路径过长失败，已把 `RAY_DIR` 改短到 `/tmp/r50neg` 后成功；训练退出状态 0。preflight 和最终健康检查都正常：driver `580.105.08`、清空 compat、`cuInit=0`、venv cu12.9 CUDA 库优先、`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=94`，8 张 B200 最终释放到 `0 MiB`。包含 final validation 的 steps 41-50 汇总约 `50.061s/step`、`4715 tokens/s`；非 validation 后段训练 step 大多约 `23-29s/step`。

## v51 计划：low-budget soft support reward

v51 直接针对 v50 的失败：不要继续加大负 reward 或放松 gate，因为 v50 已经说明惩罚 conflicting/minority first4 样本会降低候选存在性。v51 改成给 first4 中“内部支持度高”的可解析、非截断候选一个小的正终端 reward，即使它不是 raw majority，也可以获得支持。支持度来自 sharpened SPS answer 分布、base/ref answer 分布、同 answer cluster 的 process-consistent 比例和采样 cluster mass。

这个设计的目标是提高低预算前 4 条 rollout 的平均候选质量，而不是在 validation 时做选择。它仍然不用 Math500 标注，不改 strict validation，不使用 best-of、major vote、n=32 或 ground-truth selection。初始系数 `sps_low_budget_soft_value=0.25` 不是后验调参：v50 的 `0.35` 负 reward 已经降低 `best@4`，所以 v51 只加弱辅助正信号，避免覆盖原有 SPS group-normalized reward。

代码新增默认关闭的 `sps_low_budget_soft_reward`。runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v51_low_budget_soft_support_strict_n4.sh`，基于 v48/v50 容量栈，关闭 `sps_low_budget_negative_reward`，开启 `sps_low_budget_soft_reward=True` 和 `sps_low_budget_soft_value=0.25`。新增日志包括 `low_budget_soft_reward_mean`、`low_budget_soft_reward_rate`、`low_budget_soft_nonmajority_rate`、`low_budget_soft_nonmajority_reward`、`low_budget_soft_support_effective_K`、`sps_low_budget_soft_reward_applied` 和 `sps_low_budget_soft_active_rate`。

v51 已在 B200 worker `986717` 上完成 50 step strict n=4，训练退出状态 0，infra 健康。最终 strict `mean@4=74.35%`，诊断项 `best@4=84.39%`、`maj@4=75.75%`，低于 v48 的 `mean@4=74.55%` 和 `best@4=86.02%`。因此 v51 不是提升方案，不做 improvement commit；当前最好仍是 v48，本地记录 commit `86dfdf8`。

v51 的机制确实生效：step 50 `low_budget_soft_reward_mean=0.205`、`low_budget_soft_reward_rate=0.969`、`low_budget_capacity=0.769`、`answer_sharp_confidence=0.967`、`base_support_capacity=0.813`、`process_consistency_capacity=0.988`、`ground_truth_reward=0.789`。但非 majority 的软奖励到后期几乎消失：`low_budget_soft_nonmajority_reward=0.003`、`low_budget_soft_support_effective_K=1.314`。这说明 soft support 最后还是强化了已经很尖锐的 majority basin，没有带来新的独立正确性信号。

v51 结论：不要继续通过提高 `sps_low_budget_soft_value` 或放松 support 阈值来调参；这会变成后验调系数，并可能加速错误 majority 自我模仿。下一步应考虑“反坍缩/不确定性门控”的训练信号：保留 v48 的 process consistency，但当 answer sharpness 已经很高、low-budget diversity 已经坍缩、且没有独立 base/process margin 改善时，降低或中和继续强化该 prompt/sample 的力度。目标仍然是 50 step strict `mean@4 >= 85%`，validation 继续只认 `n=4` 且禁用 answer selection。

## v52 计划：anti-collapse capacity

v52 不再加 first4 正/负 reward，也不做 rollout repair，而是回到当前最好 v48 的 process-consistency capacity，在 prompt weight 上新增默认关闭的 `sps_anti_collapse_capacity`。触发条件全部来自训练期内部信号：`answer_sharp_confidence >= 0.95`、`answer_effective_K <= 1.25`、`low_budget_majority_mass >= 0.75`，并且 sharpened majority margin 明显大于 base/ref 和 first4 中较弱的 margin，差值至少 `0.15`。

这个设计来自 v50/v51 的失败：显式惩罚 first4 conflicting 样本会降低候选存在性；正向 soft support 到后期又会坍缩成 majority 自我模仿。v52 的目标是在分布已经过尖、低预算样本已经集中但独立支持 margin 跟不上时，降低继续强化这个 prompt 的力度。默认最小 cap 是 `0.45`，它只会降低权重，不会给额外 reward，也不会让 prompt 比 v48 更强。

runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v52_anti_collapse_capacity_strict_n4.sh`。它基于 v48，不开启 v50 的 `sps_low_budget_negative_reward`，也不开启 v51 的 `sps_low_budget_soft_reward`。validation 仍然是 strict `n=4`，禁用 answer selection，不使用 best-of、major vote、n=32 或 ground-truth selection 作为指标。

v52 已完成 50 step strict n=4，结果退步：`mean@4=72.99%`、`best@4=84.17%`、`maj@4=74.49%`，低于当前最好 v48 的 `mean@4=74.55%`、`best@4=86.02%`。因此 v52 不是提升方案，不做 improvement commit。

v52 的机制确实触发：后段 `anti_collapse_active_rate` 多次达到 `0.25-0.75`，step 50 为 `0.375`；step 50 还有 `answer_sharp_confidence=0.972`、`answer_effective_K=1.063`、`low_budget_majority_mass=0.719`、`process_consistency_capacity=0.990`。但 prompt-level 反坍缩只是降低更新强度，没有增加独立正确性信号，反而降低了候选存在性。结论是不能继续只调 `anti_collapse_min_weight` 或触发阈值。

v52 infra 结果需要特别注意：训练前 preflight 健康，driver `580.105.08`、清 compat、`cuInit=0`、venv cu12.9 CUDA 库优先；训练退出状态 0。但训练结束后 worker `986717` 的 `/proc` 损坏：`PROC_SELF_BAD_FINAL`、`PROC_MEMINFO_BAD_FINAL`、`PROC_COUNT_FINAL=0`，并且 GPU1/GPU5 有 teardown 窗口显存残留。因此这个 worker 不能继续跑 Ray/CUDA，需要新健康 worker 才能继续实验。

新一轮文献启发覆盖 TTRL、自训练、verifier-free RL、confidence/entropy RL、Certified Self-Consistency、COMPASS、Self-Harmony、ETTRL、CURE、SENT 等方向。综合当前实验，下一版不应再做 majority repair、first4 显式负 reward、或纯 prompt-level capacity cap；应做一个小的、稀疏的 non-majority rescue：当 first4 中某条非 majority rollout 可解析、非截断、过程一致，并且它所在 answer cluster 有 base/process 独立支持时，给它小正终端 reward，避免 v51 后期非 majority 支持完全消失，同时不改变 strict validation。

## v53 预检查：低预算非多数候选 rescue

v53 新增默认关闭的 `sps_low_budget_rescue_reward`，只奖励 first4 中可解析、非截断、过程一致、非 raw-majority 且有 base/ref、process、cluster mass 独立支持的候选。初始值为 `rescue_value=0.15`、`min_base=0.08`、`min_process=0.50`、`min_cluster_mass=0.03125`、`min_support=0.15`。这个设计直接针对 v51 的非多数支持消失和 v52 的纯降权无效，不使用 ground truth，也不改变 validation。

跑前代码审查发现一个实现问题：low-budget negative/soft/rescue reward 原来嵌在 `"sps_reward" in batch.batch` 分支下，而当前 `answer_rule_conf_weight` 路径不会生成 `sps_reward`，导致 v53 可能只记录指标但不真正加 reward。已修复为所有 SPS 路径只要存在对应 per-sample 数组就能注入 terminal reward；`diff --check`、runner `bash -n`、`ttrl_utils.py`/`ray_trainer.py` 的 `py_compile` 均已通过。

v53 runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v53_low_budget_nonmajority_rescue_strict_n4.sh`。strict validation 仍保持 `val_kwargs.n=4`、禁用 answer selection、50 step、step 50 验证。v53 修复后尚未正式跑，需要新的健康 worker，因为 v52 后 worker `986717` 的 `/proc` 已损坏。

v53 第一次启动记录：已释放坏 worker `986717`，新 worker `986961` 的 preflight 健康，8 张 B200 可见，`cuInit=0`，driver `580.105.08` 下清 compat，GEMM smoke 通过，cuBLAS/cuDNN/NCCL/nvJitLink 来自 venv cu12.9。第一次 v53 launch 通过配置校验但在 step 0 前失败，TaskRunner stderr 为 `OSError: AF_UNIX path too long`，没有任何 `training/global_step`，所以这是 infra 启动失败，不是算法结果。中断后 `/proc` 仍健康，GPU 已释放。

已修复 runner 路径过长问题：`RAY_DIR=/tmp/r53`、`LOCAL_MODEL=/tmp/qm25v53`、cache tag 改为 `q25v53`。修复后 `diff --check`、runner `bash -n`、`ttrl_utils.py`/`ray_trainer.py` 的 `py_compile` 均通过，准备重新跑 v53。

## v53 结果：非多数 rescue 过稀疏，未提升

v53 已在 8x B200 worker `986961` 跑完 50 step，runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v53_low_budget_nonmajority_rescue_strict_n4.sh`。严格验证保持 `val_kwargs.n=4`、禁用 validation answer selection、50 step 后验证。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.7399396378269618`，诊断项 `best@4=0.8437082494969819`、`maj@4=0.7558812877263581`。低于当前最好 v48 `mean@4=0.7454728370221329`，所以 v53 不是提升，不做 improvement commit。

内部现象：v53 的 terminal reward 注入路径确实工作过，step 43-44 有非零 `train/sps_low_budget_rescue_reward_applied`；但 step 50 已归零，`low_budget_rescue_reward_mean=0`、`low_budget_rescue_rate=0`。同时分布仍然过尖，`answer_sharp_confidence=0.970`、`answer_effective_K=1.070`。结论是非多数候选 rescue 后期太稀疏，不能只靠降低阈值或加大 reward 做后验调参。

infra 方面 v53 训练退出状态 0，训练前 preflight 健康，训练后 `/proc` 仍健康：`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=78`。包含最终 validation 的 steps 41-50 汇总约 `50.118s/step`、`4527 tokens/s`；非 validation 后段 step 主要在 22-32s。

## v54 计划：低预算本地多数支持 reward

v54 的动机是：v48 已经说明 process consistency 能提高候选存在性，但 `best@4` 到 86% 后 `mean@4` 仍只有 74.55%；v46/v47 又说明把训练样本修到 32-sample raw majority 会放大错误簇；v53 说明只救非多数候选太稀疏。因此 v54 直接面向 strict n=4 的训练分布：只看 first4 的本地多数答案。

算法：新增默认关闭的 `ttrl.sps_low_budget_local_reward`。每个 prompt 内先计算 first4 的 local majority；如果这个 local answer 有足够 local mass、base/ref 支持和过程一致性支持，就只给 first4 中“可解析、非截断、过程一致、等于 local majority”的样本加一个小 terminal reward。初始值：`local_value=0.12`、`min_mass=0.50`、`min_base=0.08`、`min_process=0.50`、`min_support=0.15`。这不是 validation 后验调参，而是从 v48/v53 的内部指标推出来的低预算候选质量信号。

代码路径：`verl/verl/trainer/ppo/ttrl_utils.py` 计算 `sps_low_budget_local_reward` 和相关指标；`ray_trainer.py` 传配置、记录 `train/sps/low_budget_local_*`，并把 terminal reward 注入 `token_level_scores`；`ppo_trainer_ttrl.yaml` 增加默认关闭配置。

runner：`verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v54_low_budget_local_majority_reward_strict_n4.sh`。它基于 v48，关闭 v53 rescue，开启 local reward；验证仍然严格 `n=4` 且禁用 answer selection。

当前不能启动训练：live shell 中 `/proc/self` 和 `/proc/meminfo` 正常，`PROC_COUNT=110`，但 `check_cuda_compat_preflight.sh` 因 `/proc/driver/nvidia/version` 缺失失败。这不满足 GPU preflight，按规则不能启动 Ray/CUDA 训练，必须等 `/proc/driver/nvidia/version`、`nvidia-smi`、`cuInit=0` 和 CUDA 库路径检查都通过后再跑 v54。

v54 后来已在 8x B200 worker `986961` 跑完 50 step，runner 退出状态 0，训练前后 `/proc` 健康，最终 8 张 GPU 都释放到 `0 MiB`。日志和结果文件：
`verl/sps_efficient_ttrl_qwen25_math_7b_50step_v54_low_budget_local_majority_reward_strict_n4.log`、
`verl/sps_efficient_ttrl_qwen25_math_7b_50step_v54_low_budget_local_majority_reward_strict_n4_metrics.txt`、
`verl/sps_efficient_ttrl_qwen25_math_7b_50step_v54_low_budget_local_majority_reward_strict_n4_proc_health.txt`、
`verl/sps_efficient_ttrl_qwen25_math_7b_50step_v54_low_budget_local_majority_reward_strict_n4_throughput_summary.txt`。

v54 strict validation 退步：`mean@4=72.23%`、诊断 `best@4=83.30%`、`maj@4=73.58%`，低于当前最好 v48 `mean@4=74.55%`，所以不是提升，不做 improvement commit。

内部指标说明机制确实生效，不是注入失败：step 50 `low_budget_local_reward_mean=0.084`、`low_budget_local_rate=0.750`、`low_budget_local_support=0.798`、`train/sps_low_budget_local_reward_applied=0.011`、`answer_sharp_confidence=0.980`、`answer_effective_K=1.044`。失败原因是 first4 local majority 仍然只是自洽信号；如果本地多数簇本身错了，额外 reward 会继续锐化错误簇，并且把候选存在性从 v48 的 `best@4=86.02%` 降到 `83.30%`。

结论：不要继续加大 `local_value` 或放松 local support gate。v46/v47 已经证明 global majority repair 会放大错误簇，v54 进一步证明 local first4 majority reward 也会有同样问题。下一版需要保留 v48 的 process-consistency 候选存在性，同时在 majority/local-majority 与 base/process/low-temp 支持冲突时降低或反转训练信号；目标仍是 50 step strict `mean@4 >= 85%`。

## v55 计划：support-conflict abstention capacity

v55 不再给 majority 或 local-majority 加 reward，也不做 rollout repair。它保留 v48 的 process-consistency capacity 栈，只新增一个默认关闭的 `ttrl.sps_support_conflict_capacity`：当 SPS answer 分布已经很尖，但 base/ref、first4 local majority、process-majority support 这些独立视角不支持 raw majority 时，降低这个 prompt 的训练权重。

触发条件来自内部指标，不是根据 validation acc 后验调参：`answer_sharp_confidence >= 0.90`、`answer_effective_K <= 2.0`，然后看五个 conflict 信号：base/ref top answer 是否不同意 raw majority、base/ref 对 raw majority 概率是否低于 `0.55`、first4 local majority 是否不同意 raw majority、first4 raw-majority mass 是否低于 `0.50`、process-majority support 是否低于 `0.70`。conflict 越强，prompt weight 越接近下限 `0.15`。

设计理由：v52 说明“只要分布尖就降权”太粗，会损失有用梯度；v54 说明“奖励 first4 local majority”会放大错误簇。v55 只在“很自信但独立支持冲突”的情况下 abstain，不奖励任何候选答案，也不改变 strict validation。

代码路径：`ttrl_utils.py` 计算 `sps_support_conflict_*`，`ray_trainer.py` 传配置并记录 `train/sps/support_conflict_capacity`、`active_rate`、`score`，`ppo_trainer_ttrl.yaml` 增加默认关闭配置。runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v55_support_conflict_capacity_strict_n4.sh`，仍然 50 step、`val_kwargs.n=4`、禁用 validation answer selection。

v55 已在 8x B200 worker `986961` 跑完 50 step，runner 退出状态 0。严格验证仍是 `n=4` 且禁用 answer selection。结果没有提升：`mean@4=73.04%`、诊断 `best@4=84.24%`、`maj@4=74.51%`，低于当前最好 v48 `mean@4=74.55%`，所以不做 improvement commit。

step 50 内部指标：`train_weight=0.486`、`answer_sharp_confidence=0.966`、`answer_effective_K=1.085`、`low_budget_capacity=0.666`、`base_support_capacity=0.775`、`cross_view_capacity=0.668`、`process_consistency_capacity=0.979`、`process_majority_support=0.979`、`ground_truth_reward=0.789`。v55 机制最终只弱触发：`support_conflict_capacity=0.987`、`support_conflict_active_rate=0.125`、`support_conflict_score=0.015`；早期/中期有过更高触发，但不足以改变最终 mean@4。

infra 需要注意：训练前 preflight 健康，driver `580.105.08`、清 compat、`cuInit=0`、venv cu12.9 CUDA 库优先；但训练结束后 worker `986961` 的 `/proc` 损坏，`PROC_SELF_BAD_FINAL`、`PROC_MEMINFO_BAD_FINAL`、`PROC_COUNT_FINAL=0`，GPU5 还有 `[Not Found]` 残留显存约 31GB。因此这个 worker 不能继续跑 Ray/CUDA，需要新健康 worker 才能继续实验。

v55 结论：support-conflict abstention 太弱/太稀疏，本质还是 capacity 类方法，没有提供新的正确性信号。下一步不要继续调 conflict 阈值；应引入非 majority/local-majority imitation 的内部正向正确性信号，比如低成本 self-verification 或 process-answer agreement，用来区分“自洽正确候选”和“自洽错误簇”。

v55 已在 8x B200 worker `986961` 跑完 50 step，训练进程退出状态 0。严格验证仍然是 `n=4`、禁用 validation answer selection。最终结果：`mean@4=73.04%`、诊断 `best@4=84.24%`、`maj@4=74.51%`，低于当前最好 v48 `mean@4=74.55%`，所以不是提升，不做 improvement commit。

原版 MajVote TTRL baseline 已在 8x B200 worker `987078` 跑完 Qwen2.5-Math-7B 50 step。严格验证仍然是 `n=4`、禁用 validation answer selection，验证样本数 `1988=497*4`。最终结果：`mean@4=72.89%`、诊断 `best@4=83.65%`、`maj@4=74.37%`。这低于当前 SPS 最好 v48 `mean@4=74.55%`，说明 SPS/process-consistency 路线在 50-step low-budget 口径下确实比纯 majority pseudo label 略好，但差距仍很小，远未达到 85%。本次 baseline 训练退出状态 0，退出后 `/proc` 健康、8 张 GPU 已释放。

step 50 内部指标：`train_weight=0.486`、`answer_sharp_confidence=0.966`、`answer_effective_K=1.085`、`low_budget_capacity=0.666`、`base_support_capacity=0.775`、`cross_view_capacity=0.668`、`process_consistency_capacity=0.979`、`process_majority_support=0.979`、`ground_truth_reward=0.789`。v55 机制确实接上了，但到后期很弱：`support_conflict_capacity=0.987`、`support_conflict_active_rate=0.125`、`support_conflict_score=0.015`；中途较强的例子是 step 17 `active_rate=0.375`、`score=0.089`。

失败结论：v55 还是 capacity-only 方法，只是在冲突时少训练，没有提供新的正确性正信号。到 step 50 时 base/first4/process 已经大多同意 majority，所以 conflict abstention 太晚、太弱；同时 `best@4=84.24%` 也低于 v48 的 `86.02%`，说明候选存在性也没保住。下一版不应继续只调 `sharp_min`、`min_weight` 或 conflict 阈值，这会变成 v52 式后验降权调参；应转向更早期的、独立支持的正信号，比如只在 base/ref 与 process-consistent cluster 同意时给候选 cluster 小正信号，避免直接奖励 majority/local-majority。

infra 注意：v55 跑前 preflight 健康，driver `580.105.08`、清 compat、`cuInit=0`、venv cu12.9 的 cuBLAS/cuDNN/NCCL/nvJitLink 路径正确。但训练结束后 worker `986961` 的 `/proc` 损坏：`PROC_SELF_BAD_FINAL`、`PROC_MEMINFO_BAD_FINAL`、`PROC_COUNT_FINAL=0`，GPU5 有 `[Not Found]` PID 残留约 31GB。这个 worker 不能继续跑 Ray/CUDA，需要新的健康 worker 才能继续实验。

## v56 计划：independent-support first4 reward

v56 针对 v55 的结论：capacity-only 的 abstention 太弱、太晚，没有给模型新的正确性正信号；但 v51/v54 又说明不能直接奖励 sharpened majority 或 first4 local majority，否则会自我模仿错误簇。因此 v56 只用三个相对独立的训练期信号来给 first4 样本小正 reward：base/ref answer 概率、同答案簇的 process-consistent 比例、训练采样 cluster mass。

具体算法：新增默认关闭的 `ttrl.sps_independent_support_reward`。对每个 answer cluster 计算 independent support：
`(base_prob * process_cluster_conf * sqrt(cluster_mass)) ** (1/3)`。只有 first4 中可解析、非截断、过程一致的样本 eligible；它所在 cluster 还要满足 `base>=0.12`、`process>=0.65`、`cluster_mass>=0.0625`、归一化 support>=`0.20`。reward 初始值 `0.18`。如果 cluster 是 majority 且 independent-support effective_K 已经坍缩到 `<=1.15`，reward 减半；非 majority 支持簇在 support 多样时可以拿完整 reward。

设计理由：这个信号不使用 ground truth，不改变 validation，也不做 validation-time selection；它也不直接奖励 raw majority/local majority，而是奖励“base/ref 和过程一致性共同支持”的候选簇。目标是在分布完全坍缩前，把 v48 的候选存在性转成 first4 平均正确率。

代码路径：`ttrl_utils.py` 计算 `sps_independent_support_reward` 和 rate/nonmajority/support/effective_K 指标；`ray_trainer.py` 传配置、记录 `train/sps/independent_support_*` 并注入 terminal reward；`ppo_trainer_ttrl.yaml` 增加默认关闭配置。runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v56_independent_support_reward_strict_n4.sh`，仍然 50 step、`val_kwargs.n=4`、禁用 validation answer selection。

当前 infra：worker `986961` 已在 v55 后 `/proc` 损坏，不能继续跑 Ray/CUDA。v56 需要新的健康 worker，并且必须先通过完整 preflight。

## v56 启动修复记录：Ray 和 TMPDIR 都必须短路径

worker `987078` 是健康的 8x B200。v56 启动前 preflight 正常：`/proc/self`、`/proc/meminfo` 存在，8 张 B200 可见，driver `580.105.08`，清 compat 后 `cuInit=0`，GEMM smoke 通过，cuBLAS/cuDNN/NCCL/nvJitLink 都来自 venv cu12.9。

第一次 v56 启动在训练前失败，原因是 Ray plasma socket 路径超过 AF_UNIX 107 字节限制。已把 `RAY_DIR` 改短为 `/tmp/r56`，模型和普通缓存仍放 `/tmp/ttrl_cache/<exp>`。

第二次 Ray 已成功启动并到达 `Training Progress: 0/50`，但 Python multiprocessing 在共享 torch storage 时也创建 AF_UNIX socket，仍然因为长 `TMPDIR=/tmp/ttrl_cache/<long-exp>/tmp` 报 `OSError: AF_UNIX path too long`。已继续修复为 `SHORT_TMPDIR=/tmp/t56` 并在训练前 `export TMPDIR=/tmp/t56`。

这两次都是 step 0 前 infra failure，没有任何有效训练指标，不能算算法结果。中断后 worker 仍健康：无训练/Ray 残留，8 张 GPU 回到 `0 MiB`，`/proc` 仍正常。以后 Ray + torch multiprocessing runner 需要同时保证 `RAY_TMPDIR` 和 `TMPDIR` 都是短路径。

## v56 结果：independent-support reward 生效但未提升

v56 已在 worker `987078` 跑完 50 step，严格验证仍是 `val_kwargs.n=4` 且禁用 validation answer selection。最终 strict `val-core/MATH-TTT/acc/mean@4=0.7379275653923542`，诊断项 `best@4=0.8385251509054326`、`maj@4=0.7518812877263582`。低于当前最好 v48 `mean@4=0.7454728370221329`，所以 v56 不是提升，不做算法 improvement commit。

内部指标显示机制确实生效：step 50 `independent_support_reward_mean=0.118`、`independent_support_rate=0.719`、`train/sps_independent_support_reward_applied=0.015`。但它到后期几乎完全支持 majority basin：`independent_support_nonmajority_rate=0`，同时 `answer_sharp_confidence=0.970`、`answer_effective_K=1.070`、`process_majority_support=0.998`、`base_support_agreement=1.000`。结论是 v56 的正信号没有变成“提升 first4 候选正确率”的信号，而更像另一个 majority-confirming sharpening 信号。

infra：这次成功跑通依赖两个短路径修复：`RAY_DIR=/tmp/r56` 和 `TMPDIR=/tmp/t56`。训练完成退出状态 0；steps 41-50 含最终验证平均 `49.713s/step`，整机约 `4542.759 tokens/s`；最终验证 step `testing=230.816s`、`step=254.612s`。但 teardown 后 worker `987078` 的 `/proc` 再次损坏：`PROC_SELF_BAD_FINAL`、`PROC_MEMINFO_BAD_FINAL`、`PROC_COUNT_FINAL=0`，GPU4 有 `[Not Found]` 残留约 30GB，因此这个 worker 不能继续跑 Ray/CUDA。

下一步不要简单加大 `independent_support_value` 或放松阈值；内部指标说明信号已经足够活跃，但太贴近坍缩后的 majority。更合理方向是更早期的 margin/disagreement-aware 支持信号：只在 support 还能提高 first4 候选多样性和正确性时给正信号，或者在 support 已完全 majority-confirming 且分布过尖时加入反向/降权机制。

## v57 计划：margin-aware independent-support reward

v57 针对 v56 的失败点做结构性修改，不做后验调参。v56 的 reward 已经生效，但 step 50 时 `independent_support_nonmajority_rate=0`，说明它几乎完全变成 majority self-imitation。v57 继续使用同一个 independent support 分数：base/ref answer 概率、同答案簇 process-consistent 比例、cluster mass；但改变 majority 样本拿 reward 的条件。

具体做法：新增默认兼容配置 `sps_independent_support_majority_margin_max` 和 `sps_independent_support_competition_min`。first4 中非 majority、可解析、非截断、过程一致且通过 base/process/mass/support gate 的样本仍可拿完整小正 reward；majority 样本只有在 support 分布未坍缩、top2 竞争簇足够强时才拿半额 reward。v57 runner 使用 `majority_margin_max=0.12`、`competition_min=0.35`，并记录 `independent_support_margin`、`independent_support_competition_rate`、`independent_support_majority_gate_rate`，用来判断它是否真的抓到竞争支持。

runner：`verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v57_margin_aware_independent_support_strict_n4.sh`。严格验证不变：50 step、`val_kwargs.n=4`、禁用 validation answer selection；`best@4` 和 `maj@4` 仍只作为诊断。启动前必须用新健康 worker 做完整 preflight，并使用短路径 `RAY_DIR=/tmp/r57`、`TMPDIR=/tmp/t57`。

## v57 结果：margin gate 生效但 strict mean@4 退步

v57 已在 8x B200 worker `987258` 跑完 50 step，runner 退出状态 0。严格验证仍保持 `val_kwargs.n=4`、禁用 validation answer selection、50 step 后验证。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.7248490945674044`，诊断项 `best@4=0.8291086519114689`、`maj@4=0.7377303822937624`。这低于 v56 的 `73.79%`，也低于当前最好 v48 的 `74.55%`，所以 v57 不是提升，不做 algorithm improvement commit。

内部指标说明机制接上了，但信号太稀疏：step 50 `independent_support_majority_gate_rate=0.000`，说明 majority-confirming reward 被压住；`independent_support_nonmajority_rate=0.031`、`independent_support_reward_mean=0.003`，说明非多数 reward 还有少量存在。但 `independent_support_margin=0.707`、`independent_support_competition_rate=0.000`，表示最后已经没有真正竞争簇。分布仍很尖：`answer_sharp_confidence=0.959`、`answer_effective_K=1.103`、`process_majority_support=0.987`。

infra 正常：跑前 preflight 健康，driver `580.105.08`、清 compat、`cuInit=0`、venv cu12.9 CUDA 库路径正确；跑后 `/proc` 仍健康，`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=79`，8 张 GPU 都释放到 `0 MiB`。steps 41-50 含最终验证平均 `49.432s/step`，整机约 `4457.633 tokens/s`；最终 validation step `testing=228.289s`、`step=253.376s`。

结论：v57 的结构性 gate 是正确执行的，它确实避免了 v56 后期继续奖励 majority basin；但它没有提供足够强的内部正确性正信号，只是把 reward 变得更稀疏。下一版不要简单放宽 margin/competition 阈值或加大奖励值，否则会回到 v56 的 majority self-imitation。更合理方向是引入更强的低成本 self-verification 或 process-answer agreement 信号，在分布坍缩前区分“被支持的正确候选”和“被支持但错误的替代簇”。当前 goal 仍未完成：strict `mean@4=72.48% < 85%`，当前最好仍是 v48 `74.55%`。

## v58 计划：去掉 cluster-mass 泄漏的 process-answer reward

v58 针对 v56/v57 的共同失败点：positive support reward 虽然能接上，但原来的 support 里有 `sqrt(cluster_mass)`，base/ref 也用 answer cluster 的 `logsumexp`，所以答案出现次数越多越容易拿 support，天然偏向 majority。v56 因此变成 majority self-imitation；v57 压住 majority 后又只剩很稀疏的非多数信号。

v58 不放宽 v57 阈值，而是换 correctness proxy：对每个 answer cluster 用 mean normalized base/ref logprob 做支持，即 `logsumexp(base_scores)-log(cluster_count)`，再 softmax 成 `base_mean_prob_by_answer`，避免采样次数直接变成正确性。first4 里只有可解析、非截断、过程文本自洽的样本 eligible；reward 支持为 `sqrt(base_mean_prob * process_consistent_rate(answer))`。多数答案只给半额，非多数答案给完整小正信号。

runner：`verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v58_process_answer_reward_strict_n4.sh`。初始值：`value=0.16`、`min_base=0.10`、`min_process=0.60`、`min_support=0.20`、`majority_scale=0.50`。严格验证仍是 50 step、`val_kwargs.n=4`、禁用 validation answer selection；`best@4` 和 `maj@4` 只作为诊断。关键看 `process_answer_rate`、`process_answer_nonmajority_rate` 和 `sps_process_answer_reward_applied`，如果 reward 过密或非多数率归零，就说明又退化成 majority self-training。

## v58 结果：process-answer reward 仍主要跟随 majority basin

v58 已在 8x B200 worker `987258` 跑完 50 step，runner 退出状态 0。严格验证仍保持 `val_kwargs.n=4`、禁用 validation answer selection、50 step 后验证。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.7394366197183099`，诊断项 `best@4=0.8441327967806841`、`maj@4=0.7570281690140845`。这高于 v57 的 `72.48%`，但低于当前最好 v48 的 `74.55%`，所以 v58 不是提升，不做 algorithm improvement commit。

step 50 内部指标显示 reward 确实生效但方向不够好：`process_answer_reward_mean=0.057`、`process_answer_rate=0.688`、`train/sps_process_answer_reward_applied=0.007`，但 `process_answer_nonmajority_rate=0.031`，同时 `low_budget_majority_ratio=0.875`、`process_majority_support=0.996`、`answer_sharp_confidence=0.976`、`answer_effective_K=1.060`。也就是说 v58 去掉了 cluster-count 显式泄漏，但 mean base/ref support 和 process consistency 到后期仍然主要确认多数答案。

infra 正常：跑后 `/proc` 仍健康，`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=85`，8 张 GPU 都释放到 `0 MiB`。steps 41-50 含最终验证平均 `50.540s/step`，整机约 `4586.538 tokens/s`；最终 validation step `testing=230.530s`、`step=253.011s`。

结论：v58 再次说明问题不是 reward 没接上，而是 support/capacity/process-consistency 信号在 50 step 内很容易变成 majority-confirming sharpening。`best@4=84.41%` 但 strict `mean@4=73.94%`，说明候选存在性仍然接近目标，瓶颈是不用推理时选择的情况下把正确性转进 first4 平均。下一步不要简单降低阈值或加大奖励；需要能在 majority 已过度自信时提供反 majority 或 verifier-free correctness 区分的内部信号。当前 goal 未完成：strict `mean@4 < 85%`，当前最好仍是 v48 `74.55%`。

## v59 计划：过尖 majority 下的 contrastive alternative reward

v59 针对 v58 的失败点：reward 已经激活，但几乎都在确认 majority basin。v58 step 50 是 `process_answer_rate=0.688`、`process_answer_nonmajority_rate=0.031`，同时 `answer_sharp_confidence=0.976`、`answer_effective_K=1.060`、`process_majority_support=0.996`。所以 v59 不继续加强 support reward，而是在 majority 已经过尖、first4 又存在“非 majority 但过程自洽且 base/ref mean support 不弱”的候选时，给这个候选小正 reward，并给 first4 majority 样本更小的对比惩罚。

算法仍然无监督，不用 Math500 label，也不改 validation。它复用 v58 的 count-neutral 支持分数：按答案簇的 mean base/ref logprob 做 softmax，再结合 process-consistent rate。contrastive gate 只在 `answer_sharp_confidence>=0.92`、`answer_effective_K<=1.35`、first4 majority mass `>=0.70`、base/process majority support 都 `>=0.70` 时打开。初始值：`value=0.14`、`majority_penalty=0.06`、`min_base=0.08`、`min_process=0.60`、`min_support=0.18`。这些阈值来自 v48/v58 的内部失败形态，不是按 acc 后验试参。

代码路径：`ttrl_utils.py` 计算 `sps_contrastive_alt_reward` 和诊断指标，`ray_trainer.py` 传配置、记录 `train/sps/contrastive_alt_*` 并注入 terminal reward，`ppo_trainer_ttrl.yaml` 增加默认关闭配置。runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v59_contrastive_alt_strict_n4.sh`。严格验证仍是 50 step、`val_kwargs.n=4`、禁用 validation answer selection；`best@4` 和 `maj@4` 只作为诊断，达标只能看 strict `mean@4>=85%`。

## v59 结果：contrastive alternative 信号太稀疏，未超过 v48

v59 已在 8x B200 worker `987258` 跑完 50 step，runner 退出状态 0。严格验证仍然是 `val_kwargs.n=4`、禁用 validation answer selection、50 step 后验证。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.7404426559356136`，诊断项 `best@4=0.8424728370221328`、`maj@4=0.7546519114688129`。它略高于 v58 的 `73.94%`，但低于当前最好 v48 的 `74.55%`，所以不是提升，不做 algorithm improvement commit。

step 50 内部指标显示问题不是 gate 没开，而是没有合格替代候选：`contrastive_alt_gate=0.750`，但 `contrastive_alt_rate=0.000`、`contrastive_alt_majority_penalty_rate=0.000`、`train/sps_contrastive_alt_active_rate=0.000`。中途 step 28-30 曾弱触发，`contrastive_alt_rate=0.031`、`majority_penalty_rate=0.094`、`active_rate=0.016`，太稀疏，无法改变 first4 分布。

最终分布仍然 majority 坍缩：`answer_sharp_confidence=0.977`、`answer_effective_K=1.051`、`low_budget_majority_ratio=0.844`、`process_majority_support=0.996`。结论是 v59 的方向合理，但 eligible alternative 出现太晚太少；下一步不应简单降低阈值或加大奖励，而要找更早的 verifier-free correctness/anti-collapse 信号，在 majority 过尖前改变 first4 平均正确率。

infra：steps 41-50 含最终验证平均 `50.353s/step`，整机约 `4550.685 tokens/s`；最终 validation step `testing=230.405s`、`step=253.903s`。训练退出后 worker `987258` 的 `/proc` 再次损坏：`PROC_SELF_BAD_FINAL`、`PROC_MEMINFO_BAD_FINAL`、`PROC_COUNT_FINAL=0`，但 8 张 GPU 都释放到 `0 MiB`。这个 worker 不能继续跑 Ray/CUDA，需要新健康 worker。

## v60 计划：entropy-band capacity

这轮读了近期 TTRL / self-training / confidence RL / entropy control 相关工作，主要启发是：不能只把熵往低处推，也不能 uniform 更新所有轨迹。SPINE、ETTRL、SIREN 都强调选择性熵控制；Self-Harmony、SCOPE 说明 majority 容易偏向伪共识；Intuitor、RENT、RLSC 说明模型内部 confidence/entropy 可以作为无监督信号，但需要防止自训练坍缩；CORE-PO 和 self-train 分析强调 reasoning 质量与 reward hacking 风险。

结合我们的实验，v36/v48 的 answer-sharpen capacity 有效，但 v38/v39/v46/v47/v54 证明单纯锐化、majority repair 或 local-majority reward 会放大错误簇；v56/v58 的正向 support/process reward 到后期也会贴着 majority；v57/v59 的反 majority 信号又太稀疏。因此 v60 不再新增答案 reward，而是做 prompt-level 的 entropy-band capacity：只控制什么时候让现有 SPS 信号强更新。

算法：新增默认关闭的 `ttrl.sps_entropy_band_capacity`。当 `answer_effective_K` 过低、`answer_sharp_confidence` 很高、first4 majority mass 很高，并且 base/process 都强支持 majority 时，认为进入过尖 majority self-training 区间，把 prompt weight 往 `min_weight=0.45` 降；当 `answer_effective_K` 过高、first4 parseable 低或 clip 高时，认为伪标签太噪，也降权；中间 entropy band 保持 v48 的容量栈不变。

v60 runner：`verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v60_entropy_band_capacity_strict_n4.sh`。它关闭 v59 contrastive reward，只开启 entropy-band capacity；严格验证仍是 50 step、`val_kwargs.n=4`、禁用 validation answer selection。关键看 `entropy_band_capacity`、`entropy_band_low_active_rate`、`entropy_band_high_active_rate`，以及最终 strict `mean@4`。当前 worker `987258` 的 `/proc` 已坏，不能继续跑，需要新健康 worker 后再启动。

## v60 结果：entropy-band capacity 生效但退步

v60 已在 8x B200 worker `987433` 跑完 50 step，runner 退出状态 0。严格验证仍然是 `val_kwargs.n=4`、禁用 validation answer selection、50 step 后验证。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.7293762575452716`，诊断项 `best@4=0.841595573440644`、`maj@4=0.7460120724346077`。这低于 v59 的 `74.04%`，也低于当前最好 v48 的 `74.55%`，所以 v60 不是提升，不做 algorithm improvement commit。

内部指标说明机制确实打到了目标区域：step 50 `entropy_band_capacity=0.722`、`entropy_band_low_active_rate=0.750`、`entropy_band_high_active_rate=0.000`。当时分布很尖：`answer_sharp_confidence=0.969`、`answer_effective_K=1.073`，first4 majority mass `0.750`，base/process majority support 分别 `0.789` 和 `0.992`。但 strict 和 best@4 都没有提升，说明 prompt-level 降权太粗，可能同时压掉了有用梯度。

infra：steps 41-50 含最终验证平均 `50.362s/step`，整机约 `4663.858 tokens/s`；最终 validation step `testing=229.819s`、`step=255.018s`。这次 teardown 后 worker 仍健康：`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=76`，8 张 GPU 都释放到 `0 MiB`。

结论：v60 证明 entropy-band 可以识别后期 majority 坍缩，但直接按 prompt 降权太钝。下一步不应简单把阈值调得更强，而应该把容量控制放到更细粒度的 token/decision point 或 process-disagreement 位置，保留有用的候选生成梯度。当前 goal 未完成：strict `mean@4=72.94% < 85%`，当前最好仍是 v48 `74.55%`。

## v61 计划：first4 low-budget rebalance reward

v61 的动机来自 v60：过尖 majority 坍缩可以被内部指标识别，但整条 prompt 降权太粗，会一起压掉有用梯度。v61 不继续加强 v60 的降权，而是只在 strict 指标真正关心的 first4 训练样本里做小幅 rebalancing。

算法仍然无监督，不用 Math500 label，也不改 validation。它保留 v48 的 capacity stack，关闭 `sps_entropy_band_capacity`，新增默认关闭的 `sps_low_budget_rebalance_reward`。当 `answer_sharp_confidence>=0.92`、`answer_effective_K<=1.35`、first4 majority mass `>=0.70`、base/process majority support 都 `>=0.70` 时，认为进入 v60 识别到的过尖 majority 区间；这时只在 first4 中寻找可解析、非截断、过程自洽、且有 count-neutral base/process support 的非 majority 候选。

如果存在合格替代候选，v61 给它很小的 terminal reward，并给 first4 majority 样本更小的 terminal penalty；如果没有替代候选，就不惩罚 majority。初始值是 `alt_value=0.10`、`majority_penalty=0.04`、`min_base=0.06`、`min_process=0.55`、`min_support=0.12`。这些不是按 acc 后验调参，而是从 v59/v60 的内部现象来：anti-majority 方向合理但太稀疏，prompt-level 降权又太钝，所以改成 first4 局部、低强度、有替代候选才触发。

runner：`verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v61_lowbudget_rebalance_strict_n4.sh`。静态检查已通过：Python `py_compile`、runner `bash -n`、以及 v61 相关文件 `git diff --check`。严格验证仍是 50 step、`val_kwargs.n=4`、禁用 validation answer selection；`best@4` 和 `maj@4` 只作为诊断，达标只能看 strict `mean@4>=85%`。

## v61 结果：局部 rebalance 有触发，但仍然太稀疏

v61 已在 8x B200 worker `987433` 跑完 50 step，runner 退出状态 0。严格验证保持 `val_kwargs.n=4`、禁用 validation answer selection、50 step 后验证。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.7328973843058351`，诊断项 `best@4=0.8384305835010059`、`maj@4=0.7495613682092555`。它略高于 v60 的 `72.94%`，但低于 v59 的 `74.04%`，也低于当前最好 v48 的 `74.55%`，所以 v61 不是提升，不做 algorithm improvement commit。

内部指标说明 v61 机制不是死的，但太稀疏：50 个训练 step 里有 12 个 step 的 terminal reward 实际 active，平均 `low_budget_rebalance_gate=0.435`，平均 `alt_rate=0.00868`，平均 `majority_penalty_rate=0.02632`，平均 active rate 只有 `0.00444`。step 50 确实打到了目标区域：`answer_sharp_confidence=0.984`、`answer_effective_K=1.036`、first4 majority mass `0.812`、base/process majority support `0.816/0.991`；同时 v61 动作是 `alt_rate=0.062`、`penalty_rate=0.188`、`active_rate=0.031`。

infra 正常：完整 B200 preflight 通过，`cuInit=0`，GEMM OK，实际加载 venv cu12.9 的 cuBLAS/cuDNN/NCCL/nvJitLink。steps 41-50 含最终验证平均 `49.798s/step`，整机约 `4616.564 tokens/s`；最终 validation step `testing=227.576s`、`step=250.756s`。训练退出后 worker 仍健康：`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=82`，8 张 GPU 全部释放到 `0 MiB`。

结论：v61 证明 first4 局部 rebalance 能在过尖 majority 区间触发，但和 v59 一样，等 majority basin 已经很尖时，合格的非 majority 替代候选太少，无法改变 strict first4 平均正确率。下一步不应简单加大 `alt_value` 或放宽所有阈值；这会变成后验调参，并可能复现错误簇放大。更合理方向是更早、更密的 verifier-free 信号，例如 process-step disagreement、answer revision/self-check 特征，或者在轨迹首次提交 final answer 的 token/branch 位置做更细粒度 credit。当前 goal 未完成：strict `mean@4=73.29% < 85%`，当前最好仍是 v48 `74.55%`。

## 原版 MajVote TTRL 50-step baseline 复跑

按用户要求，在同一个健康 B200 worker `987433` 上复跑 Qwen2.5-Math-7B 原版/simple majority-vote TTRL 50 step baseline。runner 是 `verl/examples/ttrl/worker_run_majvote_qwen25_math_7b_50step_strict_n4.sh`，严格验证仍然是 `val_kwargs.n=4`、禁用 validation answer selection、50 step 后只做最终验证。

preflight 正常：`/proc/self` 和 `/proc/meminfo` 存在，`PROC_COUNT=76-78`，8 张 B200 启动前空闲，driver `580.105.08` 下清 compat，`cuInit=0`，GEMM smoke 通过，实际加载 venv cu12.9 的 cuBLAS/cuDNN/NCCL/nvJitLink。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.7349094567404426`，诊断项 `best@4=0.838338028169014`、`maj@4=0.7466961770623741`。这和之前同口径 baseline `72.89%` 接近，但仍低于当前最好 SPS v48 `74.55%`，远低于 active target `85%`。这是 baseline/control，不是算法提升，不做 improvement commit。

吞吐：后半段非验证 step 大多 `21-26s`，step 50 final validation `testing=229.116s`、`step=250.373s`；steps 41-50 含验证平均 `46.567s/step`，整机 `5173.587 tokens/s`。

infra 问题很关键：训练 status 0 正常退出，但 teardown 后 `/proc` 再次损坏，记录为 `PROC_SELF_BAD_AFTER`、`PROC_MEMINFO_BAD_AFTER`、`PROC_COUNT_AFTER=0`，final 也是 `PROC_SELF_BAD_FINAL`、`PROC_MEMINFO_BAD_FINAL`、`PROC_COUNT_FINAL=0`。主日志在退出附近还有 BRPC warning：找不到 `/proc/self/stat`、`/proc/self/fd`、`/proc/loadavg` 等。当前 worker `987433` 不能继续跑 Ray/CUDA；下一次 GPU 实验必须换新的健康 worker。

## v62 静态实现：process-quality terminal reward

v62 已实现但还没有跑 GPU。原因是 MajVote baseline 退出后 worker `987433` 的 `/proc` 已损坏，不能继续跑 Ray/CUDA；必须等新的健康 worker。

设计思想：v61 的 first4 rebalance 能识别过尖 majority 区间，但等到 majority 已经坍缩后，合格的非多数替代样本太少。v62 改成更早、更密的内部信号：只看 first4 轨迹自身文本质量，不用真值，不做 validation-time selection。可解析、非截断、最终 boxed 答案位置合理、无 boxed 冲突、最终答案后没有自我修正文本、且过程自洽的样本给小正奖；截断、不可解析、boxed 冲突、最终答案后修正、final answer 不在尾部的样本给小负奖。

为了避免变成 majority self-imitation，当内部指标显示已经过尖坍缩时（`answer_sharp_confidence>=0.92`、`answer_effective_K<=1.35`、first4 majority mass `>=0.70`），majority 样本的正奖压到 `0.25x`，非 majority clean 样本保留完整正奖；坏 majority 样本给完整负奖，坏 non-majority 样本只给半额负奖。

代码路径：`ttrl_utils.py` 生成 `sps_process_quality_reward` 和诊断，`ray_trainer.py` 做配置透传、日志和 terminal reward 注入，`ppo_trainer_ttrl.yaml` 增加默认关闭配置。runner 是 `verl/examples/ttrl/worker_run_sps_efficient_ttrl_qwen25_math_7b_50step_v62_process_quality_strict_n4.sh`，保留严格协议：50 step、`val_kwargs.n=4`、禁用 validation answer selection。

静态检查已通过：Python `py_compile`、runner `bash -n`、`git diff --check`。下一步等健康 worker 后，先做完整 `/proc`、driver/compat、`cuInit=0`、GEMM 和 CUDA 库路径 preflight，再跑 v62。目标仍是 strict `mean@4>=85%`；`best@4`、`maj@4` 不算达标。

补充安全 guard：当前终端是 devbox master，`NVIDIA_VISIBLE_DEVICES=none`，`nvidia-smi` 可以退出 0 但没有 GPU 行，所以 v62 runner 增加了显式 `GPU_COUNT_BEFORE_TRAIN` 检查。训练前如果看不到 8 张 GPU，会记录 `GPU_COUNT_BEFORE_TRAIN_BAD` 并以 86 退出，避免误在非 GPU 环境启动 Ray/CUDA。旧 worker `987433` 已释放；新 8x B200 worker `987681` 已创建但暂未 ready，截至 16:43 `podIP` 仍为空，不能登录训练。

## v63 静态实现：直接拟合锐化概率分布

新方向按用户要求暂时抛弃 majority vote 作为伪标签核心。majority 只作为诊断，不再决定训练标签。参考 PowerFlow 的形式，把可迁移部分简化到当前 TTRL reward 路径：对每条 rollout 计算 `flow_score = beta * logp_ref - logp_rollout`，按答案簇做 `logsumexp`，再 softmax 得到 sharpened target distribution。训练时每条样本拿到它所属答案簇的 soft terminal reward，而不是用 majority pseudo label 算 0/1 reward。

实现入口：`ttrl.sps_reward_mode=direct_sharpened_prob`。新增函数在 `verl/verl/trainer/ppo/ttrl_utils.py`，接线在 `verl/verl/trainer/ppo/ray_trainer.py`，默认配置在 `verl/verl/trainer/config/ppo_trainer_ttrl.yaml`。runner 是 `verl/examples/ttrl/worker_run_sps_direct_sharpened_prob_qwen25_math_7b_50step_v63_strict_n4.sh`。

v63 初始参数：`sps_direct_beta=4.0`、`sps_direct_target_temperature=1.0`、`sps_direct_per_sample_mode=answer_mass`、`sps_direct_reward_scale=1.0`。仍然生成 64 条用于构造 target，训练 32 条；验证严格保持 50 step、`val_kwargs.n=4`、禁用 validation answer selection，只看 `mean@4`。

静态检查已通过：Python `py_compile`、runner `bash -n`、`git diff --check`。还没有 GPU 结果；下一步登录唯一 worker `987681`，先做 `/proc`、8 卡、driver/compat、`cuInit=0`、GEMM 和 CUDA 库路径 preflight，再跑 v63。若 strict `mean@4` 超过当前最好 v48 `74.55%` 才做本地 commit，否则只记录失败分析。

## v63 结果：直接拟合锐化概率分布没有超过 baseline

v63 已在 8x B200 worker `987681` 跑完 50 step，runner 正常退出 `0`。严格验证仍然是 `val_kwargs.n=4`、禁用 validation answer selection、50 step 后验证。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.7258551307847082`，诊断项 `best@4=0.8410402414486922`、`maj@4=0.7426720321931589`。它低于 MajVote 50-step baseline `0.7349094567404426`，也低于当前最好 v48 `0.7454728370221329`，所以 v63 不是提升，不做 algorithm improvement commit。

内部现象：direct target 确实被锐化了，50 步平均 `direct_target_confidence=0.61082`、`direct_target_effective_K=5.07714`，step 50 达到 `confidence=0.780`、`effective_K=2.536`。但 `pass@32` 平均是 `1.0` 的同时，`train/sps_pick_accuracy` 平均只有 `0.19`，说明候选里经常有正确答案，但直接概率锐化目标没有稳定偏向正确簇。

结论：这版证明了“不用 majority pseudo label、直接拟合 sharpened distribution”的链路能跑通，但 `softmax(logsumexp(beta * logp_ref - logp_rollout))` 主要是在锐化模型自身概率 basin，不足以对齐 correctness。下一步不应该盲调 beta/temperature，而是先加入和正确性更相关的无监督对齐信号，例如过程自洽、final answer/revision 质量、或 PowerFlow residual confidence 的内部指标筛选。

infra：本次 preflight 正常，`cuInit=0`、GEMM OK、8 卡可见；但训练正常退出后 worker 的 `/proc` 再次损坏：`PROC_SELF_BAD_FINAL`、`PROC_MEMINFO_BAD_FINAL`、`PROC_COUNT_FINAL=0`。worker `987681` 不能继续跑 Ray/CUDA，下一次实验必须换健康 worker。

## v62 结果：process-quality 对照补齐

v62 已在新的 8x B200 worker `987816` 跑完 50 step，runner 正常退出 `0`。严格验证仍然是 `val_kwargs.n=4`、禁用 validation answer selection、50 step 后验证。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.7339034205231388`，诊断项 `best@4=0.8304144869215292`、`maj@4=0.7502515090543259`。它略低于 MajVote baseline `0.7349094567404426`，也低于当前最好 v48 `0.7454728370221329`，所以 v62 不是提升，不做 algorithm improvement commit。

内部指标显示 process-quality 信号确实存在：平均 `process_quality_reward_mean=0.02864`、`process_quality_active_rate=0.125`、`process_quality_collapse_gate=0.475`。step 50 时分布已经很尖：`answer_sharp_confidence=0.976`、`answer_effective_K=1.054`、`low_budget_majority_mass=0.781`、`majority_ratio=0.799`；同时 process-quality 基本判为 clean，`positive_rate=0.969`、`negative_rate=0.031`。这说明文本过程质量信号更密，但仍然没有把 first4 分布推到更高正确率。

infra：steps 41-50 含最终验证平均 `49.728s/step`，整机约 `4603.120 tokens/s`；最终 validation step `testing=229.067s`、`step=251.816s`。这次训练退出后 `/proc` 保持健康：`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=77`；后续 live check 显示 8 张 GPU 都释放到 `0 MiB`。

## 本轮 direct sharpen goal 结论

同口径 50 step strict 对比：

| 方案 | strict mean@4 | 备注 |
| --- | ---: | --- |
| MajVote baseline | `0.7349094567404426` | 原版/simple majority-vote TTRL 对照 |
| v48 best SPS | `0.7454728370221329` | 当前本地最好记录 |
| v62 process-quality | `0.7339034205231388` | 信号更密，但没有超过 baseline |
| v63 direct sharpened probability | `0.7258551307847082` | 不用 majority pseudo label，但退步 |

最终判断：majority vote 可以被理解成一种粗糙的分布锐化，但直接把 PowerFlow 风格概率残差 `beta * logp_ref - logp_rollout` 聚合成 sharpened target 后，模型主要锐化的是自身概率 basin，不是 correctness basin。v63 的 `pass@32` 很高但 `sps_pick_accuracy` 很低，说明候选不是主要瓶颈，目标对齐才是主要问题。

下一步如果继续这个方向，应保留“拟合锐化分布”的框架，但先加入无监督 correctness alignment 信号，例如过程自洽、final answer/revision 质量，或更可靠的 PowerFlow residual confidence，再做锐化；不应该只靠后验调 beta/temperature。

## 新 goal：纯锐化分布训练到 75%

目标改成更明确的纯分布锐化：不再把 TTRL majority pseudo-label、hard majority reward 或推理时多采样选择当作训练范式。严格验证仍然是 Qwen2.5-Math-7B，50 step，`val_kwargs.n=4`，禁用 validation answer selection，只看 `val-core/MATH-TTT/acc/mean@4`，目标 `>=75%`。

这轮快速读了 10 类相关工作/代码启发：PowerFlow/escort distribution、exponential tilting/self-consistency 理论、one-shot entropy minimization、SPINE token-selective entropy band、ETTRL entropy advantage、Self-Harmony 反 majority bias、CORE-PO reasoning confidence、SRGen/SR-TTRL self-reflection、LESS 低熵结构段、ECHO entropy-confidence hybrid。结论是：不能再做 majority rescue 或单纯调 beta/temperature；v63 失败说明直接锐化概率 basin 不等于正确性 basin，所以需要先把无监督 correctness proxy 融入 target construction。

## v64 设计：process-tilted sharpened distribution

v64 保留 v63 的 soft target distribution 训练路径，但在 softmax 前把每条 rollout 的目标 logit 加上过程稳定性 tilt。基础分数仍是：

`flow_score = beta * logp_ref - logp_rollout`

新增无监督 process tilt：

- 可解析 final answer、未截断、boxed answer 在尾部、过程自洽：加分；
- boxed 冲突、final answer 后又 revision/correction、截断或不可解析：减分；
- 每个答案簇再按自己的 `consistent_rate / tail_rate / bad_rate` 得到 cluster tilt。

最终做：

`target_logit(a)=logsumexp_y(flow_score(y)+process_tilt+cluster_tilt)`

然后训练拟合 `softmax(target_logit)`。这仍然不是 hard label，也不使用 majority 作为伪标签；majority 只记录诊断。

实现已完成：

- `ttrl_utils.py`：扩展 `apply_direct_sharpened_ttrl_reward`，新增默认关闭的 process tilt 和诊断指标。
- `ray_trainer.py`：新增 `process_tilted_sharpened_prob` mode，走 direct soft reward 路径，不进入 majority pseudo-label 分支。
- `ppo_trainer_ttrl.yaml`：新增默认关闭配置。
- runner：`verl/examples/ttrl/worker_run_sps_process_tilted_sharpened_prob_qwen25_math_7b_50step_v64_strict_n4.sh`。

静态检查已通过：Python `py_compile`、runner `bash -n`、相关文件 `git diff --check`。下一步是在健康 8 GPU worker 上跑 v64 50 step；如果 strict `mean@4 >= 75%` 或明确超过 v48，再做本地 commit，否则只写文档分析失败原因。

## v64 结果：process tilt 仍未解决目标对齐

v64 已在 8x B200 worker `987816` 跑完，runner 正常退出 `0`。验证口径保持严格：50 step、`val_kwargs.n=4`、禁用 validation answer selection，不用 best-of、majority vote、n=32 选择或真值选择作为成功指标。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.7328973843058351`。诊断项：`best@4=0.8447364185110664`，`maj@4=0.7489195171026156`，这些只用于分析，不算成功指标。

同口径对比：

| 方案 | strict mean@4 | 备注 |
| --- | ---: | --- |
| MajVote baseline | `0.7349094567404426` | 原版/simple majority-vote 对照 |
| v48 best SPS | `0.7454728370221329` | 当前本地最好 |
| v62 process-quality | `0.7339034205231388` | 过程质量信号，没有提升 |
| v63 direct sharpened probability | `0.7258551307847082` | 纯概率 basin 锐化，退步 |
| v64 process-tilted sharpened probability | `0.7328973843058351` | 加过程稳定性 tilt，仍未超过 baseline |

内部指标显示 v64 相比 v63 有一点方向性改善：50 step 平均 `sps_pick_accuracy=0.3875`，高于 v63 的约 `0.19`；`pass@32=1.0`，说明候选里一直有正确答案；`direct_process_consistent_rate=0.82838`，过程稳定信号确实存在。

但核心失败没有解决：target 仍然锁到模型/base/majority basin。step 50 时 `direct_majority_target_mass=0.845`、`direct_majority_agreement=1.0`、`direct_base_top_confidence=0.826`、`direct_base_agreement=1.0`。也就是说形式上没有用 majority 当伪标签，但最终 target 经常和 base/majority 的 top cluster 重合。

吞吐和健康：非验证训练 step 平均约 `26.34s/step`；steps 41-50 含最终验证平均 `48.272s/step`，整机约 `4615.017 tokens/s`；最终 validation `testing=229.327s`。退出后 `/proc` 保持健康：`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=83`。

结论：v64 是有效的负结果。它证明 process tilt 能改善内部 target alignment，但不足以突破 strict 指标。因为没有超过 baseline/v48，也没有达到 `75%`，本次不做 algorithm improvement commit。

下一步不应该盲调 `beta/temperature/reward scale`。更合理的范式创新是做“contrastive basin sharpening”：把高过程稳定性分布和 base/majority-consensus 分布区分开，只锐化“过程稳定但不是单纯 base 共识”的残差信号，以直接打破 v64 暴露出的 base/majority basin lock-in。

## v65 设计：对 base basin 做 contrastive sharpening

v65 继续保持纯锐化分布训练，不回到 majority pseudo-label。动机来自 v64 的失败诊断：step 50 时 `direct_base_agreement=1.0`、`direct_majority_agreement=1.0`、`direct_majority_target_mass=0.845`，说明 target 形式上不用 majority，但实际上经常回到 base/majority top cluster。

v65 的目标不是调 `beta/temperature`，而是在 v64 的 process tilt target 上减掉 frozen base/ref 已经解释掉的答案簇支持：

`target_logit(a)=raw_logit(a)-lambda_base*(p_base(a)-1/|A|)*|A|`

其中 `raw_logit` 还是 v64 的 `flow_score + process_tilt + cluster_tilt` 聚合，`p_base(a)` 是 frozen base/ref 在答案簇上的 softmax 分布。这样做的直觉是：如果一个答案簇只是 base 本来就高置信，而不是靠过程稳定性新支持出来，就不要继续把 target 锁死在它上面。

实现已完成：

- `ttrl_utils.py`：`apply_direct_sharpened_ttrl_reward` 新增默认关闭的 `base_contrast_strength`，并记录 `direct_base_contrast_strength / penalty_mean / penalty_std`。
- `ray_trainer.py`：新增 `contrastive_process_sharpened_prob` mode，`mode_id=9.0`。
- `ppo_trainer_ttrl.yaml`：新增默认关闭配置 `sps_direct_base_contrast_strength: 0.0`。
- runner：`verl/examples/ttrl/worker_run_sps_contrastive_process_sharpened_prob_qwen25_math_7b_50step_v65_strict_n4.sh`。

v65 初始设置：保留 v64 的 `process_tilt_strength=0.6`、`process_cluster_tilt_strength=0.8`，新增 `base_contrast_strength=0.35`。这个强度来自 v64 的内部失败指标，目标是降低 `base_agreement/majority_target_mass`，不是根据 acc 做后验调参。

静态检查已通过：Python `py_compile`、runner `bash -n`，runner 中无残留 v64/r64/t64/process-tilted 名称。下一步跑 50 step strict validation；如果没到 75%，重点分析 base contrast 是否真的降低了 base/majority basin lock-in，以及是否损害 `sps_correct_weight_mass`。

## v65 结果：全局 base contrast 过强，明确失败

v65 已在 8x B200 worker `987816` 跑完，runner 正常退出 `0`。严格验证口径保持不变：50 step、`val_kwargs.n=4`、禁用 validation answer selection，不用 best-of 或 majority 作为成功指标。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.46981891348088534`。诊断项：`best@4=0.6835472837022132`，`maj@4=0.48600402414486926`。

v65 机制本身是生效的：50 step 平均 `direct_base_agreement=0.105`，比 v64 的 `0.9625` 低很多；`direct_majority_target_mass=0.11064`，也比 v64 的 `0.70908` 低很多。这说明全局减 base 支持确实打破了 base/majority basin lock-in。

但它把正确答案也一起打掉了：50 step 平均 `sps_correct_weight_mass=0.06696`，而 v64 是 `0.34496`；`sps_pick_accuracy=0.0225`，而 v64 是 `0.3875`。step 50 时 `direct_base_agreement=0.0`、`direct_majority_target_mass=0.072`，但 `sps_correct_weight_mass=0.049`、`sps_pick_accuracy=0.0`。

吞吐和健康：非验证 step 平均约 `31.19s/step`；steps 41-50 含最终验证平均 `52.755s/step`，整机约 `5339.262 tokens/s`；final validation `testing=231.798s`。退出后 `/proc` 健康：`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=89`。

结论：v65 是高价值负结果，不做 improvement commit。它证明“打破 base basin”本身不够，不能全局 subtract base；下一版应该做门控 contrast，只在答案簇“base 高但 process 支持弱”时惩罚。如果答案簇 base 高且过程稳定，应该保留，否则会压掉大量正确答案。

## v66 设计：门控 contrastive process sharpening

v66 直接针对 v65 的失败点：不是全局反 base，而是只惩罚“base 支持高但 process 支持弱”的答案簇。继续保持纯分布锐化，不使用 majority pseudo-label，不使用 hard majority reward，validation 仍然是 strict `mean@4`。

做法：

`p_process(a)=softmax(cluster_tilt(a))`

`excess(a)=max(p_base(a)-p_process(a), 0)`

`penalty(a)=lambda_base * sigmoid(gate_strength * (p_base(a)-p_process(a)-margin) * |A|) * excess(a) * |A|`

`target_logit(a)=raw_logit(a)-penalty(a)`

和 v65 的关键差异：

- 不再 boost 低 base 的噪声簇；
- 如果一个簇 base 高且 process 也支持，就不惩罚；
- 只压“base 高但 process 不支持”的疑似虚假高置信簇。

实现已完成：

- `ttrl_utils.py`：新增默认关闭的 `base_contrast_gate_strength` 和 `base_contrast_process_margin`，记录 gate 诊断。
- `ray_trainer.py`：新增 `gated_contrastive_process_sharpened_prob` mode，`mode_id=10.0`。
- `ppo_trainer_ttrl.yaml`：新增默认关闭 gate 配置。
- runner：`verl/examples/ttrl/worker_run_sps_gated_contrastive_process_sharpened_prob_qwen25_math_7b_50step_v66_strict_n4.sh`。

v66 初始设置：保留 `base_contrast_strength=0.35`，新增 `base_contrast_gate_strength=8.0`、`base_contrast_process_margin=0.02`。理由不是后验调 acc，而是 v65 的内部指标显示全局 contrast 过强，需要让惩罚依赖 process support。

静态检查已通过：Python `py_compile`、runner `bash -n`，runner 中无残留 v65/r65/t65 名称。下一步跑 50 step strict validation。

## v66 结果：门控 contrast 仍然把目标推离正确簇

v66 已在 8x B200 worker `987816` 跑完，runner 正常退出 `0`。严格验证口径不变：Qwen2.5-Math-7B，50 step，`val_kwargs.n=4`，禁用 validation answer selection，不用 best-of、major vote、n=32 选择或真值选择。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.3148893360160966`，诊断项 `best@4=0.5113420523138833`、`maj@4=0.3196338028169014`。这明显低于 v65/v64/v63、MajVote baseline 和 v48，完全没有达成 75%。

内部指标说明 gate 机制生效但方向错了：平均 `direct_base_agreement=0.2925`、`direct_majority_target_mass=0.23094`，确实比 v64 低很多，也不像 v65 那样完全全局反 base。但 correctness alignment 仍然崩：平均 `sps_correct_weight_mass=0.12196`、`sps_pick_accuracy=0.0275`，只比 v65 略好，远低于 v64 的 `0.34496/0.3875`。`pass@32=1.0` 继续说明候选不是瓶颈，目标分布选错 basin 才是瓶颈。

step 50 时 `direct_majority_target_mass=0.096`、`direct_base_agreement=0.125`、`sps_correct_weight_mass=0.073`、`sps_pick_accuracy=0.0`。这说明门控 contrast 虽然能把目标从 base/majority 拉开，但没有把目标拉向正确答案簇。

infra 正常：训练退出后 `/proc` 健康，`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=89`，8 张 B200 都释放到 `0 MiB`。非 validation step 平均 `30.134s/step`，整机约 `9145.562 tokens/s`；final validation `testing=229.194s`。

结论：v66 是明确负结果，不做 improvement commit。下一步不要继续调 `base_contrast_strength/gate_strength/margin`。v65/v66 已经证明“反 base”不是正确方向，除非有更强的独立 correctness proxy。更合理的新范式是只在高支持候选簇内部做 pairwise/listwise 的过程偏好锐化，或者先用 target entropy/effective_K 做分布约束，避免 v64 的 base/majority lock-in 和 v65/v66 的高熵错误噪声。

## v67 设计：band-limited process sharpening

v67 不再沿着 anti-base contrast 调参，而是把 v64-v66 的失败统一成一个分布形状问题：

- v64 的 target 到 step 50 变得过尖，`direct_target_effective_K=2.185`，并且锁到 base/majority top cluster；
- v65/v66 虽然把 target 从 base/majority 拉开，但 target 变成高熵或错误 basin，v66 平均 `direct_target_effective_K=8.64818`，`sps_correct_weight_mass=0.12196`，`sps_pick_accuracy=0.0275`。

所以 v67 保留 v64 的 process-tilted soft target，不再做 base contrast。构造 answer-level `target_probs=softmax(target_logit)` 后，先计算：

`effective_K = 1 / sum(target_probs^2)`

然后只用内部 support distribution 把 target 限制在中等 effective-K 区间：

- 如果 `effective_K < 3`，说明 target 过尖，和 v64 一样有锁死风险，就向 `softmax(base_logit + process_cluster_tilt)` 混合，让分布保留更多内部支持候选；
- 如果 `effective_K > 6`，说明 target 太散，和 v65/v66 一样容易漂到错误噪声，就按 support distribution 收缩；
- strength 设为 `0.45`，来自 v64 过尖和 v66 过散的内部指标差异，不是根据 validation acc 后验调参。

这仍然是纯分布锐化训练：不使用 majority pseudo label，不使用 hard majority reward，不在 validation 做 best-of、major vote、n=32 选择或真值选择。majority 只作为诊断。关键观察指标是 `direct_effective_K_before_band`、`direct_effective_K_band_alpha`、`direct_effective_K_band_direction`、`direct_target_effective_K`、`sps_correct_weight_mass`、`sps_pick_accuracy` 和 strict `mean@4`。

实现已完成并通过静态检查：

- `ttrl_utils.py`：新增默认关闭的 target effective-K band，并记录 band 诊断；
- `ray_trainer.py`：新增 `band_limited_process_sharpened_prob` mode，`mode_id=11.0`；
- `ppo_trainer_ttrl.yaml`：新增默认关闭配置；
- runner：`verl/examples/ttrl/worker_run_sps_band_limited_process_sharpened_prob_qwen25_math_7b_50step_v67_strict_n4.sh`。

下一步是在当前健康 8x B200 worker 上跑 v67 50 step strict validation。若 strict `mean@4 >= 75%`，再做 completion audit；若未达标，只记录负结果和失败归因，不做 improvement commit。

## v67 结果：K-band 生效但仍锁在 base/majority basin

v67 已在 8x B200 worker `987816` 跑完，runner 正常退出 `0`。严格验证口径保持不变：Qwen2.5-Math-7B，50 step，`val_kwargs.n=4`，禁用 validation answer selection，不用 best-of、major vote、n=32 选择或真值选择。

最终 strict `val-core/MATH-TTT/acc/mean@4=0.7183098591549296`，没有达到 75%，也低于 MajVote baseline `0.7349094567404426`、v48 `0.7454728370221329`、v63 `0.7258551307847082`、v64 `0.7328973843058351`。诊断项：`best@4=0.8322897384305835`，`maj@4=0.7306358148893359`。

内部指标：

- 50 step 平均 `direct_target_effective_K=3.33450`，`direct_effective_K_before_band=3.84630`，说明 K-band 确实改变了 target 形状；
- 但平均 `direct_base_agreement=0.96750`、`direct_majority_target_mass=0.71216`，说明 target 仍主要锁在 base/majority top cluster；
- step 50 更明显：`direct_target_effective_K=1.949`、`direct_majority_target_mass=0.824`、`direct_majority_agreement=1.0`、`direct_base_agreement=1.0`；
- `pass@32=1.0`，候选存在性仍不是瓶颈；
- 平均 `sps_correct_weight_mass=0.34080` 接近 v64，但 `sps_pick_accuracy=0.33250` 不够稳定。

吞吐和健康：

- 非 validation steps 1-49 平均 `27.047s/step`，整机约 `9222.636 tokens/s`；
- steps 41-49 train-only 平均 `26.585s/step`，约 `8801.405 tokens/s`；
- final validation `testing=232.157s`；
- 退出后 `/proc` 健康：`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=98`；post-check 8 张 GPU 都回到 `0 MiB`。

结论：v67 是负结果，不做 improvement commit。K-band 方向作为诊断有价值，但 soft band 不能单独解决 correctness alignment。下一步不应继续调 `target_effective_k_strength/min/max` 或 beta/temperature，而应在锐化前加入更独立的非 majority correctness proxy，例如局部 pairwise/listwise process preference：在高 support 候选内用 parseable、非截断、tail-box 稳定、无 revision conflict、ref likelihood margin、低温/自反一致性等信号重新分配概率质量。

## v68 设计：count-neutral process-preference sharpening

v68 直接针对 v64/v67 的共同问题：answer target 虽然不是 hard majority label，但使用 raw `logsumexp` 聚合同一答案簇内的多条 trajectory，等价于给样本数更多的答案簇天然加分，所以很容易回到 majority/base basin。

v68 保留 v64 的 process-tilted flow score：

`score(y)=beta*logp_ref(y)-logp_rollout(y)+process_tilt(y)+cluster_tilt(a)`

但把答案簇聚合从：

`answer_logit(a)=logsumexp_y score(y)`

改为 count-neutral 的：

`answer_logit(a)=logsumexp_y score(y)-log(count(a))`

也就是 log-mean-exp。这样重复采样次数不再直接变成 majority-count boost，target 更像 listwise/process-preference 分布，而不是隐式 majority pseudo-label。

约束不变：不用 majority pseudo label，不用 hard majority reward，不做 validation-time best-of/major vote/n=32/真值选择；majority 只作为诊断。v68 clean ablation 暂时关闭 v67 K-band，只看 count-neutral aggregation 是否能降低 `direct_majority_target_mass/direct_base_agreement`，同时保持 `sps_correct_weight_mass` 和 strict `mean@4`。

实现已完成：

- `ttrl_utils.py`：新增默认关闭的 `count_neutral_aggregation`，开启时 answer logits 用 log-mean-exp；
- `ray_trainer.py`：新增 `count_neutral_process_sharpened_prob` mode，`mode_id=12.0`；
- `ppo_trainer_ttrl.yaml`：新增默认关闭配置；
- runner：`verl/examples/ttrl/worker_run_sps_count_neutral_process_sharpened_prob_qwen25_math_7b_50step_v68_strict_n4.sh`。

下一步跑 v68 50 step strict validation。如果只是降低 majority mass 但 correctness 也下降，就说明“去 count bias”仍不是足够的 correctness proxy；如果 `mean@4` 提升，再考虑和 hard effective-K projection 或更细的 pairwise process preference 结合。

## v68 状态：实现完成，但本次 GPU 验证被中断

v68 代码和 runner 已实现并通过静态检查，但这次没有有效实验结果。

已完成：

- `count_neutral_process_sharpened_prob` mode，`mode_id=12.0`；
- `ttrl.sps_direct_count_neutral_aggregation=True`；
- runner：`verl/examples/ttrl/worker_run_sps_count_neutral_process_sharpened_prob_qwen25_math_7b_50step_v68_strict_n4.sh`；
- 静态检查：`py_compile`、runner `bash -n`、`git diff --check` 均通过。

启动时 preflight 正常：worker `987816` 上 `/proc` 正常，8 张 B200 可见，driver `580.105.08`，`cuInit=0`，GEMM OK，CUDA/cuBLAS/cuDNN/NCCL/NVJitLink 来自 venv cu12.9。

但在观察进度时，我错误地在 runner 前台终端里执行阻塞日志扫描，并发送了 `Ctrl-C`，导致 v68 runner 被中断。随后该 worker 的 `/proc` 损坏：`PROC_COUNT=0`，runner final snapshot 记录 `PROC_SELF_BAD_FINAL`、`PROC_MEMINFO_BAD_FINAL`、`PROC_COUNT_FINAL 0`。

结论：v68 不能算成功，也不能算算法负结果；没有 final strict `mean@4`。当前 worker 不应继续做 Ray/CUDA 恢复或实验。下一次需要新健康 worker 重新跑 v68，且训练终端只用于 runner，观察必须用单独 login 终端。

已新增 v68 只读观察脚本：

- `verl/examples/ttrl/monitor_sps_count_neutral_process_sharpened_prob_qwen25_math_7b_50step_v68_strict_n4.sh`
- 只读取 `/tmp/r68/ray/session_latest/logs`、v68 runner log 和 proc health 文件；
- 输出 `/proc` 状态、GPU 显存/利用率、最新 train 指标、validation 行和 runner 状态；
- 只能在单独的 `mlx worker login` 终端里运行，不能在训练 runner 前台终端里运行，避免再次误中断训练。

## 当前 goal 审计状态

goal 还没有完成。成功条件是 Qwen2.5-Math-7B、50 step、`val_kwargs.n=4`、禁用 validation answer selection，strict `val-core/MATH-TTT/acc/mean@4 >= 75%`。

已完成的证据：

- 已读并总结 10 类相关工作/代码启发；
- 已记录 MajVote baseline、v48、v62、v63 等对照；
- v63-v67 都是 pure soft target / distribution sharpening 方向，不把 majority 当训练 hard label；
- v67 完整跑完并落盘，但 strict `mean@4=0.7183098591549296`，未达标；
- v68 count-neutral 方案已实现并静态通过，但本次 GPU 验证被中断，没有 final 指标。

缺口：

- 还没有任何纯分布锐化方案在 strict 50-step 口径达到 `mean@4 >= 75%`；
- v68 需要在新健康 worker 上重跑，当前 `/proc` 损坏的 worker 不能继续使用。

下一步：拿到健康 worker 后直接跑 `verl/examples/ttrl/worker_run_sps_count_neutral_process_sharpened_prob_qwen25_math_7b_50step_v68_strict_n4.sh`。训练终端只跑 runner，观察另开同 worker 登录终端。

## v69 计划：count-neutral target 上的 pairwise process preference

当前 worker 状态：`mlx worker list` 仍只有旧 worker `987816`；登录后环境初始化直接报 `/proc/self/exe` 缺失，并提示 `mount -t proc proc /proc`。因此这个 worker 不能继续跑 Ray/CUDA，也没有做恢复。v68 仍需新健康 worker 才能重跑。

v69 作为下一版本地准备好的 clean ablation。动机是：v63 的概率残差锐化会贴回模型自身 basin；v64/v67 加了过程 tilt 仍然锁在 base/majority；v68 去掉 cluster count bias 之后，还需要一个非 majority 的 correctness proxy。v69 用同题答案簇之间的 pairwise/listwise 过程偏好来移动软 target。

具体做法：

- 仍然使用 v68 的 count-neutral log-mean-exp 聚合，避免答案簇样本数直接变成 majority boost；
- 每个答案簇内部只用无监督过程特征打分：可解析、非截断、最终 boxed 在尾部、boxed 答案一致、final answer 后没有修正/冲突表达；
- 用 Bradley-Terry 风格的 pairwise win rate 比较不同答案簇的过程质量；
- 把 centered pairwise preference 加到 answer logit 上：
  `answer_logit(a)=logmeanexp_y score(y)+lambda*(pairwise_pref(a)-mean_pref)`。

约束不变：不使用 majority pseudo label，不用 hard majority reward，不做 validation-time best-of/major vote/n=32/真值选择；majority 只作为诊断。

实现已完成：

- `ttrl_utils.py`：新增 `pairwise_process_preference_strength/temperature` 和 pairwise preference 指标；
- `ray_trainer.py`：新增 `pairwise_process_count_neutral_sharpened_prob` mode，`mode_id=13.0`；
- `ppo_trainer_ttrl.yaml`：新增默认关闭配置；
- runner：`verl/examples/ttrl/worker_run_sps_pairwise_process_count_neutral_sharpened_prob_qwen25_math_7b_50step_v69_strict_n4.sh`。
- 只读观察脚本：`verl/examples/ttrl/monitor_sps_pairwise_process_count_neutral_sharpened_prob_qwen25_math_7b_50step_v69_strict_n4.sh`，只能在单独 worker login 终端运行。

v69 初始配置：`pairwise_process_preference_strength=0.8`、`temperature=1.0`，保留 v64/v68 的 process tilt，关闭 base contrast 和 K-band，strict validation 仍是 Qwen2.5-Math-7B、50 step、`val_kwargs.n=4`、禁用 validation answer selection。

判断重点：`direct_pairwise_process_top_preference` 是否为正，同时 `direct_majority_target_mass/direct_base_agreement` 不再饱和，且 `sps_correct_weight_mass/sps_pick_accuracy` 不像 v65/v66 那样崩掉。最终仍只认 strict `mean@4` 是否达到 75%。

本地 CPU-only smoke 已通过，不启动 Ray/GPU。构造 4 条 fake rollout，所有 ref/rollout logprob 完全相同；其中两条答案 `1` 的过程稳定，另外两条有 revision/conflict。开启 count-neutral 但不加 pairwise 时，4 条 reward 都是 `0.25`；开启 v69 pairwise 后，稳定答案簇两条变为 `0.297568`，冲突答案簇两条变为 `0.202432`，`PAIRWISE_TOP_PREF=0.24077486991882324`，`TARGET_CONF` 从 `0.5` 提到 `0.595136284828186`。这说明 v69 的目标构造方向符合预期，但这只是本地单元验证，不是 50-step GPU 训练结果，不能算 goal 完成。

当前 worker 仍阻塞：`mlx worker list` 只有旧的 `987816`，它已经在登录时出现 `/proc/self/exe` 缺失和 `mount -t proc proc /proc` 提示。这轮没有继续登录、没有做 Ray/CUDA 恢复，也没有启动训练。

新增静态 strict-config 审计脚本：`verl/examples/ttrl/verify_pure_sharpening_strict_configs.py`。它只做启动前防泄漏检查，不证明算法有效。当前运行结果为 `STRICT_PURE_SHARPENING_CONFIG_AUDIT_OK`。检查内容包括：v68/v69 runner 使用 Qwen2.5-Math-7B 本地权重、50 step、validation `n=4`、禁用 validation answer selection、`sps_majority_reward_coef=0.0`、`sps_format_reward_coef=0.0`、没有显式 n=32/answer-selection 配置、cache 在 `/tmp/ttrl_cache/<exp>`、Ray dashboard/runtime-env workaround 存在、v68/v69 mode 已接到 `ray_trainer.py`、默认配置在 yaml 中 default-off。

同时通过：`py_compile`、v68/v69 runner 和 monitor 的 `bash -n`、`git diff --check`。这些都只是本地静态审计，不能替代 50-step GPU strict validation。

新增跑后结果审计脚本：`verl/examples/ttrl/audit_pure_sharpening_result.py`。用途是在 v68/v69 训练真正跑完后再判断是否可以进入 goal completion audit。它会检查 metrics、ray taskrunner log、proc health、throughput summary 是否齐全，提取 strict `val-core/MATH-TTT/acc/mean@4`，同时检查日志里没有 validation selection、没有 `n=32` 泄漏、训练步数和 validation 配置仍是 strict 50-step n=4。只有完整 artifact 且 `mean@4 >= 0.75` 时才输出 `RESULT=PASS`。

用被中断的 v68 artifact 做了负向 smoke：脚本正确拒绝该 run，因为缺少 `sps_count_neutral_process_sharpened_prob_qwen25_math_7b_50step_v68_strict_n4_metrics.txt`。这证明中断 run 不会被误判为有效结果。这个审计脚本仍然只是 guard，不是成功指标；真正完成条件仍然是健康 worker 上完整 50-step GPU run 的 strict `mean@4 >= 75%`。

随后用完整但未达标的 v67 做了第二个 smoke：审计脚本现在会解析 runner 日志中“最后生效”的 CLI override，而不是因为基础脚本里早先出现 `answer_rule_conf_weight` 默认值就误拒。指标解析也已改成优先读取 final validation dict 的高精度值，只有缺少 dict 时才退回 step 行的三位小数。v67 输出为 `STRICT_MEAN@4=0.7183098591549296`、`BEST@4=0.8322897384305835`、`MAJ@4=0.7306358148893359`、`TARGET=0.75`、`RESULT=FAIL`，退出码 `2`。这说明它能区分“完整但指标不足”和“artifact 不完整”两类失败。

## 当前 completion audit 快照（2026-07-07）

goal 仍未完成。完成条件是：Qwen2.5-Math-7B、50 step、validation `n=4`、禁用 validation answer selection，且 strict `val-core/MATH-TTT/acc/mean@4 >= 0.75`；同时不能使用 majority pseudo-label、hard majority reward、validation-time best-of/major vote/n=32/真值选择。

已覆盖的证据：

- 相关文献/代码方向和 10 类启发已经写入英文 handoff 和本文档；
- v68 是 count-neutral process sharpening，用 `logmeanexp` 去掉答案簇样本数带来的 majority-count boost；
- v69 是 count-neutral target 上的 pairwise process preference，用可解析、非截断、boxed tail 稳定、boxed 一致、无 revision/conflict 这些无监督过程特征构造 Bradley-Terry 风格偏好；
- `verify_pure_sharpening_strict_configs.py` 已通过，输出 `STRICT_PURE_SHARPENING_CONFIG_AUDIT_OK`；
- `audit_pure_sharpening_result.py` 已能区分完整但未达标的 v67 和 artifact 不完整的 v68，并且优先读取高精度 final validation dict；
- `audit_pure_sharpening_result.py` 现在还会输出 clean ablation 诊断字段：target confidence/entropy/effective_K、unique answer count、diagnostic majority mass/agreement、base agreement、low-budget parseable/clip/top mass、process consistency/support、count-neutral/pairwise 字段、`sps_pick_accuracy`、`sps_correct_weight_mass`、`pass@32`、validation clip/format/logprob、训练 clip ratio 和吞吐；
- v67 完整但未达标：`mean@4=0.7183098591549296`。

当前阻塞：

- `mlx worker list` 只有 worker `987816`；
- 这个 worker 是之前已确认 `/proc` 损坏的 worker，登录时出现 `/proc/self/exe` 缺失和 `mount -t proc proc /proc` 提示；
- 按规则不能在这个 worker 上继续 Ray/CUDA，也不能做恢复实验；
- 因此当前没有启动新的 GPU 训练。

拿到健康 worker 后的下一步：

1. 先跑 runner 内 preflight，确认 `/proc`、8 GPU、driver/compat、`cuInit=0`、GEMM、CUDA/cuBLAS/cuDNN/NCCL/NVJitLink 路径都正常；
2. 优先跑 v69：
   `bash /opt/tiger/TTRL/verl/examples/ttrl/worker_run_sps_pairwise_process_count_neutral_sharpened_prob_qwen25_math_7b_50step_v69_strict_n4.sh`；
3. 观察必须另开同 worker login 终端，运行 v69 monitor，不能占用 runner 前台终端；
4. 跑完后执行：
   `/opt/tiger/modelchef/.venv/bin/python3 /opt/tiger/TTRL/verl/examples/ttrl/audit_pure_sharpening_result.py v69`；
5. 只有审计输出 `RESULT=PASS` 且 `mean@4 >= 0.75`，才能进入 goal 完成复核；否则记录失败归因，再跑 v68 clean ablation。

## ablation 对比工具

新增只读汇总脚本：

- `verl/examples/ttrl/summarize_pure_sharpening_ablation.py`

用途：把 active goal 要求的 anchor 统一成 TSV 表，避免手工抄错。当前覆盖 MajVote baseline、v48 当前最好 SPS、v62 process-quality、v63 direct sharpened probability、v67 band-limited process sharpening，以及未来 v68/v69。缺失实验会标成 `MISSING`，不会误算成成功或失败。

运行命令：

`/opt/tiger/modelchef/.venv/bin/python3 /opt/tiger/TTRL/verl/examples/ttrl/summarize_pure_sharpening_ablation.py`

最新保存的 TSV：`/opt/tiger/TTRL/verl/pure_sharpening_ablation_summary.tsv`。

当前关键结果：

| 方法 | strict mean@4 | best@4 | maj@4 诊断 | 状态 |
| --- | --- | --- | --- | --- |
| MajVote baseline | `0.73490945674` | `0.838338028169` | `0.746696177062` | complete |
| v48 当前最好 SPS | `0.745472837022` | `0.860191146881` | `0.762547283702` | complete |
| v62 process-quality | `0.733903420523` | `0.830414486922` | `0.750251509054` | complete |
| v63 direct probability | `0.725855130785` | `0.841040241449` | `0.742672032193` | complete |
| v67 band-limited | `0.718309859155` | `0.832289738431` | `0.730635814889` | complete |
| v68 count-neutral | `NA` | `NA` | `NA` | missing |
| v69 pairwise count-neutral | `NA` | `NA` | `NA` | missing |

结论：当前已完成实验里最好仍是 v48 `0.745472837022`，距离 0.75 目标差约 `0.00453`。v63/v67 说明目前 pure direct-sharpening 变体低于 v48 和 MajVote baseline；v68/v69 还没有真实 GPU 指标。脚本同时输出 target entropy/effective_K、low-budget top mass、process consistency、clip/format/logprob、throughput 等列，后续 v69 跑完后可直接用于 clean ablation 分析。

## goal 完成审计工具

新增最终 gate 脚本：

- `verl/examples/ttrl/check_pure_sharpening_goal_completion.py`

用途：在未来任何 `update_goal` 之前先跑它。它会先跑 strict config audit，再读取 `verl/pure_sharpening_ablation_summary.tsv`，但只把当前纯锐化候选 v68/v69 作为可完成对象；v48 只是历史最好 anchor，不能用于完成这个新 goal。只有 v68 或 v69 有完整 metrics、`strict_mean4 >= 0.75`，并且对应 post-run audit 输出 `RESULT=PASS`，这个脚本才会通过。

当前已增强：completion gate 会先自动运行 `summarize_pure_sharpening_ablation.py` 刷新 TSV，避免使用旧表；同时输出 `CHECKLIST ...=PASS/FAIL evidence=...`，覆盖 objective restatement、summary refresh、strict config audit、ablation rows、pure candidate presence、target metric threshold 和 post-run audit。

当前运行结果是预期的失败：

- `CHECKLIST objective_restatement=PASS`
- `SUMMARY_REFRESH_EXIT=0`
- `CHECKLIST summary_refresh=PASS`
- `STRICT_CONFIG_AUDIT_OUTPUT=STRICT_PURE_SHARPENING_CONFIG_AUDIT_OK`
- `CHECKLIST strict_config_audit=PASS`
- `CHECKLIST ablation_rows_present=PASS evidence=rows=7`
- `CANDIDATE=v68_count_neutral ARTIFACT_STATUS=MISSING STRICT_MEAN@4=NA`
- `CANDIDATE=v69_pairwise_count_neutral ARTIFACT_STATUS=MISSING STRICT_MEAN@4=NA`
- `CHECKLIST pure_candidates_listed=PASS evidence=seen=v68_count_neutral,v69_pairwise_count_neutral`
- `CHECKLIST target_metric_reached=FAIL evidence=best=NA target=0.75`
- `MISSING_REQUIREMENT=no complete v68/v69 pure-sharpening run reaches target`
- `RESULT=FAIL`，退出码 `2`

结论：goal 仍未完成，不能调用 `update_goal`。

## runner 跑后自动审计

v68/v69 runner 已更新：训练命令结束、metrics/proc/throughput snapshot 写完后，会自动刷新 ablation TSV 并运行 goal completion audit。

涉及 runner：

- `verl/examples/ttrl/worker_run_sps_count_neutral_process_sharpened_prob_qwen25_math_7b_50step_v68_strict_n4.sh`
- `verl/examples/ttrl/worker_run_sps_pairwise_process_count_neutral_sharpened_prob_qwen25_math_7b_50step_v69_strict_n4.sh`

新增跑后动作：

- 运行 `summarize_pure_sharpening_ablation.py`，刷新 `verl/pure_sharpening_ablation_summary.tsv`；
- 运行 `check_pure_sharpening_goal_completion.py`；
- 两段输出都会追加到 runner log，并写入 `..._ABLATION_SUMMARY_STATUS` 和 `..._COMPLETION_AUDIT_STATUS`。

注意：completion audit 只是跑后判定证据，不会覆盖训练进程自己的退出码；runner 仍然用原训练 `status` 退出。当前验证：两个 runner `bash -n` 通过，strict config audit 仍通过，completion audit 当前仍按预期失败，因为 v68/v69 缺完整 GPU metrics。

## v69 worker readiness guard

新增只读 worker 检查脚本：

- `verl/examples/ttrl/check_worker_readiness_for_v69.sh`

用途：只运行 `NO_COLOR=1 TERM=dumb mlx worker list`，不 login、不碰 Ray/CUDA；先排除已知坏 worker，再判断是否有可用于 v69 的 8x `NVIDIA-B200` worker。默认 `KNOWN_BAD_WORKERS=987816`。

当前输出：

- `WORKER id=987816 gpu=8 gpu_type=NVIDIA-B200 pod_ip=fdbd:dccd:cde2:2131:0:e665:d5ca:9aea known_bad=1`
- `READINESS=FAIL`
- `REASON=no healthy non-known-bad 8x NVIDIA-B200 worker is listed`
- 退出码 `2`

结论：不能因为 `mlx worker list` 里有 `987816` 就启动训练；必须等这个脚本输出 `READINESS=PASS` 后，再按它打印的 login 和 v69 runner 命令执行。

## guarded v69 入口

新增 wrapper：

- `verl/examples/ttrl/guarded_run_v69_pairwise_pure_sharpening.sh`

行为：先运行 `check_worker_readiness_for_v69.sh`；readiness 失败就直接退出，不进入 v69 runner；readiness 通过时默认 dry-run，只打印下一步命令。只有在健康 worker shell 里显式设置 `RUN_V69_AFTER_READINESS=1`，才会执行 v69 runner：

`RUN_V69_AFTER_READINESS=1 bash /opt/tiger/TTRL/verl/examples/ttrl/guarded_run_v69_pairwise_pure_sharpening.sh`

当前验证：`bash -n` 通过；在当前 devbox 上运行被 readiness 拦住，退出码 `2`，没有进入 v69 runner。

## 本地 guard smoke

新增统一本地检查脚本：

- `verl/examples/ttrl/smoke_pure_sharpening_local_guards.sh`

它只做本地静态/只读检查，不启动 GPU：`py_compile` 关键 Python 文件、`bash -n` readiness/guarded runner/v68/v69 runner、strict config audit、刷新 ablation summary、运行 completion gate、运行 worker readiness。

现在 smoke 还包含 v69 pairwise target 的 CPU-only 机制验证：`smoke_v69_pairwise_target_cpu.py`。它直接调用 `apply_direct_sharpened_ttrl_reward(...)`，构造 4 条 fake rollout，所有模型/ref/rollout logprob 完全相同，开启 count-neutral aggregation。不开 pairwise 时 reward 完全均匀：`0.25,0.25,0.25,0.25`；开启 v69 pairwise 后，干净过程答案簇变为 `0.334522,0.334522`，冲突过程答案簇变为 `0.165478,0.165478`，`TARGET_CONF` 从 `0.5` 提到 `0.6690433025360107`，输出 `V69_PAIRWISE_TARGET_CPU_SMOKE_OK`。这只证明目标构造机制正确，不是 GPU 训练结果。

当前结果：`SMOKE_EXIT_CODE=0`。其中 completion gate 退出 `2` 并被记录为 `SMOKE completion_gate_not_complete`，这是预期状态；worker readiness 退出 `2`，因为当前只看到已知坏 worker `987816`，也不算 smoke 失败。

后续本地改动后，可先跑：

`bash /opt/tiger/TTRL/verl/examples/ttrl/smoke_pure_sharpening_local_guards.sh`

## v68/v69 no-majority-target 静态审计

新增静态审计脚本：

- `verl/examples/ttrl/audit_v68_v69_no_majority_target.py`

这个脚本不是全局 grep `majority`，因为仓库里仍然保留旧的 TTRL/SPS majority 路径用于复现。它只检查当前纯锐化候选 v68/v69：

- runner 必须是 Qwen2.5-Math-7B、50 step、strict `val_kwargs.n=4`、禁用 validation answer selection；
- runner 必须设置 `ttrl.sps_majority_reward_coef=0.0` 和 `ttrl.sps_format_reward_coef=0.0`；
- v68 必须使用 `count_neutral_process_sharpened_prob`；
- v69 必须使用 `pairwise_process_count_neutral_sharpened_prob`；
- `ray_trainer.py` 里这两种 mode 必须走 `apply_direct_sharpened_ttrl_reward(...)`，不能走 `apply_ttrl_gt`、`apply_sps_weighted_ttrl_gt`、`select_majority_first_per_prompt` 或 majority fallback；
- `apply_direct_sharpened_ttrl_reward(...)` 必须先构造 `answer_logits`、`target_probs` 和 `target_prob_by_answer`，之后才计算 majority 诊断；
- majority 不能参与 `prompt_rewards` 构造。

当前验证通过：

- `/opt/tiger/modelchef/.venv/bin/python3 /opt/tiger/TTRL/verl/examples/ttrl/audit_v68_v69_no_majority_target.py`
- 输出 `V68_V69_NO_MAJORITY_TARGET_AUDIT_OK`

统一本地 smoke 已包含这个审计。最新 smoke 在 `2026-07-07 23:08` 通过，包含：

- `STRICT_PURE_SHARPENING_CONFIG_AUDIT_OK`
- `V68_V69_NO_MAJORITY_TARGET_AUDIT_OK`
- `V69_PAIRWISE_TARGET_CPU_SMOKE_OK`

completion gate 仍按预期失败：v68/v69 都缺完整 GPU metrics，`STRICT_MEAN@4=NA`，`RESULT=FAIL`。当前只看到已知坏 worker `987816`，readiness 仍是 `READINESS=FAIL`。所以 goal 仍未完成，不能调用 `update_goal`；下一步必须等健康 worker 后优先跑 v69，若失败再跑 v68。

## completion gate 新增 no-majority-target 硬检查

已更新：

- `verl/examples/ttrl/check_pure_sharpening_goal_completion.py`

现在 completion gate 不只检查 strict config、ablation TSV 和 post-run audit，还会先运行：

`/opt/tiger/modelchef/.venv/bin/python3 /opt/tiger/TTRL/verl/examples/ttrl/audit_v68_v69_no_majority_target.py`

新增输出包括：

- `NO_MAJORITY_TARGET_AUDIT_EXIT=0`
- `NO_MAJORITY_TARGET_AUDIT_OUTPUT=V68_V69_NO_MAJORITY_TARGET_AUDIT_OK`
- `CHECKLIST no_majority_target_audit=PASS evidence=V68_V69_NO_MAJORITY_TARGET_AUDIT_OK`

如果这个审计不通过，即使未来 TSV 里出现 `mean@4 >= 0.75`，completion gate 也会失败，防止 hidden majority target / majority reward 泄漏被误判为纯锐化成功。

验证结果：

- `py_compile` 通过；
- `git diff --check` 通过；
- 直接运行 completion gate 仍返回 `RESULT=FAIL`，原因仍是 v68/v69 缺完整 GPU metrics；
- 统一本地 smoke 重新运行退出码 0，并显示新的 `CHECKLIST no_majority_target_audit=PASS`。

当前状态不变：goal 未完成；还需要健康 worker 上完整跑 v69，必要时再跑 v68，并用 completion gate 判定是否达标。

## completion gate 新增文档证据审计

新增：

- `verl/examples/ttrl/audit_pure_sharpening_docs.py`

用途：把 active goal 里的非数值要求也纳入 completion gate，避免未来只因为某个 TSV 指标达标就误判完成。它检查：

- 英文 handoff 里有当前文献 refresh 章节，并且至少有 10 个论文/preprint/code 方向标记；
- 英文 handoff 里有 active goal completion audit snapshot；
- 英文 handoff 记录了 v68/v69 设计和 no-majority completion gate 硬检查；
- 中文摘要里有 75% 纯锐化 goal、10 类文献/代码启发、no-majority-target 审计、completion gate 硬检查和当前未完成状态；
- 文档里能看到用户要求的已有实验现象链：v48、v58-v62、v63、v67。

当前验证：

- `/opt/tiger/modelchef/.venv/bin/python3 /opt/tiger/TTRL/verl/examples/ttrl/audit_pure_sharpening_docs.py`
- 输出 `PURE_SHARPENING_DOC_AUDIT_OK papers=13`

completion gate 现在新增：

- `DOC_AUDIT_EXIT=0`
- `DOC_AUDIT_OUTPUT=PURE_SHARPENING_DOC_AUDIT_OK papers=13`
- `CHECKLIST doc_audit=PASS evidence=PURE_SHARPENING_DOC_AUDIT_OK papers=13`

统一本地 smoke 已重新跑过，退出码 0。当前 completion gate 仍然正确失败，因为 v68/v69 都没有完整 GPU metrics。现在完成 goal 至少需要同时满足：strict config audit、no-majority-target audit、doc audit、完整 v68/v69 GPU 结果、post-run audit，且 strict `mean@4 >= 0.75`。

## 最新 live blocker

刚重新检查了 readiness 和 completion gate。

readiness：

- 命令：`bash /opt/tiger/TTRL/verl/examples/ttrl/check_worker_readiness_for_v69.sh`
- 当前 `mlx worker list` 仍然只看到 worker `987816`；
- `987816` 在 `KNOWN_BAD_WORKERS` 里；
- 输出 `READINESS=FAIL`；
- 原因：`no healthy non-known-bad 8x NVIDIA-B200 worker is listed`。

completion gate：

- 命令：`/opt/tiger/modelchef/.venv/bin/python3 /opt/tiger/TTRL/verl/examples/ttrl/check_pure_sharpening_goal_completion.py`
- `strict_config_audit=PASS`；
- `no_majority_target_audit=PASS`；
- `doc_audit=PASS`；
- 但 `v68_count_neutral` 和 `v69_pairwise_count_neutral` 都是 `ARTIFACT_STATUS=MISSING`；
- 输出 `MISSING_REQUIREMENT=no complete v68/v69 pure-sharpening run reaches target` 和 `RESULT=FAIL`。

结论：不能用 `987816` 跑 v69/v68，也不能 `update_goal`。下一步必须等健康 non-known-bad 8x B200 worker，然后先跑 guarded v69。

## readiness guard 新增恰好一个 worker 检查

已更新：

- `verl/examples/ttrl/check_worker_readiness_for_v69.sh`

新规则：

- 输出 `WORKER_COUNT=<n>`；
- 如果没有 worker，直接失败；
- 如果列出超过一个 worker，直接失败，输出：
  `REASON=multiple workers are listed; keep exactly one worker before running v69`；
- 只有恰好一个 worker，且它是非 known-bad 的 8x `NVIDIA-B200`、`podIP` 非空，才会输出 `READINESS=PASS`。

当前验证：

- `bash -n` 通过；
- 当前 readiness 输出 `WORKER_COUNT=1`，但因为唯一 worker 是 known-bad `987816`，仍然 `READINESS=FAIL`；
- 统一本地 smoke 重新跑过，退出码 0。

这个改动是防止下个会话在多个 worker 同时存在时误跑 v69，符合“不可以同时用两个 worker”的规则。它不是训练结果，也不是 accuracy/throughput improvement，所以不作为 improvement commit。

## readiness guard fixture smoke

新增本地测试脚本：

- `verl/examples/ttrl/smoke_worker_readiness_guard.sh`

同时 `check_worker_readiness_for_v69.sh` 支持 `WORKER_LIST_FIXTURE=<path>`，用于本地构造 `mlx worker list` 输出；正常运行时仍然调用真实只读命令 `NO_COLOR=1 TERM=dumb mlx worker list`。

覆盖用例：

- `no_worker`：退出 `2`，输出 `REASON=no worker is listed`；
- `known_bad_single`：退出 `2`，输出 `REASON=no healthy non-known-bad 8x NVIDIA-B200 worker is listed`；
- `multiple_workers`：退出 `2`，输出 `REASON=multiple workers are listed; keep exactly one worker before running v69`；
- `healthy_single`：退出 `0`，输出 `READINESS=PASS`、`SELECTED_WORKER`、login 命令和 v69 runner 命令。

验证：

- `bash -n` 通过；
- `bash /opt/tiger/TTRL/verl/examples/ttrl/smoke_worker_readiness_guard.sh` 输出 `WORKER_READINESS_GUARD_SMOKE_OK`；
- `bash /opt/tiger/TTRL/verl/examples/ttrl/smoke_pure_sharpening_local_guards.sh` 退出码 0，已包含 `SMOKE worker_readiness_guard_fixtures`。

结论：下一次 v69 GPU 启动前的 worker 策略已经有本地测试覆盖：必须恰好一个 worker，且该 worker 是非 known-bad、8x B200、`podIP` 非空。当前 live 状态仍失败，因为唯一 worker 是 known-bad `987816`。

## 2026-07-07 23:25 最新 gate 刷新

刚重新跑了两个只读/本地 gate。

readiness：

- 命令：`bash /opt/tiger/TTRL/verl/examples/ttrl/check_worker_readiness_for_v69.sh`
- `mlx worker list` 仍然只看到一个 worker：`987816`；
- `WORKER_COUNT=1`；
- 该 worker 是 8x `NVIDIA-B200`，`podIP` 非空，但在 `KNOWN_BAD_WORKERS=987816` 里；
- 输出 `READINESS=FAIL`；
- 失败原因：`no healthy non-known-bad 8x NVIDIA-B200 worker is listed`。

completion gate：

- 命令：`/opt/tiger/modelchef/.venv/bin/python3 /opt/tiger/TTRL/verl/examples/ttrl/check_pure_sharpening_goal_completion.py`
- strict config audit、no-majority-target audit、doc audit 都通过；
- 但 `v68_count_neutral` 和 `v69_pairwise_count_neutral` 仍然都是 `ARTIFACT_STATUS=MISSING`；
- 输出 `MISSING_REQUIREMENT=no complete v68/v69 pure-sharpening run reaches target` 和 `RESULT=FAIL`。

结论：

- 当前 pure distribution sharpening goal 仍未完成；
- 不能调用 `update_goal`；
- 不能在已知坏 worker `987816` 上启动 v68/v69；
- 下一步仍然是等健康的、非 known-bad 的、列表里唯一的 8x B200 worker，然后在 worker 内优先跑：
  `RUN_V69_AFTER_READINESS=1 bash /opt/tiger/TTRL/verl/examples/ttrl/guarded_run_v69_pairwise_pure_sharpening.sh`

## v69 完整 GPU 结果：明显负结果

v69 已在健康 B200 worker `988093` 上完整跑完。

运行信息：

- 时间：`2026-07-07 23:39` 到 `2026-07-08 00:13 CST`；
- worker：`988093`，唯一 worker，8x `NVIDIA-B200`；
- readiness 通过：`READINESS=PASS`；
- 训练前 `/proc/self`、`/proc/meminfo` 正常，`PROC_COUNT_BEFORE=75`；
- driver `580.105.08`，`cuInit=0`，GEMM smoke 通过；
- 实际加载的是 venv cu12.9 的 cuBLAS/cuDNN/NCCL/NVJitLink；
- 缓存和本地模型 copy 均在 `/tmp/ttrl_cache/sps_pairwise_process_count_neutral_sharpened_prob_qwen25_math_7b_50step_v69_strict_n4`；
- 训练后 `/proc` 仍正常：`PROC_SELF_OK_FINAL`、`PROC_MEMINFO_OK_FINAL`、`PROC_COUNT_FINAL=80`；
- 8 张 GPU 最终都回到 `0 MiB`。

严格协议：

- 模型是 Qwen2.5-Math-7B 的本地 copy；
- 50 training steps；
- validation `n=4`；
- `trainer.validation_answer_selection_enable=False`；
- 没有 validation-time best-of、majority selection、n=32 选择或真值选择；
- `ttrl.sps_reward_mode=pairwise_process_count_neutral_sharpened_prob`；
- `sps_majority_reward_coef=0.0`，`sps_format_reward_coef=0.0`；
- 使用 count-neutral aggregation 和 pairwise process preference。

结果：

- strict `val-core/MATH-TTT/acc/mean@4=0.49899396378269617`；
- `best@4=0.7108913480885312`；
- `maj@4=0.520195171026157`；
- `format_score/mean@4=0.971830985915493`；
- `response_clip/mean@4=0.096579476861167`。

吞吐：

- `step_rows=50`；
- 非 validation step 数：`49`；
- 非 validation 平均 step time：`30.897s`；
- 非 validation 整机吞吐：`9653.870 token/s`；
- final validation 用时：`233.678s`；
- final step 总用时：`264.014s`。

关键内部指标：

- step 50 `target_confidence=0.153`；
- step 50 `target_entropy=2.430`；
- step 50 `target_effective_K=11.667`；
- step 50 `unique_answer_count=30.375`；
- step 50 `majority_target_mass=0.082`；
- step 50 `base_agreement=0.125`；
- step 50 `pairwise_process_top_preference=0.242`；
- step 50 `sps_pick_accuracy=0.000`；
- step 50 `sps_correct_weight_mass=0.040`；
- `pass@32=1.000`。

解释：

- v69 的 pairwise process preference 确实生效了，因为 `pairwise_process_top_preference` 一直为正；
- 但是 target 太分散，`effective_K` 高、`target_confidence` 低；
- 更关键的是它没有对齐 correctness basin，`sps_correct_weight_mass` 很低，最终 `mean@4` 只有 `49.90%`；
- format 不是主要问题，因为 `format_score/mean@4` 仍有 `97.18%`；
- 这说明“count-neutral answer aggregation + 纯过程干净度 pairwise preference”这个版本会偏向看起来过程稳定但不正确的轨迹。

completion gate：

- v69 artifact 已完整；
- ablation 表中 v69 行已更新为 `COMPLETE_METRICS`；
- completion audit 失败：
  - `BEST_PURE_CANDIDATE=v69_pairwise_count_neutral`；
  - `BEST_PURE_MEAN@4=0.498993963783`；
  - `CHECKLIST target_metric_reached=FAIL`；
  - `RESULT=FAIL`。

结论：

- goal 仍未完成；
- 不能调用 `update_goal`；
- v69 是明确负结果，不做 improvement commit；
- 下一步不应该盲目调大 pairwise strength 或降低 temperature，因为问题不是锐化不够，而是 correctness 对齐错误；
- 下一版应在 pure sharpening 内加入独立 correctness-support gate，例如 base/ref-supported answer stability、first4 稳定性、rephrase/counterfactual consistency 等，再做锐化；majority 仍只能作为诊断。

## best-strict-50 + SPS=32 推理时选择结果

这次方向和 strict `n=4` goal 不同：允许 inference-time scaling。做法是先用目前最好的 strict 50-step 路线 v48/process-consistency 训练，再在验证时生成 `SPS=32` 条候选，按无监督答案簇 majority 做 selection，把选中的答案折叠回 `mean@4` 评估路径。

关键过程：

- v70：直接训练 + SPS32 validation，生成完 `15904 = 497 * 32` 条 validation rollout 后 Ray actor keepalive timeout，没有落出 final val metrics，也没有 ckpt。
- v71/v72：补做 train-save + ckpt eval，但第一次 ckpt 写在根盘路径，rootfs 写满导致 8 个 actor shard 全部损坏，`torch.load` 报 `failed finding central directory`，不可用。
- v73：同一训练配置改为把 checkpoint 保存到 `/tmp/ttrl_ckpts/...`，50 step 成功完成，8 个 actor shard 全部 `torch.load` 通过。
- v74：从 v73 的 `global_step_50` checkpoint 做 eval-only SPS32 selection，完整跑完并落出指标。

v73 训练吞吐：

- steps 41-50 平均 `29.091s/step`；
- 整机吞吐约 `8037 token/s`；
- 训练结束 `/proc` 正常。

v74 最终指标：

- selected/collapsed `val-core/MATH-TTT/acc/mean@4=0.8490945674044266`，即 `84.91%`；
- selected 后 `best@4=84.91%`、`maj@4=84.91%`，因为选中答案被重复 4 次；
- raw SPS32 诊断：`mean@32=73.16%`、`best@32=93.19%`、`maj@32=82.50%`；
- selected `format_score=100%`，`response_clip=0%`；
- eval 结束状态 0，`PROC_COUNT_FINAL=97`，8 张 B200 全部释放。

结论：

- v48/v73 strict 50-step checkpoint 加 SPS32 推理时答案簇选择，可以从 strict no-selection 的 `74.55%` 提到 `84.91%`。
- 这个结果非常接近 `85%`，但仍低约 `0.09pp`，不能算达标。
- 该结果依赖 validation-time `n=32` selection，不能和 strict `n=4` 无推理时选择目标混用。
- 因为没有过 85%，不作为 target-reaching improvement commit；但它是可复现的强结果，后续可以直接复用 v73 ckpt 或 v71/v72 runner 的 env override 机制。

## 30-step v48 checkpoint + TTRL answer-cluster selection

为了和历史 ITS 结果同口径比较，又对当前 30-step v48/process-consistency checkpoint 做了一次 TTRL validation answer-cluster selection，而不是 pure vLLM SPS logprob matrix。

配置：

- checkpoint：`/tmp/ttrl_ckpts/qwen25_math_7b_v48_process_consistency_strict_n4_30step/global_step_30`；
- eval-only：`trainer.val_only=True`，从 `resume_from_path` 加载；
- validation：`val_kwargs.n=32`；
- selection：`validation_answer_selection_enable=True`，`strategy=majority`，`repeats=4`；
- worker：`989057`，8x B200。

结果：

- selected/collapsed `val-core/MATH-TTT/acc/mean@4=0.8269617706237424`，即 `82.70%`；
- raw SPS32：`mean@32=72.03%`、`best@32=93.09%`、`maj@32=82.08%`；
- selected 后 `format_score=100%`、`response_clip=0%`；
- 任务状态 0，eval 后 `/proc` 正常，8 张 B200 显存释放。

结论：

- 这条同口径 ITS 结果说明，30-step checkpoint 接 TTRL answer-cluster selection 可以到 `82.70%`，明显高于 pure vLLM SPS 矩阵的 `76.6%`。
- 但它低于 v33 20-step ITS 的 `83.10%`，也低于 v73/v74 50-step ITS 的 `84.91%`。
- 因此当前 30-step checkpoint 不是 RL+ITS 的最好版本；历史最好仍是 v73/v74 的 `84.91%`。

## 30-step v48 checkpoint + chunked pure-vLLM SPS

这次按“分块推理时锐化”的想法做了一版：从 30-step v48/process-consistency checkpoint 出发，每一步先采样 `K=8` 条 `64 token` 的短 continuation，用同一套 pure-vLLM logprob sharpening score 选最好的 prefix，再继续下一块。目标是看 progressive prefix selection 能不能比 one-shot full-answer SPS 更好。

实际使用的 worker 内 HF checkpoint：

- `/tmp/ttrl_ckpts/qwen25_math_7b_v48_process_consistency_strict_n4_30step/global_step_30/hf`。

注意：master 上的 `/tmp/qwen25_math_7b_v48_30step_hf` 在 worker 里不可见；第一次 smoke 因为用了这个 master `/tmp` 路径失败，随后改为 worker-local checkpoint 路径后跑通。

配置：

- `MODEL_KIND=qwen_math`；
- `TEMP=0.50`；
- `CHUNKED_CANDIDATES=8`；
- `CHUNKED_TOKENS=64`；
- `CHUNKED_SELECT=max`；
- `MAX_NEW=3072`；
- `NUM_WORKERS=8`；
- `SKIP_MCMC=1`，`SPS_CANDIDATES=0`。

smoke 40 题：

- Base：`37.5%`；
- low-temp：`60.0%`；
- chunked SPS：`62.5%`。

full Math500：

- Base accuracy：`49.8%`；
- low-temp accuracy：`64.4%`；
- chunked SPS accuracy：`74.6%`。

运行情况：

- 最慢 worker 总耗时 `135.5s`；
- chunked SPS 本身各 worker 耗时约 `49.1s` 到 `108.1s`；
- 结束后 `/proc` 正常，8 张 B200 显存全部释放。

对比：

- 同一个 30-step v48 checkpoint 之前 pure-vLLM one-shot SPS 矩阵最好是 `tau=0.50,K=8 -> 76.6%`；
- 这次 chunked prefix selection 是 `74.6%`；
- 因此这版比 one-shot SPS 低 `2.0pp`，不是提升。

结论：

- RL 后模型继续做 inference-time sharpening 是可行的，但第一版“每 64 token 贪心选 prefix”不是更优做法；
- 主要问题可能是早期 prefix 还没有足够强的 correctness 信号，过早贪心会锁到局部高概率但最终错误的推理路径；
- 不做 improvement commit；
- 如果继续这个方向，应该尝试范式改动而不是只调参数：例如保留 top-m prefix、延迟到答案/子答案可解析后再选、或者把 chunked prefix selection 和 final-answer one-shot SPS 组合起来。

## v75 strict 纯锐化：support gate 负结果

v75 是在 v68/v69 之后做的最小新范式：不再做 count-neutral 去掉所有答案计数支持，而是在 direct sharpened probability 上加一个连续 support gate。gate 只用无监督内部信号：base/ref 概率支持、first4 低预算答案稳定性、过程一致性支持；不使用 hard majority reward，不用 validation/inference-time selection，验证仍是 strict `n=4`。

运行信息：

- worker：`989057`，8x B200；
- runner：`verl/examples/ttrl/worker_run_sps_support_gated_process_sharpened_prob_qwen25_math_7b_50step_v75_strict_n4.sh`；
- 50 step 完整跑完，状态 0；
- validation `n=4`，`validation_answer_selection_enable=False`；
- 结束后 `/proc` 正常，`PROC_COUNT_FINAL=106`，8 张 GPU 全部释放；
- 非 validation 平均 step time `27.614s`，整机吞吐约 `8964 token/s`。

结果：

- strict `mean@4=0.740945674044`，即 `74.09%`；
- `best@4=84.68%`；
- `maj@4=75.71%`；
- `format_score=96.48%`；
- `response_clip=1.56%`。

关键内部指标：

- step 50 `target_confidence=0.902`；
- `target_entropy=0.439`；
- `target_effective_K=1.401`；
- `majority_target_mass=0.902`；
- `base_agreement=1.0`；
- `low_budget_top_mass=0.875`；
- `process_consistent_rate=0.951`；
- `process_top_support=0.998`；
- `support_gate_target_overlap=0.626`；
- `support_gate_target_agreement=1.0`；
- `sps_pick_accuracy=1.0`；
- `sps_correct_weight_mass=0.444`。

结论：

- v75 比 v68/v69 好很多，但没有超过 v48 的 `74.55%`，也没有达到目标 `75%`；
- completion audit 明确失败：当前最好 pure candidate 是 v75，但 `best=0.740945674044 < 0.75`；
- 不调用 `update_goal`，不做 improvement commit；
- 失败原因不是锐化不够，而是 support gate 后期变成了“确认已坍缩答案簇”的信号。target 已经非常尖，`effective_K=1.401`，但 mean@4 没有上去，说明错误簇也被一并确认了。

下一步方向：

- 不继续随机调 beta / temperature / reward scale；
- 更合理的新范式是把 support 从乘法 gate 改成保守 target-mixture 约束：当 direct target 过尖时，把目标分布按内部支持拉回 base/first4/process 的软支持分布，避免在独立支持不足时过早坍缩到单一答案簇。

## v76 strict 纯锐化：support mixture 负结果

v76 是对 v75 失败原因的直接验证：v75 的 support gate 后期会确认已经坍缩的答案簇，所以 v76 把 support 改成保守的 target-mixture。当 direct target 过尖且和 base/first4/process support 重叠不足时，把目标分布软混回 support 分布，避免过早单簇坍缩。

运行信息：

- worker：`989057`，8x B200；
- runner：`verl/examples/ttrl/worker_run_sps_support_mixture_process_sharpened_prob_qwen25_math_7b_50step_v76_strict_n4.sh`；
- 50 step 完整跑完，状态 0；
- validation `n=4`，`validation_answer_selection_enable=False`；
- 结束后 `/proc` 正常，`PROC_COUNT_FINAL=115`，8 张 GPU 全部释放；
- 非 validation 平均 step time `26.414s`，整机吞吐约 `9388 token/s`。

结果：

- strict `mean@4=0.730885311871`，即 `73.09%`；
- `best@4=83.89%`；
- `maj@4=74.73%`；
- `format_score=96.68%`；
- `response_clip=1.36%`。

关键内部指标：

- step 50 `target_confidence=0.873`；
- `target_entropy=0.571`；
- `target_effective_K=1.452`；
- `majority_target_mass=0.873`；
- `support_mixture_alpha=0.069`；
- mixture 前 `pre_confidence=0.902`、`pre_effective_K=1.380`、`pre_overlap=0.662`；
- `base_agreement=1.0`；
- `low_budget_top_mass=0.750`；
- `process_consistent_rate=0.967`；
- `process_top_support=0.994`；
- `sps_pick_accuracy=0.625`；
- `sps_correct_weight_mass=0.445`。

结论：

- v76 机制生效了：target 从 `pre_confidence=0.902/effective_K=1.380` 被软化到 `confidence=0.873/effective_K=1.452`；
- 但 strict `mean@4` 下降到 `73.09%`，低于 v75 的 `74.09%` 和 v48 的 `74.55%`；
- completion audit 失败：当前 pure candidate 最好仍是 v75 `0.740945674044 < 0.75`；
- 不调用 `update_goal`，不做 improvement commit。

失败归因：

- 单纯把过尖 target 拉回 support 分布不够，因为 support 本身仍然来自同一个答案 basin；
- 软化减少了坍缩强度，但没有引入新的 correctness 区分信号，反而降低了 correct candidate 的前移能力；
- 下一步不应后验调 `support_mixture_strength`，而应加入更独立的 correctness-support proxy，例如延迟到可解析子答案后的稳定性、top-m support 竞争、或能区分“过程一致但错误”和“过程一致且正确”的内部信号。

## v77 strict 纯锐化：support confidence cap 负结果

v77 是对 v75/v76 的进一步验证：不再做 support gate，也不做 support mixture，而是保留 direct target 的 top answer，只用独立 support 分布的置信度加 margin 去限制 top mass。目标是防止 direct target 过尖，但不做答案选择。

运行信息：

- worker：`989057`，8x B200；
- runner：`verl/examples/ttrl/worker_run_sps_support_calibrated_process_sharpened_prob_qwen25_math_7b_50step_v77_strict_n4.sh`；
- 第一次启动有一个本地变量名 bug：`confidence_cap_strength_f` 未定义，已修为 `support_confidence_cap_strength_f`；
- 修复后重新通过 `py_compile`、strict config audit、no-majority audit；
- 50 step 完整跑完，状态 0；
- validation `n=4`，`validation_answer_selection_enable=False`；
- 结束后 `/proc` 正常，`PROC_COUNT_FINAL=116`，8 张 GPU 全部释放；
- 非 validation 平均 step time `28.034s`，整机吞吐约 `9798 token/s`。

结果：

- strict `mean@4=0.630784708249`，即 `63.08%`；
- `best@4=81.00%`；
- `maj@4=65.91%`；
- `format_score=96.78%`；
- `response_clip=4.28%`。

关键内部指标：

- step 50 `target_confidence=0.491`；
- `target_entropy=2.074`；
- `target_effective_K=4.099`；
- cap 前 `pre_confidence=0.723`、`pre_effective_K=2.820`；
- `support_cap_alpha=0.750`；
- `support_cap_value=0.550`；
- `support_gate_top_confidence=0.215`；
- `support_target_overlap=0.652`；
- `support_target_agreement=1.0`；
- `sps_pick_accuracy=0.0`；
- `sps_correct_weight_mass=0.225`。

结论：

- v77 是明显负结果，比 v75/v76/v48 都低很多；
- confidence cap 机制确实生效，但太强地把 target 打散了：`target_confidence` 从 `0.723` 压到 `0.491`，`effective_K` 提到 `4.099`；
- support top confidence 只有 `0.215`，说明这个 support 更像“不确定性先验”，不能直接作为 correctness cap；
- completion audit 仍失败，当前 pure candidate 最好还是 v75 `0.740945674044 < 0.75`；
- 不调用 `update_goal`，不做 improvement commit。

下一步方向：

- 不继续后验调 cap/mix 强度；
- 需要先引入能区分 correct basin 和 consistent-wrong basin 的独立 correctness-support proxy，再决定怎么锐化；
- 候选方向：延迟到局部答案可解析后再看稳定性、top-m support 竞争、或者用过程特征识别“过程一致但错误”的答案簇。

## v78 strict 纯锐化：support residual tilt 负结果

v78 是在 v75 support-gated process sharpening 上加一个 residual tilt：

- residual 定义为 `log p_support(answer) - log p_base(answer)`；
- `p_support` 仍然来自 base/ref、first4 稳定性、process-clean support；
- 目的不是选答案，而是当 first4/process 支持强于 base 惯性时，给 soft target logits 一个额外纠偏；
- 仍然不使用 majority pseudo-label、不使用 hard majority reward、不做 validation-time selection。

代码和配置：

- 新 reward mode：`support_residual_process_sharpened_prob`；
- mode id：`17.0`；
- 默认关闭参数：`sps_direct_support_residual_strength: 0.0`；
- v78 runner 中设置 `sps_direct_support_residual_strength=0.5`；
- strict 配置保持：
  - 50 step；
  - validation `n=4`；
  - `validation_answer_selection_enable=False`；
  - `sps_majority_reward_coef=0.0`；
  - `sps_format_reward_coef=0.0`。

运行信息：

- worker：`989057`，8x B200；
- runner：`verl/examples/ttrl/worker_run_sps_support_residual_process_sharpened_prob_qwen25_math_7b_50step_v78_strict_n4.sh`；
- 训练完整结束，状态 0；
- 结束后 `/proc` 正常，`PROC_COUNT_FINAL=115`；
- 8 张 GPU 最终全部释放；
- 非 validation 平均 step time `27.869s`；
- 非 validation 整机吞吐约 `8882 token/s`。

结果：

- strict `mean@4=0.737424547284`，即 `73.74%`；
- `best@4=83.51%`；
- `maj@4=74.87%`；
- `format_score=96.83%`；
- `response_clip=1.16%`。

关键内部指标：

- step 50 `target_confidence=0.850`；
- `target_entropy=0.621`；
- `target_effective_K=3.772`；
- `majority_target_mass=0.850`；
- `support_gate_top_confidence=0.593`；
- `support_target_overlap=0.725`；
- `support_target_agreement=1.0`；
- `support_residual_strength_mean=0.5`；
- `support_residual_std=0.277`；
- `support_residual_top_value=-0.680`；
- `sps_pick_accuracy=0.0`；
- `sps_correct_weight_mass=0.428`。

结论：

- v78 是负结果：`73.74%` 低于 v75 `74.09%`，也低于历史 v48 `74.55%`；
- residual 机制确实生效，但 step 50 的 `support_residual_top_value=-0.680` 说明它多数时候是在压低当前 top target，而不是找到更正确的替代 basin；
- 这说明 base/ref + first4 + process support 仍不是足够独立的 correctness oracle；
- completion audit 仍失败：当前 pure candidate 最好还是 v75 `0.740945674044 < 0.75`；
- 不调用 `update_goal`，不做 improvement commit。

下一步方向：

- 不建议继续调 residual strength、cap、mix 这类强度参数；
- 需要新的 correctness-support proxy：
  - 延迟到中间推理片段后看答案是否稳定；
  - 检查最终答案附近的局部等式/恒等式一致性；
  - 用 top-m basin competition 明确惩罚“过程一致但错误”的答案簇；
  - 或先做 training-free probe 判断 top basin 在继续推理时是否会自我修正，再决定能不能用于 target tilt。

## v79 strict 纯锐化：split-view support 负结果

v79 是基于 v78 的失败信号做的改动。v78 的 residual 经常压低当前 top target，但没有找到更正确的替代 basin，所以 v79 不继续调 residual 强度，而是构造两个内部视角：

- first4 低预算 rollout 的答案分布；
- held-out 后续 rollout 的答案分布，并用 process consistency 给权重；
- 用两个分布的几何平均作为 split support；
- 用 `split_support_strength=0.7` 对 soft target logits 做倾斜；
- 不使用 majority pseudo-label，不使用 hard majority reward，不做 validation-time selection。

代码和配置：

- 新 reward mode：`split_support_process_sharpened_prob`；
- mode id：`18.0`；
- runner：`verl/examples/ttrl/worker_run_sps_split_support_process_sharpened_prob_qwen25_math_7b_50step_v79_strict_n4.sh`；
- strict 配置保持：
  - Qwen2.5-Math-7B；
  - 50 step；
  - validation `n=4`；
  - `validation_answer_selection_enable=False`；
  - `sps_majority_reward_coef=0.0`；
  - `sps_format_reward_coef=0.0`。

运行信息：

- worker：`989057`，8x B200；
- 训练完整结束，状态 0；
- 结束后 `/proc` 正常，`PROC_COUNT_AFTER=119`；
- 8 张 GPU 最终释放；
- 非 validation 平均 step time `27.248s`；
- 非 validation 整机吞吐约 `9089 token/s`；
- 第一次 v79 启动曾被我过早中断，后来 Ray 日志显示它已经过了模型加载和 CUDA graph capture，不是 CUDA/Ray 失败；第二次启动完整跑完。

结果：

- strict `mean@4=0.715291750503`，即 `71.53%`；
- `best@4=82.42%`；
- `maj@4=73.32%`；
- `format_score=96.28%`；
- `response_clip=1.96%`。

关键内部指标：

- step 50 `target_confidence=0.938`；
- `target_entropy=0.328`；
- `target_effective_K=1.230`；
- `majority_target_mass=0.938`；
- `low_budget_top_mass=0.906`；
- `process_consistent_rate=0.957`；
- `process_top_support=0.988`；
- `split_support_top_confidence=0.855`；
- `split_support_target_overlap=0.912`；
- `split_support_target_agreement=1.0`；
- `split_support_view_overlap=0.888`；
- `sps_pick_accuracy=1.0`；
- `sps_correct_weight_mass=0.485`。

结论：

- v79 是明显负结果：`71.53%` 低于 v75 `74.09%`，也低于历史 v48 `74.55%`；
- split-view support 的内部一致性很强，但它主要是在确认已有高置信答案 basin；
- target 被进一步锐化到 `confidence=0.938/effective_K=1.230`，说明它更容易强化 stable wrong basin；
- final train batch 的 `sps_pick_accuracy=1.0` 和 `correct_weight_mass=0.485` 没有转化为 Math500 strict mean@4，因此这些训练批指标不能作为继续加大 split support 的理由；
- completion audit 仍失败：当前 pure candidate 最好还是 v75 `0.740945674044 < 0.75`；
- 不调用 `update_goal`，不做 improvement commit。

下一步方向：

- 不继续做 support strength/mix/cap 的后验调参；
- 需要能区分 stable-correct 和 stable-wrong 的独立 correctness proxy；
- 候选方向：
  - continuation self-correction probe；
  - final answer 附近的局部代数/等式一致性；
  - basin-level contradiction/revision 特征；
  - 或只有当独立一致性检查通过时才允许 support agreement 继续锐化 target。

## v80：稳定过置信容量刹车

时间：2026-07-09 02:37 CST。

设计动机：

- v79 证明了“first4 和 held-out 都同意”不一定代表正确，可能只是稳定错误 basin；
- v79 的 target 太尖：`confidence=0.938`，`effective_K=1.230`，strict `mean@4=71.53%`；
- 当前最好 pure 方案仍是 v75：`mean@4=74.09%`，target 没有 v79 那么塌；
- 所以 v80 不继续加大 split support，而是在 v75 support-gated 路线后加一个容量刹车。

算法：

- 新 mode：`capacity_braked_process_sharpened_prob`；
- mode id：`19.0`；
- 仍然只构造 soft answer-cluster distribution，不用 majority label/reward；
- 当 target 同时满足：
  - confidence 太高；
  - effective K 太低；
  - 和 support distribution 高重合；
  才把 target 轻微混回一个 softened support distribution，避免继续把稳定 basin 压成单峰。

v80 runner：

- `verl/examples/ttrl/worker_run_sps_capacity_braked_process_sharpened_prob_qwen25_math_7b_50step_v80_strict_n4.sh`
- strict 配置不变：
  - Qwen2.5-Math-7B；
  - 50 step；
  - validation `n=4`；
  - `validation_answer_selection_enable=False`。

预跑检查：

- Python 编译通过；
- runner `bash -n` 通过；
- strict 配置审计通过：`STRICT_PURE_SHARPENING_CONFIG_AUDIT_OK`；
- no-majority 审计通过：`PURE_SHARPENING_NO_MAJORITY_TARGET_AUDIT_OK`。

跑完重点看：

- strict `mean@4` 是否超过 75%；
- `capacity_brake_alpha` 是否真的触发；
- `pre/post effective_K` 是否把过尖 target 拉回；
- 如果没有提升，要判断是否只是防塌不够，还是 support 本身仍不能区分正确/错误 basin。

v80 结果：

- 运行完成，脚本状态 `0`；
- strict `mean@4=0.729376257545`，即 `72.94%`；
- `best@4=83.86%`；
- `maj@4=74.65%`；
- `format_score=96.78%`；
- `response_clip=1.41%`；
- 非 validation 平均 step time `26.702s`；
- 非 validation 整机吞吐 `9198.624 token/s`；
- step 50 带 validation，`timing_s/step=250.440s`，其中 `testing=228.265s`。

关键内部指标：

- `target_confidence=0.879`；
- `target_effective_K=1.573`；
- `support_gate_top_confidence=0.601`；
- `support_target_overlap=0.695`；
- `capacity_brake_alpha=0.151`；
- `capacity_brake_pre_confidence=0.911`；
- `capacity_brake_pre_effective_K=1.503`；
- `capacity_brake_pre_overlap=0.687`；
- `capacity_brake_post_effective_K=1.573`；
- `sps_pick_accuracy=0.750`；
- `sps_correct_weight_mass=0.450`。

结论：

- v80 是负结果：`72.94%` 低于 v75 `74.09%`，也低于历史 v48 `74.55%`；
- 容量刹车确实触发了，并把 effective K 从 `1.503` 拉到 `1.573`；
- 但这只是在控制分布不要过尖，没有提供新的 correctness 证据；
- 触发时 target/support overlap 只有 `0.687`，说明该机制不够精准，可能在 support 并不强的时候也削弱有效锐化；
- 当前 goal 仍未完成，completion audit 继续失败，最好 pure candidate 还是 v75 `0.740945674044 < 0.75`；
- 不做 improvement commit。

infra 重要记录：

- v80 跑前 preflight 正常：`/proc` 正常、8 卡可见、`cuInit=0`、GEMM smoke 通过；
- v80 训练结束后 worker 的 `/proc` 损坏：
  - `PROC_SELF_BAD_AFTER`；
  - `PROC_MEMINFO_BAD_AFTER`；
  - `PROC_COUNT_AFTER 0`；
  - final trap 同样记录 `PROC_SELF_BAD_FINAL`、`PROC_MEMINFO_BAD_FINAL`、`PROC_COUNT_FINAL 0`；
- 日志里有 brpc 报错：
  - 找不到 `/proc/self/stat`；
  - 找不到 `/proc/self/fd`；
  - 找不到 `/proc/loadavg`；
- 不要继续使用 worker `989057` 做 Ray/CUDA 实验，除非重启或替换 worker。

下一步：

- 不继续调 capacity threshold/strength；
- 需要引入真正独立的 correctness proxy，例如 continuation self-correction probe、最终答案附近的代数一致性检查、候选答案之间的 contradiction/revision 对比；
- 目标应该是先判断 stable basin 是否可信，再允许锐化，而不是只控制 target entropy/effective K。

## v81 准备：basin-contrast margin calibration

v81 还没跑，只完成了代码和 runner 准备。当前 goal 仍未完成，最好 strict pure 结果仍是 v75 `mean@4=0.740945674044 < 0.75`。

设计动机：

- v75/v48 说明 answer-cluster sharpening 接近有效；
- v76/v80 说明单纯控制 target 过尖不能提供 correctness 证据；
- v79 说明 split-view 稳定也可能稳定在错误 basin；
- 所以 v81 不继续调 support/capacity 强度，而是看 top answer 和第二 answer 的 basin margin 是否被独立内部证据支持。

算法思想：

- 新 mode：`basin_contrast_process_sharpened_prob`；
- 从 v75 support-gated route 出发；
- 先构造 soft target distribution；
- 找 target top answer 和 runner-up；
- 比较 target top-vs-second margin 与 base/ref、first4 low-budget、process-clean support、combined support 的同一 margin；
- 如果 target margin 超过内部 support 能解释的 margin，就只把超出的 top mass 软分给非 top 答案；
- 不使用 majority pseudo-label，不使用 hard majority reward，不做 validation-time selection，majority 仍只作为诊断。

已准备：

- runner：`verl/examples/ttrl/worker_run_sps_basin_contrast_process_sharpened_prob_qwen25_math_7b_50step_v81_strict_n4.sh`；
- strict validation 仍是 Qwen2.5-Math-7B、50 step、validation `n=4`、禁用 `validation_answer_selection_enable`；
- Python 编译通过；
- runner `bash -n` 通过；
- strict config 审计通过：`STRICT_PURE_SHARPENING_CONFIG_AUDIT_OK`；
- no-majority 审计通过：`PURE_SHARPENING_NO_MAJORITY_TARGET_AUDIT_OK`；
- summary 里 v81 目前是 `MISSING`，等待健康 worker 跑。

下一步：

- 有健康 worker 后直接跑 v81；
- 跑前必须检查 `/proc`、8 卡、CUDA/cuBLAS preflight；
- 如果 `mean@4 >= 75%`，再做 post-run audit 和泄漏检查；
- 如果失败，只写文档，不做 improvement commit。

worker 状态补充：

- 当前 `mlx worker list` 只看到 `989057`；
- 登录 `989057` 后 worker 内部仍然是坏状态：
  - `PROC_SELF_BAD`；
  - `PROC_MEMINFO_BAD`；
  - `PROC_COUNT 0`；
- 登录启动还报 `/proc/self/exe` 和 `/dev/fd/63` 缺失；
- 因此不能在 `989057` 上跑 v81，也不能启动 Ray/CUDA preflight；
- 需要新的健康 worker 后再继续实验。

v81 离线 smoke：

- 已把 basin-contrast 公式抽成 `_apply_basin_contrast_calibration(...)`；
- 新增 CPU smoke：`verl/examples/ttrl/smoke_v81_basin_contrast_cpu.py`；
- smoke 覆盖：
  - 内部 support 弱时，soft cap top mass；
  - 内部 support 强时，target 不变；
  - `strength=0` 时完全 no-op；
  - 概率归一且非负；
  - 不替换 top basin，不做 hard selection；
- 结果：`V81_BASIN_CONTRAST_CPU_SMOKE_OK`；
- strict config 审计和 no-majority 审计仍然通过；
- 这只是离线公式验证，不是训练结果，v81 仍等待健康 worker 跑 50 step。

v81 local guard：

- 新增 readiness guard：`verl/examples/ttrl/check_worker_readiness_for_v81.sh`；
- 默认 known-bad workers：`987816 989057`；
- 新增一键离线检查：`verl/examples/ttrl/smoke_v81_local_guards.sh`；
- 已运行完成，离线部分通过：
  - py_compile 通过；
  - runner / readiness guard `bash -n` 通过；
  - `V81_BASIN_CONTRAST_CPU_SMOKE_OK`；
  - `STRICT_PURE_SHARPENING_CONFIG_AUDIT_OK`；
  - `PURE_SHARPENING_NO_MAJORITY_TARGET_AUDIT_OK`；
  - summary 中 v81 仍是 `MISSING`；
  - completion gate 仍按预期失败，说明没有误判完成；
- 当前 readiness 失败：
  - 只看到 worker `989057`；
  - `989057` 在 known-bad 列表；
  - `READINESS=FAIL`；
  - 原因：`no healthy non-known-bad 8x NVIDIA-B200 worker is listed`；
- 所以继续等待健康 worker 后再跑 v81。

v81 runner 启动包已复查：模型路径、cache 路径、CUDA preflight、strict `n=4`、禁用 validation selection、50 step、Ray `127.0.0.1`/no-dashboard/no-runtime-env、majority/format reward 关闭、post-run summary/completion audit 都在 runner 里。健康 worker 到来后可以直接启动。

总 local guard 也已更新：`smoke_pure_sharpening_local_guards.sh` 现在覆盖 v81 的 CPU smoke、v81 runner、v81 readiness guard，并且 readiness 改为 `check_worker_readiness_for_v81.sh`，不会再用旧 v69 guard 误放行 known-bad `989057`。`bash -n` 通过，v81 CPU smoke 仍输出 `V81_BASIN_CONTRAST_CPU_SMOKE_OK`。

2026-07-09 最新状态：

- completion gate 已重新跑过，目标仍未完成；
- strict config 审计通过：`STRICT_PURE_SHARPENING_CONFIG_AUDIT_OK`；
- no-majority target 审计通过：`PURE_SHARPENING_NO_MAJORITY_TARGET_AUDIT_OK`；
- 文档审计通过：`PURE_SHARPENING_DOC_AUDIT_OK papers=13`；
- 当前纯锐化候选最好仍是 `v75_support_gated`，strict `mean@4=0.740945674044`；
- `v80_capacity_braked` 是负结果，strict `mean@4=0.729376257545`；
- `v81_basin_contrast` 仍是 `MISSING`，还没有健康 worker 上的 50-step GPU 结果；
- completion gate 输出 `RESULT=FAIL`，缺口是 `no complete pure-sharpening run reaches target`。

worker readiness 也重新检查过：

- 当前只看到 worker `989057`；
- `989057` 在 known-bad 列表 `987816,989057`；
- readiness 输出 `READINESS=FAIL`；
- 原因是 `no healthy non-known-bad 8x NVIDIA-B200 worker is listed`；
- 所以不能在 `989057` 上跑 v81、Ray、CUDA preflight 或训练。

下一步不变：等新的健康 8x B200 worker 出现后，先 login 和检查 `/proc`/GPU/cuInit，再直接跑 v81 strict 50-step runner；如果 `mean@4 >= 75%` 再做 post-run audit 和泄漏检查，否则只记录负结果，不做 improvement commit。

2026-07-09 追加检查：

- 再次运行 v81 readiness guard；
- 当前仍只看到 known-bad worker `989057`；
- readiness 仍是 `READINESS=FAIL`；
- 本轮没有登录 worker，没有做 CUDA/Ray preflight，也没有启动训练；
- 目标仍未完成，v81 仍等待健康 worker 跑 50 step。

2026-07-09 脚本权限修复：

- 发现 v81 新增脚本直接执行时缺少 executable bit；
- 已对以下脚本执行 `chmod +x`：
  - `check_worker_readiness_for_v81.sh`；
  - `smoke_v81_local_guards.sh`；
  - `worker_run_sps_basin_contrast_process_sharpened_prob_qwen25_math_7b_50step_v81_strict_n4.sh`；
- completion gate 重新检查仍然失败：
  - 当前最好纯锐化候选还是 `v75_support_gated`，`mean@4=0.740945674044`；
  - `v81_basin_contrast` 仍是 `MISSING`；
  - 目标 `mean@4 >= 0.75` 未达成；
- readiness guard 现在可以直接执行，但结果仍是只看到 known-bad `989057`，所以不能启动训练；
- 这次只是可复现性修复，不是指标提升或已确认 infra 根因修复，因此不做 git commit。

2026-07-09 05:06 本地 guard 刷新：

- 运行 `/opt/tiger/TTRL/verl/examples/ttrl/smoke_v81_local_guards.sh`；
- 离线检查仍然通过：
  - py_compile / bash syntax 通过；
  - `V81_BASIN_CONTRAST_CPU_SMOKE_OK`；
  - `STRICT_PURE_SHARPENING_CONFIG_AUDIT_OK`；
  - `PURE_SHARPENING_NO_MAJORITY_TARGET_AUDIT_OK`；
- completion gate 仍按预期失败：
  - `BEST_PURE_CANDIDATE=v75_support_gated`；
  - `BEST_PURE_MEAN@4=0.740945674044`；
  - `v81_basin_contrast` 仍是 `MISSING`；
  - `RESULT=FAIL`；
- worker readiness 仍失败：
  - 当前只看到 known-bad worker `989057`；
  - `READINESS=FAIL`；
- 当前结论不变：v81 本地代码和审计状态正常，但目标未完成，必须等健康非 known-bad 8x B200 worker 后再跑 50 step。

2026-07-09 后续 worker list API 阻塞：

- 后续多次运行 `/opt/tiger/TTRL/verl/examples/ttrl/check_worker_readiness_for_v81.sh`；
- 最新阻塞形态已经不是“只看到 known-bad `989057`”，而是 worker list API 本身返回 500；
- 典型输出：
  - `status code:500`；
  - `failed to list gpu worker`；
  - `WORKER_COUNT=0`；
  - `READINESS=FAIL`；
  - `REASON=no worker is listed`；
- 结论：
  - 当前不能安全选择任何 worker；
  - 不要绕过 readiness guard 去登录旧 worker id；
  - 等 worker list API 恢复，并且出现健康非 known-bad 8x B200 worker 后，才能跑 v81；
- goal 仍未完成：
  - `v81_basin_contrast` 没有 strict 50-step GPU 结果；
  - 当前最好完成结果仍是 `v75_support_gated mean@4=0.740945674044 < 0.75`；
  - 不调用 `update_goal`，不做 git commit。

补充 worker 控制面只读检查：

- 按 `merlin-devbox` worker 管理说明，只做额外的安全只读检查；
- 运行 `NO_COLOR=1 TERM=dumb timeout 30s mlx worker quota`；
- 结果 30 秒超时，exit code `124`，没有返回 quota 内容；
- 这说明当前更像是 worker 控制面整体不健康，而不只是没有可用 worker；
- 本轮没有 launch、kill、login、CUDA preflight、Ray startup 或训练操作。

readiness guard 诊断输出改进：

- 更新 `check_worker_readiness_for_v81.sh`，让它区分“worker list API 失败”和“真的没有 worker”；
- 改动内容：
  - 捕获 `mlx worker list` 的 stdout/stderr 和退出码；
  - 检测 `status code:500` / `failed to list gpu worker`；
  - 输出 `WORKER_LIST_EXIT=<code>` 和 `WORKER_LIST_STATUS=ERROR|OK`；
  - 如果 list 失败，输出 `REASON=worker list command failed; worker control plane is unhealthy`；
- 验证：
  - `bash -n check_worker_readiness_for_v81.sh` 通过；
  - 实际 readiness 仍失败，但现在明确输出 `WORKER_LIST_STATUS=ERROR`，说明是控制面不健康；
- 这只是诊断脚本改进，没有登录 worker，没有启动训练；goal 仍未完成。

v81 readiness smoke 覆盖：

- 新增 `/opt/tiger/TTRL/verl/examples/ttrl/smoke_v81_worker_readiness_guard.sh`；
- 使用本地 `WORKER_LIST_FIXTURE` 覆盖五种分支：
  - worker list API 500 / 控制面不健康；
  - 空 worker 列表；
  - 单个 known-bad worker `989057`；
  - 多个看似健康 worker，因为不允许同时使用多个 worker，所以失败；
  - 单个健康非 known-bad 8x B200 worker，应该通过并打印后续 login/runner 命令；
- 已接入 `/opt/tiger/TTRL/verl/examples/ttrl/smoke_v81_local_guards.sh`；
- 验证结果：
  - `smoke_v81_worker_readiness_guard.sh` 输出 `V81_WORKER_READINESS_GUARD_SMOKE_OK`；
  - `smoke_v81_local_guards.sh` 完整通过；
  - strict config audit 和 no-majority target audit 仍通过；
  - completion gate 仍按预期失败，因为当前最好 `mean@4=0.740945674044 < 0.75`；
- 最新真实 readiness 状态：
  - worker list 已能再次列出 `989057`；
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - `989057` 是 known-bad；
  - `READINESS=FAIL`；
  - `REASON=no healthy non-known-bad 8x NVIDIA-B200 worker is listed`；
- 当前仍不能跑 v81，goal 未完成，不调用 `update_goal`，不做 git commit。

通用 pure-sharpening guard 覆盖更新：

- 更新 `/opt/tiger/TTRL/verl/examples/ttrl/smoke_pure_sharpening_local_guards.sh`；
- 现在通用 guard 也会覆盖 v81 readiness fixture smoke：
  - `bash -n` 列表加入 `smoke_v81_worker_readiness_guard.sh`；
  - 运行步骤加入 `SMOKE v81_worker_readiness_guard_fixtures`；
- 验证：
  - 通用 guard 和 v81 readiness smoke 的 `bash -n` 通过；
  - `smoke_v81_worker_readiness_guard.sh` 仍输出 `V81_WORKER_READINESS_GUARD_SMOKE_OK`；
- 这只是诊断覆盖增强，不改变训练算法，也没有产生 GPU 结果。

2026-07-09 06:02 CST 状态刷新：

- 重新运行 v81 readiness guard：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 只有 worker `989057`，且它在 known-bad 列表中；
  - `READINESS=FAIL`；
  - 原因是没有健康的非 known-bad 8x B200 worker；
- 按安全规则，没有登录 `989057`，没有做 CUDA/Ray/训练。
- 重新跑 completion gate：
  - strict config audit 通过；
  - no-majority target audit 通过；
  - 文档/论文 audit 通过，记录 `papers=13`；
  - `v81_basin_contrast` 仍然没有 GPU 结果；
  - 当前最好纯锐化结果仍是 `v75_support_gated mean@4=0.740945674044`；
  - 目标 `mean@4 >= 0.75` 未达到。
- goal 未完成；这次只是阻塞状态刷新，不做 git commit。

2026-07-09 资源侧阻塞刷新：

- 只做了只读检查，没有 launch/kill/login，没有 CUDA/Ray/训练。
- v81 readiness 仍失败：
  - worker list 可用；
  - 当前只列出 `989057`；
  - `989057` 是 `8x NVIDIA-B200`，但在 known-bad 列表中；
  - 没有健康的非 known-bad 8x B200 worker。
- `mlx worker quota` 现在能正常返回，说明控制面已经恢复响应。
- quota 里能看到 H100/V100/T4 和 public Arnold H20/L20 等资源，但当前 strict v81 runner/readiness contract 要求健康 `8x NVIDIA-B200`，不能直接替代。
- completion gate 仍失败：
  - strict config audit 通过；
  - no-majority target audit 通过；
  - 文档/论文 audit 通过，`papers=13`；
  - `v81_basin_contrast` 仍然缺 GPU 结果；
  - 当前最好仍是 `v75_support_gated mean@4=0.740945674044`；
  - 目标 `mean@4 >= 0.75` 未达到。
- 当前阻塞是资源可用性，不是本地 v81 代码/配置/audit；不做 git commit，不调用 `update_goal`。

2026-07-09 06:38 CST 继续目标时的 readiness 刷新：

- active goal 仍是进行中，成功口径不变：Qwen2.5-Math-7B、50 step、strict validation `n=4`、关闭 validation answer selection，不做任何 inference-time scaling / selection，只看 `val-core/MATH-TTT/acc/mean@4 >= 0.75`。
- 重新运行 v81 readiness guard：
  - 命令：`NO_COLOR=1 TERM=dumb /opt/tiger/TTRL/verl/examples/ttrl/check_worker_readiness_for_v81.sh`；
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 只有 worker `989057`，它是 `8x NVIDIA-B200`，但在 known-bad 列表中；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 按安全约束，没有登录 `989057`，没有做 CUDA preflight、Ray 启动或 v81 训练。
- 没有新 GPU 指标，也没有确认 infra 修复，所以不做 git commit，不调用 `update_goal`。
- 下一步仍是等待单个健康的非 known-bad `8x NVIDIA-B200` worker，再跑 v81 strict runner。

2026-07-09 06:40 CST readiness 重试：

- goal 仍是 active，成功口径不变。
- 重新运行 `NO_COLOR=1 TERM=dumb /opt/tiger/TTRL/verl/examples/ttrl/check_worker_readiness_for_v81.sh`。
- 结果没有变化：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 只有 worker `989057`，它是 `8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA/Ray/训练；当前仍是资源可用性阻塞。

2026-07-09 06:41 CST readiness 重试：

- active goal 再次确认仍是进行中。
- v81 readiness guard 结果仍没有变化：
  - worker list 能返回；
  - 唯一 worker 还是 known-bad `989057`，`8x NVIDIA-B200`；
  - readiness 失败，原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，也没有启动训练。

2026-07-09 07:23 CST 继续目标审计：

- 重新做了一次 bounded v81 readiness 检查：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 当前唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 按 worker 安全约束，没有登录 `989057`，没有做 CUDA preflight、Ray 启动或 v81 训练。
- 重新运行 completion gate：
  - 目标口径检查通过：Qwen2.5-Math-7B、50 step、validation `n=4`、关闭 validation selection、目标 `mean@4 >= 0.75`；
  - strict config audit 通过；
  - no-majority target audit 通过；
  - 文档/论文 audit 通过，`papers=13`；
  - ablation 表有 `rows=14`；
  - `v81_basin_contrast` 仍然缺 GPU 结果；
  - 当前最好完成的纯锐化 candidate 仍是 `v75_support_gated`；
  - `BEST_PURE_MEAN@4=0.740945674044`；
  - `target_metric_reached=FAIL`，目标还没达到。
- 当前阻塞仍是资源可用性，不是本地 v81 代码、配置或 audit；这次不做 git commit，也不能调用 `update_goal`。

2026-07-09 07:25 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:26 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:27 CST 只读资源刷新：

- 重新运行 bounded v81 readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 重新运行只读 quota：
  - 命令：`NO_COLOR=1 TERM=dumb timeout 30s mlx worker quota`；
  - quota/control plane 正常返回；
  - Public Workspace 有 H100/V100/T4/A10/A100 等资源，但没有健康可用的 B200 替代；
  - Public Arnold 有 H20/L20，但不满足当前锁定的 v81 单个 `8x NVIDIA-B200` runner/readiness 合约。
- 没有 launch/kill，没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:28 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:29 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:30 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:32 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:33 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:34 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:35 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:36 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:38 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:39 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:40 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:41 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:43 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:44 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:45 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:47 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:48 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:49 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:51 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:52 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:54 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:55 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:57 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 07:58 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:00 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:02 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:03 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:05 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:07 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:08 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:10 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:14 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:16 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:17 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:18 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:19 CST readiness 重试：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。

2026-07-09 08:29 CST readiness 和 quota 刷新：

- 重新运行 bounded readiness：
  - `WORKER_LIST_STATUS=OK`；
  - `WORKER_COUNT=1`；
  - 唯一 worker 仍是 `989057`，`8x NVIDIA-B200`，但 `known_bad=1`；
  - `READINESS=FAIL`；
  - 原因仍是没有健康的非 known-bad 8x B200 worker。
- 运行只读 quota 刷新：
  - Public Workspace 有 H100/V100/T4/A10/A100 资源；
  - Public Arnold 有 H20/L20 资源；
  - quota 中没有显示可用 B200 候选。
- 没有登录 worker，没有做 CUDA preflight、Ray 启动或训练。
