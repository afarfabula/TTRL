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
