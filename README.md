<!--
This section documents the local experiment branch maintained at
experiment/ttrl-workspace-history-20260807. The original upstream TTRL README
is kept below for paper/project context.
-->

# 本分支 Claims：TTRL / Reasoning Test-Time RL 工程复现与方法探索

本分支围绕 TTRL（Test-Time Reinforcement Learning）在数学推理上的复现、扩展和大模型工程化展开。核心目标不是只跑通论文脚本，而是把“无 GT 测试时强化学习 / GRPO / self-consistency reward / reasoning test-time scaling”这条链路拆成可验证的工程 claim：哪些结果可以写进简历，哪些只是负结果或基础设施经验。

## Claim 1：在 8xB200 上复现并稳定化 Qwen2.5-Math-7B 的 TTRL/MATH500 训练链路

本分支完成了 MATH-TTT/MATH500 数据、verl 训练、vLLM rollout、reference logprob、actor update 和 validation 的端到端链路整理，并保留了可复跑脚本与日志。

Evidence：

| Run | Setting | Step | mean@16 | maj@16 | best@16 |
| --- | --- | ---: | ---: | ---: | ---: |
| Paper-style MV baseline | Qwen2.5-Math-7B, B32/R32/V64, val_n=16 | 80 | `0.8240` | `0.852` | `0.889` |
| Paper-style MV baseline | Qwen2.5-Math-7B, B32/R32/V64, val_n=16 | 150 | `0.8275` | `0.853` | `0.885` |
| Sharpened MV-anchor soft0.02 | Qwen2.5-Math-7B, B32/R32/V64, val_n=16 | 140 | `0.844` | `0.864` | `0.888` |
| Sharpened MV-anchor soft0.02 | Qwen2.5-Math-7B, B32/R32/V64, val_n=16 | 150 | `0.842125` | `0.866578` | `0.881822` |

结论：

- `sharpened mv-anchor soft0.02` 在 mean@16 和 maj@16 上超过当前 MV baseline，step80 相比历史 MV 约 `+0.012 / +0.012 / +0.003`。
- 最终 step150 达到 `mean@16=0.842125`、`maj@16=0.866578`，是当前 TTRL-native 线中最强的稳定结果之一。
- 8xB200 训练链路中保留了 validation every 20 steps、rollout old-logprob diff monitor、static batch 语义等 guardrail。

边界：

- 该 run 没达到预设 `mean@16 >= 0.85` 目标，不能写成突破性 SOTA。
- `best@16` 后期下降，说明 sharpened reward 在提升 majority/mean 的同时可能压缩了探索多样性。

## Claim 2：full-rollout posterior target + weighted NLL 是有效训练路径，PowerFlow squared-delta 在 full-response 上不稳定

本分支对 full-rollout target 的 actor objective 做了诊断。结果显示，同一类无 GT posterior target 在 weighted NLL 下能稳定传递训练信号，而直接用 full-response PowerFlow squared-delta 会显著拉低验证指标。

Evidence：

| Run | Target | Actor objective | mean@16 | maj@16 | best@16 |
| --- | --- | --- | ---: | ---: | ---: |
| FR-B0 | posterior sharpen | PowerFlow squared-delta | `0.45325` | `0.57668` | `0.83677` |
| FR-D0 | oracle correctness | PowerFlow squared-delta | `0.398875` | `0.503472` | `0.814348` |
| FR-D0-NLL | oracle correctness | weighted NLL | `0.69275` | `0.797464` | `0.904338` |
| FR-B0-NLL | posterior sharpen, no GT | weighted NLL | `0.68800` | `0.793828` | `0.908718` |
| FR-B2-NLL | margin-aware posterior, no GT | weighted NLL | `0.692875` | `0.794484` | `0.899186` |

结论：

- full-rollout target 构造链路本身可用，oracle-NLL 和无 GT posterior-NLL 能恢复到接近的指标区间。
- 失败点主要在 full-response PowerFlow squared-delta objective，而不是 reward estimation 或 rollout 数据链路完全不可用。
- 后续方法应优先围绕 weighted NLL / distribution matching 改 target，而不是继续扩大这版 squared-delta。

边界：

- FR-B0-NLL / FR-B2-NLL 仍低于强 MV-anchor TTRL 主线，不能作为最终主结果。
- Oracle correctness 只用于诊断，不属于正式无 GT 方法。

## Claim 3：chunk-level search-state TTRL 的工程链路已打通，但当前 target 质量不足

本分支实现了 chunk-level search-state 方向，包括 mid-state 构造、candidate generation、support mass / future gain / TV transport / support-anchor 等 scorer，并把 chunk target 接入 actor update。

Evidence：

| Direction | Evidence | Result |
| --- | --- | --- |
| chunk-state pipeline | `_make_chunk_state_prompts`、boundary/source metadata、diag JSONL、chunk actor batch 写入 `powerflow_flat_weights` | 多轮 3-step smoke 和 20-step pilot 可运行 |
| support-anchor no-probe | 不依赖 short-probe teacher，score 来自 full rollout support mass | `chunk_state_probe/skipped_for_support_anchor=1.0` |
| support-anchor 20-step | B32/R32/V64, val_n=16 | `mean@16=0.43725` / `maj@16=0.558596` / `best@16=0.83514` |
| infra overhead | stable steps 2-19 | chunk score 约 `0.001s`，chunk ref 约 `0.965s`，update_actor 约 `3.057s` |

结论：

- chunk-level TTRL 的基础设施是正结果：状态构造、candidate、scoring、actor batch 和诊断日志都已经可跑。
- `hardfilter + clip4` 可以稳定权重并降低 actor update 开销。
- 但当前 support-anchor / local-probe / gate 堆叠没有形成有效的 search-improvement signal，不应扩到 80-step 或包装成最终算法收益。

边界：

- 当前 chunk-level 结果是负结果定位：主矛盾是 target 没有表达 full-rollout future distribution improvement，而不是纯工程吞吐。
- 不应声称“chunk-level TTRL 已提升 MATH500”，只能声称“完成了工程闭环并定位了 target quality 问题”。

## Claim 4：Qwen3-30B-A3B MoE 的本地评测和 Expert-Sample 复现完成了可追问的负结果验证

本分支在本地可用的 `/tmp/Qwen3-30B-A3B-Base` 上完成了 MATH500 和 GPQA-Diamond 的 n=16 评测，并实现了 vLLM router 侧 Expert-Sample 环境变量开关。

Evidence：

| Run | Dataset | n | mean_acc | pass@n | elapsed |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B-Base | MATH500 | 16 | `37.25%` | `72.00%` | `3062s` |
| Expert-Sample keep7/pool16/tau0.5 | MATH500 | 16 | `34.59%` | `71.20%` | `4214s` |
| Expert-Sample keep5/pool32/tau0.5 stochastic | MATH500 | 16 | `32.45%` | `71.00%` | `4199s` |
| Qwen3-30B-A3B-Base | GPQA-Diamond | 16 | `28.38%` | `90.40%` | `1054s` |
| Expert-Sample keep5/pool32/tau0.5 stochastic | GPQA-Diamond | 16 | `28.88%` | `89.39%` | `1417s` |

结论：

- vLLM router patch 确认命中过 Expert-Sample 路由逻辑，支持 deterministic 和 stochastic tail sampling。
- 在本地 Base 模型、n=16、无 verifier Best-of-N 的设置下，没有复现论文 headline gain。
- 这是一条有价值的 reproduction audit：早期 subset gain 没有在 full MATH500 / GPQA-Diamond 上保持。

边界：

- 论文 headline 使用 `Qwen3-30B-A3B-Instruct`、`GPQA-Diamond pass@32` 和 verifier Best-of-N；本地只有 Base checkpoint，因此不能声称严格复现失败。
- 该部分应写成“本地可用条件下的复现审计和负结果”，不是论文结论否定。

## Claim 5：Qwen3-30B-A3B TTRL 大模型训练链路完成 smoke，但全参高吞吐训练仍受 sharding/offload 约束

本分支把 Qwen3-30B-A3B MoE 接入 TTRL/verl 训练链路，完成权重、vLLM、rollout、ref logprob 和 actor update 的 2xB200 smoke。

Evidence：

| Setting | Result |
| --- | --- |
| Model | `/tmp/Qwen3-30B-A3B-Base` |
| Hardware | 2 x B200, about 183GB per GPU |
| Working strategy | FSDP2 + `offload_policy=True` |
| Completed training | 1 TTRL training step |
| GPU memory after FSDP2 offload | about `36GB allocated` / `45GB reserved` |
| CPU memory used | about `1374GB` |
| Step time | about `325s`, with `update_actor=256.548s` |

结论：

- 两卡 B200 能跑通 Qwen3-30B-A3B 的 TTRL smoke 闭环。
- FSDP1 `optimizer_offload=True` 不能解决 30B 全参 Adam 首次 state 初始化峰值；FSDP2 offload 才是当前可行 smoke 路线。
- 两卡 B200 更适合兼容性验证，不适合高吞吐全参 RL 训练。

边界：

- 当前 Qwen3-30B-A3B 只是 smoke，不是长训练结果。
- 若要形成最终训练 claim，需要 8 卡 H100/B200 更稳定的 Megatron/FSDP2/ZeRO 路线和完整 validation 曲线。

## 简历写法

可以直接使用的中文项目经历：

```text
Test-Time RL / GRPO 数学推理后训练：基于 TTRL/verl 构建无 GT 测试时强化学习实验链路，在 8xB200 上复现 Qwen2.5-Math-7B + MATH-TTT/MATH500 的 rollout、majority-vote reward、reference logprob、actor update 和 val@16 评估流程；稳定运行 B32/R32/V64 配置，并通过日志化脚本记录 step time、target health、rollout diff 和验证指标。

设计并验证 sharpened MV-anchor reward：在 150-step pilot 中达到 mean@16=0.8421、maj@16=0.8666，相比 paper-style MV baseline 在 mean/majority 指标上取得稳定增益；同时保留 best@16 下降和未达 0.85 mean@16 目标的边界，形成可复盘的实验结论。

系统排查 full-rollout PowerFlow 与 chunk-level search-state TTRL：通过 oracle target、posterior target、weighted NLL、PowerFlow squared-delta、support-anchor、future-support gain 等 ablation，定位 full-response squared-delta 和 chunk target quality 是主要瓶颈；将失败路径转化为后续 distribution matching / long-horizon target 设计依据。

扩展到 Qwen3-30B-A3B MoE：完成 Base checkpoint 的 MATH500/GPQA n=16 评测与 vLLM Expert-Sample router patch，验证本地 Base 模型下 Expert-Sample 未复现 headline gain；同时打通 2xB200 Qwen3-30B-A3B TTRL smoke，定位 FSDP1 optimizer offload 不足与 FSDP2 offload 的可行边界。
```

更短的英文 bullet：

```text
Built a TTRL/verl-based test-time RL pipeline for mathematical reasoning,
covering rollout generation, majority-vote reward construction, reference
log-prob computation, actor updates, and val@16 evaluation on 8xB200; developed
a sharpened MV-anchor GRPO variant that reached 84.2% mean@16 and 86.7% maj@16
on MATH500, while systematically auditing full-rollout PowerFlow, chunk-level
search-state targets, and Qwen3-30B-A3B MoE scaling failure modes.
```

## 代码与证据索引

- 训练脚本：`verl/run_records/`
- 中文实验记录：`important_experiment_logs/`
- Qwen3-30B-A3B 评测 summary：`important_experiment_logs/qwen3_30b_a3b_*_summary.json`
- B200 环境与稳定运行配置：`important_experiment_logs/B200_ENV_USAGE.md`
- chunk-level 审计：`important_experiment_logs/chunk_level_search_state_ttrl_24h_audit_20260801.md`
- full-rollout TTRL 审计：`important_experiment_logs/full_rollout_ttrl_12h_progress_20260802.md`

---

下面保留原始 TTRL 项目 README，作为论文背景和官方使用说明。

<div align="center">

# TTRL: Test-Time Reinforcement Learning

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.16084)  [![Github](https://img.shields.io/badge/TTRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/PRIME-RL/TTRL)
[![Wandb Log of AIME](https://img.shields.io/badge/Wandb%20Log%20of%20AIME-%2300B4AB?style=for-the-badge&logo=weightsandbiases&logoColor=white&labelColor=000000)](https://wandb.ai/truman-yx-zuo-nlp/TTRL/workspace?nw=nwusertrumanyxzuo) [![HF Papers](https://img.shields.io/badge/HF--Paper-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/papers/2504.16084)  [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/zuo_yuxin/status/1915406839669572036)

</div>

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a> •
    <a href="#main-results" style="text-decoration: none; font-weight: bold;">📊 Main Results</a>
  </p>
  <p>
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#contact" style="text-decoration: none; font-weight: bold;">📨 Contact</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a> •
    <a href="#star-history" style="text-decoration: none; font-weight: bold;">🌟 Star History</a>
  </p>
</div>

> Welcome to the Era of Experience.  --David Silver, Richard S. Sutton

# 🎉News
- **[2026-03-10]** We investigate the mechanisms and potential applications of [Unsupervised RLVR (URLVR)](https://arxiv.org/pdf/2603.08660), and find that it is particularly well suited for test-time training and quantifying model priors. Here is [code](https://github.com/PRIME-RL/TTRL/tree/urlvr-dev). URLVR paper is accepted to [ICLR 2026](https://iclr.cc/Conferences/2026)!
- **[2025-09-18]** TTRL paper is accepted to [NeurIPS 2025](https://neurips.cc/Conferences/2025)!
- **[2025-08-17]** We bump into [verl v0.4.1](https://github.com/volcengine/verl/releases/tag/v0.4.1), and now you can enable TTRL by simply setting `+ttrl.enable=True`!
- **[2025-05-23]** We update both the paper and the code, with the implementation based on the [verl](https://github.com/volcengine/verl).
- **[2025-04-24]** We release the code and experimental logs. Check it out: [Getting Started](#getting-started).
- **[2025-04-23]** We present **TTRL** (Test-Time Reinforcement Learning), an open-source solution for online RL on data without ground-truth labels, especially test data.

# 📖Introduction

**We investigate Reinforcement Learning (RL) on data without explicit labels for reasoning tasks in Large Language Models (LLMs).**
The core challenge of the problem is reward estimation during inference while not having access to ground-truth information. While this setting appears elusive, we find that common practices in Test-Time Scaling (TTS), such as majority voting, yield surprisingly effective rewards suitable for driving RL training.

<p align="center">
   <img src="figs/teaser.png" alt="Performance and settings of TTRL." style="width: 80%;">
</p>


<p align="center">
   <img src="figs/overview.png" alt="Overview of TTRL." style="width: 80%;">
</p>


# 📊Main Results

Our experiments demonstrate that TTRL consistently improves performance across a variety of tasks and models. Notably, TTRL boosts the `pass@1` performance of Qwen-2.5-Math-7B by approximately 211% on `AIME 2024` with only unlabeled test data.

Furthermore, although TTRL is only supervised by the `maj@n` metric, TTRL has demonstrated performance to consistently surpass this upper limit of the initial model, and approach the performance of models trained directly on test data with ground-truth labels.

<p align="center">
   <img src="figs/results.png" alt="Main results of TTRL." style="width: 60%;">
</p>


# ✨Getting Started

## Env Setup

```bash
git clone https://github.com/PRIME-RL/TTRL.git

cd TTRL/verl

conda create -n ttrl python==3.10
conda activate ttrl
bash scripts/install_ttrl_deps.sh
pip install -e .
```

## Reproduce TTRL
You can reproduce the results on `AIME 2024` with the following commands:

```bash
bash examples/ttrl/Qwen2.5/aime.sh
```

> [!NOTE]
> - You can use the script [verl/data/preprocess.py](https://github.com/PRIME-RL/TTRL/blob/main/verl/data/preprocess.py) to convert data from the `JSON` format to the `Parquet` format for training with verl.
> - We provide scripts in the [verl/examples/ttrl](https://github.com/PRIME-RL/TTRL/tree/main/verl/examples/ttrl) directory for running TTRL on multiple models across various benchmarks.
> - For further details regarding the code, please refer to the [verl documentation](https://verl.readthedocs.io/en/latest/index.html).

We additionally conducted three independent runs using the preview version of our code. Two of the runs achieved a pass@1 (greedy) of 43.3, while one run reached 46.7. Please refer to the [Weights & Biases logs](https://wandb.ai/truman-yx-zuo-nlp/TTRL/workspace).

*All experiments were conducted on 8 x NVIDIA A100 80GB GPUs.*

<details>
<summary>
  Pseudo-Code
</summary>

The implementation of TTRL can be achieved rapidly by simply modifying the reward function. Please refer to the following code snippet for details:

<p align="center">
   <img src="figs/ttrl_reward.png" alt="The pseudo-code of the majority voting reward function." style="width: 60%;">
</p>
</details>

# 📨Contact

- Kaiyan Zhang: zhang-ky22@mails.tsinghua.edu.cn
- Ning Ding: dingning@mail.tsinghua.edu.cn

# 🎈Citation
If you find TTRL helpful, please cite us.

```bibtex
@article{zuo2025ttrl,
  title={Ttrl: Test-time reinforcement learning},
  author={Zuo, Yuxin and Zhang, Kaiyan and Sheng, Li and Qu, Shang and Cui, Ganqu and Zhu, Xuekai and Li, Haozhan and Zhang, Yuchen and Long, Xinwei and Hua, Ermo and others},
  journal={arXiv preprint arXiv:2504.16084},
  year={2025}
}
```

# 🌟Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PRIME-RL/TTRL&type=Date)](https://www.star-history.com/#PRIME-RL/TTRL&Date)
