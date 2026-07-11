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

# 🚀 在 ByteDance modelchef venv + B200 上运行（内部环境适配）

> 本章节记录如何在**复用 `modelchef` 的 uv 虚拟环境**（不使用 TTRL 自带的 conda 安装方式）、并在 **8×B200（Blackwell, sm_100）** 机器上跑通 TTRL，以及为此环境所做的全部适配与踩坑修复。
>
> 核心原则：**TTRL 使用自己的 verl 代码（`/opt/tiger/TTRL/verl`，verl 0.4.1），只借用 modelchef 的 venv 作为依赖环境（torch 2.9 / vLLM 0.14 / CUDA 12.9）。modelchef 仓库本身零改动。**

## 为什么要这样做

TTRL 官方依赖栈（`scripts/install_ttrl_deps.sh`）是 `torch==2.6.0 + cu124 + vllm==0.8.5.post1`，**这套在 B200（Blackwell）上无法运行**——Blackwell 需要 CUDA 12.8+ / torch 2.7+ 才有对应 kernel。

而 `modelchef` 的 venv 恰好提供了能在 B200 上运行的栈：

| 组件 | modelchef venv | TTRL 官方要求 | B200 可用 |
| --- | --- | --- | --- |
| Python | 3.11 | 3.10 | ✅ |
| torch | 2.9.1+cu129 | 2.6.0+cu124 | 仅 2.9 可跑 B200 |
| vLLM | 0.14.1 (v1 engine) | 0.8.5.post1 | 仅新版支持 B200 |
| transformers | 4.57.3 | >=4.51 | ✅ |
| tensordict | 0.10.0 | <=0.6.2 | ✅ |

因此“借用 modelchef venv”是 B200 上运行 TTRL 的现实可行路径。

## 启动方式

### 1. 准备数据（JSON → Parquet）

所有数据集（AIME / AMC / MATH 全系列）转换为 verl 需要的 parquet 格式：

```bash
cd /opt/tiger/TTRL/verl/data
/opt/tiger/modelchef/.venv/bin/python - <<'PY'
import os, datasets
def mk(split, source):
    def fn(ex, idx):
        return {'data_source': source,
                'prompt':[{'role':'user','content': ex['prompt']}],
                'ability':'math',
                'reward_model':{'style':'rule','ground_truth': str(ex['answer'])},
                'extra_info':{'split':split,'index':f'{source}-{idx}'}}
    return fn
for src in ['AIME-TTT','AMC-TTT','MATH-TTT','MATH-L1-TTT','MATH-L2-TTT','MATH-L3-TTT','MATH-L4-TTT','MATH-L5-TTT']:
    for split in ['train','test']:
        p=os.path.join(src, split+'.json')
        if not os.path.exists(p): continue
        ds=datasets.load_dataset('json', data_files=p, split='train')
        ds=ds.map(mk(split,src), with_indices=True, remove_columns=ds.column_names)
        ds.to_parquet(os.path.join(src, split+'.parquet'))
    print(src, 'done')
PY
```

### 2. 启动训练

本仓库在 `examples/ttrl/` 下提供了三个适配好的脚本（已内置全部 B200 环境变量与 vLLM v1 兼容配置）：

| 脚本 | 用途 | 规模 |
| --- | --- | --- |
| `examples/ttrl/smoke_aime_qwen3_4b.sh` | 冒烟测试，最快验证环境 | 2 卡 / 1 step |
| `examples/ttrl/run8_aime_qwen3_4b.sh` | 8 卡吞吐 / 稳定性测试 | 8 卡 / AIME |
| `examples/ttrl/run_paper_math_qwen3_4b.sh` | **论文级正式实验** | 8 卡 / MATH 全量 500 题 |

启动正式实验（注意用 `setsid` 让任务脱离当前 shell 进程组，避免被误杀）：

```bash
cd /opt/tiger/TTRL/verl
setsid bash examples/ttrl/run_paper_math_qwen3_4b.sh > run_paper.log 2>&1 < /dev/null &
```

监控测评曲线：

```bash
F=$(ls -t /tmp/ray/session_latest/logs/worker-*.out | head -10)
for f in $F; do grep -E "val-core/MATH-TTT/acc/mean@4" "$f"; done \
  | grep -oE "training/global_step:[0-9]+|val-core/MATH-TTT/acc/mean@4:[0-9.]+"
```

### 3. 关键环境变量（已写入脚本，单独运行时需手动 export）

```bash
# 依赖来自 modelchef venv，但 import verl 走 TTRL 自己的代码
export PYTHONPATH=/opt/tiger/TTRL/verl:${PYTHONPATH}
PY=/opt/tiger/modelchef/.venv/bin/python   # 用 modelchef 的解释器

export VLLM_USE_V1=1

# 本容器为 IPv6-only（eth0 无 IPv4）：c10d store 走 loopback，NCCL bootstrap 走 eth0/IPv6
export MY_HOST_IP=127.0.0.1                 # verl Worker._get_node_ip() 优先用它，避免解析到不可达的集群主机名
export MASTER_ADDR=127.0.0.1
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET6          # 关键：让 NCCL bootstrap 用 IPv6，否则报 "no socket interface found"
export NCCL_IB_DISABLE=1
```

## 本环境下做过的适配与踩坑修复

> 以下改动**全部在 `/opt/tiger/TTRL/verl` 内**，modelchef 仓库未改动。

### A. vLLM 0.8 → v1 (0.14.1) API 兼容（import 期报错）

TTRL 的 rollout/sharding 层引用了 vLLM 0.8 已删除的符号，做了容错改写：

| 文件 | 旧 import | 新 import（容错） |
| --- | --- | --- |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | `vllm.worker.worker_base.WorkerWrapperBase` | 回退到 `vllm.v1.worker.worker_base` |
| 同上 | `vllm.model_executor.sampling_metadata.SamplingMetadata` | 仅类型注解，缺失时降级为 `Any` |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `vllm.worker.worker_base.WorkerWrapperBase` | 回退到 `vllm.v1.worker.worker_base` |
| `verl/utils/vllm_utils.py` | `vllm.lora.models.LoRAModel` | 回退到 `vllm.lora.lora_model` |

### B. vLLM EngineArgs 参数不兼容（运行期报错）

新版 vLLM 移除了 `disable_mm_preprocessor_cache` 参数。脚本中通过 hydra 删除该 key（代码会自动跳过值为 None 的项）：

```
~actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache
```

### C. ttrl_math 判分依赖补装

ttrl 的数学判分需要 `math-verify`、`latex2sympy2_extended`（modelchef venv 默认没有）：

```bash
cd /opt/tiger/modelchef && \
  http_proxy=http://sys-proxy-rd-relay.byted.org:8118 \
  https_proxy=http://sys-proxy-rd-relay.byted.org:8118 \
  no_proxy=".byteintl.net,.byted.org" \
  uv pip install --python .venv/bin/python math-verify latex2sympy2_extended mathruler
```

### D. 单机分布式网络（c10d / NCCL）

- **c10d TCPStore 连不上**：容器里 verl 默认用 ray SDK 解析 node ip，得到不可达的集群主机名/IPv6 → 设 `MY_HOST_IP=127.0.0.1` 让 rendezvous 走 loopback。
- **NCCL `no socket interface found`**：本容器 eth0 仅有 IPv6 → 设 `NCCL_SOCKET_IFNAME=eth0` + `NCCL_SOCKET_FAMILY=AF_INET6`。

### E. GPU 驱动 / 环境故障（机器层面，非代码问题）

> 这是宿主机不定期刷驱动 + 容器重启导致的环境问题，**每次 worker 重启后可能需要重新处理**。

1. **`ld.so` compat 库与内核驱动版本错配**（`cuInit` 报 803 / `torch.cuda.is_available()=False`）。宿主驱动会在 580 / 570 之间变动，需按当前驱动版本动态校准 ld.so compat：

   ```bash
   DRV=$(cat /proc/driver/nvidia/version | head -1 | grep -oE '[0-9]{3}\.[0-9]+\.[0-9]+' | head -1)
   DRV_MAJOR=${DRV%%.*}
   if [ "$DRV_MAJOR" -ge 580 ]; then
       # 驱动 ≥ 580 原生支持 cu12.9，清掉过时的 compat
       for f in /etc/ld.so.conf.d/00-compat-*.conf; do [ -f "$f" ] && sudo truncate -s 0 "$f"; done
   else
       # 驱动 < 580，必须靠 compat 575 才能跑 cu12.9
       echo "/usr/local/cuda-12.9/compat" | sudo tee /etc/ld.so.conf.d/00-compat-active.conf
   fi
   sudo ldconfig
   ```

2. **`/proc` 从 mount namespace 脱落**（`cuInit` 报 304 / `/proc/self` 不可读）：容器级故障，用户态无法修复，需**重启 worker**。

**worker 重启后的标准恢复流程**：

```bash
# 1) 按上面 E.1 校准 ld.so compat
# 2) 验证 CUDA 可用
/opt/tiger/modelchef/.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # 期望 True
# 3) 重新启动训练
cd /opt/tiger/TTRL/verl && setsid bash examples/ttrl/run_paper_math_qwen3_4b.sh > run_paper.log 2>&1 < /dev/null &
```

## 复现结果（qwen3_4b, MATH-TTT 500 题全量, 8×B200）

完整测评曲线见 `examples/ttrl/math_acc_curve.csv`。

| 阶段 | MATH 测试集 acc@4 |
| --- | --- |
| 训练前 baseline (step 0) | **53.2%** |
| step 100 | 75.1% |
| step 150 | 86.0% |
| 峰值 (step ~425) | **90.8%** |
| 收敛区 (step 250+) | ~88–89% |

- **全程无真实标签**，奖励完全来自模型自身 64 次采样的多数投票（TTRL 核心机制），测试集准确率 **53% → 90.8%**，复现了论文效应。
- 吞吐 ~2200–2500 tok/s，8 卡满载，连续运行 ~19 小时 / 1000+ step 无崩溃。
- 注：本环境用 `qwen3_4b`（通用模型）而非论文的 `Qwen2.5-Math-7B`，故绝对数值与论文不同，但提升趋势一致。

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
