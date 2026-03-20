# AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for VLA Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](#)

本仓库包含了 **AcceRL** 的官方实现代码。AcceRL 是一个专为大规模视觉-语言-动作 (Vision-Language-Action, VLA) 模型设计的分布式异步强化学习框架。 

当前，针对大规模 VLA 模型的强化学习微调面临着严重的计算瓶颈，这主要归因于传统同步 RL 框架中的同步屏障 (Synchronization barriers) 导致昂贵的 GPU 资源在等待缓慢的物理仿真器时处于空闲状态。为此，AcceRL 提出了一种完全异步且解耦的架构，并在分布式强化学习流水线中首次集成了可训练的世界模型 (World Model)，通过在“想象”中生成高保真虚拟体验，从根本上突破了物理仿真的采样效率瓶颈。

##  核心贡献与系统架构 (Key Contributions & Architecture)

AcceRL 在系统架构与算法设计上实现了双重突破，其底层基于 Ray 构建了高度解耦的微服务 Actor 拓扑：

### 1. 系统级解耦与极速吞吐
* **宏观与微观全异步架构**：系统包含独立的 `TrainerActor`、`InferenceActor` 和 `RolloutWorkerActor`。环境交互与模型推理被物理隔离，彻底消除了阻碍集群扩展的长尾效应。
* **超线性扩展 (Super-linear Scaling)**：结合 ZeRO-2 优化与动态批处理机制，系统吞吐量在多 GPU 环境下展现出超线性扩展能力，硬件利用率始终维持在 94% 以上。

### 2. 世界模型驱动的“想象中学习” (Imagination Rollouts)
AcceRL 的 Model-Based 变体引入了独特的世界模型推演机制：
* **真实上下文初始化**：`RolloutWorkerActor` 首先与真实环境交互收集一条短轨迹，作为推演的初始视觉上下文。
* **潜在空间推演**：利用 `DenoiserInferenceActor` (转移模型) 预测下一帧虚拟观测，交由 `RewardInferenceActor` 输出成功率分布与终止信号，完成纯张量级别的虚拟环境步进。
* **极致样本效率**：该机制避免了与缓慢物理引擎的长时交互，将 VLA 模型的在线样本效率提升了惊人的 **200倍 (20,000%)**。

### 3. 高斯重要性采样策略优化 (GIPO)
* 针对异步框架中固有的策略延迟 (Policy lag) 问题，`TrainerActor` 采用平滑的高斯信任域权重替代传统 PPO 的硬裁剪机制。结合独立的价值重计算 (Value Re-computation) 和细粒度的 Token 级别优化，在陈旧数据下仍能提供严格的稳定性和优越的收敛速度。

##  实验评估 (Empirical Evaluation)

### LIBERO 基准测试 SOTA (State-of-the-Art Performance)
在 MuJoCo 驱动的 LIBERO 异构机器人操作基准测试中，AcceRL 全面超越了以 OpenVLA-OFT 为代表的监督微调基线，以及最新的 RL 框架。特别是在极易发生协变量偏移的 `LIBERO-Long` 长视距任务中，AcceRL 维持了极高的成功率。

| 算法框架 | Spatial (%) | Object (%) | Goal (%) | Long (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Ours (AcceRL)** | **99.6** | **100.0** | **98.8** | **99.1** |
| SimpleVLA-RL | 99.4 | 99.8 | 99.2 | 98.5 |
| RLinf-VLA | 99.4 | 99.8 | 98.8 | 94.0 |
| OpenVLA-OFT | 96.2 | 98.3 | 96.2 | 90.7 |

*(详细消融实验与学习曲线请参阅论文 Section 6)*

##  快速开始与本地复现 (Quick Start)

为便于研究人员在本地单机或无 GPU 环境下快速理解和验证 AcceRL 的数据流拓扑与 GIPO 算法数学逻辑，我们提供了剥离了庞大 OpenVLA/DeepSpeed 依赖的**精简版可独立运行脚本** (`main_ray_gipo.py` 和 `main_mbrl_gipo.py`)，它们内置了轻量级的 `FakeEnv` 与 `FakeModel`。

### 环境依赖安装
```bash
pip install ray torch numpy
```

### 1. 运行无模型异步 RL (Model-Free Async RL)
该脚本运行标准的分布式 GIPO 流水线。
```bash
python main_ray_gipo.py \
    --num-rollout-workers 1 \
    --num-eval-workers 1 \
    --train-iters 5 \
    --train-batch-size 16 \
    --accumulation-steps 2
```

### 2. 运行基于世界模型的 RL (Model-Based RL)
该脚本启动完整的世界模型 Actor 矩阵，展示 `Denoiser` 与 `RewardModel` 如何在虚拟空间中驱动策略更新。
```bash
python main_mbrl_gipo.py \
    --num-rollout-workers 1 \
    --imagine-horizon 4 \
    --num-step-cond 4 \
    --train-iters 5
```
*(注：默认配置下框架在纯 CPU 上运行。可通过 `--trainer-num-gpus 1` 或 `--inference-num-gpus 1` 等参数灵活分配显卡资源。)*

##  二次开发与模型替换指南 (Extension Guide)

当您需要在真实集群中训练大规模基础模型时，只需保持接口语义不变，逐步替换掉 Fake 组件：

1. **替换环境 (`FakeEnv` -> Real Simulator)**: 接入真实模拟器，保持 `reset()` 和 `step()` 的标准强化学习元组返回格式。
2. **替换转移动力学 (`FakeDenoiser` -> Real World Model)**: 在 `DenoiserInferenceActor` 中接入真实的 Diffusion 模型（例如 DIAMOND），接收历史 `obs_batch` 与 `act_batch`，输出高保真 `next_obs`。
3. **替换奖励模型 (`FakeRewardModel` -> Real RM)**: 在 `RewardInferenceActor` 中接入真实奖励模型，确保输出 `[B, 2]` 的 logits，以便框架提取 `succ_prob` 和 `end` 信号。
4. **替换策略网络 (`FakeActorCritic` -> OpenVLA)**: 实现真实 VLA 模型的 `forward` 与 `post_process` 接口，输出离散 `action_tokens` 与用于环境执行的连续 `action_env`。

替换完成后，分布式 Actor 框架和 GIPO 核心训练逻辑无需任何更改即可无缝扩展。

##  引用 (Citation)

如果您在学术研究中使用了 AcceRL 或其代码框架，请引用我们的论文：

```bibtex
@article{lu2026accerl,
  title={ACCERL: A DISTRIBUTED ASYNCHRONOUS REINFORCEMENT LEARNING AND WORLD MODEL FRAMEWORK FOR VISION-LANGUAGE-ACTION MODELS},
  author={Lu, Chengxuan and Wang, Shukuan and Li, Yanjie and Liu, Wei and Jin, Shiji and Qian, Fuyuan and Li, Peiming and Sun, Baigui and Liu, Yang},
  year={2026},
  journal={arXiv preprint}
}
```

##  许可证 (License)

本项目采用 [MIT License](LICENSE) 授权。
