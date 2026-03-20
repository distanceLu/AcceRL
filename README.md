# AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for VLA Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](#)

本仓库包含了 **AcceRL** 的官方实现代码。AcceRL 是一个专为大规模视觉-语言-动作 (Vision-Language-Action, VLA) 模型设计的分布式异步强化学习框架。 

当前，针对大规模 VLA 模型的强化学习微调面临着严重的计算瓶颈，这主要归因于传统同步 RL 框架中的同步屏障 (Synchronization barriers) 导致昂贵的 GPU 资源在等待缓慢的物理仿真器时处于空闲状态。为此，AcceRL 提出了一种完全异步且解耦的架构，并在分布式强化学习流水线中首次集成了可训练的世界模型 (World Model)，通过在“想象”中生成高保真虚拟体验，从根本上突破了物理仿真的采样效率瓶颈。

## 🌟 核心贡献 (Key Contributions)

AcceRL 在系统架构与算法设计上实现了双重突破：

### 1. 系统级解耦与极速吞吐 (Systemic Innovations)
* **宏观与微观全异步架构 (Fully Asynchronous Architecture)**：在宏观层面解耦了数据采集与策略更新，在微观层面隔离了环境交互与模型推理。消除了阻碍集群扩展的长尾效应 (Straggler effect)。
* **超线性扩展 (Super-linear Scaling)**：结合 ZeRO-2 优化与动态批处理机制，系统吞吐量在多 GPU 环境下展现出超线性扩展能力，硬件利用率始终维持在 94% 以上。

### 2. 算法保真与样本效率 (Algorithmic Innovations)
* **世界模型驱动的策略优化 (World-Model-Augmented RL)**：无缝集成基于 Diffusion 的状态转移模型与独立的奖励模型，使智能体能够在潜在空间中进行纯张量级别的大规模 Rollout。该机制将在线样本效率提升了惊人的 **200倍 (20,000%)**。
* **高斯重要性采样策略优化 (GIPO)**：针对异步框架中固有的策略延迟 (Policy lag) 问题，提出采用平滑的高斯信任域权重替代传统 PPO 的硬裁剪机制，从而在陈旧数据下仍能提供严格的稳定性和优越的收敛速度。
* **细粒度对齐**：引入词表精简 (Vocabulary Slimming)、Token 级别策略优化与价值重计算 (Value Re-computation) 机制，彻底消除自回归生成中的数值不稳定性。

## 📊 实验评估 (Empirical Evaluation)

### LIBERO 基准测试 SOTA (State-of-the-Art Performance)
在 MuJoCo 驱动的 LIBERO 异构机器人操作基准测试中，AcceRL 全面超越了以 OpenVLA-OFT 为代表的监督微调基线，以及最新的 RL 框架。特别是在极易发生协变量偏移的 `LIBERO-Long` 长视距任务中，AcceRL 维持了极高的成功率。

| 算法框架 | Spatial (%) | Object (%) | Goal (%) | Long (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Ours (AcceRL)** | **99.6** | **100.0** | 98.8 | **99.1** |
| SimpleVLA-RL | 99.4 | 99.8 | **99.2** | 98.5 |
| RLinf-VLA | 99.4 | 99.8 | 98.8 | 94.0 |
| OpenVLA-OFT | 96.2 | 98.3 | 96.2 | 90.7 |

*(详细消融实验与学习曲线请参阅论文 Section 6)*

## 🚀 快速开始 (Quick Start)

### 环境依赖安装
本框架的分布式调度基于 `ray` 构建，模型训练与推理基于 `torch`。为保证代码顺利运行，请确保您的环境满足 `requirements.txt` 中的指定版本要求：

```bash
# 推荐克隆仓库后，直接通过 requirements.txt 安装依赖
pip install -r requirements.txt

# 或者手动安装核心依赖：
pip install torch>=2.6.0 numpy>=2.2.6 ray>=2.54.0
```

### 运行极简示例
为了便于研究人员复现和理解架构，我们提供了包含 `FakeEnv` 和抽象 `FakeModel` 的精简版 Ray 分布式训练脚本。

**1. 运行无模型异步 RL (Model-Free Async RL):**
使用 GIPO 算法进行标准异步强化学习，包含完整的宏观/微观解耦流水线。

```bash
python main_ray_gipo.py \
    --num-rollout-workers 2 \
    --train-iters 20 \
    --train-batch-size 32 \
    --recompute-value
```

**2. 运行基于世界模型的 RL (Model-Based RL):**
部署包含 Policy Actor, Reward Actor 和 Denoiser (World Model) Actor 的完整微服务矩阵，执行“想象中学习”。

```bash
python main_mbrl_gipo.py \
    --num-rollout-workers 2 \
    --imagine-horizon 8 \
    --num-step-cond 4 \
    --train-iters 20
```

## 🤝 致谢 (Acknowledgments)
感谢以下开源项目对本框架的启发与支持：
* [RLinf](https://github.com/RLinf/RLinf)
* [AReaL](https://github.com/inclusionAI/AReaL)
* [OpenVLA](https://github.com/openvla/openvla)

## 📖 引用 (Citation)

如果您在学术研究中使用了 AcceRL 或其代码框架，请引用我们的论文：

```bibtex
@article{lu2026accerl,
  title={ACCERL: A DISTRIBUTED ASYNCHRONOUS REINFORCEMENT LEARNING AND WORLD MODEL FRAMEWORK FOR VISION-LANGUAGE-ACTION MODELS},
  author={Lu, Chengxuan and Wang, Shukuan and Li, Yanjie and Liu, Wei and Jin, Shiji and Qian, Fuyuan and Li, Peiming and Sun, Baigui and Liu, Yang},
  year={2026},
  journal={arXiv preprint}
}
```

## 📄 许可证 (License)

本项目采用 [MIT License](LICENSE) 授权。
