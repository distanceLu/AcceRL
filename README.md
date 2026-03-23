# AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for VLA Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2603.18464)

This repository contains the official implementation of **AcceRL**, a distributed asynchronous reinforcement learning (RL) framework specifically designed for large-scale **Vision-Language-Action (VLA)** models.

Currently, RL fine-tuning for large-scale VLA models faces significant challenges in computational efficiency and data acquisition. Traditional synchronous RL frameworks suffer from "synchronization barriers," where high-performance GPU resources remain idle while waiting for slow physical simulators. AcceRL addresses these bottlenecks through a fully asynchronous and decoupled architecture. Crucially, it is the first framework to integrate a plug-and-play, trainable **World Model** into a distributed asynchronous RL pipeline to generate virtual experiences, effectively bypassing the sampling limits of physical simulation.

##  Key Contributions & Architecture

AcceRL achieves breakthroughs in both system infrastructure and algorithmic design, utilizing a highly decoupled microservice Actor topology based on **Ray**:

### 1. System-Level Decoupling & High Throughput
* **Macro and Micro Asynchrony**: The system consists of independent `TrainerActor`, `InferenceActor`, and `RolloutWorkerActor`. By physically isolating environment interaction, model inference, and policy training, AcceRL eliminates the "straggler effect" and synchronization overhead.
* **Super-linear Scaling**: Utilizing ZeRO-2 optimizations and dynamic batching, AcceRL exhibits super-linear scaling in throughput as the cluster expands. [cite_start]Hardware utilization remains consistently high, exceeding **94%** across multi-GPU nodes.

### 2. World Model Driven "Learning in Imagination"
The Model-Based variant of AcceRL introduces a unique imagination rollout mechanism:
* **Real-World Context Initialization**: The `RolloutWorkerActor` first interacts with the real environment to collect a short trajectory, which serves as the initial visual context for imagination.
* **Pixel-Level High-Fidelity Rollouts**: The system does not rely on compressed latent spaces. Instead, it utilizes a `DenoiserInferenceActor` (a Diffusion-based transition model) to predict high-fidelity pixel-level observations. These are processed by the `RewardInferenceActor` to output success probabilities and termination signals, enabling pure tensor-level environment stepping on GPUs.
* [cite_start]**Extreme Sample Efficiency**: This mechanism bypasses slow physical simulators, boosting the online sample efficiency of VLA models by an astonishing **200x (20,000%)**.

### 3. Gaussian Importance sampling Policy Optimization (GIPO)
* **Mitigating Policy Lag**: To handle the stale data inherent in asynchronous architectures, the `TrainerActor` implements **GIPO**. Unlike standard PPO's hard clipping, GIPO uses a smooth Gaussian trust weight to softly damp extreme importance ratios, providing superior stability and convergence.
* [cite_start]**Algorithmic Fidelity**: The trainer performs **Value Re-computation** and **Token-level optimization** to ensure numerical robustness and precise credit assignment during the autoregressive generation process of large VLA backbones.

##  Empirical Evaluation

### LIBERO Benchmark SOTA Performance
AcceRL consistently outperforms standard OpenVLA-OFT baselines and contemporary RL frameworks on the LIBERO robot manipulation benchmark. [cite_start]Notably, it maintains a **99.1%** success rate in the challenging `LIBERO-Long` suite.

| Framework | Spatial (%) | Object (%) | Goal (%) | Long (%) |
| :--- | :---: | :---: | :---: | :---: |
| **AcceRL (Ours)** | **99.6** | **100.0** | 98.8 | **99.1** |
| SimpleVLA-RL | 99.4 | 99.8 | **99.2** | 98.5 |
| RLinf-VLA | 99.4 | 99.8 | 98.8 | 94.0 |
| OpenVLA-OFT | 96.2 | 98.3 | 96.2 | 90.7 |

*(For detailed ablation studies and learning curves, please refer to Section 6 of the [official paper](https://arxiv.org/abs/2603.18464)).*

## Quick Start & Local Reproduction

To help researchers understand the data flow topology and GIPO logic, we provide **minimal, standalone scripts** (`main_ray_gipo_ds_standalone.py` and `main_mbrl_gipo_ds_standalone.py`) that remove heavy dependencies and utilize built-in `FakeEnv` and `FakeModel` components.

### Installation
You can download and use AccRL by the following steps"
1. Download the Code
   ```bash
  git clone https://github.com/distanceLu/AcceRL.git
  cd AcceRL
```
2. Build the Environment:
 ```bash
   conda create -n accrl_env python=3.10 -y 
   conda activate accrl_env
```
3. Install Dependencies
 ```bash
   cd minimal_modelfree_GIPO
   pip install -r requirements.txt
```
4. Run the Standalone Test
   To verify the installation, run the standalone script which uses a fake environment for testing
    ```bash
    python main_ray_gipo_ds_standalone.py
    ```

### 1. Run Model-Free Asynchronous RL
This script executes a standard distributed GIPO pipeline.
```bash
python main_ray_gipo.py \
    --num-rollout-workers 1 \
    --num-eval-workers 1 \
    --train-iters 5 \
    --train-batch-size 16 \
    --accumulation-steps 2
```

### 2. Run Model-Based RL (World Model)
This script initializes the full World Model Actor matrix to demonstrate "learning in imagination."
```bash
python main_mbrl_gipo.py \
    --num-rollout-workers 1 \
    --imagine-horizon 4 \
    --num-step-cond 4 \
    --train-iters 5
```
*(Note: By default, the minimal version runs on CPU. Use flags like `--trainer-num-gpus 1` to assign GPU resources.)*

##  Extension Guide

To train large-scale foundation models in a real cluster, keep the interface semantics consistent and replace the "Fake" components:

1.  **Replace Environment (`FakeEnv` → Real Simulator)**: Integrate your target simulator, ensuring standard RL `reset()` and `step()` formats.
2.  **Replace Dynamics (`FakeDenoiser` → Real World Model)**: Plug in a high-fidelity Diffusion model into the `DenoiserInferenceActor` to predict `next_obs`.
3.  **Replace Reward Model (`FakeRewardModel` → Real RM)**: Plug in a binary classifier or reward model. Ensure it outputs `[B, 2]` logits to provide `succ_prob` and `end` signals.
4.  **Replace Policy (`FakeActorCritic` → OpenVLA)**: Implement the `forward` and `post_process` interfaces for a real VLA model to output `action_tokens` and continuous `action_env` commands.

##  Citation

If you use AcceRL or its codebase in your research, please cite our paper:

```bibtex
@misc{lu2026accerldistributedasynchronousreinforcement,
      title={AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for Vision-Language-Action Models}, 
      author={Chengxuan Lu and Shukuan Wang and Yanjie Li and Wei Liu and Shiji Jin and Fuyuan Qian and Peiming Li and Baigui Sun and Yang Liu},
      year={2026},
      eprint={2603.18464},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.18464}, 
}
```

## 📄 License
This project is licensed under the [MIT License](LICENSE).
