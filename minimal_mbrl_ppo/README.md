# Minimal MBRL GIPO

这是一个从 `rl/ds_wm_discrete_diffusion.py` 抽取出的最小化、可独立运行的“世界模型推演 + 分布式 GIPO”示例。它保留了核心 Ray actor 架构、imagination rollout 逻辑、以及 DeepSpeed/ZeRO 训练路径，但移除了 OpenVLA、Libero、真实奖励模型、真实扩散采样器和复杂图像预处理依赖。

目录里的 4 个文件：

- `fake_env.py`
- `fake_models.py`
- `main_mbrl_gipo.py`
- `README.md`

## 核心架构

这个最小版仍然保留了原脚本最关键的 actor 角色：

- `StatsActor`
  - 记录 rollout/eval 的回报、成功率、imagined reward 和时间统计。

- `ReplayBufferActor`
  - 存储 imagined rollout 之后得到的 GIPO 训练样本。

- `InferenceActor`
  - 运行 `FakeActorCritic`
  - 给策略提供：
    - `action_tokens`
    - 连续动作 `action_env`
    - 行为策略 `logits`
    - `value`

- `RewardInferenceActor`
  - 运行 `FakeRewardModel`
  - 给 imagined next observation 产生二分类 logits
  - `softmax(logits)[1]` 被当作 `succ_prob`
  - `argmax(logits)` 被当作 imagined rollout 的 `end` 信号

- `DenoiserInferenceActor`
  - 运行 `FakeDenoiser`
  - 接受历史 observation 序列和动作序列，直接预测一个虚拟的下一个 observation
  - 它在这里充当 transition dynamics

- `RolloutWorkerActor`
  - 先通过真实的 `FakeEnv` 收集一条真实 episode，作为 imagined rollout 的初始上下文
  - 然后进入 `for j in range(self.imagine_horizon)` 的推演循环
  - 每一步都调用：
    - `InferenceActor` 选动作
    - `DenoiserInferenceActor` 预测下一个状态
    - `RewardInferenceActor` 预测奖励和是否结束
  - 最后在 worker 内部直接计算 GAE，并把 GIPO 样本写入 `ReplayBufferActor`

- `EvaluationWorkerActor`
  - 直接在真实 `FakeEnv` 上跑确定性策略评估

- `TrainerActor`
  - 从 `ReplayBufferActor` 取 imagined GIPO 样本
  - 默认通过 `deepspeed.initialize` 执行训练，可切换 ZeRO stage
  - 保留 GIPO 的核心数学：
    - advantage 标准化
    - clipped objective
    - value loss
    - entropy bonus
    - KL penalty

## RolloutWorker 的工作方式

这是整个世界模型版的核心。

### 1. 先收集一条真实 episode

`RolloutWorkerActor.get_one_episode()` 会：

1. 在 `FakeEnv` 中重置环境
2. 用 `InferenceActor` 给策略采样动作
3. 在真实环境执行这些动作
4. 存下：
   - 真实 observation 序列
   - reward 序列
   - done 序列
   - 连续动作序列

这些真实 episode 不直接用于训练，而是作为 imagined rollout 的初始条件。

### 2. 再做 imagination rollout

`RolloutWorkerActor.run()` 中保留了原脚本最关键的结构：

```python
for j in range(self.imagine_horizon):
    # policy actor 选动作
    # denoiser actor 预测 next obs
    # reward actor 预测 succ_prob 和 end
```

推演流程如下：

1. 从真实 episode 里截出一个长度为 `num_step_cond` 的 observation 上下文窗口
2. 用当前 imagined state 调用 `InferenceActor` 获取动作
3. 把动作喂给 `DenoiserInferenceActor`，得到 imagined `next_obs`
4. 把 `next_obs` 喂给 `RewardInferenceActor`
5. 用 `succ_prob - last_succ_prob` 构造 imagined reward
6. 如果 reward actor 给出 `end=1`，则提前结束 imagined rollout
7. rollout 结束后，worker 直接计算 GAE 并写入 `ReplayBufferActor`

也就是说，训练数据不是来自真实环境长时间交互，而是来自：

- 少量真实 episode 提供起点
- 世界模型在虚拟空间中继续往前滚动

## 本地运行

安装最小依赖：

```bash
pip install ray torch numpy tensorboard
```

进入目录运行：

```bash
cd /cpfs01/qianfy_workspace/openvla_oft_rl/minimal_mbrl_ppo
python main_mbrl_gipo.py
```

更轻量的快速测试：

```bash
python main_mbrl_gipo.py \
  --train-iters 5 \
  --num-rollout-workers 1 \
  --num-eval-workers 1 \
  --train-batch-size 8 \
  --accumulation-steps 2 \
  --imagine-horizon 4
```

### 纯 CPU 运行

默认就是纯 CPU：

```bash
python main_mbrl_gipo.py
```

### 单 GPU 运行

如果你想让 trainer 和推理 actor 使用单卡：

```bash
python main_mbrl_gipo.py \
  --trainer-num-gpus 1 \
  --inference-num-gpus 0 \
  --reward-num-gpus 0 \
  --denoiser-num-gpus 0
```

如果你后续想细分策略/奖励/转移动力学 actor 到 GPU，也可以分别调：

- `--inference-num-gpus`
- `--reward-num-gpus`
- `--denoiser-num-gpus`

训练结束后会保存：

- `minimal_mbrl_gipo_ckpt.pt`

## 如何替换成真实模型

### 替换 `FakeDenoiser`

你最终要把 `fake_models.py` 里的 `FakeDenoiser` 替换成真实世界模型。

需要保持接口语义不变：

- 输入：
  - `obs_batch`: 历史 observation 序列
  - `act_batch`: 历史动作序列
- 输出：
  - `next_obs`

代码接入点在：

- `main_mbrl_gipo.py` 里的 `DenoiserInferenceActor.request()`
- `RolloutWorkerActor.predict_next_obs()`

### 替换 `FakeRewardModel`

真实奖励模型需要继续输出 `[B, 2]` 的 logits，保持：

- `softmax(logits)[1]` 代表成功概率 `succ_prob`
- `argmax(logits)` 代表是否终止 `end`

代码接入点在：

- `RewardInferenceActor.request()`
- `RolloutWorkerActor.predict_rew_end()`

### 替换 `FakeActorCritic`

真实策略模型只要继续提供这几个接口即可：

- `prepare_inputs_batch(obs_list)`
- `forward(obs_batch) -> (action_logits, values)`
- `post_process(action_logits) -> (action_tokens, action_env)`

这样 `TrainerActor`、`RolloutWorkerActor` 和 `EvaluationWorkerActor` 基本都不用大改。

## DeepSpeed 与 TensorBoard

当前版本默认开启 DeepSpeed，并使用 `deepspeed.initialize`（保留 `ds_config` 与 ZeRO 逻辑）：

```bash
python main_mbrl_gipo.py \
  --train-iters 5 \
  --train-batch-size 8 \
  --accumulation-steps 2 \
  --imagine-horizon 4
```

可通过 `--zero-stage {0,1,2,3}` 切换 ZeRO stage，或通过 `--ds-config-json /path/to/ds_config.json` 加载外部配置。若仅做纯 PyTorch 对照实验，可加 `--disable-deepspeed`。

## ds_com Standalone 版本（世界模型）

新增文件：`main_mbrl_gipo_ds_standalone.py`。该版本保留 `TrainerActorCom` / `InferenceActorCom` 通讯骨架与 DeepSpeed 初始化，并保留 `imagine_horizon` 推演循环，仅替换为 fake env / fake actor-critic / fake denoiser / fake reward。

运行示例：

```bash
cd /cpfs01/qianfy_workspace/openvla_oft_rl/minimal_mbrl_ppo
python main_mbrl_gipo_ds_standalone.py --train-iters 5 --num-rollout-workers 1 --imagine-horizon 4
```

TensorBoard 日志目录默认是 `runs/minimal_mbrl_gipo`，查看命令：

```bash
tensorboard --logdir runs/minimal_mbrl_gipo --port 6006
```

## 这个最小版保留了什么

- Ray actor 拓扑
- 真实 episode 作为 imagined rollout 起点
- reward actor + denoiser actor 共同驱动 imagination rollout
- worker 端 GAE 计算
- GIPO clip / value loss / entropy / KL
- trainer / inference 解耦

## 这个最小版去掉了什么

- OpenVLA
- Libero
- 真实视觉编码器
- 真实 reward model
- 真实 diffusion sampler
- 复杂图像预处理和反归一化逻辑

## 建议的后续扩展顺序

1. 先把 `FakeDenoiser` 换成真实 transition model
2. 再把 `FakeRewardModel` 换成真实奖励/终止预测模型
3. 最后把 `FakeActorCritic` 换成真实策略网络

这样最容易逐步验证 imagined rollout 是否工作正常。
