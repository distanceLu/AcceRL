# Minimal Ray PPO

这是从 `ds_libero_gipo_discrete.py` 中抽出的一个最小化、可独立运行的分布式 GIPO 框架示例。它保留了核心 Ray actor 拓扑和 PPO 数学逻辑，但把沉重的 OpenVLA / DeepSpeed / Libero 依赖替换成了：

- `fake_env.py`: 纯 Numpy 的 `FakeEnv`
- `fake_model.py`: 纯 PyTorch 的 `FakeActorCritic`
- `main_ray_gipo.py`: 精简后的 Ray + GIPO 主框架

## 架构概览

系统仍然保留了原始脚本中的核心 actor 角色：

- `StatsActor`
  - 汇总 rollout / eval 回报、成功率、平均 episode 长度、推理延迟等统计。

- `ReplayBufferActor`
  - 按轨迹存储采样数据。
  - `RolloutWorkerActor` 把轨迹写进去，`TrainerActor` 从里面采样。

- `RolloutWorkerActor`
  - 持有多个 `FakeEnv`。
  - 从 `InferenceActor` 请求动作，执行环境步进，拼成轨迹后写入 `ReplayBufferActor`。

- `EvaluationWorkerActor`
  - 也使用 `FakeEnv`，但始终用确定性策略（argmax）做评估。
  - 只写统计，不写 replay buffer。

- `InferenceActor`
  - 持有 `FakeActorCritic`。
  - 接收 observation，输出：
    - 离散动作 token
    - 连续动作 `action_env`
    - 行为策略 logits
    - value 估计

- `TrainerActor`
  - 从 `ReplayBufferActor` 采样轨迹。
  - 计算 GAE / advantage / value target。
  - 执行 PPO 的 clipped objective、value loss、entropy loss、KL loss。
  - 把最新权重同步给 `InferenceActor`。

数据流如下：

1. `RolloutWorkerActor` 从 `FakeEnv` 取到 observation
2. observation 发给 `InferenceActor`
3. `InferenceActor` 返回动作、logits、value
4. `RolloutWorkerActor` 在环境里执行动作并累计成 `Trajectory`
5. `Trajectory` 写入 `ReplayBufferActor`
6. `TrainerActor` 从 `ReplayBufferActor` 采样轨迹并做 PPO 更新
7. 主线程把 `TrainerActor` 的最新权重同步到所有 `InferenceActor`
8. `EvaluationWorkerActor` 持续跑评估并把结果写到 `StatsActor`

## 本地运行

建议先安装最小依赖：

```bash
pip install ray torch numpy
```

进入目录后直接运行：

```bash
cd /cpfs01/qianfy_workspace/openvla_oft_rl/minimal_ray_ppo
python main_ray_gipo.py
```

更轻量的快速测试：

```bash
python main_ray_gipo.py \
  --train-iters 5 \
  --num-rollout-workers 1 \
  --num-eval-workers 1 \
  --train-batch-size 16 \
  --accumulation-steps 2
```

如果你只有 CPU，默认就能跑，因为：

- `--trainer-num-gpus` 默认是 `0`
- `--inference-num-gpus` 默认是 `0`

如果你希望让 trainer 或 inference actor 使用 GPU，可以设置：

```bash
python main_ray_gipo.py \
  --trainer-num-gpus 1 \
  --inference-num-gpus 1 \
  --num-inference-actors 1
```

训练结束后会在当前目录保存一个权重文件：

- `minimal_ray_gipo_ckpt.pt`

## 关键保留点

这个版本保留了原脚本中最值得研究的 PPO 核心：

- 按轨迹存储经验
- 截断轨迹时使用 bootstrap value
- `_compute_gae`
- advantage 标准化
- PPO ratio / clipped surrogate objective
- value loss
- entropy regularization
- KL penalty
- trainer 与 inference 分离

## 如果接入真实环境

替换位置很明确：

1. 把 `fake_env.py` 中的 `FakeEnv` 换成真实环境包装器
2. 保持接口不变：
   - `reset(seed)` -> `(obs, info)`
   - `step(action)` -> `(next_obs, reward, terminated, truncated, info)`
3. 保证 observation 能被模型消费
4. 如果动作不再是 `(NUM_ACTIONS_CHUNK, ACTION_DIM)`，同步修改：
   - `fake_model.py` 里的输出形状
   - `RolloutWorkerActor` 里的 action 执行逻辑

## 如果接入真实神经网络

同样只需要替换 `fake_model.py` 里的 `FakeActorCritic`：

1. `forward(obs_batch)` 需要继续返回：
   - `action_logits`
   - `values`
2. `post_process(...)` 需要继续输出：
   - 离散 `action_tokens`
   - 真正送入环境的连续动作 `action_env`
3. `prepare_inputs_batch(...)` 负责把 observation 列表拼成 batch

也就是说，只要新模型继续满足这三个接口，`main_ray_gipo.py` 里的分布式 actor 框架和 GIPO 训练逻辑基本都可以不改。

## 限制说明

- 这个最小版默认只支持 `1` 个 `TrainerActor`
- 没有接入真实 DDP / 参数服务器 / DeepSpeed ZeRO
- 目标是帮助你在离线本地机器上研究：
  - actor 拓扑
  - rollout 到 replay 的数据流
  - PPO 的训练数学

如果后续你要把它扩展回真实系统，建议先替换环境，再替换模型，最后再引入多 trainer / DDP。
