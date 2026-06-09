# OpenVLA PPO RL训练 - 命令行参数说明 (无世界模型版本)

## 概述
`ds_libero_ppo_discrete.py` 现在支持通过命令行参数配置所有超参数，无需修改代码。这是不使用世界模型的标准PPO版本。

## 快速开始

### 1. 查看所有可用参数
```bash
python rl/ds_libero_ppo_discrete.py --help
```

### 2. 使用示例脚本
```bash
bash run_ppo_training_example.sh
```

### 3. 自定义训练
```bash
python rl/ds_libero_ppo_discrete.py \
  --cuda-visible-devices "0,1" \
  --benchmark libero_spatial \
  --train-iters 50000 \
  --clip-mode sapo
```

## 主要参数类别

### 环境配置
- `--cuda-visible-devices`: 可见的GPU设备 (默认: "6,7")
- `--benchmark`: Libero任务集 (默认: libero_spatial)
  - 可选: libero_spatial, libero_object, libero_goal, libero_10, libero_90

### 分布式系统参数
- `--num-trainer-gpus`: 训练器GPU数量 (默认: 1)
- `--num-inference-actors`: 推理Actor数量 (默认: 1)
- `--num-rollout-workers`: Rollout Worker数量 (默认: 2)
- `--num-eval-workers`: 评估Worker数量 (默认: 20)
- `--rollout-local-buf`: Rollout本地缓冲区大小 (默认: 64)
- `--inference-batch`: 推理批次大小 (默认: 8)
- `--inference-timeout-ms`: 推理超时(毫秒) (默认: 300)

### Ray配置
- `--object-store-memory-gb`: Ray对象存储内存(GB) (默认: 256)

### 训练参数
- `--train-batch-size`: 训练批次大小 (默认: 12)
- `--accumulation-steps`: 梯度累积步数 (默认: 21)
- `--train-iters`: 训练迭代次数 (默认: 30000)
- `--replay-capacity`: 经验回放容量 (默认: 10000)

### PPO超参数
- `--gamma`: 折扣因子 (默认: 0.99)
- `--lambda`: GAE lambda (默认: 0.95)
- `--clip-eps`: PPO裁剪epsilon (默认: 0.2)
- `--vf-coef`: 价值函数系数 (默认: 0.5)
- `--ent-coef`: 熵系数 (默认: 0.00)
- `--kl-coef`: KL散度系数 (默认: 0.1)
- `--clip-mode`: PPO裁剪模式 (默认: sapo)
  - 可选: ppo, sapo, gipo
- `--reward-scale`: 奖励缩放因子 (默认: 1.0)

### 学习率配置
- `--value-lr`: 价值网络学习率 (默认: 1e-4)
- `--policy-lr`: 策略网络学习率 (默认: 1e-5)
- `--value-warmup-steps`: 价值网络预热步数 (默认: 500)
- `--policy-warmup-steps`: 策略网络预热步数 (默认: 500)
- `--policy-train-start-step`: 策略网络开始训练步数 (默认: 0)

### 检查点与日志
- `--ckpt-dir`: 检查点保存目录 (默认: /cpfs01/liuwei_workspace/models/finetune_rl)
- `--ckpt-every-steps`: 保存检查点频率 (默认: 2000000)
- `--log-interval-seconds`: 日志记录间隔(秒) (默认: 10)
- `--moving-avg-window`: 移动平均窗口大小 (默认: 1000)

### 模型配置
- `--pretrained-checkpoint`: 预训练模型路径
- `--checkpoint2`: 第二个检查点路径
- `--use-bf16` / `--no-bf16`: 是否使用bfloat16 (默认: 启用)
- `--use-proprio`: 是否使用本体感受状态 (默认: 关闭)
- `--num-images-in-input`: 输入图像数量 (默认: 1)
- `--broadcast-group-name`: 广播组名称 (默认: trainer_to_inference_broadcast)

### 实验配置
- `--exp-name`: 实验名称 (默认: 自动生成)

## 与世界模型版本的区别

此版本 (`ds_libero_ppo_discrete.py`) 与世界模型版本 (`ds_wm_discrete_diffusion.py`) 的主要区别：

1. **无世界模型参数**: 不包含以下参数
   - `--imagine-horizon`
   - `--num-step-cond`
   - `--num-reward-inference-actors`
   - `--num-denoiser-inference-actors`
   - `--agent-config-path`
   - `--trainer-config-path`

2. **Worker数量不同**: 
   - 默认 rollout workers: 2 (vs 世界模型版本的 20)
   - 默认 eval workers: 20 (vs 世界模型版本的 10)

3. **Ray对象存储**: 添加了 `--object-store-memory-gb` 参数

## 使用技巧

### 1. 快速测试（小规模）
```bash
python rl/ds_libero_ppo_discrete.py \
  --num-rollout-workers 1 \
  --num-eval-workers 5 \
  --train-iters 1000 \
  --replay-capacity 1000
```

### 2. 大规模训练
```bash
python rl/ds_libero_ppo_discrete.py \
  --cuda-visible-devices "0,1,2,3" \
  --num-trainer-gpus 2 \
  --num-inference-actors 4 \
  --num-rollout-workers 10 \
  --num-eval-workers 40 \
  --train-iters 100000
```

### 3. 使用不同的PPO变体
```bash
# 标准PPO
python rl/ds_libero_ppo_discrete.py --clip-mode ppo

# SAPO (Soft Adaptive PPO)
python rl/ds_libero_ppo_discrete.py --clip-mode sapo

# GIPO
python rl/ds_libero_ppo_discrete.py --clip-mode gipo
```

### 4. 调整学习率和批次大小
```bash
python rl/ds_libero_ppo_discrete.py \
  --value-lr 5e-5 \
  --policy-lr 5e-6 \
  --train-batch-size 16 \
  --accumulation-steps 16
```

### 5. 使用本体感受状态
```bash
python rl/ds_libero_ppo_discrete.py \
  --use-proprio \
  --pretrained-checkpoint "path/to/proprio_checkpoint"
```

### 6. 使用多图像输入
```bash
python rl/ds_libero_ppo_discrete.py \
  --num-images-in-input 3
```

## 注意事项

1. **GPU设置**: `--cuda-visible-devices` 必须在其他GPU相关参数之前设置
2. **批次大小**: `train-batch-size * accumulation-steps` 应小于 `replay-capacity`
3. **Worker数量**: 根据可用CPU核心数调整 `num-rollout-workers` 和 `num-eval-workers`
4. **内存管理**: 
   - 大的 `replay-capacity` 需要更多RAM
   - `object-store-memory-gb` 应根据系统内存调整
5. **Rollout buffer**: `rollout-local-buf` 控制每个worker在发送数据到replay buffer前的本地缓存大小

## 示例配置

### 配置1: 快速原型测试
```bash
python rl/ds_libero_ppo_discrete.py \
  --cuda-visible-devices "0" \
  --num-rollout-workers 1 \
  --num-eval-workers 5 \
  --train-iters 5000 \
  --clip-mode ppo
```

### 配置2: 标准训练
```bash
python rl/ds_libero_ppo_discrete.py \
  --cuda-visible-devices "0,1" \
  --num-rollout-workers 2 \
  --num-eval-workers 20 \
  --train-iters 30000 \
  --clip-mode sapo
```

### 配置3: 大规模训练
```bash
python rl/ds_libero_ppo_discrete.py \
  --cuda-visible-devices "0,1,2,3" \
  --num-trainer-gpus 2 \
  --num-inference-actors 4 \
  --num-rollout-workers 10 \
  --num-eval-workers 40 \
  --train-iters 100000 \
  --object-store-memory-gb 512 \
  --clip-mode sapo
```

## 对比两个版本

| 特性 | PPO版本 | 世界模型版本 |
|------|---------|-------------|
| 文件名 | `ds_libero_ppo_discrete.py` | `ds_wm_discrete_diffusion.py` |
| 世界模型 | ❌ | ✅ |
| Rollout Workers (默认) | 2 | 20 |
| Eval Workers (默认) | 20 | 10 |
| 想象轨迹 | ❌ | ✅ |
| Ray对象存储配置 | ✅ | ❌ |
| 适用场景 | 标准RL训练 | 模型驱动RL |

