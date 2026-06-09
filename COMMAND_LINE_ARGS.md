# OpenVLA RL训练 - 命令行参数说明

## 概述
`ds_wm_discrete_diffusion.py` 现在支持通过命令行参数配置所有超参数，无需修改代码。

## 快速开始

### 1. 查看所有可用参数
```bash
python rl/ds_wm_discrete_diffusion.py --help
```

### 2. 使用示例脚本
```bash
bash run_training_example.sh
```

### 3. 自定义训练
```bash
python rl/ds_wm_discrete_diffusion.py \
  --cuda-visible-devices "0,1" \
  --benchmark LIBERO_SPATIAL \
  --train-iters 50000 \
  --clip-mode sapo
```

## 主要参数类别

### 环境配置
- `--cuda-visible-devices`: 可见的GPU设备 (默认: "1,2")
- `--benchmark`: Libero任务集 (默认: LIBERO_SPATIAL)
  - 可选: LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_10, LIBERO_90

### 分布式系统参数
- `--num-trainer-gpus`: 训练器GPU数量 (默认: 1)
- `--num-inference-actors`: 推理Actor数量 (默认: 1)
- `--num-rollout-workers`: Rollout Worker数量 (默认: 20)
- `--num-eval-workers`: 评估Worker数量 (默认: 10)
- `--inference-batch`: 推理批次大小 (默认: 8)
- `--inference-timeout-ms`: 推理超时(毫秒) (默认: 300)

### 训练参数
- `--train-batch-size`: 训练批次大小 (默认: 32)
- `--accumulation-steps`: 梯度累积步数 (默认: 8)
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
  - 可选: ppo, sapo, clippo

### 学习率配置
- `--value-lr`: 价值网络学习率 (默认: 1e-4)
- `--policy-lr`: 策略网络学习率 (默认: 1e-5)
- `--value-warmup-steps`: 价值网络预热步数 (默认: 500)
- `--policy-warmup-steps`: 策略网络预热步数 (默认: 500)
- `--policy-train-start-step`: 策略网络开始训练步数 (默认: 0)

### 世界模型参数
- `--imagine-horizon`: 想象步数 (默认: 8)
- `--num-step-cond`: 条件观测步数 (默认: 4)
- `--num-reward-inference-actors`: 奖励推理Actor数量 (默认: 1)
- `--num-denoiser-inference-actors`: 去噪器推理Actor数量 (默认: 1)

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

### 实验配置
- `--exp-name`: 实验名称 (默认: 自动生成)

## 使用技巧

### 1. 快速测试（小规模）
```bash
python rl/ds_wm_discrete_diffusion.py \
  --num-rollout-workers 5 \
  --num-eval-workers 2 \
  --train-iters 1000 \
  --replay-capacity 1000
```

### 2. 大规模训练
```bash
python rl/ds_wm_discrete_diffusion.py \
  --cuda-visible-devices "0,1,2,3" \
  --num-trainer-gpus 2 \
  --num-inference-actors 4 \
  --num-rollout-workers 40 \
  --train-iters 100000
```

### 3. 使用不同的PPO变体
```bash
# 标准PPO
python rl/ds_wm_discrete_diffusion.py --clip-mode ppo

# SAPO (Soft Adaptive PPO)
python rl/ds_wm_discrete_diffusion.py --clip-mode sapo

# CLIPPO
python rl/ds_wm_discrete_diffusion.py --clip-mode clippo
```

### 4. 调整学习率
```bash
python rl/ds_wm_discrete_diffusion.py \
  --value-lr 5e-5 \
  --policy-lr 5e-6 \
  --value-warmup-steps 1000 \
  --policy-warmup-steps 1000
```

### 5. 使用本体感受状态
```bash
python rl/ds_wm_discrete_diffusion.py \
  --use-proprio \
  --pretrained-checkpoint "path/to/proprio_checkpoint"
```

### 6. 使用多图像输入
```bash
python rl/ds_wm_discrete_diffusion.py \
  --num-images-in-input 2
```

## 注意事项

1. **GPU设置**: `--cuda-visible-devices` 必须在其他GPU相关参数之前设置
2. **批次大小**: `train-batch-size * accumulation-steps` 应小于 `replay-capacity`
3. **Worker数量**: 根据可用CPU核心数调整 `num-rollout-workers` 和 `num-eval-workers`
4. **内存管理**: 大的 `replay-capacity` 需要更多RAM

## 示例配置

### 配置1: 快速原型测试
```bash
python rl/ds_wm_discrete_diffusion.py \
  --cuda-visible-devices "0" \
  --num-rollout-workers 5 \
  --train-iters 5000 \
  --clip-mode ppo
```

### 配置2: 标准训练
```bash
python rl/ds_wm_discrete_diffusion.py \
  --cuda-visible-devices "0,1" \
  --num-rollout-workers 20 \
  --train-iters 30000 \
  --clip-mode sapo
```

### 配置3: 大规模训练
```bash
python rl/ds_wm_discrete_diffusion.py \
  --cuda-visible-devices "0,1,2,3" \
  --num-trainer-gpus 2 \
  --num-inference-actors 4 \
  --num-rollout-workers 40 \
  --num-eval-workers 20 \
  --train-iters 100000 \
  --clip-mode sapo
```


