# AcceRL + Ctrl-World

本目录包含 AcceRL 接入 Ctrl-World 后的在线训练和闭环预测评估代码：

- `ds_wm_discrete_ctrl_train_wm.py`：同时运行 OpenVLA 强化学习、Reward Model 训练和 Ctrl-World 在线训练，并把最新权重同步给对应的 Ray 推理 Actor。
- `ctrl_world_env_batch.py`：为训练脚本提供批量 Ctrl-World latent 初始化和分块预测。
- `test_ctrl_world_wm_closed_loop_predict.py`：先在 LIBERO 中收集真实轨迹，再从指定帧开始用 Ctrl-World 自回归想象，并输出视频、动作和误差指标。
- 仓库根目录的 `run_oft_ctrl_world_train_wm.sh`：当前训练配置的完整启动示例。


### 快速启动

```bash
bash run_oft_ctrl_world_train_wm.sh
```

| 参数 | 用途 |
| --- | --- |
| `--pretrained-checkpoint` | OpenVLA 基础 checkpoint |
| `--checkpoint2` | 可选的第二阶段/蒸馏 checkpoint |
| `--svd-model-path` | Ctrl-World 使用的 SVD 模型目录 |
| `--clip-model-path` | Ctrl-World 使用的 CLIP 模型目录 |
| `--ctrl-world-ckpt` | 使用 AcceRL delta action 训练的 Ctrl-World checkpoint |
| `--vae-decoder-checkpoint` | 可选的微调 VAE decoder；文件必须包含 `decoder` 键 |
| `--condition-stat-path` | action 归一化统计，需包含 `condition_p01` 和 `condition_p99` |
| `--reward-checkpoint` | Reward Model checkpoint；不传时读取 `envs/config/agent.yaml` |
| `--agent-config-path` | Reward Model/OpenVLA 配置，默认 `envs/config/agent.yaml` |
| `--trainer-config-path` | 训练器配置，默认 `envs/config/trainer.yaml` |



## 闭环预测评估

`test_ctrl_world_wm_closed_loop_predict.py` 固定使用 `libero_spatial`、双相机、`192 x 320`、6 帧历史、5 帧预测 chunk、10 步 diffusion 和 bf16。

脚本中的以下模型路径目前是源码常量，运行前需在文件顶部确认：

- `VLA_CHECKPOINT`
- `VLA_CHECKPOINT2`
- `SVD_MODEL`
- `CLIP_MODEL`


### 普通轨迹想象模式

该模式先用 VLA 在真实 LIBERO 环境中收集完整对照轨迹，在 `start-index` 之前用真实图像/action 作为历史，之后将每次 Ctrl-World 预测图像送回 VLA，使用新动作继续自回归预测：

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/data/lcx3/envs/merged-env/bin/python \
  rl/ctrl_world/test_ctrl_world_wm_closed_loop_predict.py \
  --task-id 1 \
  --initial-state-id 0 \
  --start-index 70 \
  --num-chunks 16 \
  --checkpoint /path/to/best_val_loss.pt \
  --output-dir runs/ctrl_world_closed_loop \
  --device cuda \
  --seed 0
```

### Action-matched 模式

该模式从 Ctrl-World 图像查询 VLA，然后把完全相同的 raw VLA action 同时用于 Ctrl-World 条件和真实 LIBERO 环境，更适合比较相同动作下的图像预测偏差：

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/data/lcx3/envs/merged-env/bin/python \
  rl/ctrl_world/test_ctrl_world_wm_closed_loop_predict.py \
  --action-matched \
  --task-id 1 \
  --start-index 70 \
  --num-chunks 16 \
  --checkpoint /path/to/best_val_loss.pt \
  --output-dir runs/ctrl_world_action_matched
```

`--start-index` 不能小于 6。每个 chunk 含 5 帧，相邻 chunk 共享 anchor，因此预测帧数为：

