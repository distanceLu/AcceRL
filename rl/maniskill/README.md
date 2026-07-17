# ManiSkill Training and Evaluation Flow

这个目录放的是 OpenVLA-OFT 在 ManiSkill 任务上的数据准备、SFT、PPO/RL 训练、环境封装和评估调试脚本。整体流程可以理解为：

```text
ManiSkill demos
  -> replay 成 224x224 双相机 RGBD H5
  -> 转成 TFDS/RLDS 数据集
  -> 可选预处理成 .pt shards
  -> OpenVLA SFT
  -> ManiSkill PPO/RL
  -> rollout/eval/debug
```

## 目录角色

核心训练入口：

- `finetune_maniskill.py`: ManiSkill SFT 主入口，支持直接读 RLDS，也支持读预处理后的 `.pt` shards。
- `ds_maniskill_ppo_discrete.py`: ManiSkill PPO/RL 主入口，使用 Ray actors 管理 rollout、inference、replay buffer 和 trainer。
- `maniskill_sft.sh`: SFT 启动脚本示例。
- `maniskill_rl.sh`: PPO/RL 启动脚本示例。

环境和模型封装：

- `maniskill_utils.py`: ManiSkill 环境构建、观测抽取、动作裁剪、success/done 提取等公共函数。
- `maniskill_env.py`: 单环境封装，提供类似 LIBERO wrapper 的 `reset/step/close` 接口，供 PPO worker 使用。
- `new_actor_critic.py`: OpenVLA actor-critic 封装，负责动作预测、value head、LoRA/ckpt 加载等。
- `preprocessed_dataset.py`: 读取 `.pt` shard 的 PyTorch MapDataset，用于加速 SFT 数据加载。

数据准备和检查：

- `replay_224_two_cam.py`: 早期 PickCube 双相机 replay 脚本。
- `replay_224_two_cam_7tasks.py`: 多任务 replay 脚本，默认处理额外 7 个 ManiSkill 任务。
- `maniskill_*_dataset_builder.py`: TFDS/RLDS dataset builder，将 replay 后的 H5 转为 OpenVLA 可读的数据集。
- `preprocess_rlds_to_pt.py`: 把 RLDS 样本提前解码、resize、tokenize，并保存为 `.pt` shards。
- `check_replay_dataset.py`: 检查 replay 后 H5 的相机、action、success 等字段。
- `check_rlds_dataset.py`: 检查 TFDS/RLDS 数据集结构和样本质量。

评估和调试：

- `smoke_maniskill_eval.py`: 最小环境冒烟测试，验证双相机观测和随机动作 step。
- `maniskill_actor_critic_eval.py`: 使用训练好的 actor-critic 在 ManiSkill 中 rollout 评估。
- `maniskill_actor_critic_debug.py`: PegInsertionSide 专用 debug 脚本，对比不同 action chunk 执行方式。
- `debug_rollout_render_gpu.py`: 调试 rollout/render GPU 绑定。
- `rollout_test.py`: 随机 rollout worker 测试。

## 环境准备

当前脚本默认使用本机环境：

```bash
MANISKILL_ENV=/mnt/data/lcx4/miniforge3/envs/why_maniskill
source /mnt/data/lcx4/miniforge3/etc/profile.d/conda.sh
conda activate why_maniskill
cd /mnt/data/lcx4/openvla_oft_rl
```

常用环境变量：

```bash
export CUDA_HOME="$MANISKILL_ENV"
export PATH="$MANISKILL_ENV/bin:$MANISKILL_ENV/targets/x86_64-linux/bin:$PATH"
export LD_LIBRARY_PATH="$MANISKILL_ENV/lib:$MANISKILL_ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/mnt/data/lcx4/.cache/torch_extensions/why_maniskill_py310_cu124
export LIBERO_CONFIG_PATH="$MANISKILL_ENV/libero_config"
export MPLCONFIGDIR=/tmp/matplotlib-why-maniskill
```

如果涉及 GPU 渲染，通常还需要显式绑定 Vulkan/SAPIEN：

```bash
export CUDA_VISIBLE_DEVICES=0
export VULKAN_VISIBLE_DEVICES=0
export SAPIEN_VULKAN_DEVICE=0
export EGL_DEVICE_ID=0
```

## 1. Replay ManiSkill Demonstrations

ManiSkill 原始 demonstrations 通常需要 replay 成 OpenVLA 使用的双相机 `rgbd` 轨迹。这里统一使用：

- `robot_uids=panda_wristcam`
- `base_camera` 和 `hand_camera`
- 分辨率 `224x224`
- 目标控制模式 `pd_ee_delta_pose`

多任务 replay 示例：

```bash
DATA_ROOT=/data/disk1/lcx_stu4/maniskill/demos \
TASKS=PushCube-v1,PullCube-v1,PokeCube-v1 \
SKIP_MISSING=1 \
RENDER_GPU=0 \
python rl/maniskill/replay_224_two_cam_7tasks.py
```

默认输出到：

```text
$DATA_ROOT/<TaskName>/motionplanning_rgbd_224_two_cam/
```

输出 H5 后可以检查：

```bash
python rl/maniskill/check_replay_dataset.py \
  --h5 /path/to/trajectory.rgbd.pd_ee_delta_pose.physx_cpu.h5 \
  --expect-h 224 \
  --expect-w 224 \
  --expect-action-dim 7 \
  --dump-keys
```

## 2. Build RLDS/TFDS Dataset

`maniskill_pickcube_dataset_builder.py`、`maniskill_stackcube_dataset_builder.py`、`maniskill_peginsertionside_dataset_builder.py` 会把 replay 后的 H5 转成 OpenVLA RLDS pipeline 能读取的 TFDS 数据集。

它们输出的核心字段包括：

- `observation/image`: base camera RGB
- `observation/wrist_image`: hand camera RGB
- `observation/state`: TCP xyz + axis-angle + gripper
- `observation/joint_state`: Panda arm joint state
- `action`: 7D `pd_ee_delta_pose`
- `language_instruction`: 任务语言指令

生成 TFDS 后建议先检查：

```bash
python rl/maniskill/check_rlds_dataset.py \
  --data-dir /path/to/tfds/root \
  --dataset-name maniskill_pickcube \
  --split train \
  --num-episodes 20
```

## 3. Optional: Preprocess RLDS to PT Shards

直接读取 RLDS/TFDS 可能比较慢。`preprocess_rlds_to_pt.py` 会把样本提前做确定性处理并保存为 `.pt` shards，训练时再按需做 image augmentation。

示例：

```bash
python rl/maniskill/preprocess_rlds_to_pt.py \
  --vla_path /mnt/data/lcx4/hf_cache/openvla-7b \
  --data_root_dir /mnt/data2/lcx_stu4/maniskill/demos/rlds \
  --dataset_name maniskill_three_tasks \
  --output_dir /mnt/data/lcx4/openvla_oft_rl/rl/maniskill/sft_data \
  --num_images_in_input 2 \
  --use_proprio False \
  --shard_size 500
```

输出目录通常包含：

```text
dataset_statistics.json
metadata.json
shard_00000.pt
shard_00001.pt
...
```

这些是训练产物，不建议提交到 Git。

## 4. SFT Training

SFT 入口是 `finetune_maniskill.py`。当前推荐用预处理后的 `.pt` shards，以减少 TFDS 数据加载瓶颈。

直接运行示例脚本：

```bash
bash rl/maniskill/maniskill_sft.sh
```

对应核心参数：

```bash
python rl/maniskill/finetune_maniskill.py \
  --vla_path /mnt/data/lcx4/hf_cache/openvla-7b \
  --dataset_name maniskill_three_tasks \
  --preprocessed_data_dir /mnt/data/lcx4/openvla_oft_rl/rl/maniskill/sft_data \
  --num_images_in_input 2 \
  --use_preprocessed_data True \
  --use_proprio False \
  --image_aug True \
  --batch_size 16 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --max_steps 100000 \
  --save_freq 200 \
  --run_id_note three_tasks_2cam_preprocessed \
  --use_maniskill_env_eval False
```

SFT 输出一般作为后续 PPO/RL 的 `--pretrained-checkpoint`。

## 5. PPO/RL Training

RL 入口是 `ds_maniskill_ppo_discrete.py`，示例脚本是：

```bash
bash rl/maniskill/maniskill_rl.sh
```

核心配置包括：

- `--maniskill-tasks`: 训练任务列表，例如 `PickCube-v1,StackCube-v1`
- `--camera-name` 和 `--wrist-camera-name`: 双相机输入
- `--num-images-in-input 2`: 使用 base + wrist 两张图
- `--num-rollout-workers`: rollout worker 数量
- `--num-trainer-gpus`: trainer 使用 GPU 数
- `--pretrained-checkpoint`: SFT checkpoint 或模型目录
- `--ckpt-dir`: RL checkpoint 输出目录
- `--clip-mode`: PPO/GIPO 等 clipping 策略

示例核心命令：

```bash
python rl/maniskill/ds_maniskill_ppo_discrete.py \
  --cuda-visible-devices "0,1,2,3" \
  --maniskill-tasks PickCube-v1,StackCube-v1 \
  --camera-name base_camera \
  --wrist-camera-name hand_camera \
  --robot-uids panda_wristcam \
  --camera-res 224 \
  --num-images-in-input 2 \
  --num-trainer-gpus 3 \
  --num-inference-actors 1 \
  --num-rollout-workers 30 \
  --num-eval-workers 2 \
  --pretrained-checkpoint /mnt/data/lcx4/openvla_oft_rl/rl/maniskill/sft_model \
  --ckpt-dir /mnt/data/lcx4/openvla_oft_rl/runs/rl_maniskill \
  --exp-name ManiSkill_PickCube_StackCube
```
