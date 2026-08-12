# 真机三视角模仿学习交接文档

本文档对应 AcceRL 仓库中的真机三视角、纯图像输入模仿学习流程。目标是让接手者能够完成数据检查、单卡训练、日志查看、权重加载和后续修改，并理解当前实现的边界。

## 1. 本次交接涉及的文件

- `rl/actor_critic_model_discrete.py`
  - 让 Actor 支持可配置的图像数、动作维度和 action chunk 长度。
  - 保留原始默认配置，避免破坏原先双图像、LIBERO 维度的推理路径。
  - 增加六维连续动作与离散 token 的双向转换，以及范围表随 checkpoint 保存和恢复。
- `rl/train_real_robot_imitation.py`
  - 扫描、对齐三路相机与机器人指令数据。
  - 计算六维 SE(3) 增量动作。
  - 每次实验根据全部有效样本重新生成动作范围表。
  - 执行纯图像模仿学习，并记录 JSONL/TensorBoard 指标。
- `scripts/train_real_robot_imitation_3cam.sh`
  - 真机三视角训练的统一 Bash 入口。
  - 提供数据、基础权重、GPU、batch size、步数和保存频率等默认配置。
- `.gitignore`
  - 已加入 `/data_collect/`，防止真机采集数据进入 Git。

`scripts/class Twomlp.py` 是工作区中已有的未跟踪文件，不属于本三视角流程。提交本次工作时不要因为使用 `git add .` 而误提交它。

## 2. 当前任务定义

### 2.1 网络输入

每个样本只使用以下三张同步后的 RGB 图像，顺序固定：

1. `camera_paper_aruco`
2. `camera_pool`
3. `camera_pool1`

不使用 `camera_3d_2d`，也不向神经网络输入机器人位姿、proprioception 或其他状态量。机器人指令 CSV 只用于离线计算监督动作标签，不是模型输入。

训练代码显式设置：

```text
num_images_in_input = 3
use_proprio = False
action_dim = 6
num_actions_chunk = 8
```

当前视觉骨干会把一张 RGB 图像分别送入 SigLIP 和 DINOv2 分支，因此一张图像在融合输入中对应 6 个通道。三张图像最终得到：

```text
pixel_values: (B, 18, 224, 224)
```

这里 `B` 是 batch size；`18 = 3 个视角 × 6 个融合通道`。它不是 18 张原始图片。

### 2.2 动作输出

每个时间点预测未来 8 个动作，每个动作包含 6 个维度：

```text
dx_m, dy_m, dz_m,
drot_x_rad, drot_y_rad, drot_z_rad
```

因此标签展平后是 `(B, 48)`，因为 `48 = 8 × 6`；分类 logits 是 `(B, 48, 256)`；解码后连续动作是 `(B, 8, 6)`。

平移单位为米，旋转单位为弧度。旋转增量不是简单地把欧拉角逐项相减，而是：

```text
R_delta = R_next @ R_current.T
```

随后把 `R_delta` 转成 rotation vector。这样可以正确处理旋转的组合关系和角度环绕。

## 3. 数据目录和对齐逻辑

训练器扫描：

```text
data_collect/*/*/session_meta.json
```

一个可用 session 至少要有：

```text
camera_paper_aruco/
camera_pool/
camera_pool1/
robot_state/robot_command_state.csv
session_meta.json
```

没有 `robot_command_state.csv` 的 `_infer` 目录不会用于训练。`camera_3d_2d` 即使存在也不会被读取。

数据对齐过程如下：

1. 以 `camera_paper_aruco` 的图片时间戳作为主时间轴。
2. 为每个主时间戳寻找 `camera_pool` 和 `camera_pool1` 的最近图片。
3. 为该时间戳寻找 `robot_command_state.csv` 中最近的 command TCP 位姿。
4. 超过允许时间差的样本会被丢弃。
5. 使用相邻两个有效对齐帧的 command TCP 位姿计算一个六维增量动作。
6. 每个 session 的最后一个有效帧没有下一帧，因此不会产生监督动作。

默认容差为：

```text
相机最大时间差: 150000 us
状态最大时间差: 100000 us
```

对应参数是：

```text
--max-camera-delta-us
--max-state-delta-us
```

未来 8 步 action chunk 不会跨 session。session 尾部不足 8 步时，用该 session 最后一个有效动作补齐。

## 4. 每次实验生成的六维动作范围表

训练启动时，会先收集本次全部有效对齐样本的六维动作，然后为每个维度独立计算：

```text
raw_min, raw_max, mean, std, q01, q99
```

当前实际离散化边界使用全部数据的 `raw_min` 和 `raw_max`，即每一维都有自己的 `[low, high]`。如果某一维的跨度小于 `1e-12`，代码会以该常数为中心扩展到 `±1e-6`，避免除零，并在范围表中记录 `constant_dimensions_expanded`。

连续动作先转换到 `[-1, 1]`：

```text
normalized = 2 * (action - low) / (high - low) - 1
```

随后映射为模型的动作 token。当前输出头有 256 类；实现沿用原工程边界定义，通常实际动作使用 token 1 到 255，token 0 不作为普通区间中心。

每个 run 都会保存：

```text
action_range_table.json
action_range_table.csv
```

同一份范围信息也嵌入 `agent_extra_layers.pt`。推理时必须使用目标 checkpoint 自带的范围表，不能随意拿另一轮实验的 JSON 替换，否则同一个 token 会被还原成不同的物理动作。

先只检查数据和范围表、不开始训练，可以运行：

```bash
bash scripts/train_real_robot_imitation_3cam.sh \
  --prepare-only \
  --run-name data_alignment_check
```

如果同名 run 目录已经存在，请换一个 `--run-name`，或者不传该参数以使用时间戳名称。

重点检查输出中的有效 session 数、样本数，以及范围表里六个维度是否合理。如果某些维度几乎恒定，应先确认采集任务确实没有这些方向的运动，而不是直接开始长时间训练。

## 5. 环境和基础权重

推荐先进入仓库和现有 Python 环境：

```bash
cd /mnt/data/lcx2/yanjieworkspace/AcceRL
source /mnt/data/lcx2/yanjieworkspace/clone_env_smoke_test/rlinf_env/bin/activate
```

Bash 脚本默认会优先使用该环境中的 Python。默认基础权重路径写在 `scripts/train_real_robot_imitation_3cam.sh` 中，也可以在命令前通过 `PRETRAINED_CHECKPOINT=...` 覆盖。

三视角实现没有修改视觉骨干的单图编码器参数形状，而是复用同一套视觉编码器处理第三张图。因此基础权重能够加载，原有 LoRA/模型结构也能继续作为初始化。但第三视角带来了新的输入语义，兼容权重形状不代表无需训练，也不保证初始真机成功率。

## 6. 单卡运行方法

### 6.1 一步 smoke test

以下命令使用物理 GPU 1，但进程内部只看到一张卡，所以设备编号必须是 `cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=1 \
DEVICE=cuda:0 \
BATCH_SIZE=1 \
GRAD_ACCUMULATION_STEPS=1 \
MAX_STEPS=1 \
SAVE_FREQ=1 \
DATALOADER_WORKERS=0 \
USE_TENSORBOARD=0 \
bash scripts/train_real_robot_imitation_3cam.sh --log-freq 1
```

看到类似以下输出即说明一次前向、反向、优化和 checkpoint 保存都成功：

```text
step=1/1 loss=... token_accuracy=...
Training complete: ...
```

注意：`CUDA_VISIBLE_DEVICES=1` 会把物理 GPU 1 重映射为进程内的 `cuda:0`，此时不要设置 `DEVICE=cuda:1`。脚本目前自身的默认值可能不适用于这种重映射方式，因此单卡运行时建议始终显式指定这两个变量。

### 6.2 约一轮数据的训练示例

先用 `--prepare-only` 得到本次样本数 `N`。当 `BATCH_SIZE=1`、`GRAD_ACCUMULATION_STEPS=1` 时，可把 `MAX_STEPS` 设为 `N`，大约遍历一次数据。例如有效样本数为 1394：

```bash
CUDA_VISIBLE_DEVICES=1 \
DEVICE=cuda:0 \
BATCH_SIZE=1 \
GRAD_ACCUMULATION_STEPS=1 \
MAX_STEPS=1394 \
SAVE_FREQ=200 \
DATALOADER_WORKERS=2 \
USE_TENSORBOARD=1 \
bash scripts/train_real_robot_imitation_3cam.sh --log-freq 10
```

有效 batch size 近似为：

```text
BATCH_SIZE × GRAD_ACCUMULATION_STEPS
```

一轮对应的 optimizer step 数近似为：

```text
ceil(N / effective_batch_size)
```

随着 `data_collect` 中的数据变化，`N` 也会变化，不要永久照抄 1394。

### 6.3 主要参数含义

| 参数 | 含义 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | 限制进程可见的物理 GPU |
| `DEVICE` | 进程内部使用的 PyTorch 设备 |
| `DATA_ROOT` | 真机采集数据根目录 |
| `PRETRAINED_CHECKPOINT` | 用于初始化的基础 VLA/LoRA checkpoint |
| `OUTPUT_ROOT` | 实验输出根目录 |
| `TASK_LABEL` | 固定文本任务描述 |
| `BATCH_SIZE` | 每次 dataloader 取出的样本数 |
| `GRAD_ACCUMULATION_STEPS` | 累积多少个小 batch 后更新一次参数 |
| `LEARNING_RATE` | 学习率 |
| `MAX_STEPS` | optimizer 更新总步数，不是 epoch 数 |
| `SAVE_FREQ` | 每隔多少个 optimizer step 保存一次 |
| `DATALOADER_WORKERS` | dataloader 子进程数量；排查问题可设为 0 |
| `LORA_RANK` | LoRA rank |
| `USE_TENSORBOARD` | 1 开启 TensorBoard，0 关闭 |
| `--log-freq` | 每隔多少步打印并写入训练指标 |

## 7. 输出目录与 checkpoint

每次运行会创建独立目录：

```text
runs/real_robot_imitation/<run_name>/
├── action_range_table.json
├── action_range_table.csv
├── dataset_manifest.json
├── train_metrics.jsonl
├── tensorboard/
└── checkpoints/
    └── agent_checkpoint_epoch_<step>/
        ├── agent_lora/
        └── agent_extra_layers.pt
```

目录名中的 `agent_checkpoint_epoch_60` 在当前实现里实际表示第 60 个 optimizer step，不是完整遍历了 60 个 epoch。

`SAVE_FREQ=1` 只适合 `MAX_STEPS=1` 的 smoke test。7B 模型即使只保存 LoRA 和额外层，频繁 checkpoint 仍会快速占满磁盘。正式训练建议使用 200、500 或 1000 等更大的保存间隔，并定期检查磁盘空间。

## 8. 如何看训练指标

训练日志中的 `token_accuracy` 是离散动作 token 的分类准确率，不是真机任务成功率。

当 `BATCH_SIZE=1` 时，每个样本有 48 个 token：

- `token_accuracy=0.9167` 约等于 48 个 token 中预测对 44 个。
- `token_accuracy=0.8333` 约等于 48 个 token 中预测对 40 个。

如果数据里只有 `dz` 明显变化，其他 5 维几乎是常数，那么 8 步中这 5 维就占 `8 × 5 = 40` 个容易预测的 token。模型即使没有学好 `dz`，总准确率也可能接近 83.33%。因此应优先查看：

```text
active_token_accuracy
complete_action_accuracy
各维 token accuracy
各维物理单位 MAE
loss
```

训练器会把指标写入 `train_metrics.jsonl`，开启 TensorBoard 后也会写到 `tensorboard/`。启动方式：

```bash
tensorboard --logdir runs/real_robot_imitation --port 6006
```

本机浏览器访问 `http://localhost:6006`。如果训练机是远程服务器，需要使用 VS Code 端口转发或 SSH tunnel 转发 6006 端口。

当前指标都是训练 batch 上的指标，且 batch size 为 1 时波动很大。真正的“成功率”只能通过独立验证集或真机 rollout 定义，例如成功次数除以总测试次数；当前训练脚本没有实现这个成功率评估。

## 9. 权重保存、恢复、相对偏差评测与原推理兼容性

Actor 的原始默认动作形状和图像配置仍从原 LIBERO 常量读取，因此没有显式启用三视角配置时，原来的推理路径保持不变。三视角训练只在本训练入口中设置 3 图、6 维、8 步和关闭 proprio。

三视角 checkpoint 包含两部分：

- `agent_lora/`：LoRA 权重。
- `agent_extra_layers.pt`：精简动作头、模型配置和动作范围表等额外信息。

不要只复制 `agent_lora/`，否则可能缺少六维动作头和范围表。加载完整 checkpoint 时使用 Actor 提供的入口：

```python
actor.safe_load_model(
    "/path/to/agent_checkpoint_epoch_N"
)
```

构建 Actor 时必须使用与训练一致的基础权重和三视角配置。加载范围表后，`post_process` 会把离散预测还原成对应维度的物理量；没有范围表时，它只能保持原工程的归一化动作语义。

目前没有完整的真机在线控制脚本。上线前还需要实现并验证：三路相机同步、与训练一致的预处理和视角顺序、相同 task label、action chunk 消费策略、增量动作在机器人坐标系中的执行、安全限幅、碰撞保护与急停。

训练完成后，可以用保存的 checkpoint 对真实轨迹逐样本比较：

```bash
CUDA_VISIBLE_DEVICES=2 \
DEVICE=cuda:0 \
DATA_ROOT=/mnt/data/lcx2/yanjieworkspace/data_collect \
PRINT_EVERY=1 \
PRINT_ACTIONS=1 \
bash scripts/evaluate_real_robot_imitation_3cam.sh \
  runs/real_robot_imitation/<run>/checkpoints/agent_checkpoint_epoch_<step>
```

脚本只把三张图片作为可变观测输入，CSV 位姿只在模型外部重建真实动作标签。运行时会实时打印累计相对偏差，并在 checkpoint 的 `evaluations/<timestamp>_relative_deviation/` 下保存：

```text
evaluation_config.json
predictions.jsonl
running_summary.json
summary.json
```

其中每个有效维度的相对偏差定义为：

```text
relative_deviation = MAE / (checkpoint raw_max - checkpoint raw_min)
```

设置 `PRINT_ACTIONS=1` 后，每次达到 `PRINT_EVERY` 指定的 batch 间隔，终端还会打印该 batch 中每个样本未来 8 步的预测动作、真实动作和二者误差。平移按毫米显示，旋转按弧度显示。若希望每个样本都打印，应同时设置 `PRINT_EVERY=1`；数据量较大时可设为 10 或 50，减少终端输出。

恒定维度的范围为零，脚本会将其标记为 inactive，并把相对偏差保存为 `null`，不会通过添加任意分母制造误导性百分比。`predictions.jsonl` 保留每个样本完整的 8×6 预测和真实动作，`summary.json` 保存每维 MAE、signed bias、RMSE、相对偏差及 token accuracy。

默认会评测全部对齐数据；如果这些 session 也参与过训练，结果属于训练集拟合误差，不能当作未见数据上的泛化误差。可用 `--session-regex` 只选预留的完整 session，避免按相邻帧随机切分造成泄漏。

## 10. 常见修改需求及入口

### 10.1 更换或增减相机

修改 `rl/train_real_robot_imitation.py` 中的 `CAMERA_DIRS`，并同步修改 `num_images_in_input` 和在线推理的相机顺序。当前骨干融合后每个视角占 6 个通道，因此 N 个视角对应 `(B, 6N, 224, 224)`。

只改目录名、不改模型配置会造成通道数或拆图数量不一致。训练和推理的视角顺序也必须完全一致。

### 10.2 修改动作维度

同步检查并修改：

- `ACTION_NAMES`
- `STATE_COLUMNS`
- 位姿差分/`pose_delta` 逻辑
- Actor 的 `action_dim`
- 真机端动作执行和安全范围

不能只改 `action_dim`，否则标签含义、范围表和执行端会错位。

### 10.3 修改 action chunk 长度

统一修改训练入口传入的 `--num-actions-chunk`，并让推理端使用相同值。checkpoint 的输出头形状依赖 `action_dim × num_actions_chunk`，旧 checkpoint 不一定能直接加载到不同长度。

### 10.4 修改动作标签来源

当前标签来自 `robot_command_state.csv` 中相邻 command TCP 位姿。如果要改成实际反馈位姿、速度、关节量或固定时间间隔动作，需要修改数据扫描、列定义、时间对齐和动作差分逻辑，并重新定义单位和执行语义。不要在不知道含义的情况下只换 CSV 文件名。

### 10.5 修改范围策略

当前使用全数据的 raw min/max。若要改为 q01/q99、固定安全范围或训练集统计量，需要同时保证：

1. 范围表生成逻辑一致。
2. 连续动作到 token 的训练编码一致。
3. token 到连续动作的推理解码一致。
4. checkpoint 保存和恢复的范围一致。

使用分位数时还要明确超范围动作是 clip、报错还是保留特殊 token。

### 10.6 更换基础权重

优先通过环境变量覆盖，不必修改脚本：

```bash
PRETRAINED_CHECKPOINT=/new/checkpoint/path \
bash scripts/train_real_robot_imitation_3cam.sh --prepare-only
```

开始训练前必须先做一步 smoke test，确认词表、动作头、LoRA target module 和视觉配置能够正确加载。

### 10.7 增加验证集

当前没有独立验证集。建议按完整 session 切分 train/validation，不能随机按相邻帧切分，否则同一条轨迹的高度相似图片会同时进入训练集和验证集，造成数据泄漏。验证时至少记录六维 token accuracy、物理 MAE 和完整 action accuracy。

### 10.8 增加断点续训

当前 checkpoint 主要用于模型加载，没有完整保存 optimizer、当前 step、随机数状态和 dataloader 状态，因此不属于严格的断点续训。增加 resume 时应一起保存和恢复这些状态，并确认 scheduler 的步数连续。

## 11. 已知限制和风险

- 当前部分采集数据可能主要只有 `dz` 变化，模型容易靠常数维度得到较高总 token accuracy。
- 三视角权重在形状上兼容基础视觉编码器，但第三视角仍有分布偏移，必须训练和验证。
- 动作由相邻有效相机帧产生；丢帧或帧率变化会改变动作对应的时间间隔。
- 每次训练重新统计范围，因此不同 run 的 token 物理含义可能不同。
- task label 当前是固定文本；修改后会改变语言条件分布。
- TensorFlow/XLA 打印 cuDNN、cuFFT、cuBLAS 重复注册信息通常不是 PyTorch 训练失败；应以是否继续完成模型加载、反向和保存为准。
- 当前没有真机成功率评估、在线安全控制和完整 resume，需要在部署或大规模实验前补齐。

## 12. 提交前检查

语法和格式检查：

```bash
python -m py_compile \
  rl/actor_critic_model_discrete.py \
  rl/train_real_robot_imitation.py

bash -n scripts/train_real_robot_imitation_3cam.sh
git diff --check
```

确认真机数据被忽略且没有历史跟踪文件：

```bash
git check-ignore -v data_collect/2026-08-06/11-42-22/session_meta.json
git ls-files data_collect
```

第一条应显示命中 `.gitignore` 中的 `/data_collect/`；第二条应没有输出。

建议仅提交本流程相关文件：

```text
.gitignore
rl/actor_critic_model_discrete.py
rl/train_real_robot_imitation.py
scripts/train_real_robot_imitation_3cam.sh
scripts/REAL_ROBOT_IMITATION_HANDOFF.md
```

不要使用未经检查的 `git add .`。先运行 `git status --short`，避免把采集数据、实验输出或无关的 `scripts/class Twomlp.py` 一并提交。
