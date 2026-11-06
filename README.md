# AcceRL · WEAVER original95

这个分支运行 OpenVLA Policy 与 WEAVER World/Reward Model 在 LIBERO Spatial 上的联合训练。入口是仓库根目录的 `./run.sh`。默认从原始 OpenVLA、蒸馏 Policy 和 native3200 World 权重开始，重新建立 optimizer、Replay 和 Advantage EMA。训练持续到手动停止。

## 在另一台服务器上启动

```bash
git clone --single-branch --branch codex/weaver-original95-runnable https://github.com/distanceLu/AcceRL.git
cd AcceRL
cp .weaver.env.example .weaver.env
# 在 .weaver.env 中填好本机的权重目录、已分配的四张卡和 MPS 管道。
./run.sh
```

**首次配置。** 使用 Linux x86-64 单机、四张同级大显存 GPU、兼容 CUDA 13 的 NVIDIA 驱动，以及该用户已有的空闲 MPS server（默认和 server active thread 均为 100%）。把 `WEAVER_GPU_IDS` 设为分配到的四个物理编号，也可直接继承作业的 `CUDA_VISIBLE_DEVICES`。把 `CUDA_MPS_PIPE_DIRECTORY` 设为对应管道。原配方的 Ray 使用 32 个 CPU、328 GiB object store；启动时要求 `/dev/shm` 可用至少 336 GiB，主机和受限 cgroup 可用内存至少 392 GiB。输出盘要为模型快照、Ray spill 和遥测预留空间。

**权重由运行者提供。** 默认根目录是仓库下的 `assets/`，也可以将 `WEAVER_ASSETS` 指向共享盘。沿用原文件名的用户可通过环境文件分别指定五个绝对路径，无须重命名或复制大文件。

| 根目录下的默认位置 | 内容 | 单独覆盖 |
| --- | --- | --- |
| `openvla/` | 原始 OpenVLA 模型目录，含 tokenizer、配置和权重分片 | `WEAVER_POLICY` |
| `policy_distilled.pt` | 原始蒸馏 Policy checkpoint | `WEAVER_DISTILLED` |
| `native3200/` | 完整 WEAVER 第 3200 步 inference bundle，保留原七个文件 | `WEAVER_BUNDLE` |
| `image_encoder/` | WEAVER 图像编码器 | `WEAVER_IMAGE_ENCODER` |
| `task_encoder/` | WEAVER 任务编码器 | `WEAVER_TASK_ENCODER` |

**Python 环境。** 已有匹配环境时，在环境文件里填写 `WEAVER_CONTROL_PYTHON` 和 `WEAVER_NATIVE_PYTHON` 的绝对路径，并配置已打补丁的 `WEAVER_LIBERO_ROOT` 或已有的 LIBERO 配置。也可由入口首次安装到仓库自己的两个虚拟环境。安装需要 Python 3.11（含 venv 和开发头文件）、Git、网络、C/C++ 编译器、CUDA toolkit/NVCC、OSMesa/OpenGL 和 MuJoCo 所需系统库。Policy 用 PyTorch 2.12 / CUDA 13.0，World 用 PyTorch 2.7 / CUDA 12.6。依赖版本、Git 源码版本和 LIBERO 补丁随本分支提供。需要内网镜像或预编译 wheel 的站点可使用 pip 配置或预配环境。

## 运行、检查和停止

```bash
./run.sh check               # 校验初始资源、训练实参和仿真 reset；不启动 Ray 训练
./run.sh original --id exp1  # 给新实验命名；./run.sh 可自动命名
./run.sh status --id exp1    # 在另一个终端查询
./run.sh stop --id exp1      # 请求在安全边界保存退出；再用 status 确认 finished
```

在普通服务器上，训练由当前终端的前台 supervisor 管理，长训可在 tmux 或批作业中运行。默认输出在 `.weaver-runs/<实验名>/`，通过 `WEAVER_OUTPUTS` 可换盘。日志为 `train.log`，TensorBoard 事件在 `runs/Libero/`，Policy 与 World 权重在 `checkpoints/`。保存后跳过单独的最终评估，训练中的 10 个 Eval worker 正常运行。在原实验服务器上，同一入口的启动、查询和停止走已有队列，保留该站的资源预约及身份核验。

## 复现口径

本分支对应 `weaverintegrated_1106_142611_cafe8ae1` 使用的 original95 配方及其历史参照 `weaverintegrated_1103_172031_a7557d7d`。每个 Policy batch 有 144 个真实 chunk 和 432 个想象 chunk，Policy/World 的 global batch 为 576/1024，每 5 次 Policy 更新配 1 次 World 更新。完整想象轨迹、fused JIT GAE、全局 EMA、双训练组并行、反馈调权和 Eval10 的 1000-episode 滚动窗口均来自该配方。历史约 17 小时到达的 95% 是跨多个 Policy 版本的滚动观测；新实验应按实际评估记录判断。

`weaver_run/recipe/` 保存从运行进程提取的命令、82 个解析参数及资源清单。`weaver_run/reference-source.json` 记录原始 267 个引擎文件的 SHA-256；可移植版本在三个文件中适配物理 GPU 编号和 DeepSpeed 安装位置，另外 264 个逐文件一致。`weaver_run/vendor/` 附所需第三方源码及许可证。`weaver_run/reference_site/` 保存原站点调度入口的参考代码，其他服务器只需根入口。

仓库提供同一训练配方及启动前的 CPU 校验。异步采集、抽样和 GPU 执行顺序可能改变后续学习轨迹。保存的权重可用于推理；精确续训还需要 optimizer、Replay 和随机数状态。
