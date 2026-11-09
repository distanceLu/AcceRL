# AcceRL · WEAVER original95

这个分支运行 OpenVLA Policy 与 WEAVER World/Reward Model 在 LIBERO Spatial task 0 上的联合训练。入口是仓库根目录的 `./run.sh`。当前默认为用户选定的橙线 `1793355532` 配方，从原始 OpenVLA、蒸馏 Policy 和 native3200 World 权重开始，重新建立 optimizer 和 Replay。真实训练采集最多执行150个动作；任务和评估上限仍为220。训练等待用户明确停止，不设次日自动截止；原1M模拟调用预算和故障保护保留。

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

在普通服务器上，训练由当前终端的前台 supervisor 管理，长训可在 tmux 或批作业中运行。默认输出在 `.weaver-runs/<实验名>/`，通过 `WEAVER_OUTPUTS` 可换盘。日志为 `train.log`，TensorBoard 事件在 `runs/Libero/`，Policy 与 World 权重在 `checkpoints/`。普通服务器入口仍跳过单独的最终评估。在原实验服务器上，启动、查询和停止走已有队列；本次开启保存退出后的四卡固定评估，初始/最终Policy各greedy/stochastic、各50初态共200回合，每回合最多220动作。训练中的10个Eval worker沿用橙线设置。

## 复现口径

2026-11-09按用户选择恢复橙线 `weaverintegrated_1030_181824_cf621e5b` 的算法实现：共享推理池current-critic刷新、batch优势归一化、独立chunk抽样、单训练组准入和原03_adaptive调度。每个 Policy batch 有144个真实chunk和432个想象chunk，Policy/World global batch为576/1024，每5次Policy更新配1次World更新。学习配方唯一新增变量是采集150动作预算；150步未成功按budget censor处理，Policy bootstrap，RM标签mask为0，World仍可训练已观察前缀。正式任务失败仍以220步为界。详细变更见[采集预算实验](docs/collection150.md)。

此前粉线的完整想象轨迹、fused JIT GAE、Advantage EMA和双训练组并行保存在Git历史提交 `5d728230e87a915908d3a53ad9580b536468482a` 中。当前橙线实验不启用这些改动。历史滚动成功率可混合Policy版本，并不保证新实验达到同一曲线。

## 代码和实验版本

在 `codex/weaver-original95-runnable` 分支直接修改训练代码和配置，测试后只提交本次变更。训练入口和原站队列共用 `weaver_run/recipe/` 下的参数与配置。历史 original95 的首个可移植提交是 `87a65a2c336877cc8877dd56a21c7208ea41e8d5`。

原站的开发目录是 `/mnt/data/lcx1/dong/WEAVER/code`。在 `/mnt/data/lcx1/dong/WEAVER` 运行 `./run.sh`，每次新的默认训练入队前会执行普通 Git push，并核对 GitHub 的分支提交。未提交改动或远端分叉会阻止新启动。队列从一个重复使用的独立 Git worktree 执行该提交，长训期间可在开发目录准备下一次修改。实验的 `experiment.json` 记录 `git_branch` 和 `git_commit`，`./run.sh status --id 实验名` 也可查看提交。其他服务器上的启动入口校验已发布的提交，使用者可用只读仓库权限运行。

仓库提供同一训练配方及启动前的 CPU 校验。异步采集、抽样和 GPU 执行顺序可能改变后续学习轨迹。保存的权重可用于推理；精确续训还需要 optimizer、Replay 和随机数状态。
