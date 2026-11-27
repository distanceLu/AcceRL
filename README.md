# AcceRL · WEAVER original95

这个分支运行 OpenVLA Policy 与 WEAVER World/Reward Model 在 LIBERO Spatial task 0 上的联合训练。入口是仓库根目录的 `./run.sh`。当前代码采用用户选定的橙线 `1793355532` 算法，从原始 OpenVLA、蒸馏 Policy 和 native3200 World 权重开始，重新建立 optimizer 和 Replay。新实验规则把真实训练任务上限设为150动作：达到上限仍未成功记为正式失败，RM使用有效零标签，Policy按终止处理且不bootstrap；正式评估仍为220动作。代码配置不代表新进程已启动，实际运行以队列回执中的Git提交号为准。训练等待用户明确停止，不设次日自动截止；原1M模拟调用预算和故障保护保留。

LIBERO-Spatial Task 0 的四卡 Model-free 训练已有两条实测基线。追求首次在线 990/1,000 的墙钟纪录，可复现 `mf8-compile-sat-1126-a`；需要每秒至少 50 个真实环境动作时，可用 `mf8-compile90-whole-1127-a`。实测条件和启动命令见下文的[四卡训练基准](#四卡训练基准)。

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

在普通服务器上，训练由当前终端的前台 supervisor 管理，长训可在 tmux 或批作业中运行。默认输出在 `.weaver-runs/<实验名>/`，通过 `WEAVER_OUTPUTS` 可换盘。日志为 `train.log`，TensorBoard 事件在 `runs/Libero/`，Policy 与 World 权重在 `checkpoints/`。普通服务器入口仍跳过单独的最终评估。在原实验服务器上，启动、查询和停止走已有队列；新训练默认在保存退出后执行四卡固定评估，初始/最终Policy各greedy/stochastic、各50初态共200回合，每回合最多220动作。用户要求跳过的是旧150动作预算截断实验的最终评估，不改变新任务的默认设置。

**在线Eval配置。** 当前配方启用3个Eval worker，`--eval-inference-replicas 1,2,3` 明确连接四个Policy推理副本中的后三个。在GPU4–7上对应GPU5–7；在其他服务器上对应所分配四张卡的第二至第四张，绝不额外选择未分配的GPU。第一张卡保留真实采集、想象、Value重算和训练，只移除其Eval请求。worker自身ID仍为0、1、2，220动作评估上限和greedy动作语义保持。

主曲线 `Eval/_Global/Average_Return` 使用 `Eval worker数×100` 条完成episode，本次300。`--moving-avg-window 1000` 只控制非Eval统计，真实采集、想象奖励及耗时统计不随Eval缩短。`ACCERL_EVAL_MIN_EPISODES=auto` 与实际Eval窗口对齐；显式设置大于窗口的门槛会在创建Actor前拒绝。启动事件 `online_evaluation_config` 记录worker数、逻辑副本、物理GPU、窗口及最低样本数。减少Eval可能改变共享随机数调用顺序及异步数据经历，不承诺学习等价。当前分支配方不代表已有训练已热切换。[Eval3验收说明](docs/eval3-acceptance.md)

## 可选 Model-free 模式

`--model-free` 汇总所有真实采集 worker 最近完成的 1,000 个有效训练回合。首次达到 980 次成功时记录时间，首次达到 990 次成功时在安全更新边界保存权重并停止训练。固定 50 初态评测继续记录冻结策略的表现。

`--model-free` 默认关闭，完整设置与验证边界见[Model-free 模式说明](docs/model-free.md)。启用后创建四个 Policy trainer 和四个推理副本。默认四个真实采集 worker 各绑定一张卡的 EGL renderer，下面的训练基准显式设为八个 worker。在线 Eval 默认关闭；显式 `--num-eval-workers 4` 时连接副本0、1、2、3。训练仍使用 task0、双图像、无 proprio、原 pretrained + distilled Policy 初始化、GIPO 和当前 Value 重算 GAE。真实预算1M步、训练150动作及固定评测220动作保持。

```bash
./run.sh check --model-free                       # CPU验收，无训练
./run.sh original --model-free --id model-free-01 # 获得新训练授权后使用
./run.sh original --model-free --num-eval-workers 4 --id model-free-eval4
```

原站仍从 `/mnt/data/lcx1/dong/WEAVER/run.sh` 提交，入队前同步并核对同一 Git 分支。模式、有效 pipeline、Eval 数及 `git_commit` 写入实验记录。Model-free 不检查或加载 World/Reward 权重、编码器和独立 native Python，不生成想象数据或 WM replay；当前版本关闭 cost scheduler 与库存补货。仅需原 Policy 资源及控制环境/LIBERO 资源。

TensorBoard 使用含 `ModelFree_Real4_Imag0_Eval0`（或 `Eval4`）的实验名，原站继续写入现有 `AcceRL-scheduler/runs/Libero/` 监控目录。真实采集、Policy/Value 更新与资源曲线保留；不存在的 World/Reward/Imag 和关闭的在线 Eval 不写伪零曲线。固定 checkpoint 评测保持原有独立流程；在线 Eval0 不关闭训练结束后的固定评测。

### 四卡训练基准

2026-11-24 至 11-27，原站 GPU 4–7 共完成 14 场可比的 Model-free 正式训练，均为 4× H20-3e、LIBERO-Spatial Task 0、同一 P0 和 8 个 rollout worker。下表四场均使用提交 `af68a45`、全局每步 576 个 chunk（每卡 `16×9`）和 rollout CUDA Graph。「编译」指 Trainer 的 Inductor 开关。MPS 百分比是 Trainer 客户端的配置。

| 实验 | 编译 / MPS / 采样 | 秒/步 ↓ | 环境动作/s ↑ | 推理 P95，服务 / 客户端（ms）↓ | 首次在线 990 ↓ | 触发时已发布 → 终评版本 | 冻结终评，贪心 / 随机 |
|---|---|---:|---:|---:|---:|---|---|
| `mf8-compile-sat-1126-a` | 开 / 100% / `uniform_blocks` | **26.94** | 40.90 | 1,291 / 1,980 | **1:38:06** | P209 → P210 | 49/50 · 50/50 |
| `mf8-compile90-whole-1127-a` | 开 / 90% / `whole_episodes` | **28.89** | **55.87** | 694 / 1,051 | **2:00:01** | P237 → P238 | 50/50 · 50/50 |
| `mf8-eager-sat-1126-a` | 关 / 100% / `uniform_blocks` | 31.31 | 47.70 | 1,020 / 1,391 | 2:12:16 | P244 → P244 | 50/50 · 50/50 |
| `mf8-compile90-1127-r1` | 开 / 90% / `uniform_blocks` | 30.14 | 56.73 | 692 / 1,044 | 2:27:41 | P279 → P280 | 50/50 · 49/50 |

**选配方。** 100% MPS 编译组保持这 14 场的步时与在线达标双纪录。若要求 ≤32 秒/步且 ≥50 动作/s，90% MPS 整回合组是已测配置中步时最短的一组。与同 MPS 的 `uniform_blocks` 相比，它的 238 次采样均成功按完整回合装批，稳态无梯度后缀前向从 0.468 降到 0 秒/rank/步，反向耗时维持约 14.75 秒/rank/步。四组分别只有一次达标观测，冻结终评每种解码方式各 50 回合，跨种子的平均达标时间与策略质量排序仍待验证。

**计量口径。** 稳态取 P20 至 P88 四卡权重发布 ACK 之间的 68 个更新间隔，服务 P95 取 `policy_infer_gpu_forward_and_copy`，客户端 P95 取 `policy_infer` 的等待。首次在线 990 从 GPU 队列实际启动作业算到最近 1,000 个有效回合首次出现 990 个成功，排队和冻结终评均在该区间之外。每场的原始记录位于原站 `experiments/<实验名>/` 下的 `experiment.json`、`formal_v3/training/collection.sqlite` 和 `policy_eval/summary.json`。

**微批仍用 `16×9`。** 固定 P189 权重的四卡短测 `mf8-micro16-24-p189-1127-b`（`aaf41eb`）在同一批 576 个样本上核对了 `24×6`。GAE advantage 和 value target 的最大绝对差分别为 0.00854 和 0.00109，均通过预设容差。裁剪前的完整 ZeRO-2 累积梯度相对 L2 误差为 **11.46%**，余弦相似度为 **0.99342**，超出验收范围（相对 L2 ≤2%，余弦 ≥0.999）。短测在正式计时前结束，`24×6` 的稳态显存、步时和推理 P95 仍无配对测量。完整误差在该实验的 `result.json`，方法见[配对短测协议](docs/model-free.md#p189-四卡微批配对短测)。

复现速度纪录时，在原站使用新实验名并固定实际训练提交。复现 90% MPS 整回合组时，将下面的 `--trainer-mps-percent` 改为 `90`、`--real-sampling` 改为 `whole_episodes`，另取新实验名。

```bash
cd /mnt/data/lcx1/dong/WEAVER
./run.sh start restore --id mf8-compile100-record-repeat \
  --source-commit af68a4541a844dead9be02b874b4bc8da1a0596e \
  --model-free --num-eval-workers 0 --fixed-validation-interval-env-steps 0 \
  --batch-autotune --trajectory-jit --real-workers 8 --no-micro-cache-trim \
  --compile-action-inference --trainer-compile on \
  --trainer-mps-percent 100 --real-sampling uniform_blocks
```

## 训练仿真、想象轨迹与评估回放

新任务自动抽样记录两路相机，通过同一入口打开只读观察台：

```bash
./run.sh view --id exp1 --port 8765
# 原服务器：cd /mnt/data/lcx1/dong/WEAVER 后使用同一命令
# 也可明确选择已有实验输出：
./run.sh view --root /absolute/path/to/experiment --port 8765
```

在 VS Code 的 Ports 面板转发服务器的 `8765` 端口，然后打开 `http://127.0.0.1:8765`。观察台仅监听服务器回环地址；结束观察台不会停止训练。录像写入实验目录的 `visualization/`，独立于 TensorBoard。已运行的旧进程不会自动加载新录像代码，旧日志没有保存的图像不能补造。

- **训练仿真**：全景/腕部相机、实际动作位置、服务版本、环境奖励；新规则下达到150动作仍未成功记为训练任务失败。累计模拟预算不足导致更早终止时仍标为未知结果；旧150动作预算截断录像保留原标签，不重标历史数据。
- **想象轨迹**：真实起始画面及 WM 已输出的决策边界图像、Policy/World 版本、RM 分数、预测奖励和丢弃原因。默认不额外解码，中间帧不补造；RM 分数不是已校准的成功概率。
- **评估回放**：在线或冻结 checkpoint 的视频，按初态、成功/失败、greedy/stochastic、任务或 checkpoint 搜索。抽样录像不能用于重新计算总体成功率。

抽样和保留策略在 `weaver_run/recipe/visualization.json`，每次启动复制到实验记录。默认每个 worker 首条及之后每20条真实/评估 episode、每100条想象 seed 录一次，仿真每2动作取一帧、上限128帧，末状态保留；每4秒最多发布一张最新抽样画面。每个生产者保留最多12条/64MiB视频和元数据，不是整机硬磁盘配额。帧上限达到后会记录时间缺口。设置 `enabled:false` 可对后续任务关闭。

CPU 后台队列深度2、单编码器线程，满时只跳过预览。仅复制已有 CPU 图像，不增加模型前向、训练 RPC 或 CUDA 同步，不保留训练计算图/IPC 引用。CPU转换、复制和写盘仍有成本，尚未用长训证明吞吐不变。控制环境已有 `imageio-ffmpeg==0.6.0` 依赖；缺失时显示错误且保留 JPEG 预览，不阻止训练。最近画面的时间戳和编码错误在页面中可见，画面陈旧不等于训练停止。

## 复现口径

2026-11-09按用户选择恢复橙线 `weaverintegrated_1030_181824_cf621e5b` 的算法实现：共享推理池current-critic刷新、batch优势归一化、独立chunk抽样、单训练组准入和原03_adaptive调度。每个 Policy batch 有144个真实chunk和432个想象chunk，Policy/World global batch为576/1024，每5次Policy更新配1次World更新。2026-11-10的新实验设置为 `real_training_action_horizon:150`、`real_collection_action_limit:null`，将真实训练任务定义为150动作内成功；超时失败的RM目标为有效0，Policy末端不bootstrap。仅累计1M模拟预算不足造成的更早截断保留budget censor语义。评估使用原220动作，想象horizon与其它橙线规则保持不变。这个训练目标与原橙线220动作目标不同，不能把新旧训练成功率当作同口径结果；比较Policy效果应使用相同的固定220动作评估。详细说明见[150动作训练与旧预算截断的区别](docs/collection150.md)。

此前粉线的完整想象轨迹、fused JIT GAE、Advantage EMA和双训练组并行保存在Git历史提交 `5d728230e87a915908d3a53ad9580b536468482a` 中。当前橙线实验不启用这些改动。历史滚动成功率可混合Policy版本，并不保证新实验达到同一曲线。

当前维护配方启用[critic 依赖裁剪](docs/critic-dependency-pruning.md)：先保留原副本分配与 micro 配对，再跳过没有 GAE 依赖的整块 Value 前向。完整后缀、非终止 bootstrap、原抽样和 batch 归一化保持原路径。历史固定样本预计减少约20.1%的刷新 micro，实际 GPU 提速和数值等价性尚待验证；文档包含回退方法。

维护配方同时启用[库存驱动的提前补货](docs/inventory-refill.md)：预测下一次联合发布后的首批需求，用不同 chunk 数及剩余合法使用次数判断缺口，只花已有生产额度，最多授权最早缺口的一批。宏观调度使用生产端的实际就绪状态；有界 seed 软保留在帧空间不足时让位。新逻辑在 Replay 初始填充完成后启动，不增加 GPU 前向或逐 micro 控制消息。当前运行不会在线切换；新提交的实际吞吐和学习结果需要另行 GPU 验收。

## 代码和实验版本

在 `codex/weaver-original95-runnable` 分支直接修改训练代码和配置，测试后只提交本次变更。训练入口和原站队列共用 `weaver_run/recipe/` 下的参数与配置。历史 original95 的首个可移植提交是 `87a65a2c336877cc8877dd56a21c7208ea41e8d5`。

原站的开发目录是 `/mnt/data/lcx1/dong/WEAVER/code`。在 `/mnt/data/lcx1/dong/WEAVER` 运行 `./run.sh`，每次新的默认训练入队前会执行普通 Git push，并核对 GitHub 的分支提交。未提交改动或远端分叉会阻止新启动。队列从一个重复使用的独立 Git worktree 执行该提交，长训期间可在开发目录准备下一次修改。实验的 `experiment.json` 记录 `git_branch` 和 `git_commit`，`./run.sh status --id 实验名` 也可查看提交。其他服务器上的启动入口校验已发布的提交，使用者可用只读仓库权限运行。

仓库提供同一训练配方及启动前的 CPU 校验。异步采集、抽样和 GPU 执行顺序可能改变后续学习轨迹。保存的权重可用于推理；精确续训还需要 optimizer、Replay 和随机数状态。
