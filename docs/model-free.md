# Model-free 模式说明

现有受管训练入口使用 `--model-free`。不开开关仍使用原 WEAVER 配方。model-free 默认关闭常驻在线 Eval，并在固定间隔运行有限的50例验证；仍可明确设为4个在线 Eval。代码修改不启动训练，也不切换已运行进程。

## 实际训练什么

LIBERO 环境给出外部相机、手腕相机及任务文字，Policy 输出一组离散动作，解码成机器人 7 维控制量。每次预测最多执行 8 个动作。执行后从真实仿真环境取得成功标签和下一次观察，写入 Policy Replay。这里的“真实”指真实运行的仿真环境，相对于世界模型生成的想象轨迹，不代表真机。

训练保留 GIPO 的 Policy loss 和 Value loss。Value critic 预测回报，为 GAE/优势估计提供依据；model-free 并不取消 critic。实际行为 logits、Policy 版本、末个不完整动作块的 mask 都保留。每次抽样使用当前 Policy Value 刷新 GAE，沿用 Policy 最大年龄16、最多成功使用4次的约束。

model-free 不创建 World/Reward 模型、想象 worker、WM Replay、WM 训练窗口或想象 seed。新的 CPU `RealCollectionCoordinator` 只管理初态分配和仿真预算的元数据，不保存相机帧。Policy Replay 允许一整个训练 batch 全部来自真实数据，不再等待想象样本。

## 与现有运行配方的区别

| 项目 | 原 WEAVER 模式 | model-free 默认 |
| --- | --- | --- |
| Policy Trainer / Inference | 各4，四卡各一份 | 保留 |
| World Trainer / Inference | 各4，含集成 RM | 都不创建 |
| Real / Imagine worker | 1 / 4 | 4 / 0 |
| 常驻在线 Eval worker | 3，推理副本1/2/3 | 0；可选4，对应副本0/1/2/3 |
| 训练中固定验证 | 无新增 | 默认每消耗10万仿真步；P0基线及最终版本也验证 |
| Policy batch 来源 | 144 real + 432 imagined | 576 real |
| Policy 微批次/累积 | 普通8/18；每5步4/36 | 恒8/18，全局576 |
| World 更新 | 每5次 Policy 更新一次 | 无 |
| cost scheduler、想象库存补货 | 开启 | 关闭 |
| Value 刷新 | 当前 critic，经 cost dispatcher 裁剪 | 当前 critic，普通与 autobatch 均按所选动作块的 GAE 依赖裁剪，详见[裁剪与回退](critic-dependency-pruning.md) |
| 渲染/物理 | EGL / CPU | 保留 |
| 训练任务/初态 | spatial task0、50初态 | 保留 |
| 训练动作上限 | 150；到限失败 | 保留 |
| 固定评测动作上限 | 220 | 保留 |
| 真实仿真预算 | 100万步，含 reset settling | 保留，4个worker共享一个全局预算 |

四个 Real worker 是 CPU 仿真进程，各自的 EGL 渲染和 Policy 请求绑定到对应 GPU。GPU4–7 每卡常驻一份 Policy trainer 和一份 Policy inference；不是四个独立策略分别训练，trainer 是同一策略的四个分布式 rank。

只有累计仿真预算不足而导致的更早截断才保留 bootstrap。150动作内没有成功属于训练任务失败，不能重标未知成功。训练成功率与220动作固定评测不是完全相同口径。

这版关闭旧的 cost scheduler，而非实现新的 Buffer 双向准入研究方案。实际吞吐、显存和成功率收益需要 GPU 实验验证，不能由进程数下降直接推出。

## 当前 checkpoint 与两个视角

当前 Policy 本来就是 `num_images_in_input=2`、`use_proprio=false`。`rl/utils.py::prepare_one_obs` 读取外部相机和腕部相机；model-free 继续调用同一函数。去掉世界模型不改变 Policy 输入接口。

初始化先读 `pretrained_checkpoint`，再从非空 `checkpoint2` 取 `student_state_dict`，对整个 ActorCritic 执行 strict load。因此当前初始策略是 **pretrained + distilled checkpoint2 的组合**；checkpoint2 不只是 critic 参数。model-free 保留该组合及原校验和，不能将其成功率归因于清空 checkpoint2 后的预训练模型。

现有组合的历史初始评测曾有 greedy 38/50、stochastic 40/50；那是特定任务、初态与渲染协议的结果，不能证明 model-free 训练后的成功率，也不能当作 teacher 示例的 object 权重结果。应使用固定 checkpoint、同50 case/seed/greedy/batch1/EGL 比较。

如要使用后来训练得到的 Policy snapshot，需要按 `accerl_parallel_weights_v1` 的恢复语义加载。它不是旧脚本要求的 `student_state_dict` 文件，不能仅把路径替换进 `--checkpoint2`。本次没有改变初始化来源，也不提供 optimizer/Replay/RNG 的精确恢复。

## 老师示例与这次实现

`run_ppo_training_example.sh` 确实调用无世界模型的旧入口 `rl/ds_libero_ppo_discrete.py`，但示例同时选了 object benchmark、默认全部10任务、开启 proprio、另一份预训练权重、空 checkpoint2、SAPO、10 Real 和1 Eval。旧代码强制 OSMesa，并没有当前四卡常驻共置的放置约束。直接运行它会同时改变很多变量。

老师建议的 `--clip-mode gipo` 是支持的；当前 WEAVER 配方已经使用 GIPO。此实现保留当前 Policy 配方，使去掉 World/Imagine 的影响更容易解释。Eval 不参与训练梯度，Eval0 可以减少评测开销；仍保留训练结束后的独立固定 checkpoint 评测。

## 启动与日志

未来获得启动授权并完成同分支同步后，原站只使用：

```bash
cd /mnt/data/lcx1/dong/WEAVER
./run.sh original --model-free --id model-free-first
# 如确实需要4个在线Eval：
./run.sh original --model-free --num-eval-workers 4 --id model-free-eval4
```

开关、Eval数量、有效 pipeline 和 `git_commit` 保存到实验记录。不可复用绑定了另一配置的实验名。模式开启时不做 World 权重或独立 native 环境的 admission，但仍校验 Policy 资源和当前控制环境、两路观察及初态。

TensorBoard 原站仍写到 `AcceRL-scheduler/runs/Libero/` 的监控目录，实验名明确包含 `ModelFree_Real4_Imag0_Eval0` 或 `Eval4`。保留 Policy/Value loss、真实平均回报与幕长、Policy熵/KL，新增显式真实成功率、Policy/Value学习率、累计真实动作数、完成回合的动作吞吐及更新速度。完成回合动作吞吐按两次日志间已完成回合累计动作数计算，可能随回合完成出现波动，不是逐动作即时计时。

Eval0 不写伪造的0%在线 Eval曲线；固定验证只有完整跑完50例才记录成功率。不存在的 World/RM/Imagine 服务也不写伪零曲线。终端明确显示在线 Eval 关闭。低频抽样双相机预览沿用现有设置，不额外调用世界模型。

每25次 Policy 更新保存现有权重快照；初始快照和最终保存流程保留。model-free manifest 只包含 Policy，外层验收按 Policy-only 服务与四rank计数检查，固定评测读取同一权重格式。保存权重不等于精确续训状态。

## 自动停止与固定验证

**正式达标条件为最近 1,000 个有效真实训练回合中至少 990 次成功。** 四个 Real worker 共享一个窗口，按统计端收到完整回合的顺序，每完成一个有效回合就更新一次。首次达到 980 次成功时记录 98% 里程碑，首次达到 990 次成功时锁定 99% 里程碑。窗口未满时继续采集，即使前 999 个回合全部成功也继续训练。150 动作到限仍未完成任务算一次失败。因剩余预算不足而提前截断的回合、在线 Eval 和固定验证均不进入窗口。

达到 99% 后，在安全更新边界结束训练，记录 `rollout_success_target_reached`，覆盖保存 `latest_weights.pt`，再进入原站的收尾评测。已发出的更新可以完成，后续回合导致成功率回落也保留首次达标记录。比较到达 98% 或 99% 的速度时，可查看事件 `model_free_rollout_success_milestone`，或读取最终权重清单 `metadata.model_free_success.crossings` 中对应的 `elapsed_seconds`（从作业启动到统计端收到达标回合）。最终保存的 Policy 版本对应更新边界，窗口中的回合可能由多个 Policy 版本采集。

若仍未达标，累计 100 万仿真步的预算仍会结束训练，原因记录为 `real_budget_exhausted`。预算包含动作和 reset settling。所有已发出的回合均须结清，且无采集故障。剩余步数不足以完成一次 reset 再执行一个动作时也视为预算耗尽。人工停止与故障沿用各自的处理流程。

默认在Real worker开始采集前先测P0，然后每累计消耗10万仿真步，在完整 Policy 更新边界暂停新采集、排空现有回合并固定当前策略版本，用同一 spatial task0 的50初态、greedy、推理batch1、EGL、220动作上限验证。单个临时验证worker串行调用GPU4的推理副本0；全部Policy副本继续常驻。验证不加入 Replay或训练统计，不参与训练梯度，也不消耗100万步训练预算，但会增加实际运行时间。完成后释放临时验证环境、恢复推理批次上限及训练，常驻在线 Eval仍为0。

P0 或中途验证得到 50/50 时继续训练。固定评测的首次满分和连续未改善次数用于观察趋势。如果最终策略版本已有完整验证结果，则复用这次结果。原站收尾评测分别比较初始与最终 checkpoint 的 greedy 和 stochastic 表现。收尾时保存的是实际结束训练时的权重。

固定验证的间隔、平台期诊断参数会写入实验记录。例如：

```bash
./run.sh original --model-free --id model-free-fixed \
  --fixed-validation-interval-env-steps 100000 \
  --fixed-validation-patience 3 \
  --fixed-validation-min-env-steps 300000 \
  --fixed-validation-min-delta-successes 1
```

`--fixed-validation-interval-env-steps 0` 关闭训练中固定验证。显式启用常驻在线 Eval 时，固定验证默认关闭。两种配置都按真实训练回合判定 99%。

日常趋势看 TensorBoard 的 `Rollout/_Global/Success_Rate`。固定评测看 `FixedValidation/Success_Rate`、`Success_Count`、`Budget_Env_Steps` 和 `Duration_Seconds`。每次完整验证的 50 例明细保存在训练数据目录的 `fixed_validation/P<版本>.json`，最近一次也写入 `latest.json`。这些初态来自训练初态池，每例占 2 个百分点。固定评测适合在相同条件下比较 checkpoint，未见初态上的泛化需要独立评估。

原站在验收 99% 达标时核对首次达标的窗口、最终 Policy 更新步与权重记录。四个 rank 的更新计数、Replay 使用记录、最终权重 CPU 审计和进程清理也纳入作业验收。

## 显存感知的 batch 校准（默认关闭）

在新实验中显式加入 `--batch-autotune`：

```bash
cd /mnt/data/lcx1/dong/WEAVER
./run.sh original --model-free --batch-autotune --id model-free-batch-tuned
```

这个开关只接受 model-free、常驻在线 Eval0。未提供开关时仍使用训练 micro8/GAS18 和原来的同步 critic micro2；开关不会改变已经运行的固定源码实验。启动仍须同步并核对同一 GitHub 分支，实验记录、runner状态和实际命令均冻结 `batch_autotune` 与 `git_commit`。

校准使用新实验前期的正常训练更新，不会额外抽取一批 Replay 数据做隐藏更新。先比较 critic 刷新 batch2/4/8，选定后重新测训练 micro8 的基准，再比较 micro16/GAS9、micro24/GAS6；全部配置保持四卡 global576、每卡 local144。每个候选窗口包含6次完整更新，首轮用于预热，其余测量取中位数；无跳过时共36次更新，之后固定选择。校准期间同样按真实回合达标、预算或人工请求结束训练。

每次放大前，统一检查四张卡的预计峰值，预留 `max(显存总量的15%, 8 GiB)`，对观测到的激活增量使用1.25倍系数；没有有效测量或预计空间不足时跳过候选。这个预测不能证明下一批一定不会 OOM，不在某个分布式 rank 内捕获 OOM 后擅自减 batch 重跑。

刷新耗时按观测数量归一化，训练耗时按固定576个样本的一次更新计算；另记录获准更新后至权重发布完成的耗时（包含刷新、训练、发布，不含等待 Replay 补货及固定验证）。端到端收益仍需结合原有采集速度和总墙钟记录比较。候选还要通过真实动作推理延迟约束；达到显存上限并不是选择目标。训练 loss 保留原来连续8个样本一组的有效动作归一化权重，再合并成更大的 micro，以避免部分动作 chunk 改变隐含样本权重。浮点计算和随机算子的执行顺序仍可能不同，不能承诺逐位一致。

开启后 critic 使用按 chunk 让出执行机会的刷新端点，保持同一个已发布 Policy 版本与观测顺序。真实 rollout 仍然是一卡一个worker，动作推理上限不会因为这个开关自动增加；固定验证保持原来的50例、greedy、batch1、EGL和220动作上限，不纳入校准。

配置选择、显存保护与跳过原因通过 `BatchAutotune` 遥测、TensorBoard以及实验内JSON记录。CPU测试能够验证边界、权重计算和参数传递；真实GPU上的显存峰值、四卡ZeRO行为和提速幅度仍需要首次开启后的实验测量。

## 验证范围

CPU 测试覆盖开关与旧模式回归、真实采集/150步失败、真实数据独立成批、当前Value GAE、四rank分片、四卡映射、Eval0/4、TensorBoard筛选、预算并发、Policy-only保存/读取/最终审计，以及受管/可移植启动参数传递。实际 GPU 模型加载、EGL上下文、首个 NCCL/ZeRO 更新和长训吞吐仍需新实验验收；本次不占用现有训练资源。
