# Model-free 模式说明（2026-11-16）

这次修改在现有受管训练入口增加 `--model-free`。不开开关仍使用原 WEAVER 配方。开关默认关闭在线 Eval，这是用户确认后的选择；可明确设为 4。代码修改不启动训练，也不切换已运行进程。

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
| 在线 Eval worker | 3，推理副本1/2/3 | 0；可选4，对应副本0/1/2/3 |
| Policy batch 来源 | 144 real + 432 imagined | 576 real |
| Policy 微批次/累积 | 普通8/18；每5步4/36 | 恒8/18，全局576 |
| World 更新 | 每5次 Policy 更新一次 | 无 |
| cost scheduler、想象库存补货 | 开启 | 关闭 |
| Value 刷新 | 当前 critic，支持依赖裁剪 | 当前 critic 完整刷新；没有 cost dispatcher 时不走裁剪路径 |
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

Eval0 不写伪造的0% Eval曲线；不存在的 World/RM/Imagine 服务也不写伪零曲线。终端明确显示在线 Eval 关闭。低频抽样双相机预览沿用现有设置，不额外调用世界模型。

每25次 Policy 更新保存现有权重快照；初始快照和最终保存流程保留。model-free manifest 只包含 Policy，外层验收按 Policy-only 服务与四rank计数检查，固定评测读取同一权重格式。保存权重不等于精确续训状态。

## 验证范围

CPU 测试覆盖开关与旧模式回归、真实采集/150步失败、真实数据独立成批、当前Value GAE、四rank分片、四卡映射、Eval0/4、TensorBoard筛选、预算并发、Policy-only保存/读取/最终审计，以及受管/可移植启动参数传递。实际 GPU 模型加载、EGL上下文、首个 NCCL/ZeRO 更新和长训吞吐仍需新实验验收；本次不占用现有训练资源。
