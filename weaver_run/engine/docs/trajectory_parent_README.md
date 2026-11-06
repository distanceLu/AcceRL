# WEAVER JIT：完整想象轨迹候选版本

本候选在冻结的 JIT GAE＋全局 Advantage EMA 父版本上实现完整想象轨迹抽样。
当前 GPU run `weaverintegrated_1103_120142_36ef1f8b` 的源码和配置未修改；本候选未部署。

- 想象侧每次抽 108 条完整四-chunk 轨迹，每 rank 27 条；同一想象轨迹不跨 rank。
- 每 rank 固定 36 个真实、108 个想象 chunk；global576、144:432、5:1、年龄16/3及最多4次成功消费保持。
- 真实侧保留均匀 chunk 抽样和完整后缀 GAE，尽量把同片段样本放在同一 rank。
- terminal 不计算末状态价值；非 terminal 截断保留真实 N+1 状态与最后动作。缺失末状态直接报错。
- 取样、成功消费、缺货反馈、H2 补货及宏观库存都识别完整想象轨迹，避免把碎片库存算成可用供给。
- 默认仍为旧 chunk 模式。新配置：`examples/trajectory_01_imagined.json`。

完整设计、边界、损失权重与验证限制见 [docs/trajectory_sampling.md](docs/trajectory_sampling.md)。
CPU 验收入口为 `python validate_trajectory_candidate.py`，报告见 `validation/trajectory_cpu_validation.json`；
四进程 Linux CPU Gloo 独立回执见 `validation/trajectory_gloo_remote_cpu.json`。
这些验证不替代真实 Ray/ZeRO2/OpenVLA/NCCL/CUDA 验收，尚未证明端到端加速或学习效果。

新实验入口为 `bash run_trajectory.sh 01_imagined`，旧抽样对照为 `bash run_trajectory.sh 00_chunks`；
两者要求独立的 `ACCERL_EXPERIMENT_ROOT`，不提交队列任务。

下面保留父版本说明，供理解 JIT 实现和历史约束。

---

# WEAVER JIT GAE + Advantage EMA 候选

这是独立候选，不是当前服务器 GPU 运行源码。父版本为执行调度实验的228文件冻结源码，清单见 [parent_source_manifest.json](parent_source_manifest.json)。用户在2026-11-03明确要求“开始实现”；此前训练已在P225/W45安全停止，最终评估继续使用旧冻结源码。

候选将重算移到 Policy trainer：随机抽中的 chunk 使用训练前向的当前 Value，按完整后缀依赖反向计算 detached GAE；整个梯度累积窗口固定同一组四 rank 全局 EMA。首次初始化需要额外前向，随后所选状态在 fused 模式下只做一次训练前向。未选中的后续状态仍需 no-grad 前向，activation checkpoint 的反向重计算也仍存在。

- 原 GIPO/PPO/SAPO loss 分支保留；GAE 是优势估计，与 GIPO 不冲突。
- 144真实/432想象、Policy576/World1024、5:1、年龄16/3、最多4次成功消费，以及旧奖励、行为logits和真实动作mask保持。
- EMA方差包括均值漂移项，仅在四rank optimizer成功校验后提交。失败不继续后续窗口；不承诺回滚已执行的optimizer。
- 微批反向重排保留原微批动作token分母，避免改变部分执行chunk的loss权重。
- 不再占用共享推理池执行整批critic刷新。补充上下文与GAE计入Policy训练成本；调度器对该模式不再预测共享刷新需求。
- Replay和图像输入仍在CPU侧；GAE、detached carry、EMA和累积统计在trainer设备上。日志合并到更新边界回读，有限性检查仍保留。

实现入口：[jit_gae_runtime.py](jit_gae_runtime.py)、[jit_gae_plan.py](jit_gae_plan.py)、[advantage_ema.py](advantage_ema.py)、[详细分析](docs/jit_gae_design.md)。

| 配置 | 刷新位置 | 归一化 | 所选状态前向 |
|---|---|---|---|
| `00_shared_exact` | 原共享推理池 | 当前全局batch | 刷新 + 训练 |
| `01_trainer_exact` | trainer独立刷新 | 当前全局batch | 刷新 + 训练 |
| `02_trainer_ema` | trainer独立刷新 | 历史EMA Z-score | 刷新 + 训练 |
| `03_fused_ema` | trainer复用训练前向 | 历史EMA Z-score | 初始化后一次 |
| `04_fused_rms` | trainer复用训练前向 | 历史EMA RMS，不减均值 | 初始化后一次 |

EMA默认beta=0.9，scale floor=0.001。第4项是推荐候选；RMS为独立消融，不与Z-score混为同一算法。

本地CPU验收：

```text
python weaver_jit_gae_v1/validate_jit_candidate.py
```

四进程CPU Gloo验证：

```text
python weaver_jit_gae_v1/validate_jit_gloo_cpu.py
```

本机Windows的Gloo初始化返回unsupported device；已在独立Linux CPU目录完成四rank验证，无CUDA初始化。记录见 [validation/jit_gloo_remote_cpu.json](validation/jit_gloo_remote_cpu.json)。完整本地验收结果见 [validation/jit_cpu_validation.json](validation/jit_cpu_validation.json)。这些测试不等于真实Ray、OpenVLA、ZeRO2或NCCL/GPU验收。

未来经独立GPU验收后可使用候选启动器；本次没有提交GPU训练：

```bash
export ACCERL_EXPERIMENT_ROOT=/absolute/new/isolated/experiment
bash run_jit_gae.sh 03_fused_ema
```

启动器默认沿用H2和03_adaptive，要求新输出目录。正式受监护实验也可通过`runner.py --world-backend-config`显式选择对应`examples/jit_*.json`，不能依赖未传递的环境变量。未启用JIT时，原入口默认保持旧逻辑。

**性能收益尚未证实。** 固定原rank分片可能重复计算同一片段的后续状态，新增遥测会同时记录全局唯一后缀和各rank实际依赖数量。比较应排除初始化，以完整5:1周期、有效新样本入池、年龄/复用和固定条件成功率为准，不能只比较一次前向或GPU busy。

保存仍是weights-only：新run会重置optimizer、Replay及EMA。EMA模块的state_dict仅提供可测试的状态接口，未接入完整训练恢复协议。参考仓库的文件SHA已保存在 [reference_snapshot.json](reference_snapshot.json)，未获得可验证的上游commit号。

父版本说明保留在 [docs/parent_execution_README.md](docs/parent_execution_README.md)，其中部署与验收历史属于父版本。
