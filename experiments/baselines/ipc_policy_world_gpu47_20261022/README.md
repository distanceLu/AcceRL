# CUDA IPC / Ray 328 GiB 训练基线归档

本目录保存用户指定的“当前训练速度最快”记录对应的冻结代码、运行配置和审计证据。它是已经停止的历史基线，速度按下文的完整 5:1:1 周期口径记录。

TensorBoard 目录：

```text
1792656951_OpenVLA_IPCPolicyWorld_Ray328GiB_JointPolicy4_World4_RewardFSDP2_gpu47_fresh
```

运行 ID 为 `ipcpwmanual_1022_161523_17963ef5`，2026-10-22 16:15:23（北京时间）启动，20:52:07 收到用户停止请求，在完整同步边界完成 **Policy 132 / World 26 / Reward 26** 次成功更新并保存。三个训练组各四 rank，对账一致、skip 0，无未解决错误。该版本使用 **CTRL-World**。

## 文件与来源

- `source_v3/`：当时实际运行的 57 个冻结文件；不得为了便携性修改其中字节。
- `source_manifest_v3.json`：57 文件 SHA256。原 archive SHA256 为 `aa5dec65122dd78d0e4e27892c8741b6134c4c4f729c971c332b56cfadb85878`；archive 的打包字节与工作树文件集合是两项不同校验。
- `run_record.json`：机器可读的 TB、run ID、全部文件哈希、配方、最终状态与限制。
- `evidence/validation_v3.json`：历史 155 项 CPU 验证、语法与编译检查，以及四卡合成 IPC 验证引用。
- `evidence/final_stop_audit_v3_summary.json`、`evidence/final_training_audit_v3_corrected_bound.json`、`evidence/final_weights_finite.json`：最终停止、四 rank Replay / 动态 GAS / cohort / 权重审计。
- `speed_evidence_compact.json`、`verify_speed_evidence.py`、`speed_recomputed.json`：可离线独立重算的速度证据与结果，无 PyTorch、Ray、GPU或网络依赖。
- `host_support/`：三个外部 CPU 支持源码、各自哈希及只读运行环境观察；保留在57文件冻结目录之外。

大权重、Replay/SQLite、原始 TB event、完整请求血缘和大型原始日志不随本归档发布。审计记录中的服务器路径用于标识证据来源，不代表文件已在 Git 中。源码 README 保留其冻结时的候选描述；本文件与最终审计记录说明实际结局。

## 实际配方

仅物理 GPU 4–7，每卡三个 Trainer 加三个 Inference，共 24 模型进程。三个训练组各 world size 4，并发执行；每五轮包含五次 Policy、一次 World、一次 Reward 更新。

| 配置 | Policy 普通四轮 | Policy 每五轮联合轮 | World | Reward |
|---|---:|---:|---:|---:|
| 后端 | ZeRO2 | ZeRO2 | ZeRO2 | FSDP2 |
| micro batch / GAS | 8 / 18 | 4 / 36 | 4 / 64 | 16 / 16 |
| 全局更新 batch | 576 | 576 | 1024 | 1024 |
| warmup（各模型自身成功更新数） | 100，Value 同为100 | 同左 | 20 | 20 |

峰值学习率 Policy `1e-5`、Value `1e-4`、World/Reward `1e-4`；原余弦 horizon 保持 P/Value 30000、World 6000、Reward 30000。30000 是学习率时间尺度，不是手动模式下的自动停止预算。每 25 个完整 epoch 进行周期 weights-only 保存。

推理并发 cap P/W/R 为 `2/3/2`，推理 batch 8、超时 300 ms。Reward 保留 hidden-only 特征路径、Decoder 与 51 个 vision 块的非重入重计算；World/Policy 原 500M 通信桶保留。后台只预取 CPU/Ray 原始数据。

P/W 权重通过 CUDA IPC 元数据交接，接收端复制进自己独立的模型参数。保留完整源/目标 CPU SHA256、四 rank hash、版本 ACK 与 release，因此仍有 D2H 校验成本；不能称整个同步完全零拷贝。Reward 保持原 CPU-Ray 发布路径。Ray 对象池是 **328 GiB = 352187318272 bytes**，不是整个任务 RAM 的上限。

## 速度证据与统计口径

与 `jointp4manual_1022_104329_229b879f` 对照，取两轮相同 Policy 更新 **31–100**：每轮 14 个完整 `5P + 1W + 1R` 周期，70 次 Policy 更新。周期墙钟用前一周期末同步完成到本周期末同步完成，包含实际同步与周期之间的后处理。

| 指标 | 原 CPU-Ray 对照 | 本 IPC 配置 |
|---|---:|---:|
| 完整周期吞吐（Policy 更新/小时） | 28.134829 | **32.584051** |
| 平均周期墙钟 | 639.776406 s | **552.417502 s** |
| 平均训练阶段主机墙钟 | 528.733991 s | 515.948071 s |
| 平均同步阶段主机墙钟 | 108.192009 s | **34.527135 s** |

观测吞吐提高 **15.813928%**。每周期省下 87.358904 s，其中约 **84.3244%** 来自同步时间减少。独立复核命令只读取本目录小型证据：

```bash
python verify_snapshot.py
python verify_speed_evidence.py --output speed_recomputed.json
```

compact 文件仅约 64 KB，从原 9.5 MB CPU 只读提取中保留每轮 epoch 30–100 的实际整数纳秒时间戳、事件 ID、四 rank 成功步、联合轮标记和原始 train/sync 数值。epoch 30 是第一个周期的左端点。原提取与分析器 SHA 均保留；它足够重算上述数字，不足以重建全部发布/owner/request 分解。同步和训练数值是主机控制器墙钟跨度，不是 GPU kernel 时间或资源份额。

范围 31–100 避开已知对照轮的第 14/15 轮采集、第 25 轮 CPU 分析，以及 IPC 早期血缘观测保守排除区间。不能据此声称所有观测扰动都被消除。IPC 全部稳态 6–130 为 **32.538681 P/h**；含初始化与填池的本轮 132 次更新为 **28.568659 P/h**，不可与稳态数字混用。

这是配置组合的观察对照：P/W IPC、Ray 64→328 GiB、warmup 500→100/20、额外 lineage 遥测同时变化，在线数据与推理负载也会变化。不能把全部收益归因于 IPC，不能据吞吐证明收敛或 TimeTo85 改善，也未测本轮 SM Active/Occupancy、Nsight kernel overlap 或物理带宽。

## 最终结果与恢复语义

全局真实样本曝光 P/W/R 为 **76032 / 26624 / 26624**；P 每 rank 2844 个真实 microstep，完整动态 `8/18 ↔ 4/36` 收据均已核验。26 个 WM cohort 均有八个消费者，同 rank W/R 有序样本一致、跨 rank 不交叉。

用户 stop.flag 与 run ID/UID 匹配；学习代码记录的 `experiment_completed.stop_reason=scheduled_stop` 已与 `until_manual_stop=true`、无 deadline 和实际最终步数一起核对，不能把单独 exit 0 当作人工停止证据。三模型均保存 world4 / run ID / weights-only final，CPU 有限张量计数 P/W/R 为 **891 / 1434 / 736**。本轮 GPU/Ray/driver/监督/guard/sample 已退出，MPS 保留，57 源文件不变。

这些是 weights-only 权重，不含 optimizer / Replay / RNG 的精确恢复状态，不能按完整断点续训使用。最后滚动1000完整评估 episode 成功率为 **78.2%**、本轮最高 **78.6%**，85%未达到，TimeTo85 为空。速度基线不等于效果基线。

## 部署与复现边界

**本目录是机器绑定的历史源码归档，不能 clone 后直接启动，也不能重跑历史命令覆盖 `formal_v3`。** 冻结 `runner.py` 只接受 `safe.MASTER/formal_v3`，该目录已经存在；其拒绝复用已有输出是预期行为。此处不提供一条复制即启动的训练命令。

准备新实验时，应从归档建立新的候选副本，在副本中适配 CPU 控制绑定并重新验证、冻结，保留本归档不动：

1. 安装原 AcceRL/OpenVLA/CTRL-World/LIBERO 依赖。冻结脚本从完整仓库导入 `rl`、`prismatic`、`experiments`、`envs` 与 CTRL-World；57 文件不是一个独立模型软件包。
2. 本机原 Python 为 `/mnt/data/lcx3/envs/merged-env/bin/python`。实际模块核验为 **Python 3.11.13、torch 2.12.0+cu130 / compiled CUDA 13.0、Ray 2.55.1**；包元数据中的 torch 却写为2.2.0，不能把该元数据当实际运行版本。DeepSpeed 记录为0.19.0，但动态 GAS 还逐源码绑定安装版本方法的哈希（`installed_deepspeed_boundary_sources.json`），仅装同名包不足以保证相容。运行环境记录不是完整依赖 lock。
3. `run.sh` 中 Python、OpenVLA/Value/Reward 权重、CTRL-World/SVD/CLIP/VAE/condition-stat 路径均为原机器绝对路径，需要预先准备同语义权重并审查新副本配置。最终历史权重不随本发布提供；该轮从原始 checkpoint fresh 启动，没有从先前短测恢复。
4. `gpu47_safety.py` 和 `cpu_sampler.py` 会导入冻结目录之外的三个 CPU 支持源码：`/mnt/data/lcx1/dong/need_data/archive/{serial_split_verified_run_20261016.py,mps_target_run_20261014.py,real_fill_experiment_20261014.py}`。支持文件可在归档外单独保存并核其 manifest，不计入57文件哈希；它们含旧入口、旧日期和机器路径，**只能按已审查的 helper 导入用途使用，不可执行旧实验入口**。
5. 新部署须核实际 UID、GPU UUID、MPS server/control 的 PID 出生身份、已有客户端和空闲资源，并建立新输出、run ID、`mps.json` 与独立监督绑定。原 CPU guard/sample、首更新120分钟、后续每模型停滞60分钟、真实错误120秒与人工保存900秒保护都需要保留并验证。不要复用历史 PID 作为当前身份。

仅 GPU4–7、既有 MPS 默认/server100%、客户端不限额。该源码不应启停 MPS、设置旧配额或操作 GPU0–3/其他用户；不允许全局 `ray stop`、`pkill`、`sudo`，不修改驱动、VPN、SSH、网络。新实验必须先通过实际 admission，不能与正在运行的评估或后续队列训练重叠。

人工停止必须针对**新候选冻结后的实际 data 和 run ID**使用其 `request_stop.py`，等待完整同步边界及权重保存。历史运行的 stop、launch 和审计 JSON 是证据，不是再次执行的授权。
