# Policy / World CUDA IPC + Ray 328 GiB 候选

源自已人工停止并审计通过的 jointp4manual_1022_104329_229b879f（P100/W20/R20）。
本目录是新候选，不能用于热改旧 source_v1。学习核心、模型、微批与GAS、推理配方保持原值。

- Ray对象池：328 GiB = 352187318272 bytes；不是整个Python任务的内存上限。
- Policy/World：同物理GPU、同UID/run_id的IPC句柄，经Ray只传纯元数据。
- Inference复制到自己的原Parameter/buffer，不与Trainer的后续更新共享活动权重。
- 保留源/目标CPU SHA256、四rank一致、版本ACK和完成后release；校验仍有D2H开销。
- Reward保留原FSDP2 + hidden-only/重计算与CPU-Ray权重发布。
- 手动模式仍按用户明确停止，完整边界保存；不开Nsight，不自动重试或CPU降级。
- GPU actor显式禁止重启/任务重试；同token重复请求只返缓存ACK，不重建IPC映射。

运行前须完成独立CPU/四卡Ray IPC验证，并核本目录manifest。真实六服务显存/吞吐未由小模型证明。
CPU helper无GPU/Profiler注入；MPS只复用GPU4–7已有server，不修改其配额或驱动。
生产入口是本目录run.sh或runner.py --mode manual --data <newmaster>/formal；需要fresh输出目录。
本次是否启动长训以实际launch.json为准，不能仅凭代码存在宣称新轮已运行。

source_v2补充CPU依赖记录：真实动作chunk的frame前缀、想象seed/transition、GAE后缀及bootstrap用共享DAG引用；样本ID连到实际microstep和optimizer attempt，成功更新再连到发布ACK及实际served version。评估推理不生成训练样本，Replay预留不算实际消费，跳过更新的曝光不算成功权重贡献。原model_versions末次快照为legacy记录，精确版本来自request_id对应的client_wait/served_versions。依赖记录限于直接数值输入，环境选择历史及RNG不在本次血缘范围。

source_v3保留失败v2现场，补回原启动日志/统计初始化并提前创建首次同步事件writer。按用户随后明确授权，只将Policy/Value warmup改为100、World/Reward改为20；峰值LR、衰减horizon和其余配方不变。唯一新输出formal_v3，人工stop也须指向formal_v3，旧formal不得覆盖或续训。
