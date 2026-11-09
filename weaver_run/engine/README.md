# Current engine: orange 1793355532 with collection budget 150

Active configuration is in `../recipe/`. Task and evaluation horizon remain 220. The original shared-critic, chunk-replay, batch-normalization and single-training-gang implementation is active. Code is identified by its Git commit.

The text below documents the historical November 3 candidate; its optional JIT/EMA/parallel-cycle modules are retained for history and are not enabled by the current entry. See `../../docs/collection150.md` for the active experiment.

---

# WEAVER：完整周期调权与双训练组并行候选

当前GPU训练 `weaverintegrated_1103_145816_a9fe5a02` 保持原255文件源码冻结。
本目录是独立候选，尚未部署；父版本清单见 `parent_trajectory_source_manifest.json`。

已实现两类修复：

- 修复JIT在 `needs_batch` 阶段有库存却不被视为可训练的调权状态缺口。
- 以完整5:1周期实测进行有界调权试探、两周期保持、接受或回退；保留库存、新鲜度和成功消费约束。
- 两个独立四rank训练组允许同版本并行准入，每卡保留推理提交容量；保留原collective阶段协议和失败收尾。
- 实际调权与共同训练准入新增遥测，区分逻辑许可、真实CUDA重叠和端到端吞吐。

详细架构、串行根因、旧run不调权的原始证据、候选边界与消融配置：
[docs/architecture_and_changes.md](docs/architecture_and_changes.md)。

```text
python validate_cycle_parallel_candidate.py
```

未来GPU实验入口（要求独立输出目录；不提交队列）：

```bash
bash run_cycle_parallel.sh 00_fixed_serial
bash run_cycle_parallel.sh 01_feedback_serial
bash run_cycle_parallel.sh 02_parallel_fixed
bash run_cycle_parallel.sh 03_feedback_parallel
```

`00_parent`保留原宏观算法与串行配置，但本候选的可执行性修正仍生效；严格旧版对照应使用冻结父源码。

CPU结果在 `validation/cycle_parallel_cpu_validation.json`，原始调权诊断在
`validation/parent_macro_diagnosis.json`。未运行本候选的Ray/OpenVLA/ZeRO2/NCCL/GPU并行验收，
不宣称吞吐或学习效果提升。原学习实现逐文件保持，详见验证记录。
父版本文档保存在 [docs/trajectory_parent_README.md](docs/trajectory_parent_README.md)，其运行状态属于历史记录。
