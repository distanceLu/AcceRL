# 当前 critic 的依赖裁剪

共享 Policy 推理池刷新 Value 时，先按原逻辑生成完整片段的副本分配、观察顺序和 micro 边界，再跳过与本次抽样 GAE 无关的整块 micro。保留的 micro 使用原来的成员、顺序、尺寸和完整模型前向，不把剩余状态重新拼批。

对片段中最早被抽到的 chunk `j`，GAE 仍使用 `j` 到片段末尾的完整后缀。非 terminal 末端保留实际 next observation 的当前 Value；terminal 末值只填入被 mask 掉的零占位。存储的 K 组动作/奖励和 K+1 个状态保持完整。被保留 micro 的无关伙伴仍会计算，目的是保留原 batch 形状和 padding。

原抽样 ID、顺序、来源配额、rank 分片、旧奖励、行为 logits、动作 mask、年龄和成功消费限制保持原路径。后缀使用原 float64 GAE 递推，后续使用原 batch 全局优势归一化。当前橙线仍采用共享 critic 刷新，不引入 JIT GAE、EMA、Value-only、缓存或并行策略变更。

## 启用与回退

通用引擎缺省 `ACCERL_CRITIC_DEPENDENCY_PRUNING=0`。维护配方的 `weaver_run/recipe/env.json` 显式设置为 `"1"`，且只在 cooperative cost dispatcher 启用时使用裁剪；其它路径沿用完整刷新。开关在 Replay actor 创建时读取。

要回退，修改该 JSON 为 `"0"`，测试、提交并推送，然后仍从根目录 `./run.sh` 启动新实验。启动器会清理继承的 `ACCERL_*` 并加载配方，因此只在 shell 里 export 不能覆盖这个配方。运行中的旧进程不会在线切换。

## 调度、校验与遥测

实际请求携带显式 `batch_sizes`，dispatcher 按原边界执行，登记和结算的额度只包含保留的观察。原版本锁、发布等待、已提交请求的安全完成及失败清理仍生效。返回的当前版本、位置覆盖、长度和有限值必须匹配；缺失依赖时释放样本预留并报错，不回退到生成时的 Value。

`fresh_critic_refresh_plan` 的策略名为 `original_micro_pairs_dependency_pruned`。其中：

- `original_observations_by_replica` / `original_batches_by_replica`：原完整计划。
- `observations_by_replica` / `executed_batches_by_replica`：本次实际请求的观察和 micro 数。
- `required_observations_by_replica`：GAE 必需的观察数，不包含仅为保持配对而计算的伙伴。
- `stored_k_plus_one_preserved` 与 `required_dependencies_covered`：存储和依赖合同。
- `all_k_plus_one_observations_preserved`：是否实际评估了所有观察。裁剪生效时可以为 false，不能用此旧字段替代存储完整性检查。

## 验证与限制

历史 P491–590 的固定样本回放预计 micro 从 81,395 降至 65,031（减少约20.1%），这是工作量估计，不是实际 GPU 或端到端加速。CPU 测试比较同一组当前标量 Value 下的原 GAE、targets、样本顺序和 mask，并执行实际 Replay / actor 方法的 CPU fixture、取消/失败、序列化与回退检查。复现命令：

```bash
cd weaver_run/engine
WEAVER_TEST_UPSTREAM="$(cd ../vendor/weaver && pwd)" python -B validate_cost_scheduler_candidate.py
```

同一保留 micro 的输入形状虽然保持，OpenVLA / Ray / CUDA 的实际数值与梯度等价性尚未验证。异步时序变化也可能改变后续采集和学习轨迹。被跳过的无关预测不再执行原有限值检查，因此不能声称失败行为完全相同。本次提交不启动训练。
