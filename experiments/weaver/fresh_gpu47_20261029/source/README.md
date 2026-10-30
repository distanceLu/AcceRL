# WEAVER FRESH CPU 候选

本目录是从 optimized v2 本地源码复制出的独立候选，实现学习信号与数据时效改动。生产启动使用新的冻结副本、独立实验目录和新 optimizer/Replay；原 optimized v2 源码保持冻结。本目录的 CPU 验收记录不代表实际 GPU 启动或学习效果通过。

## 四项独立开关

配置位于 `backend_options.fresh`，严格定义见 `fresh_config.py`。所有开关默认关闭，错误字段和错误类型在模型加载前拒绝。

| 工作 | 开关 | 实际路径 |
|---|---|---|
| 折扣成功结果回归＋信号记录 | `dense_reward` / `signal_telemetry` | 独立派生 targets/mask，native RM masked half-MSE；生成信号按 sample_id 关联实际消费，片段 Replay 另记刷新和成功提交事件 |
| 片段 Replay、当前 critic GAE、年龄/次数限制 | `refresh_gae` / `replay_age_limits` | K+1 observation、原始 reward、原始行为 token/logits；成功 rank shard commit 才累计 use |
| 成败位置分桶与早期上下文 | `seed_balancing` / `early_seed_context` | 真 outcome × 三个位置桶；有界 reset-left-repeat synthetic history |
| 真实 chunk 进入 Policy Replay | `real_policy_replay` | 保存真实行为记录；未执行尾部 token loss mask；固定来源配额 |

默认限制是 Policy age≤16、World/RM age≤3、每 sample 最多4次实际消费；phase4 的真实来源配额25%。这些是候选超参数，不是实测最优值。不足 global576 或任一来源配额时等待新数据，不用旧数据或另一来源替代。reservation、实际 micro 曝光、成功 update 后 commit 各有独立语义。

## 配置与消融顺序

| profile（`examples/fresh_*.json`） | 作用 |
|---|---|
| `00_baseline` | 全部关闭，原 optimized v2 语义 |
| `01_dense_signal` | 仅稠密结果监督与信号记录 |
| `02_replay` | 再开启 GAE 重算、短龄和use cap |
| `03_seeds` | 再开启成败/位置均衡及早期上下文 |
| `04_real` | 再开启25%真实chunk配额 |
| `ablate_replay_only` | 原binary奖励，单独Replay改变 |
| `ablate_seeds_only` | 原binary奖励，单独seed改变 |
| `ablate_real_only` | 原binary奖励，单独真实来源改变 |

`run_fresh.sh <profile>` 显式选择配置并调用现有 launcher，要求新的 `ACCERL_EXPERIMENT_ROOT`。原生产 master/source 路径列入保护范围。该入口是待验收候选；本工作未调用它。

受控对比应从相同的原始 Policy 和已评估 native3200 World 初始权重、新 optimizer/Replay 开始。保留 Real1/Imag4、4卡、Policy global576、World global1024、P:W=5:1、NFE16 和原1M真实模拟器调用预算；不要把旧run的Replay/optimizer接到新目标上。

## 奖励目标的定义

原始 `EpisodeRecord.success` 仍为真实逐帧布尔 `info.is_success`。新字段 `reward_targets` 与 `reward_target_mask` 独立派生：

- 完整成功：`target[t] = gamma_action ** (T_success - t)`，末成功帧为1。
- 观察到完整 task-horizon 或环境终止失败：target为0，仍是合法回归监督。
- 预算截断、未完成或未观察到完整结果：mask为0，不伪标失败；World本体仍可训练。

`gamma_action = 0.99 ** (1/8)`，单位是一次真实action，与原8-action chunk折扣0.99对齐；不是直接把Policy的chunk gamma当frame gamma。新目标使用无类别加权的masked half-MSE；取消binary目标下数百倍的正类权重。归一化按整个cohort实际有效帧数，保持DDP/GAS与原生两branch归约正确；全censored时RM梯度/Adam/decay跳过，全失败时RM正常更新。checkpoint使用独立的 `reward_target_counts`，不把soft targets伪装成binary正类计数。

输出仍未校准，不是成功概率。想象仍使用明示的 `Phi_next-Phi_previous` 经验奖励；本候选不声称标准potential-based shaping的策略不变性。

## GAE、行为分布和真实来源

完整segment保留K+1 observation、K个reward和行为token/logits、原始value、Policy/World/RM版本、Phi、termination/truncation及实际执行mask。取样只在成功发布后的训练边界发生；现有4个Policy infer owner执行有界critic-only forward，保留行为logits作为ratio分母。

只重算当前critic/GAE，**不重标旧RM奖励，不重新生成旧World状态**。事件明确记录 `rewards_refreshed=false`。年龄上限用于限制剩余陈旧性，不宣称数据重新变为on-policy。

真实chunk的reward来源为 `environment_success`，想象chunk为生成时native score delta。二者是明确区分来源的混合目标实验；不能称统一奖励尺度或额外独立真实数据。真实成功/环境终止及完整task_horizon失败在学习目标上不bootstrap；仅提前real_env_budget等人为censoring保留末状态bootstrap。原环境标志与learning terminal分别记录。未执行的chunk尾部不进入surrogate、KL和entropy。

## 早期上下文与来源

早期anchor使用真实reset帧/状态左侧重复，padding/action mask/dt均显式标记：padding valid=false、无动作、dt=0；真实reset valid=true但无incoming action。canonical EpisodeRecord不改。cache只使用真实非负source frame index，重复reset对应同一frame0身份。

上游attention没有接收padding attention mask：重复token仍进入encoder和history/memory。这是新的、显式synthetic reset-repeat输入分布，CPU可运行不等于真实模型已接受该分布或效果已验证。训练窗口继续拒绝padding。

按真实完整episode的success/failure × early/middle/late六桶调度；缺桶跳过、不制造数据，不基于未校准score判成败。random消融使用私有RNG。所有帧库、pending、在途和generation credits额度保留。

## 信号与验收

`fresh_policy_signal` 记录reward、advantage、value_target、Phi、来源、seed anchor及版本。阶段1沿用原Replay，生成信号通过同一 sample_id 与实际 micro 的 `replay_use` 关联；预约不算消费，也不声称此阶段刷新过critic。启用片段Replay后另有 `refreshed` 和成功update提交后的 `consumed` 信号事件；是否真正刷新critic仍以 `critic_refreshed` 为准。记录只允许有界有限primitive，不写图像或模型tensor。seed各桶有pending/admit/complete/cancel/drop计数。World sample统计已从reserve→coordinator→bridge贯通，原零占位值不用于新鲜度结论。

`validation/cpu_validation.json` 保存实际CPU验证记录：178项通过，failure/error/skip均为0，CUDA未初始化；此前173项的时间戳记录保留。另有8份profile与7个单开关准入通过。测试覆盖纯CPU合同、真实上游tiny native forward、实际类方法的CPU提取执行及checkpoint CPU序列化；不冒充生产模型、真实Ray/DDP/CUDA或LIBERO效果测试。

2026-10-29实停旧run时发现，SQLite TEXT affinity将发布版本返回为字符串，原最终验收器因此拒绝实际一致的四rank发布。新版在读取边界严格转换规范非负十进制版本；四rank清单、hash、ACK和release的严格验收保持。新增5项真实SQLite/实际collect方法回归，旧run的失败历史保留，另存恢复审计结果。

上线前还需独立GPU准入：完整模型与新目标数值/显存、critic刷新耗时与版本一致、首轮global576/1024消费、有限性/ACK/release、真实与想象来源配额、背压与退出。随后按逐阶段profile测固定预算/初态/seed下的实际收益，不同时打开四项后推断单项因果。留出真实帧的RM AUROC/校准、各seed位置 |A| 需真实新运行的记录，当前不编造结果。

## 已有方法与诊断边界

成功前0/成功帧1在“当前是否成功”的原目标下定义一致；dense改变的是监督目标。现有诊断尚未检出相对初始策略的净提升，不等于证明唯一因果根因。Brier反推的RMS不是逐帧平均分或留出AUROC。

- [MERL Appendix A.2](https://arxiv.org/html/2610.16517v1#A1.SS2) 已有类似discounted success-to-go目标。
- [AcceRL §5/Appendix C](https://arxiv.org/html/2603.18464v3#S5) 已有current-value/GAE与行为概率校正机制。
- [Rollback](https://arxiv.org/html/2610.36068v1)、[WoVR+](https://arxiv.org/html/2610.17922v1) 是分支/阶段采样的相关工作，本实现不声称复现它们。

FRESH在这里表示本项目的候选实现与消融组织，不宣称上述各机制的新颖性。
