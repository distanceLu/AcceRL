# Eval3运行验收

本次只调整在线评估的请求分布与统计窗口；不修改模型、训练batch、Value重算、动作采样实现、Replay规则、补货或优化器。当前源码直接在同一Git分支修改，通过提交回退；不复制训练源码。新进程需另行授权启动，已有进程保持原提交。

## 启动检查

- 配方：`--num-eval-workers 3 --eval-inference-replicas 1,2,3`。
- 需要禁用在线Eval时，训练CLI同时传`--num-eval-workers 0 --eval-inference-replicas ""`，显式覆盖配方的三个目标；非零worker不接受空路由。
- `online_evaluation_config`：worker IDs `[0,1,2]`，replicas `[1,2,3]`，在原站physical GPUs `[5,6,7]`，window/min_samples均300，non_eval_window1000。
- 启动记录仍有四个Policy推理、四个World推理、两组四rank训练；不将GPU4改成真实采集专卡。
- 逐episode重算主Eval窗口，与TensorBoard快照累计watermark对齐。200次固定最终评估是另一协议，不受在线300窗口影响。
- 保留四rank skip0、发布SHA/ACK/release、原144:432、576/1024、5:1、年龄16/3、usecap4及220动作Eval上限。

## 四项性能检查

同阶段比较至少四个完整5:1周期，排除P1–20冷启动。保留run、git_commit、起止时间、原始日志与样本量；以下服务耗时不是SM利用率或MFU。

| 检查 | 计算口径 | 不能替代它的指标 |
| --- | --- | --- |
| 第一张卡真实请求排队 | `policy_infer` 的 `service`，`workflow_counts.real_rollout`；按实际请求数加权queue_mean_ns，连同请求量报告 | GPU忙碌率、只统计Eval排队 |
| 四副本Value拖尾 | 用 `scheduler_job_id` 配对 `fresh_critic_refresh_requested` 与完成事件；按同一次refresh plan比较四副本完成时间、状态数及最后返回次数 | 单副本计算状态数、CUDA事件当作排队时间 |
| 完整周期吞吐 | 连续`synchronized_epoch_completed`边界，5个Policy与1个World；报告更新/小时、窗口长度、普通/联合轮 | 阶段耗时直接相加、单个快轮 |
| 有效想象供给 | 成功入池的不同chunk、其年龄/合法剩余消费次数、库存变化、缺货与首次/重复消费 | 请求数、生成帧数、短时消耗旧库存 |

## 已保存的旧配置基线

运行`weaverintegrated_1112_180139_d5ee1e47`，提交`11a3981387d0735164a375955326e4747a97fb9b`。2026-11-13 09:03:47–10:58:40北京时间，P236–265，6个完整周期，114.881分钟。工作区只读证据在`analysis/weaver_eval2_review_20261113/balance_raw.json`、`balance_review.json`及`balance_manifest.json`；实验数据不上传GitHub。

| 物理GPU | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: |
| 原Eval worker数 | 3 | 3 | 2 | 2 |
| 每轮Value状态均值 | 351.97 | 351.30 | 350.17 | 350.40 |
| Value提交至完成均值/秒 | 147.65 | 134.72 | 123.40 | 127.99 |
| 30轮中最后返回次数 | 24 | 3 | 1 | 2 |

第一张卡真实动作2452次，平均排队0.6817秒。这个窗口为历史基线，新的Eval3 GPU结果尚未取得；CPU测试只验证实现和配置。少开Eval会改变请求合批、随机数消耗及生产节奏，历史非随机对照无法证明学习等价，也不能保证24秒拖尾差全部消失。
