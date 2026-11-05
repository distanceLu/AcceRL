# JIT GAE与EMA实现分析

目标是去掉共享Policy推理池上的批量刷新阻塞，并复用所选训练样本已经需要的前向。当前critic重算本来已存在，本候选改变其执行位置与归一化时序；不把它包装成新的奖励学习算法。

## 计算及抽样边界

Replay仍按原随机chunk抽样和来源配额reserve。每条选中观测携带完整片段引用、chunk索引、当前published版本和动作mask。发送的是CPU依赖描述，不预先把生成时Value当成新target。待计算target在信号记录中为None，batch占位数组不会进入JIT loss。

每个rank保留原144条样本身份和分片。对每个涉及的片段，从最早被选chunk一直保留到片段末尾；非terminal末尾还包括bootstrap观测。跳过最早被选位置之前的前缀，以及不参与terminal bootstrap的末尾观测。处理顺序从后往前，future advantage和Value只作为detached标量保存。

一个有界bundle恰有当前训练micro数量的所选状态。先为其依赖中的非选中状态做micro<=2的no-grad前向，再一次性前向所选状态，计算GAE、loss并立即backward。下一bundle继承detached carry；不保留所有片段的梯度图。上下文不进入loss、EMA样本数或Replay消费。跨rank的共享片段可能各算一次，已明确计数。

GAE沿用原边界：

```text
delta[t] = reward[t] + gamma * (0 if terminated[t] else V[t+1]) - V[t]
A[t] = delta[t] + gamma * lambda * not(terminated[t] or truncated[t]) * A[t+1]
value_target[t] = stop_gradient(A[t] + V[t])
```

预算censor与想象horizon可以bootstrap，但不把GAE递推跨过新episode。LIBERO完整失败horizon的learning-terminal语义由父版本产生，本候选不重解释。旧Reward、Phi、行为logits不重标。

## EMA不是当前batch标准化的严格等价替换

第一次窗口先做无梯度精确统计，整个窗口使用这组固定统计；之后使用此前成功窗口的EMA。每次窗口只对原选中chunk累积count/sum/sumsq，四rank合并；由全局population variance更新：

```text
mu_new = beta * mu_old + (1-beta) * batch_mean
var_new = beta * var_old + (1-beta) * batch_var
          + beta * (1-beta) * (batch_mean-mu_old)^2
Z-score: (A-mu_frozen) / max(sqrt(var_frozen), 0.001)
RMS: A / max(sqrt(var_frozen + mu_frozen^2), 0.001)
```

beta=0.9。末项不可省略，否则均值发生漂移时会低估方差。统计使用float64，训练优势/target使用float32；和父版本float32 reduction仍可能有舍入差异。

历史EMA减少跨批尺度波动，但会滞后于奖励/critic分布变化；它没有“只有好处”的保证。应观察normalization_scale、EMA均值/方差、KL、value loss、固定条件成功率。不能因训练reward上升直接宣称机器人表现改善。

## 保留原loss权重

父版本对一个micro内实际执行的action token求均值，value loss按chunk均值，再由DeepSpeed进行GAS缩放及DP平均。反向重排改变了micro组成，直接对新micro重新除以有效token总数会改变部分真实chunk的权重。

因此每个chunk保存其原micro的token分母。新micro对各chunk分别乘该权重后求和；整个GAS窗口的政策、KL、熵梯度权重保持。value仍按等长micro平均。GIPO系数仍来自detached行为比率，GAE/EMA没有替换它。

## 通信、版本和失败

四rank仍以完整optimizer update为调度量子；无新增任意micro暂停。当前trainer参数在整个GAS内不变，开头校验版本、GAS、EMA状态及配置一致；no-grad上下文调用同一ZeRO2复制参数module，所选梯度前向调用engine。FSDP/TP/其他未审查边界被拒绝。

当前片段预处理仍需要CPU到GPU传输。GAE、carry、EMA及统计在设备上，微批无Ray调用；每micro保留一个共同的错误/有限性投票，再进入backward。窗口末合并统计和小型日志/target记录回读。原模型prepare_inputs_batch内的同步、DeepSpeed通信、父版本缓存trim未被悄悄删掉。

更新后先验证四rank的optimizer delta=1且skip=0，再提交EMA；原序列阶段barrier之后才提交Replay成功消费。收据只允许包含选中样本的实际advantage、value_target和refresh版本。失败取消仍未消费的reservation并阻止后续窗口。optimizer已执行后发生RPC或发布失败时，系统失败退出，不声称能回滚权重或实现跨进程原子恢复。

首次预遍历保护并恢复Torch RNG，拒绝RNG消耗或buffer变化；主动dropout、DropPath、训练BatchNorm及非零LoRA dropout被拒绝。实际OpenVLA依赖仍需GPU数值预检，包括autocast、padding、checkpoint重计算及batch形状的影响。

weights-only保存不恢复optimizer/Replay/EMA；新run初始化EMA，training_state和逐窗口遥测显式记录此约定。仅调用EMA.load_state_dict不能构成完整恢复。

## 成本收益的边界

令N=576；U为原共享刷新涉及片段的全部K+1观测数；D为各rank从最早选中点到末尾依赖状态数之和。理想忽略batch形状、缓存和通信，仅数显式状态前向：

| 模式 | 一次窗口显式状态前向数 |
|---|---:|
| 原共享刷新 | U + N |
| trainer独立刷新 | D + N |
| fused EMA初始化 | 2D |
| fused EMA初始化后 | D |

D包含N个所选状态，剩余D-N仍必须做上下文前向。若同一片段横跨多rank，D可能大于U+N；因此省掉重复所选前向并不保证总工作量下降。activation checkpoint的backward重计算不计入该表。

新增事件`jit_gae_optimizer_window`记录global_rank_dependency_states、global_unique_suffix_states、original_shared_full_refresh_states、selected_grad_states、no_grad前向数和EMA状态。host_call_wall_s是Python调用墙钟时间，可能含排队或隐式同步，也可能只记录GPU提交，不能当GPU kernel耗时或与并发阶段简单相加。传输payload计数明确只覆盖所选actions/logits/masks/weights及合并报告，不冒充全系统PCIe字节数。

整个训练仍可能受到新想象数据供给限制。必须同时比较新real/imag入池速度、成功消费、样本年龄、复用、缺货等待、普通/联合训练及发布、显存峰值；不放宽4次消费或16/3年龄来制造提速。

建议先用相同调度配置比较00与01定位位置变化，再01与02隔离EMA变化，最后02与03隔离前向复用。报告完整5:1周期并排除首次初始化与P1-P20冷启动。固定条件评估和多个种子用于效果判断，历史非随机单次A/B只能作为观察。

## 实现与验证边界

CPU检查包含完整后缀GAE、terminal/truncation、detached target、原micro权重、实际Replay类及训练hook、跳步/非有限值/身份失败、原JIT-off行为和父版本冻结清单。四进程Linux Gloo验证不同长度依赖的全局统计、单rank跳步和单rank错误共同失败。Windows本地Gloo初始化失败如实记录；Linux验证只使用CPU toy optimizer，未加载OpenVLA、Ray、ZeRO2或CUDA。

真实GPU验收仍须核验同一完整576样本窗口的target/梯度、至少P5/P6闭环、EMA四rank一致、没有共享刷新请求、GAS18/36正确、原发布/消费收尾和峰值显存。当前候选没有提交长期GPU训练。

参考：[AcceRL-Agent](https://github.com/distanceLu/AcceRL-Agent)，已保存agent_textworld.py、ppo_value.py和rl_data.py的SHA快照；上游commit未锁定。参考项目的token/trajectory统计口径不能直接等同本项目的chunk及动作mask权重。
