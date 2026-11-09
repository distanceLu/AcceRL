# 橙线配方：训练采集150、评估220

用户于2026-11-09选择橙线1793355532，只修改真实采集上限，从同一初始权重新开实验。开发仍在同一Git仓库和分支；历史源文件仅用于复核基线，运行以git_commit标识。

基线为2026-10-30 stall-fixed版：原信用额度死锁修复、共享critic刷新、原batch归一化、chunk Replay、03_adaptive单训练gang。保留25%/75%来源配额、age16/3、usecap4、Policy576、World1024、5:1及原10个滚动Eval worker。

`weaver_run/recipe/pipeline_config.json` 新增 `real_collection_action_limit: 150`。真实任务初态池仍声明220动作；分配时仅收紧每次采集的动作预算，若累计1M预算更少则以剩余额度为准。10个reset step照常计费。每个8动作chunk的最后一块最多执行剩余6动作，未执行动作不进入Policy loss。

- 150动作内成功：记录成功，正常生成dense目标。
- 150动作未成功：预算截断，结果未知；Policy末端bootstrap，RM整段mask0，World保留已观察前缀。
- 220动作正式评估仍未成功：任务失败。评估不读取采集150预算。

原站入口在训练保存、CPU权重审计、所有训练进程退出后执行四独立Policy/环境worker的固定200回合评估，保留同50初态、配对seeds、greedy/stochastic和220动作上限。不增加训练时GPU模型。

不承诺150步一定加速或提高成功率。它会放弃150步之后的恢复机会，减少失败前缀以后的采样，也改变可观测RM监督数量。应分别报告每秒有效真实chunk、截断率、RM有效标签、每完整更新耗时和固定220步成功率。

训练为manual，timeout=0，无明早自动停止、日期或成功率截止。等待用户明确停止，现有累计预算与故障保护保留。
