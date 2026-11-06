# 固定权重的全初态配对评估

队列必须先结束训练进程和所有训练子进程，确认最终权重 CPU 审计通过，再启动本程序。程序不使用 Ray、在线推理 actor 或 IPC，不连接正在更新的策略；仅使用队列指定的一张 GPU 4–7。

```bash
CUDA_VISIBLE_DEVICES=GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c \
python frozen_policy_eval.py \
  --training-root /path/to/experiment/formal_v3 \
  --output-root /path/to/experiment/policy_eval \
  --max-seconds 7200 --initial-state-count 50 \
  --modes greedy,stochastic --seed 20261026 --repeats 1
```

默认 baseline 是 `formal_v3/training/checkpoints_initial/latest_weights.pt`，final 是 `formal_v3/training/checkpoints/latest_weights.pt`。训练启动时必须单独保留初始快照，不能让周期性快照覆盖它。可用 `--baseline-checkpoint` 和 `--final-checkpoint` 显式指定其他不可变快照，比较旧、新 checkpoint 时应选同一步数并保持原始模型配方相同。快照格式必须是 `accerl_parallel_weights_v1`，使用内嵌 manifest；缺少训练的 LoRA、lm_head、value_head、attn_pool 或必要的 proprio projector 会报错，不会回退到基座模型。

两种模式、两个 checkpoint、每个50初态、各一次重复，一共200条 episode。若明确设置 `--modes greedy`，则是100条。只比较 `libero_spatial` 的任务0，评估动作上限220步，reset稳定动作10步；这50个初态也用于训练，不能称为未见任务或留出测试集。

固定 seed 生成初态排列。每个 `(repeat, initial_state_id)` 的环境 seed 和动作 seed 独立派生，同一配对在两份 checkpoint 中完全相同。环境 reset 显式选择初态，策略的 Torch RNG 用动作 seed 设置。沿用训练代码中的图像预处理、策略构造、动作 token 解码和动作反归一化；greedy 使用 argmax，stochastic 使用 Categorical，batch固定为1。每条 episode 使用完整固定权重，不会混合多个策略版本。

输出文件：

- `plan.json`：不可变 checkpoint SHA256、保存的模型配方、配对计划和计划 SHA256。
- `initial_state_pool.json`：实际初态池的数量、形状和规范字节 SHA256。
- `episodes.jsonl`：逐条保存 checkpoint 指纹、固定策略版本、初态ID/哈希、环境与动作 seed、模式、成功、步数、回报、终止原因、动作 token 摘要和时间；完成一条即 flush/fsync。
- `progress.json`：最近完成的一条及累计进度。
- `summary.json`：每份 checkpoint/模式的结果和初态缺失，以及只按完整配对计算的成功率差、单边成功和双边成功/失败数量。
- `completed.json`：机器验收回执，含 `complete`、50初态、checkpoint SHA256、模式、实际/期望条数及错误。完整且无错误时退出码0，部分或失败为2。

重启使用同一输出目录会核验完全相同的计划、权重指纹和初态池，只跳过已有完整记录。仅允许修复写入中断的最后一个不完整 JSON 行；中间损坏、重复记录或配对初态哈希不同都会失败。首次 `started.json` 保存的截止时间不会在续跑时延长。收到 SIGTERM/SIGINT 或预算到期会保留已有逐条结果并生成部分报告。

`--validate-only` 只做 CPU 权重 SHA、manifest 和计划准入，不加载 GPU 模型，也不能替代正式模型恢复验证。CPU 测试命令是 `python -m unittest test_frozen_policy_eval_cpu -v`。一次随机动作重复只能提供50对结果，增加重复次数后才能更充分评估动作随机性。部分结果只对已完成配对汇总，必须同时查看缺失覆盖，不能拿部分总体均值直接比较。
