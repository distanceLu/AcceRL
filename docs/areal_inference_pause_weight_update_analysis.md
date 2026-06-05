# AReaL 推理暂停与模型权重更新机制分析

本文分析 AReaL 如何在在线训练中暂停推理、更新 rollout 模型权重，再恢复生成，并对比当前 `AcceRL/accerl_vllm/vllm_local_infer.py` 的本地 demo 实现。

分析基于以下代码路径：

- `AReaL/areal/engine/vllm_remote.py`
- `AReaL/areal/infra/remote_inf_engine.py`
- `AReaL/areal/engine/vllm_ext/areal_vllm_server.py`
- `AReaL/areal/engine/vllm_ext/vllm_worker_extension.py`
- `AReaL/areal/engine/fsdp_engine.py`
- `AReaL/areal/trainer/rl_trainer.py`
- `AReaL/areal/experimental/inference_service/inf_bridge.py`
- `AReaL/areal/experimental/inference_service/vllm/bridge.py`

## 总体结论

AReaL 的设计目标不是“在单个请求中途原地换权重并继续使用原 KV cache”，而是：

1. 训练侧完成一次 actor update。
2. 暂停 rollout 任务提交。
3. 通知所有推理 server 暂停 generation，并 abort 正在运行或等待的请求。
4. 通过 disk 或 XCCL/NCCL/HCCL 将训练后的权重同步到推理 worker。
5. 恢复推理 server 的 generation。
6. 更新全局 version，让后续生成 token 能记录来自哪个权重版本。

因此，AReaL 的关键语义是 **weight update boundary**：在权重更新边界上，正在推理的请求可以被中断；上层会通过重新提交、重新拼接 prompt 或重新采样的方式继续，而不是依赖旧 KV cache 与新权重混用。

你当前的 `vllm_local_infer.py` 是一个单进程/本地多进程 vLLM demo：它在生成达到 token 阈值后主动停止消费 stream，调用 `collective_rpc("reload_weights")` 从磁盘重载权重，再用“原 prompt + 已生成文本”作为新 prompt 重新发起请求。这个思路和 AReaL 的 abort/resubmit 方向一致，但实现层级、并发控制、权重传输方式和版本管理都更简化。

## AReaL 的主要对象

### 训练侧

典型路径在 `areal/trainer/rl_trainer.py`：

```python
self.rollout.pause()
new_version = global_step + 1
versioned_meta = self.weight_update_meta.with_version(new_version)
self.actor.update_weights(versioned_meta)
self.actor.set_version(new_version)
self.rollout.set_version(new_version)
```

这里有两个重要动作：

- `rollout.pause()`：暂停 rollout workflow 的新任务提交，避免训练期间继续产生过多旧版本样本。
- `actor.update_weights(...)`：训练 actor 把新权重同步给 rollout inference engine。

### Rollout 推理客户端

vLLM 路径是 `RemotevLLMEngine`，它只是外壳，实际委托给 `RemoteInfEngine`：

```python
self._engine = RemoteInfEngine(config, VLLMBackend())
```

`VLLMBackend` 负责把 AReaL 的统一接口翻译成 vLLM server 的 HTTP endpoint：

- 生成：`/v1/completions` 或 `/v1/chat/completions`
- 暂停：`/areal_pause_generation`
- 恢复：`/areal_continue_generation`
- 磁盘权重更新：`/areal_update_weights`
- XCCL 权重更新：`/areal_set_update_weight_meta` + `/areal_update_weights_xccl`
- 初始化权重同步通信组：`/areal_init_weights_update_group`

### vLLM server 扩展

AReaL 没有直接用原生 vLLM OpenAI server，而是通过 `areal_vllm_server.py` monkey patch vLLM 的 `build_app`，替换 `/v1/completions` 路由并加入自定义 endpoint。

核心全局状态是：

```python
_generation_run_event = asyncio.Event()
_generation_run_event.set()
```

被替换后的 `/v1/completions` 在真正调用 vLLM 原始 completion handler 前会先等这个 event：

```python
await _wait_if_paused()
response = await original_create_completion(request, raw_request)
```

也就是说，pause 期间新请求不会进入 vLLM 生成逻辑，而会阻塞在 HTTP server 包装层。

## AReaL 的 pause/resume 语义

### 客户端侧暂停

`RemoteInfEngine.pause_generation()` 做两件事：

1. 对所有 server 广播 backend 的 pause request。
2. 等待 `pause_grace_period`，给远端 server 时间调度并真正 abort 请求。

```python
pause_req = self.backend.get_pause_request()
self._run_request_on_all_servers(pause_req)
time.sleep(self.config.pause_grace_period)
```

`VLLMBackend.get_pause_request()` 对应：

```python
HttpRequest(endpoint="/areal_pause_generation", payload={})
```

### vLLM server 侧暂停

`/areal_pause_generation` 做三件事：

1. 清除 `_generation_run_event`，阻止新请求进入 completion。
2. 调用 `llm.pause_generation(wait_for_inflight_requests=False, clear_cache=True)`。
3. 调用 `llm.reset_mm_cache()`。

这意味着它不是“等待当前请求自然结束”，而是主动中断当前正在执行或排队的请求，并清理缓存。`wait_for_inflight_requests=False` 是关键：AReaL 更关心尽快建立权重更新边界，而不是让旧权重请求跑完。

### vLLM server 侧恢复

`/areal_continue_generation` 做两件事：

1. 调用 `llm.resume_generation()`。
2. 设置 `_generation_run_event.set()`，让新请求重新进入 completion。

### 和 `rollout.pause()` 的区别

AReaL 有两层暂停：

| 层级 | 方法 | 作用 |
| --- | --- | --- |
| workflow 调度层 | `rollout.pause()` | 暂停本地 workflow executor 提交新 rollout 任务，已有任务可能还在跑 |
| inference server 层 | `pause_generation()` | 广播到 vLLM/SGLang server，abort 正在生成的请求并清 cache |

这两层解决的是不同问题。前者控制“不要再生产新任务”，后者控制“正在推理的服务进入可更新权重的状态”。

## AReaL 的权重更新方式

AReaL 支持两类更新：从磁盘加载、通过分布式通信直接广播。

### 方式一：从磁盘更新

训练侧流程在 `FSDPEngine._update_weights_from_disk`：

1. rank 0 调用 `rollout_engine.pause_generation()`。
2. rank 0 异步向 rollout engine 发起 `update_weights_from_disk(meta)`。
3. 训练侧保存 HF checkpoint 到 `meta.path`。
4. 通过 name resolve 通知 checkpoint 已保存完成。
5. rollout server 收到通知后请求 `/areal_update_weights`。
6. vLLM server pause generation，调用 worker extension 加载权重。
7. rank 0 等待 future 完成后调用 `rollout_engine.continue_generation()`。

server 侧 `/areal_update_weights` 内部还会再次执行：

```python
await llm.pause_generation(wait_for_inflight_requests=False, clear_cache=True)
await llm.reset_mm_cache()
ret_list = await llm.collective_rpc("areal_update_weights", args=(model_path,))
await llm.resume_generation()
```

这个“双重 pause”有点冗余，但有工程意义：即使外层调用者没有先 pause，更新 endpoint 本身仍然保证更新前后暂停/恢复。

worker extension 中的实际加载逻辑是：

```python
self.model_runner.model_config.model = model_path
model_loader = get_model_loader(self.model_runner.vllm_config.load_config)
model_loader.load_weights(self.model_runner.model, model_config=self.model_runner.model_config)
self.sync()
```

它是在 vLLM worker 内部原地加载模型权重，不重启 server。

### 方式二：通过 XCCL/NCCL/HCCL 更新

这是 AReaL 在线训练中更关键的路径。训练 engine 和 rollout server 先建立一个额外的 weight update process group。

初始化：

1. 训练 rank 0 创建 `init_method`、group name、world size。
2. 调用 rollout engine 的 `init_weights_update_group(meta)`。
3. rollout server 收到 `/areal_init_weights_update_group`。
4. vLLM worker extension 执行 `areal_init_update_weight_group`。
5. 每个 vLLM worker 用 `rank=self.rank + rank_offset` 加入通信组。

更新：

1. 训练 rank 0 调用 `rollout_engine.pause_generation()`。
2. 所有训练 rank 通过 CPU group barrier 对齐。
3. 训练侧遍历模型参数。
4. 对 FSDP/DTensor 参数调用 `full_tensor()` 得到完整 tensor。
5. cast 到推理 compute dtype，例如 bf16。
6. 按 `weight_chunked_mem_mb` 切 bucket，避免一次广播全部参数导致内存峰值过大。
7. 对每个 bucket，先向 rollout server 发送参数元信息：

   ```python
   names, dtypes, shapes, group_name
   ```

8. rollout server 收到 `/areal_set_update_weight_meta`，把元信息存到 worker extension。
9. rollout server 收到 `/areal_update_weights_xccl`，在 worker 中为每个参数分配空 tensor。
10. vLLM worker 从 group rank 0 接收 broadcast。
11. worker 调用：

    ```python
    self.model_runner.model.load_weights(weights=[(name, tensor)])
    ```

12. 训练 rank 0 对每个 tensor 执行 `dist.broadcast(tensor, src=0, group=...)`。
13. bucket 完成后等待 future。
14. 全部参数更新完成后，rank 0 调用 `rollout_engine.continue_generation()`。

这个路径和你的 demo 最大不同是：AReaL 不需要先保存 checkpoint 再读回来，训练侧参数直接通过通信组送到推理 worker，速度和内存控制都更适合在线 RL。

## 中断请求如何继续

AReaL 有两套相关逻辑。

### 旧版 RemoteInfEngine 的循环重试

`RemoteInfEngine.agenerate()` 是非 streaming 的。它向 server 发一次 `/v1/completions`，解析返回 token。如果返回 `stop_reason == "abort"`，它不会直接失败，而是：

1. 把已经返回的 `output_tokens` 累积起来。
2. 把这些 token 拼到 `req.input_ids` 后面。
3. 减少 `max_new_tokens`。
4. 等待 workflow executor 不再 paused。
5. 再发下一次请求。

伪代码如下：

```python
while not finished and len(accumulated_output_tokens) < ori_max_new_tokens:
    while self.workflow_executor.is_paused():
        await asyncio.sleep(0.5)

    result = await request_server(req)
    accumulated_output_tokens.extend(result.output_tokens)
    req.input_ids += result.output_tokens
    req.gconfig.max_new_tokens -= len(result.output_tokens)
```

这和你当前 demo 的思路非常接近：**中断后不恢复旧 KV cache，而是把已生成 token 追加到 prompt，重新 prefill。**

### 新版 experimental InfBridge 的自动 resubmit

`experimental/inference_service/inf_bridge.py` 把这个逻辑做得更明确：

- `PauseState` 是一个异步安全的 paused flag。
- `pause()` 先设置 paused，再调用 backend pause。
- `resume()` 先调用 backend resume，再清 paused。
- `agenerate()` 在循环中检查 pause state。
- 收到 `abort` 后自动 patch request，追加已生成 tokens，缩短剩余 token budget，然后 resubmit。

关键接口由 backend 实现。vLLM 版本在 `experimental/inference_service/vllm/bridge.py`：

```python
http_req.payload["max_tokens"] = remaining_tokens
http_req.payload["prompt"] = list(req.input_ids) + accumulated_tokens
```

这比旧版 `RemoteInfEngine` 更模块化：abort/resubmit 是 backend-agnostic 的，vLLM 和 SGLang 只负责 payload 格式差异。

## 版本管理

AReaL 会记录 weight version：

- 训练 step 后 `new_version = global_step + 1`。
- actor、critic、rollout、eval_rollout 都设置 version。
- 推理返回的每个 output token 都附带 `output_versions`。

`RemoteInfEngine.agenerate()` 中：

```python
accumulated_versions.extend([self.get_version()] * len(gen_result.output_tokens))
```

这对异步 RL 很重要：一个 trajectory 可能跨过权重更新边界，AReaL 可以知道每个 token 是哪个 policy version 生成的，从而控制 off-policy/staleness。

你的当前 demo 没有 version 字段，只保存了 pre-update 与 post-update 两段文本/token ids。它能展示行为，但还不能支持 RL 训练里的 staleness 统计、loss mask 版本标注、或者跨版本样本过滤。

## 和当前 `vllm_local_infer.py` 的差异

### 架构层级

当前实现：

- 一个 Python 脚本直接创建 `AsyncLLMEngine`。
- 通过 `engine.generate(...)` stream 本地请求。
- 通过 `engine.collective_rpc("reload_weights", ...)` 直接调用 worker。
- 没有 HTTP server、controller、workflow executor。

AReaL：

- 训练进程、rollout engine、vLLM server、vLLM worker 分层。
- rollout engine 通过 HTTP 控制 server。
- vLLM server 通过 `collective_rpc` 控制 worker。
- 训练侧和推理侧可以跨机器、跨进程、跨调度器。

### 暂停触发点

当前实现：

- 某个请求生成 token 数达到 `PAUSE_TOKEN_THRESHOLD` 后设置本地 `asyncio.Event`。
- 其他本地 stream task 看到 event 后返回。
- 没有调用 `engine.pause_generation()` 来主动 abort 所有 vLLM 内部请求。

AReaL：

- 训练 step 完成后统一触发暂停。
- 先暂停 rollout workflow 提交，再广播 server pause。
- server 内部调用 vLLM `pause_generation(wait_for_inflight_requests=False, clear_cache=True)`，主动中断所有请求。

### 正在生成的请求处理

当前实现：

- 你的 task 从 async generator 返回，但没有显式 abort request id。
- 未完成请求的状态由应用层 `GenerationState` 保存。
- 后续重新发起新 request。

AReaL：

- server pause 会让 vLLM abort 运行/等待请求并清 cache。
- HTTP response 可能带 `finish_reason == "abort"`。
- `RemoteInfEngine` 或 `InfBridge` 自动累积已返回 token，然后 resubmit。
- 如果 abort 前没有 token，结果为空，继续等 resume 后重试。

### KV cache 语义

当前实现：

- 明确不恢复旧 KV cache。
- restart prompt = original prompt + generated text。
- post-update continuation 需要重新 prefill。

AReaL：

- 同样不跨权重更新复用旧 KV cache。
- `clear_cache=True` 明确清理缓存。
- 中断续跑靠 token 拼接和重新请求。

这点上你的 demo 和 AReaL 的正确性方向是一致的。

### 权重来源

当前实现：

- 从同一个 `MODEL_NAME` 路径 reload。
- 如果磁盘权重没有变化，更新前后实际模型可能完全一样。
- `reload_weights` 是你当前 vLLM 环境里的 worker RPC 方法。

AReaL：

- disk 模式：训练侧先保存新 HF checkpoint，server 再原地加载。
- XCCL 模式：训练侧直接广播参数 tensor，server 原地 `load_weights`。
- LoRA 模式还会按 version 注册新的 LoRA name。

### 并发与一致性

当前实现：

- 只有 4 个 prompt，应用层 `asyncio.gather`。
- 没有 staleness manager。
- 没有队列 backpressure。
- 没有多 server 地址、round-robin、request retry。

AReaL：

- `WorkflowExecutor` 后台 producer/consumer 线程持续提交 rollout。
- `StalenessManager` 限制并发和 off-policy 程度。
- `RemoteInfEngine` 记录 `rid -> server_addr`，同一 rid 固定路由以利于 KV cache 复用。
- HTTP 请求有 retry、timeout、健康检查。
- pause 会同时影响 workflow 层和 server 层。

### 版本与训练数据

当前实现：

- 只有 `generated_text`、`generated_token_ids`、`post_update_text`。
- 没有 token-level version。

AReaL：

- 每个输出 token 都有 `output_versions`。
- trajectory dump 会记录 `head_version` 和 `tail_version`。
- staleness/off-policy 控制依赖这些版本。

## 如果把当前 demo 向 AReaL 靠拢，可以优先补什么

### 1. 显式 abort / pause vLLM generation

当前 `stream_until_pause` 在达到阈值后只是返回。更接近 AReaL 的方式是：

- 在触发更新时调用 `engine.pause_generation(...)`。
- 确认是否需要 `clear_cache=True`。
- 对当前 request id 做显式 abort 或依赖 engine pause abort。

这样能避免内部还有未消费/未清理的请求状态。

### 2. 把 resubmit 逻辑抽成通用循环

当前 restart 只发生一次。AReaL 的 `InfBridge` 是一个循环：

- wait while paused
- patch prompt
- generate
- accumulate tokens
- abort 则继续
- stop/length 则结束

这能处理多次权重更新、多次 abort，以及 abort 时返回 0 token 的情况。

### 3. 改用 token ids 拼接，而不是 text 拼接

当前：

```python
restart_prompt = original_prompt + generated_text
```

这在 tokenizer 边界上可能有细微差异。例如空格、特殊 token、chat template、中文/英文子词边界，都可能导致重新 tokenize 后的 token 序列不完全等价。

AReaL 用的是 token ids：

```python
req.input_ids += gen_result.output_tokens
```

如果你的目标是研究正确性，建议改为 token-id 级续跑，而不是字符串级续跑。

### 4. 加版本字段

可以在 `GenerationState` 中加入：

```python
pre_update_version: int
post_update_version: int
output_versions: list[int]
```

这样你能判断某条生成是否跨版本，也更接近 RL rollout 数据结构。

### 5. 区分“暂停提交”和“暂停 engine”

如果后续你会持续投喂很多 prompts，建议分两层：

- 应用层 submitter paused：不再创建新 `engine.generate` task。
- engine 层 generation paused：正在执行的请求 abort，cache 清理，准备更新权重。

这能避免更新时仍然有新任务进入 engine。

## 简短对照表

| 维度 | 当前 `vllm_local_infer.py` | AReaL |
| --- | --- | --- |
| 部署形态 | 本地脚本直接持有 `AsyncLLMEngine` | 训练进程 + rollout engine + HTTP inference server + worker |
| 暂停入口 | 本地 `asyncio.Event` | `rollout.pause()` + `/areal_pause_generation` |
| 是否主动 abort vLLM 请求 | 当前代码没有显式调用 | 是，`pause_generation(wait_for_inflight_requests=False, clear_cache=True)` |
| 权重更新 | `collective_rpc("reload_weights")` 从磁盘 reload | disk reload 或 XCCL/NCCL/HCCL tensor broadcast |
| 续跑方式 | 原 prompt + 已生成文本，重新请求 | 原 input_ids + 已生成 token ids，自动 resubmit |
| KV cache | 不复用 | 不复用，且 `clear_cache=True` |
| 多次 abort | 未封装成循环 | 支持循环 resubmit |
| token 版本 | 无 | `output_versions` 记录每个 token 的权重版本 |
| 并发控制 | 简单 `asyncio.gather` | workflow executor + queue + staleness manager |
| 多 server | 无 | server 列表、round-robin、rid 固定路由 |

## 最关键的一句话

AReaL 的“暂停推理更新模型”不是暂停一个 Python 生成器然后原地换权重，而是在系统层建立一个清晰的权重更新边界：停止新 rollout、abort 旧 generation、清 KV cache、同步权重、恢复服务；如果请求被中断，则用已经返回的 token 重新构造输入并再次生成。你当前实现已经抓住了“不要跨权重复用 KV cache，而是重新 prefill”的核心，但还缺少 AReaL 在分布式、并发、版本和多次中断恢复上的工程外壳。
