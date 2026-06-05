# `vllm_fsdp.py` 代码讲解

本文解释 `AcceRL/accerl_vllm/vllm_fsdp.py` 的整体逻辑、关键代码路径和 Python/PyTorch/vLLM 相关语法。它不是生产训练脚本，而是一个最小化 demo，用来验证：

1. Ray 启动多个 FSDP 训练 worker。
2. vLLM 使用独立 GPU 运行可中断推理。
3. 训练到同步边界时暂停/abort 推理。
4. 训练侧通过 NCCL 把 FSDP 权重同步给 vLLM。
5. 推理侧把被打断的请求重新提交并继续生成。

## 整体目标和运行架构

文件顶部的 docstring 已经给出核心设定：

```text
8-GPU layout:
  Training  - 4 GPUs, PyTorch FSDP2 (fully_shard)
  Inference - 4 GPUs, vLLM AsyncLLMEngine with EP+DP
```

默认常量是：

```python
FSDP_WORLD_SIZE = 4
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 4
```

含义是：4 个 Ray actor 各占 1 张 GPU，组成训练侧 FSDP process group；vLLM 推理侧使用 4 个 data-parallel worker，tensor parallel 为 1，并启用 expert parallel。权重同步时会额外建立一个 NCCL weight-transfer group，world size 为：

```python
transfer_world_size = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE + 1
```

也就是 1 个训练 rank 0 发送方，加 4 个 vLLM 推理 worker 接收方。

注意：代码假设单机 8 GPU。虽然 `--ray-address` 可以接已有 Ray 集群，但脚本里的 IP/端口、GPU placement 和 NCCL rendezvous 都按单机 demo 写法组织，没有实现完整跨节点部署策略。

## 入口链路

主入口很短：

```python
def main() -> None:
    args = parse_args()
    validate_args(args)
    asyncio.run(run_weight_sync_demo(args))
```

执行顺序是：

1. `parse_args()` 定义 CLI 参数。
2. `validate_args()` 检查参数必须为正数或合法范围。
3. `asyncio.run(run_weight_sync_demo(args))` 进入异步主流程。

这里 `asyncio.run(...)` 会创建事件循环并运行 `run_weight_sync_demo`。因为 vLLM 的 `AsyncLLMEngine.generate()`、pause/resume、weight update 都是 async 接口，主流程必须运行在 asyncio 事件循环里。

## 辅助函数

`get_local_ip()` 和 `find_open_port()` 用来构造 NCCL rendezvous 地址。

```python
def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
```

这里的 UDP socket 并不是真的向 8.8.8.8 发送业务数据，而是借助系统路由选择本机出站 IP。失败时回退到 localhost。

`pick_dtype()` 根据 `--dtype` 选择模型加载 dtype：

- `float32`、`float16`、`bfloat16` 显式指定。
- `auto` 在 CUDA 可用时优先 bf16，否则 fp16。
- CPU 下回退 fp32。

## Dummy SFT 数据构造

脚本内置了 `DUMMY_CHAT_EXAMPLES`，每条样本包含 `user` 和 `assistant`。它不是实际 RL 数据，而是为了让训练 loop 有可前向、可反向的数据。

`EncodedExample` 是一个简单容器：

```python
class EncodedExample:
    def __init__(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        labels: List[int],
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
```

这里的 `List[int]` 是类型标注，说明字段应是一串 token id。类型标注不会在运行时自动检查，但能帮助 IDE、静态检查器和读代码的人理解数据形状。

`DummyChatDataset` 实现了 PyTorch Dataset 需要的两个魔术方法：

```python
def __len__(self) -> int:
    return len(self.examples)

def __getitem__(self, index: int) -> EncodedExample:
    return self.examples[index]
```

`__len__` 告诉 DataLoader 数据集长度，`__getitem__` 让 DataLoader 能按 index 取样本。

## Prompt、labels 和 response-only loss

`build_prompt()` 只构造用户 prompt，`build_full_text()` 构造完整对话：

```python
messages = [{"role": "user", "content": user}]
```

如果 tokenizer 有 chat template，就调用 `tokenizer.apply_chat_template(...)`；否则使用普通字符串格式：

```python
return f"User: {user}\nAssistant: "
```

`encode_chat_example()` 的关键是 labels masking：

```python
labels = list(input_ids)
prompt_len = min(len(prompt_ids), len(labels))
labels[:prompt_len] = [-100] * prompt_len
```

在 Hugging Face causal LM 训练中，`labels` 里值为 `-100` 的位置会被 loss 忽略。因此这段代码只让 assistant 回复部分参与 loss，用户 prompt 不参与 loss，这就是 response-only SFT 的最小实现。

后面的保护逻辑：

```python
if all(label == -100 for label in labels) and labels:
    labels[-1] = input_ids[-1]
```

用于避免样本被截断后所有 label 都是 `-100`，否则该样本没有任何可训练 token。

## `make_collate_fn()` 和闭包

`make_collate_fn(tokenizer)` 返回内部函数 `collate()`：

```python
def make_collate_fn(tokenizer):
    pad_token_id = tokenizer.pad_token_id

    def collate(examples: List[EncodedExample]) -> Dict[str, torch.Tensor]:
        ...

    return collate
```

这是一个闭包。内部的 `collate()` 会记住外层作用域里的 `pad_token_id`。DataLoader 每次组 batch 时调用 `collate()`，它会：

1. 找到 batch 内最大序列长度。
2. 用 `pad_token_id` pad `input_ids`。
3. 用 0 pad `attention_mask`。
4. 用 `-100` pad `labels`。
5. 返回 `Dict[str, torch.Tensor]`。

`Dict[str, torch.Tensor]` 表示字典 key 是字符串，value 是 PyTorch tensor。返回值形如：

```python
{
    "input_ids": torch.tensor(..., dtype=torch.long),
    "attention_mask": torch.tensor(..., dtype=torch.long),
    "labels": torch.tensor(..., dtype=torch.long),
}
```

## 模型加载和可训练参数选择

`build_tokenizer()` 从本地路径加载 tokenizer：

```python
AutoTokenizer.from_pretrained(
    args.model_path,
    local_files_only=True,
    trust_remote_code=args.trust_remote_code,
)
```

`local_files_only=True` 表示只读本地缓存，不联网下载。若 tokenizer 没有 pad token，就用 eos token 作为 pad token。

`build_model()` 加载 causal LM：

```python
model = AutoModelForCausalLM.from_pretrained(...)
model.to(device)
model.train()
model.config.use_cache = False
```

训练时关闭 `use_cache`，避免生成用 KV cache 和梯度计算逻辑冲突。若传了 `--gradient-checkpointing`，则开启 gradient checkpointing 来省显存。

`configure_trainable_parameters()` 支持三种训练范围：

- `full`：所有参数可训练。
- `lm_head`：只训练 `lm_head`。
- `last_layer`：训练最后一层 transformer block 和 `lm_head`。

实现方式是先设置 `requires_grad`：

```python
for param in model.parameters():
    param.requires_grad = False
```

再根据参数名把目标参数打开：

```python
if any(keyword in name for keyword in target_keywords):
    param.requires_grad = True
```

后续 optimizer 只接收 `requires_grad=True` 的参数。

## vLLM 可加载权重名转换

`iter_vllm_loadable_weights()` 是权重同步里很重要的小函数。Transformers 里的 Qwen MoE expert 权重可能是融合后的 3D tensor，例如：

- `experts.gate_up_proj`
- `experts.down_proj`

但 vLLM 的 loader 期望 checkpoint 风格的逐 expert 权重名。于是代码把 fused tensor 拆回 vLLM 可识别的名字：

```python
if name.endswith(".mlp.experts.gate_up_proj"):
    prefix = name.removesuffix(".gate_up_proj")
    gate_proj, up_proj = tensor.chunk(2, dim=1)
    for expert_idx in range(tensor.shape[0]):
        yield f"{prefix}.{expert_idx}.gate_proj.weight", gate_proj[expert_idx]
        yield f"{prefix}.{expert_idx}.up_proj.weight", up_proj[expert_idx]
```

这里 `yield` 说明函数是生成器。调用方不是一次性拿到 list，而是逐个迭代 `(name, tensor)`。在 `gather_and_broadcast_weights()` 里还用了：

```python
yield from iter_vllm_loadable_weights(name, full_param)
```

`yield from` 的意思是把另一个生成器产生的所有元素继续向外产出。

`get_vllm_weight_metadata()` 使用同一套转换逻辑提前生成 metadata：

```python
names.append(load_name)
dtype_names.append(str(load_tensor.dtype).split(".")[-1])
shapes.append(list(load_tensor.shape))
```

metadata 和真实发送的 tensor 顺序必须一致，否则 vLLM 接收侧会按错误的名字、dtype 或 shape 解释权重。

## FSDP 训练 worker

`FSDPTrainWorker` 是 Ray actor 类。主流程里这样创建 actor 类型：

```python
remote_worker = ray.remote(num_gpus=1)(FSDPTrainWorker)
```

含义是：每个 actor 分配 1 张 GPU。实例化 actor 时调用：

```python
remote_worker.remote(args, rank, args.fsdp_world_size, fsdp_master_addr, fsdp_master_port)
```

`.remote(...)` 不会在当前进程里直接执行，而是提交给 Ray worker 进程执行，并返回 object ref。需要结果时用：

```python
ray.get([w.get_rank.remote() for w in fsdp_workers])
```

### 初始化分布式训练

每个 actor 构造时设置 rendezvous 环境变量：

```python
os.environ["MASTER_ADDR"] = fsdp_master_addr
os.environ["MASTER_PORT"] = str(fsdp_master_port)
```

然后初始化 NCCL process group：

```python
dist.init_process_group(backend="nccl", rank=rank, world_size=fsdp_world_size)
```

这里的 `rank` 是训练侧 FSDP rank，范围是 `0..fsdp_world_size-1`。

Ray 已经给每个 actor 分配一张可见 GPU，所以 actor 内部使用 `cuda:0`：

```python
torch.cuda.set_device(0)
self.device = torch.device("cuda:0")
```

### FSDP2 包装

模型加载完成后，代码先保存参数名和 metadata：

```python
named_parameters = list(model.named_parameters())
self.all_param_names = [name for name, _ in named_parameters]
self.trainable_param_names = [
    name for name, param in named_parameters if param.requires_grad
]
```

然后再执行 FSDP2 sharding：

```python
for layer in model.model.layers:
    fully_shard(layer)
fully_shard(model)
```

这里顺序很重要：metadata 是根据 pre-FSDP 参数名和形状准备的；FSDP 包装后参数会被 shard，但后续仍需要按原始参数名和 vLLM loader 约定同步。

`fully_shard(layer)` 先包每层 transformer layer，`fully_shard(model)` 再包顶层模型。这是 PyTorch FSDP2 风格，用于把参数、梯度和 optimizer state 分片到多个 rank。

### DataLoader 和 DistributedSampler

每个 rank 都构造同一个 dummy dataset，但 sampler 不同：

```python
self.sampler = DistributedSampler(
    dataset,
    num_replicas=fsdp_world_size,
    rank=rank,
    shuffle=True,
    seed=args.seed,
    drop_last=False,
)
```

`DistributedSampler` 根据 rank 切分数据，让不同训练 rank 看到不同样本顺序。每个 epoch 前调用：

```python
self.sampler.set_epoch(self.train_epoch)
```

这是为了让分布式 shuffle 在每个 epoch 一致但又能变化。

### 持久训练 loop

`train_until_next_sync()` 不是从头训练一次，而是从 actor 内部保存的状态继续训练：

```python
self.train_epoch = 0
self.train_micro_step = 0
self.optimizer_step = 0
self._dataloader_iter = None
```

每次主流程调用它时，它继续跑若干 optimizer step，直到：

- 达到 `self.optimizer_step + num_optimizer_steps`。
- 或达到全局 `args.max_steps`。

核心训练步骤是：

```python
batch = move_batch_to_device(self._next_training_batch(), self.device)
outputs = self.model(**batch)
loss = outputs.loss / self.args.grad_accum_steps
loss.backward()
```

因为用了梯度累积，loss 先除以 `grad_accum_steps`。每个 micro step 都 backward，但只有满足条件时才 optimizer step：

```python
should_step = self.train_micro_step % self.args.grad_accum_steps == 0
if not should_step:
    continue
```

真正更新参数时：

```python
torch.nn.utils.clip_grad_norm_(self.trainable_parameter_list, max_norm=1.0)
self.optimizer.step()
self.optimizer.zero_grad(set_to_none=True)
self.optimizer_step += 1
```

函数结尾有：

```python
dist.barrier()
```

这是训练侧所有 FSDP rank 的同步点，确保所有 rank 都完成当前 segment 后再回到主流程。

## 权重同步的训练侧

训练侧 rank 0 负责建立 weight-transfer endpoint：

```python
def setup_transfer_endpoint(self):
    assert self.rank == 0
    self.transfer_port = find_open_port()
    self.transfer_master_address = get_local_ip()
    return self.transfer_master_address, self.transfer_port
```

`assert self.rank == 0` 明确这个方法只能 rank 0 调用。之后 rank 0 加入额外的 NCCL weight-transfer group：

```python
self.model_update_group = NCCLWeightTransferEngine.trainer_init(
    dict(
        master_address=self.transfer_master_address,
        master_port=self.transfer_port,
        world_size=transfer_world_size,
    ),
)
```

真正发送权重的是 `gather_and_broadcast_weights()`。

```python
param_names = self.param_names_by_scope[scope]
```

`scope` 只能是：

- `all`：全量参数。
- `trainable`：当前训练范围内的参数。

rank 0 会构造生成器 `_full_param_iter()`：

```python
params_by_name = dict(self.model.named_parameters())
for name in param_names:
    full_param = params_by_name[name].full_tensor().detach()
    yield from iter_vllm_loadable_weights(name, full_param)
```

`full_tensor()` 是 FSDP2/DTensor 上的 collective 操作。它会让所有 FSDP rank 共同参与 all-gather，把 shard 还原成完整 tensor。即使只有 rank 0 发送给 vLLM，其他 FSDP rank 也必须以同样顺序调用 `full_tensor()`：

```python
else:
    params_by_name = dict(self.model.named_parameters())
    for name in param_names:
        params_by_name[name].full_tensor()
```

这是最重要的死锁风险点之一：如果某个 rank 没有调用，或者参数顺序不同，collective 会卡住。

rank 0 最后调用 vLLM 提供的权重传输接口：

```python
trainer_args = NCCLTrainerSendWeightsArgs(
    group=self.model_update_group,
    packed=packed,
)
NCCLWeightTransferEngine.trainer_send_weights(
    iterator=_full_param_iter(),
    trainer_args=trainer_args,
)
```

`packed=True` 表示使用 packed 传输格式，减少大量小 tensor 广播的开销。

## vLLM AsyncLLMEngine 创建

`create_async_engine()` 手动创建 `vllm.AsyncLLMEngine`：

```python
engine_args = vllm.AsyncEngineArgs(**kwargs)
vllm_config = engine_args.create_engine_config()
executor_class = Executor.get_class(vllm_config)
return vllm.AsyncLLMEngine(...)
```

主流程中传入的关键参数是：

```python
engine = create_async_engine(
    model=local_model_path,
    enforce_eager=True,
    tensor_parallel_size=INFERENCE_TP_SIZE,
    data_parallel_size=INFERENCE_DP_SIZE,
    enable_expert_parallel=True,
    distributed_executor_backend="ray",
    data_parallel_backend="ray",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
    load_format="dummy",
    gpu_memory_utilization=0.3,
    max_num_seqs=args.vllm_max_num_seqs,
    max_num_batched_tokens=args.vllm_max_num_batched_tokens,
)
```

几个重点：

- `load_format="dummy"`：vLLM 先用 dummy weights 启动，不从磁盘加载真实权重。真实权重随后由 FSDP 初始同步送入。
- `weight_transfer_config=WeightTransferConfig(backend="nccl")`：启用 NCCL 权重传输后端。
- `distributed_executor_backend="ray"` 和 `data_parallel_backend="ray"`：推理 worker 也由 Ray 调度。
- `max_num_seqs`：vLLM scheduler 同时 active/running 的 sequence 上限。
- `max_num_batched_tokens`：一次调度允许的 batched token 总量上限。

## 可中断生成状态

`OnlineGenerationState` 是 dataclass：

```python
@dataclass
class OnlineGenerationState:
    index: int
    prompt: str
    input_ids: List[int]
    requested_max_tokens: int
    output_tokens: List[int] = field(default_factory=list)
    output_versions: List[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None = None
    completed: bool = False
    attempts: int = 0
```

`@dataclass` 会自动生成 `__init__` 等样板代码。`field(default_factory=list)` 的作用是每个实例都创建自己的新 list，避免多个请求共享同一个默认 list。

`Literal["length", "stop", "tool_calls", "abort"]` 表示 stop reason 只允许这几个语义值。后面的 `| None` 是 Python 3.10+ union 语法，表示也可以为 `None`。

两个 property 是动态计算字段：

```python
@property
def remaining_max_tokens(self) -> int:
    return max(0, self.requested_max_tokens - len(self.output_tokens))

@property
def restart_prompt_token_ids(self) -> List[int]:
    return self.input_ids + self.output_tokens
```

中断后继续生成时，不复用旧 KV cache，而是把“原始 prompt token + 已生成 token”作为新 prompt 重新 prefill。

`RepeatingInferenceStats` 也是 dataclass，用于记录完成请求数、生成 token 数、并发度和最近完成的请求状态。

## `InterruptibleGenerationRunner`

这个类把一次用户请求包装成“可被 abort、多次重试”的生成状态机。

初始化时有几个重要字段：

```python
self.version = 0
self.paused = asyncio.Event()
self.paused.clear()
self._active_attempts = 0
self._active_changed = asyncio.Condition()
```

- `version`：当前权重版本。每次 trainable 权重同步完成后加 1。
- `paused`：本地暂停标志。设置后不再启动新的 `engine.generate()` attempt。
- `_active_attempts`：当前正在跑的 vLLM generate attempt 数。
- `_active_changed`：异步条件变量，用来等待 active attempt 清零。

`asyncio.Event` 常用于“开关”语义：

```python
while self.paused.is_set():
    await asyncio.sleep(0)
```

如果 pause 已设置，新的 attempt 就让出事件循环，直到恢复。

`asyncio.Condition` 用于等待更复杂的状态变化：

```python
async with self._active_changed:
    await self._active_changed.wait_for(lambda: self._active_attempts == 0)
```

这里 `wait_for_idle()` 会等所有正在跑的 generate attempt 正常结束或被 abort 后退出。

### 单次 generate attempt

核心生成代码是：

```python
async for request_output in self.engine.generate(
    {"prompt_token_ids": state.restart_prompt_token_ids},
    sampling_params,
    request_id=request_id,
):
    final_output = request_output
    request_finished = bool(getattr(request_output, "finished", False))
```

`async for` 表示异步迭代 vLLM streaming 输出。vLLM 每产生一段输出就 yield 一个 `request_output`，代码只保存最后一个 `final_output`。

`try/finally` 确保 active attempt 计数一定会减少：

```python
await self._increment_active_attempts()
try:
    async for ...:
        ...
finally:
    await self._decrement_active_attempts()
```

如果 `pause_generation(mode="abort")` 打断请求，`engine.generate()` 可能没有正常 final output。这时：

```python
if final_output is None:
    state.stop_reason = "abort"
    continue
```

runner 会进入下一次 attempt。

如果拿到了 token：

```python
attempt_tokens = _tokens_from_output(final_output)[:remaining]
state.output_tokens.extend(attempt_tokens)
state.output_versions.extend([attempt_version] * len(attempt_tokens))
```

它把新 token 追加到总输出，并记录这些 token 是哪个权重版本生成的。若停止原因是 `stop`、`tool_calls` 或 `length`，请求完成；如果是 `abort` 或没有 finished，则继续重试。

## 持续推理队列

`run_repeating_inference()` 负责保持固定数量的用户级请求并发：

```python
worker_tasks = [
    asyncio.create_task(infer_worker(worker_id))
    for worker_id in range(infer_concurrency)
]
```

`asyncio.create_task(...)` 会把 coroutine 放到事件循环后台运行。每个 `infer_worker` 不断：

1. 调 `make_next_state()` 创建新请求状态。
2. 调 `runner.generate(state)` 执行可中断生成。
3. 调 `record_completed_state(...)` 更新统计。

`make_next_state()` 的返回类型是：

```python
async def make_next_state() -> OnlineGenerationState | None:
```

`OnlineGenerationState | None` 表示可能返回一个状态，也可能返回 `None`。当 `stop_after_current_cycle` 被设置后，它返回 `None`，worker 退出。

内部用了 `asyncio.Lock()` 保护共享计数器：

```python
async with stats_lock:
    ...
```

因为多个 inference worker 会同时更新 `next_request_index`、`next_prompt_index` 和 stats。

## `sync_weights_to_vllm()`

这是训练权重同步到 vLLM 的统一函数。步骤是：

1. 从 FSDP rank 0 获取 metadata。
2. 打印逻辑权重大小和推理侧聚合接收量。
3. 通知 vLLM 进入 weight update。
4. 并发触发所有 FSDP rank gather/broadcast。
5. 通知 vLLM 按 metadata 接收并加载权重。
6. 等训练侧广播完成。
7. 通知 vLLM 结束 weight update。

代码对应如下：

```python
names, dtype_names, shapes = ray.get(
    fsdp_workers[0].get_weight_metadata.remote(scope)
)
```

这里只向 rank 0 要 metadata，因为所有 rank 的模型结构一致。

```python
await engine.start_weight_update()
```

告诉 vLLM 接下来要更新权重。

```python
broadcast_handles = [
    worker.gather_and_broadcast_weights.remote(scope=scope, packed=packed)
    for worker in fsdp_workers
]
```

关键点是所有 FSDP worker 都要同时进入 `gather_and_broadcast_weights()`，因为里面的 `full_tensor()` 是 collective。

随后 vLLM 接收侧开始 update：

```python
await engine.update_weights(
    WeightTransferUpdateRequest(
        update_info=asdict(
            NCCLWeightTransferUpdateInfo(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                packed=packed,
            )
        )
    )
)
```

`asdict(...)` 来自 dataclasses，会把 dataclass 对象转成普通 dict，适配 vLLM 的 request 类型。

最后：

```python
ray.get(broadcast_handles)
await engine.finish_weight_update()
```

先确认训练侧发送完成，再告诉 vLLM 结束更新。

## 主流程 `run_weight_sync_demo()`

`run_weight_sync_demo()` 是整个 demo 的调度中心。

### 1. 初始化 Ray 和 FSDP workers

```python
if args.ray_address:
    ray.init(address=args.ray_address)
else:
    ray.init()
```

然后创建 `args.fsdp_world_size` 个训练 actor：

```python
fsdp_workers = [
    remote_worker.remote(
        args,
        rank,
        args.fsdp_world_size,
        fsdp_master_addr,
        fsdp_master_port,
    )
    for rank in range(args.fsdp_world_size)
]
```

`ray.get([w.get_rank.remote() for w in fsdp_workers])` 既拿 rank，也起到等待所有 actor 初始化完成的作用。

### 2. 创建 vLLM engine

vLLM engine 使用 dummy weights 创建，然后等待训练侧初始同步真实权重。这样可以验证“训练权重直接通过 NCCL 注入推理 engine”的路径，而不是从磁盘 checkpoint 加载。

### 3. 初始化 weight-transfer NCCL group

训练 rank 0 创建 endpoint：

```python
transfer_addr, transfer_port = ray.get(
    fsdp_workers[0].setup_transfer_endpoint.remote()
)
```

然后训练侧 rank 0 和 vLLM worker 同时加入 NCCL group：

```python
train_handle = fsdp_workers[0].init_weight_transfer_group.remote(
    transfer_world_size
)
await engine.init_weight_transfer_engine(...)
ray.get(train_handle)
```

vLLM worker 使用 `rank_offset=1`，意思是训练 rank 0 是 weight-transfer group rank 0，vLLM workers 从 rank 1 开始。

### 4. 初始全量同步

第一次同步必须是全量：

```python
await engine.pause_generation(mode="abort", clear_cache=True)
await sync_weights_to_vllm(..., scope="all", ...)
await engine.resume_generation()
```

因为 vLLM 是 dummy weights 启动的，必须先把所有真实权重送进去，推理才有意义。

### 5. 启动持续推理

初始同步完成后，代码创建 runner 和后台推理任务：

```python
runner = InterruptibleGenerationRunner(engine, temperature=0.0)
stop_inference = asyncio.Event()
inference_loop_task = asyncio.create_task(
    run_repeating_inference(...)
)
```

这个任务会持续保持 `--infer-concurrency` 个用户级请求在跑。

### 6. 训练 segment 和周期性同步

主循环每次让所有 FSDP worker 训练一段：

```python
train_handles = [
    worker.train_until_next_sync.remote(args.sync_every_optimizer_steps)
    for worker in fsdp_workers
]
train_future = asyncio.create_task(
    asyncio.to_thread(ray.get, train_handles)
)
```

`ray.get(...)` 是阻塞调用，所以用 `asyncio.to_thread(...)` 放到线程里，避免阻塞事件循环。

训练 segment 完成后，如果还需要同步：

```python
runner.paused.set()
await engine.pause_generation(mode="abort", clear_cache=True)
await runner.wait_for_idle()
```

这里有两层控制：

- `runner.paused.set()`：阻止本地 runner 启动新的 generate attempt。
- `engine.pause_generation(mode="abort", clear_cache=True)`：让 vLLM abort 当前请求并清缓存。
- `runner.wait_for_idle()`：等待已经进入 `engine.generate()` 的 attempt 退出。

然后只同步 trainable 参数：

```python
await sync_weights_to_vllm(..., scope="trainable", ...)
runner.version += 1
await engine.resume_generation()
runner.paused.clear()
```

后续生成的新 token 会记录新的 `runner.version`。

### 7. 停止和清理

训练结束或同步轮数达到上限后：

```python
stop_inference.set()
inference_stats = await inference_loop_task
```

这表示不再启动新推理请求，但已经进入 worker 的请求会完成当前 cycle。

`finally` 块负责清理：

```python
await shutdown_vllm_engine(engine)
ray.get([worker.close.remote() for worker in fsdp_workers])
ray.shutdown()
```

`shutdown_vllm_engine()` 用 `getattr(engine, "shutdown", None)` 做兼容：如果当前 vLLM engine 没有 shutdown 方法，就直接返回。

## CLI 参数说明

训练和数据相关参数：

- `--model-path`：本地模型路径，默认是 `MODEL_NAME`。
- `--dtype`：模型 dtype，支持 `auto`、`bfloat16`、`float16`、`float32`。
- `--train-mode`：训练范围，支持 `lm_head`、`last_layer`、`full`。
- `--max-length`：样本最大 token 长度。
- `--batch-size`：每个 FSDP rank 的 batch size。
- `--max-steps`：最多 optimizer step 数。
- `--learning-rate`、`--weight-decay`：AdamW 参数。
- `--grad-accum-steps`：梯度累积步数。
- `--dataset-repeat`：内置 dummy 数据重复次数。
- `--log-every`：rank 0 每多少 optimizer step 打印一次 loss。
- `--seed`：随机种子。
- `--trust-remote-code`：传给 Transformers loader。
- `--gradient-checkpointing`：是否开启 gradient checkpointing。

FSDP/Ray 相关参数：

- `--fsdp-world-size`：训练侧 FSDP rank 数，默认 4。
- `--fsdp-master-addr`、`--fsdp-master-port`：训练 process group rendezvous 地址；不传则自动找本机 IP 和空闲端口。
- `--ray-address`：可选 Ray 集群地址；不传则 `ray.init()` 启本地 Ray。

同步和推理相关参数：

- `--sync-every-optimizer-steps`：每训练多少 optimizer step 触发一次 trainable 权重同步。
- `--infer-max-tokens`：每个用户请求最多生成 token 数。
- `--infer-concurrency`：用户级并发请求数。
- `--vllm-max-num-seqs`：vLLM scheduler 同时 active sequence 上限。
- `--vllm-max-num-batched-tokens`：vLLM scheduler 一次调度的 token 总量上限。
- `--max-sync-rounds`：最多执行多少轮 trainable-only 同步；为 `None` 时不限制。

## 关键语法速查

### dataclass 和 default_factory

```python
@dataclass
class RepeatingInferenceStats:
    last_completed_states: List[OnlineGenerationState] = field(default_factory=list)
```

`@dataclass` 自动生成初始化方法。可变默认值要用 `default_factory=list`，否则多个实例可能共享同一个 list。

### 类型标注

```python
def move_batch_to_device(batch: Dict, device) -> Dict:
```

表示输入输出都是 dict。更具体的标注出现在 collate：

```python
def collate(examples: List[EncodedExample]) -> Dict[str, torch.Tensor]:
```

表示输入是 `EncodedExample` 列表，输出是字符串到 tensor 的字典。

### Literal 和 union

```python
stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None = None
```

`Literal[...]` 限定语义值集合，`| None` 表示也可以为空。

### 闭包

```python
def make_collate_fn(tokenizer):
    pad_token_id = tokenizer.pad_token_id

    def collate(...):
        ...

    return collate
```

内部函数 `collate` 捕获外部变量 `pad_token_id`，这就是闭包。

### 生成器和 `yield from`

```python
def iter_vllm_loadable_weights(name: str, tensor: torch.Tensor):
    yield name, tensor
```

含 `yield` 的函数返回生成器。`yield from other_generator` 会把另一个生成器的元素逐个转发出去。

### async/await

```python
async def sync_weights_to_vllm(...):
    await engine.start_weight_update()
```

`async def` 定义 coroutine，`await` 等待异步操作完成并让出事件循环。

### async for

```python
async for request_output in self.engine.generate(...):
    final_output = request_output
```

用于消费异步流。这里 vLLM generate 返回 streaming 输出。

### asyncio.Event

```python
self.paused = asyncio.Event()
self.paused.set()
self.paused.clear()
self.paused.is_set()
```

可以理解为异步世界里的开关。设置后表示暂停，清除后表示恢复。

### asyncio.Condition

```python
async with self._active_changed:
    await self._active_changed.wait_for(lambda: self._active_attempts == 0)
```

用于等待某个共享状态满足条件。这里等待所有 active generate attempt 结束。

### Ray actor

```python
remote_worker = ray.remote(num_gpus=1)(FSDPTrainWorker)
worker = remote_worker.remote(...)
result = ray.get(worker.get_rank.remote())
```

`ray.remote` 把普通类变成 actor 类；`.remote()` 是远程调用；`ray.get()` 等待并取回结果。

### PyTorch distributed/FSDP

```python
dist.init_process_group(backend="nccl", rank=rank, world_size=fsdp_world_size)
fully_shard(layer)
params_by_name[name].full_tensor()
```

`init_process_group` 建立训练侧通信组；`fully_shard` 做 FSDP2 参数分片；`full_tensor()` 从 FSDP shard all-gather 出完整 tensor，是 collective 操作。

## 容易踩坑的点

1. `full_tensor()` 是 collective。所有 FSDP rank 必须同时、同顺序调用，否则很容易死锁。
2. 初始同步必须用 `scope="all"`。vLLM 是 dummy weights 启动，只有 trainable 权重不够。
3. 后续同步用 `scope="trainable"`。这和 `--train-mode` 配套，可以减少传输量。
4. abort 后不会恢复旧 KV cache。代码使用 `input_ids + output_tokens` 重新提交请求，也就是重新 prefill。
5. `metadata` 的名字、dtype、shape 顺序必须和实际发送 tensor 的顺序一致。
6. `runner.paused` 和 `engine.pause_generation()` 是两层不同控制：前者阻止本地新 attempt，后者让 vLLM 中断/暂停生成。
7. `ray.get(...)` 会阻塞，所以主循环里用 `asyncio.to_thread(ray.get, train_handles)` 避免卡住 asyncio 事件循环。
8. 这个脚本不保存 checkpoint，也不通过磁盘 reload 权重；权重同步路径是 NCCL 直接传输。
9. 默认 `--infer-concurrency=2048` 是用户级请求并发，不等于 vLLM 同时 active 的 sequence 数；真正调度上限还受 `--vllm-max-num-seqs` 和 `--vllm-max-num-batched-tokens` 限制。
10. 脚本以单机 8 GPU demo 为前提，没有完整处理跨节点网络、Ray placement group 细节或生产级故障恢复。

## 一句话总结

`vllm_fsdp.py` 展示的是一个在线训练/推理权重更新边界：训练侧用 FSDP 分片训练，推理侧用 vLLM 持续服务；每到同步边界，系统 abort 当前生成、all-gather FSDP 参数、通过 NCCL 把权重送入 vLLM，再把未完成请求用“原 prompt + 已生成 token”重新提交。
