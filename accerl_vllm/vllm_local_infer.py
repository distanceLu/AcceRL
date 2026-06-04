# SPDX-License-Identifier: Apache-2.0
"""Run local vLLM inference with a mid-generation weight-update pause.

This demo intentionally aborts unfinished requests before a weight update and
resubmits them afterwards instead of resuming their old KV cache.  The restart
input is built from the original prompt token IDs plus the tokens already
generated before the pause, so the post-update continuation is prefilling from
the updated weights without relying on string re-tokenization.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import vllm
from vllm import AsyncLLMEngine, RequestOutput, SamplingParams
from vllm.v1.executor import Executor

MODEL_NAME = "/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat"
PAUSE_TOKEN_THRESHOLD = 20000
DEFAULT_MAX_TOKENS = 2560
MAX_RESUBMIT_RETRIES = 20
INFER_BATCH_SIZE = 1024
VLLM_MAX_NUM_BATCHED_TOKENS = 4096
VLLM_MAX_NUM_SEQS = 1024

# vLLM worker processes inherit PATH from this parent process.  Some shells run
# the env's python directly without fully activating the conda env, leaving
# env-local tools such as ninja invisible to FlashInfer JIT compilation.
# ENV_BIN_DIR = os.path.dirname(sys.executable)
# if ENV_BIN_DIR and ENV_BIN_DIR not in os.environ.get("PATH", "").split(os.pathsep):
#     os.environ["PATH"] = ENV_BIN_DIR + os.pathsep + os.environ.get("PATH", "")


@dataclass
class GenerationState:
    """Application-level state preserved across the pause/restart boundary."""

    index: int
    original_prompt: str
    original_prompt_token_ids: list[int]
    requested_max_tokens: int
    latest_output: Optional[RequestOutput] = None
    latest_prompt_token_ids: list[int] = field(default_factory=list)
    generated_text: str = ""
    generated_token_ids: list[int] = field(default_factory=list)
    generated_versions: list[int] = field(default_factory=list)
    post_update_text: str = ""
    post_update_token_ids: list[int] = field(default_factory=list)
    post_update_versions: list[int] = field(default_factory=list)
    completed_before_update: bool = False
    restarted: bool = False

    @property
    def remaining_max_tokens(self) -> int:
        generated_tokens = len(self.generated_token_ids) + len(
            self.post_update_token_ids
        )
        return max(0, self.requested_max_tokens - generated_tokens)

    @property
    def restart_prompt_token_ids(self) -> list[int]:
        prompt_token_ids = self.latest_prompt_token_ids or self.original_prompt_token_ids
        return prompt_token_ids + self.generated_token_ids + self.post_update_token_ids

    @property
    def full_generated_text(self) -> str:
        return self.generated_text + self.post_update_text

    @property
    def output_versions(self) -> list[int]:
        return self.generated_versions + self.post_update_versions


class InferTokenStats:
    """Aggregate generated-token throughput across concurrent infer streams."""

    def __init__(self) -> None:
        self.total_tokens = 0
        self._start_time = time.monotonic()
        self._interval_tokens = 0
        self._last_snapshot_time = time.monotonic()
        self._lock = asyncio.Lock()

    async def add(self, token_count: int) -> None:
        if token_count <= 0:
            return
        async with self._lock:
            self.total_tokens += token_count
            self._interval_tokens += token_count

    async def snapshot_and_reset_interval(self) -> tuple[int, int, float]:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_snapshot_time
            self._last_snapshot_time = now
            interval_tokens = self._interval_tokens
            self._interval_tokens = 0
            return interval_tokens, self.total_tokens, elapsed

    async def final_summary(self) -> tuple[int, float, float]:
        async with self._lock:
            elapsed = time.monotonic() - self._start_time
            tokens_per_s = self.total_tokens / elapsed if elapsed > 0 else 0.0
            return self.total_tokens, elapsed, tokens_per_s


async def report_infer_token_stats(
    stats: InferTokenStats,
    interval_seconds: float = 1.0,
) -> None:
    """Print aggregate generated tokens once per interval."""
    while True:
        await asyncio.sleep(interval_seconds)
        interval_tokens, total_tokens, elapsed = (
            await stats.snapshot_and_reset_interval()
        )
        tokens_per_s = interval_tokens / elapsed if elapsed > 0 else 0.0
        print(
            "[infer-stats] "
            f"interval={elapsed:.2f}s "
            f"tokens={interval_tokens} "
            f"tokens_per_s={tokens_per_s:.2f} "
            f"total_tokens={total_tokens}"
        )


async def flush_infer_token_stats(stats: InferTokenStats) -> None:
    """Print the final partial interval, if it contains generated tokens."""
    interval_tokens, total_tokens, elapsed = await stats.snapshot_and_reset_interval()
    if interval_tokens <= 0:
        return
    tokens_per_s = interval_tokens / elapsed if elapsed > 0 else 0.0
    print(
        "[infer-stats] "
        f"interval={elapsed:.2f}s "
        f"tokens={interval_tokens} "
        f"tokens_per_s={tokens_per_s:.2f} "
        f"total_tokens={total_tokens}"
    )


async def print_final_infer_token_stats(stats: InferTokenStats) -> None:
    total_tokens, elapsed, tokens_per_s = await stats.final_summary()
    print(
        "[infer-summary] "
        f"elapsed={elapsed:.2f}s "
        f"total_tokens={total_tokens} "
        f"tokens_per_s={tokens_per_s:.2f}"
    )


def create_async_engine() -> AsyncLLMEngine:
    """Create an AsyncLLMEngine for 4-GPU local inference.

    AsyncLLMEngine exposes streaming request outputs plus
    pause_generation/resume_generation, which the synchronous LLM wrapper does
    not provide.
    """
    engine_args = vllm.AsyncEngineArgs(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        enable_expert_parallel=False,
        gpu_memory_utilization=0.7,
        # Keep vLLM's initialization/profile dummy batch small for this demo.
        # The default can be 16K tokens, which is very slow for single-GPU MoE.
        max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=VLLM_MAX_NUM_SEQS,
        disable_custom_all_reduce=True,
        max_model_len=2048,
    )
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    return AsyncLLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_requests=engine_args.enable_log_requests,
        log_stats=not engine_args.disable_log_stats,
    )

    # Single-GPU fallback if TP/EP is not needed or the 4-GPU config fails:
    # engine_args = vllm.AsyncEngineArgs(
    #     model=MODEL_NAME,
    #     trust_remote_code=True,
    #     dtype="bfloat16",
    #     enforce_eager=True,
    #     gpu_memory_utilization=0.7,
    # )


async def reload_model_weights_from_disk(engine: AsyncLLMEngine) -> None:
    """Reload local checkpoint weights into the existing vLLM engine.

    This keeps the AsyncLLMEngine and its worker processes alive.  The running
    requests are aborted before this hook is called, and restarted afterwards
    with their partial generations as fresh prompts.
    """
    print(f"[sync] Reloading model weights from {MODEL_NAME}...")
    await engine.collective_rpc(
        "reload_weights",
        kwargs={
            "weights_path": MODEL_NAME,
            "is_checkpoint_format": True,
        },
    )
    print("[sync] Model weights reloaded.")


async def pause_generation_for_weight_update(engine: AsyncLLMEngine) -> None:
    """Pause vLLM itself, aborting in-flight requests and clearing KV cache."""
    # start_time = asyncio.get_event_loop().time()
    await engine.pause_generation(mode="abort", clear_cache=True)
    # end_time = asyncio.get_event_loop().time()
    # print(
    #     f"!!!!!!!!!!!!!!!!!!![sync] pause_generation_for_weight_update took {end_time - start_time:.2f} seconds."
    # )

def _copy_generation_from_output(
    state: GenerationState,
    output: RequestOutput,
    version: int,
) -> int:
    """Refresh pre-update text/token state from the latest streamed output."""
    state.latest_output = output
    if output.prompt_token_ids is not None:
        state.latest_prompt_token_ids = list(output.prompt_token_ids)
    if not output.outputs:
        return 0

    completion = output.outputs[0]
    state.generated_text = completion.text
    state.generated_token_ids = list(completion.token_ids)
    state.generated_versions = [version] * len(state.generated_token_ids)
    return len(state.generated_token_ids)


def _copy_post_update_from_output(
    state: GenerationState,
    output: RequestOutput,
    version: int,
    base_text: str,
    base_token_ids: list[int],
) -> str | None:
    """Merge one resubmitted request's partial output into post-update state."""
    state.latest_output = output
    if not output.outputs:
        return None

    completion = output.outputs[0]
    attempt_token_ids = list(completion.token_ids)
    state.post_update_text = base_text + completion.text
    state.post_update_token_ids = base_token_ids + attempt_token_ids
    state.post_update_versions = [version] * len(state.post_update_token_ids)
    return getattr(completion, "finish_reason", None)


async def stream_until_pause(
    engine: AsyncLLMEngine,
    state: GenerationState,
    sampling_params: SamplingParams,
    pause_requested: asyncio.Event, # 由主协程设置以触发权重更新
    pause_after_tokens: int,
    version: int,
    token_stats: InferTokenStats,
) -> None:
    """Stream one request until it finishes or a global pause is requested."""
    request_id = f"pre-update-{state.index}-{uuid.uuid4()}"
    async for request_output in engine.generate(
        {"prompt": state.original_prompt},
        sampling_params,
        request_id=request_id,
    ):
        previous_generated_tokens = len(state.generated_token_ids)
        generated_tokens = _copy_generation_from_output(
            state,
            request_output,
            version,
        )
        await token_stats.add(generated_tokens - previous_generated_tokens)

        if request_output.finished:
            state.completed_before_update = True
            return

        if generated_tokens >= pause_after_tokens and not pause_requested.is_set():
            print(
                "[generate] Request "
                f"{state.index} reached {generated_tokens} generated tokens; "
                "requesting weight update."
            )
            pause_requested.set()
            return

        if pause_requested.is_set():
            return


async def stream_after_update(
    engine: AsyncLLMEngine,
    state: GenerationState,
    version: int,
    token_stats: InferTokenStats,
) -> None:
    """Restart an unfinished request, resubmitting on abort until it finishes."""
    if state.completed_before_update:
        return

    state.restarted = True
    for attempt in range(1, MAX_RESUBMIT_RETRIES + 1):
        remaining_max_tokens = state.remaining_max_tokens
        if remaining_max_tokens <= 0:
            return

        base_text = state.post_update_text
        base_token_ids = list(state.post_update_token_ids)
        restart_sampling_params = SamplingParams(
            temperature=0,
            max_tokens=remaining_max_tokens,
        )
        request_id = f"post-update-{state.index}-try{attempt}-{uuid.uuid4()}"
        finish_reason = None
        request_finished = False

        async for request_output in engine.generate(
            {"prompt_token_ids": state.restart_prompt_token_ids},
            restart_sampling_params,
            request_id=request_id,
        ):
            previous_post_update_tokens = len(state.post_update_token_ids)
            finish_reason = _copy_post_update_from_output(
                state,
                request_output,
                version,
                base_text,
                base_token_ids,
            )
            await token_stats.add(
                len(state.post_update_token_ids) - previous_post_update_tokens
            )
            request_finished = request_output.finished

        if finish_reason in {"stop", "length"} or state.remaining_max_tokens <= 0:
            return

        if not request_finished or finish_reason == "abort":
            print(
                "[generate] Request "
                f"{state.index} post-update attempt {attempt} ended with "
                f"finish_reason={finish_reason!r}; resubmitting with "
                f"{state.remaining_max_tokens} tokens remaining."
            )
            await asyncio.sleep(0)
            continue

        # Unknown terminal reason: keep the partial result but avoid spinning.
        return

    print(
        "[generate] Request "
        f"{state.index} reached MAX_RESUBMIT_RETRIES={MAX_RESUBMIT_RETRIES}; "
        "keeping the partial post-update result."
    )


async def pause_update_and_restart(
    engine: AsyncLLMEngine,
    states: list[GenerationState],
    current_version: int,
    token_stats: InferTokenStats,
) -> None:
    """Reload weights, then restart unfinished requests."""
    # start_time = asyncio.get_event_loop().time()
    await reload_model_weights_from_disk(engine)
    # end_time = asyncio.get_event_loop().time()
    # print(
    #     f"!!!!!!!!!!!!!!!![sync] pause_update_and_restart's weight reload step took "
    #     f"{end_time - start_time:.2f} seconds."
    # )
    next_version = current_version + 1

    print("[sync] Resuming generation for fresh restarted requests...")
    # start_time = asyncio.get_event_loop().time()
    await engine.resume_generation()
    # end_time = asyncio.get_event_loop().time()
    # print(f"!!!!!!!!!!!!!!![sync] Generation resumed with weight version {next_version}.")
    # print(
    #     f"[sync] pause_update_and_restart's generation resume step took "
    #     f"{end_time - start_time:.2f} seconds."
    # )

    restart_tasks = [
        asyncio.create_task(
            stream_after_update(
                engine,
                state,
                version=next_version,
                token_stats=token_stats,
            )
        )
        for state in states
        if not state.completed_before_update and state.remaining_max_tokens > 0
    ]
    if restart_tasks:
        await asyncio.gather(*restart_tasks)


def print_results(states: list[GenerationState]) -> None:
    print("-" * 60)
    for state in states:
        print(f"Prompt: {state.original_prompt!r}")
        print(
            "Prompt token IDs "
            f"({len(state.original_prompt_token_ids)}): "
            f"{state.original_prompt_token_ids!r}"
        )
        print(
            "Pre-update generated "
            f"({len(state.generated_token_ids)} tokens): {state.generated_text!r}"
        )
        print(f"Pre-update token IDs: {state.generated_token_ids!r}")

        if state.completed_before_update:
            print("Post-update generated: <not restarted; request finished early>")
        elif state.restarted:
            print(
                "Post-update generated "
                f"({len(state.post_update_token_ids)} tokens): "
                f"{state.post_update_text!r}"
            )
            print(f"Post-update token IDs: {state.post_update_token_ids!r}")
        else:
            print("Post-update generated: <not restarted; no remaining token budget>")

        print(f"Output versions: {state.output_versions!r}")
        print(f"Full generated: {state.full_generated_text!r}")
        print("-" * 60)


async def main() -> None:
    base_prompts = [
        "The president of the United States is",
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = [
        base_prompts[index % len(base_prompts)]
        for index in range(INFER_BATCH_SIZE)
    ]

    sampling_params = SamplingParams(temperature=0, max_tokens=DEFAULT_MAX_TOKENS)
    # 把这个中断事件改成加载模型
    pause_requested = asyncio.Event()

    print(f"[init] Loading local model from {MODEL_NAME}")
    engine = create_async_engine()
    tokenizer = engine.get_tokenizer()
    current_version = 0
    states = [
        GenerationState(
            index=index,
            original_prompt=prompt,
            original_prompt_token_ids=tokenizer.encode(prompt),
            requested_max_tokens=DEFAULT_MAX_TOKENS,
        )
        for index, prompt in enumerate(prompts)
    ]

    print(
        "[generate] Starting streaming generation... "
        f"batch_size={len(prompts)} "
        f"max_num_seqs={VLLM_MAX_NUM_SEQS} "
        f"max_num_batched_tokens={VLLM_MAX_NUM_BATCHED_TOKENS}"
    )
    token_stats = InferTokenStats()
    stats_reporter_task = asyncio.create_task(report_infer_token_stats(token_stats))
    generation_tasks = [
        asyncio.create_task(
            stream_until_pause(
                engine=engine,
                state=state,
                sampling_params=sampling_params,
                pause_requested=pause_requested,
                pause_after_tokens=PAUSE_TOKEN_THRESHOLD,
                version=current_version,
                token_stats=token_stats,
            )
        )
        for state in states
    ]
    pause_wait_task = asyncio.create_task(pause_requested.wait())

    pending_generation_tasks = set(generation_tasks)
    while pending_generation_tasks:
        done, pending = await asyncio.wait(
            [*pending_generation_tasks, pause_wait_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if pause_wait_task in done:
            print(
                "[sync] Pausing vLLM generation with mode='abort' "
                "and clear_cache=True..."
            )
            # abort 模式会让正在生成的请求立即停止；clear_cache=True 避免旧权重
            # 生成的 KV cache 在权重更新后被复用。
            await pause_generation_for_weight_update(engine)
            print("[sync] Generation paused.")
            # 等待所有生成任务完成（无论是正常完成还是被 pause 中断），再进行权重更新和重启
            await asyncio.gather(*pending_generation_tasks, return_exceptions=True)
            # 进行权重更新并重启未完成的请求
            await pause_update_and_restart(
                engine,
                states,
                current_version=current_version,
                token_stats=token_stats,
            )
            current_version += 1
            break

        for task in done:
            if task is not pause_wait_task:
                task.result()

        pending_generation_tasks = {
            task for task in pending if task is not pause_wait_task
        }
    else:
        pause_wait_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pause_wait_task
        print("[generate] All requests completed before any weight update trigger.")

    print("[generate] Generation complete.")
    stats_reporter_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await stats_reporter_task
    await flush_infer_token_stats(token_stats)
    await print_final_infer_token_stats(token_stats)
    # Detailed final generations are noisy for large batch throughput runs.
    # print_results(states)


if __name__ == "__main__":
    asyncio.run(main())
