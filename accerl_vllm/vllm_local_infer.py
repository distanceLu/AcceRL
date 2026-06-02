# SPDX-License-Identifier: Apache-2.0
"""Run local vLLM inference with a mid-generation weight-update pause.

This demo intentionally restarts unfinished requests after a weight update
instead of resuming their old KV cache.  The restart prompt is built from the
original prompt plus the tokens already generated before the pause, so the
post-update continuation is prefilling from the updated weights.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Optional

import vllm
from vllm import AsyncLLMEngine, RequestOutput, SamplingParams
from vllm.v1.executor import Executor

MODEL_NAME = "/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat"
PAUSE_TOKEN_THRESHOLD = 10
DEFAULT_MAX_TOKENS = 64

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
    requested_max_tokens: int
    latest_output: Optional[RequestOutput] = None
    generated_text: str = ""
    generated_token_ids: list[int] = field(default_factory=list)
    post_update_text: str = ""
    post_update_token_ids: list[int] = field(default_factory=list)
    completed_before_update: bool = False
    restarted: bool = False

    @property
    def remaining_max_tokens(self) -> int:
        return max(0, self.requested_max_tokens - len(self.generated_token_ids))

    @property
    def restart_prompt(self) -> str:
        return self.original_prompt + self.generated_text

    @property
    def full_generated_text(self) -> str:
        return self.generated_text + self.post_update_text


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
        enable_expert_parallel=True,
        gpu_memory_utilization=0.7,
        # Keep vLLM's initialization/profile dummy batch small for this demo.
        # The default can be 16K tokens, which is very slow for single-GPU MoE.
        max_num_batched_tokens=1024,
        max_num_seqs=4,
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


def _copy_generation_from_output(state: GenerationState, output: RequestOutput) -> int:
    """Refresh pre-update text/token state from the latest streamed output."""
    state.latest_output = output
    if not output.outputs:
        return 0

    completion = output.outputs[0]
    state.generated_text = completion.text
    state.generated_token_ids = list(completion.token_ids)
    return len(state.generated_token_ids)


async def stream_until_pause(
    engine: AsyncLLMEngine,
    state: GenerationState,
    sampling_params: SamplingParams,
    pause_requested: asyncio.Event, # 由主协程设置以触发权重更新
    pause_after_tokens: int,
) -> None:
    """Stream one request until it finishes or a global pause is requested."""
    request_id = f"pre-update-{state.index}-{uuid.uuid4()}"
    async for request_output in engine.generate(
        {"prompt": state.original_prompt},
        sampling_params,
        request_id=request_id,
    ):
        generated_tokens = _copy_generation_from_output(state, request_output)

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
) -> None:
    """Restart one unfinished request using the partial result as prompt."""
    if state.completed_before_update:
        return

    remaining_max_tokens = state.remaining_max_tokens
    if remaining_max_tokens <= 0:
        return

    state.restarted = True
    restart_sampling_params = SamplingParams(
        temperature=0,
        max_tokens=remaining_max_tokens,
    )
    request_id = f"post-update-{state.index}-{uuid.uuid4()}"
    async for request_output in engine.generate(
        {"prompt": state.restart_prompt},
        restart_sampling_params,
        request_id=request_id,
    ):
        state.latest_output = request_output
        if not request_output.outputs:
            continue

        completion = request_output.outputs[0]
        state.post_update_text = completion.text
        state.post_update_token_ids = list(completion.token_ids)


async def pause_update_and_restart(
    engine: AsyncLLMEngine,
    states: list[GenerationState],
) -> None:
    """Reload weights, then restart unfinished requests."""
    await reload_model_weights_from_disk(engine)

    print("[sync] Resuming generation for fresh restarted requests...")
    await engine.resume_generation()
    print("[sync] Generation resumed.")

    restart_tasks = [
        asyncio.create_task(stream_after_update(engine, state))
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
            "Pre-update generated "
            f"({len(state.generated_token_ids)} tokens): {state.generated_text!r}"
        )

        if state.completed_before_update:
            print("Post-update generated: <not restarted; request finished early>")
        elif state.restarted:
            print(
                "Post-update generated "
                f"({len(state.post_update_token_ids)} tokens): "
                f"{state.post_update_text!r}"
            )
        else:
            print("Post-update generated: <not restarted; no remaining token budget>")

        print(f"Full generated: {state.full_generated_text!r}")
        print("-" * 60)


async def main() -> None:
    prompts = [
        "The president of the United States is",
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=DEFAULT_MAX_TOKENS)
    states = [
        GenerationState(
            index=index,
            original_prompt=prompt,
            requested_max_tokens=DEFAULT_MAX_TOKENS,
        )
        for index, prompt in enumerate(prompts)
    ]
    # 把这个中断事件改成加载模型
    pause_requested = asyncio.Event()

    print(f"[init] Loading local model from {MODEL_NAME}")
    engine = create_async_engine()

    print("[generate] Starting streaming generation...")
    generation_tasks = [
        asyncio.create_task(
            stream_until_pause(
                engine=engine,
                state=state,
                sampling_params=sampling_params,
                pause_requested=pause_requested,
                pause_after_tokens=PAUSE_TOKEN_THRESHOLD,
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
            print("[sync] Pausing generation with mode='abort'...")
            # aboret 模式会让正在生成的请求立即停止并抛出异常，未完成的请求会被标记为 finished=True 以触发后续重启逻辑，KV cache 不会被保留
            await engine.pause_generation(mode="abort")
            print("[sync] Generation paused.")
            # 等待所有生成任务完成（无论是正常完成还是被 pause 中断），再进行权重更新和重启
            await asyncio.gather(*pending_generation_tasks, return_exceptions=True)
            # 进行权重更新并重启未完成的请求
            await pause_update_and_restart(engine, states)
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
    print_results(states)


if __name__ == "__main__":
    asyncio.run(main())
