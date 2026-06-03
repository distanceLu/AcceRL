# SPDX-License-Identifier: Apache-2.0
"""Local rollout engine for vLLM pause/update/resume experiments.

This module turns the standalone logic from ``vllm_local_infer.py`` into a
small local engine object with the same shape the trainer will eventually call:

    pause_generation() -> update_weights(...) -> continue_generation()

The real vLLM dependency is imported lazily so the token-level resubmit logic
can be unit-tested with a fake AsyncLLMEngine on machines without GPUs.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

MODEL_NAME = "/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat"
PAUSE_TOKEN_THRESHOLD = 20
DEFAULT_MAX_TOKENS = 64


@dataclass
class GenerationConfig:
    n_samples: int = 1
    max_new_tokens: int = 16384
    min_new_tokens: int = 0
    max_tokens: int = 32768
    greedy: bool = False
    top_p: float = 1.0
    top_k: int = int(1e8)
    temperature: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    stop: list[str] | None = None
    frequency_penalty: float = 0.0
    use_beam_search: bool = False

    def new(self, **kwargs):
        args = asdict(self)
        args.update(kwargs)
        return GenerationConfig(**args)


@dataclass
class RolloutRequest:
    rid: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: list[int] = field(default_factory=list)
    gconfig: GenerationConfig = field(default_factory=GenerationConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    tokenizer: Any | None = None

    def copy(self):
        return RolloutRequest(
            rid=self.rid,
            input_ids=self.input_ids.copy(),
            gconfig=self.gconfig.new(),
            metadata=self.metadata.copy(),
            tokenizer=self.tokenizer,
        )


@dataclass
class RolloutResponse:
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    output_versions: list[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] = "stop"
    tokenizer: Any | None = None
    latency: float = float("inf")
    ttft: float = float("inf")
    itl: list[float] = field(default_factory=list)


def _load_sampling_params_cls():
    try:
        from vllm import SamplingParams
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "vLLM is required to create real SamplingParams. "
            "Install/activate the vLLM environment or pass sampling_params_cls "
            "to LocalRolloutEngine for tests."
        ) from exc
    return SamplingParams


def _finish_reason_from_output(request_output: Any) -> str | None:
    if not getattr(request_output, "outputs", None):
        return None
    completion = request_output.outputs[0]
    return getattr(completion, "finish_reason", None)


def _tokens_from_output(request_output: Any) -> list[int]:
    if not getattr(request_output, "outputs", None):
        return []
    return list(getattr(request_output.outputs[0], "token_ids", []) or [])


def _logprobs_from_output(request_output: Any, token_ids: list[int]) -> list[float]:
    if not getattr(request_output, "outputs", None):
        return []

    raw_logprobs = getattr(request_output.outputs[0], "logprobs", None)
    if not raw_logprobs:
        return [0.0] * len(token_ids)

    parsed: list[float] = []
    for token_id, token_logprobs in zip(token_ids, raw_logprobs):
        value = 0.0
        if isinstance(token_logprobs, dict):
            entry = token_logprobs.get(token_id)
            if entry is None and token_logprobs:
                entry = next(iter(token_logprobs.values()))
            if entry is not None:
                value = float(getattr(entry, "logprob", entry))
        elif token_logprobs is not None:
            value = float(getattr(token_logprobs, "logprob", token_logprobs))
        parsed.append(value)

    if len(parsed) < len(token_ids):
        parsed.extend([0.0] * (len(token_ids) - len(parsed)))
    return parsed[: len(token_ids)]


class LocalRolloutEngine:
    """Local vLLM rollout engine with token-level abort/resubmit handling."""

    def __init__(
        self,
        async_llm_engine: Any,
        max_resubmit_retries: int = 20,
        resubmit_wait: float = 0.0,
        sampling_params_cls: type | None = None,
    ):
        self.engine = async_llm_engine
        self.version = 0
        self.paused = asyncio.Event()
        self.paused.clear()
        self.max_resubmit_retries = max_resubmit_retries
        self.resubmit_wait = resubmit_wait
        self._sampling_params_cls = sampling_params_cls

    async def agenerate(self, request: RolloutRequest) -> RolloutResponse:
        req = request.copy() if hasattr(request, "copy") else request
        gconfig = req.gconfig

        if gconfig.n_samples != 1:
            raise ValueError(
                "LocalRolloutEngine only supports n_samples=1. "
                "Call agenerate multiple times for multiple samples."
            )

        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids), gconfig.max_new_tokens
        )
        if max_new_tokens <= 0:
            raise RuntimeError(
                f"max_new_tokens ({max_new_tokens}) is non-positive: "
                f"max_tokens={gconfig.max_tokens}, input_len={len(req.input_ids)}, "
                f"configured_max_new_tokens={gconfig.max_new_tokens}."
            )

        start_time = time.perf_counter()
        last_token_time: float | None = None
        ttft = float("inf")
        itl: list[float] = []
        accumulated_tokens: list[int] = []
        accumulated_logprobs: list[float] = []
        accumulated_versions: list[int] = []
        stop_reason: Literal["length", "stop", "tool_calls", "abort"] | None = None

        for attempt in range(1, self.max_resubmit_retries + 1):
            while self.paused.is_set():
                await asyncio.sleep(self.resubmit_wait)

            remaining = max_new_tokens - len(accumulated_tokens)
            if remaining <= 0:
                stop_reason = "length"
                break

            attempt_version = self.version
            prompt_token_ids = list(req.input_ids) + accumulated_tokens
            sampling_params = self._make_sampling_params(gconfig, remaining)
            request_id = f"{req.rid}-v{attempt_version}-try{attempt}-{uuid.uuid4()}"

            final_output = None
            request_finished = False
            try:
                async for request_output in self.engine.generate(
                    {"prompt_token_ids": prompt_token_ids},
                    sampling_params,
                    request_id=request_id,
                ):
                    final_output = request_output
                    request_finished = bool(getattr(request_output, "finished", False))
                    stream_tokens = _tokens_from_output(request_output)[:remaining]
                    await self._notify_progress(
                        req,
                        len(accumulated_tokens) + len(stream_tokens),
                        attempt_version,
                    )
            except asyncio.CancelledError:
                raise

            if final_output is None:
                stop_reason = "abort"
                continue

            attempt_tokens = _tokens_from_output(final_output)
            attempt_tokens = attempt_tokens[:remaining]
            attempt_logprobs = _logprobs_from_output(final_output, attempt_tokens)
            now = time.perf_counter()
            if attempt_tokens:
                if ttft == float("inf"):
                    ttft = now - start_time
                if last_token_time is not None:
                    itl.append(now - last_token_time)
                if len(attempt_tokens) > 1:
                    itl.extend([0.0] * (len(attempt_tokens) - 1))
                last_token_time = now

            accumulated_tokens.extend(attempt_tokens)
            accumulated_logprobs.extend(attempt_logprobs)
            accumulated_versions.extend([attempt_version] * len(attempt_tokens))

            raw_finish_reason = _finish_reason_from_output(final_output)
            stop_reason = self._normalize_stop_reason(raw_finish_reason)

            if stop_reason in ("stop", "tool_calls", "length"):
                break
            if len(accumulated_tokens) >= max_new_tokens:
                stop_reason = "length"
                break
            if not request_finished or stop_reason == "abort":
                await asyncio.sleep(self.resubmit_wait)
                continue
            break
        else:
            stop_reason = "length" if len(accumulated_tokens) >= max_new_tokens else "abort"

        if stop_reason == "abort" and len(accumulated_tokens) >= max_new_tokens:
            stop_reason = "length"
        if stop_reason is None:
            stop_reason = "length" if len(accumulated_tokens) >= max_new_tokens else "abort"

        latency = time.perf_counter() - start_time
        return RolloutResponse(
            input_tokens=list(request.input_ids),
            output_tokens=accumulated_tokens,
            output_logprobs=accumulated_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            tokenizer=getattr(request, "tokenizer", None),
            latency=latency,
            ttft=ttft,
            itl=itl,
        )

    async def _notify_progress(
        self, req: RolloutRequest, output_len: int, version: int
    ) -> None:
        callback = getattr(req, "metadata", {}).get("on_token_update")
        if callback is None:
            return
        result = callback(req, output_len, version)
        if inspect.isawaitable(result):
            await result

    async def pause_generation(self) -> None:
        self.paused.set()
        await self.engine.pause_generation(mode="abort", clear_cache=True)

    async def update_weights(self, path: str) -> None:
        await self.engine.collective_rpc(
            "reload_weights",
            kwargs={
                "weights_path": path,
                "is_checkpoint_format": True,
            },
        )
        self.version += 1

    async def continue_generation(self) -> None:
        await self.engine.resume_generation()
        self.paused.clear()

    def _make_sampling_params(self, gconfig: GenerationConfig, max_tokens: int):
        sampling_params_cls = self._sampling_params_cls or _load_sampling_params_cls()
        kwargs = {
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": max_tokens,
            "stop_token_ids": gconfig.stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "frequency_penalty": gconfig.frequency_penalty,
            "logprobs": 0,
        }
        if gconfig.stop:
            kwargs["stop"] = gconfig.stop
        if gconfig.use_beam_search:
            kwargs["use_beam_search"] = True
        return sampling_params_cls(**kwargs)

    @staticmethod
    def _normalize_stop_reason(
        stop_reason: Any,
    ) -> Literal["length", "stop", "tool_calls", "abort"]:
        if stop_reason in ("length", "stop", "tool_calls", "abort"):
            return stop_reason
        if stop_reason in ("eos", "stop_token", "stop_sequence"):
            return "stop"
        if stop_reason is None:
            return "abort"
        return "abort"


def create_async_engine():
    """Create an AsyncLLMEngine for the local 4-GPU vLLM demo."""
    try:
        import vllm
        from vllm import AsyncLLMEngine
        from vllm.v1.executor import Executor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "vLLM is required for the real demo. Activate the vLLM environment "
            "before running this script."
        ) from exc

    env_bin_dir = os.path.dirname(sys.executable)
    if env_bin_dir and env_bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = env_bin_dir + os.pathsep + os.environ.get("PATH", "")

    engine_args = vllm.AsyncEngineArgs(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        enable_expert_parallel=True,
        gpu_memory_utilization=0.7,
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


async def demo_main() -> None:
    prompts = [
        "The president of the United States is",
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    print(f"[init] Loading local model from {MODEL_NAME}")
    engine = create_async_engine()
    rollout = LocalRolloutEngine(engine)
    tokenizer = engine.get_tokenizer()
    pause_requested = asyncio.Event()

    def request_pause_at_threshold(
        req: RolloutRequest, output_len: int, version: int
    ) -> None:
        if output_len >= PAUSE_TOKEN_THRESHOLD and not pause_requested.is_set():
            print(
                "[generate] Request "
                f"{req.rid} reached {output_len} generated tokens at "
                f"version {version}; requesting weight update."
            )
            pause_requested.set()

    requests = [
        RolloutRequest(
            rid=f"demo-{index}",
            input_ids=tokenizer.encode(prompt),
            gconfig=GenerationConfig(
                n_samples=1,
                max_new_tokens=DEFAULT_MAX_TOKENS,
                max_tokens=len(tokenizer.encode(prompt)) + DEFAULT_MAX_TOKENS,
                greedy=True,
            ),
            metadata={"on_token_update": request_pause_at_threshold},
            tokenizer=tokenizer,
        )
        for index, prompt in enumerate(prompts)
    ]

    async def generate_one(req: RolloutRequest) -> RolloutResponse:
        return await rollout.agenerate(req)

    tasks = [asyncio.create_task(generate_one(req)) for req in requests]

    wait_for_pause = asyncio.create_task(pause_requested.wait())
    wait_for_all = asyncio.gather(*tasks)
    done, _ = await asyncio.wait(
        {wait_for_pause, wait_for_all},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if wait_for_pause in done and any(not task.done() for task in tasks):
        print("[sync] Pausing generation for weight reload...")
        await rollout.pause_generation()
        print("[sync] Reloading weights...")
        await rollout.update_weights(MODEL_NAME)
        print("[sync] Resuming generation...")
        await rollout.continue_generation()
    else:
        wait_for_pause.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await wait_for_pause

    responses = await wait_for_all
    print("[generate] Generation complete.")
    print("-" * 60)
    for prompt, req, resp in zip(prompts, requests, responses):
        print(f"Prompt: {prompt!r}")
        print(f"Prompt token IDs ({len(req.input_ids)}): {req.input_ids!r}")
        print(f"Output token IDs ({len(resp.output_tokens)}): {resp.output_tokens!r}")
        print(f"Output versions: {resp.output_versions!r}")
        print(f"Stop reason: {resp.stop_reason}")
        print(f"Generated: {tokenizer.decode(resp.output_tokens)!r}")
        print("-" * 60)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(demo_main())
