# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


MODULE_PATH = Path(__file__).with_name("vllm_local_rollout_engine.py")
SPEC = importlib.util.spec_from_file_location("vllm_local_rollout_engine", MODULE_PATH)
rollout_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = rollout_mod
SPEC.loader.exec_module(rollout_mod)


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def output(token_ids, finished, finish_reason=None, logprobs=None):
    return SimpleNamespace(
        finished=finished,
        outputs=[
            SimpleNamespace(
                token_ids=token_ids,
                finish_reason=finish_reason,
                logprobs=logprobs,
            )
        ],
    )


class PlannedEngine:
    def __init__(self, plans):
        self.plans = list(plans)
        self.generate_calls = []
        self.pause_calls = []
        self.resume_calls = 0
        self.collective_rpc_calls = []

    async def generate(self, prompt, sampling_params, request_id):
        self.generate_calls.append((prompt, sampling_params, request_id))
        plan = self.plans.pop(0)
        for item in plan:
            if callable(item):
                item()
                await asyncio.sleep(0)
                continue
            yield item

    async def pause_generation(self, **kwargs):
        self.pause_calls.append(kwargs)

    async def collective_rpc(self, method, kwargs):
        self.collective_rpc_calls.append((method, kwargs))

    async def resume_generation(self):
        self.resume_calls += 1


class BlockingAbortEngine(PlannedEngine):
    def __init__(self):
        super().__init__(plans=[])
        self.partial_ready = asyncio.Event()
        self.pause_called = asyncio.Event()

    async def generate(self, prompt, sampling_params, request_id):
        self.generate_calls.append((prompt, sampling_params, request_id))
        if len(self.generate_calls) == 1:
            yield output([10, 11], finished=False)
            self.partial_ready.set()
            await self.pause_called.wait()
            yield output([10, 11], finished=True, finish_reason="abort")
        else:
            yield output([12, 13], finished=True, finish_reason="length")

    async def pause_generation(self, **kwargs):
        await super().pause_generation(**kwargs)
        self.pause_called.set()


class LocalRolloutEngineTest(unittest.IsolatedAsyncioTestCase):
    def make_request(self, input_ids=None, max_new_tokens=4, **gconfig_kwargs):
        input_ids = [1, 2] if input_ids is None else input_ids
        gconfig = rollout_mod.GenerationConfig(
            n_samples=1,
            max_new_tokens=max_new_tokens,
            max_tokens=len(input_ids) + max_new_tokens,
            greedy=True,
            **gconfig_kwargs,
        )
        return rollout_mod.RolloutRequest(
            rid="req",
            input_ids=input_ids,
            gconfig=gconfig,
        )

    async def test_normal_generation_finishes_once_with_version_zero(self):
        engine = PlannedEngine(
            [[output([10, 11, 12], finished=True, finish_reason="stop")]]
        )
        rollout = rollout_mod.LocalRolloutEngine(
            engine, sampling_params_cls=FakeSamplingParams
        )

        response = await rollout.agenerate(self.make_request(max_new_tokens=3))

        self.assertEqual(response.input_tokens, [1, 2])
        self.assertEqual(response.output_tokens, [10, 11, 12])
        self.assertEqual(response.output_versions, [0, 0, 0])
        self.assertEqual(response.stop_reason, "stop")
        self.assertEqual(len(engine.generate_calls), 1)
        self.assertEqual(engine.generate_calls[0][0], {"prompt_token_ids": [1, 2]})

    async def test_pause_update_resume_resubmits_with_new_version(self):
        engine = BlockingAbortEngine()
        rollout = rollout_mod.LocalRolloutEngine(
            engine, sampling_params_cls=FakeSamplingParams
        )
        task = asyncio.create_task(rollout.agenerate(self.make_request(max_new_tokens=4)))

        await engine.partial_ready.wait()
        await rollout.pause_generation()
        await rollout.update_weights("/tmp/checkpoint")
        await rollout.continue_generation()
        response = await task

        self.assertEqual(response.output_tokens, [10, 11, 12, 13])
        self.assertEqual(response.output_versions, [0, 0, 1, 1])
        self.assertEqual(response.stop_reason, "length")
        self.assertEqual(engine.generate_calls[1][0], {"prompt_token_ids": [1, 2, 10, 11]})

    async def test_generation_never_exceeds_max_new_tokens(self):
        engine = PlannedEngine(
            [
                [output([10, 11, 12, 13, 14], finished=True, finish_reason="length")],
            ]
        )
        rollout = rollout_mod.LocalRolloutEngine(
            engine, sampling_params_cls=FakeSamplingParams
        )

        response = await rollout.agenerate(self.make_request(max_new_tokens=3))

        self.assertEqual(response.output_tokens, [10, 11, 12])
        self.assertEqual(response.output_versions, [0, 0, 0])
        self.assertEqual(response.stop_reason, "length")
        self.assertEqual(engine.generate_calls[0][1].kwargs["max_tokens"], 3)

    async def test_lifecycle_methods_call_vllm_and_manage_pause_state(self):
        engine = PlannedEngine([])
        rollout = rollout_mod.LocalRolloutEngine(
            engine, sampling_params_cls=FakeSamplingParams
        )

        await rollout.pause_generation()
        self.assertTrue(rollout.paused.is_set())
        self.assertEqual(engine.pause_calls, [{"mode": "abort", "clear_cache": True}])

        await rollout.update_weights("/tmp/weights")
        self.assertEqual(rollout.version, 1)
        self.assertEqual(
            engine.collective_rpc_calls,
            [
                (
                    "reload_weights",
                    {
                        "weights_path": "/tmp/weights",
                        "is_checkpoint_format": True,
                    },
                )
            ],
        )

        await rollout.continue_generation()
        self.assertFalse(rollout.paused.is_set())
        self.assertEqual(engine.resume_calls, 1)


if __name__ == "__main__":
    unittest.main()
