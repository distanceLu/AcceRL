"""Exercise explicit-state allocation through actual Replay/collector CPU code.

The policy RPC is fake and the simulator is a small deterministic CPU adapter;
Replay, the initial-state scheduler, reset helper, collector loop, episode
packing, and producer orchestration are the actual implementation.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from initial_state_env import initial_state_pool_metadata, reset_with_initial_state
from initial_state_scheduler import CheckpointMismatchError, UnresolvedLeaseError
from pipeline_runtime import run_producer
from test_pipeline_integration_cpu import actor, payload
from test_pipeline_producers_cpu import make_worker, Remote


class TinyExplicitEnv:
    """Imitate the existing LIBERO reset contract, with counted simulator steps."""
    def __init__(self, count=50, horizon=5, reset_steps=10, success_at=None):
        self.initial_states = np.arange(count * 4, dtype=np.float64).reshape(count, 4)
        self.max_episode_steps = horizon
        self.num_steps_wait = reset_steps
        self.episode_idx = 0
        self.task_description = "pick bowl"
        self.steps = 0
        self.total_sim_steps = 0
        self.seed_calls, self.selected_ids = [], []
        self.success_at = success_at
        self.env = SimpleNamespace(seed=self.seed)

    def seed(self, value):
        self.seed_calls.append(value)

    def observation(self):
        return {"full_image": np.full((8, 8, 3), self.steps, np.uint8),
                "wrist_image": np.full((8, 8, 3), self.steps + 10, np.uint8)}

    def reset(self, seed=None):
        self.steps = 0
        state_id = self.episode_idx % len(self.initial_states)
        self.selected_ids.append(state_id)
        self.episode_idx += 1
        self.total_sim_steps += self.num_steps_wait
        return self.observation(), {"is_success": False, "task_description": self.task_description}

    def step(self, action):
        self.steps += 1
        self.total_sim_steps += 1
        terminated = self.success_at is not None and self.steps >= self.success_at
        truncated = self.steps >= self.max_episode_steps
        return self.observation(), float(terminated), terminated, truncated, {"is_success": terminated}

    def get_name(self):
        return "tiny-explicit"


def owner_for(**changes):
    config = dict(initial_state_enabled=True, initial_state_count=50,
                  initial_state_seed=20261026, real_env_step_budget=100,
                  max_episode_frames=6, frame_budget=120)
    config.update(changes)
    return actor(**config)


def episode_payload(lease, identity="train-0", steps=5, success=False):
    result = payload(identity, count=steps + 1)
    result["terminated"][:] = False
    result["truncated"][-1] = not success
    result["terminated"][-1] = success
    result["success"][:] = False
    result["success"][-1] = success
    assigned = lease["initial_state"]
    result["metadata"] = dict(
        initial_state_id=assigned["initial_state_id"], env_seed=assigned["env_seed"],
        reset_env_steps=assigned["reset_steps"], initial_state_hash="b" * 64,
        termination_reason="success" if success else "task_horizon",
        policy_versions=[1, 2])
    return result


class InitialStatePipelineTests(unittest.TestCase):
    def test_actual_actor_accepts_integer_worker_and_finishes_metadata(self):
        owner = owner_for()
        pool = initial_state_pool_metadata(TinyExplicitEnv())
        lease = owner.acquire_real(0, pool)
        assigned = lease["initial_state"]
        self.assertEqual(assigned["worker_id"], "0")
        result = owner.finish_real_episode(lease["token"], episode_payload(lease, success=True))
        self.assertEqual(result["windows"], 4)
        status = owner.pipeline_status()["initial_states"]
        self.assertEqual(status["known_action_steps"], 5)
        self.assertEqual(status["known_reset_steps"], 10)
        self.assertEqual(status["episodes_completed"], 1)
        state = owner.initial_state_checkpoint()
        self.assertTrue(state["enabled"])
        terminal = state["scheduler"]["terminal"][assigned["token"]]
        self.assertEqual(terminal["metadata"]["policy_versions"], [1, 2])
        self.assertEqual(terminal["metadata"]["initial_state_hash"], "b" * 64)
        self.assertEqual(state["pool"], pool)

    def test_actual_collector_partial_episode_stops_mid_policy_chunk(self):
        owner = owner_for(real_env_step_budget=11)
        env = TinyExplicitEnv(horizon=5)
        pool = initial_state_pool_metadata(env)
        lease = owner.acquire_real(0, pool)
        self.assertEqual(lease["initial_state"]["max_action_steps"], 1)
        events, collected = [], []

        def finish(token, result):
            collected.append(result)
            return owner.finish_real_episode(token, result)

        wm_replay = SimpleNamespace(finish_real_episode=Remote(events, "finish_real", finish))
        worker, _ = make_worker(events, None, wm_replay, mode="real")
        worker.envs = [env]
        worker.initial_state_enabled = True
        worker.initial_state_pool = pool
        worker.pipeline_job, worker.pipeline_lease = lease, lease["token"]
        worker.ctrl_wm_frames = 3
        worker.get_one_episode()
        self.assertEqual(env.steps, 1)
        self.assertEqual(env.total_sim_steps, 11)
        self.assertEqual(env.seed_calls, [lease["initial_state"]["env_seed"]])
        self.assertEqual(env.selected_ids, [lease["initial_state"]["initial_state_id"]])
        result = collected[0]
        self.assertEqual(result["frames"].shape[0], 2)
        self.assertEqual(result["metadata"]["termination_reason"], "real_env_budget")
        self.assertTrue(result["metadata"]["budget_censored"])
        np.testing.assert_array_equal(result["terminated"], [False, False])
        np.testing.assert_array_equal(result["truncated"], [False, True])
        self.assertNotIn("submit:stats", events)
        status = owner.pipeline_status()["initial_states"]
        self.assertEqual(status["consumed_env_steps"], 11)
        self.assertEqual(status["episodes_completed"], 1)
        self.assertTrue(owner.acquire_real(0, pool)["budget_exhausted"])

    def test_actual_collector_success_refunds_and_reports_uncensored_success(self):
        owner = owner_for(real_env_step_budget=50)
        env = TinyExplicitEnv(horizon=5, success_at=2)
        lease = owner.acquire_real(0, initial_state_pool_metadata(env))
        events, collected = [], []

        def finish(token, result):
            collected.append(result)
            return owner.finish_real_episode(token, result)

        worker, _ = make_worker(events, None,
            SimpleNamespace(finish_real_episode=Remote(events, "finish_real", finish)), mode="real")
        worker.envs = [env]
        worker.initial_state_enabled = True
        worker.pipeline_job, worker.pipeline_lease = lease, lease["token"]
        worker.ctrl_wm_frames = 3
        worker.get_one_episode()
        metadata = collected[0]["metadata"]
        self.assertEqual(metadata["termination_reason"], "success")
        self.assertFalse(metadata["budget_censored"])
        self.assertIn("submit:stats", events)
        status = owner.pipeline_status()["initial_states"]
        self.assertEqual(status["consumed_env_steps"], 12)
        self.assertEqual(status["available_env_steps"], 38)
        self.assertEqual(status["successful_episodes"], 1)

    def test_pool_and_assignment_mismatch_rejected_without_replay_insertion(self):
        owner = owner_for()
        pool = initial_state_pool_metadata(TinyExplicitEnv())
        lease = owner.acquire_real(0, pool)
        for key, change in (("initial_state_id", (lease["initial_state"]["initial_state_id"] + 1) % 50),
                            ("env_seed", lease["initial_state"]["env_seed"] + 1),
                            ("reset_env_steps", lease["initial_state"]["reset_steps"] - 1)):
            bad = episode_payload(lease)
            bad["metadata"][key] = change
            with self.subTest(field=key), self.assertRaises(ValueError):
                owner.finish_real_episode(lease["token"], bad)
            self.assertEqual(owner.initial_state_scheduler.status()["episodes_completed"], 0)
            self.assertNotIn("train-0", owner.frame_store)
        other_pool = dict(pool, sha256="c" * 64)
        with self.assertRaises(ValueError):
            owner.acquire_real(1, other_pool)
        owner.cancel_real(lease["token"])
        self.assertEqual(owner.initial_state_scheduler.status()["conservative_unknown_env_steps"], 15)

    def test_journal_restart_blocks_ambiguous_lease_and_preserves_cursor(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ACCERL_DATA_DIR": tmp}):
            pool = initial_state_pool_metadata(TinyExplicitEnv())
            first = owner_for()
            lease = first.acquire_real(0, pool)
            second = owner_for()
            with self.assertRaises(UnresolvedLeaseError):
                second.acquire_real(0, pool)
            self.assertEqual(second.initial_state_scheduler.status()["reserved_env_steps"], 15)
            self.assertEqual(second.initial_state_scheduler.status()["cursor"], 1)
            snapshot = json.loads((Path(tmp) / "initial_state_queue.json").read_text())
            self.assertEqual(snapshot["inflight"][0]["initial_state_id"], lease["initial_state"]["initial_state_id"])

    def test_budget_exhaustion_does_not_exit_producer_before_stop_signal(self):
        owner = owner_for(real_env_step_budget=11)
        pool = initial_state_pool_metadata(TinyExplicitEnv())
        lease = owner.acquire_real(0, pool)
        owner.finish_real_episode(lease["token"], episode_payload(lease, steps=1))
        events, polls = [], []

        def acquire(worker_id, state_pool):
            polls.append(1)
            if len(polls) == 2:
                owner.stop_pipeline()
            return owner.acquire_real(worker_id, state_pool)

        rpc = SimpleNamespace(acquire_real=Remote(events, "acquire_real", acquire),
                              acquire_imagine=Remote(events, "unused", lambda *a: None),
                              cancel_real=Remote(events, "cancel_real", owner.cancel_real),
                              cancel_imagine=Remote(events, "unused", lambda *a: None))
        worker, fake_ray = make_worker(events, None, rpc, mode="real")
        worker.initial_state_enabled, worker.initial_state_pool = True, pool
        worker.get_one_episode = lambda: self.fail("budget-exhausted worker started a new episode")
        with patch.dict(sys.modules, {"ray": fake_ray}), \
             patch("pipeline_runtime.time.sleep", side_effect=lambda delay: events.append("budget_poll")):
            run_producer(worker)
        self.assertEqual(len(polls), 2)
        self.assertIn("budget_poll", events)

    def test_reset_helper_explicit_id_seed_and_horizon_restoration(self):
        env = TinyExplicitEnv()
        pool = initial_state_pool_metadata(env)
        _, info = reset_with_initial_state(env, 37, 123, max_action_steps=2)
        self.assertEqual(env.selected_ids, [37])
        self.assertEqual(env.seed_calls, [123])
        self.assertEqual(env.max_episode_steps, 2)
        self.assertEqual(info["task_action_horizon"], 5)
        self.assertEqual(info["reset_env_steps"], 10)
        self.assertEqual(initial_state_pool_metadata(env), pool)
        _, other = reset_with_initial_state(env, 0, 456)
        self.assertEqual(env.selected_ids, [37, 0])
        self.assertEqual(env.max_episode_steps, 5)
        self.assertNotEqual(info["initial_state_hash"], other["initial_state_hash"])
        for index, seed, steps in ((50, 1, 2), (-1, 1, 2), (True, 1, 2), (1, -1, 2),
                                   (1, 2**32, 2), (1, 1, 0), (1, 1, 6)):
            with self.subTest(index=index, seed=seed, steps=steps), self.assertRaises(ValueError):
                reset_with_initial_state(env, index, seed, steps)


if __name__ == "__main__":
    unittest.main()
