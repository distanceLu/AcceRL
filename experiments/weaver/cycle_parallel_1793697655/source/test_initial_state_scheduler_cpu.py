"""CPU-only behavior tests for global initial states and exact step budgeting."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from initial_state_scheduler import (
    BudgetExceededError, CheckpointMismatchError, InitialStateScheduler,
    InitialStateSchedulerError, UnresolvedLeaseError,
)


class InitialStateSchedulerTests(unittest.TestCase):
    def finish(self, queue, lease, action_steps=1, success=True):
        return queue.finish(lease["token"], action_steps=action_steps,
                            reset_steps=lease["reset_steps"], success=success)

    def test_worker_count_does_not_change_global_permutation(self):
        one = InitialStateScheduler(max_env_steps=100000)
        many = InitialStateScheduler(max_env_steps=100000)
        one_ids, many_ids = [], []
        for index in range(150):
            a = one.acquire("one")
            b = many.acquire(f"worker-{index % 10}")
            one_ids.append(a["initial_state_id"])
            many_ids.append(b["initial_state_id"])
            self.assertEqual(a["env_seed"], b["env_seed"])
            self.finish(one, a)
            self.finish(many, b)
        self.assertEqual(one_ids, many_ids)
        for start in (0, 50, 100):
            self.assertEqual(sorted(one_ids[start:start + 50]), list(range(50)))
        self.assertNotEqual(one_ids[:50], one_ids[50:100])
        self.assertEqual(many.status()["completed_coverage_count"], 50)

    def test_permutation_independent_of_completion_order(self):
        queue = InitialStateScheduler(state_count=7, max_env_steps=10000)
        leases = [queue.acquire(f"worker-{i}") for i in range(7)]
        self.assertEqual(len({lease["initial_state_id"] for lease in leases}), 7)
        for lease in reversed(leases):
            self.finish(queue, lease)
        self.assertEqual(queue.status()["episodes_completed"], 7)
        self.assertEqual(queue.status()["completed_initial_state_ids"], list(range(7)))
        next_lease = queue.acquire("worker-0")
        self.assertEqual((next_lease["cycle"], next_lease["index"]), (1, 0))

    def test_exact_11000_includes_reset_and_partial_final_episode(self):
        queue = InitialStateScheduler()
        lengths = []
        while (lease := queue.acquire("collector")) is not None:
            lengths.append(lease["max_action_steps"])
            self.finish(queue, lease, action_steps=lease["max_action_steps"], success=False)
        self.assertEqual(lengths, [220] * 47 + [180])
        status = queue.status()
        self.assertEqual(status["known_action_steps"], 10520)
        self.assertEqual(status["known_reset_steps"], 480)
        self.assertEqual(status["consumed_env_steps"], 11000)
        self.assertEqual(status["available_env_steps"], 0)
        self.assertEqual(status["completed_coverage_count"], 48)
        self.assertEqual(len(status["uncompleted_initial_state_ids"]), 2)

    def test_concurrent_reservations_never_oversubscribe_and_refund(self):
        queue = InitialStateScheduler(max_env_steps=500)
        a, b, c = (queue.acquire(worker) for worker in ("a", "b", "c"))
        self.assertEqual([a["reserved_env_steps"], b["reserved_env_steps"], c["reserved_env_steps"]],
                         [230, 230, 40])
        self.assertIsNone(queue.acquire("d"))
        self.assertEqual(queue.status()["committed_env_steps"], 500)
        self.finish(queue, a, action_steps=20)
        self.assertEqual(queue.status()["available_env_steps"], 200)
        d = queue.acquire("d")
        self.assertEqual(d["max_action_steps"], 190)
        for lease in (b, c, d):
            self.finish(queue, lease, action_steps=lease["max_action_steps"])
        self.assertEqual(queue.status()["consumed_env_steps"], 500)

    def test_reload_preserves_order_rng_and_ambiguous_reservations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            queue = InitialStateScheduler(state_count=5, checkpoint_path=path)
            for _ in range(7):
                self.finish(queue, queue.acquire("collector"))
            abandoned = queue.acquire("crashed")
            control = InitialStateScheduler.from_snapshot(queue.snapshot())
            restored = InitialStateScheduler(state_count=5, checkpoint_path=path)
            self.assertEqual(restored.status()["reserved_env_steps"], 230)
            self.assertEqual(restored.status()["recovery_required_tokens"], [abandoned["token"]])
            with self.assertRaises(UnresolvedLeaseError):
                restored.acquire("crashed")
            with self.assertRaises(UnresolvedLeaseError):
                restored.finish(abandoned["token"], action_steps=1, reset_steps=10, success=True)
            restored.recover(abandoned["token"])
            control.recover(abandoned["token"])
            self.assertEqual(restored.status()["conservative_unknown_env_steps"], 230)
            for _ in range(12):
                a, b = restored.acquire("collector"), control.acquire("collector")
                self.assertEqual((a["initial_state_id"], a["env_seed"], a["cycle"], a["index"]),
                                 (b["initial_state_id"], b["env_seed"], b["cycle"], b["index"]))
                self.finish(restored, a)
                self.finish(control, b)
            disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(disk["events"]), 2 * len(disk["allocations"]))
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_idempotent_acquire_and_completion_cannot_double_spend(self):
        queue = InitialStateScheduler(max_env_steps=300)
        a = queue.acquire("worker")
        self.assertEqual(a, queue.acquire("worker"))
        self.assertEqual(queue.status()["episodes_assigned"], 1)
        result = queue.finish(a["token"], action_steps=20, reset_steps=10, success=True, episode_id="e1")
        self.assertEqual(result, queue.finish(a["token"], action_steps=20, reset_steps=10,
                                             success=True, episode_id="e1"))
        self.assertEqual(queue.status()["consumed_env_steps"], 30)
        with self.assertRaises(InitialStateSchedulerError):
            queue.finish(a["token"], action_steps=21, reset_steps=10, success=True)
        with self.assertRaises(InitialStateSchedulerError):
            queue.cancel(a["token"])

    def test_unknown_cancellation_charges_full_known_cancel_refunds(self):
        queue = InitialStateScheduler(max_env_steps=500)
        a = queue.acquire("a")
        result = queue.cancel(a["token"], reason="worker_lost")
        self.assertEqual(result["charged_env_steps"], 230)
        self.assertEqual(queue.status()["conservative_unknown_env_steps"], 230)
        self.assertEqual(result, queue.cancel(a["token"], reason="worker_lost"))
        b = queue.acquire("b")
        queue.cancel(b["token"], action_steps=0, reset_steps=0, reason="never_started")
        self.assertEqual(queue.status()["consumed_env_steps"], 230)
        self.assertEqual(queue.status()["available_env_steps"], 270)
        self.assertNotEqual(b["token"], queue.acquire("b")["token"])
        self.assertEqual(queue.status()["assigned_coverage_count"], 3)
        self.assertEqual(queue.status()["completed_coverage_count"], 0)

    def test_overspend_invalid_inputs_and_coverage_report(self):
        queue = InitialStateScheduler(state_count=3, max_env_steps=230)
        a = queue.acquire("worker")
        before = queue.snapshot()
        for actions, reset in ((221, 10), (220, 11)):
            with self.assertRaises(BudgetExceededError):
                queue.finish(a["token"], action_steps=actions, reset_steps=reset, success=True)
        for actions, reset in ((-1, 10), (True, 10), (1.0, 10), (1, None)):
            with self.assertRaises(ValueError):
                queue.cancel(a["token"], action_steps=actions, reset_steps=reset)
        with self.assertRaises(ValueError):
            queue.finish(a["token"], action_steps=2, reset_steps=10, success=1)
        with self.assertRaises(InitialStateSchedulerError):
            queue.finish("unknown", action_steps=0, reset_steps=0, success=False)
        self.assertEqual(before, queue.snapshot())
        status = queue.status()
        self.assertEqual(status["assigned_not_completed_initial_state_ids"], [a["initial_state_id"]])
        self.assertEqual(len(status["unassigned_initial_state_ids"]), 2)

    def test_rejects_mismatched_config_asset_and_corrupt_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            queue = InitialStateScheduler(checkpoint_path=path, state_asset_sha256="a" * 64)
            queue.acquire("worker")
            for change in (dict(seed=1), dict(state_count=49), dict(max_env_steps=100),
                           dict(state_asset_sha256="b" * 64)):
                config = dict(checkpoint_path=path, state_asset_sha256="a" * 64)
                config.update(change)
                with self.assertRaises(CheckpointMismatchError):
                    InitialStateScheduler(**config)
            malformed = deepcopy(queue.snapshot())
            malformed["allocations"][0]["initial_state_id"] = 50
            with self.assertRaises(CheckpointMismatchError):
                InitialStateScheduler.from_snapshot(malformed)
            malformed = deepcopy(queue.snapshot())
            malformed["permutation"][0] = malformed["permutation"][1]
            with self.assertRaises(CheckpointMismatchError):
                InitialStateScheduler.from_snapshot(malformed)

    def test_snapshot_detached_and_global_rng_untouched(self):
        random.seed(17)
        before = random.getstate()
        queue = InitialStateScheduler()
        lease = queue.acquire("worker")
        view = queue.snapshot()
        view["inflight"][0]["initial_state_id"] = 1234
        lease["initial_state_id"] = 4567
        self.assertLess(queue.snapshot()["inflight"][0]["initial_state_id"], 50)
        self.assertEqual(random.getstate(), before)

    def test_explicit_recovery_with_exact_completion_refunds(self):
        queue = InitialStateScheduler()
        lease = queue.acquire("worker")
        restored = InitialStateScheduler.from_snapshot(queue.snapshot())
        result = restored.recover(lease["token"], mode="finish", action_steps=30, reset_steps=10,
                                  success=True, episode_id="reconciled")
        self.assertEqual(result["refunded_env_steps"], 190)
        self.assertEqual(restored.status()["consumed_env_steps"], 40)
        self.assertEqual(restored.status()["completed_initial_state_ids"], [lease["initial_state_id"]])
        self.assertNotEqual(restored.acquire("worker")["token"], lease["token"])

    def test_tiny_remaining_budget_never_executes_reset_alone(self):
        queue = InitialStateScheduler(max_env_steps=10)
        self.assertIsNone(queue.acquire("worker"))
        self.assertEqual(queue.status()["available_env_steps"], 10)
        lease = queue.acquire("worker", max_action_steps=50, reset_steps=0)
        self.assertEqual(lease["max_action_steps"], 10)
        self.finish(queue, lease, action_steps=10)
        self.assertIsNone(queue.acquire("worker", reset_steps=0))

    def test_atomic_write_failure_retry_cannot_hand_out_unjournaled_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            queue = InitialStateScheduler(checkpoint_path=path)
            with patch("initial_state_scheduler.os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    queue.acquire("worker")
            self.assertEqual(json.loads(path.read_text())["allocations"], [])
            lease = queue.acquire("worker")
            self.assertEqual(len(json.loads(path.read_text())["allocations"]), 1)
            with patch("initial_state_scheduler.os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    queue.finish(lease["token"], action_steps=20, reset_steps=10, success=True)
            self.assertEqual(json.loads(path.read_text())["terminal"], {})
            queue.finish(lease["token"], action_steps=20, reset_steps=10, success=True)
            self.assertIn(lease["token"], json.loads(path.read_text())["terminal"])
            self.assertFalse(list(Path(tmp).glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
