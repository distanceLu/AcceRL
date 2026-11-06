"""Behavioral CPU tests for bounded producers and non-draining replay demand."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from pipeline_budget import (PipelineBudget, PipelineBudgetConfig, PipelineBudgetError,
                             LeaseExpiredError, uniform_seed_starts)


class PipelineBudgetTests(unittest.TestCase):
    def config(self, **changes):
        base = PipelineBudgetConfig(max_episode_frames=20, bytes_per_frame=10,
                                    max_resident_frame_bytes=2000,
                                    policy_high_watermark=8, policy_low_watermark=4,
                                    generation_per_sample=0.5, seeds_per_episode=4)
        return replace(base, **changes)

    def episode(self, budget, name="episode", worker="collector", frames=12, context=4):
        lease = budget.acquire_real(worker)
        self.assertIsNotNone(lease)
        return budget.complete_real(lease["token"], name, frames, context, frames * 10)

    def test_config_strict_and_json_roundtrip(self):
        default = PipelineBudgetConfig.from_json(None)
        self.assertIs(default, PipelineBudgetConfig.from_json(default))
        self.assertEqual(default.initial_policy_credit, 4096)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budget.json"
            path.write_text(json.dumps(default.to_dict()), encoding="utf-8")
            self.assertEqual(default, PipelineBudgetConfig.from_json(path))
        for field, value in (("generation_per_sample", float("nan")),
                             ("poll_seconds", float("inf")),
                             ("lease_timeout_seconds", 0),
                             ("max_real_in_flight", True), ("real_workers", 0),
                             ("max_episode_frames", 3.0),
                             ("replace_old_pending", 1)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                PipelineBudgetConfig.from_json({field: value})
        with self.assertRaises(ValueError):
            PipelineBudgetConfig.from_json({"typo": 4})
        with self.assertRaises(ValueError):
            self.config(policy_low_watermark=8)
        with self.assertRaises(ValueError):
            self.config(max_resident_frame_bytes=199)

    def test_uniform_starts_valid_deterministic_and_bounded(self):
        self.assertEqual(uniform_seed_starts(7, 7, 16), [])
        self.assertEqual(uniform_seed_starts(8, 7, 16), [0])
        self.assertEqual(uniform_seed_starts(20, 7, 4), [0, 4, 8, 12])
        for frames in range(0, 1026):
            starts = uniform_seed_starts(frames, 7, 16)
            self.assertEqual(starts, sorted(set(starts)))
            self.assertLessEqual(len(starts), 16)
            self.assertTrue(all(start + 7 < frames for start in starts))
            if len(starts) > 1:
                self.assertEqual(starts[-1], frames - 8)

    def test_full_replay_refills_on_consumption_with_hysteresis(self):
        budget = PipelineBudget(self.config(), policy_chunks_per_seed=2)
        self.episode(budget)
        budget.update_demand(policy_size=10000, policy_added=0, policy_sampled=0)
        for index in range(4):
            lease = budget.acquire_imagine(f"imag-{index}")
            budget.finish_imagine(lease["token"], 2)
        self.assertTrue(budget.stats()["policy_paused"])
        self.episode(budget, "second")
        self.assertIsNone(budget.acquire_imagine("waiting"))
        budget.policy_drawn(4, policy_size=10000)  # Two credits < hysteresis refill four.
        self.assertIsNone(budget.acquire_imagine("waiting"))
        budget.policy_drawn(4, policy_size=10000)
        lease = budget.acquire_imagine("waiting")
        self.assertIsNotNone(lease)
        self.assertEqual(budget.stats()["policy_size"], 10000)
        self.assertEqual(budget.stats()["policy_sampled"], 8)
        self.assertEqual(budget.stats()["policy_added"], 8)

    def test_demand_credit_cannot_be_banked_without_bound(self):
        budget = PipelineBudget(self.config(), 2)
        budget.policy_drawn(1_000_000)
        self.assertEqual(budget.stats()["available_policy_credits"], 8)
        self.episode(budget)
        lease = budget.acquire_imagine("imag")
        budget.policy_drawn(1_000_000)
        self.assertEqual(budget.stats()["available_policy_credits"], 6)
        self.assertEqual(budget.stats()["reserved_policy_samples"], 2)
        budget.finish_imagine(lease["token"], 1)
        self.assertEqual(budget.stats()["available_policy_credits"], 7)

    def test_round_robin_seeds_do_not_starve_older_episodes(self):
        budget = PipelineBudget(self.config(policy_high_watermark=100), 2)
        for name in ("A", "B", "C"):
            self.episode(budget, name)
        seen = []
        for _ in range(6):
            lease = budget.acquire_imagine("imag")
            seen.append(lease["episode_id"])
            budget.finish_imagine(lease["token"], 2)
        self.assertEqual(seen, ["A", "B", "C", "A", "B", "C"])

    def test_duplicate_worker_leases_and_double_completion_are_errors(self):
        budget = PipelineBudget(self.config(), 2)
        lease = budget.acquire_real("real")
        with self.assertRaises(PipelineBudgetError):
            budget.acquire_real("real")
        budget.cancel_real(lease["token"])
        with self.assertRaises(PipelineBudgetError):
            budget.cancel_real(lease["token"])
        self.episode(budget)
        seed = budget.acquire_imagine("imag")
        with self.assertRaises(PipelineBudgetError):
            budget.acquire_imagine("imag")
        budget.finish_imagine(seed["token"], 2)
        with self.assertRaises(PipelineBudgetError):
            budget.finish_imagine(seed["token"], 2)

    def test_flight_caps_and_actual_frame_reservation(self):
        budget = PipelineBudget(self.config(max_real_in_flight=2, max_imagine_in_flight=1), 2)
        first = budget.acquire_real("real-1")
        second = budget.acquire_real("real-2")
        self.assertIsNone(budget.acquire_real("real-3"))
        budget.complete_real(first["token"], "A", 12, 4, 120)
        seed = budget.acquire_imagine("imag-1")
        self.assertIsNone(budget.acquire_imagine("imag-2"))
        budget.cancel_imagine(seed["token"])
        self.assertIsNotNone(budget.acquire_imagine("imag-2"))
        budget.cancel_real(second["token"])

        limited = PipelineBudget(self.config(max_resident_frame_bytes=400), 2)
        lease = limited.acquire_real("first", resident_bytes=150)
        self.assertIsNotNone(lease)
        self.assertIsNone(limited.acquire_real("second", resident_bytes=150))
        limited.cancel_real(lease["token"])
        self.assertIsNone(limited.acquire_real("second", resident_bytes=201))
        self.assertIsNotNone(limited.acquire_real("second", resident_bytes=200))

    def test_real_can_replace_pending_while_imagination_demand_pauses(self):
        budget = PipelineBudget(self.config(max_pending_episodes=2), 2)
        self.episode(budget, "A")
        for _ in range(4):
            lease = budget.acquire_imagine("imag")
            budget.finish_imagine(lease["token"], 2)
        self.assertTrue(budget.stats()["policy_paused"])
        self.episode(budget, "B")
        self.episode(budget, "C")
        self.episode(budget, "D")
        self.assertEqual(budget.pinned_episode_ids(), ["C", "D"])
        self.assertEqual(budget.stats()["episodes_dropped"], 1)
        self.assertEqual(budget.stats()["seeds_dropped"], 4)
        self.assertEqual(budget.drain_released_episode_ids(), ["A", "B"])

    def test_pending_eviction_protects_slow_active_imagination(self):
        budget = PipelineBudget(self.config(max_pending_episodes=2), 2)
        self.episode(budget, "active")
        slow = budget.acquire_imagine("slow")
        self.episode(budget, "old-pending")
        self.episode(budget, "new-pending")
        self.assertIn("active", budget.pinned_episode_ids())
        self.assertEqual(budget.drain_released_episode_ids(), ["old-pending"])
        self.assertEqual(budget.drop_oldest_pending(), "new-pending")
        self.assertIsNone(budget.drop_oldest_pending())
        budget.cancel_imagine(slow["token"])
        self.assertEqual(budget.drop_oldest_pending(), "active")

    def test_all_active_slots_block_real_admission_until_release(self):
        budget = PipelineBudget(self.config(max_pending_episodes=2), 2)
        for name in ("A", "B"):
            self.episode(budget, name)
        first = budget.acquire_imagine("slow-1")
        second = budget.acquire_imagine("slow-2")
        self.assertIsNone(budget.acquire_real("real"))
        budget.cancel_imagine(first["token"])
        lease = budget.acquire_real("real")
        self.assertIsNotNone(lease)
        self.assertIn(second["episode_id"], budget.pinned_episode_ids())

    def test_episode_is_released_only_after_last_active_seed_finishes(self):
        budget = PipelineBudget(self.config(seeds_per_episode=2), 2)
        self.episode(budget)
        first = budget.acquire_imagine("one")
        second = budget.acquire_imagine("two")
        result = budget.finish_imagine(first["token"], 2)
        self.assertIsNone(result["released_episode_id"])
        self.assertEqual(budget.pinned_episode_ids(), ["episode"])
        result = budget.finish_imagine(second["token"], 1)
        self.assertEqual(result["released_episode_id"], "episode")
        self.assertEqual(budget.pinned_episode_ids(), [])
        self.assertEqual(budget.drain_released_episode_ids(), ["episode"])
        self.assertEqual(budget.drain_released_episode_ids(), [])

    def test_short_episodes_release_and_oversized_episodes_fail_without_truncation(self):
        budget = PipelineBudget(self.config(), 2)
        result = self.episode(budget, "short", frames=4, context=4)
        self.assertEqual(result["seed_count"], 0)
        self.assertEqual(budget.drain_released_episode_ids(), ["short"])
        lease = budget.acquire_real("real")
        for frames, frame_bytes in ((21, 200), (20, 201)):
            with self.assertRaises(ValueError):
                budget.complete_real(lease["token"], "oversized", frames, 4, frame_bytes)
        budget.cancel_real(lease["token"])
        self.assertEqual(budget.stats()["reserved_frame_bytes"], 0)

    def test_completion_never_exceeds_reserved_samples(self):
        budget = PipelineBudget(self.config(), 2)
        self.episode(budget)
        lease = budget.acquire_imagine("imag")
        with self.assertRaises(ValueError):
            budget.finish_imagine(lease["token"], 3)
        budget.cancel_imagine(lease["token"])
        self.assertEqual(budget.stats()["policy_added"], 0)
        self.assertEqual(budget.stats()["available_policy_credits"], 8)

    def test_stop_and_failure_clean_all_coordinator_pins_and_reservations(self):
        budget = PipelineBudget(self.config(), 2)
        self.episode(budget)
        budget.acquire_real("real")
        budget.acquire_imagine("imag")
        state = budget.stop()
        self.assertEqual(state["active_real"], 0)
        self.assertEqual(state["active_imagine"], 0)
        self.assertEqual(state["available_policy_credits"], 8)
        self.assertEqual(budget.pinned_episode_ids(), [])
        self.assertEqual(budget.drain_released_episode_ids(), ["episode"])
        self.assertIsNone(budget.acquire_real("real"))
        self.assertIsNone(budget.acquire_imagine("imag"))
        budget.fail("worker crashed")
        with self.assertRaisesRegex(PipelineBudgetError, "worker crashed"):
            budget.acquire_real("real")

    def test_stale_leases_failfast_and_release_instead_of_silent_retry(self):
        now = [0.0]
        budget = PipelineBudget(self.config(lease_timeout_seconds=3), 2, clock=lambda: now[0])
        self.episode(budget)
        budget.acquire_real("real")
        budget.acquire_imagine("imag")
        now[0] = 4
        with self.assertRaisesRegex(LeaseExpiredError, "refusing silent retry"):
            budget.check_stale_leases()
        self.assertEqual(budget.stats()["reserved_frame_bytes"], 0)
        self.assertEqual(budget.stats()["reserved_policy_samples"], 0)
        self.assertEqual(budget.drain_released_episode_ids(), ["episode"])
        with self.assertRaises(PipelineBudgetError):
            budget.acquire_imagine("retry")

    def test_counter_sync_monotonic_and_no_double_charge(self):
        budget = PipelineBudget(self.config(), 2)
        self.episode(budget)
        lease = budget.acquire_imagine("imag")
        budget.finish_imagine(lease["token"], 2)
        budget.update_demand(100, 2, 0)
        self.assertEqual(budget.stats()["available_policy_credits"], 6)
        budget.update_demand(100, 2, 4)
        self.assertEqual(budget.stats()["available_policy_credits"], 8)
        with self.assertRaises(ValueError):
            budget.update_demand(100, 1, 4)
        with self.assertRaises(ValueError):
            budget.update_demand(100, 2, 3)


if __name__ == "__main__":
    unittest.main()
