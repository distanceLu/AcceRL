import unittest
from types import SimpleNamespace

from real_collection import RealCollectionCoordinator
from test_fresh_policy_replay_cpu import Ledger, Writer, runtime_replay_class, segment
from test_model_free_foundation_cpu import POOL, collection_config, completion


class RolloutInventoryGateTests(unittest.TestCase):
    def test_gate_stays_closed_in_middle_band_then_opens_at_exact_low_mark(self):
        replay = runtime_replay_class()(64, dict(real_only=True, real_policy_replay=True,
            real_fraction=1., replay_age_limits=True, max_policy_age=5, max_sample_uses=4))
        replay.capture_ledger, replay.telemetry = Ledger(), Writer()
        replay.configure_fresh([object()])
        replay.set_fresh_versions(4, 0)
        for version, count in ((1, 2), (2, 1), (4, 5)):
            replay.add_segment(segment(str(version), source='real', policy=version, count=count))
        self.assertFalse(any(kind == 'rollout_inventory_gate_decision' for kind, _ in replay.telemetry.events))
        calls = []
        collector = SimpleNamespace(set_rollout_gate=SimpleNamespace(remote=lambda *a: calls.append(a)))
        replay.configure_rollout_inventory_gate(collector, 4)
        self.assertTrue(calls[-1][1])
        replay.set_fresh_versions(6, 0)
        self.assertTrue(calls[-1][1])
        replay.set_fresh_versions(7, 0)
        self.assertFalse(calls[-1][1])
        snapshot = replay.telemetry.events[-1][1]['details']
        self.assertEqual(snapshot['eligible_next'], 5)
        self.assertTrue(snapshot['next_two_versions_ready'])

    def test_replay_notifies_at_hysteresis_edges_and_publication(self):
        replay = runtime_replay_class()(64, dict(real_only=True, real_policy_replay=True,
            real_fraction=1., replay_age_limits=True, max_policy_age=3, max_sample_uses=1))
        replay.capture_ledger, replay.telemetry = Ledger(), Writer()
        replay.configure_fresh([object()])
        calls = []
        collector = SimpleNamespace(set_rollout_gate=SimpleNamespace(remote=lambda *a: calls.append(a)))
        replay.configure_rollout_inventory_gate(collector, 4)
        replay.set_fresh_versions(4, 0)
        replay.add_segment(segment('r1', source='real', count=7))
        self.assertFalse(any(closed for _, closed in calls))
        replay.add_segment(segment('r2', source='real', count=1))
        self.assertTrue(calls[-1][1])
        batch = replay.admit_model_free_batch(4)
        self.assertTrue(batch['ready'])
        self.assertFalse(calls[-1][1])
        replay.cancel_model_free_admission(batch['reservation_token'])
        self.assertTrue(calls[-1][1])
        replay.set_fresh_versions(6, 0)
        self.assertFalse(calls[-1][1])
        events = [v['details'] for kind, v in replay.telemetry.events
                  if kind == 'rollout_inventory_gate_decision']
        self.assertEqual(events[-1]['eligible_next'], 8)
        self.assertFalse(events[-1]['next_two_versions_ready'])

    def test_lease_gate_preserves_inflight_pause_budget_and_stop(self):
        owner = RealCollectionCoordinator(collection_config(real_env_step_budget=30))
        first = owner.acquire_real('first', POOL)
        owner.set_rollout_gate(1, True)
        self.assertIsNone(owner.acquire_real('second', POOL))
        self.assertEqual(owner.acquire_real('first', POOL), first)
        owner.finish_real_episode(first['token'], completion(first))
        self.assertEqual(owner.pipeline_status()['budget']['rollout_gate_deferred_polls'], 1)
        owner.set_paused(True)
        owner.set_rollout_gate(3, False)
        owner.set_rollout_gate(2, True)
        self.assertIsNone(owner.acquire_real('second', POOL))
        owner.set_paused(False)
        second = owner.acquire_real('second', POOL)
        self.assertEqual(second['token'], 'real:2')
        owner.finish_real_episode(second['token'], completion(second, episode_id='other'))
        owner.set_rollout_gate(4, True)
        self.assertTrue(owner.acquire_real('third', POOL)['budget_exhausted'])
        owner.stop_pipeline()
        self.assertTrue(owner.acquire_real('third', POOL)['stopped'])
