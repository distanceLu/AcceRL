import asyncio
import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gpu_trial_placement import ordered_positions
from jit_gae_plan import build_plan
from test_fresh_policy_replay_cpu import Ledger, Writer, runtime_replay_class
from test_model_free_critic_pruning_cpu import real_segment
from training_topology import GlobalBatchCoordinator


class TrajectoryReplayTests(unittest.TestCase):
    def test_episode_protocol_preserves_admitted_rank_members_and_full_suffix_geometry(self):
        for requested in ('whole_episodes', 'uniform_blocks'):
            with self.subTest(requested=requested), patch.dict(os.environ,
                    ACCERL_TRAJECTORY_JIT='1', ACCERL_REAL_SAMPLING=requested):
                replay = runtime_replay_class()(2400, dict(real_only=True, real_fraction=1.,
                    real_policy_replay=True, refresh_gae=True, signal_telemetry=True))
                replay.capture_ledger, replay.telemetry = Ledger(), Writer()
                replay.configure_fresh([object()]*4)
                replay.set_fresh_versions(5, 0)
                for episode in range(83):
                    replay.add_segment(real_segment(episode, 9))
                replay.fresh_store.rng = random.Random(17)
                admission = replay.admit_model_free_batch(576)
                draw = replay._model_free_admission['sample_ids']
                shards = asyncio.run(replay.trajectory_jit_shards(576))
                placement = next(row['details'] for kind, row in replay.telemetry.events
                                 if kind == 'trajectory_jit_placement')
                self.assertEqual(placement['real_sampling']['actual'], requested)
                for rank, shard in enumerate(shards):
                    observations, _, _, _, _, meta = shard
                    self.assertEqual(meta['real_sampling'], requested)
                    self.assertEqual(set(meta['sample_ids']), set(draw[144*rank:144*(rank+1)]))
                    segments, bundles = build_plan(observations, 16, 5)
                    contexts = [sum(n.selected is None for n in b.nodes) for b in bundles]
                    self.assertEqual(placement['context_by_rank_bundle'][rank], contexts)
                    if requested == 'whole_episodes':
                        self.assertEqual(sum(contexts), sum(not s['terminated'][-1] for s in segments.values()))
                self.assertEqual(admission['reservation_token'], shards[0][-1]['fresh_reservation_token'])

    def test_admission_placement_receipts_and_exact_uses(self):
        with patch.dict(os.environ, ACCERL_TRAJECTORY_JIT='1'):
            replay = runtime_replay_class()(2400, dict(real_only=True, real_fraction=1., real_policy_replay=True,
                                                       refresh_gae=True, signal_telemetry=True))
        replay.capture_ledger, replay.telemetry = Ledger(), Writer()
        replay.configure_fresh([object()] * 4)
        replay.set_fresh_versions(5, 0)
        for episode in range(100):
            replay.add_segment(real_segment(episode, 9))
        replay.fresh_store.rng = random.Random(11)
        token = replay.admit_model_free_batch(576)['reservation_token']
        original = list(replay._model_free_admission['sample_ids'])
        original_denom = {}
        for start in range(0, 576, 8):
            group = original[start:start+8]
            count = sum(1 if sid.endswith(':chunk8') else 2 for sid in group)
            original_denom.update(dict.fromkeys(group, count))
        shards = asyncio.run(replay.trajectory_jit_shards(576))
        self.assertEqual(set(original), {sid for shard in shards for sid in shard[-1]['sample_ids']})
        seen = set()
        for rank in range(4):
            obs, _, _, _, _, metadata = shards[rank]
            self.assertTrue(metadata['jit_pending'])
            ids = metadata['sample_ids']
            self.assertEqual(ids, [row['_jit_sample_id'] for row in obs])
            self.assertEqual([original_denom[sid] for sid in ids], [row['_jit_original_denominator'] for row in obs])
            segments, bundles = build_plan(obs, 16, 5)
            self.assertEqual(len(bundles), 9)
            self.assertFalse(seen.intersection(segments))
            seen.update(segments)
            receipts = [dict(sample_id=sid, advantage=1., value_target=2., refresh_policy_version=5)
                        for sid in reversed(ids)]
            with self.assertRaises(ValueError):
                replay.commit_fresh_use(token, ids, receipts[:-1])
            self.assertTrue(replay.commit_fresh_use(token, ids, receipts))
        self.assertFalse(replay.fresh_store.reservations)
        self.assertFalse(replay.fresh_records)
        self.assertEqual(sum(s['uses'] for s in replay.fresh_store.samples.values()), 576)
        signals = [event['details'] for kind, event in replay.telemetry.events if kind == 'fresh_policy_signal']
        self.assertEqual(len(signals), 1152)
        self.assertTrue(all(r['advantage']==1 and r['value_target']==2 for r in signals))

    def test_oversize_trajectory_still_covers_four_exact_shards(self):
        ids = [f'one:chunk{i}' for i in range(576)]
        positions, counts = ordered_positions(ids, {'one': (576, True)})
        self.assertEqual([len(row) for row in positions], [144] * 4)
        self.assertEqual(sorted(sum(positions, [])), list(range(576)))
        self.assertEqual([len(row) for row in counts], [9] * 4)

    def test_coordinator_hands_out_refs_once_per_draw(self):
        states = []
        class Sender:
            def options(self, *, num_returns):
                self.count = num_returns
                return self
            def remote(self, size):
                states.append((self.count, size))
                return [object() for _ in range(self.count)]
        with patch.dict(os.environ, ACCERL_TRAJECTORY_JIT='1'):
            coordinator = GlobalBatchCoordinator(SimpleNamespace(trajectory_jit_shards=Sender()),
                                                 None, 4, wm_consumers=())
        async def run_draws():
            for _ in range(2):
                refs = await asyncio.gather(*(coordinator.sample_policy(i, 144) for i in (2,1,3,0)))
                self.assertEqual(len({r['trajectory_jit_shard_ref'] for r in refs}), 4)
        asyncio.run(run_draws())
        self.assertEqual(states, [(4,576)] * 2)


if __name__ == '__main__':
    unittest.main()
