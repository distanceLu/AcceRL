"""Regression checks for comparisons on one immutable real Replay inventory."""
import asyncio
import itertools
import random
import unittest
from types import SimpleNamespace

import numpy as np

from fresh_policy_replay import PolicySegment, SegmentReplay
import gpu_trial_replay
import paired_gpu_driver


class EpisodeBenchmarkTests(unittest.TestCase):
    def test_one_pool_three_draws_without_consumption_or_cross_rank_repacking(self):
        cfg = SimpleNamespace(real_only=True, real_policy_replay=True, refresh_gae=True,
                              replay_age_limits=True, max_policy_age=16, max_sample_uses=4)
        store = SegmentReplay(1800, cfg, rng=random.Random(71))
        store.set_versions(0, 0)
        for index in range(100):
            n = 8
            s = PolicySegment.create(segment_id=f'seg-{index}', episode_id=str(index), source='real',
                observations=[{'pixel':j} for j in range(n+1)],
                rewards=np.zeros(n), action_tokens=np.zeros((n,8,7),np.int64),
                behaviour_logits=np.zeros((n,8,7,3)), action_mask=np.ones((n,8,7),bool),
                terminated=[False]*(n-1)+[True], truncated=[False]*n,
                versions=[{'policy':0}]*n, generation_values=[0.]*(n+1), gamma=.99, lambda_=.95)
            store.add(s)
        owner = SimpleNamespace(fresh_store=store, fresh_config=cfg, fresh_admitted_chunks={'real':800})
        self.assertTrue(hasattr(gpu_trial_replay, 'freeze_sampling_round'))
        receipt = gpu_trial_replay.freeze_sampling_round(owner, 576, 'round-00')
        self.assertTrue(receipt['ready'])
        self.assertFalse(store.reservations)
        self.assertTrue(all(not x['reserved'] and not x['uses'] for x in store.samples.values()))
        self.assertEqual({len(set(x['sample_ids'])) for x in receipt['draws'].values()}, {576})
        self.assertEqual(len({x['sample_ids_sha256'] for x in receipt['draws'].values()}), 3)
        # A later logical version of the same frozen weights must not cause a redraw.
        store.set_versions(1, 0)
        shards = asyncio.run(gpu_trial_replay.trial_shards(owner, 'round-00', 'whole_episodes'))
        self.assertEqual([r['version'] for r in shards], [1]*4)
        ids = receipt['draws']['whole_episodes']['sample_ids']
        for rank, shard in enumerate(shards):
            self.assertEqual(set(shard['sample_ids']), set(ids[rank*144:(rank+1)*144]))
            self.assertEqual(sum(shard['bundle_context_counts']), 0)
        self.assertTrue(gpu_trial_replay.release_trial(owner, 'round-00'))

    def test_measured_orders_are_balanced_and_ignore_warmup(self):
        self.assertTrue(hasattr(paired_gpu_driver, 'sampling_order'))
        orders = [tuple(paired_gpu_driver.sampling_order(i)) for i in range(1,7)]
        modes = ('chunks','whole_episodes','uniform_blocks')
        self.assertEqual(set(orders), set(itertools.permutations(modes)))


if __name__ == '__main__':
    unittest.main()
