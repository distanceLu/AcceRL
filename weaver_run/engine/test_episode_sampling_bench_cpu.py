"""Regression checks for comparisons on one immutable real Replay inventory."""
import asyncio
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fresh_policy_replay import PolicySegment, SegmentReplay
import gpu_trial_replay
import paired_gpu_driver


class EpisodeBenchmarkTests(unittest.TestCase):
    def make_owner(self):
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
        return SimpleNamespace(fresh_store=store, fresh_config=cfg, fresh_admitted_chunks={'real':800},
                               fresh_critics=[None]*4, fresh_refresh_prune=True)

    def freeze(self, owner):
        return asyncio.run(gpu_trial_replay.freeze_trial(owner, 576, 'round-00', sampling=True))

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        env = patch.dict(os.environ, ACCERL_JIT_BENCH_OUTPUT=str(self.output))
        env.start()
        self.addCleanup(env.stop)

    def test_one_pool_three_draws_without_consumption_or_cross_rank_repacking(self):
        owner = self.make_owner()
        store = owner.fresh_store
        receipt = self.freeze(owner)
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

    def test_saved_inventory_reproduces_each_draw_and_four_rank_orders(self):
        owner = self.make_owner()
        live = owner.fresh_store
        live.samples['seg-3:chunk2']['uses'] = 4
        live.samples['seg-3:chunk3']['reserved'] = 1
        live.segments['seg-3'].versions[4]['policy'] = 1
        receipt = self.freeze(owner)
        self.assertIn('inventory_manifest', receipt)
        data = (self.output / receipt['inventory_manifest']).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), receipt['inventory_sha256'])
        inventory = json.loads(gzip.decompress(data))
        self.assertEqual(self.freeze(owner), receipt)
        self.assertEqual(len(list(self.output.rglob('*.gz'))), 1)

        rng = random.Random()
        version, state, gaussian = inventory['rng_state']
        rng.setstate((version, tuple(state), gaussian))
        replay = SegmentReplay(inventory['capacity'], SimpleNamespace(**inventory['config']), rng=rng)
        replay.set_versions(*inventory['versions'])
        geometry = {}
        for row in inventory['segments']:
            key = row['segment_id']
            seg = SimpleNamespace(segment_id=key, episode_id=row['episode_id'], count=row['count'],
                                  source=row['source'], versions=[{'policy': v} for v in row['policy_versions']],
                                  terminated=[row['terminated']], truncated=[row['truncated']],
                                  sample_id=lambda i, key=key: f'{key}:chunk{i}')
            replay.segments[key] = seg
            geometry[key] = (seg.count, seg.terminated[-1])
            for i in range(seg.count):
                sample = dict(segment_id=key, index=i, uses=row['uses'][i], reserved=row['reserved'][i])
                replay.samples[seg.sample_id(i)] = sample
                self.assertEqual(replay._eligible(sample), row['eligible'][i])
        self.assertEqual(list(replay.samples), list(live.samples))
        self.assertEqual(replay.samples, live.samples)
        self.assertEqual(replay.segments['seg-3'].episode_id, '3')
        self.assertEqual(replay.segments['seg-3'].versions[4]['policy'], 1)
        for mode in inventory['draw_order']:
            token, ids = replay.reserve(receipt['n'], real_sampling=mode)
            replay.cancel(token)
            draw = receipt['draws'][mode]
            self.assertEqual(ids, draw['sample_ids'])
            positions, _ = gpu_trial_replay.ordered_positions(ids, geometry,
                                      preserve_rank_shards=draw['preserve_rank_shards'])
            actual = asyncio.run(gpu_trial_replay.trial_shards(owner, 'round-00', mode))
            self.assertEqual([[ids[i] for i in rank] for rank in positions],
                             [rank['sample_ids'] for rank in actual])

    def test_measured_orders_are_balanced_and_ignore_warmup(self):
        self.assertTrue(hasattr(paired_gpu_driver, 'sampling_order'))
        orders = [tuple(paired_gpu_driver.sampling_order(i)) for i in range(1,7)]
        modes = ('chunks','whole_episodes','uniform_blocks')
        self.assertEqual(set(orders), set(itertools.permutations(modes)))


if __name__ == '__main__':
    unittest.main()
