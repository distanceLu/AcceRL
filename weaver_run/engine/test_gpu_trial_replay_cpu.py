import asyncio
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fresh_policy_replay import PolicySegment, SegmentReplay
from gpu_trial_replay import freeze_trial, release_trial, trial_shards
from jit_gae_plan import build_plan


class FrozenTrialTests(unittest.TestCase):
    def test_persist_both_arms_and_real_suffix_on_the_same_draw(self):
        async def exercise(directory):
            cfg = SimpleNamespace(real_only=True, real_policy_replay=True, refresh_gae=True,
                                  replay_age_limits=True, max_policy_age=16, max_sample_uses=4)
            store = SegmentReplay(1800,cfg,rng=random.Random(19))
            store.set_versions(0,0)
            for k in range(160):
                n=8
                segment=PolicySegment.create(segment_id=f'real-{k}',episode_id=str(k),source='real',
                    observations=[{'pixel':j+.1*k} for j in range(n+1)],
                    rewards=np.linspace(0,1,n),action_tokens=np.zeros((n,8,7),np.int64),
                    behaviour_logits=np.zeros((n,8,7,3)), action_mask=np.ones((n,8,7),bool),
                    terminated=[False]*(n-1)+[k%2==0],truncated=[False]*(n-1)+[k%2==1],
                    versions=[{'policy':0}]*n,generation_values=[-9.]*(n+1),gamma=.99,lambda_=.95)
                store.add(segment)

            async def remote(inputs, version, batch_sizes):
                self.assertEqual(sum(batch_sizes),len(inputs))
                return dict(policy_version=version,values=[row['pixel'] for row in inputs])

            critic=SimpleNamespace(autotuned_evaluate_values=SimpleNamespace(remote=remote))
            owner=SimpleNamespace(fresh_store=store,fresh_config=cfg,fresh_critics=[critic]*4,
                                  fresh_admitted_chunks={'real':1280},fresh_refresh_prune=True)
            capture=await freeze_trial(owner,576,'pair-00')
            manifest=json.loads(Path(capture['manifest']).read_text())
            sid=manifest['sample_ids'][0].rsplit(':chunk',1)[0]
            data=torch.load(Path(capture['manifest']).parent/manifest['segments'][sid]['file'],weights_only=False)
            self.assertEqual(data['observations'],store.segments[sid].observations)
            np.testing.assert_array_equal(data['behaviour_logits'],store.segments[sid].behaviour_logits)
            self.assertFalse(store.reservations)
            raw=await trial_shards(owner,'pair-00','pruned')
            store.set_versions(1,0)
            jit=await trial_shards(owner,'pair-00','jit')
            original=[i for rank in raw for i in rank['sample_ids']]
            self.assertEqual(original,manifest['sample_ids'])
            self.assertEqual(set(original),{i for rank in jit for i in rank['sample_ids']})
            self.assertTrue(all(r['version']==0 and len(r['targets'])==144 for r in raw))
            seen=set()
            for r in jit:
                self.assertEqual((r['version'],r['freeze_version'],len(r['sample_ids'])),(1,0,144))
                segments,bundles=build_plan(r['observations'],16,1)
                self.assertFalse(seen.intersection(segments))
                seen.update(segments)
                self.assertEqual(r['bundle_context_counts'],[len(b.nodes)-len(b.selected) for b in bundles])
                for b in bundles:
                    for node in b.nodes:
                        self.assertIn('pixel',segments[node.segment]['observations'][node.index])
            self.assertTrue(release_trial(owner,'pair-00'))
            self.assertFalse(release_trial(owner,'pair-00'))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,ACCERL_JIT_BENCH_OUTPUT=directory):
            asyncio.run(exercise(directory))


if __name__ == '__main__':
    unittest.main()
