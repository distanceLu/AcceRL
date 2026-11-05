"""CPU checks for ordered targets, fresh cohort leases, and runtime wiring."""
from __future__ import annotations

import ast
import asyncio
from collections import deque
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np

from fresh_config import FreshConfig
from fresh_policy_replay import PolicySegment, SegmentReplay, FreshReplayNotReady, generalized_advantages
from fresh_signal import record_signal


def segment(name='imag', source='imagined', policy=4, world=1, count=2, terminal=False):
    return PolicySegment.create(segment_id=name,episode_id='ep',source=source,
        observations=[{'value':float(i)} for i in range(count+1)],
        action_tokens=np.zeros((count,2,1),np.int64),behaviour_logits=np.zeros((count,2,1,3),np.float32),
        rewards=np.arange(1,count+1),generation_values=np.zeros(count+1),
        terminated=[False]*(count-1)+[terminal],truncated=[False]*(count-1)+[not terminal],
        versions=[{'policy':policy, **({'world_bundle_version':world,'world_reward_bundle_version':world}
                                     if source=='imagined' else {})} for _ in range(count)],
        gamma=.9,lambda_=.8)


class Ledger:
    def __init__(self):self.ids=set()
    def add(self,sid,*args,**kwargs):self.ids.add(sid)
    def evict(self,sid,*args):self.ids.remove(sid)
    def reserve(self,ids):return {'reservation_id':'ledger-reservation:one','sample_ids':ids,
                                 'replay':'policy','reserved_ns':123,'consumer':'training'}


class Writer:
    def __init__(self):self.events=[]
    def event(self,kind,**kwargs):self.events.append((kind,kwargs))


def runtime_replay_class():
    tree=ast.parse((Path(__file__).parent/'train.py').read_text(encoding='utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReplayBufferActor')
    node.decorator_list=[]
    ns=dict(np=np,asyncio=asyncio,deque=deque,random=random,FreshConfig=FreshConfig,
            SegmentReplay=SegmentReplay,FreshReplayNotReady=FreshReplayNotReady,
            setup_debugger=lambda *args:None,record_signal=record_signal)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<actual ReplayBufferActor>','exec'),ns)
    return ns['ReplayBufferActor']


class PureReplayTests(unittest.TestCase):
    def test_terminal_and_truncation_have_different_bootstrap(self):
        terminal,_=generalized_advantages([1.],[2.,10.],[True],[False],.9,.95)
        truncated,_=generalized_advantages([1.],[2.,10.],[False],[True],.9,.95)
        self.assertEqual(float(terminal[0]),-1.)
        self.assertEqual(float(truncated[0]),8.)

    def test_refresh_preserves_rewards_and_behaviour_distribution(self):
        replay=SegmentReplay(8,FreshConfig(refresh_gae=True),rng=random.Random(1))
        item=segment();replay.add(item);replay.set_versions(5,1)
        token,ids=replay.reserve(2)
        old,_=replay.materialize(ids)
        new,records=replay.materialize(ids,{'imag':[2.,3.,4.]})
        self.assertFalse(np.array_equal(old[2],new[2]))
        np.testing.assert_array_equal(old[3],new[3])
        np.testing.assert_array_equal(item.rewards,[1,2])
        self.assertTrue(all(not record['rewards_refreshed'] for record in records))
        replay.cancel(token)

    def test_age_and_use_caps_never_fall_back(self):
        replay=SegmentReplay(8,FreshConfig(replay_age_limits=True,max_policy_age=2,max_world_age=1,max_sample_uses=1))
        replay.add(segment());replay.set_versions(5,2)
        token,ids=replay.reserve(2)
        with self.assertRaises(FreshReplayNotReady):replay.reserve(1)
        replay.commit(token,ids[:1]);replay.commit(token,ids[1:])
        with self.assertRaises(FreshReplayNotReady):replay.reserve(1)
        with self.assertRaises(ValueError):replay.commit(token,ids)
        replay.add(segment('old',policy=0,world=0))
        with self.assertRaises(FreshReplayNotReady):replay.reserve(1)

    def test_whole_segments_evicted_and_inflight_refs_remain_pinned(self):
        replay=SegmentReplay(2,FreshConfig())
        replay.add(segment());replay.set_versions(5,1)
        token,ids=replay.reserve(1)
        with self.assertRaises(FreshReplayNotReady):replay.add(segment('second'))
        replay.cancel(token);replay.add(segment('second'))
        self.assertEqual(list(replay.segments),['second'])

    def test_mixed_sources_have_fixed_quota_and_no_substitution(self):
        replay=SegmentReplay(16,FreshConfig(real_policy_replay=True,real_fraction=.25),rng=random.Random(4))
        replay.add(segment('imag',count=6));replay.set_versions(5,1)
        with self.assertRaises(FreshReplayNotReady):replay.reserve(4)
        replay.add(segment('real',source='real'))
        token,ids=replay.reserve(4)
        sources=[replay.segments[replay.samples[sid]['segment_id']].source for sid in ids]
        self.assertEqual(sources.count('real'),1)
        replay.cancel(token)

    def test_partial_real_chunk_mask_and_immutable_tokens(self):
        item=segment(source='real')
        with self.assertRaises(ValueError):item.action_tokens[0,0,0]=1
        values={**item.__dict__,'action_mask':np.array([[[True],[False]],[[True],[True]]])}
        item=PolicySegment(**values)
        replay=SegmentReplay(8,FreshConfig());replay.add(item);replay.set_versions(5,1)
        result,_=replay.materialize([item.sample_id(0)])
        self.assertEqual(result[0][0]['_fresh_action_mask'].tolist(),[[True],[False]])


class ActualReplayRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_actor_refreshes_with_exact_version_and_records_commit(self):
        calls=[]
        async def evaluate(inputs,version):
            calls.append((len(inputs),version));return {'policy_version':version,'values':[obs['value']+2 for obs in inputs]}
        replay=runtime_replay_class()(8,FreshConfig(refresh_gae=True,replay_age_limits=True,signal_telemetry=True).to_dict())
        replay.capture_ledger=Ledger();replay.telemetry=Writer()
        replay.configure_fresh([SimpleNamespace(evaluate_values=SimpleNamespace(remote=evaluate))])
        replay.set_fresh_versions(5,1);replay.add_segment(segment())
        result=await replay.sample(2)
        self.assertEqual(calls,[(3,5)])
        self.assertEqual(result[-1]['reservation_id'],'ledger-reservation:one')
        self.assertTrue(result[-1]['fresh_reservation_token'].startswith('fresh-reservation:'))
        self.assertNotEqual(result[-1]['reservation_id'],result[-1]['fresh_reservation_token'])
        self.assertEqual(result[-1]['source_counts'],{'real':0,'imagined':2})
        replay.commit_fresh_use(result[-1]['fresh_reservation_token'],result[-1]['sample_ids'])
        consumed=[kwargs['details'] for kind,kwargs in replay.telemetry.events
                  if kind=='fresh_policy_signal' and kwargs['details']['phase']=='consumed']
        self.assertEqual(len(consumed),2)
        self.assertTrue(all(row['refresh_policy_version']==5 and not row['rewards_refreshed'] for row in consumed))

    async def test_actual_actor_yields_under_shortage_and_accepts_new_segments(self):
        replay=runtime_replay_class()(8,FreshConfig(replay_age_limits=True).to_dict())
        replay.capture_ledger=Ledger();replay.telemetry=Writer()
        replay.configure_fresh([object()]);replay.set_fresh_versions(5,1)
        pending=asyncio.create_task(replay.sample(2))
        await asyncio.sleep(.01);self.assertFalse(pending.done())
        replay.add_segment(segment())
        result=await asyncio.wait_for(pending,.3)
        self.assertEqual(len(result[0]),2)

    async def test_disabled_actual_actor_keeps_frozen_legacy_targets(self):
        replay=runtime_replay_class()(8)
        row=SimpleNamespace(obs={'x':1},action_token=np.zeros((2,1)),advantage=9.,
                            behaviour_logits=np.zeros((2,1,3)),value_target=7.)
        replay.add_batch([row]);result=await replay.sample(1)
        self.assertEqual(float(result[2][0]),9.);self.assertEqual(float(result[4][0]),7.)


if __name__=='__main__':unittest.main()
