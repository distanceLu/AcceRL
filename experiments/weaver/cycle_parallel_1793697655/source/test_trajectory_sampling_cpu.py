"""Whole-segment admission, actual transport/planner and CPU JIT regressions."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay, FreshReplayNotReady, generalized_advantages
from fresh_replay_pressure import shortage
from inventory_lookahead import forecast
from cost_scheduler_runtime import replay_inventory
from cycle_service_controller import InventoryBucket, CycleSnapshot, CycleServiceController
from jit_gae_plan import build_plan
from test_fresh_policy_replay_cpu import segment, runtime_replay_class, Ledger, Writer
from test_jit_gae_cpu import ToyModel, owner_for, execute
from trajectory_sampling import validate_rank


def config():
    return FreshConfig(refresh_gae=True, replay_age_limits=True, real_policy_replay=True,
        jit_gae_mode='fused', advantage_norm='ema_zscore', policy_sampling='imagined_trajectories')


def replay(imagined=120, *, fragmented=False):
    r = SegmentReplay(10000, config(), rng=random.Random(71))
    for i in range(28):
        r.add(segment('r'+str(i),'real',count=8+i%21,terminal=bool(i%2)))
    for i in range(imagined):
        s=segment('i'+str(i), count=4, terminal=bool(i%3==0))
        r.add(s)
        if fragmented:r.samples[s.sample_id(3)]['uses']=4
    r.set_versions(5,1)
    return r


def owner_for_rank(r, ids, rank, micro=8):
    batch, records=r.materialize(ids,jit=True)
    owner=owner_for(ToyModel(),batch,ids,rank=rank,micro=micro)
    owner.fresh_config=config();owner.world_size=4
    return owner,batch,records


class SamplingTests(unittest.TestCase):
    def test_global_and_rank_quotas_intact_and_no_imagined_duplicate_context(self):
        r=replay();token,ids=r.reserve(576)
        self.assertEqual(len(set(ids)),576)
        imagined_owners={}
        for rank in range(4):
            shard=ids[144*rank:144*(rank+1)]
            batch,records=r.materialize(shard,jit=True)
            validate_rank(batch[0],rank,4)
            self.assertEqual(Counter(x['source'] for x in records),{'real':36,'imagined':108})
            for micro in (4,8):
                segments,bundles=build_plan(batch[0],micro,5)
                self.assertEqual(sum(len(b.selected) for b in bundles),144)
                for name,s in segments.items():
                    if s['source']=='imagined':
                        self.assertEqual([n.index for b in bundles for n in b.nodes if n.segment==name and not n.bootstrap],[3,2,1,0])
                        self.assertEqual(sum(n.bootstrap for b in bundles for n in b.nodes if n.segment==name),int(not s['terminated'][-1]))
                        if micro==4:
                            self.assertNotIn(name,imagined_owners);imagined_owners[name]=rank
            r.commit(token,shard)
        self.assertEqual(len(imagined_owners),108)
        self.assertFalse(r.reservations)
        self.assertEqual(sum(v['uses'] for v in r.samples.values()),576)

    def test_real_sampling_exact_same_first_uniform_draw(self):
        r=replay();eligible=[sid for sid,s in r.samples.items() if r.segments[s['segment_id']].source=='real' and r._eligible(s)]
        expected=random.Random(71).sample(eligible,144)
        _,ids=r.reserve(576)
        self.assertEqual(set(expected),{sid for sid in ids if sid.startswith('r')})

    def test_fragmented_chunks_are_neither_ready_nor_macro_supply(self):
        r=replay(160,fragmented=True)
        self.assertEqual(sum(r._eligible(s) for s in r.samples.values() if s['segment_id'].startswith('i')),480)
        before=deepcopy(r.samples);rng=r.rng.getstate()
        with self.assertRaises(FreshReplayNotReady):r.reserve(576)
        self.assertEqual(before,r.samples);self.assertEqual(rng,r.rng.getstate())
        self.assertEqual(shortage(r,576)['missing']['imagined'],432)
        self.assertEqual(forecast(r,2)['new_chunks_needed'],432)
        inv=replay_inventory(r)
        snap=CycleSnapshot(1,5,1,tuple(InventoryBucket(**b) for b in inv['inventory']))
        d=CycleServiceController().demands(snap)
        self.assertEqual(d['eligible_distinct_chunks']['imagined'],0)
        self.assertEqual(d['next_batch_missing']['imagined'],432)

    def test_partial_commit_rejected_without_mutation_cancel_releases(self):
        r=replay();token,ids=r.reserve(576);sid=next(i for i in ids if i.startswith('i'))
        before=deepcopy(r.samples)
        with self.assertRaises(ValueError):r.commit(token,[sid])
        self.assertEqual(before,r.samples)
        r.cancel(token)
        self.assertTrue(all(s['reserved']==0 and s['uses']==0 for s in r.samples.values()))
        self.assertFalse(r.reservations)

    def test_four_successes_then_no_fallback_and_context_not_consumed(self):
        r=replay(108)
        for _ in range(4):
            token,ids=r.reserve(576)
            for rank in range(4):r.commit(token,ids[rank*144:(rank+1)*144])
        self.assertTrue(all(s['uses']==4 for s in r.samples.values() if s['segment_id'].startswith('i')))
        self.assertEqual(sum(s['uses'] for s in r.samples.values()),4*576)
        with self.assertRaises(FreshReplayNotReady):r.reserve(576)

    def test_one_aged_member_excludes_whole_group_now_and_in_future(self):
        r=replay(108);s=r.segments['i0'];versions=list(s.versions)
        versions[3]=dict(versions[3],policy=0)
        r.segments['i0']=replace(s,versions=tuple(versions))
        r.set_versions(16,3)
        self.assertEqual(shortage(r,576)['missing']['imagined'],0)
        self.assertEqual(forecast(r,1)['new_chunks_needed'],4)
        r.set_versions(17,3)
        self.assertEqual(shortage(r,576)['missing']['imagined'],4)

    def test_group_flow_one_remaining_use_requires_54_new_groups_for_two_updates(self):
        r=replay(108)
        for sample in r.samples.values():
            if sample['segment_id'].startswith('i'):sample['uses']=3
        p=forecast(r,2)
        self.assertEqual(p['feasible_exposures'],432)
        self.assertEqual(p['new_chunks_needed'],216)
        self.assertEqual(p['sampling_unit_chunks'],4)
        token,_=r.reserve(576)
        self.assertEqual(forecast(r,2)['new_chunks_needed'],432)
        r.cancel(token)

    def test_wrong_geometry_and_config_fail_before_mutation(self):
        r=replay()
        before=len(r.samples)
        with self.assertRaises(ValueError):r.add(segment('wrong',count=3))
        self.assertEqual(len(r.samples),before)
        for size in (144,575,577):
            with self.assertRaises(ValueError):r.reserve(size)
        for kw in ({'jit_gae_mode':'off'},{'real_fraction':.5},{'replay_age_limits':False},{'max_sample_uses':8}):
            with self.assertRaises(ValueError):replace(config(),**kw)
        with self.assertRaises(ValueError):InventoryBucket('bad','imagined',431,5,1,sampling_unit_chunks=4)


class BoundaryTests(unittest.TestCase):
    def test_terminal_never_forwards_endpoint_and_truncation_keeps_last_action(self):
        r=replay(108);_,ids=r.reserve(576)
        batch,_=r.materialize(ids[:144],jit=True)
        segments,bundles=build_plan(batch[0],8,5)
        for name,s in segments.items():
            nodes=[n for b in bundles for n in b.nodes if n.segment==name]
            self.assertEqual(any(n.bootstrap for n in nodes),not bool(s['terminated'][-1]))
            if s['source']=='imagined':self.assertIn(3,[n.index for n in nodes if n.selected is not None])
        # Missing endpoint is rejected even for terminal storage; no repeat/fake terminal.
        for key in ('observations','terminated','truncated','rewards'):
            bad=deepcopy(batch[0]);s=bad[0]['_jit_segment'];s[key]=s[key][:-1]
            with self.subTest(key=key),self.assertRaises((ValueError,IndexError)):build_plan(bad,8,5)

    def test_incomplete_imagined_rank_rejected_without_shortened_gae(self):
        r=replay();_,ids=r.reserve(576);batch,_=r.materialize(ids[:144],jit=True)
        pos=next(i for i,o in enumerate(batch[0]) if o['_jit_segment']['source']=='imagined')
        changed=list(batch[0]);del changed[pos]
        with self.assertRaises(ValueError):build_plan(changed,1,5)

    def test_real_selected_chunk_still_depends_on_entire_original_suffix(self):
        r=replay();s=r.segments['r20'];ids=[s.sample_id(0)]
        batch,_=r.materialize(ids,jit=True);_,bundles=build_plan(batch[0],1,5)
        self.assertEqual([n.index for b in bundles for n in b.nodes if not n.bootstrap],list(reversed(range(s.count))))


class RuntimeTests(unittest.TestCase):
    def test_actual_cpu_jit_two_windows_keep_all_targets_masks_and_ema_count(self):
        torch.manual_seed(9);r=replay();token,ids=r.reserve(576)
        shard=ids[:144];owner,batch,records=owner_for_rank(r,shard,0)
        owner.base_model.layer.weight.data.zero_();owner.base_model.layer.bias.data.zero_()
        # Explicit partial real action: masked tail is still masked in training.
        realpos=next(i for i,x in enumerate(records) if x['source']=='real')
        batch[0][realpos]['_fresh_action_mask'][1]=False
        for window in range(2):
            if window:
                owner.capture_policy_metadata['fresh_versions']=[6,1]
                for obs in batch[0]:obs['_jit_version']=6
            expected={}
            for name in {x['segment_id'] for x in records}:
                s=r.segments[name]
                with torch.no_grad():values=owner.base_model(owner.base_model.prepare_inputs_batch(s.observations))[1].numpy()
                expected[name]=generalized_advantages(s.rewards,values,s.terminated,s.truncated,s.gamma,s.lambda_)
            execute(owner,batch)
            self.assertEqual(len(owner._jit_receipts),144)
            for receipt in owner._jit_receipts:
                sample=r.samples[receipt['sample_id']];adv,target=expected[sample['segment_id']]
                self.assertAlmostEqual(receipt['advantage'],float(adv[sample['index']]),places=4)
                self.assertAlmostEqual(receipt['value_target'],float(target[sample['index']]),places=4)
            event=owner.events[-1][1]['details']
            self.assertEqual(event['selected_grad_states'],144)
            self.assertEqual(event['whole_imagined_segments'],27)
            self.assertEqual(event['initialization_prepass'],not bool(window))
            if window:self.assertEqual(event['prepass_selected_and_context_states'],0)
        r.commit(token,shard);r.cancel(token)
        self.assertEqual(sum(s['uses'] for s in r.samples.values()),144)


class ActorTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_replay_and_coordinator_shards_keep_trajectories_and_credit_units(self):
        from training_topology import GlobalBatchCoordinator
        store=replay(108)
        actor=runtime_replay_class()(10000,config().to_dict())
        actor.capture_ledger=Ledger();actor.telemetry=Writer()
        async def forbidden(*args):raise AssertionError('Shared critic refresh called')
        actor.configure_fresh([SimpleNamespace(evaluate_values=SimpleNamespace(remote=forbidden))])
        actor.set_fresh_versions(5,1)
        for seg in store.segments.values():actor.add_segment(seg)
        credits=[]
        async def drawn(count):credits.append(count)
        coord=GlobalBatchCoordinator(SimpleNamespace(sample=SimpleNamespace(remote=actor.sample)),
            SimpleNamespace(policy_drawn=SimpleNamespace(remote=drawn)),4,pipeline_enabled=True)
        segment_owners={}
        for rank in range(4):
            batch=await coord._sample('policy',rank,144)
            meta=batch[-1];ids=meta['sample_ids'];token=meta['fresh_reservation_token']
            validate_rank(batch[0],rank,4)
            for obs in batch[0]:
                s=obs['_jit_segment']
                if s['source']=='imagined':
                    self.assertEqual(segment_owners.setdefault(s['segment_id'],rank),rank)
            # Transport/commit contract; optimizer math is exercised separately.
            receipts=[dict(sample_id=sid,advantage=1.,value_target=2.,refresh_policy_version=5) for sid in ids]
            actor.commit_fresh_use(token,ids,receipts)
        self.assertEqual(credits,[432])
        self.assertEqual(len(segment_owners),108)
        self.assertFalse(actor.fresh_store.reservations)
        self.assertFalse(actor.fresh_records)
        self.assertEqual(sum(s['uses'] for s in actor.fresh_store.samples.values()),576)


class ProfileTests(unittest.TestCase):
    def test_new_profile_changes_only_sampler_and_parent_remains_frozen(self):
        root=Path(__file__).resolve().parent
        base=json.loads((root/'examples/jit_03_fused_ema.json').read_text('utf8'))
        raw=json.loads((root/'examples/trajectory_01_imagined.json').read_text('utf8'))
        from world_backends import WorldBackendConfig
        WorldBackendConfig.from_mapping(raw).validate_admission()
        self.assertEqual(FreshConfig.from_world_config(raw),replace(config(),
            dense_reward=True,signal_telemetry=True,seed_balancing=True,early_seed_context=True))
        self.assertEqual(raw['backend_options']['fresh'].pop('policy_sampling'),'imagined_trajectories')
        self.assertEqual(base,raw)
        parent=Path(os.environ.get('ACCERL_TEST_WORKSPACE',str(root.parent)))/'output/weaver_jit_scheduler_20261103/source_v1/weaver_integrated_v1'
        manifest=json.loads((root/'trajectory_parent_manifest.json').read_text('utf8'))
        self.assertEqual(len(manifest['files']),247)
        for name,digest in manifest['files'].items():
            with self.subTest(file=name):self.assertEqual(hashlib.sha256((parent/name).read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
