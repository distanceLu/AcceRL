"""Actual candidate runtime on CPU/toy engine; not Ray/CUDA/ZeRO acceptance."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from advantage_ema import AdvantageEMA, raw_sums
from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay, generalized_advantages
from jit_gae import ReverseGAE, masked_policy_loss
from jit_gae_plan import build_plan
from jit_gae_runtime import run_jit_policy_update
from jit_replay import validate_receipts
from test_fresh_policy_replay_cpu import segment, runtime_replay_class, Ledger, Writer

torch.set_num_threads(1)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1,7)
        self.calls = []

    def prepare_inputs_batch(self, rows):
        if any(any(k.startswith('_') for k in row) for row in rows):
            raise AssertionError('Transport metadata reached model inputs')
        return torch.tensor([[row['value']/10.] for row in rows])

    def forward(self, inputs):
        self.calls.append((len(inputs),torch.is_grad_enabled()))
        y = self.layer(inputs)
        return y[:,:6].reshape(-1,2,1,3), y[:,6]


class ToyEngine(torch.nn.Module):
    def __init__(self, module, gas, skip=False):
        super().__init__();self.module=module;self.gas=gas;self.skip=skip
        self.global_steps=0;self.skipped_steps=0;self.micro_steps=0
        self.optimizer=torch.optim.SGD(module.parameters(),lr=.003)

    def forward(self, inputs): return self.module(inputs)
    def backward(self, loss): (loss/self.gas).backward()
    def step(self):
        self.micro_steps+=1
        if self.micro_steps % self.gas == 0:
            if torch.distributed.is_initialized():
                for parameter in self.module.parameters():
                    torch.distributed.all_reduce(parameter.grad)
                    parameter.grad.div_(torch.distributed.get_world_size())
            self.global_steps+=1
            if self.skip:self.skipped_steps+=1
            else:self.optimizer.step()
            self.optimizer.zero_grad()


def cohort(version=5, rank=0):
    cfg=FreshConfig(refresh_gae=True,replay_age_limits=True,jit_gae_mode='fused',advantage_norm='ema_zscore')
    replay=SegmentReplay(64,cfg)
    a=segment('a'+str(rank),count=7,terminal=True)
    b=segment('b'+str(rank),source='real',count=5,terminal=False)
    mask=b.action_mask.copy();mask[1:,1]=False
    b=replace(b,action_mask=mask)
    replay.add(a);replay.add(b);replay.set_versions(version,1)
    ids=[a.sample_id(1),b.sample_id(3),a.sample_id(5),b.sample_id(1)]
    batch,records=replay.materialize(ids,jit=True)
    return replay,batch,records,ids


def owner_for(model, batch, ids, *, mode='fused', norm='ema_zscore', micro=2, rank=0, skip=False):
    cfg=FreshConfig(refresh_gae=True,jit_gae_mode=mode,advantage_norm=norm)
    engine=ToyEngine(model,len(ids)//micro,skip=skip)
    events=[]
    def verify(engine,kind,before,skips):
        if engine.global_steps-before != 1 or engine.skipped_steps-skips != 0:
            raise RuntimeError('Failed optimizer window')
    return SimpleNamespace(model=engine,base_model=model,fresh_config=cfg,cfg=SimpleNamespace(lora_dropout=0),
        global_step=5,rank=rank,super_batch_size=len(ids),train_batch_size=micro,
        capture_policy_metadata={'sample_ids':ids,'fresh_versions':[5,1],'jit_pending':True},
        vf_coef=.5,ent_coef=.01,kl_coef=.1,clip_mode='gipo',clip_eps=.2,policy_train_start_step=0,
        _verify_optimizer_update=verify,telemetry=SimpleNamespace(event=lambda name,**kw:events.append((name,kw))),events=events)


def execute(owner,batch):
    captures=[]
    def begin(*args,**kwargs):
        captures.extend(kwargs['sample_indices']);return None
    result=run_jit_policy_update(owner,batch[0],batch[1],batch[3],
        micro_begin=begin,micro_end=lambda *a,**k:None,trim=lambda phase:None)
    return result,captures


class MathTests(unittest.TestCase):
    def test_reverse_suffix_matches_full_gae_and_bootstrap(self):
        replay,batch,_,ids=cohort()
        segments,bundles=build_plan(batch[0],2,5)
        gae=ReverseGAE(segments,'cpu');seen={}
        for bundle in bundles:
            vals={(n.segment,n.index):torch.tensor(float(n.index)*.13,requires_grad=True) for n in bundle.nodes}
            adv,target=gae.consume(bundle,vals)
            self.assertFalse(adv.requires_grad);self.assertFalse(target.requires_grad)
            for i,a,t in zip(bundle.selected,adv,target):seen[i]=(float(a),float(t))
        for i,sid in enumerate(ids):
            s=replay.samples[sid];seg=replay.segments[s['segment_id']]
            a,t=generalized_advantages(seg.rewards,np.arange(seg.count+1)*.13,
                seg.terminated,seg.truncated,seg.gamma,seg.lambda_)
            np.testing.assert_allclose(seen[i],[a[s['index']],t[s['index']]],rtol=1e-6)
        self.assertEqual(sorted(seen),list(range(len(ids))))
        self.assertNotIn(('a0',0),[(n.segment,n.index) for b in bundles for n in b.nodes])

    def test_ema_drift_variance_and_transaction(self):
        ema=AdvantageEMA('cpu',beta=.5)
        with self.assertRaises(ValueError):ema.frozen('ema_zscore')
        first=raw_sums(torch.tensor([0.,2.]))
        m,s=ema.frozen('ema_zscore',first)
        self.assertEqual(ema.updates,0)
        ema.commit(first,expected_updates=0,optimizer_succeeded=True)
        frozen=ema.frozen('ema_zscore')
        second=raw_sums(torch.tensor([10.,12.]))
        self.assertFalse(ema.commit(second,expected_updates=1,optimizer_succeeded=False))
        self.assertEqual(ema.updates,1)
        ema.commit(second,expected_updates=1,optimizer_succeeded=True)
        self.assertEqual(float(ema.mean),6.);self.assertEqual(float(ema.variance),26.)
        self.assertEqual(float(frozen[0]),1.);self.assertEqual(float(frozen[1]),1.)
        with self.assertRaises(RuntimeError):ema.commit(second,expected_updates=1,optimizer_succeeded=True)

    def test_ema_recovery_and_rms_floor(self):
        ema=AdvantageEMA('cpu');ema.commit(raw_sums(torch.tensor([4.,4.])),expected_updates=0,optimizer_succeeded=True)
        restored=AdvantageEMA('cpu');restored.load_state_dict(ema.state_dict())
        self.assertEqual(restored.updates,1)
        self.assertEqual(float(restored.frozen('ema_rms')[1]),4.)
        self.assertEqual(float(restored.frozen('ema_zscore')[1]),.001)
        state=ema.state_dict();state['variance']=torch.tensor(float('nan'))
        with self.assertRaises(ValueError):restored.load_state_dict(state)

    def test_masks_preserve_original_micro_gradient_after_regrouping(self):
        torch.manual_seed(17)
        logits=torch.randn(4,2,1,3,requires_grad=True);values=torch.randn(4,requires_grad=True)
        old=torch.randn_like(logits);actions=torch.zeros((4,2,1),dtype=torch.long)
        adv=torch.tensor([1.,-2.,.5,1.2]);target=torch.ones(4)
        masks=torch.tensor([[[1],[1]],[[1],[0]],[[1],[0]],[[1],[0]]],dtype=torch.bool)
        weights=torch.tensor([1/3,1/3,1/2,1/2])
        kw=dict(vf_coef=.5,ent_coef=.1,kl_coef=.2,clip_mode='gipo',clip_eps=.2)
        def loss(order):
            total=0
            for indices in order:
                l,_=masked_policy_loss(logits[indices],values[indices],actions[indices],old[indices],
                    adv[indices],target[indices],masks[indices],weights[indices],**kw)
                total+=l/2
            return total
        a=loss([[0,1],[2,3]]);ga=torch.autograd.grad(a,(logits,values),retain_graph=True)
        b=loss([[2,0],[3,1]]);gb=torch.autograd.grad(b,(logits,values))
        torch.testing.assert_close(a,b)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
        self.assertTrue(torch.equal(ga[0][1,1],torch.zeros_like(ga[0][1,1])))

    def test_plan_rejects_duplicates_wrong_version_and_partial_micro(self):
        _,batch,_,_=cohort()
        for rows,micro,version in [(batch[0],2,6),([batch[0][0]]*4,2,5),(batch[0],3,5)]:
            with self.assertRaises(ValueError):build_plan(rows,micro,version)


class RuntimeTests(unittest.TestCase):
    def test_actual_capture_hooks_keep_completed_step_and_selected_identities(self):
        from capture_training import micro_begin, micro_end
        from contextlib import AbstractContextManager
        class Span(AbstractContextManager):
            def __init__(self):self.details={}
            def __exit__(self,*args):return False
        class Capture:
            def __init__(self):self.events=[];self.used=[]
            def span(self,*a,**k):return Span()
            def event(self,name,**kw):self.events.append((name,kw))
            def replay_used(self,*a,**kw):self.used.extend(kw['sample_ids'])
            def flush(self):pass
        _,batch,_,ids=cohort();owner=owner_for(ToyModel(),batch,ids)
        owner.capture_telemetry=Capture();owner.training_role='policy';owner.world_size=1
        owner.capture_versions={'policy':5}
        owner.model.is_gradient_accumulation_boundary=lambda: (owner.model.micro_steps+1)%owner.model.gas==0
        owner.model.train_batch_size=lambda:4
        run_jit_policy_update(owner,batch[0],batch[1],batch[3],micro_begin=micro_begin,micro_end=micro_end,trim=lambda _:None)
        events=[r for name,r in owner.capture_telemetry.events if name=='optimizer_update']
        self.assertEqual(len(events),1);self.assertEqual(events[0]['step'],6)
        self.assertEqual(owner.global_step,6)
        self.assertEqual(sorted(owner.capture_telemetry.used),sorted(ids))

    def test_fused_matches_separate_exact_update_and_actual_receipts(self):
        torch.manual_seed(9);_,batch,_,ids=cohort()
        model=ToyModel();a=owner_for(deepcopy(model),batch,ids,norm='optimizer_window')
        b=owner_for(deepcopy(model),batch,ids,mode='trainer_separate',norm='optimizer_window')
        _,order=execute(a,batch);execute(b,batch)
        self.assertEqual(sorted(order),list(range(4)))
        for x,y in zip(a.model.parameters(),b.model.parameters()):torch.testing.assert_close(x,y)
        ar=sorted(a._jit_receipts,key=lambda r:r['sample_id']);br=sorted(b._jit_receipts,key=lambda r:r['sample_id'])
        for x,y in zip(ar,br):
            self.assertEqual(x['sample_id'],y['sample_id'])
            self.assertAlmostEqual(x['advantage'],y['advantage'],places=5)
        self.assertEqual(a.advantage_ema.updates,1)

    def test_initialized_ema_uses_one_selected_forward_and_frozen_statistics(self):
        _,batch,_,ids=cohort();owner=owner_for(ToyModel(),batch,ids)
        owner.advantage_ema=AdvantageEMA('cpu')
        owner.advantage_ema.commit(raw_sums(torch.tensor([-2.,2.])),expected_updates=0,optimizer_succeeded=True)
        execute(owner,batch)
        detail=owner.events[-1][1]['details']
        self.assertFalse(detail['initialization_prepass'])
        self.assertEqual(detail['selected_grad_states'],4)
        self.assertEqual(detail['selected_grad_forward_calls'],2)
        self.assertEqual(detail['normalization_mean'],0.)
        self.assertEqual(detail['normalization_scale'],2.)
        self.assertGreater(detail['context_only_states'],0)
        self.assertEqual(owner.advantage_ema.updates,2)

    def test_skip_does_not_initialize_ema_or_emit_consumption_receipts(self):
        _,batch,_,ids=cohort();owner=owner_for(ToyModel(),batch,ids,skip=True)
        with self.assertRaisesRegex(RuntimeError,'Failed optimizer'):execute(owner,batch)
        self.assertEqual(owner.advantage_ema.updates,0);self.assertIsNone(owner._jit_receipts)

    def test_failed_identity_and_active_dropout_fail_before_optimizer(self):
        _,batch,_,ids=cohort()
        for bad_identity in (True,False):
            model=ToyModel();owner=owner_for(model,batch,ids)
            if bad_identity:owner.capture_policy_metadata['sample_ids']=ids[::-1]
            else:model.drop=torch.nn.Dropout(.1)
            with self.assertRaises(RuntimeError):execute(owner,batch)
            self.assertEqual(owner.model.global_steps,0)

    def test_value_only_warmup_still_records_real_advantages(self):
        _,batch,_,ids=cohort();owner=owner_for(ToyModel(),batch,ids);owner.policy_train_start_step=100
        before=owner.base_model.layer.weight.detach().clone()
        result,_=execute(owner,batch)
        self.assertTrue(all(x==0 for x in result['epoch_p_losses']))
        torch.testing.assert_close(owner.base_model.layer.weight[:6],before[:6])
        self.assertEqual(len(owner._jit_receipts),4)

    def test_nonfinite_and_mutating_buffer_stop_before_optimizer(self):
        _,batch,_,ids=cohort()
        class Mutates(ToyModel):
            def __init__(self):
                super().__init__();self.register_buffer('counter',torch.zeros(()))
            def forward(self,x):
                self.counter.add_(1);return super().forward(x)
        for model in (Mutates(),ToyModel()):
            if type(model) is ToyModel:
                with torch.no_grad():model.layer.weight.fill_(float('nan'))
            owner=owner_for(model,batch,ids)
            with self.assertRaises((RuntimeError,ValueError)):execute(owner,batch)
            self.assertEqual(owner.model.global_steps,0);self.assertEqual(owner.advantage_ema.updates,0)

    def test_configuration_does_not_silently_enable_or_disable_ema(self):
        for changes in ({'advantage_norm':'ema_zscore'}, {'jit_gae_mode':'fused'},
                        {'advantage_ema_beta':1.}, {'advantage_scale_floor':0.}, {'jit_context_micro':0}):
            with self.assertRaises(ValueError):FreshConfig(**changes)


class ReplayRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_replay_bypasses_inference_and_commits_only_selected(self):
        async def forbidden(*args):raise AssertionError('Shared inference refresh called')
        cfg=FreshConfig(refresh_gae=True,signal_telemetry=True,replay_age_limits=True,jit_gae_mode='fused',advantage_norm='ema_zscore')
        replay=runtime_replay_class()(20,cfg.to_dict());replay.capture_ledger=Ledger();replay.telemetry=Writer()
        replay.configure_fresh([SimpleNamespace(evaluate_values=SimpleNamespace(remote=forbidden))])
        replay.set_fresh_versions(5,1);replay.add_segment(segment(count=8))
        batch=await replay.sample(4);ids=batch[-1]['sample_ids'];token=batch[-1]['fresh_reservation_token']
        self.assertTrue(batch[-1]['jit_pending']);self.assertFalse(batch[-1]['critic_refreshed'])
        self.assertTrue(all(r['advantage'] is None for r in replay.fresh_records[token].values()))
        owner=owner_for(ToyModel(),batch,ids);execute(owner,batch)
        replay.commit_fresh_use(token,ids,owner._jit_receipts)
        self.assertEqual(sum(r['uses'] for r in replay.fresh_store.samples.values()),4)
        self.assertEqual(sum(r['reserved'] for r in replay.fresh_store.samples.values()),0)
        consumed=[r for _,kw in replay.telemetry.events if (r:=kw.get('details',{})).get('phase')=='consumed']
        self.assertEqual(len(consumed),4)
        self.assertTrue(all(r['refresh_policy_version']==5 and not r['rewards_refreshed'] for r in consumed))
        with self.assertRaises((ValueError,KeyError)):replay.commit_fresh_use(token,ids,owner._jit_receipts)

    async def test_receipt_relabel_and_wrong_version_are_rejected_without_consumption(self):
        replay,batch,records,ids=cohort()
        rows=[dict(sample_id=s,advantage=1.,value_target=2.,refresh_policy_version=5) for s in ids]
        mapping={r['sample_id']:r for r in records}
        for changes in ({'refresh_policy_version':4},{'advantage':float('inf')},{'reward':99.}):
            bad=deepcopy(rows);bad[0].update(changes)
            with self.assertRaises(ValueError):validate_receipts(mapping,ids,bad,5)
        self.assertEqual(sum(s['uses'] for s in replay.samples.values()),0)

    async def test_scheduler_never_forecasts_phantom_inference_refresh(self):
        from cost_scheduler_runtime import CostAdmissionBroker, replay_inventory
        from test_cost_scheduler_runtime_cpu import settings, cycle, SOURCE
        replay,_,_,_=cohort()
        inventory=dict(replay_inventory(replay),inventory_revision=1,inventory_versions=[5,1])
        broker=CostAdmissionBroker(settings(),'jit-test',SOURCE)
        await broker.update_inventory(inventory)
        await broker.update_cycle(cycle(policy=5,world=1))
        self.assertEqual(broker.latest_cycle_snapshot.critic_observations_per_future_update,0.)


if __name__ == '__main__':unittest.main()
