import itertools
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from critical_jit_bench import schedule, calibrate, competition_state, run_critical_jit_benchmark
from gpu_trial_replay import _digest, freeze_critical, critical_decision, commit_critical, release_trial
from test_fresh_policy_replay_cpu import segment
from fresh_config import FreshConfig
from fresh_policy_replay import SegmentReplay


class CriticalBenchTests(unittest.TestCase):
    def test_orders_and_noise_controls(self):
        orders = schedule(20261125)
        self.assertEqual(len(orders), 8)
        self.assertCountEqual([tuple(row) for row in orders if 'P-target' in row],
                              itertools.permutations(['N', 'P-target', 'P-other']))
        self.assertCountEqual([tuple(row) for row in orders if 'N1' in row], [('N1','N2'),('N2','N1')])
        self.assertEqual(orders, schedule(20261125))

    def test_calibration_selects_late_and_other_with_clock_uncertainty(self):
        rows = [dict(rank=r,gpu_uuid=f'gpu-{r}',clock_error_ns=20,
                     bundles=[dict(bundle=j,forward_ready_ns=1000*j+(400 if (r,j)==(2,5) else r),
                                   clock_error_ns=20) for j in range(9)]) for r in range(4)]
        selected = calibrate(rows)
        self.assertEqual((selected['target_rank'],selected['target_bundle'],selected['other_rank']), (2,5,0))
        self.assertTrue(selected['resolved_late_rank'])
        rows[2]['bundles'][5]['clock_error_ns']=500
        self.assertFalse(calibrate(rows)['resolved_late_rank'])
        rows[2]['bundles'][5].update(forward_ready_lo_ns=5390,forward_ready_hi_ns=5410)
        self.assertTrue(calibrate(rows)['resolved_late_rank'])

    def test_pretreatment_state_does_not_use_treated_service_duration(self):
        def event(kind,t,request,batch=None):
            return dict(kind=kind,ts_ns=t,request_id=request,batch_id=batch,rank=1,gpu_uuid='gpu-1')
        rows = [event('enqueue',1,'a'),event('gpu_start',5,'a','b'),event('gpu_end',100,'a','b'),
                event('enqueue',9,'q'), event('service_start',3,'a','b')]
        result = competition_state(rows,1,'gpu-1',10,{1:200,2:200},100,150)
        self.assertEqual(result['state'],'long')
        self.assertEqual(result['queued_requests'],1)
        self.assertEqual(result['projected_gpu_residual_ns'],195)
        rows[2]['ts_ns']=11
        changed = competition_state(rows,1,'gpu-1',10,{1:200,2:200},100,150)
        self.assertEqual(changed['state'],'long')
        self.assertNotEqual(changed['realized_gpu_residual_ns'],result['realized_gpu_residual_ns'])

    def test_reserved_inventory_and_lifetime_are_checked_each_update(self):
        cfg = FreshConfig(real_only=True,real_policy_replay=True,real_fraction=1.,refresh_gae=True,
                          replay_age_limits=True,max_sample_uses=4,max_policy_age=5)
        store = SegmentReplay(4000,cfg)
        store.set_versions(5,0)
        for i in range(10):
            store.add(segment('old'+str(i),'real',policy=1,count=10))
        for i in range(126):
            store.add(segment(str(i),'real',policy=5,count=10))
        owner = SimpleNamespace(fresh_store=store,real_sampling='uniform_blocks',fresh_config=cfg,
                                fresh_admitted_chunks={'real':1360},fresh_critics=[None]*4,fresh_refresh_prune=True)
        frozen=freeze_critical(owner,'tiny',581,123,4)
        ids=frozen['sample_ids']
        self.assertFalse(any(s.startswith('old') for s in ids))
        for version in range(5,9):
            store.set_versions(version,0)
            snap=critical_decision(owner,'tiny',version,1152)
            self.assertEqual(snap['reserved_ids'],576)
            self.assertEqual(snap['min_uses_before'],version-5)
            self.assertEqual(snap['ready'],version<8)
            if version==8:
                self.assertLess(snap['stock']['eligible_next'],1152)
                for i in range(60):
                    store.add(segment('new'+str(i),'real',policy=8,count=10))
                self.assertTrue(critical_decision(owner,'tiny',version,1152)['ready'])
            committed=commit_critical(owner,'tiny',version,ids)
            self.assertEqual(committed['charged'],576)
            with self.assertRaises(ValueError):
                commit_critical(owner,'tiny',version,ids)
        self.assertEqual(len(store.reservations),0)
        self.assertTrue(all(store.samples[s]['uses']==4 for s in ids))
        self.assertTrue(release_trial(owner,'tiny'))

    def test_crossover_uses_actual_probe_and_ack_contract_and_no_extra_updates(self):
        from critical_probe import configure_probe
        class Rpc:
            def __init__(self, fn):
                self.remote = fn
            def options(self, **opts):
                return self

        def nothing(*args,**kwargs):
            pass

        ids = [f'a:chunk{i}' for i in range(576)]
        signature = _digest(ids)
        plan=dict(comparison='critical_jit',checkpoint_policy_step=189,real_workers=8,git_commit='p189',
            real_sampling='uniform_blocks',no_micro_cache_trim=True,compile_action_inference=False,learning_rate=0.,
            global_batch=576,rank_size=144,train_micro=16,context_micro=2,warmup_updates=1,measured_blocks=6,
            noise_blocks=2,updates_per_arm=1,minimum_replay_chunks=2753,high_next_chunks=1152,
            gate_ttl_s=5.,protocol_seed=20261125)
        version, actions, probes, artifacts = 0, [], [], {}
        spec = {}
        def configure(output,**kwargs):
            spec.update(configure_probe(output,**kwargs))
            return dict(spec)
        def freeze(cid,pool,seed,count):
            return dict(ready=True,sample_ids=ids,sample_ids_sha256=signature,original_group_denominators=[40]*72)
        def charge(cid,v,consumed):
            self.assertEqual(v,version)
            self.assertCountEqual(ids,consumed)
            return dict(charged=576)
        def shards(cid,mode):
            self.assertEqual(mode,'preserve')
            return [ids[r*144:(r+1)*144] for r in range(4)]
        replay = SimpleNamespace(freeze_critical_jit=Rpc(freeze),critical_jit_decision=Rpc(lambda *a:dict(ready=True)),
            critical_jit_inventory=Rpc(lambda:dict(policy_version=version)),
            gpu_trial_batch=Rpc(shards),commit_critical_jit=Rpc(charge),release_gpu_trial=Rpc(nothing))
        workers = [SimpleNamespace(run=Rpc(lambda:'worker')) for _ in range(8)]
        ray = SimpleNamespace(get=lambda x,timeout=0:x,wait=lambda refs,**kw: ([],refs) if refs[0]=='worker' else (refs[:1],refs[1:]))
        def trainer(rank):
            def compute(mode, selected):
                now = time.time_ns()
                if rank==0:
                    probes.clear()
                rows=[]
                for j in range(9):
                    t=now+j*10
                    row=dict(bundle=j,forward_start_ns=t,forward_ready_ns=t+3,forward_ready_lo_ns=t+2,
                        forward_ready_hi_ns=t+4,vote_done_ns=t+6,vote_done_lo_ns=t+5,
                        vote_done_hi_ns=t+7,clock_error_ns=1)
                    if rank==spec['target_rank'] and j==spec['target_bundle']:
                        row.update(gate_issued_ns=t+1,gate_closed_ns=t+8,gate_until_ns=t+5_000_000_001,
                            gate_rank=spec['gate_rank'],gate_effective=spec['condition'].startswith('P-'))
                    rows.append(row)
                probes.append(dict(spec,rank=rank,gpu_uuid=f'gpu{rank}',physical_gpu=str(rank+4),bundles=rows))
                rid=f'{version}-{rank}'
                for k,event in enumerate(('ready','dispatch','enqueue','service_start','gpu_start','gpu_end','service_end','response')):
                    actions.append(dict(kind=event,rank=rank,gpu_uuid=f'gpu{rank}',physical_gpu=str(rank+4),
                        ts_ns=now+k,request_id=rid,batch_id=rid,held_ns=0))
                return dict(rank=rank,version=version,version_after=version+1,learning_rate=0,sample_ids_sha256=signature,
                    optimizer_delta=1,skipped_delta=0,train_micro=16,gas=9,actor_loss_enabled=True,layout_arm='preserve',
                    cache_trim=dict(micro='0',idle='1'),initialization_prepass=version==0,consumed_sample_ids=selected,
                    consumed_sample_ids_sha256=_digest(selected),consumed_token_denominators=[40]*144,
                    planned_bundle_context_counts=[0]*9,context=dict(calls_detail=[]))
            return SimpleNamespace(paired_gpu_weight_digest=Rpc(lambda:dict(rank=rank,sha256='fixed')),
                prepare_layout_ema=Rpc(lambda cid,v,first,repeated:dict(rank=rank,version=v,updates=v,mean=0.,variance=1.,scale_floor=.001)),
                run_paired_gpu_arm=Rpc(compute),
                read_layout_normalization=Rpc(lambda v:dict(rank=rank,version=v,mean=0.,scale=1.)))
        def publish():
            nonlocal version
            version+=1
            now=time.time_ns()
            return dict(version=version,receiver_ranks=list(range(4)),rank_ACK_observed_ns=[now]*4,
                        ack_completed_ns=now,inventories=[dict(version=version)]*4)
        def replica(rank):
            return SimpleNamespace(set_training_batch_limit=Rpc(nothing),configure_batch_autotune=Rpc(nothing),
                get_batch_autotune_stats=Rpc(lambda step:dict(rank=rank,action_latencies=[.3],refresh_observations=0)))
        with tempfile.TemporaryDirectory() as output, patch.dict(os.environ,ACCERL_TRAJECTORY_JIT='1',
                ACCERL_REAL_SAMPLING='uniform_blocks',ACCERL_RUN_ID='test'), patch('builtins.print'), \
                patch('time.time_ns',itertools.count(1_000_000_000,1000).__next__), \
                patch('critical_probe._replace',nothing), \
                patch('critical_probe.configure_probe',configure), patch('critical_probe.read_probe',lambda *args:list(probes)), \
                patch('critical_probe.read_actions',lambda output:actions), \
                patch('frozen_policy_eval.atomic_json',lambda path,report:artifacts.update({path.name:report})):
            run_critical_jit_benchmark(ray=ray,plan=plan,output=Path(output),replay=replay,
                trainers=[trainer(i) for i in range(4)],inference=[replica(i) for i in range(4)],workers=workers,
                collector=SimpleNamespace(stop_pipeline=Rpc(nothing)),sync_policy=publish,
                experiment=SimpleNamespace(event=nothing,flush=nothing),stop_requested=lambda:False)
        result=artifacts['result.json']
        self.assertEqual(result['completed_updates'],31)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(result['summary']['matched_contrasts']),6)
        self.assertEqual(len(result['summary']['noise_differences_s']),2)
        self.assertTrue(all(not r['target_gate_held'] for r in result['summary']['matched_contrasts']))
        self.assertTrue(all(a['ready_to_response']['all']['complete']==4 for b in result['blocks'] for a in b['arms']))


if __name__ == '__main__':
    unittest.main()
