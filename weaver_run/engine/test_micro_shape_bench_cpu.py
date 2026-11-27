import itertools
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from micro_shape_bench import compare_audits, latency_samples, shape_schedule


class ShapeDriverTests(unittest.TestCase):
    def test_switches_only_at_common_boundary(self):
        steps = 0
        measured = []
        audit = []
        for phase, micro, pair in shape_schedule():
            gas = 144 // micro
            self.assertEqual(steps % gas, 0)
            steps += gas
            if phase == 'audit':
                audit.append(micro)
            if phase == 'time':
                self.assertEqual(audit, [16, 24])
                measured.append((pair, micro))
        self.assertEqual(measured, [(1,16),(1,24),(2,24),(2,16),(3,24),(3,16),(4,16),(4,24)])

    def _audit(self, candidate):
        rows = []
        for rank in range(4):
            ids = [str(rank*144+i) for i in range(144)]
            if candidate:
                ids.reverse()
            grad = dict(elements=2, logical_elements=8, schema_sha256='group-schema',
                        reference_sq=4., finite=True)
            if candidate:
                grad.update(candidate_sq=4., diff_sq=0., dot=4., max_abs_error=0.)
                grad['groups'] = [{k:grad[k] for k in ('reference_sq','candidate_sq','diff_sq','dot','max_abs_error')}]
            rows.append(dict(rank=rank, normalization={'mean':0.,'scale':1.}, shape_gradient=grad,
                             consumed_sample_ids=ids, advantage_value_targets=[[int(s)/576.,1.] for s in ids]))
        return rows

    def test_aligns_by_id_and_rejects_gradient_or_target_drift(self):
        a, b = self._audit(False), self._audit(True)
        self.assertTrue(compare_audits(a,b,8)['passed'])
        b[1]['shape_gradient']['diff_sq'] = 1.
        self.assertFalse(compare_audits(a,b,8)['passed'])
        b = self._audit(True)
        # A small second optimizer group's errors can disappear in a pooled norm.
        for row in b:
            row['shape_gradient']['groups'].append(dict(reference_sq=1e-12,candidate_sq=1e-12,
                                                       diff_sq=4e-12,dot=-1e-12,max_abs_error=2e-6))
        self.assertFalse(compare_audits(a,b,8)['passed'])
        b = self._audit(True)
        b[2]['advantage_value_targets'][0][1] = 2.
        self.assertFalse(compare_audits(a,b,8)['passed'])
        b[2]['advantage_value_targets'][0][1] = float('nan')
        with self.assertRaises(ValueError):
            compare_audits(a,b,8)

    def test_real_driver_contract_stops_after_failed_audit_and_completes_four_valid_pairs(self):
        from gpu_trial_replay import _digest
        from micro_shape_bench import run_micro_shapes
        class Rpc:
            def __init__(self, fn): self.remote = fn
            def options(self, **kw): return self
        def idle(*args,**kw): pass
        for mismatch in (False,True):
            with self.subTest(mismatch=mismatch):
                version, draws, released = 0, [], []
                ids = [str(i) for i in range(576)]
                binding = [dict(uuid=str(r),physical_gpu=r+4,mps_client_percent=90,visible_sm_count=70) for r in range(4)]
                db = sqlite3.connect(':memory:')
                db.execute('create table events(id integer primary key,run_id,kind,details_json,workload,actor_id,end_ns,duration_ns,status)')
                db.execute('insert into events(run_id,kind,details_json) values(?,?,?)',
                    ('r','trainer_mps_verified',json.dumps(dict(trainer_mps_percent=90,trainers=binding,
                       policy_inference=[dict(g,mps_client_percent=100,visible_sm_count=78) for g in binding]))))
                def freeze(n,cid,**kw):
                    draws.append(cid)
                    return dict(ready=True,sample_ids=ids,original_group_denominators=[8]*72)
                def shards(cid,arm):
                    self.assertEqual((cid,arm),('shape-cohort','preserve'))
                    return [ids[r*144:(r+1)*144] for r in range(4)]
                replay = SimpleNamespace(size=Rpc(lambda:2753),freeze_gpu_trial=Rpc(freeze),
                    gpu_trial_batch=Rpc(shards),release_gpu_trial=Rpc(released.append))
                def train(rank):
                    def run(arm,selected):
                        m = int(arm[-2:])
                        audit = 'audit' in arm
                        grad = self._audit(m==24)[rank]['shape_gradient'] if audit else None
                        if audit and m==24 and mismatch:
                            grad['diff_sq'] = 4.
                        t0,t1 = time.time_ns(),time.time_ns()
                        for load,kind,index in [('policy_infer_gpu_forward_and_copy','service',rank),
                                                ('policy_infer','client_wait',2*rank),('policy_infer','client_wait',2*rank+1)]:
                            db.execute('insert into events(run_id,workload,kind,actor_id,end_ns,duration_ns,status) values(?,?,?,?,?,?,?)',
                                ('r',load,kind,index,t1,1_000_000,'ok'))
                        return dict(rank=rank,version=version,version_after=version+1,cohort_id='shape-cohort-preserve',
                            micro=m,gas=144//m,arm=arm,learning_rate=0.,optimizer_steps=1,skipped_steps=0,
                            cache_trim={'micro':'0'},shape_gradient=grad,consumed_sample_ids=selected,
                            consumed_sample_ids_sha256=_digest(selected),consumed_token_denominators=[8]*144,
                            input_sample_ids=selected,input_order_sha256=_digest(selected),normalization={'mean':0.,'scale':1.},
                            advantage_value_targets=[[.1,1.]]*144,trainer_compile=dict(registered={'compiled':True,'backend':'inductor'},
                               delta=dict(unique_graphs=int(version in (0,2)),fxgraph_cache_miss=int(version in (0,2)),fxgraph_cache_hit=0)),
                            timing_start_ns=t0,timing_end_ns=t1,peak_allocated_bytes=1,peak_reserved_bytes=2,
                            sampled_peak_used_bytes=3,estimated_peak_used_bytes=4)
                    return SimpleNamespace(run_paired_gpu_arm=Rpc(run),paired_gpu_weight_digest=Rpc(lambda:dict(rank=rank,sha256='f',elements=8)))
                def publish():
                    nonlocal version
                    version+=1
                    stamp = time.time_ns()
                    for load,kind,n in [('policy_infer_gpu_forward_and_copy','service',4),('policy_infer','client_wait',8)]:
                        for i in range(n):
                            db.execute('insert into events(run_id,workload,kind,actor_id,end_ns,duration_ns,status) values(?,?,?,?,?,?,?)',
                                       ('r',load,kind,i,stamp,200_000,'ok'))
                    return dict(version=version,receiver_ranks=list(range(4)),rank_ACK_observed_ns=[stamp]*4,
                        ack_completed_ns=stamp,export_submitted_ns=stamp,inventories=[dict(version=version)]*4)
                def replica(rank):
                    return SimpleNamespace(set_training_batch_limit=Rpc(idle),configure_batch_autotune=Rpc(idle),
                        get_batch_autotune_stats=Rpc(lambda *a,**k:dict(rank=rank,compiled_policy=dict(backend='cuda_graph',
                               graph_replays=version*20,fallback_counts={},capture_failures=0))))
                artifacts = {}
                plan = dict(git_commit='f',train_shapes=[[16,9],[24,6]],real_workers=8,global_batch=576,rank_size=144,
                    real_sampling='uniform_blocks',trainer_mps_percent=90,trainer_compile='on',compile_action_inference=True,
                    no_micro_cache_trim=True,learning_rate=0.,parity_pairs=1,warmup_pairs=1,measured_pairs=4,
                    minimum_replay_chunks=2753,protocol_seed=1127,checkpoint_policy_step=189)
                kwargs = dict(ray=SimpleNamespace(get=lambda x,timeout:x,wait=lambda rows,**kw:([],rows)),plan=plan,
                    output=Path('.'),replay=replay,trainers=[train(i) for i in range(4)],inference=[replica(i) for i in range(4)],
                    collector=SimpleNamespace(stop_pipeline=Rpc(idle)),workers=[SimpleNamespace(run=Rpc(idle)) for _ in range(8)],
                    sync_policy=publish,experiment=SimpleNamespace(run_id='r',connection=db,flush=idle,event=idle),stop_requested=lambda:False)
                with patch('builtins.print'), patch('time.time_ns',itertools.count(10000,10).__next__), \
                     patch('frozen_policy_eval.atomic_json',lambda path,r:artifacts.update({path.name:r})):
                    if mismatch:
                        with self.assertRaisesRegex(RuntimeError,'parity exceeded'):
                            run_micro_shapes(**kwargs)
                    else:
                        run_micro_shapes(**kwargs)
                result = artifacts['result.json']
                self.assertEqual((draws,released),(['shape-cohort'],['shape-cohort']))
                self.assertEqual(result['parity_passed'],not mismatch)
                self.assertEqual(len(result['pairs']),0 if mismatch else 4)
                if not mismatch:
                    self.assertEqual(result['summary']['by_micro']['24']['pooled_rollout_latency']['client_wait_ms']['n'],32)
                db.close()

    def test_telemetry_quantiles_use_completed_successful_calls_on_each_arm(self):
        db = sqlite3.connect(':memory:')
        self.addCleanup(db.close)
        db.execute('create table events(run_id,workload,kind,actor_id,end_ns,duration_ns,status)')
        for load,kind,n in [('policy_infer','client_wait',8),('policy_infer_gpu_forward_and_copy','service',4)]:
            for rank in range(n):
                for end,status,value in [(100,'ok',999),(105,'ok',rank+1),(115,'ok',999),(103,'error',999)]:
                    db.execute('insert into events values(?,?,?,?,?,?,?)',
                               ('run',load,kind,str(rank),end,value*1_000_000,status))
        rows = latency_samples(db,'run',100,110)
        self.assertEqual(rows['client_wait_ms'], list(range(1,9)))
        self.assertEqual(rows['inference_service_ms'], list(range(1,5)))


if __name__ == '__main__':
    unittest.main()
