"""Crossover and frozen production-epoch invariants, without CUDA or Ray."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from admission_pulse_bench import CONDITIONS, block_order, fixed_weight_epoch, replay_snapshot, recent_demand


class PulseBenchTest(unittest.TestCase):
    def test_every_condition_at_every_position(self):
        orders = [block_order(i) for i in range(3)]
        self.assertTrue(all(set(order) == set(CONDITIONS) for order in orders))
        self.assertTrue(all(set(column) == set(CONDITIONS) for column in zip(*orders)))

    def test_recent_demand_uses_actual_observation(self):
        rows = [dict(rank=i, last_request_ns=0, observed_ns=2_000_000_000) for i in range(4)]
        self.assertFalse(recent_demand(rows, 2_000_000_000))
        rows[2]['last_request_ns'] = 1_500_000_000
        self.assertTrue(recent_demand(rows, 2_000_000_000))
        self.assertFalse(recent_demand(rows, 2_251_000_000))

    def test_next_publication_expiry_keeps_reuse_reservations(self):
        samples = [dict(segment_id='s', index=i, uses=u, reserved=r)
                   for i,(u,r) in enumerate([(0,0),(1,0),(3,1)])]
        store = SimpleNamespace(samples=dict(enumerate(samples)), segments={'s': SimpleNamespace(
            versions=[{'policy':0},{'policy':1},{'policy':15}])}, versions=(16,0),
            config=SimpleNamespace(real_only=True, max_policy_age=16, max_sample_uses=4))
        owner = SimpleNamespace(fresh_store=store, fresh_admitted_chunks={'real':9}, real_sampling='uniform_blocks')
        row = replay_snapshot(owner)
        self.assertEqual((row['eligible_now'], row['eligible_after_next_publication']), (2,1))
        self.assertEqual(row['admitted_real_chunks'],9)

    def test_real_trainer_path_freezes_lr_and_checks_use_receipts(self):
        owner = SimpleNamespace(_trajectory_jit=True, cpu_replay_prefetch=True, global_step=3,
              rank=0, model=SimpleNamespace(global_steps=3,skipped_steps=0), policy_lr=.2,
              value_lr=.3, policy_train_start_step=100)
        ids = [str(i) for i in range(144)]
        async def train(step):
            self.assertEqual((owner.policy_lr,owner.value_lr,owner.policy_train_start_step),(0,0,0))
            owner.global_step=owner.model.global_steps=step
            owner.capture_policy_metadata=dict(sample_ids=ids, fresh_versions=[step-1,0],
                    real_sampling='uniform_blocks', fresh_reservation_token='token')
            owner._jit_receipts = [dict(sample_id=s) for s in ids]
            return (1.,0.,0.,0.,0.,dict(policy=0.,value=0.),step)
        owner.run_training_epoch=train
        with patch.dict(os.environ, ACCERL_JIT_BENCH_PLAN='p', ACCERL_REAL_SAMPLING='uniform_blocks'):
            out=asyncio.run(fixed_weight_epoch(owner,4))
        self.assertEqual(out['sample_ids'],ids)
        self.assertEqual(out['version'],4)

    def test_crossover_counts_the_scheduled_delay_and_all_28_publications(self):
        report = self.run_trial(normal_only=False, real_workers=4)
        self.assertEqual(report['completed_updates'],28)
        for block in report['blocks']:
            for arm in block['arms']:
                first=arm['updates'][0]
                self.assertGreater(first['dispatch_ns'],arm['intent_wall_ns'])
                if arm['condition']=='trainer_yields':
                    self.assertGreaterEqual(first['dispatch_ns']-arm['intent_wall_ns'],5_000_000_000)

    def test_normal_only_runs_three_windows_of_three_updates(self):
        for collectors in (4,8):
            with self.subTest(collectors=collectors):
                report = self.run_trial(normal_only=True, real_workers=collectors)
                self.assertEqual(report['completed_updates'], 10)
                self.assertEqual([[(a['condition'], a['pulse'], len(a['updates'])) for a in b['arms']]
                                  for b in report['blocks']], [[('normal', None, 3)]] * 3)
                self.assertEqual(report['actual_real_workers'], collectors)

    def run_trial(self, *, normal_only, real_workers):
        from paired_gpu_driver import run_benchmark
        def actor(**methods):
            return SimpleNamespace(**{key:SimpleNamespace(remote=fn) for key,fn in methods.items()})
        version = 0
        def publish():
            nonlocal version
            version += 1
        def rank_worker(rank):
            def epoch(step):
                self.assertEqual(step,version+1)
                return dict(rank=rank,version=step,reservation='r'+str(version),
                    sample_ids=[f'{rank}-{i}' for i in range(144)],
                    started_ns=time.time_ns(),finished_ns=time.time_ns())
            return actor(run_admission_pulse_epoch=epoch, configure_batch_autotune=lambda *_:None,
                paired_gpu_weight_digest=lambda:dict(rank=rank,sha256='same_weights'))
        def infer(rank):
            return actor(set_training_batch_limit=lambda *_:None, configure_batch_autotune=lambda *_:None,
                         pulse_action_demand=lambda:dict(rank=rank,last_request_ns=time.time_ns()))
        def snapshot():
            return dict(policy_version=version,eligible_now=2048)
        replay = actor(pulse_inventory=snapshot,
                       admit_model_free_batch=lambda n:dict(ready=True,eligible={'real':2048},
                           versions=[version,0],reservation_token='r'+str(version)),
                       configure_batch_autotune=lambda *_:None)
        plan=dict(comparison='admission_pulse',warmup_updates=1,measured_blocks=3,updates_per_arm=3,
                  pulse_seconds=5.,real_sampling='uniform_blocks',minimum_replay_chunks=2048,
                  git_commit='abc',checkpoint='P189')
        ns = 100_000_000_000
        def now():
            nonlocal ns
            ns += 1000
            return ns
        def sleep(seconds):
            nonlocal ns
            ns += round(seconds*1e9)
        if normal_only:
            plan.update(normal_only=True, real_workers=real_workers)
        with (tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,
                ACCERL_TRAJECTORY_JIT='1',ACCERL_REAL_SAMPLING='uniform_blocks',ACCERL_RUN_ID='trial',
                ACCERL_JIT_BENCH_OUTPUT=tmp, ACCERL_JIT_BENCH_PLAN=str(Path(tmp)/'plan.json'),
                WEAVER_BENCH_NORMAL_ONLY='1' if normal_only else '0', WEAVER_BENCH_REAL_WORKERS=str(real_workers)),
                patch('time.time_ns',now),patch('time.sleep',sleep),patch('builtins.print')):
            (Path(tmp)/'plan.json').write_text(json.dumps(plan))
            run_benchmark(ray=SimpleNamespace(get=lambda v,timeout:v,wait=lambda *a,**kw:([],None)),
                args=SimpleNamespace(num_eval_workers=0,batch_autotune=True),
                replay=replay,trainers=[rank_worker(i) for i in range(4)],
                inference=[infer(i) for i in range(4)],collector=actor(stop_pipeline=lambda:None),
                workers=[actor(run=lambda:None) for _ in range(real_workers)],sync_policy=publish,
                experiment=SimpleNamespace(event=lambda *a,**kw:None,flush=lambda:None),stop_requested=lambda:False)
            return json.loads((Path(tmp)/'result.json').read_text())


if __name__ == '__main__':
    unittest.main()
