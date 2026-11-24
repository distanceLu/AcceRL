"""Fixed-input checks for the diagnostic placement crossover."""
import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path
import os
import random
import tempfile
import unittest
import torch
from advantage_ema import AdvantageEMA
from types import SimpleNamespace
from unittest.mock import patch

from gpu_trial_replay import _digest
import paired_gpu_driver as driver
from real_episode_sampling import _uniform_block
from test_fresh_policy_replay_cpu import runtime_replay_class
from test_model_free_critic_pruning_cpu import real_segment
from jit_gae_plan import build_plan


class LayoutTests(unittest.TestCase):
    def test_pool_sizes_and_arm_order_are_balanced_within_each_condition(self):
        self.assertTrue(hasattr(driver, 'layout_round'))
        schedule = [driver.layout_round(i) for i in range(9)]
        self.assertEqual([n for n,_ in schedule], [2753,581,2753,581,2753,581,2753,581,2753])
        for n in (581,2753):
            orders = [names for size,names in schedule[1:] if size==n]
            self.assertEqual(Counter(orders), {('preserve','repack'):2, ('repack','preserve'):2})

    def test_both_layouts_execute_the_same_latest_eligible_draw_with_original_weights(self):
        replay = runtime_replay_class()(2000, dict(real_only=True, real_policy_replay=True,
            real_fraction=1., refresh_gae=True, replay_age_limits=True))
        replay.fresh_critics = [object()]*4
        replay.fresh_refresh_prune = True
        store = replay.fresh_store
        store.set_versions(5,0)
        for index in range(78):
            store.add(real_segment(index,9))
        raw = list(store.samples)
        store.samples[raw[-4]]['uses'] = store.config.max_sample_uses
        expected_pool = [sid for sid in raw if store._eligible(store.samples[sid])][-581:]
        groups = defaultdict(list)
        for sid in expected_pool:
            groups[store.samples[sid]['segment_id']].append(sid)
        expected = _uniform_block(random.Random(1125),groups)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,ACCERL_JIT_BENCH_OUTPUT=folder):
            self.assertIn('layout_pool_size', replay.freeze_gpu_trial.__code__.co_varnames)
            async def check():
                short = await replay.freeze_gpu_trial(576,'unused',layout_pool_size=2753,layout_seed=1124)
                self.assertFalse(short['ready'])
                frozen = await replay.freeze_gpu_trial(576,'pair-01',layout_pool_size=581,layout_seed=1125)
                self.assertEqual(frozen['sample_ids'],expected)
                self.assertEqual(frozen['pool_ids_sha256'],_digest(expected_pool))
                self.assertEqual(frozen['pool_size'],581)
                self.assertEqual(frozen['n'],576)
                denominators = {sid:sum(int(store.segments[store.samples[s]['segment_id']].action_mask[
                    store.samples[s]['index']].sum()) for s in expected[i:i+8])
                    for i in range(0,576,8) for sid in expected[i:i+8]}
                self.assertGreater(len(set(denominators.values())),1)
                self.assertEqual(frozen['original_group_denominators'],[denominators[s] for s in expected[::8]])
                for arm in ('preserve','repack'):
                    shards = await replay.gpu_trial_batch('pair-01',arm)
                    self.assertEqual({sid for shard in shards for sid in shard['sample_ids']},set(expected))
                    for rank,shard in enumerate(shards):
                        ids = shard['sample_ids']
                        self.assertEqual(shard['cohort_id'],'pair-01-'+arm)
                        self.assertEqual(len(set(ids)),144)
                        self.assertEqual(shard['layout_arm'],arm)
                        self.assertGreaterEqual(shard['layout_planning_s'],0)
                        self.assertEqual(shard['original_denominators'],[denominators[sid] for sid in ids])
                        _, bundles = build_plan(shard['observations'],16,5)
                        self.assertEqual(shard['bundle_context_counts'],[len(b.nodes)-16 for b in bundles])
                        if arm=='preserve':
                            self.assertEqual(set(ids),set(expected[rank*144:(rank+1)*144]))
                self.assertTrue(replay.release_gpu_trial('pair-01'))
            asyncio.run(check())
            self.assertEqual(list(Path(folder).iterdir()),[])
        self.assertEqual(sum(r['uses'] for r in store.samples.values()),store.config.max_sample_uses)
        self.assertFalse(store.reservations)

    def test_exposures_verified_from_jit_receipts_and_cross_rank_union(self):
        self.assertTrue(hasattr(driver, 'verify_layout_exposures'))
        ids = [f't:chunk{i}' for i in range(576)]
        ranks = []
        for i in range(4):
            exposed = ids[i*144:(i+1)*144][::-1]
            ranks.append(dict(rank=i,consumed_sample_ids=exposed,consumed_sample_ids_sha256=_digest(exposed),
                              consumed_token_denominators=[(ids.index(s)//8)+1 for s in exposed]))
        denominators=list(range(1,73))
        driver.verify_layout_exposures('preserve',ids,ranks,denominators)
        ranks[0], ranks[1] = dict(ranks[1],rank=0), dict(ranks[0],rank=1)
        with self.assertRaises(ValueError):
            driver.verify_layout_exposures('preserve',ids,ranks,denominators)
        driver.verify_layout_exposures('repack',ids,ranks,denominators)
        ranks[0]['consumed_token_denominators'][0]+=1
        with self.assertRaises(ValueError):
            driver.verify_layout_exposures('repack',ids,ranks,denominators)
        ranks[0]['consumed_token_denominators'][0]-=1
        ranks[0]['consumed_sample_ids'] = ranks[1]['consumed_sample_ids']
        ranks[0]['consumed_sample_ids_sha256'] = ranks[1]['consumed_sample_ids_sha256']
        with self.assertRaises(ValueError):
            driver.verify_layout_exposures('repack',ids,ranks,denominators)

    def test_pair_reset_replays_numeric_moments_but_keeps_real_update_count(self):
        import paired_gpu_trainer as trainer
        self.assertTrue(hasattr(trainer,'prepare_layout_ema'))
        ema=AdvantageEMA('cpu')
        for i in range(2):
            ema.commit(torch.tensor([2.,1.,2.],dtype=torch.float64),expected_updates=i,optimizer_succeeded=True)
        owner=SimpleNamespace(rank=0,global_step=2,advantage_ema=ema)
        first=trainer.prepare_layout_ema(owner,'p',2,True)
        mean0,scale0=ema.frozen('ema_zscore')
        ema.commit(torch.tensor([2.,6.,20.],dtype=torch.float64),expected_updates=2,optimizer_succeeded=True)
        owner.global_step+=1
        second=trainer.prepare_layout_ema(owner,'p',3,False)
        self.assertEqual((first['mean'],first['variance']),(second['mean'],second['variance']))
        self.assertEqual(ema.updates,3)
        mean1,scale1=ema.frozen('ema_zscore')
        self.assertTrue(torch.equal(mean0,mean1) and torch.equal(scale0,scale1))
        with self.assertRaises(ValueError):
            trainer.prepare_layout_ema(owner,'other',3,False)

    def test_rank_receipt_follows_the_optimizer_consumption_order_and_effective_trim(self):
        import paired_gpu_trainer as trainer
        self.assertTrue(hasattr(trainer, 'layout_receipt'))
        ids = [f't:chunk{i}' for i in range(144)]
        owner = SimpleNamespace(_jit_receipts=[dict(sample_id=sid,refresh_policy_version=3,advantage=1.5,value_target=2.5)
                                               for sid in reversed(ids)])
        with patch.dict(os.environ,ACCERL_TRIM_MICRO_CACHE='0',ACCERL_TRIM_IDLE_CACHE='1'):
            receipt = trainer.layout_receipt(owner,ids,3)
            self.assertEqual(receipt['consumed_sample_ids'],ids[::-1])
            self.assertEqual(receipt['advantage_value_targets'],[[1.5,2.5]]*144)
            self.assertEqual(receipt['cache_trim'],dict(micro='0',idle='1'))
        owner._jit_receipts[-1]['refresh_policy_version']=4
        with self.assertRaises(ValueError):
            trainer.layout_receipt(owner,ids,3)

    def test_driver_records_eighteen_jit_updates_and_both_pool_summaries(self):
        class Rpc:
            def __init__(self,fn):
                self.remote = fn
            def options(self,**kwargs):
                return self
        def noop(*args):
            return None
        ids=[f's:chunk{i}' for i in range(576)]
        versions=[]
        ray = SimpleNamespace(get=lambda value,timeout:value, wait=lambda *args,**kwargs:([],[]))
        def freeze(n,cid,*,layout_pool_size,layout_seed):
            self.assertEqual(layout_seed,1124+int(cid.split('-')[1]))
            return dict(n=n,ready=True,sample_ids=ids,sample_ids_sha256=_digest(ids),pool_size=layout_pool_size,
                        original_group_denominators=[16]*72)
        replay=SimpleNamespace(size=Rpc(lambda:3000),freeze_gpu_trial=Rpc(freeze),
            gpu_trial_batch=Rpc(lambda cid,arm:[(arm,ids[144*i:144*(i+1)]) for i in range(4)]),
            release_gpu_trial=Rpc(noop))
        def worker(rank):
            def prepare(cid,version,first):
                return dict(rank=rank,version=version,updates=version,mean=.25,variance=.25,scale_floor=.001)
            def execute(compute,shard):
                self.assertEqual(compute,'jit')
                name,exposed=shard
                version=len(versions)
                return dict(rank=rank,optimizer_delta=1,skipped_delta=0,version=version,version_after=version+1,
                    train_micro=16,gas=9,actor_loss_enabled=True,cache_trim=dict(micro='0',idle='1'),
                    initialization_prepass=version==0,sample_ids_sha256=_digest(ids),layout_arm=name,
                    consumed_sample_ids=exposed,consumed_sample_ids_sha256=_digest(exposed),layout_planning_s=.001,
                    consumed_token_denominators=[16]*144,planned_bundle_context_counts=[0]*9)
            trainer=SimpleNamespace(paired_gpu_weight_digest=Rpc(lambda:dict(rank=rank,sha256='fixed')),
                                    run_paired_gpu_arm=Rpc(execute),prepare_layout_ema=Rpc(prepare),
                                    read_layout_normalization=Rpc(lambda v:dict(rank=rank,version=v,mean=.25,scale=.5)))
            replica=SimpleNamespace(set_training_batch_limit=Rpc(noop),configure_batch_autotune=Rpc(noop),
                get_batch_autotune_stats=Rpc(lambda v:dict(rank=rank,action_latencies=[.2],
                                                           refresh_observations=0,refresh_seconds=0)))
            return trainer,replica
        trainers,replicas=zip(*(worker(i) for i in range(4)))
        with tempfile.TemporaryDirectory() as folder:
            plan=Path(folder)/'plan.json'
            plan.write_text(json.dumps(dict(comparison='trajectory_layout', real_sampling='uniform_blocks',
                no_micro_cache_trim=True, layout_pool_sizes=[581,2753], warmup_pairs=1,
                measured_pairs=8, minimum_replay_chunks=2753)),encoding='utf8')
            env=dict(ACCERL_RUN_ID='test', ACCERL_JIT_BENCH_PLAN=str(plan),
                     ACCERL_JIT_BENCH_CHECKPOINT='P189', ACCERL_JIT_BENCH_OUTPUT=folder)
            with patch.dict(os.environ,env), patch('builtins.print'):
                driver.run_benchmark(ray=ray,args=SimpleNamespace(num_eval_workers=0,batch_autotune=True,replay_capacity=10000),
                    replay=replay,trainers=trainers,inference=replicas,collector=SimpleNamespace(stop_pipeline=Rpc(noop)),
                    workers=[SimpleNamespace(run=Rpc(noop)) for _ in range(4)],sync_policy=lambda:versions.append(len(versions)),
                    experiment=SimpleNamespace(event=lambda *a,**kw:None,flush=noop),stop_requested=lambda:False)
            result=json.loads((Path(folder)/'result.json').read_bytes())
        self.assertEqual(len(versions),18)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(set(result['summary']),{'581','2753'})
        for pair in result['pairs'][1:]:
            self.assertEqual(pair['arms'][0]['normalization'],pair['arms'][1]['normalization'])
        self.assertEqual([tuple(a['name'] for a in p['arms']) for p in result['pairs']],
                         [driver.layout_round(i)[1] for i in range(9)])


if __name__=='__main__':
    unittest.main()
