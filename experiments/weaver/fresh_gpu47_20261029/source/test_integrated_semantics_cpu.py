"""Real-label admission, four-consumer streaming and production GAE, no CUDA."""
import ast
import asyncio
import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import numpy as np
from world_backends import WorldBackendConfig,build_service_plan
from fresh_config import FreshConfig
from world_backends.rollout import consume_predicted_chunk
from replay_projection import native_reward_cohort
from training_topology import GlobalBatchCoordinator,shard_metadata
from weaver_checkpoint import admit_metadata,WeaverAdmissionError

ROOT=Path(__file__).parent
def config():return json.loads((ROOT/'examples/weaver_integrated_internal.json').read_text(encoding='utf-8'))
def flattened(raw):
    values=dict(raw);values.update(values.pop('backend_options'));return values


class Admission(unittest.TestCase):
    def test_actual_native_metadata_admitted_without_fabricating_calibration(self):
        raw=config();cfg=WorldBackendConfig.from_mapping(raw);cfg.validate_admission()
        self.assertIs(admit_metadata(flattened(raw),raw['checkpoint_metadata']),raw['checkpoint_metadata'])
        self.assertEqual(raw['checkpoint_metadata']['load_report']['tensor_count'],866)
        self.assertFalse(cfg.capabilities().provides_success)
        self.assertFalse(cfg.capabilities().provides_terminal)
        plan=build_service_plan(cfg);self.assertEqual(plan.expected_gpu_model_processes,16)
        self.assertEqual(plan.active_kinds,('policy','ctrl'));self.assertEqual(plan.wm_consumers,('ctrl',))
    def test_low_level_admission_rejects_false_head_claim_source_tensor_schema_and_spec(self):
        for mutation in (lambda r:r['checkpoint_metadata']['success_head'].update(calibrated=True),
                         lambda r:r['backend_options'].update(initial_checkpoint_sha256='0'*64),
                         lambda r:r['checkpoint_metadata']['load_report'].update(tensor_count=865),
                         lambda r:r['schema'].update(normalizer_id='other'),
                         lambda r:r['backend_options']['native_reward_head'].update(link='identity'),
                         lambda r:r.update(terminal_rule='success_probability')):
            raw=config();mutation(raw)
            with self.assertRaises(WeaverAdmissionError):admit_metadata(flattened(raw),raw['checkpoint_metadata'])
    def test_legacy_success_mode_still_rejects_uncalibrated_current_checkpoint(self):
        raw=config();raw['reward_semantics']='delta_success';raw['terminal_rule']='argmax'
        with self.assertRaises(WeaverAdmissionError):admit_metadata(flattened(raw),raw['checkpoint_metadata'])
    def test_checkpoint_metadata_remains_exact_initial_bundle_metadata(self):
        # The new head initialization report belongs to the run, never an
        # invented modification of the immutable evaluated checkpoint.
        self.assertNotIn('reward_head_provenance',config()['checkpoint_metadata'])


class Window:
    def __init__(self,index,positive=True):
        self.sample_id=str(index);self.start=0;self.length=40
        labels=np.zeros(40,bool);labels[-1]=bool(positive and index%2)
        self.episode=types.SimpleNamespace(metadata={'label_source':'libero_env_step'},success=labels)
    @property
    def world_observations(self):raise AssertionError('Counting labels touched pixels')


class CohortLabels(unittest.TestCase):
    def test_global_counts_use_exact_ten_frames_and_no_pixels(self):
        rows=[Window(i) for i in range(1024)]
        result=native_reward_cohort(rows)
        self.assertEqual((result['positive_count'],result['negative_count']),(512,9728))
        self.assertEqual(result['positive_weight'],19.)
        self.assertTrue(result['rm_active']);self.assertEqual(result['frame_slice'],[30,40])
    def test_missing_class_disables_head_without_made_up_positive_labels(self):
        result=native_reward_cohort([Window(i,False) for i in range(1024)])
        self.assertFalse(result['rm_active']);self.assertEqual(result['positive_weight'],0.)
        self.assertEqual(result['negative_count'],10240)
    def test_pseudo_label_source_rejected(self):
        rows=[Window(i) for i in range(1024)];rows[2].episode.metadata['label_source']='imagined'
        with self.assertRaisesRegex(ValueError,'measured'):native_reward_cohort(rows)


class NativeCohort(unittest.IsolatedAsyncioTestCase):
    async def test_exact_four_world_consumers_release_same_global_draw(self):
        ids=[str(i) for i in range(1024)]
        cohort=native_reward_cohort([Window(i) for i in range(1024)])
        full={'sample_ids':ids,'reservation_id':'r1','reward_cohort':cohort}
        releases=[]
        def method(fn):
            async def call(*args):return fn(*args)
            return types.SimpleNamespace(remote=call)
        def describe(ticket,rank,size,consumer):return {'capture_meta':shard_metadata(full,rank*size,(rank+1)*size,rank,4)}
        def project(ticket,rank,size,start,end):
            meta=shard_metadata(full,rank*size+start,rank*size+end,rank,4)
            return ([0]*4,[0]*4,[0]*4,['task']*4,{'capture_meta':meta})
        replay=types.SimpleNamespace(reserve_windows=method(lambda size:{'ticket_id':'one','global_batch_size':size}),
            describe_window_shard=method(describe),project_window_microshard=method(project),
            release_windows=method(lambda ticket:releases.append(ticket)))
        coordinator=GlobalBatchCoordinator(None,replay,4,('ctrl',),pipeline_enabled=True,world_streaming=True)
        used=[]
        for rank in range(4):
            stream=await coordinator.reserve_wm_stream(rank,256)
            self.assertEqual(stream['capture_meta']['reward_cohort'],cohort)
            local=[]
            for start in range(0,256,4):
                local.extend((await coordinator.sample_wm_stream(stream['stream_id'],start,start+4))[-1]['capture_meta']['sample_ids'])
            self.assertEqual(local,ids[rank*256:(rank+1)*256]);used.extend(local)
            await coordinator.complete_wm_stream(stream['stream_id'])
            self.assertEqual(len(releases),int(rank==3))
        self.assertEqual(len(set(used)),1024)
        with self.assertRaisesRegex(ValueError,'Unknown WM'):await coordinator.sample_wm(0,256,'reward')


class ScoreConsumption(unittest.TestCase):
    def result(self):return {'observations':[1,2],'predicted_states':[3,4],
        'reward_predictions':[.4,.99],'terminal_predictions':[False,False],
        'reward_semantics':'delta_native_score','reward_calibrated':False}
    def test_score_high_does_not_stop_and_delta_is_explicit(self):
        result=consume_predicted_chunk(self.result(),.2,2.,reward_semantics='delta_native_score')
        self.assertAlmostEqual(result[0],1.58);self.assertFalse(result[2]);self.assertEqual(result[-1],2)
    def test_no_thresholding_or_external_reward_or_fake_success_probabilities(self):
        for mutate in (lambda r:r.update(terminal_predictions=[False,True]),
                       lambda r:r.update(success_probabilities=[.4,.99]),
                       lambda r:r.update(reward_predictions=[.4,1.1]),
                       lambda r:r.update(reward_calibrated=True)):
            raw=self.result();mutate(raw)
            with self.assertRaises(ValueError):consume_predicted_chunk(raw,.2,reward_semantics='delta_native_score')
        with self.assertRaises(ValueError):consume_predicted_chunk(self.result(),.2,external_reward=lambda x:(1,True),reward_semantics='delta_native_score')


class ProductionImagine(unittest.TestCase):
    def run_seed(self,terminal_anchor=False):
        from pipeline_imagination import imagine_weaver_seed
        calls=[];batches=[]
        class Tensor(np.ndarray):
            def float(self):return self.astype(np.float32)
        torch=types.SimpleNamespace(from_numpy=lambda a:np.asarray(a).view(Tensor))
        hooks=types.ModuleType('capture_hooks')
        hooks.lineage_begin=lambda *a,**k:None;hooks.lineage_end=lambda *a:'group'
        hooks.lineage_frame_parent=lambda *a:();hooks.lineage_sample=lambda *a,**k:None
        def rpc(worker,role,endpoint,method,*args,**kwargs):
            calls.append((role,method,kwargs.get('_capture_context',{}).get('workflow')))
            if role=='reward_infer':raise AssertionError('Removed external Reward called')
            if role=='policy_infer':
                return None,np.zeros((2,7),np.float32),np.array([1]),np.zeros(3),(.4 if calls[-1][-1]=='imagined_bootstrap' else .1)
            if method=='init_rollout':return {'version':0,'initial_reward_score':.2}
            emitted=sum(c[0]=='ctrl_infer' and c[1]=='request' for c in calls)
            scores=[.3,.4] if emitted==1 else [.5,.6]
            return {'observations':np.zeros((2,2,3,2,2)), 'predicted_states':np.zeros((2,8)),
                'next_state':args[0],'served_version':0,'reward_predictions':scores,
                'terminal_predictions':[False,False],'reward_semantics':'delta_native_score','reward_calibrated':False}
        hooks.profile_rpc=rpc
        tree=ast.parse((ROOT/'train.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RolloutWorkerActor')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_process_traj')
        namespace={'np':np,'Experience':lambda **k:types.SimpleNamespace(sample_id='sample',**k),
            'lineage_begin':hooks.lineage_begin,'lineage_end':hooks.lineage_end,'lineage_sample':hooks.lineage_sample,
            'ray':types.SimpleNamespace(get=lambda v:v)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'actual GAE','exec'),namespace)
        worker=types.SimpleNamespace(world_config=WorldBackendConfig.from_mapping(config()),fresh_config=FreshConfig(),num_step_cond=32,
            ctrl_world_infer_stride=8,num_policy_chunks=2,capture_last_versions={},local_buffer=[],wid=0,
            gamma=.9,lambda_=.8,reward_scale=1.,infer=object(),ctrl_world_infer=object(),
            obs2inp=lambda obs,task:obs,stats_actor=types.SimpleNamespace(add_imagine_reward=types.SimpleNamespace(remote=lambda *a:None)),
            replay=types.SimpleNamespace(add_batch=types.SimpleNamespace(remote=lambda values:batches.extend(values))))
        worker._process_traj=types.MethodType(namespace['_process_traj'],worker)
        job={'real_anchor_terminal':terminal_anchor,'instruction':'task','episode_id':'real',
            'start':0,'frame_groups':[],'frames':np.zeros((32,2,3,2,2),np.uint8),
            'states':np.zeros((32,8)),'actions':np.zeros((32,7))}
        with mock.patch.dict(sys.modules,{'torch':torch,'capture_hooks':hooks}):imagine_weaver_seed(worker,job)
        return calls,batches
    def test_production_score_path_no_reward_rpc_and_finite_horizon_actual_gae_bootstrap(self):
        calls,batches=self.run_seed()
        self.assertEqual(len(batches),2)
        self.assertAlmostEqual(batches[1].value_target,.56,places=6)
        self.assertAlmostEqual(batches[0].value_target,.6212,places=6)
        self.assertTrue(any(c[-1]=='imagined_bootstrap' for c in calls))
        self.assertFalse(any(c[0]=='reward_infer' for c in calls))
        self.assertTrue(all(b.model_versions['world_reward_bundle_version']==0 for b in batches))
    def test_actual_terminal_or_truncated_real_anchor_produces_no_imagined_sample(self):
        calls,batches=self.run_seed(True);self.assertEqual(calls,[]);self.assertEqual(batches,[])


if __name__=='__main__':unittest.main()
