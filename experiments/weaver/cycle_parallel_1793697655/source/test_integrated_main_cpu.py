"""Execute this candidate's actual main startup and P0 stop using inert actors.

This checks orchestration wiring, not CUDA or physical model admission.
"""
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
# Imported helpers are pure AST/inert actors. An integrated candidate includes
# its own copy; the local partial patch falls back to the frozen test fixture.
SUPPORT=HERE if (HERE/'test_main_startup_cpu.py').is_file() else HERE.parents[2]/'weaver_policy_v1'
if str(SUPPORT) not in sys.path:sys.path.append(str(SUPPORT))
import test_main_startup_cpu as support
from pipeline_budget import PipelineBudgetConfig
from training_stop_policy import TrainingStopPolicy
from world_backend_runtime import isolated_online_weaver, save_online_split_snapshot, validate_online_runtime_arguments

TREE=ast.parse((HERE/'train.py').read_text(encoding='utf-8'))


class IntegratedMain(unittest.TestCase):
    def smoke(self, stop_zero=False):
        args=support.arguments()
        config=json.loads((HERE/'examples/weaver_integrated_internal.json').read_text(encoding='utf-8'))
        args.world_backend_config=config
        args.ctrl_world_ckpt=None
        args.num_step_cond=32
        args.num_reward_inference_actors=0
        args.pipeline_config=PipelineBudgetConfig(real_workers=1,max_real_in_flight=1,
            imagination_workers=4,max_imagine_in_flight=4,initial_state_enabled=True,
            real_env_step_budget=1000000).to_dict()
        context=support.Context()
        namespace=context.namespace()
        namespace.update(isolated_online_weaver=isolated_online_weaver,
            save_online_split_snapshot=save_online_split_snapshot,
            validate_online_runtime_arguments=validate_online_runtime_arguments,
            WeaverOnlineTrainerActor=support.ActorClass(context,'WeaverOnlineTrainerActor'),
            WeaverOnlineWorldInferenceActor=support.ActorClass(context,'WeaverOnlineWorldInferenceActor'))
        namespace['os'].environ.update(ACCERL_RUN_ID='weaverintegrated_cpu_fixture',
            ACCERL_RUN_UNTIL_MANUAL_STOP='1',ACCERL_STOP_ON_TARGET_85='0')
        namespace['TrainingStopPolicy']=TrainingStopPolicy
        namespace['ThresholdTracker']=lambda **kwargs:types.SimpleNamespace(as_dict=lambda:{})
        namespace['stop_requested']=lambda:stop_zero
        context.ray.shutdown=lambda:context.actions.append(('shutdown',))
        original_init=support.Actor.__init__
        original_call=support.Actor.call
        def initialize(actor,*values,**keywords):
            original_init(actor,*values,**keywords)
            if actor.name in ('WeaverOnlineTrainerActor','WeaverOnlineWorldInferenceActor'):actor.kind='ctrl'
        def call(actor,method,values,keywords):
            if method=='export_snapshot_component':
                return {'kind':values[0],'policy_step':0,'model_step':0,'fake_cpu_component':True}
            if method=='export_native_snapshot':
                context.actions.append(('native_snapshot',values[1],values[2]))
                return {'protocol':'weaver_online_joint_weights_v1','trained_parameter_families':['wm','rm'],'rm_updates':0,'backend':'weaver','world_trainable':True,
                    'run_id':'weaverintegrated_cpu_fixture','world_size':4,'policy_step':0,'successful_updates':0,
                    'model_version':0,'tensor_count':866,'source_optimizer_steps':3200,
                    'source_bundle_sha256':config['backend_options']['initial_checkpoint_sha256'],
                    'weights_only':True,'weights_file':'cpu_fixture_native.pt','weights_sha256':'c'*64}
            if method=='initial_state_checkpoint':
                return {'enabled':True,'pool':{'count':50},'scheduler':{'max_env_steps':1000000}}
            if method=='stop_pipeline':return True
            return original_call(actor,method,values,keywords)
        preservation=types.ModuleType('checkpoint_evaluation')
        preservation.preserve_initial_checkpoint=mock.Mock()
        with mock.patch.object(support.Actor,'__init__',initialize),mock.patch.object(support.Actor,'call',call), \
                mock.patch.dict(sys.modules,{'checkpoint_evaluation':preservation}):
            state=support.execute_definition(support.startup_ast(tree=TREE),namespace)(args)
        return context,state
    def test_actual_main_all_eight_owners_and_inference_initial_two_syncs_save_fill_caps(self):
        context,state=self.smoke()
        counts=Counter(actor.name for actor in context.actors)
        self.assertEqual(counts['TrainerActor'],4)
        self.assertEqual(counts['WeaverOnlineTrainerActor'],4)
        self.assertEqual(counts['WeaverOnlineWorldInferenceActor'],4)
        self.assertEqual(counts['WeaverWorldInferenceActor'],0)
        self.assertEqual(counts['RewardInferenceActor'],0)
        self.assertEqual(len(state['all_trainers']),8)
        self.assertEqual(sum(actor.kind is not None for actor in context.actors),16)
        self.assertEqual(len(context.summaries),1)
        self.assertEqual(len([x for x in context.actors if x.name=='StatsActor']),1)
        self.assertEqual(len([c for c in context.calls if c[2]=='setup_deepspeed_group']),8)
        self.assertEqual(len([a for a in context.actions if a[0]=='ACK']),8)
        self.assertEqual(len([a for a in context.actions if a[0]=='release']),8)
        publications=[event[1] for event in state['experiment'].events if event[0]=='weight_publication_committed']
        self.assertEqual({item['details']['kind'] for item in publications},{'policy','ctrl'})
        self.assertEqual({item['model_version'] for item in publications},{0})
        self.assertEqual(set(context.snapshots[0][0]),{'policy'})
        self.assertEqual(context.snapshots[0][2]['metadata']['online_world']['successful_updates'],0)
        self.assertTrue(any(a[0]=='native_snapshot' for a in context.actions))
        coordinator=next(a for a in context.actors if a.name=='GlobalBatchCoordinator')
        self.assertEqual(coordinator.kwargs['wm_consumers'],('ctrl',))
        self.assertTrue(coordinator.kwargs['world_streaming'])
        self.assertEqual(Counter(a.kwargs['worker_mode'] for a in context.actors if a.name=='RolloutWorkerActor'),
                         {'real':1,'imagine':4})
        self.assertEqual(state['replay_window_frames'],40)
        self.assertEqual(state['pipeline_config'].real_env_step_budget,1000000)
        self.assertEqual(len(context.limits),8)
        self.assertTrue(any(e[0]=='replay_ready' for e in state['experiment'].events))
    def test_actual_main_stop_during_fill_saves_zero_world_without_claimed_update_and_closes(self):
        context,state=self.smoke(stop_zero=True)
        self.assertIsNone(state)
        self.assertEqual([s[2]['reason'] for s in context.snapshots],['initial','final_before_first_update'])
        self.assertEqual(context.snapshots[-1][2]['metadata']['online_world']['model_version'],0)
        events=[event for telemetry in context.telemetries for event in telemetry.events if event[0]=='experiment_completed']
        self.assertEqual(events[-1][1]['details']['global_step'],0)
        self.assertEqual(events[-1][1]['details']['stop_reason'],'scheduled_stop')
        self.assertTrue(context.summaries[0].closed)
        self.assertTrue(all(t.closed for t in context.telemetries if t.actor_id=='fill_experiment'))
        self.assertIn(('shutdown',),context.actions)


if __name__=='__main__':unittest.main()
