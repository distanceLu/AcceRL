"""Run the actual driver through six epochs with inert Ray/model boundaries.

This complements the startup and real-replay tests: it exercises the complete
driver loop, including ordinary step five, metrics, publication and final stop.
It does not construct models, launch Ray, access a simulator or use a GPU.
"""
from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import test_main_startup_cpu as startup
from split_serial import epoch_groups
from training_topology import combine_wm_metrics


class ResultRef:
    def __init__(self, value):
        self.value = value


class SixEpochStopPolicy(startup.Policy):
    def deadline_reached(self):
        return False
    def should_continue(self, **state):
        return state['global_step'] < 6

    def target_stop_active(self, *args):
        return False

    def stop_reason(self, **state):
        return 'scheduled_stop'


class ModelFreeDriverLoop(unittest.TestCase):
    def test_six_epochs_log_publish_save_and_stop_without_world_owners(self):
        self.run_loop(autotune=False)

    def test_enabled_autotune_configures_all_ranks_and_records_a_real_window(self):
        self.run_loop(autotune=True)

    def run_loop(self, *, autotune):
        context = startup.Context()
        namespace = context.namespace()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        namespace['os'].environ['ACCERL_DATA_DIR'] = temporary.name
        namespace.update(TrainingStopPolicy=SixEpochStopPolicy, np=np,
                         combine_wm_metrics=combine_wm_metrics, epoch_groups=epoch_groups,
                         # Real shard identities are covered by resource tests;
                         # the inert trainers here return no sample payloads.
                         validate_sample_group=lambda *args, **kwargs: {})
        namespace['NsightCaptureWindows'] = lambda *args: SimpleNamespace(
            before_epoch=lambda *args: None, after_epoch=lambda *args: None,
            close=lambda: None)
        namespace['os'].environ.update(ACCERL_CAPTURE_ONLY='1', ACCERL_SNAPSHOT_INTERVAL='5')
        context.ray.shutdown = lambda: None

        def get(value, **kwargs):
            if isinstance(value, ResultRef):
                return value.value
            return [get(item) for item in value] if isinstance(value, list) else value

        def wait(refs, **kwargs):
            return ([refs[0]], refs[1:]) if refs and isinstance(refs[0], ResultRef) else ([], refs)

        context.ray.get, context.ray.wait = get, wait
        startup.execute_definition(startup.definition(startup.TREE, 'merge_split_epoch_results'), namespace)
        original = startup.Actor.call

        def call(actor, method, args, kwargs):
            if method == 'configure_batch_autotune':
                actor.autotune_choice = args
                context.actions.append(('autotune_configure',actor.name,actor.rank,args))
                if actor.name == 'ReplayBufferActor':return args[0]
                key='micro' if actor.name == 'TrainerActor' else 'refresh_micro'
                return dict(rank=actor.rank,policy_step=args[0],**{key:args[1]})
            if method == 'get_batch_autotune_stats':
                step,micro=actor.autotune_choice
                self.assertEqual(step,args[0])
                memory=dict(total_bytes=140*2**30,peak_used_bytes=70*2**30,activation_bytes=10*2**30)
                if actor.name == 'TrainerActor':
                    return dict(rank=actor.rank,policy_step=step,micro=micro,train_seconds=40.,memory=memory)
                return dict(rank=actor.rank,policy_step=step,refresh_micro=micro,refresh_seconds=20.,
                            refresh_observations=100,action_latencies=[.1,.3],memory=memory)
            if method in ('configure_fresh', 'set_fresh_versions', 'set_fresh_replay_versions', 'stop_pipeline'):
                return True
            if method == 'run_training_epoch':
                if autotune:
                    self.assertEqual(actor.autotune_choice,(args[0],8))
                actor.step = args[0]
                return ResultRef((0., 0., 0., 0., 0., {'policy': 1e-5, 'value': 1e-4},
                                  actor.step, 0., 0., {'policy_sample_time': 0.,
                                  'policy_prep_time': 0., 'policy_train_time': 0.}, {}))
            if method == 'serial_sample_identity':
                return {}
            if method == 'get_stats':
                values = dict(avg_return=0., avg_ep_len=150, total_episodes_processed=10,
                              total_env_steps=1500, avg_step_time=.01, avg_success_rate=.17)
                return {'_global_rollout_': dict(values), '_global_eval_': dict(values)}
            if method == 'pipeline_status':
                return {}
            if method == 'admit_model_free_batch':
                return {'ready': True, 'batch_size': args[0]}
            result = original(actor, method, args, kwargs)
            if method == 'get_training_state':
                result['policy_steps'] = actor.__dict__.get('step', 0)
                result['engines'][actor.kind]['global_steps'] = result['policy_steps']
            return result

        def inventory(actor, kind):
            return {'kind': kind, 'version': actor.__dict__.get('step', 0),
                    'hash': 'a' * 64, 'bytes': 64, 'tensor_count': 1}

        args = startup.arguments(overrides=('--model-free', '--num-eval-workers', '0',
                                             '--fixed-validation-interval-env-steps', '0')
                                            + (('--batch-autotune',) if autotune else ()))
        args.log_interval_seconds = -1  # Exercise logging on every completed epoch.
        args.pipeline_config = startup.HERE.parent/'recipe/pipeline_config.json'
        with patch.object(startup.Actor, 'call', call), patch.object(startup.Actor, 'inventory', inventory), \
                patch('checkpoint_evaluation.preserve_initial_checkpoint'):
            startup.execute_definition(startup.definition(startup.TREE, 'main'), namespace)(args)

        events = [(kind, fields) for writer in context.telemetries for kind, fields in writer.events]
        epochs = [fields['details'] for kind, fields in events if kind == 'synchronized_epoch_completed']
        self.assertEqual([epoch['global_step'] for epoch in epochs], list(range(1, 7)))
        self.assertTrue(all(epoch['rank_steps'] == [epoch['global_step']] * 4 for epoch in epochs))
        self.assertTrue(all(not epoch['trained_ctrl'] and not epoch['trained_reward'] for epoch in epochs))
        publications = [fields for kind, fields in events if kind == 'weight_publication_committed']
        self.assertEqual([row['model_version'] for row in publications], list(range(7)))
        self.assertTrue(all(row['details']['kind'] == 'policy' for row in publications))
        self.assertEqual([row[2]['reason'] for row in context.snapshots], ['initial', 'periodic', 'final'])
        self.assertTrue(all(set(row[0]) == {'policy'} and row[2]['metadata']['model_free']
                            for row in context.snapshots))
        completion = [fields['details'] for kind, fields in events if kind == 'experiment_completed']
        self.assertEqual(len(completion), 1)
        self.assertEqual((completion[0]['global_step'], completion[0]['completed_epochs']), (6, 6))
        self.assertEqual(completion[0]['stop_reason'], 'scheduled_stop')
        self.assertEqual([state['policy_steps'] for state in completion[0]['states']], [6] * 4)
        scalars = context.summaries[0].scalars
        for tag in ('Loss/Policy', 'Loss/Value', 'Rollout/_Global/Success_Rate'):
            self.assertEqual([row[0][2] for row in scalars if row[0][0] == tag], list(range(1, 7)))
        self.assertEqual([row[0][1] for row in scalars if row[0][0] == 'Rollout/_Global/Success_Rate'],
                         [0.] * 6)  # Strict cross-task window, independently of legacy per-task averages.
        self.assertTrue(context.summaries[0].closed)
        observations=[fields['details'] for kind,fields in events if kind == 'batch_autotune_observation']
        if autotune:
            self.assertEqual([x['policy_step'] for x in observations],list(range(1,7)))
            record=json.loads((Path(temporary.name)/'batch_autotune.json').read_text())
            self.assertEqual(record['decision']['next'],{'train_micro':8,'refresh_micro':4})
            self.assertEqual(record['decision']['window']['measured_updates'],5)
            configurations=[x for x in context.actions if x[0]=='autotune_configure']
            self.assertEqual(len(configurations),6*9)  # Four trainers, four infer, one Replay per update.
        else:
            self.assertEqual(observations,[])


if __name__ == '__main__':
    unittest.main()
