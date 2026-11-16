"""Run actual main through budget/manual/plateau completion without GPUs.

Only actor/model transport and fixed-case inference are inert. The real driver,
runtime admission, controller and stop/snapshot ordering are exercised together.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import test_main_startup_cpu as startup
from test_model_free_loop_cpu import ResultRef
from model_free_runtime import ModelFreeRuntime
from split_serial import epoch_groups
from training_topology import combine_wm_metrics


class BoundaryStopPolicy(startup.Policy):
    def deadline_reached(self):
        return False

    def should_continue(self, **state):
        if state['global_step'] >= 8:
            raise AssertionError('Automatic stop never reached the main driver')
        return not state['requested_stop']

    def target_stop_active(self, *args):
        return False

    def stop_reason(self, **state):
        return 'requested_stop' if state.get('requested_stop') else 'unexpected_stop'


class ModelFreeStoppingDriverTests(unittest.TestCase):
    def run_driver(self, scenario, *, suppress_boundary=False):
        context = startup.Context()
        state = dict(consumed=0, step=0, paused=False, stopped=False, requested=False,
                     fill_reads=0, admissions=[], epochs=[], validations=[])
        context.ray.shutdown = lambda: context.actions.append(('shutdown',))
        namespace = context.namespace()
        namespace.update(TrainingStopPolicy=BoundaryStopPolicy, np=np,
            combine_wm_metrics=combine_wm_metrics, epoch_groups=epoch_groups,
            validate_sample_group=lambda *a, **k: {},
            stop_requested=lambda: state['requested'],
            NsightCaptureWindows=lambda *a: SimpleNamespace(before_epoch=lambda *a: None,
                after_epoch=lambda *a: None, close=lambda: None))
        namespace['os'].environ.update(ACCERL_CAPTURE_ONLY='1', ACCERL_SNAPSHOT_INTERVAL='50')
        original_save = context.save

        def save(*args, **kwargs):
            context.actions.append(('snapshot', kwargs['reason']))
            return original_save(*args, **kwargs)

        namespace['save_split_weights'] = save

        def get(value, **kwargs):
            if isinstance(value, ResultRef):
                return value.value
            return [get(item) for item in value] if isinstance(value, list) else value

        context.ray.get = get
        context.ray.wait = lambda refs, **kw: (([refs[0]], refs[1:])
            if refs and isinstance(refs[0], ResultRef) else ([], refs))
        startup.execute_definition(startup.definition(startup.TREE, 'merge_split_epoch_results'), namespace)
        original_call = startup.Actor.call

        def pipeline_status():
            scheduler = dict(max_env_steps=1000000, consumed_env_steps=state['consumed'],
                reserved_env_steps=0, available_env_steps=1000000 - state['consumed'],
                episodes_inflight=0, conservative_unknown_env_steps=0, episodes_cancelled=0)
            return dict(budget=dict(config={'real_env_step_budget': 1000000}, active_real=0,
                reset_steps=10, budget_exhausted=state['consumed'] >= 999990,
                paused=state['paused'], stopped=state['stopped'], failure=None), initial_states=scheduler)

        def call(actor, method, args, kwargs):
            if method in ('configure_fresh', 'set_fresh_versions', 'set_fresh_replay_versions'):
                return True
            if method == 'pipeline_status':
                return pipeline_status()
            if method == 'initial_state_checkpoint':
                return {'enabled': True, 'pool': None, 'scheduler': None}
            if method == 'set_paused':
                state['paused'] = args[0]
                context.actions.append(('paused', args[0]))
                return pipeline_status()
            if method == 'stop_pipeline':
                state['stopped'] = True
                context.actions.append(('stop_pipeline',))
                return pipeline_status()
            if method == 'size' and actor.name == 'ReplayBufferActor':
                state['fill_reads'] += 1
                if scenario == 'fill_budget':
                    state['consumed'] = 1000000
                    return 0
                if scenario == 'boundary_budget':
                    state['consumed'] = 1000000
                return 10000
            if method == 'admit_model_free_batch':
                state['admissions'].append(args[0])
                if scenario == 'empty_manual':
                    state['requested'] = True
                    return {'ready': False}
                return {'ready': True, 'batch_size': args[0]}
            if method == 'run_training_epoch':
                actor.step = state['step'] = args[0]
                state['epochs'].append((actor.rank, args[0]))
                if scenario == 'plateau':
                    state['consumed'] = args[0] * 100000
                return ResultRef((0., 0., 0., 0., 0., {'policy': 1e-5, 'value': 1e-4},
                    actor.step, 0., 0., {'policy_sample_time': 0., 'policy_prep_time': 0.,
                    'policy_train_time': 0.}, {}))
            if method == 'serial_sample_identity':
                return {}
            result = original_call(actor, method, args, kwargs)
            if method == 'get_training_state':
                result['policy_steps'] = actor.__dict__.get('step', 0)
                result['engines'][actor.kind]['global_steps'] = result['policy_steps']
            return result

        def inventory(actor, kind):
            return dict(kind=kind, version=actor.__dict__.get('step', 0),
                        hash='a' * 64, bytes=64, tensor_count=1)

        def validate(runtime, policy_step, *, force=False):
            runtime.status()
            consumed = runtime.last_budget['consumed_env_steps']
            if runtime.control.should_validate(consumed, policy_step, force=force):
                runtime.control.observe(10, policy_step=policy_step, consumed_env_steps=consumed)
                runtime.last_result = {'success_count': 10, 'policy_version': policy_step}
                state['validations'].append(policy_step)
                context.actions.append(('validation', policy_step))
            if force or runtime.control.plateau_reached:
                state['paused'] = True
            return runtime.last_result

        args = startup.arguments(overrides=('--model-free', '--num-eval-workers', '0',
            '--pipeline-config', str(startup.HERE.parent/'recipe/pipeline_config.json'),
            '--fixed-validation-interval-env-steps', '100000' if scenario == 'plateau' else '0'))
        args.log_interval_seconds = float('inf')
        boundary_patch = (patch.object(ModelFreeRuntime, 'before_update', return_value=None)
                          if suppress_boundary else nullcontext())
        with patch.object(startup.Actor, 'call', call), patch.object(startup.Actor, 'inventory', inventory), \
                patch('checkpoint_evaluation.preserve_initial_checkpoint'), \
                patch('model_free_runtime.time.sleep', return_value=None), \
                patch.object(ModelFreeRuntime, 'validate', validate), boundary_patch:
            startup.execute_definition(startup.definition(startup.TREE, 'main'), namespace)(args)
        events = [(kind, fields) for writer in context.telemetries for kind, fields in writer.events]
        completion = [fields['details'] for kind, fields in events if kind == 'experiment_completed']
        self.assertEqual(len(completion), 1)
        self.assertTrue(context.summaries[0].closed)
        self.assertEqual(context.actions[-1], ('shutdown',))
        return context, state, events, completion[0]

    def test_budget_exhausted_during_initial_fill_saves_without_any_update(self):
        context, state, events, complete = self.run_driver('fill_budget')
        self.assertEqual(state['epochs'], [])
        self.assertEqual(state['admissions'], [])
        self.assertEqual((complete['global_step'], complete['completed_epochs']), (0, 0))
        self.assertEqual(complete['stop_reason'], 'real_budget_exhausted')
        self.assertTrue(complete['budget_status']['budget_exhausted'])
        self.assertEqual([row[2]['reason'] for row in context.snapshots],
                         ['initial', 'final_before_first_update'])
        self.assertEqual(len([kind for kind, fields in events if kind == 'fill_snapshot']), 1)

    def test_budget_exhausted_after_fill_never_starts_first_collective(self):
        context, state, events, complete = self.run_driver('boundary_budget')
        self.assertEqual(state['epochs'], [])
        self.assertEqual(state['admissions'], [])
        self.assertEqual(complete['stop_reason'], 'real_budget_exhausted')
        self.assertEqual([row[2]['reason'] for row in context.snapshots], ['initial', 'final'])
        self.assertTrue(any(kind == 'replay_ready' for kind, _ in events))

    def test_empty_eligible_replay_manual_stop_never_starts_any_collective(self):
        context, state, events, complete = self.run_driver('empty_manual')
        self.assertEqual(state['admissions'], [576])
        self.assertEqual(state['epochs'], [])
        self.assertEqual(complete['stop_reason'], 'requested_stop')
        self.assertEqual(complete['global_step'], 0)
        self.assertEqual([row[2]['reason'] for row in context.snapshots], ['initial', 'final'])

    def test_plateau_stops_before_fourth_update_and_final_snapshot_keeps_latest_policy(self):
        context, state, events, complete = self.run_driver('plateau')
        self.assertEqual(state['validations'], [0, 1, 2, 3])
        self.assertEqual(state['epochs'], [(rank, step) for step in (1, 2, 3) for rank in range(4)])
        self.assertEqual(state['admissions'], [576, 576, 576])
        self.assertEqual(complete['stop_reason'], 'fixed_validation_plateau')
        self.assertEqual((complete['global_step'], complete['completed_epochs']), (3, 3))
        self.assertEqual([row['policy_steps'] for row in complete['states']], [3] * 4)
        final = context.snapshots[-1][2]
        self.assertEqual(final['reason'], 'final')
        self.assertEqual(final['metadata']['fixed_validation']['last']['policy_step'], 3)
        self.assertEqual(final['metadata']['fixed_validation']['best_policy_step'], 0)
        self.assertLess(context.actions.index(('validation', 3)), context.actions.index(('stop_pipeline',)))
        self.assertLess(context.actions.index(('snapshot', 'final')),
                        context.actions.index(('event', 'experiment_completed')))

    def test_regression_removing_preupdate_gate_is_detected_by_actual_driver(self):
        with self.assertRaisesRegex(AssertionError, 'Automatic stop never reached'):
            self.run_driver('boundary_budget', suppress_boundary=True)


if __name__ == '__main__':
    unittest.main()
