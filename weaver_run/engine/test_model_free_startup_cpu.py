"""Exercise actual startup wiring while replacing only model/Ray boundaries."""
from collections import Counter
import ast
import copy
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import test_main_startup_cpu as startup
from world_backends.runtime import load_backend_config
from model_free_mode import configure_training_mode, pipeline_configuration


class ModelFreeStartupTests(unittest.TestCase):
    def run_startup(self, evaluation=0):
        args = startup.arguments(overrides=(
            '--model-free', '--num-eval-workers', str(evaluation),
            '--eval-inference-replicas', '', '--world-backend-config', 'DOES_NOT_EXIST.json'))
        context = startup.Context()
        original = startup.Actor.call
        def call(actor, method, args, kwargs):
            if method in ('configure_fresh', 'set_fresh_versions', 'set_fresh_replay_versions'):
                return True
            return original(actor, method, args, kwargs)
        with patch.object(startup.Actor, 'call', call):
            state = startup.execute_definition(startup.startup_ast(), context.namespace())(args)
        return context, state

    def test_switch_never_reads_world_assets(self):
        args = startup.arguments(overrides=('--model-free', '--world-backend-config', 'MISSING.json'))
        self.assertEqual(load_backend_config(args).backend, 'model_free')

    def test_model_free_default_eval_and_saved_training_budget(self):
        args = startup.arguments(overrides=('--model-free',))
        args.num_eval_workers = None
        configure_training_mode(args)
        self.assertEqual(args.num_eval_workers, 0)
        args.pipeline_config = startup.HERE.parent/'recipe/pipeline_config.json'
        config = pipeline_configuration(args)
        self.assertEqual((config.real_workers, config.max_real_in_flight, config.imagination_workers), (4, 4, 0))
        self.assertEqual((config.initial_state_count, config.real_training_action_horizon,
                          config.real_env_step_budget), (50, 150, 1000000))
        self.assertFalse(config.inventory_refill_enabled)
        self.assertIsNone(config.real_collection_action_limit)

    def test_serial_role_schedule_is_rejected_before_any_actor(self):
        args = startup.arguments(overrides=('--model-free', '--num-eval-workers', '0'))
        context = startup.Context()
        namespace = context.namespace()
        namespace['serial_enabled'] = lambda *a, **k: True
        with self.assertRaisesRegex(ValueError, 'concurrent Policy'):
            startup.execute_definition(startup.startup_ast(), namespace)(args)
        self.assertEqual(context.actors, [])

    def test_real_only_startup_reaches_first_fill_and_policy_snapshot(self):
        context, state = self.run_startup()
        counts = Counter(actor.name for actor in context.actors)
        self.assertEqual(counts['TrainerActor'], 4)
        self.assertEqual(counts['InferenceActor'], 4)
        self.assertEqual(counts['RolloutWorkerActor'], 4)
        for name in ('WMReplayBufferActor', 'RewardInferenceActor', 'CtrlWorldInferenceActor',
                     'WeaverWorldInferenceActor', 'EvaluationWorkerActor'):
            self.assertEqual(counts[name], 0, name)
        self.assertEqual(counts['RealCollectionCoordinator'], 1)
        self.assertEqual(state['active_kinds'], ('policy',))
        self.assertEqual(state['wm_replay_buffers'], [])
        self.assertIsNone(state['threshold_tracker'])
        self.assertEqual(set(context.snapshots[0][0]), {'policy'})
        self.assertEqual(state['wm_min_buffer_size'], 0)
        self.assertEqual(state['cfg'].num_images_in_input, 2)
        self.assertFalse(state['cfg'].use_proprio)
        workers = [a for a in context.actors if a.name == 'RolloutWorkerActor']
        self.assertEqual([a.positional[0].kwargs['actor_id'] for a in workers], [0, 1, 2, 3])
        self.assertTrue(all(a.kwargs['worker_mode'] == 'real' for a in workers))
        self.assertTrue(all(a.positional[10] is None for a in workers))

    def test_optional_four_eval_workers_route_one_per_replica(self):
        context, state = self.run_startup(4)
        workers = [a for a in context.actors if a.name == 'EvaluationWorkerActor']
        self.assertEqual([a.positional[0].kwargs['actor_id'] for a in workers], [0, 1, 2, 3])
        self.assertEqual(state['evaluation_plan']['window_size'], 400)

    def test_completed_policy_only_epoch_can_be_merged_even_at_step_five(self):
        fn = startup.execute_definition(startup.definition(startup.TREE, 'merge_split_epoch_results'), {})
        policy = [(0., 0., 0., 0., 0., {'policy': 1e-5, 'value': 1e-4}, 5, 0., 0.,
                   {'policy_train_time': 1.}, {}) for _ in range(4)]
        result = fn(policy, 4, 5, 5, 5, ('policy',))
        self.assertEqual(len(result), 4)
        self.assertTrue(all(row[6] == 5 and not row[10]['trained_ctrl'] for row in result))

    def test_actual_summary_omits_absent_services_and_keeps_model_free_measurements(self):
        cls = copy.deepcopy(next(n for n in startup.TREE.body if isinstance(n, ast.ClassDef)
                                 and n.name == 'CaptureSummaryWriter'))
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                     '<actual summary>', 'exec'), namespace)
        writer = object.__new__(namespace['CaptureSummaryWriter'])
        events, scalars = [], []
        writer.model_free, writer.eval_workers = True, 0
        writer.native_world_losses = writer.native_reward_losses = False
        writer.telemetry = SimpleNamespace(event=lambda *a, **k: events.append(k))
        writer.tensorboard = SimpleNamespace(add_scalar=lambda *a, **k: scalars.append(a))
        for tag in ('Eval/_Global/Average_Return', 'System/Eval_Total_Env_Steps',
                    'Loss/World_RM', 'Loss/CtrlWorld', 'System/WM_Replay_Buffer_Size_Total',
                    'Rollout/_Global/Average_Imagine_Reward'):
            writer.add_scalar(tag, 0., 1)
        self.assertEqual(events, [])
        self.assertEqual(scalars, [])
        for tag in ('Loss/Policy', 'Loss/Value', 'Rollout/_Global/Success_Rate',
                    'Metrics/Training_Speed_Steps_per_Sec', 'System/Total_Env_Steps'):
            writer.add_scalar(tag, .8, 1)
        self.assertEqual(len(scalars), 5)
        writer.eval_workers = 4
        writer.add_scalar('Eval/_Global/Average_Return', .9, 2)
        self.assertEqual(scalars[-1][:3], ('Eval/_Global/Average_Return', .9, 2))


if __name__ == '__main__':
    unittest.main()
