"""Reject mixed pink/orange recipes before a managed GPU job can start."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('orange_admission_training', HERE / 'training_entry.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class OrangeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.package = Path(self.temp.name).resolve()
        self.source = self.package / 'engine'
        self.recipe = self.package / 'recipe'
        self.root = self.package / 'output'
        for p in (self.source, self.recipe, self.root):
            p.mkdir()
        self.scheduler = dict(schema=1, enabled=True, macro_enabled=True, allow_probe=True,
                              training_gangs=1, card_inflight=2)
        self.pipeline = dict(training.PIPELINE_RECIPE, policy_high_watermark=4096,
                             policy_low_watermark=2048, generation_per_sample=.25)
        self.write_recipes()

    def write_recipes(self):
        for name, value in [('cost_scheduler_config.json', self.scheduler),
                            ('pipeline_config.json', self.pipeline)]:
            (self.recipe / name).write_text(json.dumps(value), encoding='utf8')

    def state(self):
        scheduler = training.validate_cost_scheduler_admission(self.source)
        scheduler.update(configuration_path=str(self.root / 'formal_v3/cost_scheduler_config.json'),
                         configuration_sha256=scheduler['source_sha256'])
        return dict(cost_scheduler=scheduler, pipeline_config=copy.deepcopy(self.pipeline),
                    pipeline_config_path=str(self.root / 'formal_v3/pipeline_config.json'))

    def test_orange_scheduler_and_formal_training_horizon_activate_together(self):
        self.assertEqual(training.validate_cost_scheduler_admission(self.source)['profile'], '03_adaptive')
        self.assertTrue(training.check_cost_scheduler_state(self.state(), self.source, self.root))

    def test_parallel_training_or_extra_card_lane_is_rejected(self):
        for key, value in [('training_gangs', 2), ('training_gangs', True), ('card_inflight', 3)]:
            with self.subTest(key=key, value=value):
                old = self.scheduler[key]
                self.scheduler[key] = value
                self.write_recipes()
                with self.assertRaises(ValueError):
                    training.validate_cost_scheduler_admission(self.source)
                self.scheduler[key] = old

    def test_later_execution_and_feedback_extensions_are_rejected_even_if_disabled(self):
        for key in ('execution', 'cycle_feedback'):
            with self.subTest(key=key):
                self.scheduler[key] = {'enabled': False}
                self.write_recipes()
                with self.assertRaises(ValueError):
                    training.validate_cost_scheduler_admission(self.source)
                self.scheduler.pop(key)

    def test_scheduler_cannot_silently_disable(self):
        for key in ('enabled', 'macro_enabled', 'allow_probe'):
            with self.subTest(key=key):
                self.scheduler[key] = False
                self.write_recipes()
                with self.assertRaises(ValueError):
                    training.validate_cost_scheduler_admission(self.source)
                self.scheduler[key] = True

    def test_collection_censor_cap_cannot_silently_replace_formal_horizon(self):
        for value in (150, 220, 150.0, True, False, 0):
            p = dict(self.pipeline, real_collection_action_limit=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                training.validate_pipeline_admission(p)
        p.pop('real_collection_action_limit')
        with self.assertRaises(ValueError):
            training.validate_pipeline_admission(p)

    def test_training_horizon_is_required_and_strictly_integer_150(self):
        for value in (None, 220, 150.0, '150', True, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                training.validate_pipeline_admission(dict(self.pipeline, real_training_action_horizon=value))
        p = dict(self.pipeline)
        p.pop('real_training_action_horizon')
        with self.assertRaises(ValueError):
            training.validate_pipeline_admission(p)

    def test_managed_recipe_uses_formal_150_not_the_old_censor_recipe(self):
        recipe = training.read_json(HERE.parents[1] / 'recipe/pipeline_config.json')
        self.assertTrue(training.validate_pipeline_admission(recipe))
        self.assertIsNone(recipe['real_collection_action_limit'])
        self.assertEqual(recipe['real_training_action_horizon'], 150)
        reference = training.read_json(HERE.parents[1] / 'recipe/reference.json')
        self.assertTrue(reference['training_horizon_failure_labels_valid'])
        self.assertTrue(reference['training_horizon_failure_is_terminal'])
        self.assertFalse(reference['training_horizon_failure_bootstrap'])
        self.assertEqual(reference['training_horizon_failure_reward_target'], 0.0)
        self.assertEqual(reference['final_evaluation_action_horizon'], 220)

    def test_lookahead_is_rejected_even_when_zero(self):
        for value in (0, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                training.validate_pipeline_admission(dict(self.pipeline, fresh_lookahead_updates=value))

    def test_real_budget_pool_and_source_mix_contracts_remain_fixed(self):
        for key, value in [('real_workers', 2), ('imagination_workers', 2),
                           ('initial_state_count', 49), ('real_env_step_budget', 2000000)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                training.validate_pipeline_admission(dict(self.pipeline, **{key: value}))

    def test_runner_cannot_report_original_recipe_but_activate_another_budget(self):
        for key, value in [('real_collection_action_limit', 150), ('real_training_action_horizon', 220)]:
            state = self.state()
            state['pipeline_config'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                training.check_cost_scheduler_state(state, self.source, self.root)
        state = self.state()
        state['pipeline_config'].pop('real_collection_action_limit')
        with self.assertRaises(ValueError):
            training.check_cost_scheduler_state(state, self.source, self.root)

    def test_runner_scheduler_settings_and_pipeline_path_must_match(self):
        state = self.state()
        state['cost_scheduler']['settings']['training_gangs'] = 2
        with self.assertRaises(ValueError):
            training.check_cost_scheduler_state(state, self.source, self.root)
        state = self.state()
        state['pipeline_config_path'] = '/unbound/pipeline.json'
        with self.assertRaises(ValueError):
            training.check_cost_scheduler_state(state, self.source, self.root)

    def test_runner_fresh_contract_rejects_jit_or_trajectory_fields(self):
        state = dict(run_id='weaverintegrated_test', candidate='weaver_integrated_v1',
                     source_path=str(self.source), experiment_root=str(self.root), uid=1004,
                     mode='manual', until_manual_stop=True, stop_on_target=False, deadline_unix=None,
                     world_backend_config={'backend_options': {'fresh': dict(training.FRESH_OPTIONS)}})
        self.assertEqual(training.check_state(state, self.source, self.root), state['run_id'])
        for key, value in [('jit_gae_mode', 'fused'), ('advantage_norm', 'ema_zscore'),
                           ('policy_sampling', 'imagined_trajectories')]:
            fresh = state['world_backend_config']['backend_options']['fresh']
            fresh[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                training.check_state(state, self.source, self.root)
            fresh.pop(key)

    def test_backend_admission_accepts_orange_and_rejects_mixed_fresh_recipe(self):
        backend = dict(backend='weaver', reward_provider='world_head',
                       reward_semantics='delta_native_score', terminal_rule='horizon_only',
                       critic_mode='disabled', world_trainable=True,
                       checkpoint_metadata={'load_report': {'checkpoint_sha256':
                           training.INITIAL_WORLD['source_bundle_sha256']}},
                       backend_options=dict(native_reward_head=dict(training.NATIVE_REWARD),
                           fresh=dict(training.FRESH_OPTIONS), optimizations=dict(training.OPTIMIZATIONS),
                           training_python='/native/python'))
        admission = dict(configuration_admitted=True, assets_checked=True, errors=[],
                         environment={'compatible': True}, online_training_environment=dict(
                             compatible=True, distributed_available=True, nccl_available=True,
                             cuda_initialized=False, gpu_access_performed=False, python='/native/python'))
        self.assertTrue(training.validate_backend_admission(backend, self.pipeline, admission))
        backend['backend_options']['fresh']['jit_gae_mode'] = 'off'
        with self.assertRaises(ValueError):
            training.validate_backend_admission(backend, self.pipeline, admission)
        backend['backend_options']['fresh'].pop('jit_gae_mode')
        with self.assertRaises(ValueError):
            training.validate_backend_admission(backend, dict(self.pipeline, real_collection_action_limit=220), admission)

    def test_git_identity_is_preserved_instead_of_a_source_snapshot(self):
        package = HERE.parents[1]
        commit = 'a' * 40
        fake = SimpleNamespace(revision=lambda root, expected: expected)
        with patch.dict(sys.modules, {'git_version': fake}):
            self.assertEqual(training.verify_git(package / 'engine', commit), {'git_commit': commit})
        with self.assertRaises(ValueError):
            training.verify_git(self.source, commit)

    def test_manual_and_error_envelopes_are_unchanged(self):
        contract = training.validate_manual_contract()
        self.assertIsNone(contract['deadline_ns'])
        self.assertEqual(training.ERROR_WORK_GRACE_SECONDS, 120)
        self.assertEqual(training.MANUAL_GPU_SAVE_GRACE_SECONDS, 900)
        self.assertEqual(training.ERROR_TEARDOWN_GRACE_SECONDS, 90)


if __name__ == '__main__':
    unittest.main()
