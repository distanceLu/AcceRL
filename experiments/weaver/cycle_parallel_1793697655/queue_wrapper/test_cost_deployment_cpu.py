"""Exercise the real launch adapter functions, without Ray, torch or CUDA."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import training_entry as wrapper


CANDIDATE = Path(os.environ.get('COST_CANDIDATE_SOURCE', os.environ.get(
    'FRESH_CANDIDATE_SOURCE', Path(__file__).resolve().parents[1] / 'staged_v1/weaver_integrated_v1'))).resolve()


def runner_functions(source):
    names = {'cost_scheduler_launch_configuration', 'freeze_cost_scheduler_configuration',
             'cost_scheduler_environment_configuration', 'formal_environment'}
    tree = ast.parse((CANDIDATE / 'runner.py').read_text(encoding='utf-8'))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    namespace = {'Path': Path, 'json': json, 'hashlib': hashlib, 'SOURCE': source,
                 'ROOT': Path('/base'), 'COST_SCHEDULER_PROFILE': wrapper.COST_SCHEDULER_PROFILE,
                 'COST_SCHEDULER_RELATIVE': wrapper.COST_SCHEDULER_CONFIG,
                 'safe': SimpleNamespace(cpu_helper_environment=lambda: {'CUDA_VISIBLE_DEVICES': ''})}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(CANDIDATE / 'runner.py'), 'exec'), namespace)
    return namespace


class CostDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'examples').mkdir()
        self.profile = self.source / wrapper.COST_SCHEDULER_CONFIG
        self.profile.write_bytes((CANDIDATE / wrapper.COST_SCHEDULER_CONFIG).read_bytes())
        (self.source / wrapper.PIPELINE_CONFIG).write_bytes((CANDIDATE / wrapper.PIPELINE_CONFIG).read_bytes())
        self.bind_manifest()
        self.namespace = runner_functions(self.source)
        self.old_path = list(sys.path)
        sys.path.insert(0, str(CANDIDATE))
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.old_path))

    def bind_manifest(self):
        digest = hashlib.sha256(self.profile.read_bytes()).hexdigest()
        (self.source / 'source_manifest.json').write_text(json.dumps({
            'candidate': 'weaver_integrated_v1', 'files': {wrapper.COST_SCHEDULER_CONFIG: digest}}), encoding='utf-8')

    def launch(self, path=None):
        return self.namespace['cost_scheduler_launch_configuration'](path or self.profile)

    def frozen(self):
        data = self.root / 'formal_v3'
        data.mkdir()
        value = self.namespace['freeze_cost_scheduler_configuration'](data, self.launch())
        self.addCleanup(lambda: (data / 'cost_scheduler_config.json').chmod(0o666))
        return data, value

    def test_explicit_adaptive_profile_is_loaded(self):
        value = self.launch()
        self.assertEqual(value['profile'], wrapper.COST_SCHEDULER_PROFILE)
        self.assertTrue(all(value['settings'][key] is True for key in ('enabled', 'macro_enabled', 'allow_probe')))

    def test_legacy_no_argument_does_not_use_inherited_environment(self):
        self.assertIsNone(self.namespace['cost_scheduler_launch_configuration']())
        self.assertEqual(self.namespace['cost_scheduler_environment_configuration']({}, self.root), {})

    def test_another_profile_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exact frozen'):
            self.launch(self.profile.with_name('cost_01_fixed.json'))

    def test_source_profile_sha_mismatch_is_rejected(self):
        self.profile.write_bytes(self.profile.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.launch()

    def test_disabled_or_nonadaptive_profile_is_rejected(self):
        for key in ('enabled', 'macro_enabled', 'allow_probe'):
            original = json.loads((CANDIDATE / wrapper.COST_SCHEDULER_CONFIG).read_text(encoding='utf-8'))
            original[key] = False
            self.profile.write_text(json.dumps(original), encoding='utf-8')
            self.bind_manifest()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.launch()

    def test_formal_environment_restores_verified_config_after_cleaning(self):
        data, configuration = self.frozen()
        state = {'run_id': 'weaverintegrated_unit', 'started_ns': 1,
                 'experiment_root': str(self.root), 'cost_scheduler': configuration}
        pre = {'cards': [{'uuid': 'gpu4'}], 'mps_pipe': '/pipe', 'mps_log': '/log'}
        env = self.namespace['formal_environment'](state, data, pre)
        expected = data / 'cost_scheduler_config.json'
        self.assertEqual(env['ACCERL_COST_SCHEDULER_CONFIG'], str(expected))
        self.assertEqual(expected.read_bytes(), self.profile.read_bytes())
        from cost_scheduler_runtime import load_settings
        loaded = load_settings(env['ACCERL_COST_SCHEDULER_CONFIG'])
        self.assertTrue(loaded['enabled'] and loaded['macro_enabled'] and loaded['allow_probe'])

    def test_frozen_config_path_mismatch_is_rejected(self):
        data, configuration = self.frozen()
        configuration['configuration_path'] = str(self.root / 'other.json')
        with self.assertRaisesRegex(ValueError, 'path/profile'):
            self.namespace['cost_scheduler_environment_configuration']({'cost_scheduler': configuration}, data)

    def test_frozen_config_sha_mismatch_is_rejected(self):
        data, configuration = self.frozen()
        path = data / 'cost_scheduler_config.json'
        path.chmod(0o666)
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.namespace['cost_scheduler_environment_configuration']({'cost_scheduler': configuration}, data)

    def test_wrapper_records_profile_and_checks_activated_state(self):
        data, configuration = self.frozen()
        result = wrapper.validate_cost_scheduler_admission(self.source)
        self.assertEqual(result['source_sha256'], configuration['source_sha256'])
        state = {'cost_scheduler': configuration,
                 'pipeline_config': json.loads((self.source / wrapper.PIPELINE_CONFIG).read_text('utf8')),
                 'pipeline_config_path': str(self.root / 'formal_v3/pipeline_config.json')}
        self.assertTrue(wrapper.check_cost_scheduler_state(state, self.source, self.root))
        bad_execution = copy.deepcopy(state)
        bad_execution['cost_scheduler']['settings']['execution']['parent_policy_budget'] = False
        with self.assertRaisesRegex(ValueError, 'did not activate'):
            wrapper.check_cost_scheduler_state(bad_execution, self.source, self.root)
        bad = copy.deepcopy(state)
        bad['pipeline_config']['fresh_lookahead_updates'] = 0
        with self.assertRaisesRegex(ValueError, 'inventory lookahead'):
            wrapper.check_cost_scheduler_state(bad, self.source, self.root)
        with self.assertRaisesRegex(ValueError, 'did not activate'):
            wrapper.check_cost_scheduler_state({}, self.source, self.root)

    def test_wrapper_runner_command_uses_explicit_path_not_caller_environment(self):
        args = SimpleNamespace(source=self.source, root=self.root)
        command = wrapper.runner_command(args)
        index = command.index('--cost-scheduler-config')
        self.assertEqual(command[index + 1], str(self.profile))
        self.assertIn('--mode', command)
        self.assertIn('manual', command)


if __name__ == '__main__':
    unittest.main()
