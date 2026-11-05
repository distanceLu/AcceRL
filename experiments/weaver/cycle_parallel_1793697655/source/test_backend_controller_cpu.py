"""AST-isolated controller tests: no production runner import, subprocess or GPU."""
from __future__ import annotations

import ast
import copy
import json
import os
import sqlite3
import time
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_manual_controller_cpu import load, result, RUN, SOURCE
from world_backend_audit import validate_world_zero2_receipts
from world_backends import build_service_plan
from world_backends.runtime import context_geometry, load_backend_config, prepare_runtime
from candidate_paths import experiment_root, candidate_run_id, candidate_data, launch_paths, validate_candidate_state


PLAN = build_service_plan({"backend": "weaver", "reward_provider": "world_head"}).to_dict()


def state():
    return {"run_id": RUN, "started_ns": 100, "service_plan": PLAN}


def integrated_result(steps=27):
    value = result(steps)
    value['training'].pop('reward')
    for key in ('fsdp_sharding_verified', 'reward_features_only_enabled', 'reward_checkpointing_verified'):
        value.pop(key)
    value['replay'][1][5] = 0
    value['experiment_completed'][0]['details']['service_plan'] = PLAN
    value['split_wm_cohort_validated'] = ([{'details': {'wm_consumers': ['ctrl'], 'expected_wm_consumers': 4}}]
                                         if steps >= 5 else [])
    return value


class BackendControllerCPU(unittest.TestCase):
    def test_candidate_root_requires_explicit_isolated_absolute_path(self):
        for value in (None, '', 'relative', str(SOURCE), str(SOURCE.parent),
                      str(SOURCE.parent/'analysis/cuda_ipc_weight_sync_gpu47_20261022'),
                      '/mnt/data/lcx1/dong/need_data/cuda_ipc_weight_sync_gpu47_20261022/formal_v3'):
            with self.subTest(root=value), self.assertRaises(ValueError):
                experiment_root(value, environ={})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.assertEqual(experiment_root(environ={'ACCERL_EXPERIMENT_ROOT': str(root)}), root)
            self.assertEqual(candidate_data(root/'formal_v3', root=root), root/'formal_v3')
            with self.assertRaises(ValueError):
                candidate_data(root/'other', root=root)
            with self.assertRaises(ValueError):
                launch_paths({'ACCERL_EXPERIMENT_ROOT': str(root), 'ACCERL_DATA_DIR': str(root/'old_run')})

    def test_old_run_ids_and_cross_source_state_cannot_be_stopped(self):
        for run_id in ('ipcpwmanual_1022', 'mps_split_123', 'w4manual_cpu_unit', '', None):
            with self.assertRaises(ValueError):
                candidate_run_id(run_id)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            current = dict(state(), candidate='weaver_integrated_v1', experiment_root=str(root), source_path=str(SOURCE))
            self.assertIs(validate_candidate_state(root/'formal_v3', current, root=root, run_id=RUN), current)
            for field, value in (('source_path', '/old/source'), ('experiment_root', '/old/root'),
                                 ('candidate', 'cuda_ipc'), ('run_id', 'ipcpwmanual_old')):
                with self.assertRaises(ValueError):
                    validate_candidate_state(root/'formal_v3', dict(current, **{field:value}), root=root)

    def test_redirected_output_paths_and_unvalidated_audit_writes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            data = root/'formal_v3'
            original = Path.resolve
            redirected = data/'training'
            def resolve(path, *args, **kwargs):
                return root/'foreign_run' if path == redirected else original(path, *args, **kwargs)
            with patch.object(Path, 'resolve', autospec=True, side_effect=resolve):
                with self.assertRaises(ValueError):
                    candidate_data(data, root=root)
            verifier = SimpleNamespace(load=Mock(side_effect=AssertionError('No checkpoint access')))
            audit = load('gpu47_safety.py', 'audit_final_weights', {
                'validate_candidate_state': validate_candidate_state, 'MASTER': root})
            with self.assertRaises(ValueError):
                audit(data, state(), torch_module=verifier, persist=True)
            verifier.load.assert_not_called()

    def test_collect_discards_legacy_empty_reward_counter_placeholders(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data/'training').mkdir()
            connection = sqlite3.connect(data/'training/collection.sqlite')
            connection.executescript('''CREATE TABLE events(id INTEGER, run_id TEXT, ts_ns INTEGER,
                actor TEXT, actor_id TEXT, pid INTEGER, workload TEXT, kind TEXT, details_json TEXT);
                CREATE TABLE replay_samples(run_id TEXT, replay TEXT, resident INTEGER,
                policy_uses INTEGER, ctrl_uses INTEGER, reward_uses INTEGER);''')
            connection.close()
            training = integrated_result()['training']
            training['reward'] = {'global_updates': 0, 'ranks': {
                str(rank): {'successful_updates': 0, 'engine_global_steps': 0,
                            'first_success_ns': None, 'last_success_ns': None} for rank in range(4)}}
            namespace = {'time': time, 'json': json, 'sqlite3': sqlite3, 'SOURCE': SOURCE,
                         'stamp': lambda: 'cpu', 'safe': SimpleNamespace(
                             rank_updates=lambda *a: copy.deepcopy(training),
                             placement_evidence=lambda *a, **kw: {'complete_expected_placement': True})}
            namespace['measurement_configuration'] = load('runner.py', 'measurement_configuration', {})
            collect = load('runner.py', 'collect', namespace)
            current = dict(state(), state='running', mode='manual')
            self.assertNotIn('reward', collect(data, current)['training'])
            training['reward']['ranks']['0']['latest_ns'] = 10
            with self.assertRaises(RuntimeError):
                collect(data, current)

    def test_integrated_watchdog_does_not_wait_for_external_reward(self):
        check = load('runner.py', 'check_update_watchdogs', {'time': SimpleNamespace(time_ns=lambda: 0)})
        ready = {'training': {kind: {'ranks': {str(rank): {'successful_updates': 1,
                   'last_success_ns': 9000_000_000_000} for rank in range(4)}} for kind in ('policy', 'ctrl')}}
        check(ready, dict(state(), started_ns=1), now_ns=9001_000_000_000)
        for kind in ('policy', 'ctrl'):
            bad = copy.deepcopy(ready)
            bad['training'][kind]['ranks'] = {}
            with self.assertRaises(RuntimeError):
                check(bad, dict(state(), started_ns=1), now_ns=9001_000_000_000)

    def test_integrated_completion_still_checks_four_ranks_skips_and_consumption(self):
        validate = load('runner.py', 'validate_manual_completion',
                        {'validate_world_zero2_receipts': validate_world_zero2_receipts})
        flag = {'run_id': RUN, 'uid': 1004, 'reason': 'user_requested_stop', 'requested_ns': 1000}
        self.assertEqual(validate(integrated_result(), state(), flag)['steps'], {'policy': 27, 'ctrl': 5})
        self.assertTrue(validate(integrated_result(0), state(), flag)['stopped_before_first_update'])
        mutations = [lambda r: r['training']['policy']['ranks'].pop('3'),
                     lambda r: r['training']['ctrl']['ranks']['2'].update(skipped_steps=1),
                     lambda r: r['world_zero2_verified'].pop(),
                     lambda r: r['policy_zero2_verified'].pop(),
                     lambda r: r['replay'][1].__setitem__(5, 1024),
                     lambda r: r['replay'][1].__setitem__(4, 0),
                     lambda r: r['split_wm_cohort_validated'][0]['details'].update(expected_wm_consumers=8),
                     lambda r: r['experiment_completed'][0]['details'].update(service_plan={})]
        for mutate in mutations:
            bad = copy.deepcopy(integrated_result())
            mutate(bad)
            with self.assertRaises(RuntimeError):
                validate(bad, state(), flag)

    def test_placement_counts_exactly_sixteen_and_rejects_ghost_reward_process(self):
        rows = []
        for rank in range(4):
            for index, role in enumerate(('policy_trainer', 'ctrl_trainer', 'policy_infer', 'ctrl_infer')):
                rows.append({'actor': role, 'actor_id': str(rank), 'pid': 100 + rank * 4 + index,
                    'details': {'role': role, 'physical_gpus': [{'physical_index': rank+4, 'uuid': f'gpu{rank}'}],
                                'physical_gpu_uuid': f'gpu{rank}', 'cpu_threads': 1}})
        safe = SimpleNamespace(query_events=lambda *a, **kw: rows)
        check = load('gpu47_safety.py', 'placement_evidence', {'safe': safe})
        accepted = check(None, RUN, service_plan=PLAN)
        self.assertTrue(accepted['complete_expected_placement'])
        self.assertEqual(accepted['observed_gpu_processes'], 16)
        rows.append({**rows[0], 'actor': 'reward_infer', 'pid': 999,
                     'details': {**rows[0]['details'], 'role': 'reward_infer'}})
        self.assertFalse(check(None, RUN, service_plan=PLAN)['complete_expected_placement'])

    def test_optional_config_is_forwarded_through_clean_environment(self):
        clean = load('gpu47_safety.py', 'cpu_helper_environment', {'os': os})
        formal = load('runner.py', 'formal_environment', {'os': os, 'ROOT': Path('/repo'),
                      'safe': SimpleNamespace(cpu_helper_environment=clean)})
        with patch.dict(os.environ, {}, clear=True):
            env = formal(dict(state(), world_backend_config_path='/candidate/backend.json'), Path('/run'),
                         {'cards': [{'uuid': str(i)} for i in range(4)], 'mps_pipe': '/pipe', 'mps_log': '/log'})
        self.assertEqual(env['ACCERL_WORLD_BACKEND_CONFIG'], '/candidate/backend.json')

    def test_unadmitted_backend_rejected_before_preflight_and_process_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / 'bad.json'
            config.write_text(json.dumps({'backend': 'weaver', 'reward_provider': 'world_head'}))
            safe = SimpleNamespace(MASTER=root, preflight=Mock(side_effect=AssertionError('GPU preflight must not run')))
            run = load('runner.py', 'run', {'safe': safe, 'os': os, 'Path': Path})
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
                run(root/'formal_v3', 'manual', False, config)
            safe.preflight.assert_not_called()
            self.assertFalse((root/'formal_v3').exists())

    def test_final_two_owner_snapshot_requires_integrated_bundle_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            directory = data/'training/checkpoints'
            directory.mkdir(parents=True)
            path = directory/'latest_weights.pt'
            path.write_bytes(b'fixture')
            steps = {'policy': 27, 'ctrl': 5}
            manifest = {'metadata': {'run_id': RUN, 'service_plan': PLAN, 'stop_policy': {
                    'until_manual_stop': True, 'until_deadline': False, 'deadline_unix_seconds': None,
                    'stop_on_target': False}}, 'world_size': 4, 'reason': 'final', 'steps': steps,
                    'weights_only': True, 'tensor_counts': {'policy': 1, 'ctrl': 2}}
            (directory/'manifest.json').write_text(json.dumps(dict(manifest, weights_file=path.name, weights_bytes=7)))
            tensor = SimpleNamespace(device=SimpleNamespace(type='cpu'))
            payload = {'manifest': manifest, 'models': {
                'policy': {'state_dict': {'p': tensor}, 'trainable_names': ['p'], 'full_state': False},
                'ctrl': {'state_dict': {'dynamics': tensor, 'reward_head': tensor},
                         'trainable_names': ['dynamics', 'reward_head'], 'full_state': False,
                         'publication_restore_names': ['dynamics', 'reward_head'], 'publication': {
                             'parameter_names': ['dynamics', 'reward_head'], 'buffer_names': [],
                             'bundle_version': 5, 'backend_identity': {'name': 'weaver'},
                             'bundle_metadata': {'reward_provider': 'world_head'}}}}}
            torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False),
                                    load=Mock(return_value=payload), is_tensor=lambda t: t is tensor)
            audit = load('gpu47_safety.py', 'audit_final_weights', {'os': SimpleNamespace(environ={}),
                          'legacy': SimpleNamespace(stamp=lambda: 'cpu'), 'json': json})
            current = dict(state(), report={'training': {kind: {'global_updates': step} for kind, step in steps.items()}})
            self.assertTrue(audit(data, current, torch_module=torch, persist=False)['valid'])
            payload['models']['ctrl']['state_dict'].pop('reward_head')
            self.assertFalse(audit(data, current, torch_module=torch, persist=False)['valid'])

    def test_launcher_retains_default_argv_and_overrides_backend_geometry_last(self):
        script = (SOURCE/'run.sh').read_text()
        self.assertIn('WORLD_BACKEND_ARGS=()', script)
        self.assertIn('config = load_backend_config()', script)
        self.assertIn('plan = prepare_runtime(config, world_size=4)', script)
        self.assertIn('geometry["required_context_frames"]', script)
        self.assertLess(script.index('plan = prepare_runtime'), script.index('cd "$ACCERL_REPO_ROOT"'))
        self.assertLess(script.rindex('"$@"'), script.rindex('"${WORLD_BACKEND_ARGS[@]}"'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
