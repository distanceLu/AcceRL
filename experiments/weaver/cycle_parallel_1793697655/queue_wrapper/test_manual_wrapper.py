"""CPU contracts for a truly manual train phase and separately bounded evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import training_entry as train
import queue_entry as queue


def mps_authorization():
    return {'protocol': 'gpu47_mps_authorization_v1', 'uid': 1004,
            'boot_id': '31dafb78-e50a-4824-a1ba-12f7ddef8741', 'scope': [4,5,6,7],
            'gpu_uuids': list(train.GPUS), 'pipe': train.PIPE,
            'server': {'pid': 410001, 'uid':1004, 'birth_ticks':123456},
            'control': {'pid': 410000, 'uid':1004, 'birth_ticks':123450}}


def mps_receipt():
    authorization = mps_authorization()
    return {**authorization, 'cards':[{'index':i,'uuid':u} for i,u in zip(range(4,8),train.GPUS)],
            'log':str(Path(train.PIPE).parent/'log')}


def good_state(updates=15):
    counts = {'policy': updates, 'ctrl': updates // 5}
    world = {**train.INITIAL_WORLD, 'protocol': 'weaver_online_joint_weights_v1', 'rm_updates':max(0,updates//5-1),
             'run_id': 'weaverintegrated_test', 'world_size': 4, 'policy_step': updates,
             'successful_updates': updates // 5, 'training_steps': updates // 5,
             'model_version': updates // 5, 'weights_only': True, 'map_location': 'cpu',
             'mmap': True, 'valid': True, 'finite': True, 'tensor_count': 866,
             'weights_sha256': 'a' * 64, 'cuda_initialized': False,
             'python': '/mnt/data/lcx1/dong/envs/weaver-libero-20261022/bin/python'}
    valid = max(0, updates // 5 - 1) * 10240
    world.update(native_reward_head=dict(train.NATIVE_REWARD), reward_target_counts={
        'scope':'global_selected_frames_once_per_successful_world_update',
        'valid_count':valid, 'masked_count':(updates//5)*10240-valid,
        'disabled_updates':min(1,updates//5), 'target_sum':valid*.5})
    state = {'run_id': 'weaverintegrated_test', 'state': 'stopped_by_user',
             'source_path': '/frozen/source', 'training_stopped': True,
             'cleanup': {'survivors': []},
             'manual_completion': {'steps': counts, 'checkpoint_reason': 'final'},
             'checkpoint_cpu_audit': {'valid': True, 'steps': counts, 'manifest_reason': 'final',
                                      'monolith_steps': {'policy': counts['policy']},
                                      'online_world': world}}
    report = {'run_id': state['run_id'], 'state': 'stopped_by_user', 'source': state['source_path'],
              'training': {key: {'global_updates': value} for key, value in counts.items()},
              'experiment_completed': [{'details': {'global_step': updates}}],
              'synchronized_epoch_completed': [{'details': {'global_step': updates}}]}
    flag = {'run_id': state['run_id'], 'uid': 1004, 'reason': 'user_requested_stop', 'requested_ns': 1}
    return state, report, flag


def preflight_report(run_id='weaverintegrated_preflight_test'):
    identities=[{'pid':100+rank+(4 if role=='inference' else 0),'uid':1004,'birth':1000+rank,
        'role':role,'rank':rank,'gpu_uuid':train.GPUS[rank]} for role in ('trainer','inference') for rank in range(4)]
    publications=[]
    for version in range(3):
        inventory={'version':version,'hash':str(version)*64}
        publications.append({'version':version,'inventories':[dict(inventory) for _ in range(4)],
            'acks':[dict(inventory) for _ in range(4)],'released':[dict(inventory) for _ in range(4)],
            'served':[{'served_version':version,'hash':inventory['hash']} for _ in range(4)]})
    mps={'mps_server':{'pid':410001,'uid':1004,'birth':123456},'clients':[],
         'default_cap':100.,'server_cap':100.}
    return {'run_id':run_id,'passed':True,'gpu_started':True,'expected_children':8,
        'child_identities':identities,'successful_updates_per_rank':2,
        'successful_rm_updates_per_rank':1,'missing_class_kept_rm_weights':True,'rm_weights_changed_on_active_update':True,
        'updates':[[{'state':{'global_steps':step,'skipped_steps':0,'distributed_world_size':4,'rm_updates':1,'rm_disabled_updates':step-1}}
                    for _ in range(4)] for step in (1,2)],'publications':publications,
        'cleanup':{'all_owned_exited':True},'preflight':dict(mps),'postflight':dict(mps)}


class ManualWrapperTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(train, 'mps_authorization', return_value=mps_authorization())
        patcher.start(); self.addCleanup(patcher.stop)

    def test_preflight_rejects_missing_child_update_ACK_release_or_postflight_idle(self):
        report=preflight_report();run_id=report['run_id']
        self.assertTrue(train.native_preflight_acceptance(report,report,0,[],run_id=run_id)['passed'])
        for edit in (
            lambda r:r.update(gpu_started=False),
            lambda r:r['child_identities'].pop(),
            lambda r:r['child_identities'][0].update(gpu_uuid='another GPU'),
            lambda r:r['updates'][1][0]['state'].update(global_steps=1),
            lambda r:r['updates'][1][0]['state'].update(skipped_steps=1),
            lambda r:r['publications'][2]['released'].pop(),
            lambda r:r['publications'][2]['served'][0].update(served_version=1),
            lambda r:r['postflight'].update(clients=['999']),
            lambda r:r['postflight']['mps_server'].update(birth=1),
        ):
            changed=copy.deepcopy(report);edit(changed)
            self.assertFalse(train.native_preflight_acceptance(changed,changed,0,[],run_id=run_id)['passed'])

    def test_formal_training_requires_explicit_native_gpu_preflight_flag(self):
        args=SimpleNamespace(post_stop_grace_seconds=1800,native_gpu_preflight=False)
        with self.assertRaisesRegex(ValueError,'native-gpu-preflight'):train.run(args)
    def test_source_manifest_requires_native_online_files_and_detects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root).resolve()
            names=('runner.py','request_stop.py','candidate_paths.py','pipeline_default.json',
                train.BACKEND_CONFIG,'frozen_policy_eval.py','inspect_world_backend.py',
                'world_backend_audit.py','world_backend_runtime.py','weaver_online_bridge.py',
                'weaver_online_engine.py','weaver_online_gpu_worker.py','weaver_online_smoke.py','world_backends/weaver.py',
                'run_fresh.sh','fresh_config.py','fresh_reward_targets.py','fresh_policy_replay.py',
                'fresh_seed_scheduler.py','fresh_context.py','fresh_signal.py',
                'cost_scheduler.py','cost_scheduler_runtime.py','collective_phase_gate.py',
                'cycle_service_controller.py','policy_service_dispatch.py',train.COST_SCHEDULER_CONFIG)
            files={}
            for name in names:
                member=source/name;member.parent.mkdir(exist_ok=True,parents=True);member.write_text(name)
                files[name]=hashlib.sha256(member.read_bytes()).hexdigest()
            manifest=source/'source_manifest.json'
            manifest.write_text(json.dumps({'candidate':'weaver_integrated_v1','files':files}))
            digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(train.verify_manifest(source,digest)['verified_files'],len(names))
            (source/'weaver_online_engine.py').write_text('changed source')
            with self.assertRaises(ValueError):train.verify_manifest(source,digest)

    def test_admission_requires_two_compatible_interpreters_and_one_million_real_steps(self):
        backend={'backend':'weaver','reward_provider':'world_head','reward_semantics':'delta_native_score',
            'terminal_rule':'horizon_only','critic_mode':'disabled','world_trainable':True,
            'backend_options':{'training_python':'/weaver/bin/python','optimizations':dict(train.OPTIMIZATIONS),
                'native_reward_head':dict(train.NATIVE_REWARD),'fresh':dict(train.FRESH_OPTIONS)},
            'checkpoint_metadata':{'load_report':{'checkpoint_sha256':train.INITIAL_WORLD['source_bundle_sha256']}}}
        admitted={'configuration_admitted':True,'assets_checked':True,'errors':[],
            'environment':{'compatible':True},'online_training_environment':{
                'python':'/weaver/bin/python','compatible':True,'distributed_available':True,
                'nccl_available':True,'cuda_initialized':False,'gpu_access_performed':False}}
        self.assertTrue(train.validate_backend_admission(backend,train.PIPELINE_RECIPE,admitted))
        for edit in (
            lambda b,p,a:p.update(real_env_step_budget=11000),
            lambda b,p,a:p.update(real_env_step_budget=True),
            lambda b,p,a:p.update(real_workers=4),
            lambda b,p,a:b.update(world_trainable=False),
            lambda b,p,a:a['environment'].update(compatible=False),
            lambda b,p,a:a['online_training_environment'].update(compatible=False),
            lambda b,p,a:a['online_training_environment'].update(distributed_available=False),
            lambda b,p,a:a['online_training_environment'].update(cuda_initialized=True),
            lambda b,p,a:a['online_training_environment'].update(python='/merged/bin/python'),
        ):
            b,p,a=copy.deepcopy(backend),dict(train.PIPELINE_RECIPE),copy.deepcopy(admitted);edit(b,p,a)
            with self.assertRaises(ValueError):train.validate_backend_admission(b,p,a)

    def test_manual_contract_has_no_queue_or_training_deadline(self):
        value = train.validate_manual_contract()
        self.assertTrue(value['run_until_manual_stop'])
        self.assertEqual(value['expected_queue_timeout_seconds'], 0)
        self.assertIsNone(value['deadline_ns'])
        self.assertIsNone(value['soft_stop_seconds'])
        self.assertIsNone(value['target_policy_epochs'])

    def test_neither_morning_nor_epochs_nor_many_days_stop(self):
        for epochs in (0, 15, 25, 500, 30000):
            for elapsed in (3600, 24 * 3600, 366 * 86400):
                self.assertIsNone(train.stop_cause(epochs=epochs, elapsed=elapsed))
        self.assertEqual(train.stop_cause(signal_number=15), 'external_signal_15')

    def test_post_stop_grace_does_not_become_training_budget(self):
        for invalid in (0, 899, 3601, True, 900.0):
            with self.assertRaises(ValueError):
                train.validate_manual_contract(invalid)
        self.assertEqual(train.validate_manual_contract(900)['post_stop_grace_seconds'], 900)

    def test_external_stop_is_accepted_without_wrapper_stop_ack(self):
        state, report, flag = good_state()
        result = train.acceptance(state, report, 0, [], flag)
        self.assertTrue(result['passed'], result)

    def test_stop_before_first_update_is_truthfully_accepted(self):
        for reason in ('final', 'final_before_first_update'):
            state, report, flag = good_state(0)
            state['checkpoint_cpu_audit']['manifest_reason'] = reason
            state['manual_completion']['checkpoint_reason'] = reason
            result = train.acceptance(state, report, 0, [], flag)
            self.assertTrue(result['passed'], result)
            self.assertTrue(result['stopped_before_first_update'])

    def test_world_must_record_its_own_synchronized_online_updates(self):
        state, report, flag = good_state()
        report['training']['ctrl'] = {'global_updates': 0}
        self.assertFalse(train.acceptance(state, report, 0, [], flag)['passed'])

    def test_aggregate_has_only_policy_world_while_monolith_has_only_policy(self):
        state, report, flag = good_state()
        self.assertEqual(set(state['checkpoint_cpu_audit']['steps']), {'policy', 'ctrl'})
        self.assertEqual(set(state['checkpoint_cpu_audit']['monolith_steps']), {'policy'})
        self.assertTrue(train.acceptance(state, report, 0, [], flag)['passed'])
        state['checkpoint_cpu_audit']['monolith_steps']['ctrl'] = 3
        self.assertFalse(train.acceptance(state, report, 0, [], flag)['passed'])

    def test_initial_world_bundle_cannot_silently_change(self):
        state, report, flag = good_state()
        state['checkpoint_cpu_audit']['online_world']['source_bundle_sha256'] = 'b' * 64
        self.assertIn('online_world_cpu_audit_identity_or_steps_mismatch',
                      train.acceptance(state, report, 0, [], flag)['reasons'])

    def test_frozen_world_audit_cannot_masquerade_as_online_training(self):
        state, report, flag = good_state()
        state['checkpoint_cpu_audit']['frozen_world'] = {'fixed_step': 3200}
        self.assertIn('online_world_cannot_use_frozen_world_audit',
                      train.acceptance(state, report, 0, [], flag)['reasons'])

    def test_separate_world_cpu_audit_requires_exact_counter_run_hash_and_no_cuda(self):
        for field, value in (('run_id', 'another'), ('policy_step', 10), ('model_version', 0),
                             ('successful_updates', 0), ('training_steps', 3203), ('finite', False),
                             ('mmap', False), ('cuda_initialized', True), ('tensor_count', 383),
                             ('python', ''), ('weights_sha256', 'unknown')):
            state, report, flag = good_state()
            state['checkpoint_cpu_audit']['online_world'][field] = value
            with self.subTest(field=field):
                self.assertFalse(train.acceptance(state, report, 0, [], flag)['passed'])

    def test_invalid_stop_or_missing_audit_or_survivor_fails(self):
        for change in ('foreign', 'no_flag', 'no_audit', 'survivor', 'failure'):
            state, report, flag = good_state()
            remaining = []
            if change == 'foreign': flag['run_id'] = 'another'
            if change == 'no_flag': flag = None
            if change == 'no_audit': state['checkpoint_cpu_audit'] = {}
            if change == 'survivor': remaining = [{'pid': 10}]
            if change == 'failure': state['state'] = 'failed'
            self.assertFalse(train.acceptance(state, report, 0, remaining, flag)['passed'])

    def test_zero_exit_without_stop_is_not_user_completion(self):
        state, report, _ = good_state()
        self.assertIn('exact_manual_stop_evidence_missing',
                      train.acceptance(state, report, 0, [], None)['reasons'])

    def test_stop_file_requires_exact_identity_and_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / 'training').mkdir()
            self.assertIsNone(train.read_manual_stop(data, 'weaverintegrated_test'))
            _, _, flag = good_state()
            path = data / 'training/stop.flag'
            path.write_text(json.dumps(flag))
            self.assertEqual(train.read_manual_stop(data, flag['run_id']), flag)
            for key, value in (('uid', 1005), ('reason', 'stall_guard'), ('run_id', 'foreign'), ('requested_ns', 0)):
                bad = dict(flag, **{key: value})
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    train.read_manual_stop(data, flag['run_id'])

    def test_two_hour_eval_is_independent_and_bounded(self):
        args = SimpleNamespace(post_stop_grace_seconds=1800, evaluation_seconds=7200)
        result = queue.validate_budgets(args)
        self.assertIsNone(result['deadline_ns'])
        self.assertEqual(result['evaluation_seconds'], 7200)
        for invalid in (0, 59, 7201, True):
            args.evaluation_seconds = invalid
            with self.assertRaises(ValueError): queue.validate_budgets(args)

    def test_evaluation_is_exactly_initial_final_by_50_by_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'policy_eval'
            path.mkdir()
            receipt = {'complete': True, 'initial_state_count': 50,
                       'modes': ['greedy', 'stochastic'], 'expected_episode_records': 200,
                       'completed_episode_records': 200,
                       'checkpoint_sha256': {'baseline': 'a' * 64, 'final': 'b' * 64}}
            (path / 'completed.json').write_text(json.dumps(receipt))
            for name in ('summary.json', 'episodes.jsonl', 'plan.json'): (path / name).write_text('{}')
            self.assertTrue(queue.evaluation_acceptance(root, 0, [])['passed'])
            receipt['completed_episode_records'] = 199
            (path / 'completed.json').write_text(json.dumps(receipt))
            self.assertFalse(queue.evaluation_acceptance(root, 0, [])['passed'])

    def test_launch_environment_keeps_explicit_manual_queue_identity(self):
        env = train.launch_environment(Path('/out'), Path('/source'), {
            'EXPERIMENT_QUEUE_JOB_ID': 'own', 'EXPERIMENT_QUEUE_MANUAL': '1',
            'EXPERIMENT_QUEUE_TIMEOUT_SECONDS': '0', 'ACCERL_RUN_ID': 'old',
            'SOFT_STOP_AT': 'old', 'MEASURE_STOP_AFTER_EPOCHS': '15', 'LD_PRELOAD': 'profiler'})
        self.assertEqual(env['EXPERIMENT_QUEUE_MANUAL'], '1')
        self.assertEqual(env['EXPERIMENT_QUEUE_TIMEOUT_SECONDS'], '0')
        self.assertEqual(env['ACCERL_WORLD_BACKEND_CONFIG'], str(Path('/source') / train.BACKEND_CONFIG))
        for key in ('ACCERL_RUN_ID', 'SOFT_STOP_AT', 'MEASURE_STOP_AFTER_EPOCHS', 'LD_PRELOAD'):
            self.assertNotIn(key, env)

    def test_cli_refuses_hidden_bounded_training_flags(self):
        for flag in ('--hard-seconds', '--soft-seconds', '--target-policy-epochs', '--stop-on-target'):
            result = subprocess.run([sys.executable, str(Path(queue.__file__)), '--source', '/source',
                                     '--root', '/out', '--manifest-sha256', 'a'*64, flag, '15'],
                                     capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('unrecognized arguments', result.stderr)


if __name__ == '__main__':
    unittest.main()
