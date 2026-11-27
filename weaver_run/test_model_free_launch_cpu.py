"""CPU launch contracts: no model construction, queue submission or GPU work."""
import importlib.util
import ast
import copy
import datetime
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import time
import uuid
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'engine'))
if os.name != 'posix':
    sys.modules.setdefault('fcntl', types.SimpleNamespace())
import launch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


training = module('model_free_training_entry', HERE / 'reference_site/queue_wrapper/training_entry.py')
site = module('model_free_site_cli', HERE / 'reference_site/cli.py')


class ModelFreeLaunchTests(unittest.TestCase):
    def test_trainer_mps_reaches_only_policy_trainers(self):
        from training_topology import actor_options, SPLIT_LAYOUT, validate_trainer_mps_bindings
        flag, key = '--trainer-mps-percent', 'ACCERL_TRAINER_MPS_PERCENT'
        options = dict(model_free=True, trajectory_jit=True, batch_autotune=True, num_eval_workers=0,
                       real_sampling='uniform_blocks', real_workers=8, trainer_mps_percent=80)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'mps.json').write_bytes(b'{}')
            with patch.object(site, 'PARENT', root), patch.object(site, 'revision', return_value='a'*40), \
                 patch.object(site, 'PROFILES', {'restore': {'mps_receipt': site.digest(b'{}')}}):
                record = site.make_spec('restore', 'mps80', root, root, checkout=site.CHECKOUT, **options)
            self.assertEqual(record['trainer_mps_percent'], 80)
            argv = record['job']['argv']
            self.assertEqual(argv[argv.index(flag)+1], '80')
            args = types.SimpleNamespace(source=HERE/'engine', root=root, **options)
            mode = training.mode_kwargs(args)
            self.assertEqual(mode['trainer_mps_percent'], 80)
            self.assertEqual(training.launch_environment(root, args.source, {}, **mode)[key], '80')
            command = training.runner_command(args)
            with patch.dict(os.environ, dict(ACCERL_PIPELINE_CONFIG='p', ACCERL_FINAL_CHECKPOINT='w'), clear=True), \
                 patch.object(os, 'chdir'), patch.object(os, 'execve') as execute:
                launch.exec_managed_driver(command[command.index('--model-free'):])
                self.assertEqual(execute.call_args.args[2][key], '80')
        cards = [dict(index=str(i), uuid=str(i), resource=str(i)) for i in range(4)]
        cap_key = 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'
        with patch.dict(os.environ, {key:'80', 'ACCERL_USE_MPS':'1', 'CUDA_MPS_PIPE_DIRECTORY':str(HERE)}, clear=True):
            for i in range(4):
                trainer = actor_options(cards, 'policy_trainer', i, SPLIT_LAYOUT, ('policy',))
                infer = actor_options(cards, 'policy', i, SPLIT_LAYOUT, ('policy',))
                self.assertEqual(trainer['runtime_env']['env_vars'][cap_key], '80')
                self.assertNotIn(cap_key, infer['runtime_env']['env_vars'])
        trainers = [dict(uuid=str(i), physical_gpu=str(i), mps_client_percent=80, visible_sm_count=60) for i in range(4)]
        inference = [dict(uuid=str(i), physical_gpu=str(i), mps_client_percent=100, visible_sm_count=78) for i in range(4)]
        validate_trainer_mps_bindings(trainers, inference, 80)
        with self.assertRaises(ValueError):
            validate_trainer_mps_bindings(trainers, inference[::-1], 80)
        trainers[0]['visible_sm_count'] = 78
        with self.assertRaises(ValueError):
            validate_trainer_mps_bindings(trainers, inference, 80)
        self.assertNotIn(flag, launch.rollout_arguments())
        for value in (False, 0, 101):
            with self.assertRaises(ValueError):
                launch.rollout_options(True, trainer_mps_percent=value)

    def test_fixed_episode_rate_reaches_driver_and_frozen_experiment(self):
        options = dict(model_free=True, trajectory_jit=True, batch_autotune=True, num_eval_workers=0,
                       real_sampling='uniform_blocks', real_workers=8, real_episodes_per_minute=22.2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'mps.json').write_bytes(b'{}')
            with patch.object(site, 'PARENT', root), patch.object(site, 'revision', return_value='a'*40), \
                 patch.object(site, 'PROFILES', {'restore': {'mps_receipt': site.digest(b'{}')}}):
                record = site.make_spec('restore', 'fixedrate', root, root, checkout=site.CHECKOUT, **options)
            self.assertEqual(record['real_episodes_per_minute'], 22.2)
            argv = record['job']['argv']
            self.assertEqual(argv[argv.index('--real-episodes-per-minute')+1], '22.2')
            args = types.SimpleNamespace(source=HERE/'engine', root=root, **options)
            mode = training.mode_kwargs(args)
            self.assertEqual(mode['real_episodes_per_minute'], 22.2)
            self.assertEqual(training.launch_environment(root, args.source, {}, **mode)
                             ['ACCERL_REAL_EPISODES_PER_MINUTE'], '22.2')
            command = training.runner_command(args)
            with patch.dict(os.environ, dict(ACCERL_PIPELINE_CONFIG='p', ACCERL_FINAL_CHECKPOINT='w'), clear=True), \
                 patch.object(os, 'chdir'), patch.object(os, 'execve') as execute:
                launch.exec_managed_driver(command[command.index('--model-free'):])
                self.assertEqual(execute.call_args.args[2]['ACCERL_REAL_EPISODES_PER_MINUTE'], '22.2')
        for rate, gate in ((-1, False), (float('nan'), False), (float('inf'), False), (22.2, True)):
            with self.subTest(rate=rate, gate=gate), self.assertRaises(ValueError):
                launch.rollout_options(True, 8, rollout_inventory_gate=gate, trajectory_jit=True,
                                      real_sampling='uniform_blocks', real_episodes_per_minute=rate)

    def test_compiled_action_request_is_frozen_and_forwarded_to_training_workers(self):
        from test_manual_controller_cpu import env as formal_env
        flag = '--compile-action-inference'
        key = 'ACCERL_COMPILE_ACTION_INFERENCE'
        options = dict(model_free=True, trajectory_jit=True, batch_autotune=True,
                       real_sampling='uniform_blocks', real_workers=8, no_micro_cache_trim=True,
                       num_eval_workers=0, compile_action_inference=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'mps.json').write_bytes(b'{}')
            with patch.object(site, 'PARENT', root), patch.object(site, 'revision', return_value='a'*40), \
                 patch.object(site, 'PROFILES', {'restore': {'mps_receipt': site.digest(b'{}')}}):
                record = site.make_spec('restore', 'compiled', root, root, checkout=site.CHECKOUT, **options)
            self.assertIs(json.loads((root/'experiment.json').read_bytes())['compile_action_inference'], True)
            self.assertIn(flag, record['job']['argv'])
            self.assertFalse(record['rollout_inventory_gate'])
            args = types.SimpleNamespace(source=HERE/'engine', root=root, **options)
            selected = training.mode_kwargs(args)
            self.assertIs(selected['compile_action_inference'], True)
            self.assertEqual(training.launch_environment(root, args.source, {}, **selected)[key], '1')
            state = dict(options, run_id='weaverintegrated_test', started_ns=1,
                         initial_policy={'checkpoint': 'p0'}, experiment_root=str(root))
            self.assertEqual(formal_env(state=state)[key], '1')
            del state['compile_action_inference']
            self.assertEqual(formal_env({key: '1'}, state=state)[key], '0')
            command = training.runner_command(args)
            overrides = command[command.index('--model-free'):]
            self.assertIn(flag, overrides)
            env = dict(ACCERL_PIPELINE_CONFIG='pipeline.json', ACCERL_FINAL_CHECKPOINT='weights',
                       ACCERL_EXPERIMENT_ROOT=str(root), ACCERL_COMPILE_ACTION_INFERENCE='1')
            for enabled in (True, False):
                with self.subTest(enabled=enabled), patch.dict(os.environ, env, clear=True), \
                     patch.object(os, 'chdir'), patch.object(os, 'execve') as execute:
                    launch.exec_managed_driver([arg for arg in overrides if enabled or arg != flag])
                    _, argv, child_env = execute.call_args.args
                    self.assertEqual(child_env[key], str(int(enabled)))
                    self.assertNotIn(flag, argv)

    def test_compiled_action_requires_model_free_trajectory_jit(self):
        for model_free, trajectory_jit in [(False, False), (False, True), (True, False)]:
            with self.subTest(model_free=model_free, trajectory_jit=trajectory_jit), \
                 self.assertRaisesRegex(ValueError, 'compile-action-inference'):
                launch.rollout_options(model_free, trajectory_jit=trajectory_jit, compile_action_inference=True)

    def automatic_case(self, reason='real_budget_exhausted', steps=5):
        from test_manual_controller_cpu import result, RUN
        from world_backends import build_service_plan
        config={'backend':'model_free','world_trainable':False,'reward_provider':'environment'}
        plan=build_service_plan(config).to_dict()
        report=result(steps)
        report['training']={'policy':report['training']['policy']}
        report['replay']=report['replay'][:1]
        for kind in ('world_zero2_verified','fsdp_sharding_verified','reward_features_only_enabled',
                     'reward_checkpointing_verified','world_memory_policy_verified','split_wm_cohort_validated'):
            report[kind]=[]
        validation=dict(interval_env_steps=100000,patience=3,min_env_steps=300000,min_delta_successes=1)
        budget=dict(initialized=True,max_env_steps=1000000,consumed_env_steps=999995,
            available_env_steps=5,reset_steps=10,reserved_env_steps=0,active_real=0,episodes_inflight=0,
            conservative_unknown_env_steps=0,episodes_cancelled=0,budget_exhausted=True,failure=None)
        from model_free_control import ModelFreeControl, ModelFreeControlConfig
        controller=ModelFreeControl(ModelFreeControlConfig(**validation))
        for step,consumed in ((0,1000),(1,101000),(3,201000),(5,301000)):
            controller.observe(38,policy_step=step,consumed_env_steps=consumed)
        summary=controller.summary()
        report['experiment_completed'][0]['details'].update(model_free=True,service_plan=plan,
            stop_reason=reason,budget_status=budget,fixed_validation=summary)
        state=dict(run_id=RUN,started_ns=100,model_free=True,world_backend_config=config,service_plan=plan,
            pipeline_config={'real_env_step_budget':1000000},fixed_validation_config=validation)
        return state,report

    def test_automatic_completion_accepts_exhausted_budget_without_manual_flag(self):
        from test_manual_controller_cpu import load
        state,report=self.automatic_case()
        validate=load('runner.py','validate_manual_completion',{})
        with self.assertRaises(RuntimeError):
            validate(report,state,{})
        try:
            receipt=validate(report,state,{},automatic=True)
        except TypeError as exc:
            self.fail('Runner does not support audited automatic completion: '+str(exc))
        self.assertEqual(receipt['stop_reason'],'real_budget_exhausted')
        zero_state,zero_report=self.automatic_case(steps=0)
        zero=validate(zero_report,zero_state,{},automatic=True)
        self.assertTrue(zero['stopped_before_first_update'])
        self.assertEqual(zero['checkpoint_reason'],'final_before_first_update')
        for mutation in ('legacy','inflight','cancelled','unknown','failure','capacity','flag'):
            modified=copy.deepcopy(report)
            changed=copy.deepcopy(state)
            flag={}
            budget=modified['experiment_completed'][0]['details']['budget_status']
            if mutation=='legacy': changed['model_free']=False
            if mutation=='inflight': budget['active_real']=1
            if mutation=='cancelled': budget['episodes_cancelled']=1
            if mutation=='unknown': budget['conservative_unknown_env_steps']=1
            if mutation=='failure': budget['failure']='worker crashed'
            if mutation=='capacity': budget['max_env_steps']=2
            if mutation=='flag': flag={'reason':'worker_failed'}
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):
                validate(modified,changed,flag,automatic=True)

    def test_plateau_completion_requires_matching_validation_and_final_policy(self):
        from test_manual_controller_cpu import load
        state,report=self.automatic_case('fixed_validation_plateau')
        validate=load('runner.py','validate_manual_completion',{})
        try:
            receipt=validate(report,state,{},automatic=True)
        except TypeError as exc:
            self.fail('Runner does not support audited plateau completion: '+str(exc))
        self.assertEqual(receipt['stop_reason'],'fixed_validation_plateau')
        for key,value in [('stale_count',2),('plateau_reached',False),('enabled',False),('case_count',49)]:
            modified=copy.deepcopy(report)
            modified['experiment_completed'][0]['details']['fixed_validation'][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                validate(modified,state,{},automatic=True)
        modified=copy.deepcopy(report)
        modified['experiment_completed'][0]['details']['fixed_validation']['last']['policy_step']=4
        with self.assertRaises(RuntimeError):
            validate(modified,state,{},automatic=True)
        modified=copy.deepcopy(report)
        modified['experiment_completed'][0]['details']['fixed_validation']['validations'][-1]['successes']=50
        with self.assertRaises(RuntimeError):
            validate(modified,state,{},automatic=True)

    def target_case(self, steps=5):
        from model_free_control import ModelFreeControl, ModelFreeControlConfig
        state,report=self.automatic_case('fixed_validation_target_reached',steps=steps)
        controller=ModelFreeControl(ModelFreeControlConfig(**state['fixed_validation_config']))
        if steps:
            controller.observe(38,policy_step=0,consumed_env_steps=0)
        summary=controller.observe(50,policy_step=steps,consumed_env_steps=steps*1000)
        details=report['experiment_completed'][0]['details']
        details['fixed_validation']=summary
        details['budget_status'].update(consumed_env_steps=steps*1000,
            available_env_steps=1000000-steps*1000,budget_exhausted=False)
        report['checkpoint']['steps']={'policy':steps}
        return state,report

    def test_target_completion_accepts_exact_saved_policy_before_plateau_minimum_including_p0(self):
        from test_manual_controller_cpu import load
        from policy_batch_receipts import validate_policy_batch_mode_receipts
        validate=load('runner.py','validate_manual_completion',{})
        for steps in (0,5):
            with self.subTest(steps=steps):
                state,report=self.target_case(steps)
                if steps == 0:
                    state['batch_autotune']=True
                    report['policy_batch_mode_validation']=validate_policy_batch_mode_receipts(
                        [],run_id=state['run_id'],steps=0,model_free=True,autotune=True)
                receipt=validate(report,state,{},automatic=True)
                self.assertEqual(receipt['stop_reason'],'fixed_validation_target_reached')
                self.assertEqual(receipt['steps'],{'policy':steps})
                self.assertEqual(receipt['stopped_before_first_update'],steps == 0)

    def rollout_case(self, steps=5):
        state, report = self.automatic_case('rollout_success_target_reached', steps=steps)
        state['model_free_success_target'] = {'window_size': 1000, 'window_successes': 990}
        start = state['started_ns']
        def observation(episodes, successes):
            return dict(window_size=1000, window_episodes=1000, window_successes=successes,
                        success_rate=successes / 1000, total_episodes_processed=episodes,
                        total_env_steps=episodes * 20, observed_ns=start+episodes,
                        elapsed_seconds=episodes / 1e9, run_started_ns=start)
        snapshot = dict(observation(1002, 989), crossings={
            '98': observation(1000, 989), '99': observation(1001, 990)})
        details = report['experiment_completed'][0]['details']
        details['model_free_success'] = snapshot
        details['budget_status'].update(consumed_env_steps=30060, available_env_steps=969940,
                                       budget_exhausted=False)
        report['checkpoint'].update(steps={'policy': steps}, model_free_success=copy.deepcopy(snapshot))
        return state, report

    def test_rollout_target_requires_a_latched_full_window_and_its_exact_final_checkpoint(self):
        from test_manual_controller_cpu import load
        validate = load('runner.py', 'validate_manual_completion', {})
        for steps in (0, 5):
            with self.subTest(steps=steps):
                state, report = self.rollout_case(steps)
                receipt = validate(report, state, {}, automatic=True)
                snapshot = report['experiment_completed'][0]['details']['model_free_success']
                self.assertEqual(receipt['model_free_success'], snapshot)
                state.update(state='completed_automatically', training_stopped=True,
                             cleanup={'survivors': []}, source_path='/source', automatic_completion=receipt,
                             checkpoint_cpu_audit=dict(valid=True, steps={'policy': steps},
                                monolith_steps={'policy': steps}, manifest_reason=receipt['checkpoint_reason'],
                                model_free_success=copy.deepcopy(snapshot)))
                report.update(run_id=state['run_id'], state=state['state'], source='/source',
                              synchronized_epoch_completed=[{}] if steps else [])
                self.assertTrue(training.acceptance(state, report, 0, [], None)['passed'])
                del state['checkpoint_cpu_audit']['model_free_success']['crossings']['99']
                self.assertFalse(training.acceptance(state, report, 0, [], None)['passed'])
                report['checkpoint']['steps'] = {'policy': steps+1}
                with self.assertRaises(RuntimeError):
                    validate(report, state, {}, automatic=True)

    def test_new_stop_contract_rejects_fixed_scores_and_invalid_rollout_evidence(self):
        from test_manual_controller_cpu import load
        validate = load('runner.py', 'validate_manual_completion', {})
        for key in ('fixed_target', 'plateau', 'partial', '98_only', 'wrong_start',
                    'inflight', 'charged_actions', 'wrong_snapshot'):
            with self.subTest(key=key):
                state, report = self.rollout_case()
                details = report['experiment_completed'][0]['details']
                if key == 'fixed_target':
                    state, report = self.target_case()
                    state['model_free_success_target'] = {'window_size': 1000, 'window_successes': 990}
                elif key == 'plateau': details['stop_reason'] = 'fixed_validation_plateau'
                elif key == 'partial': details['model_free_success']['crossings']['99']['window_episodes'] = 999
                elif key == '98_only': del details['model_free_success']['crossings']['99']
                elif key == 'wrong_start': details['model_free_success']['run_started_ns'] += 1
                elif key == 'inflight': details['budget_status']['episodes_inflight'] = 1
                elif key == 'charged_actions':
                    details['budget_status'].update(consumed_env_steps=1, available_env_steps=999999)
                elif key == 'wrong_snapshot': del report['checkpoint']['model_free_success']['crossings']['99']
                with self.assertRaises(RuntimeError):
                    validate(report, state, {}, automatic=True)

    def test_new_budget_stop_records_the_98_crossing_without_claiming_99(self):
        from test_manual_controller_cpu import load
        validate = load('runner.py', 'validate_manual_completion', {})
        state, report = self.rollout_case()
        details = report['experiment_completed'][0]['details']
        details['stop_reason'] = 'real_budget_exhausted'
        details['budget_status'].update(consumed_env_steps=999999, available_env_steps=1, budget_exhausted=True)
        with self.assertRaises(RuntimeError):
            validate(report, state, {}, automatic=True)
        del details['model_free_success']['crossings']['99']
        report['checkpoint']['model_free_success'] = copy.deepcopy(details['model_free_success'])
        receipt = validate(report, state, {}, automatic=True)
        self.assertEqual(receipt['stop_reason'], 'real_budget_exhausted')
        self.assertEqual(set(receipt['model_free_success']['crossings']), {'98'})

    def test_runner_child_exit_routes_target_to_automatic_completion(self):
        from test_manual_controller_cpu import load
        from world_backends import build_service_plan
        tree=ast.parse((HERE/'engine/runner.py').read_text(encoding='utf8'))
        run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == 'run')
        exited=next(node for node in ast.walk(run) if isinstance(node,ast.If)
            and isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name)
            and node.test.left.id == 'rc')
        # Execute the actual child-exit branch with CPU reports in place of
        # subprocess/SQLite transports; preserve its automatic routing and audit.
        entry=ast.parse('for unused in (0,):\n    pass').body[0]
        entry.body=[exited]
        for make_case, steps in ((self.target_case, 0), (self.target_case, 5),
                                 (self.rollout_case, 0), (self.rollout_case, 5)):
            with self.subTest(case=make_case.__name__, steps=steps):
                state,result=make_case(steps)
                ns=dict(rc=0,model_free=True,batch_autotune=False,state=state,data=Path('/cpu'),
                    plan=build_service_plan(state['world_backend_config']),
                    report=lambda *_:result,observe_stop_request=lambda *_:None,
                    validate_policy_batch_modes_from_db=lambda *a,**kw:result['policy_batch_mode_validation'],
                    validate_manual_completion=load('runner.py','validate_manual_completion',{}))
                try:
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[entry],type_ignores=[])),
                        '<actual runner child-exit branch>','exec'),ns)
                except RuntimeError as exc:
                    self.fail('Complete target should route to automatic completion: '+str(exc))
                self.assertEqual(state['state'],'completed_automatically')
                self.assertEqual(state['automatic_completion']['stop_reason'],
                                 result['experiment_completed'][0]['details']['stop_reason'])

    def test_target_completion_rejects_partial_forged_or_wrong_saved_version(self):
        from test_manual_controller_cpu import load
        validate=load('runner.py','validate_manual_completion',{})
        for mutation in ('partial','not_full','no_target','history_missing','history_forged',
                         'continued_after_target','wrong_checkpoint','missing_checkpoint',
                         'wrong_event_step','fault','stop','undrained','disabled'):
            with self.subTest(mutation=mutation):
                state,report=self.target_case()
                details=report['experiment_completed'][0]['details']
                summary=details['fixed_validation']
                flag={}
                if mutation == 'partial': summary['last']['cases']=49
                if mutation == 'not_full': summary['last']['successes']=49
                if mutation == 'no_target': summary['target_reached']=False
                if mutation == 'history_missing': summary.pop('validations')
                if mutation == 'history_forged': summary['validations'][-1]['successes']=49
                if mutation == 'continued_after_target':
                    summary['validations'][0]['successes']=50
                    summary['baseline']['successes']=50
                if mutation == 'wrong_checkpoint': report['checkpoint']['steps']={'policy':6}
                if mutation == 'missing_checkpoint': report['checkpoint'].pop('steps')
                if mutation == 'wrong_event_step': details['global_step']=6
                if mutation == 'fault': state['error']='evaluation failed'
                if mutation == 'stop': flag={'reason':'worker_failed'}
                if mutation == 'undrained': details['budget_status']['active_real']=1
                if mutation == 'disabled': summary['enabled']=False
                with self.assertRaises(RuntimeError):
                    validate(report,state,flag,automatic=True)

    def test_queue_target_acceptance_binds_saved_and_reported_policy_to_complete_validation(self):
        for steps in (0,5):
            with self.subTest(steps=steps):
                state,report=self.target_case(steps)
                reason='final' if steps else 'final_before_first_update'
                state.update(state='completed_automatically',training_stopped=True,cleanup={'survivors':[]},
                    source_path='/source',automatic_completion=dict(steps={'policy':steps},
                        checkpoint_reason=reason,stop_reason='fixed_validation_target_reached'),
                    checkpoint_cpu_audit=dict(valid=True,steps={'policy':steps},
                        monolith_steps={'policy':steps},manifest_reason=reason))
                report.update(run_id=state['run_id'],state=state['state'],source='/source',
                    synchronized_epoch_completed=[{}] if steps else [])
                self.assertTrue(training.acceptance(state,report,0,[],None)['passed'])
                wrong=copy.deepcopy(report)
                wrong['experiment_completed'][0]['details']['fixed_validation']['last']['cases']=49
                self.assertFalse(training.acceptance(state,wrong,0,[],None)['passed'])
                # Even internally matching runner/report/audit counters cannot
                # substitute a later policy for the complete successful one.
                wrong_state,wrong=copy.deepcopy((state,report))
                wrong_state['automatic_completion']['steps']={'policy':steps+1}
                wrong_state['checkpoint_cpu_audit'].update(steps={'policy':steps+1},
                    monolith_steps={'policy':steps+1},manifest_reason='final')
                wrong_state['automatic_completion']['checkpoint_reason']='final'
                wrong['training']['policy']['global_updates']=steps+1
                wrong['checkpoint'].update(reason='final',steps={'policy':steps+1})
                wrong['synchronized_epoch_completed']=[{}]
                self.assertFalse(training.acceptance(wrong_state,wrong,0,[],None)['passed'])

    def test_watchdog_validation_pause_is_bounded_and_resumes_after_completion(self):
        from test_manual_controller_cpu import load
        state,report=self.automatic_case()
        state['started_ns']=1000_000_000_000
        for row in report['training']['policy']['ranks'].values():
            row['last_success_ns']=1000_000_000_000
        watchdog=load('runner.py','check_update_watchdogs',{})
        report['fixed_validation_activity']=dict(run_id=state['run_id'],active=True,
            started_unix_seconds=4000,deadline_unix_seconds=9400,policy_step=5,phase='evaluate')
        try:
            watchdog(report,state,now_ns=5000_000_000_000)
        except RuntimeError as exc:
            self.fail('Finite validation should suspend the optimizer watchdog: '+str(exc))
        for field,value in [('deadline_unix_seconds',999999),('run_id','another-run')]:
            changed=copy.deepcopy(report)
            changed['fixed_validation_activity'][field]=value
            with self.assertRaises(RuntimeError):
                watchdog(changed,state,now_ns=5000_000_000_000)
        with self.assertRaises(RuntimeError):
            watchdog(report,state,now_ns=9401_000_000_000)
        report['fixed_validation_activity']['active']=False
        watchdog(report,state,now_ns=5001_000_000_000)
        with self.assertRaises(RuntimeError):
            watchdog(report,state,now_ns=8601_000_000_000)

    def test_queue_automatic_acceptance_requires_independent_stop_evidence(self):
        state,report=self.automatic_case()
        completion=dict(steps={'policy':5},checkpoint_reason='final',stop_reason='real_budget_exhausted')
        state.update(state='completed_automatically',training_stopped=True,cleanup={'survivors':[]},
            source_path='/source',automatic_completion=completion,
            checkpoint_cpu_audit=dict(valid=True,steps={'policy':5},monolith_steps={'policy':5},manifest_reason='final'))
        report.update(run_id=state['run_id'],state=state['state'],source='/source',synchronized_epoch_completed=[{}])
        self.assertTrue(training.acceptance(state,report,0,[],None)['passed'])
        report['experiment_completed'][0]['details']['budget_status']['reserved_env_steps']=1
        self.assertFalse(training.acceptance(state,report,0,[],None)['passed'])

    def test_all_spatial_tasks_freeze_scope_in_queue_and_driver(self):
        parser = argparse.ArgumentParser()
        launch.add_rollout_arguments(parser)
        self.assertTrue(parser.parse_args(['--all-spatial-tasks']).all_spatial_tasks)
        for percent, sampler in ((100,'uniform_blocks'), (90,'whole_episodes')):
            with self.subTest(percent=percent, sampler=sampler):
                rollout = launch.rollout_options(True, 8, True, trajectory_jit=True,
                    real_sampling=sampler, trainer_compile='on', trainer_mps_percent=percent,
                    compile_action_inference=True, all_spatial_tasks=True)
                self.assertTrue(rollout['all_spatial_tasks'])
                self.assertIn('--all-spatial-tasks',launch.rollout_arguments(**rollout))
                options = dict(model_free=True, num_eval_workers=0, batch_autotune=True,
                               trajectory_jit=True, real_sampling=sampler, **rollout)
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    (root/'mps.json').write_bytes(b'{}')
                    with patch.object(site,'PARENT',root), patch.object(site,'revision',return_value='a'*40), \
                         patch.object(site,'PROFILES',{'restore':{'mps_receipt':site.digest(b'{}')}}):
                        record = site.make_spec('restore','multi',root,root,checkout=HERE.parent,**options)
                    self.assertTrue(record['all_spatial_tasks'])
                    self.assertEqual(record['pipeline']['initial_state_count'],500)
                    self.assertEqual(record['real_training_action_horizons_by_task'], [150] + [220] * 9)
                    self.assertEqual(record['pipeline']['real_training_action_horizons_by_task'],
                                     record['real_training_action_horizons_by_task'])
                    self.assertIsNone(record['task_action_horizon'])
                    self.assertIn('--all-spatial-tasks',record['job']['argv'])
                    args = types.SimpleNamespace(source=HERE/'engine',root=root,**options)
                    kwargs = training.mode_kwargs(args)
                    self.assertTrue(kwargs['all_spatial_tasks'])
                    self.assertEqual(training.launch_environment(root,args.source,{},**kwargs)['ACCERL_ALL_SPATIAL_TASKS'],'1')
                    command = training.runner_command(args)
                    with patch.dict(os.environ,{'ACCERL_PIPELINE_CONFIG':'p','ACCERL_FINAL_CHECKPOINT':'w'},clear=True), \
                         patch.object(os,'chdir'), patch.object(os,'execve') as execute:
                        launch.exec_managed_driver(command[command.index('--model-free'):])
                    environment = execute.call_args.args[2]
                    self.assertEqual(environment['ACCERL_ALL_SPATIAL_TASKS'],'1')
                    self.assertEqual(environment['ACCERL_TRAINER_MPS_PERCENT'],str(percent))
                    self.assertEqual(environment['ACCERL_REAL_SAMPLING'],sampler)
        self.assertNotIn('all_spatial_tasks',launch.rollout_options(True))
        self.assertNotIn('--all-spatial-tasks',launch.rollout_arguments())
        self.assertEqual(500, launch.effective_pipeline(True,8,all_spatial_tasks=True)['initial_state_count'])
        self.assertEqual(50, launch.effective_pipeline(True,8)['initial_state_count'])
        with self.assertRaises(ValueError):
            launch.rollout_options(False,all_spatial_tasks=True)
        with self.assertRaises(ValueError):
            launch.fixed_validation_options(True,4,all_spatial_tasks=True)
        with self.assertRaises(ValueError):
            launch.fixed_validation_options(True,0,{'interval_env_steps':100},all_spatial_tasks=True)

    def test_fixed_validation_defaults_and_overrides_are_carried_by_training_arguments(self):
        defaults = launch.training_arguments({}, model_free=True, template=[])
        for option, value in [('interval-env-steps', '100000'), ('patience', '3'),
                              ('min-env-steps', '300000'), ('min-delta-successes', '1')]:
            self.assertIn('--fixed-validation-' + option, defaults)
            self.assertEqual(defaults[defaults.index('--fixed-validation-' + option) + 1], value)
        changed = launch.training_arguments({}, model_free=True, template=[],
            fixed_validation={'interval_env_steps':200000,'patience':4})
        self.assertEqual(changed[changed.index('--fixed-validation-interval-env-steps')+1], '200000')
        self.assertEqual(changed[changed.index('--fixed-validation-patience')+1], '4')
        online = launch.training_arguments({}, model_free=True, num_eval_workers=4, template=[])
        self.assertEqual(online[online.index('--fixed-validation-interval-env-steps')+1], '0')
        with self.assertRaises(ValueError):
            launch.training_arguments({}, model_free=False, template=[],fixed_validation={'patience':3})
        with self.assertRaises(ValueError):
            launch.training_arguments({}, model_free=True,num_eval_workers=4,template=[],
                fixed_validation={'interval_env_steps':100000})

    def test_fixed_validation_configuration_reaches_managed_runner(self):
        args=types.SimpleNamespace(source=HERE/'engine',root=Path('/output'),model_free=True,
            num_eval_workers=0,fixed_validation_interval_env_steps=200000,
            fixed_validation_patience=5,fixed_validation_min_env_steps=400000,
            fixed_validation_min_delta_successes=2)
        command=training.runner_command(args)
        self.assertIn('--fixed-validation-patience',command)
        self.assertEqual(command[command.index('--fixed-validation-patience')+1],'5')
        selected=training.mode_kwargs(args)
        self.assertEqual(selected['fixed_validation']['interval_env_steps'],200000)

    def test_actual_runner_startup_freezes_policy_only_state_without_world_assets(self):
        from test_manual_controller_cpu import load
        tree=ast.parse((HERE/'engine/runner.py').read_text(encoding='utf8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        boundary=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='processes' for t in n.targets))
        node.body=node.body[:boundary]+[ast.Return(ast.Tuple([ast.Name('state',ast.Load()),ast.Name('env',ast.Load())],ast.Load()))]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            pre=dict(cards=[dict(uuid=f'GPU-{i}') for i in range(4)],mps_pipe='/existing/mps/pipe',mps_log='/existing/mps/log')
            safe=types.SimpleNamespace(MASTER=root,preflight=lambda:pre,proc_identity=lambda _:dict(pid=123,uid=1004,birth_ticks=1),
                atomic_json=launch.put,cpu_helper_environment=lambda:{})
            ns=dict(safe=safe,Path=Path,os=os,ROOT=HERE.parent,SOURCE=HERE/'engine',dt=datetime,time=time,uuid=uuid,
                TZ=datetime.timezone.utc,stamp=lambda:'now',revision=lambda _:'a'*40)
            for name in ('cost_scheduler_environment_configuration','freeze_cost_scheduler_configuration','formal_environment','measurement_configuration'):
                load('runner.py',name,ns)
            ns['cost_scheduler_launch_configuration']=lambda *_:self.fail('model-free must not inspect scheduler config')
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<actual runner startup>','exec'),ns)
            with patch.object(os,'getuid',return_value=1004,create=True):
                state,env=ns['run'](root/'formal_v3','manual',False,
                    world_backend_config=root/'missing-world.json',pipeline_config=HERE/'recipe/pipeline_config.json',model_free=True,num_eval_workers=4)
            self.assertTrue(state['model_free'])
            self.assertEqual(state['num_eval_workers'],4)
            self.assertEqual(state['service_plan']['trainer_kinds'],['policy'])
            self.assertEqual(state['pipeline_config']['imagination_workers'],0)
            self.assertIsNone(state['cost_scheduler'])
            self.assertNotIn('world_backend_config_path',state)
            self.assertNotIn('ACCERL_WORLD_BACKEND_CONFIG',env)
            self.assertEqual(env['ACCERL_NUM_EVAL_WORKERS'],'4')
            self.assertEqual(state['fixed_validation_config'],dict(interval_env_steps=0,patience=3,
                min_env_steps=300000,min_delta_successes=1))
            self.assertFalse((root/'formal_v3/world_backend_config.json').exists())

    def test_model_free_cpu_admission_only_checks_policy_and_control_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'policy-checksums.json').write_text('{}')
            calls=[]
            def run(argv,**kwargs):
                calls.append((argv,kwargs))
                return types.SimpleNamespace(returncode=0)
            with patch.object(launch,'RECIPE',root),patch.object(launch.subprocess,'run',side_effect=run):
                launch.check(root,['python','train.py','--model-free'],{'CUDA_VISIBLE_DEVICES':'4,5,6,7'}, {})
            self.assertEqual(len(calls),1)
            self.assertEqual(Path(calls[0][0][1]).name,'check_runtime.py')
            self.assertIn('--model-free',calls[0][0])
            self.assertEqual(calls[0][1]['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(calls[0][1]['env']['MUJOCO_GL'],'osmesa')

    def test_model_based_arguments_are_unchanged_by_default(self):
        template=['--num-eval-workers','3','--eval-inference-replicas','1,2,3','--exp-name','old']
        self.assertEqual(launch.training_arguments({},template=template),template)

    def test_runner_final_policy_receipts_do_not_require_frozen_world(self):
        from test_manual_controller_cpu import load, result
        from world_backends import build_service_plan
        config={'backend':'model_free','world_trainable':False,'reward_provider':'environment'}
        plan=build_service_plan(config).to_dict()
        value=result(5)
        value['training']={'policy':value['training']['policy']}
        value['replay']=[value['replay'][0]]
        for key in ('fsdp_sharding_verified','reward_features_only_enabled','reward_checkpointing_verified',
                    'world_zero2_verified','world_memory_policy_verified','split_wm_cohort_validated'):
            value[key]=[]
        value['experiment_completed'][0]['details']['service_plan']=plan
        state=dict(run_id='weaverintegrated_test',started_ns=100,model_free=True,
                   world_backend_config=config,service_plan=plan)
        value['policy_batch_mode_validation']['run_id']=state['run_id']
        validate=load('runner.py','validate_manual_completion',{})
        receipt=validate(value,state,dict(run_id=state['run_id'],uid=1004,reason='user_requested_stop',requested_ns=1000))
        self.assertEqual(receipt['steps'],{'policy':5})

    def test_portable_preparation_needs_only_policy_assets_and_preserves_recipe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = root / 'policy'
            policy.mkdir()
            distilled = root / 'distilled.pt'
            distilled.write_bytes(b'policy')
            paths = dict(policy=str(policy), distilled=str(distilled),
                         **{key: str(root / ('absent_' + key)) for key in ('bundle', 'image', 'task', 'native')})
            with patch.object(launch, 'settings', return_value=paths), patch.dict(os.environ, LIBERO_CONFIG_PATH=str(root)), patch.object(os, 'getuid', return_value=1004, create=True):
                cmd, env, _ = launch.prepare(root, ['4','5','6','7'], 'weaverintegrated_test', model_free=True)
            self.assertIn('--model-free', cmd)
            for option, expected in [('--num-eval-workers','0'), ('--eval-inference-replicas',''),
                                     ('--num-rollout-workers','4'), ('--num-ctrl-inference-actors','0'),
                                     ('--num-images-in-input','2'), ('--clip-mode','gipo')]:
                self.assertEqual(cmd[cmd.index(option)+1], expected)
            self.assertIn(str(distilled), cmd)
            self.assertIn(str(policy), cmd)
            self.assertNotIn('--world-backend-config', cmd)
            self.assertFalse((root / 'world.json').exists())
            self.assertNotIn('ACCERL_COST_SCHEDULER_CONFIG', env)
            self.assertEqual(env['ACCERL_TENSORBOARD'], '1')
            self.assertEqual(env['ACCERL_ENV_RENDER_BACKEND'], 'egl')
            effective = json.loads(Path(env['ACCERL_PIPELINE_CONFIG']).read_text())
            self.assertEqual([effective[k] for k in ('real_workers','imagination_workers','max_imagine_in_flight','inventory_refill_enabled')], [4,0,0,False])
            self.assertEqual(effective['real_env_step_budget'], 1000000)
            self.assertEqual(effective['real_training_action_horizon'], 150)

    def test_explicit_eval_four_maps_every_policy_replica(self):
        argv = launch.training_arguments({}, model_free=True, num_eval_workers=4, template=['--num-eval-workers','3','--eval-inference-replicas','1,2,3','--exp-name','old'])
        self.assertEqual(argv[argv.index('--eval-inference-replicas')+1], '0,1,2,3')
        self.assertIn('ModelFree', argv[argv.index('--exp-name')+1])

    def test_inventory_gate_is_bound_to_the_formal_job_and_managed_driver(self):
        options = dict(model_free=True, num_eval_workers=0, batch_autotune=True,
                       trajectory_jit=True, real_sampling='uniform_blocks',
                       real_workers=8, no_micro_cache_trim=True, rollout_inventory_gate=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'mps.json').write_bytes(b'{}')
            profiles = {'restore': {'mps_receipt': site.digest(b'{}')}}
            with patch.object(site, 'PARENT', root), patch.object(site, 'PROFILES', profiles), \
                 patch.object(site, 'revision', return_value='a'*40):
                record = site.make_spec('restore', 'inventory', root, root, checkout=HERE.parent, **options)
            self.assertTrue(record['rollout_inventory_gate'])
            self.assertIn('--rollout-inventory-gate', record['job']['argv'])
            args = types.SimpleNamespace(source=HERE/'engine', root=root, **options)
            mode = training.mode_kwargs(args)
            self.assertTrue(mode['rollout_inventory_gate'])
            self.assertEqual(training.launch_environment(root, args.source, {}, **mode)
                             ['ACCERL_ROLLOUT_INVENTORY_GATE'], '1')
            command = training.runner_command(args)
            self.assertIn('--rollout-inventory-gate', command)
            overrides = command[command.index('--model-free'):]
            env = dict(ACCERL_PIPELINE_CONFIG='pipeline.json', ACCERL_FINAL_CHECKPOINT='weights',
                       ACCERL_ROLLOUT_INVENTORY_GATE='1')
            for enabled in (True, False):
                with self.subTest(enabled=enabled), patch.dict(os.environ, env, clear=True), \
                     patch.object(os, 'chdir'), patch.object(os, 'execve') as execute:
                    selected = [arg for arg in overrides if enabled or arg != '--rollout-inventory-gate']
                    launch.exec_managed_driver(selected)
                    _, argv, child_env = execute.call_args.args
                    self.assertEqual(child_env['ACCERL_ROLLOUT_INVENTORY_GATE'], str(int(enabled)))
                    self.assertNotIn('--rollout-inventory-gate', argv)

    def test_reference_start_forwards_inventory_gate_and_runner_freezes_it(self):
        from test_manual_controller_cpu import env as formal_env
        argv = ['run.sh', 'original', '--id', 'gate-test', '--model-free', '--batch-autotune',
                '--trajectory-jit', '--real-sampling', 'uniform_blocks', '--real-workers', '8',
                '--rollout-inventory-gate']
        with patch.object(sys, 'argv', argv), patch.object(launch.socket, 'gethostname',
                return_value=launch.REFERENCE_SERVER), patch.object(os, 'execv', side_effect=InterruptedError) as start:
            with self.assertRaises(InterruptedError):
                launch.main()
        self.assertIn('--rollout-inventory-gate', start.call_args.args[1])
        state = dict(run_id='weaverintegrated_test', started_ns=1, model_free=True, real_workers=8,
                     trajectory_jit=True, real_sampling='uniform_blocks', initial_policy={'checkpoint': 'p0'},
                     rollout_inventory_gate=True)
        self.assertEqual(formal_env(state=state)['ACCERL_ROLLOUT_INVENTORY_GATE'], '1')
        del state['rollout_inventory_gate']
        self.assertEqual(formal_env({'ACCERL_ROLLOUT_INVENTORY_GATE':'1'}, state=state)
                         ['ACCERL_ROLLOUT_INVENTORY_GATE'], '0')

    def test_inventory_gate_rejects_incompatible_formal_configuration(self):
        options = dict(model_free=True, real_workers=8, trajectory_jit=True,
                       real_sampling='uniform_blocks', rollout_inventory_gate=True)
        for change in (dict(model_free=False), dict(real_workers=4), dict(trajectory_jit=False),
                       dict(real_sampling='whole_episodes'), dict(real_sampling='chunks')):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'inventory gate'):
                launch.rollout_options(**dict(options, **change))

    def test_formal_eight_workers_propagate_to_driver_and_replay_leases(self):
        selected = launch.rollout_options(True, real_workers=8, no_micro_cache_trim=True)
        effective = launch.effective_pipeline(True, real_workers=8)
        self.assertEqual((effective['real_workers'], effective['max_real_in_flight']), (8, 8))
        variables = {'ACCERL_PIPELINE_CONFIG':'/pipeline.json', 'ACCERL_FINAL_CHECKPOINT':'/weights'}
        with patch.dict(os.environ, variables, clear=True), patch.object(os, 'chdir'), patch.object(os, 'execve') as execute:
            launch.exec_managed_driver(['--model-free', *launch.rollout_arguments(**selected)])
        _, argv, env = execute.call_args.args
        self.assertEqual(argv[argv.index('--num-rollout-workers') + 1], '8')
        self.assertNotIn('--no-micro-cache-trim', argv)
        self.assertEqual((env['ACCERL_MODEL_FREE_REAL_WORKERS'], env['ACCERL_TRIM_MICRO_CACHE'],
                          env['ACCERL_TRIM_IDLE_CACHE']), ('8','0','1'))
        self.assertNotIn('ACCERL_JIT_BENCH_PLAN', env)
        self.assertEqual(env['ACCERL_FAST_JIT_BOUNDARIES'], '0')
        with patch.dict(os.environ, variables, clear=True), patch.object(os, 'chdir'), patch.object(os, 'execve') as execute:
            launch.exec_managed_driver(['--model-free', '--batch-autotune', '--trajectory-jit',
                                        '--num-eval-workers', '0', *launch.rollout_arguments(**selected)])
        fast_env = execute.call_args.args[2]
        self.assertEqual((fast_env['ACCERL_TRIM_MICRO_CACHE'], fast_env['ACCERL_TRIM_IDLE_CACHE'],
                          fast_env['ACCERL_FAST_JIT_BOUNDARIES']), ('0', '0', '1'))
        with self.assertRaises(ValueError):
            launch.rollout_options(False, real_workers=8)

    def test_managed_spec_persists_effective_mode_and_never_reads_world_template(self):
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp)
            (stage/'mps.json').write_bytes(b'{}')
            profiles={'restore':{'mps_receipt':site.digest(b'{}')}}
            original_read=Path.read_bytes
            def guarded_read(path):
                if path.name=='world_backend_config.json':
                    raise AssertionError('model-free spec must not read World recipe')
                return original_read(path)
            with patch.object(site,'PARENT',stage),patch.object(site,'PROFILES',profiles),patch.object(site,'revision',return_value='a'*40),patch.object(Path,'read_bytes',guarded_read):
                record=site.make_spec('restore','model-free',stage,stage,checkout=HERE.parent,model_free=True,num_eval_workers=4,
                    fixed_validation={'patience':5})
            self.assertEqual(record['git_commit'],'a'*40)
            self.assertTrue(record['model_free'])
            self.assertEqual(record['stop_policy'], 'real_rollout_success_or_budget_or_explicit_stop_or_fault')
            self.assertEqual(record['model_free_success_target'], dict(window_size=1000, window_successes=990))
            self.assertEqual(record['num_eval_workers'],4)
            self.assertEqual(record['fixed_validation_config'],dict(interval_env_steps=0,patience=5,
                min_env_steps=300000,min_delta_successes=1))
            self.assertEqual(record['job']['argv'][record['job']['argv'].index('--fixed-validation-patience')+1],'5')
            self.assertEqual(record['pipeline']['real_workers'],4)
            self.assertEqual(record['pipeline']['imagination_workers'],0)
            self.assertFalse(record['cost_scheduler_enabled'])
            self.assertNotIn('--native-gpu-preflight',record['job']['argv'])
            self.assertIn('--model-free',record['job']['argv'])
            self.assertFalse((stage/'git-world.json').exists())

    def test_managed_driver_uses_frozen_checkout_and_preserves_tensorboard_watch_directory(self):
        captured={}
        def execute(python,argv,env):
            captured.update(argv=argv,env=env)
        with patch.dict(os.environ,{'ACCERL_PIPELINE_CONFIG':'/frozen/pipeline.json','ACCERL_FINAL_CHECKPOINT':'/output/checkpoints',
                                    'ACCERL_COST_SCHEDULER_CONFIG':'/stale/scheduler.json','ACCERL_WORLD_BACKEND_CONFIG':'/stale/world.json'},clear=True),patch.object(os,'chdir') as chdir,patch.object(os,'execve',side_effect=execute):
            launch.exec_managed_driver(['--model-free','--num-eval-workers','4'])
        self.assertIn('--model-free',captured['argv'])
        self.assertNotIn('--world-backend-config',captured['argv'])
        self.assertEqual(captured['env']['ACCERL_TENSORBOARD'],'1')
        self.assertNotIn('ACCERL_COST_SCHEDULER_CONFIG',captured['env'])
        self.assertNotIn('ACCERL_WORLD_BACKEND_CONFIG',captured['env'])
        self.assertEqual(chdir.call_args.args[0],launch.ROOT)
        self.assertEqual(Path(captured['env']['ACCERL_TENSORBOARD_RUNS']),launch.SITE_ENTRY.parent.parent/'AcceRL-scheduler/runs')
        self.assertEqual(captured['env']['PYTHONPATH'].split(os.pathsep)[:2],
                         [str(launch.ENGINE),str(launch.ROOT)])

    def test_mode_mismatch_cannot_pass_managed_state_checks(self):
        state=dict(run_id='weaverintegrated_test',candidate='weaver_integrated_v1',source_path='/source',
            experiment_root='/output',uid=1004,mode='manual',until_manual_stop=True,stop_on_target=False,
            deadline_unix=None,world_backend_config={'backend_options':{'fresh':training.FRESH_OPTIONS}})
        with self.assertRaises(ValueError):
            training.check_state(state,Path('/source'),Path('/output'),model_free=True,num_eval_workers=0)

    def test_policy_only_final_snapshot_is_accepted_and_wrong_mode_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'latest_weights.pt').write_bytes(b'weights')
            launch.put(directory / 'manifest.json', dict(reason='final', steps={'policy':5}, weights_bytes=7,
                metadata={'run_id':'weaverintegrated_test','model_free':True}))
            result = launch.final_snapshot(directory, 'weaverintegrated_test', model_free=True)
            self.assertEqual(result['policy_step'],5)
            self.assertNotIn('world_updates',result)
            with self.assertRaises((ValueError,KeyError)):
                launch.final_snapshot(directory,'weaverintegrated_test',model_free=False)

    def test_portable_status_reports_rollout_stop_only_with_matching_final_weights(self):
        from model_free_success import ModelFreeSuccess
        stats = ModelFreeSuccess(1)
        for i in range(1000):
            stats.observe(i < 990, 10, i + 2)
        metadata = dict(run_id='portable', model_free=True, global_step=73,
                        stop_reason='rollout_success_target_reached', model_free_success=stats.snapshot())
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'latest_weights.pt').write_bytes(b'weights')
            manifest = dict(reason='final', steps={'policy':73}, weights_bytes=7, metadata=metadata)
            launch.put(directory / 'manifest.json', manifest)
            receipt = launch.final_snapshot(directory, 'portable', model_free=True)
            self.assertEqual(receipt['stop_reason'], metadata['stop_reason'])
            self.assertEqual(receipt['model_free_success']['crossings']['99']['window_successes'], 990)
            manifest['steps']['policy'] += 1
            launch.put(directory / 'manifest.json', manifest)
            with self.assertRaises(ValueError):
                launch.final_snapshot(directory, 'portable', model_free=True)

    def test_managed_environment_and_runner_command_carry_mode_without_world(self):
        root, source = Path('/output'), HERE / 'engine'
        env = training.launch_environment(root, source, {'ACCERL_COST_SCHEDULER_CONFIG':'stale'}, model_free=True, num_eval_workers=4)
        self.assertEqual(env['ACCERL_MODEL_FREE'],'1')
        self.assertEqual(env['ACCERL_NUM_EVAL_WORKERS'],'4')
        self.assertNotIn('ACCERL_WORLD_BACKEND_CONFIG',env)
        self.assertNotIn('ACCERL_COST_SCHEDULER_CONFIG',env)
        args = types.SimpleNamespace(source=source,root=root,model_free=True,num_eval_workers=4)
        command = training.runner_command(args)
        self.assertIn('--model-free',command)
        self.assertNotIn('--world-backend-config',command)
        self.assertNotIn('--cost-scheduler-config',command)

    def test_managed_policy_only_completion_does_not_demand_world_audit(self):
        state = dict(model_free=True, state='stopped_by_user', training_stopped=True, cleanup={'survivors':[]},
            run_id='weaverintegrated_test', source_path='/source', manual_completion={'steps':{'policy':5},'checkpoint_reason':'final'},
            checkpoint_cpu_audit={'valid':True,'steps':{'policy':5},'monolith_steps':{'policy':5},'manifest_reason':'final'})
        report = dict(run_id=state['run_id'],state=state['state'],source='/source',training={'policy':{'global_updates':5}},
            experiment_completed=[{}], synchronized_epoch_completed=[{}])
        stop = dict(run_id=state['run_id'],uid=1004,reason='user_requested_stop')
        self.assertTrue(training.acceptance(state,report,0,[],stop)['passed'])
        report['training']['ctrl']={'global_updates':1}
        self.assertFalse(training.acceptance(state,report,0,[],stop)['passed'])


if __name__ == '__main__':
    unittest.main()
