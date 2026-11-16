"""CPU launch contracts: no model construction, queue submission or GPU work."""
import importlib.util
import ast
import copy
import datetime
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

    def test_managed_driver_executes_model_free_recipe_in_tensorboard_watch_directory(self):
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
        self.assertEqual(chdir.call_args.args[0],launch.SITE_ENTRY.parent.parent/'AcceRL-scheduler')

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
