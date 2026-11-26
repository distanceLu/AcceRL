"""Exercise opt-in batch tuning through managed commands without GPU or jobs."""
import argparse
import ast
import copy
import datetime
import os
import sys
from pathlib import Path
import tempfile
import time
import types
import unittest
import uuid
from unittest.mock import patch

from test_model_free_launch_cpu import launch, training, site, HERE


class BatchAutotuneLaunchTests(unittest.TestCase):
    def test_real_sampling_reaches_queue_runner_and_driver(self):
        for mode in ('whole_episodes', 'uniform_blocks', 'first_use_blocks'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                stage = Path(temp)
                (stage / 'mps.json').write_bytes(b'{}')
                with patch.object(site, 'PARENT', stage), patch.object(site, 'revision', return_value='a'*40), \
                        patch.object(site, 'PROFILES', {'restore': {'mps_receipt': site.digest(b'{}')}}):
                    record = site.make_spec('restore', 'test-run', stage, stage, checkout=HERE.parent,
                        model_free=True, batch_autotune=True, trajectory_jit=True, real_sampling=mode)
                self.assertEqual(record['real_sampling'], mode)
                queue_args = record['job']['argv']
                self.assertEqual(queue_args[queue_args.index('--real-sampling')+1], mode)
                self.assertEqual(record['fixed_validation_config']['interval_env_steps'], 0)
                args = types.SimpleNamespace(source=HERE/'engine', root=stage, model_free=True,
                    num_eval_workers=0, batch_autotune=True, trajectory_jit=True, real_sampling=mode)
                runner = training.runner_command(args)
                self.assertEqual(runner[runner.index('--real-sampling')+1], mode)
                self.assertEqual(runner[runner.index('--fixed-validation-interval-env-steps')+1], '0')
                self.assertEqual(training.mode_kwargs(args)['real_sampling'], mode)
                captured = {}
                with patch.dict(os.environ, {'ACCERL_PIPELINE_CONFIG':'/pipeline','ACCERL_FINAL_CHECKPOINT':'/out'}), \
                        patch.object(os,'chdir'), patch.object(os,'execve',side_effect=lambda p,a,e:captured.update(e)):
                    launch.exec_managed_driver(['--model-free','--batch-autotune','--trajectory-jit',
                                                '--real-sampling', mode])
                self.assertEqual(captured['ACCERL_REAL_SAMPLING'], mode)
        with self.assertRaisesRegex(ValueError, 'trajectory-jit'):
            launch.real_sampling_options(False, 'whole_episodes')
        with self.assertRaisesRegex(ValueError, 'interval'):
            launch.fixed_validation_options(True, 0, {'interval_env_steps':100000}, real_sampling='whole_episodes')

    def test_first_use_blocks_supports_the_existing_eight_worker_controls(self):
        recipe = dict(model_free=True, real_workers=8, trajectory_jit=True, real_sampling='first_use_blocks')
        self.assertTrue(launch.rollout_options(**recipe, rollout_inventory_gate=True)['rollout_inventory_gate'])
        self.assertEqual(launch.rollout_options(**recipe, real_episodes_per_minute=22.2)['real_episodes_per_minute'], 22.2)
        with self.assertRaises(ValueError):
            launch.rollout_options(**recipe, rollout_inventory_gate=True, real_episodes_per_minute=22.2)

    def test_trajectory_jit_reaches_driver_without_a_benchmark_plan(self):
        for mf, evaluation, tuned in ((False,0,True),(True,2,True),(True,0,False)):
            with self.assertRaises(ValueError):
                launch.trajectory_jit_options(mf,evaluation,tuned,True)
        args = types.SimpleNamespace(source=HERE/'engine', root=Path('/output'), model_free=True,
                                     num_eval_workers=0, batch_autotune=True, trajectory_jit=True)
        self.assertTrue(training.mode_kwargs(args)['trajectory_jit'])
        self.assertIn('--trajectory-jit', training.runner_command(args))
        capture = {}
        with patch.dict(os.environ, {'ACCERL_PIPELINE_CONFIG':'/pipeline', 'ACCERL_FINAL_CHECKPOINT':'/out'}, clear=True), \
                patch.object(os,'chdir'), patch.object(os,'execve',side_effect=lambda p,a,e:capture.update(argv=a,env=e)):
            launch.exec_managed_driver(['--model-free','--num-eval-workers','0','--batch-autotune','--trajectory-jit',
                                        '--fixed-validation-interval-env-steps','0'])
        self.assertEqual(capture['env']['ACCERL_TRAJECTORY_JIT'], '1')
        self.assertNotIn('ACCERL_JIT_BENCH_PLAN',capture['env'])
        self.assertNotIn('--trajectory-jit', capture['argv'])
        self.assertIn('--batch-autotune', capture['argv'])

    def test_reference_policy_rejects_tampering_and_wrong_checkpoint_step(self):
        import hashlib
        import json
        from engine import trajectory_jit_initial as initial
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            policy=initial.reference_policy(root)
            weights=Path(policy['checkpoint'])
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b'original')
            policy.update(bytes=weights.stat().st_size,sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
            manifest=dict(steps={'policy':0},metadata={'run_id':policy['run_id']},tensor_counts={'policy':891})
            (weights.parent/'manifest.json').write_text(json.dumps(manifest))
            (root/'experiment.json').write_text(json.dumps({'initial_policy':policy}))
            with patch.object(initial,'reference_policy',return_value=policy):
                self.assertEqual(initial.verify_reference_policy(root),policy)
                weights.write_bytes(b'modified')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    initial.verify_reference_policy(root)
                manifest['steps']['policy']=189
                (weights.parent/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'at P0'):
                    initial.verify_reference_policy(root)

    def test_training_flag_is_explicit_and_defaults_preserve_argv(self):
        plain = launch.training_arguments({}, model_free=True, template=[])
        tuned = launch.training_arguments({}, model_free=True, template=[], batch_autotune=True)
        self.assertNotIn('--batch-autotune', plain)
        self.assertEqual(tuned, plain + ['--batch-autotune'])
        for kwargs in (dict(model_free=False), dict(model_free=True, num_eval_workers=4)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                launch.training_arguments({}, template=[], batch_autotune=True, **kwargs)

    def test_managed_driver_and_runner_carry_the_flag(self):
        args = types.SimpleNamespace(source=HERE/'engine', root=Path('/output'),
            model_free=True, num_eval_workers=0, batch_autotune=True)
        command = training.runner_command(args)
        self.assertEqual(command.count('--batch-autotune'), 1)
        self.assertTrue(training.mode_kwargs(args)['batch_autotune'])
        captured = {}
        with patch.dict(os.environ, {'ACCERL_PIPELINE_CONFIG':'/frozen/pipeline.json',
                'ACCERL_FINAL_CHECKPOINT':'/output/checkpoints'}, clear=True), \
                patch.object(os, 'chdir'), patch.object(os, 'execve',
                side_effect=lambda p,a,e: captured.update(argv=a,env=e)):
            launch.exec_managed_driver(['--model-free', '--batch-autotune'])
        self.assertEqual(captured['argv'].count('--batch-autotune'), 1)

    def test_managed_spec_freezes_flag_and_queue_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp)
            (stage/'mps.json').write_bytes(b'{}')
            with patch.object(site,'PARENT',stage), \
                    patch.object(site,'PROFILES',{'restore':{'mps_receipt':site.digest(b'{}')}}), \
                    patch.object(site,'revision',return_value='a'*40):
                record=site.make_spec('restore','tuned',stage,stage,checkout=HERE.parent,
                    model_free=True,batch_autotune=True)
        self.assertIs(record['batch_autotune'], True)
        self.assertEqual(record['job']['argv'].count('--batch-autotune'), 1)
        self.assertEqual(record['num_eval_workers'], 0)

    def test_runner_freezes_the_same_flag_before_any_gpu_launch(self):
        from test_manual_controller_cpu import load
        tree=ast.parse((HERE/'engine/runner.py').read_text(encoding='utf8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        boundary=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='processes' for t in n.targets))
        node.body=node.body[:boundary]+[ast.Return(ast.Name('state',ast.Load()))]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            safe=types.SimpleNamespace(MASTER=root,
                preflight=lambda:dict(cards=[dict(uuid=f'GPU-{i}') for i in range(4)],mps_pipe='/mps/p',mps_log='/mps/l'),
                proc_identity=lambda _:dict(pid=123,uid=1004,birth_ticks=1),
                atomic_json=launch.put,cpu_helper_environment=lambda:{})
            ns=dict(safe=safe,Path=Path,os=os,ROOT=HERE.parent,SOURCE=HERE/'engine',dt=datetime,time=time,uuid=uuid,
                TZ=datetime.timezone.utc,stamp=lambda:'now',revision=lambda _:'a'*40)
            for name in ('cost_scheduler_environment_configuration','freeze_cost_scheduler_configuration','formal_environment','measurement_configuration'):
                load('runner.py',name,ns)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), '<runner startup>', 'exec'),ns)
            with patch.object(os,'getuid',return_value=1004,create=True):
                state=ns['run'](root/'formal_v3','manual',False,model_free=True,batch_autotune=True)
        self.assertIs(state['batch_autotune'], True)

    def test_queue_and_supervisor_parsers_accept_the_frozen_command(self):
        for filename in ('queue_entry.py','training_entry.py'):
            path=HERE/'reference_site/queue_wrapper'/filename
            tree=ast.parse(path.read_text(encoding='utf8'))
            main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
            end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets))
            main.body=main.body[:end+1]+[ast.Return(ast.Name('args',ast.Load()))]
            ns=dict(argparse=argparse,Path=Path,sys=sys,__file__=str(path),__doc__='',DEFAULT_EVAL_SECONDS=3600)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(path),'exec'),ns)
            argv=[filename,'--source',str(HERE/'engine'),'--root','/output','--git-commit','a'*40,
                  '--model-free','--batch-autotune']
            with self.subTest(filename=filename),patch.object(sys,'argv',argv):
                args=ns['main']()
                self.assertTrue(training.mode_kwargs(args)['batch_autotune'])
                self.assertIn('--batch-autotune',training.runner_command(args))

    def test_final_receipt_mode_must_match_the_frozen_feature_flag(self):
        from test_model_free_launch_cpu import ModelFreeLaunchTests
        from test_manual_controller_cpu import load
        state,report=ModelFreeLaunchTests().automatic_case()
        state['batch_autotune']=True
        validate=load('runner.py','validate_manual_completion',{})
        with self.assertRaisesRegex(RuntimeError, 'batch autotune'):
            validate(report,state,{},automatic=True)
        report['policy_batch_mode_validation']['autotune']=True
        self.assertEqual(validate(report,state,{},automatic=True)['steps'],{'policy':5})

    def test_state_flag_mismatch_is_rejected_before_other_checks(self):
        with self.assertRaisesRegex(ValueError, 'batch'):
            training.check_state(dict(model_free=True,num_eval_workers=0,batch_autotune=False),
                Path('/source'),Path('/output'),model_free=True,num_eval_workers=0,batch_autotune=True)


if __name__ == '__main__':
    unittest.main()
