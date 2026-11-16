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
