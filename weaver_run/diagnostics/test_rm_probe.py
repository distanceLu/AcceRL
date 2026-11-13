import unittest
import tempfile
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import json
import ast
import argparse
import sys
from . import rm_probe


class RMProbeTests(unittest.TestCase):
    def test_outer_launcher_dispatches_rm_commands_to_reference_cli(self):
        root=Path(__file__).resolve().parents[2];path=root/'weaver_run/launch.py'
        tree=ast.parse(path.read_text('utf8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        dispatched=Mock(side_effect=RuntimeError('dispatched'))
        scope=dict(argparse=argparse,Path=Path,sys=sys,REFERENCE_SERVER='reference',HERE=path.parent,
            socket=SimpleNamespace(gethostname=lambda:'reference'),os=SimpleNamespace(execv=dispatched),
            require=lambda ok,why: None if ok else self.fail(why))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
        for command in ('diagnose-rm','diagnose-rm-status'):
            with patch.object(sys,'argv',['run.sh',command,'--id','rm-state-20261113-a']):
                with self.assertRaisesRegex(RuntimeError,'dispatched'):scope['main']()
            self.assertEqual(dispatched.call_args.args[1][-3:],[command,'--id','rm-state-20261113-a'])

    def test_new_submission_queues_behind_active_training_without_repin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkout=root/'code';checkout.mkdir();experiments=root/'experiments';experiments.mkdir()
            cli=SimpleNamespace(EXPERIMENTS=experiments,CHECKOUT=checkout,QUEUE=root/'queue',PYTHON='python',
                LABEL=re.compile(r'[a-z0-9-]+'),lookup=Mock(return_value=None),
                require=lambda ok,why: None if ok else self.fail(why),
                published=Mock(return_value='a'*40),cpu=Mock(return_value=json.dumps({'cuda_initialized':False})),
                active_jobs=Mock(return_value=[dict(id='live',state='running')]),
                create=Mock(return_value=dict(id='pending',state='queued')))
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=30*1024**3)):
                reply=rm_probe.submit(cli,'rm-state-20261113-a')
            self.assertEqual(reply['state'],'queued')
            job=cli.create.call_args.args[0]
            self.assertEqual(job['cwd'],str(checkout))
            self.assertFalse(job['manual']);self.assertLessEqual(job['timeout_seconds'],14520)
            recorded=json.loads((experiments/'rm-state-20261113-a/experiment.json').read_bytes())
            self.assertEqual(recorded['waiting_behind'],[dict(id='live',state='running')])
            self.assertTrue(recorded['no_training_stop_requested'])
            self.assertIn('--validate-only',cli.cpu.call_args.args[0])

    def test_existing_submission_is_idempotent_and_does_not_touch_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli=SimpleNamespace(EXPERIMENTS=Path(tmp),LABEL=re.compile(r'[a-z0-9-]+'),
                lookup=Mock(return_value=dict(id='existing',state='queued')),
                require=lambda ok,why: None if ok else self.fail(why),
                published=Mock(side_effect=AssertionError('No resubmission')),
                create=Mock(side_effect=AssertionError('No duplicate job')),
                active_jobs=Mock(side_effect=AssertionError('No training operation')))
            result=rm_probe.submit(cli,'rm-state-20261113-a')
            self.assertEqual(result['job_id'],'existing')
            cli.create.assert_not_called();cli.published.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_status_before_submission_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli=SimpleNamespace(EXPERIMENTS=Path(tmp),LABEL=re.compile(r'[a-z0-9-]+'),
                lookup=Mock(return_value=None),require=lambda ok,why: None if ok else self.fail(why))
            self.assertEqual(rm_probe.submit(cli,'rm-state-20261113-a',status=True)['state'],'not_submitted')
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_real_only_native_adapter_never_generates_imagined_frames(self):
        import os,sys
        from pathlib import Path
        import numpy as np
        root=Path(__file__).resolve().parents[2]
        os.environ['WEAVER_TEST_UPSTREAM']=str(root/'weaver_run/vendor/weaver')
        sys.path.insert(0,str(root/'weaver_run/engine'))
        from test_fresh_padding_backend_cpu import configuration
        from test_weaver_optimized_backend_cpu import make_backend
        from .native_panel_worker import score_real_episode
        cfg=configuration(cache=False);upstream,b=make_backend(cfg)
        def forbidden(*a,**k):raise AssertionError('No World rollout in real RM diagnosis')
        b.predict_chunk=forbidden
        ep=dict(episode_id='probe',initial_state_id=rm_probe.STATES[0],success=True,instruction='task',
                frames=np.full((20,2,3,224,224),128,np.uint8),states=np.zeros((20,8),np.float32),
                actions=np.zeros((19,7),np.float32))
        rows=score_real_episode(b,ep)
        self.assertEqual([r['frame_index'] for r in rows],list(range(1,20)))
        self.assertEqual(rows[-1]['target'],1.)
        self.assertTrue(all(0<=r['prediction']<=1 for r in rows))
        self.assertFalse(any(p.grad is not None for p in upstream.parameters()))

    def test_training_deadline_is_distinct_from_evaluation(self):
        n, success, y = rm_probe.rm_prefix(190, True)
        self.assertEqual(n, 150); self.assertFalse(success)
        self.assertEqual(set(y), {0.})
        n, success, y = rm_probe.rm_prefix(150, True)
        self.assertTrue(success); self.assertEqual(y[-1], 1.)
        self.assertAlmostEqual(y[-9], .99)

    def test_incomplete_failure_must_not_become_a_label(self):
        with self.assertRaises(ValueError): rm_probe.rm_prefix(100, False)
        self.assertEqual(set(rm_probe.rm_prefix(220, False)[2]), {0.})

    def test_scoring_context_keeps_each_action_endpoint_once(self):
        blocks=rm_probe.real_scoring_blocks(19)
        self.assertEqual([p for context, endpoints in blocks for p in endpoints], list(range(1,20)))
        self.assertEqual(blocks[0][0], [0,0,1,2,3,4,5,6,7,8])
        self.assertEqual(blocks[-1][0], list(range(10,20)))
        self.assertEqual(blocks[-1][1], [17,18,19])

    def test_probe_uses_exact_models_and_disjoint_initial_states(self):
        p=rm_probe.probe_spec()
        self.assertEqual(p['models']['refill']['policy_step'],275)
        self.assertEqual(p['models']['orange150']['policy_step'],594)
        self.assertFalse(set(p['calibration_initial_states']) & set(p['test_initial_states']))
        self.assertEqual(set(p['initial_states']),set(p['calibration_initial_states'])|set(p['test_initial_states']))
        self.assertEqual(p['rm_action_horizon'],150)
        self.assertEqual(p['collection_action_horizon'],220)
        self.assertTrue(p['no_optimizer_steps'])

if __name__=='__main__':unittest.main()
