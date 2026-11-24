import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from weaver_run.reference_site import jit_bench_queue as queue
import launch


class TrajectoryLayoutSubmissionTest(unittest.TestCase):
    def test_submit_and_lookup_use_existing_runtime_and_fixed_p189(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cli = Mock(EXPERIMENTS=base/'experiments', PARENT=base, HOME=base,
                       CHECKOUT=queue.REPO, RUNTIME=queue.REPO, PYTHON='python')
            cli.EXPERIMENTS.mkdir()
            cli.LABEL = re.compile(r'[a-z0-9-]+')
            cli.PROFILES = {'restore': {'mps_receipt':hashlib.sha256(b'{}').hexdigest()}}
            (base/'mps.json').write_bytes(b'{}')
            cli.lookup.return_value = None
            cli.published.return_value = 'a'*40
            cli.create.return_value = {'id':'job','state':'queued'}
            result = queue.submit(cli, 'layout-test', layout=True)
            root = cli.EXPERIMENTS/'layout-test'
            plan = json.loads((root/'plan.json').read_bytes())
            self.assertEqual(result['submission_id'], 'weaver-bench-layout-layout-test')
            self.assertEqual(plan['comparison'], 'trajectory_layout')
            self.assertEqual((plan['warmup_pairs'], plan['measured_pairs'], plan['minimum_replay_chunks']),
                             (1,8,2753))
            self.assertEqual(plan['layout_pool_sizes'], [581,2753])
            self.assertEqual((plan['real_workers'], plan['online_eval_workers'], plan['global_batch'],
                              plan['rank_size'], plan['train_micro'], plan['context_micro']),
                             (4,0,576,144,16,2))
            self.assertFalse(plan['fixed_validation'])
            self.assertTrue(plan['no_micro_cache_trim'])
            self.assertEqual(plan['gpu_ids'], [4,5,6,7])
            self.assertEqual((plan['checkpoint_policy_step'],plan['checkpoint_sha256']),
                             (189,queue.WEIGHTS_SHA256))
            self.assertEqual(plan['git_commit'], 'a'*40)
            cli.ready.assert_called_once()
            cli.pin_worktree.assert_called_once_with(cli.CHECKOUT, cli.RUNTIME, 'a'*40)
            self.assertEqual(cli.create.call_args.args[0]['cwd'], str(cli.RUNTIME))
            self.assertFalse((base/'library').exists())
            cli.lookup.return_value = {'id':'job','state':'queued'}
            status = queue.submit(cli, 'layout-test', layout=True, status=True)
            self.assertEqual((status['job_id'],status['state']), ('job','queued'))
            cli.lookup.assert_called_with(result['submission_id'])
            cli.create.assert_called_once()

    def test_managed_driver_keeps_paired_checkpoint_path_and_disables_micro_trim(self):
        plan = dict(comparison='trajectory_layout', no_micro_cache_trim=True)
        options = queue.benchmark_argv(plan)[2:]
        env = dict(ACCERL_JIT_BENCH_PLAN='plan.json', ACCERL_JIT_BENCH_CHECKPOINT='p189',
                   ACCERL_PIPELINE_CONFIG='pipeline.json', ACCERL_FINAL_CHECKPOINT='weights')
        with patch.dict(os.environ,env,clear=True), patch.object(os,'chdir'), patch.object(os,'execve') as run:
            launch.exec_managed_driver(options)
        _,argv,effective = run.call_args.args
        self.assertEqual(effective['ACCERL_TRIM_MICRO_CACHE'],'0')
        self.assertEqual(effective['ACCERL_TRIM_IDLE_CACHE'],'1')
        self.assertEqual(effective['ACCERL_TRAJECTORY_JIT'],'0')
        self.assertEqual(effective['ACCERL_MODEL_FREE_REAL_WORKERS'],'4')
        self.assertEqual(effective['ACCERL_NUM_EVAL_WORKERS'],'0')
        self.assertEqual(effective['ACCERL_JIT_BENCH_CHECKPOINT'],'p189')
        self.assertEqual(argv[argv.index('--num-rollout-workers')+1],'4')

    def test_public_commands_route_to_site_cli(self):
        for command in ('bench-trajectory-layout','bench-trajectory-layout-status'):
            argv=['run.sh',command,'--id','layout-test']
            with self.subTest(command=command), patch.object(sys,'argv',argv), \
                    patch.object(launch.socket,'gethostname',return_value=launch.REFERENCE_SERVER), \
                    patch.object(os,'execv',side_effect=InterruptedError) as run:
                with self.assertRaises(InterruptedError):
                    launch.main()
                self.assertEqual(run.call_args.args[1][-3:],[command,'--id','layout-test'])

    def test_site_entry_distinguishes_submission_from_status(self):
        from weaver_run.reference_site import cli
        with tempfile.TemporaryDirectory() as tmp:
            site=Path(tmp)
            (site/'.owner').write_text('WEAVER server catalog v1\n')
            for command,status in [('bench-trajectory-layout',False),('bench-trajectory-layout-status',True)]:
                with self.subTest(command=command), patch.object(cli,'HOME',site), \
                        patch.object(cli,'EXPERIMENTS',site), patch.object(os,'getuid',return_value=1004), \
                        patch.object(cli.socket,'gethostname',return_value=launch.REFERENCE_SERVER), \
                        patch.object(sys,'argv',['cli.py',command,'--id','layout-test']), \
                        patch.object(sys,'stdout',io.StringIO()), patch.object(queue,'submit',return_value={}) as submit:
                    cli.main()
                    self.assertEqual(submit.call_args.args,(cli,'layout-test'))
                    self.assertTrue(submit.call_args.kwargs['layout'])
                    self.assertEqual(submit.call_args.kwargs['status'],status)
                    self.assertFalse(submit.call_args.kwargs['pulse'] or submit.call_args.kwargs['sampling'])


if __name__=='__main__':
    unittest.main()
