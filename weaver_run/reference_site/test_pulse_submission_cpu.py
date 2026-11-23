import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from weaver_run.reference_site import jit_bench_queue


class PulseSubmissionTest(unittest.TestCase):
    def test_eight_worker_managed_argv_matches_actual_job(self):
        import launch
        plan = {'comparison':'admission_pulse', 'real_sampling':'uniform_blocks',
                'normal_only':True, 'real_workers':8}
        env = dict(jit_bench_queue.pulse_environment(plan), ACCERL_JIT_BENCH_PLAN='plan.json',
                   ACCERL_PIPELINE_CONFIG='pipeline.json', ACCERL_FINAL_CHECKPOINT='weights')
        with patch.dict(os.environ, env), patch.object(os, 'chdir'), patch.object(os, 'execve') as run:
            launch.exec_managed_driver(jit_bench_queue.benchmark_argv(plan)[2:])
        executable, argv, child_env = run.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(argv[argv.index('--num-rollout-workers')+1], '8')
        self.assertIn('Real8_Imag0', argv[argv.index('--exp-name')+1])
        self.assertEqual(child_env['WEAVER_BENCH_REAL_WORKERS'], '8')
        self.assertEqual(child_env['WEAVER_BENCH_NORMAL_ONLY'], '1')

    def test_pulse_driver_receives_jit_and_sampling_flags(self):
        pulse = jit_bench_queue.benchmark_argv({'comparison':'admission_pulse',
                                               'real_sampling':'uniform_blocks'})
        self.assertIn('--trajectory-jit', pulse)
        self.assertEqual(pulse[pulse.index('--real-sampling')+1], 'uniform_blocks')
        self.assertNotIn('--trajectory-jit', jit_bench_queue.benchmark_argv({'comparison':'episode_sampling'}))

    def test_pulse_uses_existing_runtime_and_three_live_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = Mock()
            cli.LABEL = re.compile(r'[a-z0-9-]+')
            cli.PARENT = root
            cli.EXPERIMENTS = root / 'experiments'
            cli.EXPERIMENTS.mkdir()
            cli.HOME = root
            cli.RUNTIME = jit_bench_queue.REPO
            cli.CHECKOUT = jit_bench_queue.REPO
            cli.lookup.return_value = None
            cli.published.return_value = 'a'*40
            cli.create.return_value = {'id': 'lease', 'state': 'queued'}
            cli.PYTHON = 'python'
            (root/'mps.json').write_bytes(b'{}')
            cli.PROFILES = {'restore': {'mps_receipt': hashlib.sha256(b'{}').hexdigest()}}

            receipt = jit_bench_queue.submit(cli, 'pulse-test', pulse=True)
            plan = json.loads((cli.EXPERIMENTS/'pulse-test'/'plan.json').read_bytes())
            self.assertEqual(receipt['submission_id'], 'weaver-bench-pulse-pulse-test')
            cli.ready.assert_called_once()
            cli.pin_worktree.assert_called_once_with(cli.CHECKOUT, cli.RUNTIME, 'a'*40)
            self.assertEqual(plan['comparison'], 'admission_pulse')
            self.assertEqual(plan['real_sampling'], 'uniform_blocks')
            self.assertEqual((plan['warmup_updates'], plan['measured_blocks'], plan['updates_per_arm']), (1, 3, 3))
            self.assertEqual(plan['pulse_seconds'], 5)
            self.assertNotIn('measured_pairs', plan)
            self.assertEqual(plan['gpu_ids'], [4, 5, 6, 7])
            self.assertEqual(plan['checkpoint_policy_step'], 189)
            self.assertEqual(plan['global_batch'], 576)

            jit_bench_queue.submit(cli, 'workers-eight', pulse=True, real_workers=8, normal_only=True)
            path = cli.EXPERIMENTS/'workers-eight'
            plan = json.loads((path/'plan.json').read_bytes())
            pipeline = json.loads((path/'git-pipeline.json').read_bytes())
            self.assertTrue(plan['normal_only'])
            self.assertEqual(plan['real_workers'], 8)
            self.assertEqual((pipeline['real_workers'], pipeline['max_real_in_flight']), (8, 8))
            self.assertEqual(jit_bench_queue.pulse_environment(plan), {
                'WEAVER_BENCH_REAL_WORKERS':'8', 'WEAVER_BENCH_NORMAL_ONLY':'1',
                'ACCERL_TENSORBOARD':'0'})
            self.assertEqual(jit_bench_queue.pulse_environment({'comparison':'episode_sampling'}), {})
            old = jit_bench_queue.pulse_environment({'comparison':'admission_pulse'})
            self.assertEqual((old['WEAVER_BENCH_REAL_WORKERS'], old['WEAVER_BENCH_NORMAL_ONLY']), ('4', '0'))
            with self.assertRaises(ValueError):
                jit_bench_queue.submit(cli, 'invalid-count', pulse=True, real_workers=6)


if __name__ == '__main__':
    unittest.main()
