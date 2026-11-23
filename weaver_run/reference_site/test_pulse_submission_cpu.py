import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import Mock

from weaver_run.reference_site import jit_bench_queue


class PulseSubmissionTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
