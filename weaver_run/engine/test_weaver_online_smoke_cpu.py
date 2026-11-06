"""The smoke admission and reference arithmetic are CPU-testable, with no CUDA."""
import copy
import sys
import json
from pathlib import Path
import subprocess
import tempfile
import time
from unittest import mock
import unittest

from weaver_online_smoke import GPU_UUIDS, validate_plan, tiny_backend, synthetic_row, compare_reference,synthetic_reward_cohort,wait_postflight
from weaver_online_engine import WeaverOnlineEngine

try:
    import torch
except ImportError:
    torch = None


def plan():
    return {'protocol': 'weaver_online_tiny_smoke_v1', 'run_id': 'weaverintegrated_preflight_cpu_test',
        'expected_uid': 1004, 'hostname': 'h20-xyff52ab-0001',
        'python': '/env/weaver/bin/python', 'output': '/tmp/new-smoke', 'max_seconds':180,
        'gpu_uuids': list(GPU_UUIDS), 'mps': {'pipe':'/dev/shm/mps_lcx1_gpu47_pzero2_20261020/pipe',
            'server': {'pid':691086,'birth':361065122,'uid':1004},
            'control': {'pid':690850,'birth':361065009,'uid':1004}}}


class Admission(unittest.TestCase):
    def test_exact_authorized_resources_and_bounded_plan(self):
        result = validate_plan(plan())
        self.assertFalse(result['gpu_started'])
        self.assertEqual(result['publications'], [0,1,2])
        self.assertEqual(result['gpu_processes'], 8)

    def test_wrong_gpu_uid_deadline_identity_and_marker_rejected(self):
        changes = [('expected_uid',0), ('max_seconds',301), ('run_id','formal-run'),
                   ('gpu_uuids', list(reversed(GPU_UUIDS)))]
        for key, value in changes:
            value_plan = plan(); value_plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plan(value_plan)
        value_plan = plan(); del value_plan['mps']['server']['birth']
        with self.assertRaises(ValueError):
            validate_plan(value_plan)

    def test_synthetic_rows_have_no_shared_mutable_state(self):
        first = synthetic_row(0,0,1); first['target'][0] = 100
        self.assertNotEqual(first, synthetic_row(0,0,1))

    def test_actual_plan_cli_completes_without_torch_or_gpu_start(self):
        with tempfile.TemporaryDirectory() as folder:
            plan_file = Path(folder)/'plan.json'
            plan_file.write_text(json.dumps(plan()))
            result = subprocess.run([sys.executable,str(Path(__file__).with_name('weaver_online_smoke.py')),
                '--plan',str(plan_file),'--plan-only'],text=True,capture_output=True,check=True,timeout=10)
        receipt = json.loads(result.stdout)
        self.assertFalse(receipt['gpu_started'])
        self.assertFalse(receipt['torch_imported'])

    def test_postflight_waits_for_two_idle_checks_without_restarting_any_task(self):
        with mock.patch('weaver_online_smoke.check_host',side_effect=[
                RuntimeError('GPU training/evaluation clients remain; do not overlap the smoke'),
                {'clients':[]},{'clients':[]}]) as checker,mock.patch('weaver_online_smoke.time.sleep'):
            result=wait_postflight(plan(),time.monotonic()+2)
        self.assertEqual(result['consecutive_idle_checks'],2)
        self.assertEqual(checker.call_count,3)

    def test_postflight_identity_change_fails_without_polling_or_signals(self):
        with mock.patch('weaver_online_smoke.check_host',side_effect=RuntimeError('MPS server PID/birth/UID changed')) as checker:
            with self.assertRaisesRegex(RuntimeError,'birth'):
                wait_postflight(plan(),time.monotonic()+2)
        self.assertEqual(checker.call_count,1)


@unittest.skipIf(torch is None, 'CPU Torch required for actual engine arithmetic')
class Reference(unittest.TestCase):
    def test_same_engine_two_updates_matches_global_batch_reference_and_releases_gradients(self):
        backend = tiny_backend(torch, 'cpu')
        engine = WeaverOnlineEngine(backend, world_size=1, micro_batch=1, accumulation_steps=8)
        reference = tiny_backend(torch, 'cpu')
        optimizer = torch.optim.AdamW(reference.model.parameters(), lr=1e-5, betas=(.9,.99), weight_decay=.01)
        hashes = []
        for update in (1,2):
            for policy in range(engine.policy_step+1, update*5+1):
                engine.advance_policy(policy)
            engine.begin_update(update*5,synthetic_reward_cohort(update))
            for rank in range(4):
                for micro in range(2):
                    engine.micro([synthetic_row(rank,micro,update)])
            result = compare_reference(torch,engine,reference,optimizer,update)
            self.assertLessEqual(result['weight_max_abs'],2e-7)
            self.assertTrue(engine.state()['gradients_released'])
            hashes.append(result['actual_parameter_hash'])
        self.assertEqual(engine.global_steps,2)
        self.assertEqual(engine.micro_steps,16)
        self.assertNotEqual(hashes[0],hashes[1])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
