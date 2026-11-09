"""Managed-Git integration contracts; fake receipts, no model/GPU/network."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'queue_wrapper'))
import parallel_eval as supervisor
import plan_contract as contract
import queue_entry as queue
from test_parallel_contract_cpu import BASE, TOKEN, make_plan, worker_ledgers
from test_parallel_supervisor_cpu import completion_reports
import test_parallel_supervisor_cpu as supervisor_tests


class ManagedEvaluationTests(unittest.TestCase):
    def args(self, root):
        return SimpleNamespace(source=HERE.parents[1] / 'engine', root=root,
                               git_commit='a' * 40, evaluation_modes='greedy,stochastic',
                               evaluation_seed=20261026, evaluation_repeats=1,
                               baseline_checkpoint=None)

    def test_queue_selects_four_replica_entry_and_shared_absolute_deadline(self):
        args = self.args(Path('/experiment'))
        command = queue.evaluation_command(args, 7200, run_id='exact-run', deadline_unix=123456)
        self.assertEqual(Path(command[1]), HERE / 'parallel_eval.py')
        self.assertEqual(command[command.index('--run-id') + 1], 'exact-run')
        self.assertEqual(command[command.index('--git-commit') + 1], 'a' * 40)
        self.assertEqual(command[command.index('--deadline-unix') + 1], '123456')
        self.assertEqual(command[command.index('--max-seconds') + 1], '7200')
        self.assertNotIn('frozen_policy_eval.py', command[1])

    def test_explicit_baseline_checkpoint_is_forwarded_without_changing_plan(self):
        args = self.args(Path('/experiment'))
        args.baseline_checkpoint = Path('/weights/baseline.pt')
        command = queue.evaluation_command(args, 7200)
        self.assertEqual(command[-2:], ['--baseline-checkpoint', str(args.baseline_checkpoint)])

    def test_foreign_or_unaudited_training_receipt_never_reaches_gpu_admission(self):
        mutations = [dict(job_id='another-job'), dict(run_id='another-run'),
                     dict(runner_exit_code=-15), dict(state='failed'), dict(all_owned_exited=False),
                     dict(checkpoint_cpu_audit={'valid': False}),
                     dict(final_source_validation={'git_commit': 'b' * 40}),
                     dict(acceptance={'passed': True, 'reasons': ['audit_error']})]
        for mutation in mutations:
            adapter = supervisor_tests.StoppedAdmissionTests.adapter([])
            receipt = adapter.read_json(Path('training_queue_receipt.json'))
            receipt.update(mutation)
            adapter.read_json = lambda path: receipt
            with self.subTest(mutation=mutation), \
                    mock.patch.object(supervisor.os, 'getuid', return_value=1004, create=True), \
                    mock.patch.dict(os.environ, {'EXPERIMENT_QUEUE_JOB_ID': 'cpu-reservation'}, clear=True), \
                    self.assertRaises(ValueError):
                supervisor.stopped_admission(adapter, Path('/master'), self.args(None).source,
                                             'cpu-stopped-run', 'a' * 40)
            adapter.wait_idle.assert_not_called()
            adapter.launch_environment.assert_not_called()

    def test_foreign_git_adapter_directory_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'same Git checkout'):
            supervisor.load_adapter(Path(tmp), self.args(None).source, 'a' * 40)

    def test_adapter_verifies_same_runtime_git_revision(self):
        fake = SimpleNamespace(verify_git=mock.Mock())
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=mock.Mock()))
        source = self.args(None).source
        with mock.patch.object(supervisor.importlib.util, 'spec_from_file_location', return_value=spec), \
                mock.patch.object(supervisor.importlib.util, 'module_from_spec', return_value=fake):
            self.assertIs(supervisor.load_adapter(HERE.parent / 'queue_wrapper', source, 'a' * 40), fake)
        fake.verify_git.assert_called_once_with(source.resolve(), 'a' * 40)

    def write_complete(self, root):
        output = root / 'policy_eval'
        output.mkdir()
        plan = make_plan()
        shards = contract.make_shards(plan)
        reports = completion_reports(plan, shards)
        summary = supervisor.completion_acceptance(
            plan, shards, worker_ledgers(plan, shards), reports, dict.fromkeys(range(4), 0),
            {'survivors': [], 'errors': []}, TOKEN, BASE)
        summary.update(initial_state_count=50, modes=plan['modes'],
                       checkpoint_sha256={c['name']: c['sha256'] for c in plan['checkpoints']})
        for name, value in [('plan.json', plan), ('shards.json', shards),
                            ('summary.json', summary), ('completed.json', summary)]:
            BASE.atomic_json(output / name, value)
        rows = contract.merge_rows(plan, shards, worker_ledgers(plan, shards), TOKEN, BASE)
        (output / 'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        for i, report in reports.items():
            worker = output / 'workers' / f'worker_{i}'
            worker.mkdir(parents=True)
            BASE.atomic_json(worker / 'completed.json', report)
        return output

    def test_four_complete_workers_52_52_48_48_at_220_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.write_complete(root)
            self.assertEqual([w['expected_episode_records'] for w in
                              json.loads((output / 'shards.json').read_text())['workers']], [52, 52, 48, 48])
            result = queue.evaluation_acceptance(root, 0, [])
            self.assertTrue(result['passed'], result)

    def test_legacy_single_gpu_completion_is_not_accepted_as_four_gpu_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.write_complete(root)
            (output / 'shards.json').unlink()
            self.assertFalse(queue.evaluation_acceptance(root, 0, [])['passed'])

    def test_incomplete_worker_rejected_even_when_global_file_claims_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.write_complete(root)
            path = output / 'workers/worker_2/completed.json'
            value = json.loads(path.read_text())
            value['completed_episode_records'] -= 1
            BASE.atomic_json(path, value)
            self.assertIn('parallel_worker_result_mismatch:2',
                          queue.evaluation_acceptance(root, 0, [])['reasons'])

    def test_collection_150_cap_cannot_leak_into_formal_evaluation_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.write_complete(root)
            path = output / 'plan.json'
            value = json.loads(path.read_text())
            value['horizon_action_steps'] = 150
            BASE.atomic_json(path, value)
            self.assertFalse(queue.evaluation_acceptance(root, 0, [])['passed'])

    def test_empty_episode_ledger_cannot_pass_with_only_complete_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.write_complete(root)
            (output / 'episodes.jsonl').write_text('', encoding='utf-8')
            self.assertIn('parallel_episode_ledger_coverage_or_summary_mismatch',
                          queue.evaluation_acceptance(root, 0, [])['reasons'])

    def test_process_survivors_and_nonzero_exit_reject_complete_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_complete(root)
            self.assertFalse(queue.evaluation_acceptance(root, 2, [])['passed'])
            self.assertFalse(queue.evaluation_acceptance(root, 0, [{'pid': 12}])['passed'])


if __name__ == '__main__':
    unittest.main()
