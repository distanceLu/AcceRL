import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weaver_run.reference_site import after_experiment as m


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = 10**12
        self.plan = {
            'training': dict(job_id='old', submission_id='sid-old', run_id='run-old',
                             root=str(self.root), git_commit='old-sha', cwd='/old', argv=['old']),
            'target': dict(submission_id='sid-new', root=str(self.root/'new')),
            'protected_files': {}, 'queue_state': str(self.root/'state.json')}
        self.row = dict(id='old', submission_id='sid-old', cwd='/old', argv_json='["old"]',
                        state='running', exitcode=None, started_ns=10, finished_ns=None)
        self.receipt = dict(job_id='old', run_id='run-old', source_validation={'git_commit': 'old-sha'},
                            state='succeeded', all_owned_exited=True, training_acceptance={'passed': True},
                            evaluation_acceptance={'passed': True}, finished_ns=100)
        self.state = dict(observed_ns=self.now, external_process_count=0, mps={'clients': []})
        self.write('queue_receipt.json', self.receipt)
        self.write('state.json', self.state)

    def write(self, name, value):
        (self.root/name).write_text(json.dumps(value), encoding='utf8')

    def complete(self):
        self.row.update(state='succeeded', exitcode=0, finished_ns=200)

    def gate(self, others=()):
        return m.evaluate(self.plan, [self.row, *others], self.now)['state']

    def test_training_completion_alone_does_not_start_next_job(self):
        self.assertEqual(self.gate(), 'waiting_training')
        self.complete()
        for field, value in [('evaluation_acceptance', {}), ('all_owned_exited', False), ('state', 'evaluating')]:
            with self.subTest(field=field):
                self.write('queue_receipt.json', dict(self.receipt, **{field: value}))
                self.assertEqual(self.gate(), 'waiting_training_receipt')
        self.write('queue_receipt.json', self.receipt)
        self.assertEqual(self.gate(), 'ready')

    def test_failed_or_replaced_predecessor_is_rejected(self):
        self.row['state'] = 'failed'
        with self.assertRaises(RuntimeError): self.gate()
        self.row.update(state='running', id='replaced')
        with self.assertRaises(RuntimeError): self.gate()

    def test_failed_acceptance_is_not_a_retryable_wait(self):
        self.complete()
        self.receipt['evaluation_acceptance'] = {'passed': False}
        self.write('queue_receipt.json', self.receipt)
        with self.assertRaises(RuntimeError): self.gate()

    def test_busy_gpu_queue_and_stale_telemetry_never_launch(self):
        self.complete()
        for change, expected in [({'external_process_count': 1}, 'waiting_idle'),
                                 ({'mps': {'clients': [123]}}, 'waiting_idle'),
                                 ({'observed_ns': 1}, 'waiting_telemetry')]:
            self.write('state.json', dict(self.state, **change))
            self.assertEqual(self.gate(), expected)
        self.write('state.json', self.state)
        self.assertEqual(self.gate([dict(submission_id='someone-else', state='queued')]), 'waiting_idle')

    def test_wrong_source_and_reversed_completion_times_are_rejected(self):
        self.complete()
        self.receipt['source_validation']['git_commit'] = 'another-commit'
        self.write('queue_receipt.json', self.receipt)
        with self.assertRaises(RuntimeError): self.gate()
        self.receipt['source_validation']['git_commit'] = 'old-sha'
        self.receipt['finished_ns'] = 201
        self.write('queue_receipt.json', self.receipt)
        with self.assertRaises(RuntimeError): self.gate()

    def test_submission_records_intent_and_does_not_retry_ambiguity(self):
        command = ['/workspace/run.sh', 'legacy-baseline', '--id', 'trial', '--training-seconds', '7200']
        self.plan['target'].update(label='trial', git_commit='new', checkout='/repo', launch_argv=command)
        saved = []
        def run(argv, **kwargs):
            self.assertEqual(argv, command)
            self.assertEqual(saved[0]['state'], 'submitting')
            return json.dumps(dict(id='trial', git_commit='new', job='queued-id', state='queued'))
        with patch.object(m, 'prepare'), patch.object(m, 'run', side_effect=run):
            result = m.submit_once(self.plan, {}, saved.append)
        self.assertEqual(result['job_id'], 'queued-id')
        with self.assertRaises(RuntimeError):
            m.submit_once(self.plan, {'attempted_ns': 1}, saved.append)


if __name__ == '__main__':
    unittest.main()
