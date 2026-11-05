import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import stop_control
import queue_entry


class CalendarTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name).resolve()/'stop.json'
        self.policy=dict(schema=1,submission_id='owned',revision=1,authorized_ns=10**18,
            stop_at_ns=10**18+16*3600*10**9,timezone='Asia/Shanghai',authorization='user tomorrow 10AM',
            job_id='job',run_id='run')
        self.save();self.args=SimpleNamespace(stop_control=self.path,stop_submission='owned',
            post_stop_grace_seconds=1800,evaluation_seconds=7200,measurement_seconds=14400)
        self.receipt={'run_id':'run'}
    def save(self):self.path.write_text(json.dumps(self.policy),encoding='utf8')
    def due(self,ns):return stop_control.due(self.receipt,self.args,epochs=1,monotonic_now=100000.,wall_ns=ns,job_id='job')
    def test_calendar_mode_never_uses_four_hour_cutoff(self):
        self.assertFalse(self.due(self.policy['authorized_ns']+5*3600*10**9))
        self.assertFalse(self.receipt['measurement']['four_hour_cutoff_enabled'])
        self.assertNotIn('measurement_seconds',queue_entry.validate_budgets(self.args))
    def test_exact_deadline_requests_boundary_stop(self):
        self.assertFalse(self.due(self.policy['stop_at_ns']-1))
        self.assertTrue(self.due(self.policy['stop_at_ns']))
    def test_revision_changes_are_recorded_without_restarting(self):
        self.due(self.policy['authorized_ns'])
        self.policy['revision']+=1;self.policy['stop_at_ns']+=3600*10**9;self.save()
        self.assertFalse(self.due(self.policy['stop_at_ns']-1))
        self.assertEqual(len(self.receipt['calendar_stop_history']),2)
    def test_same_revision_cannot_change_deadline(self):
        self.due(self.policy['authorized_ns'])
        self.policy['stop_at_ns']+=1;self.save()
        with self.assertRaises(ValueError):self.due(self.policy['authorized_ns'])
    def test_wrong_run_job_or_submission_is_rejected(self):
        for field in ('run_id','job_id','submission_id'):
            prior=self.policy[field];self.policy[field]='wrong';self.save()
            with self.subTest(field=field),self.assertRaises(ValueError):self.due(self.policy['authorized_ns'])
            self.policy[field]=prior
    def test_missing_control_fails_instead_of_silently_using_four_hours(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):self.due(self.policy['authorized_ns'])


if __name__=='__main__':unittest.main()
