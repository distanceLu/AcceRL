import unittest
from deadline_guard import guard_decision
from training_stop_policy import TrainingStopPolicy


class DeadlineTest(unittest.TestCase):
    def test_independent_deadline_and_save_grace(self):
        s = {'deadline_unix': 1000, 'heartbeat_ns': 1000_000_000_000}
        d = guard_decision(s, 1000, True)
        self.assertEqual(d['stop_reason'], 'user_deadline_reached')
        self.assertFalse(d['cleanup'])
        s['heartbeat_ns'] = 1899_000_000_000
        self.assertFalse(guard_decision(s, 1899, True)['cleanup'])
        s['heartbeat_ns'] = 1900_000_000_000
        self.assertTrue(guard_decision(s, 1900, True)['cleanup'])

    def test_stale_supervisor_has_short_error_grace(self):
        s = {'deadline_unix': 1000, 'heartbeat_ns': 1_000_000_000}
        d = guard_decision(s, 200, False)
        self.assertFalse(d['cleanup'])
        self.assertTrue(guard_decision(s, 320, False, d['failure_at'])['cleanup'])

    def test_preflight_cap_and_formal_deadline(self):
        common = {'ACCERL_SOFT_STOP_AT': '1792458000', 'ACCERL_STOP_ON_TARGET_85': '0',
                  'ACCERL_RUN_UNTIL_MANUAL_STOP': '0'}
        pre = TrainingStopPolicy.from_environment(dict(common, ACCERL_RUN_UNTIL_DEADLINE='0'))
        formal = TrainingStopPolicy.from_environment(dict(common, ACCERL_RUN_UNTIL_DEADLINE='1'))
        args = dict(global_step=30001, train_iters=30000, target_reached=True,
                    requested_stop=False, completed_epochs=30001, measure_stop_epochs=15)
        self.assertFalse(pre.should_continue(**args, now=1792457999))
        self.assertTrue(formal.should_continue(**args, now=1792457999))
        self.assertFalse(formal.should_continue(**args, now=1792458000))


if __name__ == '__main__':
    unittest.main()
