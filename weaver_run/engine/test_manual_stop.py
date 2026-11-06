import unittest
from training_stop_policy import TrainingStopPolicy


class ManualStopTest(unittest.TestCase):
    def test_target_and_lr_horizon_only_record(self):
        p = TrainingStopPolicy.from_environment({'ACCERL_RUN_UNTIL_MANUAL_STOP': '1'})
        self.assertFalse(p.stop_on_target)
        self.assertIsNone(p.deadline_unix_seconds)
        self.assertTrue(p.should_continue(global_step=30001, train_iters=30000,
            target_reached=True, requested_stop=False, completed_epochs=30001))
        self.assertFalse(p.should_continue(global_step=30, train_iters=30000,
            target_reached=True, requested_stop=True))
        self.assertEqual(p.stop_reason(global_step=30, train_iters=30000,
            target_reached=True, requested_stop=True), 'scheduled_stop')

    def test_bounded_preflight_is_preserved(self):
        p = TrainingStopPolicy.from_environment({'ACCERL_STOP_ON_TARGET_85': '0'})
        self.assertFalse(p.should_continue(global_step=15, train_iters=30000,
            target_reached=False, requested_stop=False, completed_epochs=15, measure_stop_epochs=15))

    def test_conflicting_modes_rejected(self):
        with self.assertRaises(ValueError):
            TrainingStopPolicy.from_environment({'ACCERL_RUN_UNTIL_MANUAL_STOP': '1',
                'ACCERL_RUN_UNTIL_DEADLINE': '1', 'ACCERL_SOFT_STOP_AT': '1792371600'})


if __name__ == '__main__':
    unittest.main()
