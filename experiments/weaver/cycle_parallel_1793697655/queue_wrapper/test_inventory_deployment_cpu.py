import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import training_entry as training


class MeasurementTests(unittest.TestCase):
    def test_four_hours_begin_after_first_successful_update(self):
        receipt = {}
        self.assertFalse(training.measurement_stop_due(receipt, 100000., 0, 14400))
        self.assertNotIn('started_monotonic', receipt['measurement'])
        self.assertFalse(training.measurement_stop_due(receipt, 200000., 1, 14400))
        self.assertFalse(training.measurement_stop_due(receipt, 214399.99, 60, 14400))
        self.assertTrue(training.measurement_stop_due(receipt, 214400., 60, 14400))
        self.assertTrue(training.measurement_stop_due(receipt, 214401., 61, 14400))
        self.assertEqual(receipt['measurement']['started_monotonic'], 200000.)
        self.assertFalse(receipt['measurement']['hard_process_deadline'])

    def test_duration_is_bounded_and_immutable(self):
        for seconds in (True, 0, 3599, 21601, 14400.0):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                training.measurement_stop_due({}, 0., 1, seconds)
        receipt = {}
        training.measurement_stop_due(receipt, 0., 1, 14400)
        with self.assertRaises(ValueError):
            training.measurement_stop_due(receipt, 1., 2, 18000)

    def test_both_launch_layers_use_explicit_lookahead_config(self):
        source, root = Path('/new/source'), Path('/new/root')
        command = training.runner_command(SimpleNamespace(source=source, root=root))
        self.assertEqual(command[command.index('--pipeline-config')+1], str(source/training.PIPELINE_CONFIG))
        env = training.launch_environment(root, source, {'ACCERL_PIPELINE_CONFIG': '/wrong'})
        self.assertEqual(env['ACCERL_PIPELINE_CONFIG'], str(source/training.PIPELINE_CONFIG))
        self.assertEqual(command[command.index('--cost-scheduler-config')+1],
                         str(source/training.COST_SCHEDULER_CONFIG))


if __name__ == '__main__':
    unittest.main()
