import unittest
from types import SimpleNamespace
from pathlib import Path
import training_entry as t
import queue_entry as q
from test_fresh_wrapper_cpu import admitted_config

class JitDeploymentTests(unittest.TestCase):
    def test_manual_has_no_duration_or_calendar(self):
        a=SimpleNamespace(post_stop_grace_seconds=1800,evaluation_seconds=7200)
        b=q.validate_budgets(a)
        self.assertIsNone(b['deadline_ns']);self.assertIsNone(b['stop_at_ns'])
        self.assertFalse(b['four_hour_cutoff_enabled'])
        self.assertNotIn('measurement_seconds',b)
        self.assertIsNone(t.stop_cause(epochs=100000,elapsed=1e10))
        self.assertEqual(t.stop_cause(signal_number=15),'external_signal_15')
    def test_reject_inherited_duration(self):
        for extra in ({'measurement_seconds':14400},{'stop_control':Path('/old')}):
            with self.assertRaises(ValueError):q.validate_budgets(SimpleNamespace(post_stop_grace_seconds=1800,evaluation_seconds=7200,**extra))
    def test_reject_silent_jit_disable_or_wrong_ema(self):
        for key,value in [('jit_gae_mode','off'),('advantage_norm','optimizer_window'),('advantage_ema_beta',.99),('jit_context_micro',8)]:
            b,p,a=admitted_config();b['backend_options']['fresh'][key]=value
            with self.assertRaises(ValueError):t.validate_backend_admission(b,p,a)
    def test_world_profile_survives_environment_cleaning(self):
        source=Path('/frozen');root=Path('/new')
        a=SimpleNamespace(source=source,root=root)
        env=t.launch_environment(root,source,{'ACCERL_WORLD_BACKEND_CONFIG':'wrong'})
        expected=str(source/'examples/trajectory_01_imagined.json')
        self.assertEqual(env['ACCERL_WORLD_BACKEND_CONFIG'],expected)
        command=t.runner_command(a)
        self.assertEqual(command[command.index('--world-backend-config')+1],expected)
