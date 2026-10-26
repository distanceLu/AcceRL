"""Updated manual-profile configuration fixtures; historical filename retained."""
import ast
from pathlib import Path
import tempfile
import unittest
from test_manual_controller_cpu import env,load,SOURCE


class ProfiledManualTest(unittest.TestCase):
    def test_exact_timer_windows_override_inherited_profile_settings_without_stop_deadline(self):
        value=env({'ACCERL_NSIGHT_NVTX':'0','ACCERL_NSIGHT_MODE':'epoch',
                   'ACCERL_NSIGHT_STEPS':'1,5,15','ACCERL_NSIGHT_SECONDS':'600',
                   'ACCERL_MEASURE_STOP_AFTER_EPOCHS':'15','ACCERL_SOFT_STOP_AT':'1'})
        self.assertEqual(value['ACCERL_NSIGHT_NVTX'],'1')
        self.assertEqual(value['ACCERL_NSIGHT_MODE'],'timer')
        self.assertEqual(value['ACCERL_NSIGHT_STEPS'],'14,15')
        self.assertEqual(value['ACCERL_NSIGHT_SECONDS'],'30')
        self.assertNotIn('ACCERL_MEASURE_STOP_AFTER_EPOCHS',value)
        self.assertNotIn('ACCERL_SOFT_STOP_AT',value)
        self.assertEqual(value['ACCERL_RUN_UNTIL_MANUAL_STOP'],'1')

    def test_actual_driver_uses_public_fail_open_capture_and_runner_has_no_dummy_window(self):
        tree=ast.parse((SOURCE/'runner.py').read_text(encoding='utf-8'))
        run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
        self.assertNotIn('NsightCaptureWindows',ast.unparse(run))
        train=ast.parse((SOURCE/'train.py').read_text(encoding='utf-8'))
        imported=[node for node in train.body if isinstance(node,ast.ImportFrom) and node.module=='nsight_capture']
        self.assertTrue(any(alias.name=='NsightCaptureWindows' for node in imported for alias in node.names))
        capture=ast.parse((SOURCE/'nsight_capture.py').read_text(encoding='utf-8'))
        aliases=[node for node in capture.body if isinstance(node,ast.Assign)
                 and any(isinstance(target,ast.Name) and target.id=='NsightCaptureWindows' for target in node.targets)]
        self.assertEqual(ast.unparse(aliases[-1].value),'ContentionCaptureWindows')

    def test_metadata_requests_two_windows_without_claiming_hardware_metrics(self):
        configuration=load('runner.py','measurement_configuration',{})()
        self.assertTrue(configuration['nsight_collection'])
        capture=configuration['nsight_capture']
        self.assertEqual(capture['seconds_per_window'],30)
        self.assertEqual(capture['policy_steps'],[14,15])
        self.assertEqual(capture['capture_range_end'],'repeat-shutdown:2:async')
        self.assertEqual(capture['kill'],'none')
        self.assertFalse(capture['trace_verified'])
        self.assertFalse(capture['inner_nsys_wrapper'])
        self.assertTrue(capture['training_continues_after_capture'])
        self.assertFalse(configuration['hardware_metrics']['enabled'])
        for metric in ('sm_active','occupancy'):self.assertIsNone(configuration['hardware_metrics'][metric])

    def test_collect_and_state_record_requested_and_actual_measurement_configuration(self):
        namespace={'stamp':lambda:'cpu','SOURCE':Path('/source')}
        configuration=load('runner.py','measurement_configuration',namespace)()
        collect=load('runner.py','collect',namespace)
        with tempfile.TemporaryDirectory() as temporary:
            state={'run_id':'manual_profile_cpu','state':'running','mode':'manual',
                   'measurement_injection_verified':False,'measurement_configuration_error':'untrusted'}
            result=collect(Path(temporary),state)
        for key,value in configuration['nsight_capture'].items():self.assertEqual(result['nsight_capture'][key],value)
        self.assertFalse(result['nsight_capture']['actual_enabled'])
        self.assertEqual(result['measurement_configuration_error'],'untrusted')
        self.assertNotIn('error',result)
        code=ast.unparse(ast.parse((SOURCE/'runner.py').read_text(encoding='utf-8')))
        self.assertIn('state.update(measurement_configuration())',code)

    def test_training_child_stays_bash_and_report_describes_limited_windows(self):
        tree=ast.parse((SOURCE/'runner.py').read_text(encoding='utf-8'))
        run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
        calls=[node for node in ast.walk(run) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute) and node.func.attr=='Popen']
        self.assertEqual(calls[0].args[0].elts[0].value,'bash')
        self.assertFalse(any('nsys' in ast.unparse(node.args[0]) for node in calls))
        report=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='report')
        text=ast.unparse(report)
        self.assertIn('第14独立轮/15联合轮各30秒',text)
        self.assertIn('随后停止采集且训练继续',text)
        self.assertNotIn('不开Nsight',text)


if __name__=='__main__':unittest.main()
