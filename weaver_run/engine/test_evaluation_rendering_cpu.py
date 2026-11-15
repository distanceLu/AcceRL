import importlib
import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment_rendering import prepare_evaluation_renderer

class EvaluationBootstrapTests(unittest.TestCase):
    def test_uuid_restored_on_import_failure(self):
        binding = {'uuid': 'GPU-test', 'egl_device_id': 3, 'pci': '0000:01:00.0'}
        def fail(name):
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(os.environ['MUJOCO_EGL_DEVICE_ID'], '3')
            raise ImportError('simulator failed')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test'}, clear=True), patch('environment_rendering.discover_evaluation_binding', return_value=binding), patch('environment_rendering.discover_render_bindings', side_effect=AssertionError('parent must not enumerate EGL')), patch.object(importlib, 'import_module', side_effect=fail):
            with self.assertRaises(ImportError): prepare_evaluation_renderer('GPU-test')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-test')

    def test_diagnostic_osmesa_needs_no_gpu_discovery(self):
        with patch.dict(os.environ, {'ACCERL_EVAL_RENDER_BACKEND': 'osmesa', 'CUDA_VISIBLE_DEVICES': 'GPU-test'}, clear=True), patch('environment_rendering.discover_render_bindings') as discover:
            self.assertEqual(prepare_evaluation_renderer('GPU-test'), 'osmesa')
            discover.assert_not_called()
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-test')

    def test_tensorflow_import_is_rejected(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test'}, clear=True), patch.dict(sys.modules, {'tensorflow': object()}):
            with self.assertRaisesRegex(RuntimeError, 'TensorFlow'): prepare_evaluation_renderer('GPU-test')

    def test_discovery_isolated_with_cuda_hidden_and_timeout(self):
        import json
        import subprocess
        from environment_rendering import discover_evaluation_binding
        binding = {'uuid': 'GPU-test', 'egl_device_id': 2, 'pci': '0000:01:00.0'}
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test'}, clear=True), patch('environment_rendering.discover_render_bindings', side_effect=AssertionError('parent enumeration')), patch('environment_rendering.subprocess.check_output', return_value=json.dumps(binding)) as run:
            self.assertEqual(discover_evaluation_binding('GPU-test'), binding)
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(run.call_args.kwargs['timeout'], 40)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-test')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test'}, clear=True), patch('environment_rendering.subprocess.check_output', side_effect=subprocess.TimeoutExpired('discover', 40)):
            with self.assertRaises(subprocess.TimeoutExpired): discover_evaluation_binding('GPU-test')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-test')

    def test_numeric_admission_normalizes_before_bootstrap(self):
        from types import SimpleNamespace
        import frozen_policy_eval as base
        selected = base.APPROVED_GPUS[0]
        def bootstrap(uuid):
            self.assertEqual(uuid, selected)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], selected)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '4'}, clear=True), patch('subprocess.check_output', return_value='4, '+selected+'\n'), patch('environment_rendering.prepare_evaluation_renderer', side_effect=bootstrap) as prepare, patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))}):
            self.assertEqual(base.verify_single_gpu(), selected)
            prepare.assert_called_once_with(selected)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}, clear=True), patch('subprocess.check_output', return_value='0, GPU-unapproved\n'), patch('environment_rendering.prepare_evaluation_renderer') as prepare:
            with self.assertRaises(ValueError): base.verify_single_gpu()
            prepare.assert_not_called()

    def test_success_preserves_uuid_and_mapped_selector(self):
        binding = {'uuid': 'GPU-test', 'egl_device_id': 7, 'pci': '0000:01:00.0'}
        def imported(name):
            self.assertEqual(name, 'robosuite.utils.binding_utils')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(os.environ['MUJOCO_EGL_DEVICE_ID'], '7')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test'}, clear=True), patch('environment_rendering.discover_evaluation_binding', return_value=binding), patch('environment_rendering.discover_render_bindings', side_effect=AssertionError('parent must not enumerate EGL')), patch.object(importlib, 'import_module', side_effect=imported):
            self.assertEqual(prepare_evaluation_renderer('GPU-test'), 'egl')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-test')
            self.assertEqual(os.environ['MUJOCO_EGL_DEVICE_ID'], '7')

    def test_initialized_cuda_is_rejected_before_discovery(self):
        from types import SimpleNamespace
        torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test'}, clear=True), patch.dict(sys.modules, {'torch': torch}), patch('environment_rendering.discover_render_bindings') as discover:
            with self.assertRaisesRegex(RuntimeError, 'Torch CUDA'): prepare_evaluation_renderer('GPU-test')
            discover.assert_not_called()
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-test')

    def test_active_device_and_vendor_must_match(self):
        from environment_rendering import validate_active_renderer
        binding = {'pci': '0000:01:00.0'}
        for actual in ({'vendor': 'Mesa', 'renderer': 'llvmpipe', 'pci': binding['pci']},
                       {'vendor': 'NVIDIA', 'renderer': 'GPU', 'pci': '0000:02:00.0'}):
            with self.subTest(actual=actual), self.assertRaises(RuntimeError):
                validate_active_renderer(binding, actual)

if __name__ == '__main__': unittest.main()
