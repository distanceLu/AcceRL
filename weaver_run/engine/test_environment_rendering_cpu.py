"""Placement contracts without loading CUDA, EGL, Ray or the simulator."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


class RenderingTests(unittest.TestCase):
    def api(self):
        spec = importlib.util.find_spec('environment_rendering')
        self.assertIsNotNone(spec, 'Explicit environment rendering admission is required')
        import environment_rendering
        return environment_rendering

    def fixtures(self):
        bindings = [{'index': str(i), 'uuid': f'GPU-{i}', 'resource': f'card{i}'} for i in (4,5,6,7)]
        inventory = [{'index': str(i), 'uuid': f'GPU-{i}', 'pci': f'00000001:{i:02x}:00.0'} for i in (4,5,6,7)]
        devices = [{'egl_device_id': 0, 'pci': None},
                   *[{'egl_device_id': j+1, 'pci': f'0001:{i:02x}:00.0'} for j,i in enumerate((7,5,4,6))]]
        return bindings, inventory, devices

    def test_egl_order_is_matched_by_pci_not_cuda_or_physical_index(self):
        rows = self.api().select_render_bindings(*self.fixtures())
        self.assertEqual([r['egl_device_id'] for r in rows], [3,2,4,1])
        self.assertEqual([r['uuid'] for r in rows], ['GPU-4','GPU-5','GPU-6','GPU-7'])

    def test_missing_ambiguous_or_changed_gpu_is_rejected(self):
        api = self.api()
        for kind in ('missing','duplicate','uuid'):
            bindings, inventory, devices = self.fixtures()
            if kind == 'missing': devices = devices[:-1]
            elif kind == 'duplicate': devices.append(dict(devices[1],egl_device_id=8))
            else: inventory[0]['uuid'] = 'GPU-wrong'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                api.select_render_bindings(bindings, inventory, devices)

    def test_workers_hide_cuda_and_select_their_policy_replica(self):
        api = self.api()
        rows = api.select_render_bindings(*self.fixtures())
        env = {'ACCERL_ENV_RENDER_BACKEND':'egl','ACCERL_EXPERIMENT_ROOT':'/mnt/data/test'}
        for replica, label, want in ((0,'real-0',3),(1,'eval-0',2),(2,'eval-1',4),(3,'eval-2',1)):
            options = api.render_worker_options(rows, replica, label, environ=env)
            self.assertEqual(options['num_gpus'], 0)
            child = options['runtime_env']['env_vars']
            self.assertEqual(child['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(child['MUJOCO_EGL_DEVICE_ID'], str(want))
            self.assertEqual(child['MUJOCO_GL'], 'egl')
            self.assertEqual(child['PYOPENGL_PLATFORM'], 'egl')
            self.assertEqual(json.loads(child['ACCERL_RENDER_BINDING'])['uuid'], f'GPU-{replica+4}')
            self.assertTrue(child['__GL_SHADER_DISK_CACHE_PATH'].startswith('/mnt/data/test/'))

    def test_backend_configuration_clears_unbound_selector_and_allows_explicit_osmesa(self):
        api = self.api()
        env = {'ACCERL_ENV_RENDER_BACKEND':'egl','MUJOCO_GL':'osmesa','PYOPENGL_PLATFORM':'osmesa',
               'MUJOCO_EGL_DEVICE_ID':'0'}
        self.assertEqual(api.configure_render_backend(env), 'egl')
        self.assertNotIn('MUJOCO_EGL_DEVICE_ID', env)
        self.assertEqual(env['PYOPENGL_PLATFORM'], 'egl')
        env['ACCERL_ENV_RENDER_BACKEND'] = 'osmesa'
        api.configure_render_backend(env)
        self.assertEqual(env['MUJOCO_GL'], 'osmesa')
        env['ACCERL_ENV_RENDER_BACKEND'] = 'automatic'
        with self.assertRaises(ValueError): api.configure_render_backend(env)

    def test_wrong_card_or_software_renderer_cannot_pass(self):
        api = self.api()
        expected = {'pci':'0001:04:00.0','uuid':'GPU-4','physical_gpu':'4','egl_device_id':3}
        actual = {'vendor':'NVIDIA Corporation','renderer':'NVIDIA H20','version':'4.6','pci':'0001:04:00.0'}
        api.validate_active_renderer(expected, actual)
        for changed in ({'pci':'0001:05:00.0'}, {'vendor':'Mesa','renderer':'llvmpipe'}):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                api.validate_active_renderer(expected, dict(actual, **changed))

    def test_osmesa_does_not_require_gpu_inventory(self):
        api = self.api()
        self.assertEqual(api.discover_render_bindings(None, backend='osmesa'), [])
        options = api.render_worker_options([],0,'real-0',environ={'ACCERL_ENV_RENDER_BACKEND':'osmesa'})
        self.assertEqual(options['runtime_env']['env_vars']['MUJOCO_GL'], 'osmesa')
        self.assertNotIn('MUJOCO_EGL_DEVICE_ID',options['runtime_env']['env_vars'])

    def test_actual_startup_routes_renderers_with_policy_and_excludes_imagination(self):
        import test_main_startup_cpu as startup
        api = self.api()
        rows = api.select_render_bindings(*self.fixtures())
        context = startup.Context()
        namespace = context.namespace()
        namespace['os'].environ.update(ACCERL_ENV_RENDER_BACKEND='egl',
                                      ACCERL_EXPERIMENT_ROOT='/mnt/data/test')
        fn = startup.execute_definition(startup.startup_ast(), namespace)
        with mock.patch.object(api, 'discover_render_bindings', return_value=rows):
            fn(startup.arguments())
        for actor in context.actors:
            if actor.name not in ('RolloutWorkerActor','EvaluationWorkerActor'):
                continue
            child = actor.options.get('runtime_env',{}).get('env_vars',{})
            if actor.kwargs.get('worker_mode') == 'imagine':
                self.assertNotIn('ACCERL_RENDER_BINDING',child)
                continue
            self.assertEqual(child.get('MUJOCO_GL'),'egl')
            target = actor.positional[0].kwargs['actor_id']
            self.assertEqual(json.loads(child['ACCERL_RENDER_BINDING'])['uuid'], f'GPU-{target+4}')

    def test_render_worker_cannot_enable_tensor_cuda(self):
        api = self.api()
        with mock.patch.dict('os.environ', {'ACCERL_ENV_RENDER_BACKEND':'egl',
                 'ACCERL_RENDER_BINDING':'{}','CUDA_VISIBLE_DEVICES':'4'}, clear=True):
            with self.assertRaisesRegex(RuntimeError,'CUDA hidden'):
                api.prepare_environment_renderer()


if __name__ == '__main__':
    unittest.main()
