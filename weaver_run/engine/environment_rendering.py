"""Explicit EGL placement for CPU simulator workers; no CUDA imports.

EGL enumeration is independent of CUDA_VISIBLE_DEVICES and MPS ordinals. Match
the driver UUID through PCI/DRM, then verify the active display after creation.
"""
from __future__ import annotations

import ctypes as C
import ctypes.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time


def render_backend(environ=None):
    env = os.environ if environ is None else environ
    backend = env.get('ACCERL_ENV_RENDER_BACKEND', env.get('MUJOCO_GL', 'osmesa'))
    if backend not in ('egl', 'osmesa'):
        raise ValueError('Environment rendering must explicitly select egl or osmesa')
    return backend


def configure_render_backend(environ=None):
    """Call before importing OpenGL, robosuite, LIBERO or their consumers."""
    env = os.environ if environ is None else environ
    backend = render_backend(env)
    env['MUJOCO_GL'] = env['PYOPENGL_PLATFORM'] = backend
    if backend != 'egl' or not env.get('ACCERL_RENDER_BINDING'):
        # A numeric EGL selector is invalid in model processes with UUID CUDA masks.
        env.pop('MUJOCO_EGL_DEVICE_ID', None)
    return backend


def normalize_pci(value):
    match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value.strip())
    if match is None:
        raise ValueError('Invalid PCI identity: '+repr(value))
    domain, bus, slot, function = (int(part, 16) for part in match.groups())
    return f'{domain:04x}:{bus:02x}:{slot:02x}.{function}'


class _EGL:
    """Device enumeration uses the driver API without creating a GL context."""
    def __init__(self):
        self.lib = C.CDLL(ctypes.util.find_library('EGL') or 'libEGL.so.1')
        self.lib.eglGetProcAddress.argtypes = [C.c_char_p]
        self.lib.eglGetProcAddress.restype = C.c_void_p
        self.string = self.proc('eglQueryDeviceStringEXT', C.c_char_p, C.c_void_p, C.c_int)

    def proc(self, name, result, *args):
        address = self.lib.eglGetProcAddress(name.encode('ascii'))
        if not address:
            raise RuntimeError('Required EGL entry point is unavailable: '+name)
        return C.CFUNCTYPE(result, *args)(address)

    def device(self, handle):
        def string(token):
            value = self.string(handle, token)
            return value.decode('utf-8') if value else None
        extensions = (string(0x3055) or '').split()
        node = string(0x3377) if 'EGL_EXT_device_drm_render_node' in extensions else None
        node = node or (string(0x3233) if 'EGL_EXT_device_drm' in extensions else None)
        pci = None
        if node:
            path = Path('/sys/class/drm') / Path(node).name / 'device'
            if path.exists():
                pci = normalize_pci(path.resolve(strict=True).name)
        return {'render_node': node, 'pci': pci}

    def devices(self):
        query = self.proc('eglQueryDevicesEXT', C.c_uint, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int))
        handles, count = (C.c_void_p * 256)(), C.c_int()
        if not query(len(handles), handles, C.byref(count)) or not 0 < count.value < len(handles):
            raise RuntimeError('Cannot enumerate EGL devices completely')
        return [dict(self.device(handles[i]), egl_device_id=i) for i in range(count.value)]

    def active_device(self, display):
        query = self.proc('eglQueryDisplayAttribEXT', C.c_uint, C.c_void_p, C.c_int, C.POINTER(C.c_ssize_t))
        handle = C.c_ssize_t()
        if not query(C.cast(display, C.c_void_p), 0x322C, C.byref(handle)) or not handle.value:
            raise RuntimeError('Cannot identify the active EGL display device')
        return self.device(C.c_void_p(handle.value))


def select_render_bindings(bindings, gpu_inventory, egl_devices):
    """Pure identity join; enumeration order and CUDA ordinals are irrelevant."""
    result = []
    for binding in bindings:
        matching_gpu = [g for g in gpu_inventory if g['index'] == binding['index'] and g['uuid'] == binding['uuid']]
        if len(matching_gpu) != 1:
            raise ValueError('Allocated GPU index/UUID no longer matches the driver inventory')
        pci = normalize_pci(matching_gpu[0]['pci'])
        matches = [d for d in egl_devices if d.get('pci') and normalize_pci(d['pci']) == pci]
        if len(matches) != 1:
            raise ValueError('Allocated GPU must match exactly one EGL device: '+binding['uuid'])
        result.append({'physical_gpu': binding['index'], 'uuid': binding['uuid'], 'pci': pci,
                       'egl_device_id': matches[0]['egl_device_id']})
    if len({r['uuid'] for r in result}) != len(result) or len({r['egl_device_id'] for r in result}) != len(result):
        raise ValueError('Render GPU bindings must be distinct')
    return result


def discover_render_bindings(bindings, *, backend=None):
    backend = render_backend() if backend is None else backend
    if backend == 'osmesa':
        return []
    if backend != 'egl' or not bindings:
        raise ValueError('EGL requires explicit allocated physical GPU bindings')
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    inventory = [dict(zip(('index','uuid','pci'), [part.strip() for part in row.split(',')]))
                 for row in output.splitlines() if row.strip()]
    return select_render_bindings(bindings, inventory, _EGL().devices())


def render_worker_options(render_bindings, replica, worker_label, *, environ=None):
    env = os.environ if environ is None else environ
    backend = render_backend(env)
    if not re.fullmatch(r'(real|eval)-[0-9]+', worker_label):
        raise ValueError('Invalid rendering worker identity')
    child = {'ACCERL_ENV_RENDER_BACKEND': backend, 'MUJOCO_GL': backend,
             'PYOPENGL_PLATFORM': backend, 'CUDA_VISIBLE_DEVICES': '',
             'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1',
             'ACCERL_RENDER_WORKER': worker_label}
    if backend == 'egl':
        if type(replica) is not int or not 0 <= replica < len(render_bindings):
            raise ValueError('Renderer must follow an allocated Policy inference replica')
        root = env.get('ACCERL_EXPERIMENT_ROOT', '')
        if not root.startswith('/') or root == '/home' or root.startswith('/home/'):
            raise ValueError('EGL requires an absolute experiment directory on the data disk')
        binding = render_bindings[replica]
        child.update(ACCERL_RENDER_BINDING=json.dumps(binding, sort_keys=True),
                     MUJOCO_EGL_DEVICE_ID=str(binding['egl_device_id']),
                     __GL_SHADER_DISK_CACHE='1',
                     __GL_SHADER_DISK_CACHE_PATH=str(PurePosixPath(root)/'rendering'/'cache'/worker_label))
    return {'num_gpus': 0, 'max_restarts': 0, 'max_task_retries': 0,
            'runtime_env': {'env_vars': child}}


def prepare_environment_renderer():
    backend = configure_render_backend()
    if backend == 'egl':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('Environment workers must keep CUDA hidden while using EGL')
        binding = json.loads(os.environ.get('ACCERL_RENDER_BINDING', '{}'))
        if not binding or os.environ.get('MUJOCO_EGL_DEVICE_ID') != str(binding['egl_device_id']):
            raise RuntimeError('EGL environment has no validated per-worker device selector')
        cache = Path(os.environ['__GL_SHADER_DISK_CACHE_PATH'])
        root = Path(os.environ['ACCERL_EXPERIMENT_ROOT']).resolve()
        if not cache.resolve().is_relative_to(root):
            raise RuntimeError('Shader cache is outside the experiment directory')
        cache.mkdir(parents=True, exist_ok=True)
    return backend


def validate_active_renderer(binding, actual):
    if 'nvidia' not in actual.get('vendor', '').lower() or any(
            word in actual.get('renderer', '').lower() for word in ('llvmpipe', 'softpipe', 'software')):
        raise RuntimeError('EGL did not create an NVIDIA hardware renderer: '+repr(actual))
    if not actual.get('pci') or normalize_pci(actual['pci']) != binding['pci']:
        raise RuntimeError('Active EGL renderer is on the wrong physical GPU: '+repr(actual))


def verify_environment_renderer():
    """Call with the newly constructed environment's rendering context current."""
    from OpenGL import GL
    actual = {}
    for key, enum in (('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER),('version',GL.GL_VERSION)):
        value = GL.glGetString(enum)
        if not value:
            raise RuntimeError('Simulator did not leave a current rendering context')
        actual[key] = value.decode('utf-8')
    backend = render_backend()
    binding = None
    if backend == 'egl':
        from OpenGL import EGL
        binding = json.loads(os.environ['ACCERL_RENDER_BINDING'])
        actual.update(_EGL().active_device(EGL.eglGetCurrentDisplay()))
        validate_active_renderer(binding, actual)
    receipt = dict(backend=backend, physics='cpu', pid=os.getpid(), time_ns=time.time_ns(),
                   worker=os.environ.get('ACCERL_RENDER_WORKER'), requested=binding, actual=actual,
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    root = os.environ.get('ACCERL_EXPERIMENT_ROOT')
    if root and receipt['worker']:
        directory = Path(root)/'rendering'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory/(receipt['worker']+'.json')
        temporary = target.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
        temporary.replace(target)
    print('[Environment renderer] '+json.dumps(receipt, sort_keys=True), flush=True)
    return receipt
