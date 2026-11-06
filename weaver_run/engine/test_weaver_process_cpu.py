"""Real Linux private-pipe worker lifetime tests; no model/CUDA is loaded."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from weaver_subprocess import WeaverProcessClient, same_process

ROOT = Path(__file__).resolve().parent
STUB = '''
import sys, time
from weaver_subprocess import process_identity, receive_frame, send_frame
from weaver_gpu_worker import arm_parent_guard
arm_parent_guard()
while True:
    try:
        msg=receive_frame(sys.stdin.buffer)
    except EOFError:
        break
    method=msg['method']
    if method=='load':
        result={'worker_identity':process_identity(__import__('os').getpid()),'geometry':{'horizon':8}}
    elif method=='echo':
        result=msg['args'][0]
    elif method=='sleep':
        time.sleep(3)
        result=True
    elif method=='fail':
        send_frame(sys.stdout.buffer,{'ok':False,'traceback':'synthetic CPU worker failure'})
        break
    else:
        raise RuntimeError(method)
    send_frame(sys.stdout.buffer,{'ok':True,'result':result})
'''


@unittest.skipUnless(os.name == 'posix' and Path('/proc/self/stat').exists(), 'Requires actual Linux process birth identities')
class ProcessLifetime(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {'ACCERL_RUN_ID': 'cpu_weaver_pipe_only'})
        self.environment.start()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.worker = self.root / 'cpu_worker.py'
        self.worker.write_text(STUB)
        self.config = {'backend_options': {'inference_python': sys.executable,
            'rpc_timeout_seconds': .3, 'load_timeout_seconds': 10}}
    def tearDown(self):
        self.directory.cleanup()
        self.environment.stop()
    def client(self):
        return WeaverProcessClient(self.config, 0, command=[sys.executable, '-u', str(self.worker)])
    def test_actual_spawn_echo_and_eof_release(self):
        import numpy as np
        client = self.client()
        identity = dict(client.identity)
        try:
            expected = np.arange(56, dtype=np.float32).reshape(8, 7)
            result = client.call('echo', {'actions': expected})
            np.testing.assert_array_equal(result['actions'], expected)
            self.assertTrue(same_process(identity))
        finally:
            client.close()
        self.assertFalse(same_process(identity))
    def test_rpc_timeout_cleans_exact_owned_process(self):
        client = self.client()
        identity = dict(client.identity)
        with self.assertRaisesRegex(TimeoutError, 'deadline exceeded'):
            client.call('sleep')
        self.assertFalse(same_process(identity))
    def test_worker_failure_propagates_without_retry(self):
        client = self.client()
        identity = dict(client.identity)
        with self.assertRaisesRegex(RuntimeError, 'synthetic CPU worker failure'):
            client.call('fail')
        self.assertFalse(same_process(identity))
    def test_parent_sigkill_does_not_orphan_worker(self):
        result_path = self.root / 'owned_child.json'
        launcher = self.root / 'launcher.py'
        launcher.write_text('import sys,time,json\n'
            + 'sys.path.insert(0,' + repr(str(ROOT)) + ')\n'
            + 'from pathlib import Path\nfrom weaver_subprocess import WeaverProcessClient\n'
            + 'client=WeaverProcessClient(' + repr(self.config) + ',0,command='
            + repr([sys.executable, '-u', str(self.worker)]) + ')\n'
            + 'Path(' + repr(str(result_path)) + ').write_text(json.dumps(client.identity))\n'
            + 'time.sleep(60)\n')
        process = subprocess.Popen([sys.executable, '-u', str(launcher)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child = None
        try:
            deadline = time.monotonic() + 10
            while not result_path.is_file() and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue(result_path.is_file(), 'CPU child did not complete its load handshake')
            child = json.loads(result_path.read_text())
            process.kill()
            process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while same_process(child) and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertFalse(same_process(child), 'PDEATHSIG failed to release the exact child')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if child and same_process(child):
                os.kill(child['pid'], 9)


if __name__ == '__main__':
    unittest.main()
