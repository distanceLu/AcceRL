"""Isolated WEAVER interpreter transport. This module never imports CUDA/Torch.

Only the privately spawned same-user child can access these stdin/stdout pipes.
Wire values are CPU numpy arrays and primitive containers; CUDA tensors and
their pickle reducers never cross the two independent Torch installations.
"""
from __future__ import annotations

import atexit
import os
from pathlib import Path
import pickle
import queue
import signal
import struct
import subprocess
import sys
import threading
import time

MAX_FRAME_BYTES = 512 * 1024 * 1024


def validate_resident_wire(handle, run_id, actor_id, owner):
    """Reject another child's capability before sending even its small frame."""
    from weaver_resident import ResidentStateError
    allowed={'backend','version','_resident','initial_reward_score','initial_success_probability',
             'initial_terminal','reward_semantics','reward_calibrated'}
    if not isinstance(handle,dict) or set(handle)-allowed or handle.get('backend')!='weaver':
        raise ResidentStateError('wire must contain only the issued capability and public metadata')
    token=handle.get('_resident')
    fields={'protocol','run_id','owner','actor_id','version','session','generation'}
    if (not isinstance(token,dict) or set(token)!=fields or token['protocol']!='weaver_resident_v1'
            or token['run_id']!=run_id or token['actor_id']!=actor_id
            or type(token['actor_id']) is not int or not isinstance(token['owner'],dict)
            or set(token['owner'])!={'pid','uid','birth'}
            or any(type(v) is not int for v in token['owner'].values())
            or token['owner']!={k:owner[k] for k in ('pid','uid','birth')}
            or type(token['version']) is not int or token['version']!=handle.get('version')
            or type(token['generation']) is not int or not 0<=token['generation']<=4096
            or not isinstance(token['session'],str) or len(token['session'])!=64):
        raise ResidentStateError('wire capability differs from the exact child/run/actor')
    return handle


def process_identity(pid):
    root = Path('/proc') / str(int(pid))
    try:
        raw = (root / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        status = (root / 'status').read_text().splitlines()
        uid = int(next(row for row in status if row.startswith('Uid:')).split()[1])
        return {'pid': int(pid), 'uid': uid, 'birth': int(fields[19]),
                'ppid': int(fields[1]), 'state': fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected):
    actual = process_identity(expected['pid'])
    return bool(actual and actual['state'] != 'Z' and all(
        actual[key] == expected[key] for key in ('pid', 'uid', 'birth')))


def cpu_wire(value):
    """Convert CPU Torch tensors without importing or initializing Torch."""
    if isinstance(value, dict):
        return {key: cpu_wire(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_wire(item) for item in value)
    if hasattr(value, 'detach'):
        if str(value.device) != 'cpu':
            raise ValueError('WEAVER bridge accepts only CPU tensors')
        tensor = value.detach()
        # numpy has no BF16 storage dtype. Its finite values are exactly
        # representable in FP32; retain those values across the version boundary.
        if str(tensor.dtype) == 'torch.bfloat16':
            tensor = tensor.float()
        return tensor.numpy().copy()
    return value


def _exact(stream, length):
    pieces = bytearray()
    while len(pieces) < length:
        part = stream.read(length - len(pieces))
        if not part:
            raise EOFError('WEAVER worker closed its private protocol pipe')
        pieces.extend(part)
    return bytes(pieces)


def send_frame(stream, value):
    payload = pickle.dumps(cpu_wire(value), protocol=5)
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError('WEAVER private RPC exceeded the bounded frame size')
    for data in (struct.pack('!Q', len(payload)), payload):
        remaining = memoryview(data)
        while remaining:
            written = stream.write(remaining)
            if written is None or written <= 0:
                raise BrokenPipeError('WEAVER private protocol stopped accepting bytes')
            remaining = remaining[written:]
    stream.flush()


def receive_frame(stream):
    length = struct.unpack('!Q', _exact(stream, 8))[0]
    if length > MAX_FRAME_BYTES:
        raise ValueError('WEAVER private RPC frame exceeds the admitted limit')
    return pickle.loads(_exact(stream, length))


class WeaverProcessClient:
    """One foreground child, one serialized request; no retries or fallback."""
    def __init__(self, config, actor_id, *, command=None, environment=None):
        self.config = config
        options = config['backend_options']
        self.timeout = float(options.get('rpc_timeout_seconds', 180))
        self.load_timeout = float(options.get('load_timeout_seconds', 900))
        if not 0 < self.timeout <= 900 or not 0 < self.load_timeout <= 1200:
            raise ValueError('WEAVER child RPC/load deadlines must be finite and bounded')
        self.parent = process_identity(os.getpid())
        if self.parent is None:
            raise RuntimeError('WEAVER isolated worker requires Linux process identities')
        self.run_id = os.environ.get('ACCERL_RUN_ID', '')
        if not self.run_id:
            raise RuntimeError('WEAVER isolated worker requires the explicit training run ID')
        self.actor_id=int(actor_id)
        self.marker = self.run_id + ':weaver:' + str(actor_id)
        env = dict(os.environ if environment is None else environment)
        # Prevent either Torch environment or a profiler preload leaking into
        # the other. Source/config/encoder paths are explicit, no downloads.
        for key in ('PYTHONHOME', 'LD_PRELOAD', 'CUDA_INJECTION64_PATH',
                    'NVTX_INJECTION64_PATH', 'NSYS_SESSION_ID'):
            env.pop(key, None)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parent)
        env['WEAVER_GPU_WORKER_ID'] = self.marker
        env['WEAVER_PARENT_PID'] = str(self.parent['pid'])
        env['WEAVER_PARENT_BIRTH'] = str(self.parent['birth'])
        env['WEAVER_PARENT_UID'] = str(self.parent['uid'])
        env['TOKENIZERS_PARALLELISM'] = 'false'
        python = options['inference_python']
        env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
        argv = command or [python, '-u', str(Path(__file__).with_name('weaver_gpu_worker.py')),
                           '--run-id', self.run_id, '--actor-id', str(actor_id)]
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=None, env=env, bufsize=0)
        self.identity = process_identity(self.process.pid)
        if self.identity is None or self.identity['uid'] != self.parent['uid']:
            # No load command has been sent, so the child owns no GPU allocation.
            # EOF is sufficient and avoids a PID-based signal without a birth.
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            self.process.stdout.close()
            raise RuntimeError('Unable to establish exact WEAVER child ownership')
        self.responses = queue.Queue(maxsize=2)
        self.lock = threading.Lock()
        self.closed = False
        self.reader = threading.Thread(target=self._reader, daemon=True)
        self.reader.start()
        atexit.register(self.close)
        try:
            self.receipt = self.call('load', config, timeout=self.load_timeout)
            if any(self.receipt['worker_identity'][key] != self.identity[key]
                   for key in ('pid', 'uid', 'birth')):
                raise RuntimeError('WEAVER load reply came from a different process')
        except BaseException:
            self.close()
            raise

    def _reader(self):
        try:
            while True:
                self.responses.put(receive_frame(self.process.stdout))
        except BaseException as error:
            self.responses.put(error)

    def call(self, method, *args, timeout=None):
        with self.lock:
            if self.closed or not same_process(self.identity):
                raise RuntimeError('WEAVER isolated worker is no longer the owned live process')
            if (method in ('request','release_rollout') and
                    getattr(self,'receipt',{}).get('resident_rollout')):
                # Validation errors are expected stale-session failures and do
                # not close the otherwise healthy, exactly owned child.
                validate_resident_wire(args[0],self.run_id,self.actor_id,self.identity)
            try:
                budget = self.timeout if timeout is None else timeout
                started = time.monotonic()
                payload = {'method': method, 'args': cpu_wire(args)}
                sent = queue.Queue(maxsize=1)
                def send():
                    try:
                        send_frame(self.process.stdin, payload)
                        sent.put(None)
                    except BaseException as error:
                        sent.put(error)
                sender = threading.Thread(target=send, daemon=True)
                sender.start()
                error = sent.get(timeout=budget)
                if error is not None:
                    raise error
                remaining = budget - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError('WEAVER private RPC timed out while sending ' + method)
                response = self.responses.get(timeout=remaining)
                if isinstance(response, BaseException):
                    raise response
                if not response.get('ok'):
                    raise RuntimeError('WEAVER isolated GPU failure: ' + response.get('traceback', 'unknown'))
                return response['result']
            except queue.Empty as error:
                self.close()
                raise TimeoutError('WEAVER private RPC deadline exceeded for ' + method) from error
            except BaseException:
                self.close()
                raise

    def close(self):
        if getattr(self, 'closed', True):
            return
        self.closed = True
        process = self.process
        if same_process(self.identity):
            try:
                process.stdin.close()
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                if same_process(self.identity):
                    os.kill(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if same_process(self.identity):
                        os.kill(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except OSError:
                pass


class WeaverRemoteOperations:
    def __init__(self, client):
        self.client = client
        self.published_version = 0
    def get_context_geometry(self):
        return self.client.receipt['geometry']
    def set_published_version(self, version):
        if version != 0:
            raise RuntimeError('Frozen WEAVER cannot publish another model version')
    def init_rollout(self, context, published_version=None):
        return self.client.call('init_rollout', context, published_version)
    def release_rollout(self, state):
        return self.client.call('release_rollout', state)
    def request(self, state, actions, instruction=None):
        result = self.client.call('request', state, actions, instruction)
        # The existing Policy/Reward image adapters require local CPU Torch
        # tensors. Rebuild only their public decoded outputs, never CUDA storage
        # or the opaque latent archive from the other Torch installation.
        import torch
        import numpy as np
        for key in ('observations', 'predicted_states', 'valid_mask', 'terminal_predictions',
                    'success_probabilities', 'reward_predictions'):
            if isinstance(result.get(key), np.ndarray):
                result[key] = torch.from_numpy(result[key])
        return result
