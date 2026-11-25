"""Single-host, benchmark-only timing and action admission for four colocated ranks."""
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import time
import weakref


@lru_cache(maxsize=4)
def _plan_enabled(path):
    return json.loads(Path(path).read_bytes()).get('comparison') == 'critical_jit'


def benchmark_output(env):
    plan = env.get('ACCERL_JIT_BENCH_PLAN')
    if plan and _plan_enabled(plan):
        return Path(env['ACCERL_JIT_BENCH_OUTPUT'])
    return None


def _directory(output):
    folder = Path(output) / 'critical_probe'
    folder.mkdir(exist_ok=True, parents=True)
    return folder


def _gate_path(output, rank):
    # Only the tiny live admission flag is ephemeral. Keep all observations in
    # the run output. All actors are required to be colocated on one host.
    shm = Path('/dev/shm')
    if shm.is_dir():
        key = hashlib.sha256(os.fsencode(Path(output).resolve())).hexdigest()[:32]
        folder = shm / f'accerl-critical-{os.getuid()}-{key}'
        folder.mkdir(exist_ok=True, mode=0o700)
    else:
        folder = _directory(output)
    return folder / f'gate-{rank}.json'


def _replace(path, data):
    # Exactly one writer per destination. Publishing a flag needs atomic
    # visibility on this host, not a storage flush inside the training micro.
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(json.dumps(data, allow_nan=False, separators=(',', ':')).encode())
    os.replace(tmp, path)


def configure_probe(output, *, step, arm_id, condition, target_rank=None,
                    target_bundle=None, gate_rank=None, ttl_s=5.):
    if (type(step) is not int or step <= 0 or not arm_id or
            condition not in ('calibration', 'N', 'P-target', 'P-other') or
            not math.isfinite(ttl_s) or not 0 < ttl_s <= 5):
        raise ValueError('Invalid critical JIT probe plan')
    if gate_rank is None:
        gate_rank = target_rank
    if condition == 'calibration':
        if any(i is not None for i in (target_rank, target_bundle, gate_rank)):
            raise ValueError('Calibration must observe all micros without a target')
    elif (target_rank not in range(4) or gate_rank not in range(4) or
          target_bundle not in range(9) or
          (condition == 'P-other') != (gate_rank != target_rank)):
        raise ValueError('Invalid critical JIT gate rank or bundle')
    spec = dict(step=step, arm_id=arm_id, condition=condition, target_rank=target_rank,
                target_bundle=target_bundle, gate_rank=gate_rank, ttl_s=ttl_s)
    _replace(_directory(output) / 'arm.json', spec)
    return spec


class GateWindow:
    def __init__(self, output, spec):
        self.spec = spec
        self.path = _gate_path(output, spec['gate_rank'])

    def open(self):
        ttl_ns = round(self.spec['ttl_s'] * 1e9)
        issued = time.time_ns()
        self.record = dict(arm_id=self.spec['arm_id'], gate_rank=self.spec['gate_rank'],
                           gate_effective=self.spec['condition'] != 'N', gate_issued_ns=issued,
                           gate_until_ns=issued + ttl_ns,
                           until_monotonic_ns=time.monotonic_ns() + ttl_ns, gate_closed_ns=None)
        _replace(self.path, self.record)
        self.record['gate_visible_by_ns'] = time.time_ns()
        return dict(self.record)

    def close(self):
        self.record['gate_closed_ns'] = time.time_ns()
        _replace(self.path, self.record)
        return self.record['gate_closed_ns']


class TargetActionGate:
    def __init__(self, output, worker_id):
        if type(worker_id) is not int or worker_id not in range(8):
            raise ValueError('Critical JIT expects eight round-robin real workers')
        self.path = _gate_path(output, worker_id % 4)
        self.actions = ActionTrace(output, worker_id % 4, worker_id=worker_id)

    def _read(self):
        try:
            return json.loads(self.path.read_bytes())
        except FileNotFoundError:
            return None

    def wait(self, ready_ns):
        pulse = self._read()
        start = time.monotonic_ns()
        if (pulse is None or pulse['gate_closed_ns'] is not None or
                ready_ns < pulse['gate_issued_ns'] or start >= pulse['until_monotonic_ns']):
            return None, 0
        if not pulse['gate_effective']:
            return pulse['arm_id'], 0
        deadline = pulse['until_monotonic_ns']
        while (remaining := deadline - time.monotonic_ns()) > 0:
            time.sleep(min(.001, remaining / 1e9))
            current = self._read()
            if current is None or current['arm_id'] != pulse['arm_id'] or current['gate_closed_ns'] is not None:
                break
        return pulse['arm_id'], time.monotonic_ns() - start


class ActionTrace:
    def __init__(self, output, rank, *, worker_id=None, gpu_uuid=None, physical_gpu=None):
        self.rank, self.worker_id = rank, worker_id
        self.identity = dict(rank=rank, gpu_uuid=gpu_uuid,
                             physical_gpu=None if physical_gpu is None else int(physical_gpu))
        label = f'worker-{worker_id}' if worker_id is not None else f'policy-{rank}'
        self.path = _directory(output) / f'actions-{label}.jsonl'
        self.fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND | getattr(os, 'O_BINARY', 0), 0o600)
        self.close = weakref.finalize(self, os.close, self.fd)
        self.batch = []
        self.batch_size = 0

    def _append(self, rows):
        if rows:
            payload = b''.join(json.dumps(r, separators=(',', ':')).encode() + b'\n' for r in rows)
            if os.write(self.fd, payload) != len(payload):
                raise OSError('Incomplete critical JIT action observation write')

    def worker(self, kind, request_id, ts_ns, **details):
        self._append([dict(self.identity, kind=kind, request_id=request_id, worker_id=self.worker_id,
                           ts_ns=ts_ns, **details)])

    def submitted(self, request_id, ready, dispatch, returned, held, arm):
        identity = dict(self.identity, request_id=request_id, worker_id=self.worker_id)
        self._append([dict(identity, kind='ready', ts_ns=ready),
                      dict(identity, kind='dispatch', ts_ns=dispatch, remote_returned_ns=returned,
                           held_ns=held, gate_arm_id=arm)])

    def _actor(self, kind, items, ts_ns, **details):
        rows = []
        for item in items:
            if item.get('workflow') != 'real_rollout':
                continue
            worker_id = int(item['caller_id'])
            if worker_id not in range(8) or worker_id % 4 != self.rank:
                raise RuntimeError('Actual real action receiver differs from benchmark card routing')
            rows.append(dict(self.identity, kind=kind, request_id=item['request_id'],
                             worker_id=worker_id, ts_ns=ts_ns, **details))
        self._append(rows)

    def enqueue(self, item):
        self._actor('enqueue', [item], item['arrival_ns'])

    def begin(self, items, ts_ns):
        self.batch = [item for item in items if item.get('workflow') == 'real_rollout']
        self.batch_size = len(items)
        self.phase('service_start', ts_ns)

    def phase(self, kind, ts_ns):
        if self.batch:
            self._actor(kind, self.batch, ts_ns, batch_id=self.batch[0]['request_id'],
                        batch_size=self.batch_size)
        if kind == 'service_end':
            self.batch = []


def read_actions(output):
    rows = []
    for file in (Path(output) / 'critical_probe').glob('actions-*.jsonl'):
        # Writers append whole lines. A live reader may see a final partial line.
        with file.open('rb') as src:
            rows.extend(json.loads(line) for line in src if line.endswith(b'\n'))
    return sorted(rows, key=lambda r: r['ts_ns'])


def align_marker(relative, anchors, issue_ns, completed_by_ns):
    """Intersect host brackets from mandatory readbacks and direct causal bounds.

    Relative CUDA event times use ns. Allow 200ppm for device/host clock rate
    skew and 2us for event precision. Inconsistent clock estimates fall back to
    the direct host interval from enqueue to the next mandatory completion.
    """
    lower, upper = issue_ns, completed_by_ns
    for offset, earliest, latest in anchors:
        delta = relative - offset
        error = 2_000 + math.ceil(abs(delta) * .0002)
        lower = max(lower, earliest + delta - error)
        upper = min(upper, latest + delta + error)
    if lower > upper:
        lower, upper = issue_ns, completed_by_ns
    return lower, (lower + upper) // 2, upper


class TrainerProbe:
    def __init__(self, output, spec, owner):
        self.output, self.spec, self.rank = output, spec, owner.rank
        self.identity = dict(gpu_uuid=owner.gpu_binding['uuid'], physical_gpu=int(owner.gpu_binding['physical_gpu']))
        self.bundles, self.events, self.votes = [], [], []
        self.window = (GateWindow(output, spec) if self.rank == spec['target_rank'] else None)

    @classmethod
    def for_update(cls, owner, step):
        output = benchmark_output(os.environ)
        if output is None:
            return None
        spec = json.loads((_directory(output) / 'arm.json').read_bytes())
        if spec['step'] != step:
            raise RuntimeError('Critical JIT trainer received a stale step configuration')
        return cls(output, spec, owner)

    def begin(self, number):
        row = dict(bundle=number, forward_start_ns=time.time_ns())
        self.bundles.append(row)
        if self.window and number == self.spec['target_bundle']:
            row.update(self.window.open())

    def mark(self, name):
        import torch
        event = torch.cuda.Event(enable_timing=True)
        issued = time.time_ns()
        event.record()
        index = len(self.bundles) - 1
        self.events.append((index, name, event, issued))
        return event, issued

    def vote_complete(self, marker):
        self.votes.append((*marker, time.time_ns()))
        if self.window and self.bundles[-1]['bundle'] == self.spec['target_bundle']:
            self.bundles[-1]['gate_closed_ns'] = self.window.close()

    def finish(self, anchor, completed_ns):
        # The caller just consumed the existing packed result on the CPU, so
        # every marker on this CUDA stream is complete. No new GPU wait here.
        if len(self.bundles) != 9 or len(self.votes) != 9:
            raise RuntimeError('Critical JIT window must contain nine voted bundles')
        last, issued = anchor
        def relative(event):
            return -round(event.elapsed_time(last) * 1e6)
        anchors = [(relative(ev), lo, hi) for ev, lo, hi in self.votes] + [(0, issued, completed_ns)]
        for index, name, ev, lo in self.events:
            if name == 'packed':
                continue
            hi = (self.votes[index][2] if name in ('forward_ready', 'vote_done') else
                  self.votes[index + 1][2] if index + 1 < len(self.votes) else completed_ns)
            lower, stamp, upper = align_marker(relative(ev), anchors, lo, hi)
            row = self.bundles[index]
            row[name + '_ns'] = stamp
            row[name + '_lo_ns'] = lower
            row[name + '_hi_ns'] = upper
            row['clock_error_ns'] = max(row.get('clock_error_ns', 0), stamp-lower, upper-stamp)
        result = dict(self.spec, **self.identity, rank=self.rank, bundles=self.bundles,
                      clock='single-host epoch ns, CUDA events bracketed by required readbacks')
        _replace(_directory(self.output) / f"step-{self.spec['step']}-rank-{self.rank}.json", result)


def read_probe(output, step):
    return [json.loads((_directory(output) / f'step-{step}-rank-{rank}.json').read_bytes())
            for rank in range(4)]
