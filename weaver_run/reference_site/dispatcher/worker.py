"""Single cooperative queue worker. No shell, preemption, or automatic job retry."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

try:
    from .store import QueueStore, QueueError, DEFAULT_CWD_ROOT
except ImportError:
    from store import QueueStore, QueueError, DEFAULT_CWD_ROOT


GPU_UUIDS = (
    'GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c',
    'GPU-fe0edfe4-b193-1af7-6a98-cbff40214f74',
    'GPU-8c9bc264-7f09-989e-1ec7-cecd6cde782c',
    'GPU-07ae0bbc-8923-bd0c-199e-c726821e655d',
)
MPS_PIPE = '/dev/shm/mps_lcx1_gpu47_pzero2_20261020/pipe'
IDENTITY_KEYS = ('pid', 'uid', 'birth_ticks')


def process_identity(pid):
    try:
        path = Path('/proc') / str(int(pid))
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        uid = int(next(x.split()[1] for x in (path / 'status').read_text().splitlines() if x.startswith('Uid:')))
        return {'pid': int(pid), 'uid': uid, 'birth_ticks': int(fields[19]), 'pgid': int(fields[2]),
                'ppid': int(fields[1]), 'state': fields[0]}
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def same_process(expected, actual):
    return bool(expected and actual and actual.get('state') not in ('Z', 'X')
                and all(expected.get(k) == actual.get(k) for k in IDENTITY_KEYS))


def marked_processes(job_id, uid, known=()):
    """Discover only exact marker processes or verified descendants of owned births."""
    identities = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            item = process_identity(int(entry.name))
            if item and item['uid'] == uid and item['state'] not in ('Z', 'X'):
                identities[item['pid']] = item
    owned = {x['pid']: identities[x['pid']] for x in known
             if same_process(x, identities.get(x['pid']))}
    marker = ('EXPERIMENT_QUEUE_JOB_ID=' + job_id).encode()
    for pid, item in identities.items():
        try:
            if marker in (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0'):
                owned[pid] = item
        except (OSError, PermissionError):
            pass  # An unreadable unrelated UID peer is never signalled.
    changed = True
    while changed:
        changed = False
        for pid, item in identities.items():
            if pid not in owned and item['ppid'] in owned:
                owned[pid] = item
                changed = True
    return list(owned.values())


def signal_owned(owned, sig):
    sent = []
    # Individual identities, not killpg, prevent affecting an unowned process.
    for item in reversed(owned):
        if same_process(item, process_identity(item['pid'])):
            try:
                if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
                    descriptor = os.pidfd_open(item['pid'])
                    try:
                        if not same_process(item, process_identity(item['pid'])):
                            continue
                        signal.pidfd_send_signal(descriptor, sig)
                    finally:
                        os.close(descriptor)
                else:
                    os.kill(item['pid'], sig)
                sent.append(item)
            except ProcessLookupError:
                pass
    return sent


class QueueWorker:
    def __init__(self, root, gate_callback, environment, snapshot_callback=None, *,
                 store=None, clock_ns=time.time_ns, termination_grace_seconds=10,
                 recovery_gate_callback=None, recovery_quiet_seconds=5):
        self.root = Path(root)
        self.store = store or QueueStore(root)
        self.gate_callback = gate_callback
        self.environment = dict(environment)
        self.snapshot_callback = snapshot_callback
        self.clock_ns = clock_ns
        self.termination_grace_ns = int(termination_grace_seconds * 1e9)
        self.children = {}
        self.terminating = {}
        self.recovered = set()
        self.lock = None
        self.monotonic_deadlines = {}
        self.recovery_gate_callback = recovery_gate_callback or gate_callback
        self.recovery_quiet_seconds = max(0, float(recovery_quiet_seconds))
        # Deliberately not restored across daemon restarts: two fresh ownership
        # and resource observations are required before releasing a claim.
        self.recovery_quiet = {}

    def close(self):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def acquire_lock(self):
        if self.lock is not None:
            return True
        if sys.platform != 'linux':
            raise QueueError('Production worker requires Linux')
        import fcntl
        handle = (self.root / 'worker.lock').open('a')
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        self.lock = handle
        return True

    def emit(self, **value):
        value['at_ns'] = self.clock_ns()
        if self.snapshot_callback:
            self.snapshot_callback(value)
        return value

    def environment_for(self, job):
        env = dict(self.environment)
        for key in list(env):
            if key.startswith(('NSYS_', 'CUDA_INJECTION', 'NVTX_INJECTION')) or key == 'LD_PRELOAD':
                env.pop(key, None)
        for key in ('WEAVER_OFFLINE_RUN_ID', 'WEAVER_EVAL_RUN_ID', 'WEAVER_EXPECTED_GPU_UUIDS',
                    'ACCERL_RUN_ID', 'ACCE_RUN_ID'):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES=','.join(GPU_UUIDS), CUDA_MPS_PIPE_DIRECTORY=MPS_PIPE,
                   CUDA_MPS_LOG_DIRECTORY=str(Path(MPS_PIPE).parent / 'log'),
                   EXPERIMENT_QUEUE_JOB_ID=job['id'], PYTHONUNBUFFERED='1')
        env['EXPERIMENT_QUEUE_MANUAL'] = '1' if job.get('manual') is True else '0'
        env['EXPERIMENT_QUEUE_TIMEOUT_SECONDS'] = str(job['timeout_seconds'])
        for key in ('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT'):
            env.pop(key, None)
        return env

    def validate_launch(self, job):
        if sys.platform != 'linux':
            raise QueueError('Production worker requires Linux birth identity checks')
        if os.getuid() != 1004 and self.store.allowed_cwd_root == DEFAULT_CWD_ROOT:
            raise QueueError('Production GPU4-7 worker is restricted to UID1004')
        self.store.validate_submission(job['name'], job['argv'], job['cwd'], job['timeout_seconds'], job.get('manual', False))
        cwd = Path(job['cwd']).resolve(strict=True)
        try:
            cwd.relative_to(Path(self.store.allowed_cwd_root).resolve(strict=True))
        except ValueError:
            raise QueueError('cwd redirects outside the allowed root') from None
        if not cwd.is_dir() or not Path(job['argv'][0]).is_file():
            raise QueueError('cwd or executable does not exist')

    def start(self, job):
        child = None
        try:
            self.validate_launch(job)
            folder = self.root / 'jobs' / job['id']
            folder.mkdir(parents=True, exist_ok=False)
            logpath = folder / 'output.log'
            deadline_ns = None if job.get('manual') is True else self.clock_ns() + job['timeout_seconds'] * 10**9
            with logpath.open('xb') as logfile:
                child = subprocess.Popen(job['argv'], cwd=job['cwd'], env=self.environment_for(job),
                    stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT,
                    start_new_session=True, shell=False)
            self.children[job['id']] = child
            if job.get('manual') is not True:
                self.monotonic_deadlines[job['id']] = time.monotonic() + job['timeout_seconds']
            identity = process_identity(child.pid)
            if not identity or identity['uid'] != os.getuid():
                raise QueueError('Spawned process birth identity unavailable')
            updated = self.store.record_started(job['id'], identity, logpath, deadline_ns)
            return self.emit(state='running', job=updated, reason='launched_once')
        except BaseException as exc:
            # If spawn might have happened, preserve reservation until identity/cleanup can be proved.
            state = 'blocked' if child is not None else 'failed'
            updated = self.store.finish(job['id'], state, 'launch_error: ' + repr(exc))
            return self.emit(state=state, job=updated, reason=updated['reason'])

    def recover_quiet_claim(self, job, owned, reason, exitcode=None):
        """Uncertain jobs fail, never succeed/retry, after two quiet proofs."""
        if job['state'] != 'blocked':
            job = self.store.finish(job['id'], 'blocked', reason, exitcode)
        reason = job.get('reason') or reason
        if owned:
            self.recovery_quiet.pop(job['id'], None)
            return self.emit(state='blocked', job=self.store.get(job['id']), reason=reason,
                             recovery={'owned_live': owned, 'quiet_observations': 0})
        try:
            gate = self.recovery_gate_callback()
        except Exception as exc:
            gate = {'ready': False, 'reason': 'recovery_resource_gate_error: ' + repr(exc)}
        if not isinstance(gate, dict) or gate.get('ready') is not True:
            self.recovery_quiet.pop(job['id'], None)
            return self.emit(state='blocked', job=self.store.get(job['id']), reason=reason,
                             recovery={'owned_live': [], 'quiet_observations': 0, 'gate': gate})
        now_mono = time.monotonic()
        first = self.recovery_quiet.get(job['id'])
        if first is None:
            first = {'monotonic': now_mono, 'at_ns': self.clock_ns()}
            self.recovery_quiet[job['id']] = first
            self.store.record_recovery_observation(job['id'],
                {'observation': 1, 'owned_live': [], 'resource_gate': gate,
                 'reason': reason, 'automatic_retry': False})
        elif now_mono - first['monotonic'] >= self.recovery_quiet_seconds:
            self.store.record_recovery_observation(job['id'],
                {'observation': 2, 'owned_live': [], 'resource_gate': gate,
                 'first_observed_ns': first['at_ns'], 'reason': reason,
                 'automatic_retry': False, 'exit_status_known': exitcode is not None})
            final_reason = reason + '; cleanup_confirmed_two_quiet_observations'
            updated = self.store.finish(job['id'], 'failed', final_reason, exitcode)
            self.recovery_quiet.pop(job['id'], None)
            self.children.pop(job['id'], None)
            self.terminating.pop(job['id'], None)
            self.monotonic_deadlines.pop(job['id'], None)
            return self.emit(state='failed', job=updated, reason=final_reason)
        return self.emit(state='blocked', job=self.store.get(job['id']), reason=reason,
                         recovery={'owned_live': [], 'quiet_observations': 1, 'gate': gate,
                                   'required_quiet_seconds': self.recovery_quiet_seconds})

    def observe(self, job):
        child = self.children.get(job['id'])
        uid = job['identity']['uid'] if job['identity'] else os.getuid()
        known = job['owned'] + ([job['identity']] if job['identity'] else [])
        owned = marked_processes(job['id'], uid, known)
        self.store.observe(job['id'], owned)
        live_parent = bool(job['identity'] and same_process(
            job['identity'], process_identity(job['identity']['pid'])))
        exitcode = child.poll() if child is not None else None
        uncertain_reason = job.get('reason') if job['state'] == 'blocked' else None
        if not job['identity']:
            uncertain_reason = uncertain_reason or 'durable_claim_without_launch_identity_no_retry'
        # Recovery may monitor the exact process, but cannot recover its exit status.
        if child is None:
            self.recovered.add(job['id'])
            if job['identity'] and not live_parent:
                uncertain_reason = uncertain_reason or 'recovered_process_exited_unknown_exit_status_no_retry'
        if job['id'] in self.terminating:
            stop = self.terminating[job['id']]
            if owned and self.clock_ns() >= stop['kill_at_ns']:
                signal_owned(owned, signal.SIGKILL)
            owned = marked_processes(job['id'], uid, self.store.get(job['id'])['owned'])
            self.store.observe(job['id'], owned)
            if child is not None:
                exitcode = child.poll()
            if not owned:
                if uncertain_reason or child is None or job['state'] == 'blocked':
                    return self.recover_quiet_claim(job, owned, uncertain_reason or stop['reason'], exitcode)
                updated = self.store.finish(job['id'], 'failed', stop['reason'], exitcode)
                self.children.pop(job['id'], None)
                self.terminating.pop(job['id'], None)
                return self.emit(state='failed', job=updated, reason=stop['reason'])
            if self.clock_ns() >= stop['kill_at_ns'] + 15 * 10**9:
                reason = uncertain_reason or 'owned_process_cleanup_incomplete'
                return self.recover_quiet_claim(job, owned, reason, exitcode)
            return self.emit(state='stopping', job=self.store.get(job['id']), reason=stop['reason'])
        # A durable claim may have survived a crash between Popen and recording
        # the PID. Its original budget still bounds any marked live processes.
        deadline_ns = job['deadline_ns']
        # A manual reservation has no date deadline. An explicit DB deadline,
        # if one was later imposed by the user, is still enforced.
        manual = job.get('manual') is True
        if manual and job['timeout_seconds'] != 0:
            return self.recover_quiet_claim(job, owned, 'invalid_manual_timeout_contract', exitcode)
        if deadline_ns is None and not manual and job['claimed_ns'] is not None:
            deadline_ns = job['claimed_ns'] + job['timeout_seconds'] * 10**9
        if job['id'] not in self.monotonic_deadlines and deadline_ns is not None:
            remaining = max(0, (deadline_ns - self.clock_ns()) / 1e9)
            if not manual:
                remaining = min(job['timeout_seconds'], remaining)
            self.monotonic_deadlines[job['id']] = time.monotonic() + remaining
        timed_out = (deadline_ns is not None and self.clock_ns() >= deadline_ns) or (
            job['id'] in self.monotonic_deadlines and time.monotonic() >= self.monotonic_deadlines[job['id']])
        orphaned = bool(owned) and ((child is not None and exitcode is not None) or
                                   (job['identity'] and child is None and not live_parent))
        if timed_out or orphaned:
            reason = 'wall_timeout' if timed_out else 'parent_exited_with_owned_descendants'
            signal_owned(owned, signal.SIGTERM)
            self.terminating[job['id']] = {'kill_at_ns': self.clock_ns() + self.termination_grace_ns, 'reason': reason}
            return self.emit(state='stopping', job=self.store.get(job['id']), reason=reason)
        if uncertain_reason:
            return self.recover_quiet_claim(job, owned, uncertain_reason, exitcode)
        if child is not None and exitcode is not None:
            state = 'succeeded' if exitcode == 0 else 'failed'
            updated = self.store.finish(job['id'], state, 'process_exit', exitcode)
            self.children.pop(job['id'], None)
            return self.emit(state=state, job=updated, reason='process_exit')
        return self.emit(state='running', job=self.store.get(job['id']),
                         reason='recovered_exact_process_no_retry' if child is None else 'in_progress')

    def tick(self):
        if not self.acquire_lock():
            return self.emit(state='waiting', reason='another_worker_holds_process_lock')
        active = self.store.active()
        if len(active) > 1:
            return self.emit(state='blocked', reason='multiple_active_reservations_require_review')
        if active:
            return self.observe(active[0])
        queued = [item for item in self.store.list() if item['state'] == 'queued']
        if not queued:
            return self.emit(state='idle', reason='queue_empty')
        try:
            gate = self.gate_callback()
        except Exception as exc:
            return self.emit(state='waiting', reason='resource_gate_error: ' + repr(exc))
        if not isinstance(gate, dict) or gate.get('ready') is not True:
            return self.emit(state='waiting', reason=(gate.get('reason', 'resource_gate_not_ready')
                             if isinstance(gate, dict) else 'invalid_resource_gate'), gate=gate)
        job = self.store.claim_next()
        if job is None:
            return self.emit(state='waiting', reason='another_worker_holds_claim')
        return self.start(job)
