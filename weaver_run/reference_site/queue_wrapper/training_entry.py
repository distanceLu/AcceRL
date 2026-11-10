"""Foreground manual Policy/WEAVER-native-RM training with exact queue ownership.

This deployment runs until an explicit manual boundary stop or a detected failure.
The queue itself has no hard process timeout. The native server supervisor retains its error/stall protections.
Only an explicit manual stop (or failure) ends training. Saving/audit and the
following fixed-checkpoint policy evaluation have separate bounded phases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import signal
import sqlite3
import subprocess
import sys
import time


GPUS = ('GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c',
        'GPU-fe0edfe4-b193-1af7-6a98-cbff40214f74',
        'GPU-8c9bc264-7f09-989e-1ec7-cecd6cde782c',
        'GPU-07ae0bbc-8923-bd0c-199e-c726821e655d')
PIPE = '/dev/shm/mps_lcx1_gpu47_pzero2_20261020/pipe'
FRESH_PROFILE = '04_real'
BACKEND_CONFIG = 'git-world.json'
COST_SCHEDULER_PROFILE = '03_adaptive'
COST_SCHEDULER_CONFIG = '../recipe/cost_scheduler_config.json'
PIPELINE_CONFIG = '../recipe/pipeline_config.json'
NATIVE_REWARD = {'mode': 'libero_discounted_success_value_v1', 'link': 'sigmoid',
                 'loss': 'masked_half_mse', 'label_source': 'libero_env_step',
                 'calibrated': False, 'terminal': 'horizon_only', 'balance': 'none',
                 'loss_coefficient': 1.0, 'gamma_per_action': .99 ** (1. / 8.),
                 'discount_unit': 'action_steps',
                 'target_semantics': 'discounted_observed_episode_success'}
FRESH_OPTIONS = {'protocol': 'weaver_fresh_v1', 'dense_reward': True,
                 'signal_telemetry': True, 'refresh_gae': True, 'replay_age_limits': True,
                 'max_policy_age': 16, 'max_world_age': 3, 'max_sample_uses': 4,
                 'seed_balancing': True, 'early_seed_context': True,
                 'seed_scheduler_mode': 'balanced', 'seed_scheduler_seed': 20261028,
                 'real_policy_replay': True, 'real_fraction': .25}
OPTIMIZATIONS = {'selected_frame_encoding': True, 'kv_cache': True, 'decode_mode': 'last',
                 'feature_cache_max_bytes': 67108864, 'resident_rollout_state': True,
                 'resident_max_sessions': 8, 'resident_max_bytes': 67108864,
                 'validate_on_cpu': True, 'task_cache_entries': 64, 'safety_checks': 'batched'}
TERMINAL = {'stopped_by_user', 'failed', 'completed'}
POST_STOP_GRACE_SECONDS = 1800
# Error work is bounded by the runner's 120s allowance. This separate envelope
# only lets its exact cleanup, helper joins and final CPU report finish before
# the outer wrapper escalates. It is never appended to a new GPU/save budget.
ERROR_WORK_GRACE_SECONDS = 120
MANUAL_GPU_SAVE_GRACE_SECONDS = 900
ERROR_TEARDOWN_GRACE_SECONDS = 90
INITIAL_WORLD = {'backend': 'weaver', 'world_trainable': True, 'source_optimizer_steps': 3200,
                 'source_bundle_sha256': '6b45705115034914803224c3653533e30236fffdac544b46400b5af68bfd3443'}
PIPELINE_RECIPE = {'real_workers': 1, 'imagination_workers': 4, 'max_real_in_flight': 1,
                   'max_imagine_in_flight': 4, 'initial_state_enabled': True,
                   'initial_state_count': 50, 'initial_state_seed': 20261026,
                   'real_env_step_budget': 1000000, 'real_collection_action_limit': None,
                   'real_training_action_horizon': 150}


def validate_pipeline_admission(pipeline):
    # The real training task ends at 150 actions, with valid failure labels
    # and no Policy bootstrap. There is no extra collection censor cap;
    # evaluation remains 220 and earlier global-budget exhaustion is censored.
    if (not isinstance(pipeline, dict) or 'fresh_lookahead_updates' in pipeline
            or any(k not in pipeline or type(pipeline[k]) is not type(v) or pipeline[k] != v
                   for k, v in PIPELINE_RECIPE.items())):
        raise ValueError('Expected orange Real1/Imag4, 1M budget, a 150-action real training horizon and an explicit null collection cap without lookahead')
    return True


def validate_backend_admission(backend, pipeline, admission):
    required_backend = {'backend': 'weaver', 'reward_provider': 'world_head',
                        'reward_semantics': 'delta_native_score', 'terminal_rule': 'horizon_only',
                        'critic_mode': 'disabled', 'world_trainable': True}
    if any(type(backend.get(k)) is not type(v) or backend.get(k) != v for k, v in required_backend.items()):
        raise ValueError('Expected trainable WEAVER with its native uncalibrated reward score and horizon-only termination')
    native_reward = backend.get('backend_options', {}).get('native_reward_head')
    if (not isinstance(native_reward, dict) or set(native_reward) != set(NATIVE_REWARD)
            or any(type(native_reward.get(k)) is not type(v) or native_reward.get(k) != v
                   for k, v in NATIVE_REWARD.items())):
        raise ValueError('FRESH discounted-success reward target/loss protocol differs')
    fresh = backend.get('backend_options', {}).get('fresh')
    if (not isinstance(fresh, dict) or set(fresh) != set(FRESH_OPTIONS)
            or any(type(fresh.get(k)) is not type(v) or fresh.get(k) != v
                   for k, v in FRESH_OPTIONS.items())):
        raise ValueError('Full FRESH profile, age/use limits or source quota differs')
    options = backend.get('backend_options', {}).get('optimizations')
    if (not isinstance(options, dict) or set(options) != set(OPTIMIZATIONS)
            or any(type(options.get(k)) is not type(v) or options.get(k) != v
                   for k, v in OPTIMIZATIONS.items())):
        raise ValueError('All four pinned WEAVER execution optimizations must be enabled')
    validate_pipeline_admission(pipeline)
    training_environment = admission.get('online_training_environment', {})
    if (admission.get('configuration_admitted') is not True
            or admission.get('assets_checked') is not True or admission.get('errors')
            or backend.get('checkpoint_metadata', {}).get('load_report', {}).get('checkpoint_sha256') != INITIAL_WORLD['source_bundle_sha256']
            or admission.get('environment', {}).get('compatible') is not True
            or training_environment.get('compatible') is not True
            or training_environment.get('distributed_available') is not True
            or training_environment.get('nccl_available') is not True
            or training_environment.get('cuda_initialized') is not False
            or training_environment.get('gpu_access_performed') is not False
            or training_environment.get('python') != backend.get('backend_options', {}).get('training_python')):
        raise ValueError('Initial WEAVER asset or independent train/infer interpreter admission failed')
    return True

def validate_manual_contract(post_stop_grace_seconds=1800):
    if type(post_stop_grace_seconds) is not int or not 900 <= post_stop_grace_seconds <= 3600:
        raise ValueError('Post-stop save/audit grace must be 900..3600 seconds')
    return {'run_until_manual_stop': True, 'expected_queue_manual': True,
            'expected_queue_timeout_seconds': 0, 'soft_stop_seconds': None,
            'deadline_ns': None, 'target_policy_epochs': None,
            'post_stop_grace_seconds': post_stop_grace_seconds}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def mps_authorization(*, check_live=True):
    """Frozen per-boot identity; no PID discovery or mutation is an authorization."""
    path = Path('/mnt/data/lcx1/dong/need_data/weaver_cycle_restore_20261105_v1/queue_wrapper_v1/mps_authorization.json')
    if path.is_symlink():
        raise ValueError('MPS authorization may not redirect through a symlink')
    value = read_json(path)
    if (value.get('protocol') != 'gpu47_mps_authorization_v1' or value.get('uid') != 1004
            or value.get('pipe') != PIPE or value.get('scope') != [4, 5, 6, 7]
            or tuple(value.get('gpu_uuids', ())) != GPUS
            or not isinstance(value.get('boot_id'), str)
            or not re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', value['boot_id'])):
        raise ValueError('Frozen MPS authorization scope/boot identity differs')
    for role in ('server', 'control'):
        identity = value.get(role, {})
        if (identity.get('uid') != 1004
                or any(type(identity.get(k)) is not int or identity[k] <= 0
                       for k in ('pid', 'birth_ticks'))):
            raise ValueError('Frozen MPS exact PID/UID/birth is required: ' + role)
    if check_live:
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != value['boot_id']:
            raise ValueError('Server rebooted after MPS authorization')
        for role in ('server', 'control'):
            expected = value[role]
            actual = proc_identity(expected['pid'])
            if (not actual or actual.get('state') in ('Z', 'X')
                    or any(actual.get(k) != expected[k] for k in ('pid', 'uid', 'birth_ticks'))):
                raise ValueError('Authorized MPS process no longer exists: ' + role)
            command = (Path('/proc') / str(expected['pid']) / 'cmdline').read_bytes().split(b'\0')
            if not command or Path(os.fsdecode(command[0])).name.strip() != 'nvidia-cuda-mps-' + role:
                raise ValueError('Authorized MPS executable differs: ' + role)
    return value


def validate_mps_receipt(mps):
    authorization = mps_authorization()
    if (mps.get('scope') != [4, 5, 6, 7] or mps.get('pipe') != PIPE
            or mps.get('log') != str(Path(PIPE).parent / 'log')
            or mps.get('boot_id') != authorization['boot_id']
            or [(x.get('index'), x.get('uuid')) for x in mps.get('cards', [])]
            != list(zip((4, 5, 6, 7), GPUS))):
        raise ValueError('MPS receipt does not identify the admitted per-boot GPU4-7 scope')
    for role in ('server', 'control'):
        if any(mps.get(role, {}).get(k) != authorization[role][k]
               for k in ('pid', 'uid', 'birth_ticks')):
            raise ValueError('MPS receipt birth identity mismatch: ' + role)
    return authorization


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def log(**fields):
    print(json.dumps({'at_ns': time.time_ns(), **fields}, ensure_ascii=False), flush=True)


def verify_git(source, expected):
    package = Path(__file__).resolve().parents[2]
    if source != package/"engine" or not re.fullmatch("[0-9a-f]{40}", expected):
        raise ValueError("Expected the engine and commit from the queued Git checkout")
    sys.path.insert(0,str(package))
    from git_version import revision
    return {"git_commit":revision(package.parent, expected)}


def validate_cost_scheduler_admission(source):
    path = (source / COST_SCHEDULER_CONFIG).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.parent != source.parent/"recipe" or (source.parent/"recipe"/path.name).is_symlink():
        raise ValueError('Adaptive scheduler config path/SHA differs from frozen source')
    settings = read_json(path)
    if (settings.get('schema') != 1
            or any(settings.get(k) is not True for k in ('enabled', 'macro_enabled', 'allow_probe'))):
        raise ValueError('Adaptive scheduler must explicitly enable scheduling, macro feedback and probes')
    if (any(key in settings for key in ('execution', 'cycle_feedback'))
            or type(settings.get('training_gangs')) is not int or settings['training_gangs'] != 1
            or type(settings.get('card_inflight')) is not int or settings['card_inflight'] != 2):
        raise ValueError('Expected the orange single-training-gang adaptive scheduler without later execution/cycle extensions')
    return {'profile': COST_SCHEDULER_PROFILE, 'source_path': str(path),
            'source_sha256': digest, 'settings': settings}


def check_cost_scheduler_state(state, source, root):
    expected = validate_cost_scheduler_admission(source)
    observed = state.get('cost_scheduler', {})
    if (any(observed.get(k) != expected[k] for k in ('profile', 'source_path', 'source_sha256'))
            or observed.get('configuration_path') != str(root / 'formal_v3/cost_scheduler_config.json')
            or observed.get('configuration_sha256') != expected['source_sha256']
            or observed.get('settings') != expected['settings']
            or any(observed.get('settings', {}).get(k) is not True
                   for k in ('enabled', 'macro_enabled', 'allow_probe'))):
        raise ValueError('Runner did not activate the exact frozen adaptive scheduler')
    actual = state.get('pipeline_config', {})
    expected_pipeline = read_json(source / PIPELINE_CONFIG)
    validate_pipeline_admission(expected_pipeline)
    validate_pipeline_admission(actual)
    if (actual != expected_pipeline
            or state.get('pipeline_config_path') != str(root / 'formal_v3/pipeline_config.json')):
        raise ValueError('Runner did not activate the exact orange pipeline with the 150-action real training horizon')
    return True


def measurement_stop_due(receipt, now_monotonic, synchronized_epochs, seconds):
    if type(seconds) is not int or not 3600 <= seconds <= 21600:
        raise ValueError('Authorized measurement duration must be 1..6 hours')
    measurement = receipt.setdefault('measurement', {
        'duration_seconds': seconds, 'anchor': 'first_successful_synchronized_update',
        'stop_method': 'user_authorized_boundary_request', 'hard_process_deadline': False})
    if measurement['duration_seconds'] != seconds:
        raise ValueError('Measurement duration cannot change during the run')
    if synchronized_epochs > 0 and 'started_monotonic' not in measurement:
        measurement.update(started_monotonic=now_monotonic, started_ns=time.time_ns(),
                           initial_synchronized_epochs=synchronized_epochs)
    start = measurement.get('started_monotonic')
    return start is not None and now_monotonic - start >= seconds


def validate(source, root, git_commit):
    if not source.is_absolute() or not root.is_absolute():
        raise ValueError('Source and experiment root must be absolute paths')
    if source.resolve() != source or root.resolve() != root:
        raise ValueError('Source/root paths must be canonical and not symlinks')
    evidence = verify_git(source, git_commit)
    scheduler = validate_cost_scheduler_admission(source)
    # This module is CPU-only, and its bytes have just been verified.
    paths = runpy.run_path(str(source / 'candidate_paths.py'))
    paths['experiment_root'](root)
    paths['candidate_data'](root / 'formal_v3', root=root)
    for name in ('formal_v3', 'queue_receipt.json', 'queue_entry.lock'):
        if os.path.lexists(root / name):
            raise ValueError('Existing output cannot be reused: ' + str(root / name))
    backend = read_json(root / BACKEND_CONFIG)
    # Probe the distinct pinned WEAVER interpreter without constructing a CUDA
    # model. The merged controller environment need not import WEAVER itself.
    cpu_env = launch_environment(root, source, os.environ)
    cpu_env['CUDA_VISIBLE_DEVICES'] = ''
    inspected = subprocess.run([sys.executable, str(source / 'inspect_world_backend.py'),
                               '--config', str(root / BACKEND_CONFIG),
                               '--check-local'], cwd=source, env=cpu_env,
                              capture_output=True, text=True, timeout=90, check=True)
    admission = json.loads(inspected.stdout)
    validate_backend_admission(backend, read_json(source / PIPELINE_CONFIG), admission)
    mps_path = root / 'mps.json'
    if mps_path.is_symlink():
        raise ValueError('MPS receipt must not redirect through a symlink')
    mps = read_json(mps_path)
    authorization = validate_mps_receipt(mps)
    return {**evidence, 'source': str(source), 'experiment_root': str(root),
            'backend': backend, 'mps_receipt_sha256': hashlib.sha256(mps_path.read_bytes()).hexdigest(),
            'world_admission': admission, 'cost_scheduler': scheduler, 'mps_boot_id': authorization['boot_id'],
            'gpu_access_performed': False}


def launch_environment(root, source, environ):
    env = {k: v for k, v in environ.items()
           if not k.startswith(('ACCERL_', 'ACCE_', 'WEAVER_', 'NSYS_',
                                'CUDA_INJECTION', 'NVTX_INJECTION'))}
    for name in ('LD_PRELOAD', 'SOFT_STOP_AT', 'MEASURE_STOP_AFTER_EPOCHS',
                 'SHORT_TEST_DEADLINE', 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',
                 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT', 'CUDA_MPS_CLIENT_PRIORITY',
                 'CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING',
                 'CUDA_MPS_SM_PARTITION'):
        env.pop(name, None)
    env.update(ACCERL_EXPERIMENT_ROOT=str(root),
               ACCERL_FRESH_PROFILE=FRESH_PROFILE,
               ACCERL_WORLD_BACKEND_CONFIG=str(root / BACKEND_CONFIG),
               ACCERL_PIPELINE_CONFIG=str(source / PIPELINE_CONFIG),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def check_state(state, source, root, run_id=None, supervisor=None):
    rid = state.get('run_id', '')
    if (not re.fullmatch(r'weaverintegrated_[A-Za-z0-9_-]+', rid)
            or (run_id is not None and rid != run_id)
            or state.get('candidate') != 'weaver_integrated_v1'
            or state.get('source_path') != str(source)
            or state.get('experiment_root') != str(root)
            or state.get('uid') != 1004 or state.get('mode') != 'manual'
            or state.get('until_manual_stop') is not True
            or state.get('stop_on_target') is not False
            or state.get('deadline_unix') is not None):
        raise ValueError('Runner state source/root/run_id/control identity mismatch')
    if supervisor and any(state.get('supervisor', {}).get(k) != supervisor.get(k)
                          for k in ('pid', 'uid', 'birth_ticks')):
        raise ValueError('Runner state supervisor birth identity mismatch')
    fresh = state.get('world_backend_config', {}).get('backend_options', {}).get('fresh', {})
    if (not isinstance(fresh, dict) or set(fresh) != set(FRESH_OPTIONS)
            or any(type(fresh.get(k)) is not type(v) or fresh.get(k) != v for k,v in FRESH_OPTIONS.items())):
        raise ValueError('Actual runner FRESH settings differ from the orange profile')
    return rid


def synchronized_epochs(data, run_id):
    """Read only, bounded query; optimizer counters alone are insufficient."""
    db = data / 'training/collection.sqlite'
    if not db.is_file():
        return 0
    begin = time.monotonic()
    connection = sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True, timeout=1)
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.set_progress_handler(lambda: int(time.monotonic() - begin > 2), 1000)
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'").fetchone():
            return 0
        row = connection.execute(
            'SELECT details_json FROM events WHERE run_id=? AND workload IS NULL '
            'AND kind=? ORDER BY ts_ns DESC, id DESC LIMIT 1',
            (run_id, 'synchronized_epoch_completed')).fetchone()
        if row is None:
            return 0
        details = json.loads(row[0])
        epochs = details.get('completed_epochs')
        if type(epochs) is not int or epochs < 0 or details.get('global_step') != epochs:
            raise ValueError('Malformed synchronized epoch receipt')
        return epochs
    finally:
        connection.close()


def stop_cause(epochs=None, elapsed=None, signal_number=None):
    # Dates, target accuracy, epoch counts and elapsed durations do not stop training.
    return 'external_signal_' + str(signal_number) if signal_number is not None else None

def read_stop_evidence(data, run_id, state=None):
    """Read exact-run evidence without translating an error into a user stop.

    The runner writes stop.flag before runnerstatus.json. A same-run nonmanual
    flag may therefore precede its error state; preserve it under the bounded
    error envelope while awaiting corroboration. It never authorizes success.
    """
    path = data / 'training/stop.flag'
    if path.is_symlink():
        raise ValueError('Stop flag may not redirect through a symlink')
    if not path.is_file():
        return None
    if os.name == 'posix' and path.stat().st_uid != 1004:
        raise ValueError('Stop flag filesystem owner differs from this run')
    flag = read_json(path)
    if (not isinstance(flag, dict) or flag.get('run_id') != run_id
            or type(flag.get('uid')) is not int or flag['uid'] != 1004
            or not isinstance(flag.get('reason'), str) or not flag['reason']
            or type(flag.get('requested_ns')) is not int or flag['requested_ns'] <= 0):
        raise ValueError('Existing stop flag UID/run/timestamp/reason differs')
    if flag['reason'] == 'user_requested_stop':
        kind = 'manual'
    elif (state and state.get('run_id') == run_id and state.get('uid') == 1004
          and state.get('state') in ('stopping_error', 'failed')
          and flag['reason'] == state.get('error')):
        kind = 'runner_error'
    else:
        kind = 'same_run_nonmanual_stop'
    return {'kind': kind, 'flag': flag}


def read_manual_stop(data, run_id):
    evidence = read_stop_evidence(data, run_id)
    if evidence is None:
        return None
    if evidence['kind'] != 'manual':
        raise ValueError('Existing stop flag is not an exact authorized manual stop')
    return evidence['flag']


def stop_deadlines(evidence, state, *, now_ns, now_monotonic,
                   post_stop_grace_seconds, prior=None):
    """Pin wall-clock evidence to monotonic deadlines; never renew a budget."""
    prior = prior or {}
    deadlines = dict(prior)
    error = (evidence is not None and evidence['kind'] != 'manual') or (
        state.get('state') in ('stopping_error', 'failed') and bool(state.get('error')))
    stamp = evidence['flag']['requested_ns'] if evidence else now_ns
    # Future/corrupt timestamps cannot extend either allowed window.
    stamp = min(stamp, now_ns)
    manual_stamp = state.get('stop_requested_ns')
    if type(manual_stamp) is not int or manual_stamp <= 0:
        manual_stamp = stamp
    manual_stamp = min(manual_stamp, now_ns, stamp)
    if error:
        error_stamp = state.get('error_started_ns')
        if type(error_stamp) is not int or error_stamp <= 0:
            error_stamp = stamp
        error_stamp = min(error_stamp, now_ns)
        work_ns = min(error_stamp + ERROR_WORK_GRACE_SECONDS * 10**9,
                      manual_stamp + MANUAL_GPU_SAVE_GRACE_SECONDS * 10**9)
        # For a nonmanual flag preceding state persistence, its timestamp is
        # the first error observation; later state writes cannot reset it.
        if evidence is not None and evidence['kind'] != 'manual':
            work_ns = min(work_ns, stamp + ERROR_WORK_GRACE_SECONDS * 10**9)
        until_ns = work_ns + ERROR_TEARDOWN_GRACE_SECONDS * 10**9
        deadlines['kind'] = ('runner_error' if evidence is None or evidence['kind'] == 'manual'
                             else evidence['kind'])
        proposed_work = now_monotonic + (work_ns - now_ns) / 1e9
        deadlines['work_deadline'] = min(prior.get('work_deadline', proposed_work), proposed_work)
    else:
        until_ns = stamp + post_stop_grace_seconds * 10**9
        deadlines['kind'] = 'manual' if evidence else 'external_signal'
    proposed = now_monotonic + (until_ns - now_ns) / 1e9
    deadlines['supervisor_deadline'] = min(prior.get('supervisor_deadline', proposed), proposed)
    return deadlines


def request_stop(source, data, run_id, env):
    command = [sys.executable, str(source / 'request_stop.py'),
               '--data', str(data), '--run-id', run_id]
    result = subprocess.run(command, env=env, cwd=source, text=True,
                            capture_output=True, timeout=30, check=True)
    receipt = json.loads(result.stdout)
    if receipt.get('run_id') != run_id or receipt.get('stop_requested') is not True:
        raise ValueError('Stop request returned no matching acknowledgement')
    return receipt


def supervisor_environment(env):
    """CPU supervisor stays CPU-only; frozen runner restores verified child GPU IDs."""
    return dict(env, CUDA_VISIBLE_DEVICES='')


def wait_idle(source, env, seconds=120):
    """Recheck only transient utilization; do not launch/retry a training run."""
    code = ("import json,sys; import gpu47_safety as s\n"
            "try: s.preflight()\n"
            "except RuntimeError as e:\n"
            " print(json.dumps({'error':str(e)})); sys.exit(3 if str(e)=='GPU4-7 is not idle' else 1)\n"
            "print(json.dumps({'preflight':'passed'}))\n")
    deadline = time.monotonic() + seconds
    while True:
        remaining = max(.1, deadline-time.monotonic())
        result = subprocess.run([sys.executable, '-c', code], cwd=source, env=env,
                                capture_output=True, text=True, timeout=min(30., remaining))
        if result.returncode == 0:
            return
        if result.returncode != 3 or time.monotonic() >= deadline:
            raise RuntimeError('Queue admission preflight failed: ' + (result.stdout + result.stderr)[-1500:])
        time.sleep(2)


def proc_identity(pid):
    try:
        base = Path('/proc') / str(int(pid))
        fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        uid = int(next(x.split()[1] for x in (base / 'status').read_text().splitlines()
                       if x.startswith('Uid:')))
        return {'pid': int(pid), 'uid': uid, 'birth_ticks': int(fields[19]),
                'state': fields[0], 'ppid': int(fields[1])}
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def same_process(expected, current):
    return bool(expected and current and current.get('state') not in ('Z', 'X')
                and all(expected.get(k) == current.get(k) for k in ('pid', 'uid', 'birth_ticks')))


def marked_children(job_id, known=(), *, marker_name='EXPERIMENT_QUEUE_JOB_ID'):
    marker = (marker_name + '=' + job_id).encode()
    found = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        identity = proc_identity(int(entry.name))
        if not identity or identity['uid'] != 1004 or identity['state'] in ('Z', 'X'):
            continue
        try:
            command = (entry / 'cmdline').read_bytes()
            if b'nvidia-cuda-mps' in command:
                continue
            if marker in (entry / 'environ').read_bytes().split(b'\0'):
                found[identity['pid']] = identity
        except OSError:
            pass
    for identity in known:
        current = proc_identity(identity['pid'])
        if identity['pid'] != os.getpid() and identity.get('uid') == 1004 and same_process(identity, current):
            found[identity['pid']] = current
    return list(found.values())


def signal_verified(identities, sig):
    sent = []
    for expected in identities:
        if expected.get('uid') != 1004 or expected['pid'] == os.getpid():
            continue
        if not same_process(expected, proc_identity(expected['pid'])):
            continue
        try:
            if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
                fd = os.pidfd_open(expected['pid'])
                try:
                    if not same_process(expected, proc_identity(expected['pid'])):
                        continue
                    signal.pidfd_send_signal(fd, sig)
                finally:
                    os.close(fd)
            else:
                os.kill(expected['pid'], sig)
            sent.append({'identity': expected, 'signal': sig})
        except ProcessLookupError:
            pass
    return sent


def scoped_cleanup(job_id, known=(), *, marker_name='EXPERIMENT_QUEUE_JOB_ID'):
    """Emergency fallback for startup/runner failure; never signals GPU peers/MPS."""
    signals = []
    tracked = marked_children(job_id, known, marker_name=marker_name)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        signals.extend(signal_verified(tracked, sig))
        until = time.monotonic() + 12
        while tracked and time.monotonic() < until:
            time.sleep(.5)
            tracked = marked_children(job_id, tracked, marker_name=marker_name)
        if not tracked:
            break
    return {'signals': signals, 'survivors': marked_children(job_id, tracked, marker_name=marker_name),
            'scope': 'exact queue marker or owned PID/UID/birth; MPS excluded'}


def native_preflight_plan(args, run_id):
    backend = read_json(args.root / BACKEND_CONFIG)
    mps = read_json(args.root / 'mps.json')
    authorization = validate_mps_receipt(mps)
    return {'protocol': 'weaver_online_tiny_smoke_v1', 'run_id': run_id,
            'expected_uid': 1004, 'hostname': 'h20-xyff52ab-0001',
            'python': backend['backend_options']['training_python'],
            'output': str(args.root / 'native_gpu_preflight'), 'max_seconds': 150,
            'gpu_uuids': list(GPUS),
            'mps': {'pipe': PIPE, **{role: {'pid': authorization[role]['pid'],
                    'uid': authorization[role]['uid'], 'birth': authorization[role]['birth_ticks']}
                    for role in ('server', 'control')}}}


def native_preflight_command(args, run_id, *, plan_only=False):
    backend = read_json(args.root / BACKEND_CONFIG)
    command = [backend['backend_options']['training_python'], str(args.source/'weaver_online_smoke.py'),
        '--plan', str(args.root / 'native_preflight_plan.json')]
    return command + (['--plan-only'] if plan_only else [])


def native_preflight_acceptance(report, completed, exit_code, remaining, *, run_id):
    authorization = mps_authorization()
    reasons = []
    if (exit_code != 0 or report.get('passed') is not True or report.get('gpu_started') is not True or completed != report
            or report.get('run_id') != run_id or remaining
            or report.get('cleanup', {}).get('all_owned_exited') is not True):
        reasons.append('native_preflight_not_cleanly_completed')
    identities = report.get('child_identities', [])
    expected = {(role, rank) for role in ('trainer', 'inference') for rank in range(4)}
    if (report.get('expected_children') != 8 or len(identities) != 8
            or {(row.get('role'), row.get('rank')) for row in identities} != expected
            or len({row.get('pid') for row in identities}) != 8
            or any(row.get('uid') != 1004 or type(row.get('birth')) is not int or row['birth'] <= 0
                   or type(row.get('pid')) is not int or row['pid'] <= 0
                   or type(row.get('rank')) is not int or not 0 <= row['rank'] < 4
                   or row.get('gpu_uuid') != GPUS[row['rank']] for row in identities)):
        reasons.append('native_preflight_missing_eight_actual_gpu_children')
    publications = report.get('publications', [])
    updates = report.get('updates', [])
    if (report.get('successful_updates_per_rank') != 2 or len(updates) != 2
            or any(len(group) != 4 for group in updates)
            or any(entry.get('state', {}).get('global_steps') != update
                   or entry.get('state', {}).get('skipped_steps') != 0
                   or entry.get('state', {}).get('distributed_world_size') != 4
                   for update, group in enumerate(updates, 1) for entry in group)
            or [row.get('version') for row in publications] != [0, 1, 2]
            or any(any(len(row.get(field, [])) != 4 for field in ('inventories','acks','released','served'))
                   for row in publications)):
        reasons.append('native_preflight_update_or_publication_coverage_missing')
    if (report.get('successful_rm_updates_per_rank') != 1 or report.get('missing_class_kept_rm_weights') is not True
            or report.get('rm_weights_changed_on_active_update') is not True
            or any(entry.get('state', {}).get('rm_updates') != 1
                   or entry.get('state', {}).get('rm_disabled_updates') != update-1
                   for update, group in enumerate(updates, 1) for entry in group)):
        reasons.append('native_preflight_head_active_or_missing_class_skip_not_verified')
    for row in publications:
        inventories = row.get('inventories', [])
        if (len(inventories) != 4 or row.get('acks') != inventories or row.get('released') != inventories
                or any(item.get('version') != row.get('version') for item in inventories)
                or len({item.get('hash') for item in inventories}) != 1
                or any(item.get('served_version') != row.get('version')
                       or not inventories or item.get('hash') != inventories[0].get('hash')
                       for item in row.get('served', []))):
            reasons.append('native_preflight_ACK_release_or_actual_served_version_mismatch')
    for stage in ('preflight', 'postflight'):
        status = report.get(stage, {})
        identity = status.get('mps_server', {})
        if (status.get('clients') != [] or status.get('default_cap') != 100. or status.get('server_cap') != 100.
                or any(identity.get(key) != authorization['server'][expected]
                       for key, expected in (('pid', 'pid'), ('uid', 'uid'), ('birth', 'birth_ticks')))):
            reasons.append('native_preflight_mps_identity_or_idle_mismatch:' + stage)
    return {'passed': not reasons, 'reasons': reasons,
            'scope': 'two tiny native-engine DDP updates with binary synthetic RM and three IPC versions; '
                     'transport qualification only, not FRESH dense targets, current-critic refresh, '
                     'mixed Replay, full-model memory or learning-effect admission'}


def run_native_preflight(args, env, *, abort=lambda: False):
    """One durable claim, at most180s including cleanup; failure never starts training."""
    claim_path = args.root/'native_gpu_preflight_claim.json'
    # An existing attempt is authoritative even if its old MPS identity is now
    # stale or its plan cannot be read. Reject it before any plan/MPS admission;
    # the exclusive open below still closes concurrent first-launch races.
    if os.path.lexists(claim_path):
        raise FileExistsError('Native preflight claim already exists; refusing a second launch: ' + str(claim_path))
    run_id = 'weaverintegrated_preflight_' + str(time.time_ns())
    started = time.monotonic(); hard_deadline = started + 180
    plan = native_preflight_plan(args, run_id)
    claim = {'run_id': run_id, 'started_ns': time.time_ns(), 'state': 'claimed',
             'total_budget_seconds': 180, 'script_budget_seconds': 150,
             'command': native_preflight_command(args, run_id), 'automatic_retry': False}
    with claim_path.open('x', encoding='utf-8') as handle:
        json.dump(claim, handle); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    # The exclusive launch claim is acquired before this write. A duplicate
    # invocation cannot change the first attempt's plan or create a new budget.
    with (args.root / 'native_preflight_plan.json').open('x', encoding='utf-8') as handle:
        json.dump(plan, handle, indent=2); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    output = args.root/'native_gpu_preflight'
    if os.path.lexists(output):
        raise RuntimeError('Native preflight output already exists; refusing a second GPU launch')
    known, process, failure = [], None, None
    marker_name = 'WEAVER_PREFLIGHT_RUN_ID'
    scoped_env = dict(env, WEAVER_PREFLIGHT_RUN_ID=run_id, ACCERL_RUN_ID=run_id)
    try:
        with (args.root/'native_gpu_preflight.log').open('x', encoding='utf-8') as handle:
            process = subprocess.Popen(claim['command'], env=scoped_env, cwd=args.source,
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            identity = proc_identity(process.pid)
            if identity is None or identity['uid'] != 1004:
                raise RuntimeError('Cannot identify native preflight controller birth')
            known.append(identity)
            claim.update(state='running', controller=identity)
            atomic_json(claim_path, claim)
            while process.poll() is None:
                known = marked_children(run_id, known, marker_name=marker_name)
                if abort():
                    raise RuntimeError('Native preflight aborted by an external queue stop')
                if time.monotonic() >= started + 150:
                    raise TimeoutError('Native preflight GPU budget exhausted; remaining30s reserved for cleanup')
                time.sleep(.2)
        remaining = marked_children(run_id, known, marker_name=marker_name)
        report = read_json(output/'report.json'); completed = read_json(output/'completed.json')
        review = native_preflight_acceptance(report, completed, process.returncode, remaining, run_id=run_id)
        claim['acceptance'] = review
        if not review['passed']:
            raise RuntimeError('Native preflight failed: ' + repr(review['reasons']))
        # Do not reuse smoke PID receipts or a stale idle observation for the
        # real-model launcher. Both checks query actual UUID/MPS identities.
        for observation in range(2):
            seconds = int(hard_deadline - time.monotonic())
            if seconds < 2:
                raise TimeoutError('Native preflight budget lacks two final idle checks')
            wait_idle(args.source, env, seconds=min(15, seconds))
            if observation == 0:
                time.sleep(1)
        claim.update(state='passed', report=str(output/'report.json'), idle_checks_after_preflight=2)
    except BaseException as error:
        failure = error
        claim.update(state='failed', error=repr(error))
    finally:
        remaining = marked_children(run_id, known, marker_name=marker_name)
        if remaining:
            claim['emergency_cleanup'] = scoped_cleanup(run_id, remaining, marker_name=marker_name)
            remaining = claim['emergency_cleanup']['survivors']
        if process is not None:
            try: process.wait(timeout=max(.1, min(2., hard_deadline-time.monotonic())))
            except subprocess.TimeoutExpired: pass
        claim.update(finished_ns=time.time_ns(), elapsed_seconds=time.monotonic()-started,
                     all_owned_exited=not remaining)
        if remaining or claim['elapsed_seconds'] > 180:
            claim.update(state='failed', error='Native preflight did not finish within its total cleanup budget')
        atomic_json(claim_path, claim)
    if failure or claim['state'] != 'passed':
        raise RuntimeError('One-time native GPU preflight failed; no retry or formal launch: ' + repr(claim))
    return claim


def validate_online_world_audit(world, run_id, policy_step, updates):
    """Consume the dedicated WEAVER interpreter's CPU audit, never its model.

    The controller combines P/R's ordinary checkpoint with the independently
    exported native World. A source step3200 identity is initialization only;
    actual version/optimizer count must describe this new online run.
    """
    reasons = []
    if not isinstance(world, dict):
        return {'passed': False, 'reasons': ['online_world_cpu_audit_missing']}
    required = {**INITIAL_WORLD, 'protocol': 'weaver_online_joint_weights_v1',
                'run_id': run_id, 'world_size': 4, 'policy_step': policy_step,
                'successful_updates': updates, 'training_steps': updates, 'model_version': updates,
                'weights_only': True, 'map_location': 'cpu', 'mmap': True,
                'valid': True, 'finite': True, 'tensor_count': 866}
    if type(updates) is not int or updates < 0:
        reasons.append('online_world_successful_update_count_invalid')
    if any(type(world.get(key)) is not type(value) or world.get(key) != value for key, value in required.items()):
        reasons.append('online_world_cpu_audit_identity_or_steps_mismatch')
    if not isinstance(world.get('weights_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', world['weights_sha256']):
        reasons.append('online_world_native_weight_hash_missing')
    if not isinstance(world.get('python'), str) or not world['python'].startswith('/'):
        reasons.append('online_world_independent_cpu_interpreter_missing')
    if world.get('cuda_initialized') is not False:
        reasons.append('online_world_cpu_audit_initialized_cuda')
    if type(updates) is not int or type(world.get('rm_updates')) is not int or not 0 <= world['rm_updates'] <= updates:
        reasons.append('native_reward_head_update_count_invalid')
    counts = world.get('reward_target_counts', {})
    if (world.get('native_reward_head') != NATIVE_REWARD or 'reward_class_counts' in world
            or not isinstance(counts, dict)
            or counts.get('scope') != 'global_selected_frames_once_per_successful_world_update'
            or any(type(counts.get(k)) is not int or counts[k] < 0
                   for k in ('valid_count', 'masked_count', 'disabled_updates'))
            or type(updates) is not int or type(world.get('rm_updates')) is not int
            or counts.get('disabled_updates', -1) + world.get('rm_updates', -1) != updates
            or counts.get('valid_count', -1) + counts.get('masked_count', -1) != updates * 10240):
        reasons.append('fresh_dense_reward_target_audit_missing_or_inconsistent')
    return {'passed': not reasons, 'reasons': reasons}


def acceptance(state, report, exit_code, remaining, stop_flag):
    reasons = []
    if exit_code != 0 or state.get('state') != 'stopped_by_user' or state.get('error'):
        reasons.append('runner_not_cleanly_stopped')
    if state.get('training_stopped') is not True or state.get('cleanup', {}).get('survivors') != [] or remaining:
        reasons.append('cleanup_not_confirmed')
    rid = state.get('run_id')
    if (not stop_flag or stop_flag.get('run_id') != rid or stop_flag.get('uid') != 1004
            or stop_flag.get('reason') != 'user_requested_stop'):
        reasons.append('exact_manual_stop_evidence_missing')
    manual = state.get('manual_completion') or {}
    steps = manual.get('steps', {})
    p = steps.get('policy')
    if (type(p) is not int or p < 0 or any(type(v) is not int for v in steps.values())
            or steps != {'policy': p, 'ctrl': p // 5}):
        reasons.append('policy_joint_world_synchronized_steps_mismatch')
    accepted_reasons = {'final', 'final_before_first_update'} if p == 0 else {'final'}
    audit = state.get('checkpoint_cpu_audit') or {}
    if (audit.get('valid') is not True or audit.get('steps') != steps
            or audit.get('manifest_reason') not in accepted_reasons
            or manual.get('checkpoint_reason') not in accepted_reasons):
        reasons.append('final_checkpoint_audit_not_confirmed')
    if audit.get('monolith_steps') != {'policy': steps.get('policy')}:
        reasons.append('policy_only_monolith_steps_mismatch')
    if 'frozen_world' in audit:
        reasons.append('online_world_cannot_use_frozen_world_audit')
    world_review = validate_online_world_audit(audit.get('online_world'), rid, p, steps.get('ctrl'))
    reasons.extend(world_review['reasons'])
    if (report.get('run_id') != rid or report.get('state') != 'stopped_by_user'
            or report.get('source') != state.get('source_path')):
        reasons.append('final_report_identity_mismatch')
    reported = {k: report.get('training', {}).get(k, {}).get('global_updates')
                for k in ('policy', 'ctrl')}
    if report.get('training', {}).get('reward', {}).get('global_updates', 0):
        reasons.append('disabled_external_reward_has_updates')
    if reported != steps or not report.get('experiment_completed'):
        reasons.append('final_report_completion_mismatch')
    if type(p) is int and p > 0 and not report.get('synchronized_epoch_completed'):
        reasons.append('synchronized_completion_receipt_missing')
    return {'passed': not reasons, 'reasons': reasons, 'observed_steps': steps,
            'stopped_before_first_update': p == 0,
            'scope': 'manual boundary checkpoint and cleanup; no minimum update or model-quality claim'}


def runner_command(args):
    return [sys.executable, str(args.source / 'runner.py'), '--mode', 'manual',
            '--data', str(args.root / 'formal_v3'),
            '--pipeline-config', str(args.source / PIPELINE_CONFIG),
            '--world-backend-config', str(args.root / BACKEND_CONFIG),
            '--cost-scheduler-config', str((args.source / COST_SCHEDULER_CONFIG).resolve())]


def run(args):
    budgets = validate_manual_contract(args.post_stop_grace_seconds)
    if getattr(args, 'stop_control', None) is not None or getattr(args, 'measurement_seconds', None) is not None:
        raise ValueError('Orange deployment is manual-stop only; no inherited calendar/duration')
    if getattr(args, 'native_gpu_preflight', False) is not True:
        raise ValueError('This online candidate requires explicit --native-gpu-preflight once after queue admission')
    source, root = args.source, args.root
    verified = validate(source, root, args.git_commit)
    if args.validate_only:
        log(validation=verified, budgets=budgets, valid=True, outputs_created=False,
            native_preflight_argv=native_preflight_command(args, 'weaverintegrated_preflight_plan_only', plan_only=True),
            native_preflight_plan=native_preflight_plan(args, 'weaverintegrated_preflight_plan_only'),
            native_gpu_preflight_executed=False)
        return 0
    job_id = os.environ.get('EXPERIMENT_QUEUE_JOB_ID', '')
    if sys.platform != 'linux' or os.getuid() != 1004 or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', job_id):
        raise ValueError('Run requires Linux UID1004 and an exact queue job marker')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(GPUS)
            or os.environ.get('CUDA_MPS_PIPE_DIRECTORY') != PIPE):
        raise ValueError('Queue GPU/MPS environment does not match the validated receipt')
    if (os.environ.get('EXPERIMENT_QUEUE_MANUAL') != '1'
            or os.environ.get('EXPERIMENT_QUEUE_TIMEOUT_SECONDS') != '0'):
        raise ValueError('Manual training requires an explicitly unlimited manual queue reservation')
    with (root / 'queue_entry.lock').open('x', encoding='utf-8') as lock:
        lock.write(job_id + '\n')
    data = root / 'formal_v3'
    receipt_path = root / 'queue_receipt.json'
    env = launch_environment(root, source, os.environ)
    command = runner_command(args)
    receipt = {'schema': 3, 'job_id': job_id, 'state': 'starting', 'started_ns': time.time_ns(),
               'source_validation': verified, 'argv': command, **budgets,
               'stop_protocol_reason': 'user_requested_stop',
               'stop_authorization': 'user authorized new JIT GPU training; manual stop with no automatic date or duration',
               'measurement': {'anchor': 'manual_stop', 'four_hour_cutoff_enabled': False, 'stop_at_ns': None},
               'stop_trigger': None, 'run_id': None, 'last_synchronized_epochs': 0}
    if getattr(args,'stop_control',None) is not None:
        receipt['stop_authorization']='explicit user calendar stop; see bound stop_control and revision history'
    atomic_json(receipt_path, receipt)
    received = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: received.append(number))
    start = time.monotonic()
    stop_at = None
    stop_limits = None
    child = None
    owned = []
    state = {}
    next_log = 0.0
    last_stop_attempt = float('-inf')
    failure = None
    try:
        wait_idle(source, env)
        receipt['native_gpu_preflight'] = run_native_preflight(args, env, abort=lambda: bool(received))
        atomic_json(receipt_path, receipt)
        child = subprocess.Popen(command, cwd=source, env=supervisor_environment(env), stdin=subprocess.DEVNULL)
        identity = proc_identity(child.pid)
        if identity is None or identity['uid'] != 1004:
            raise RuntimeError('Cannot identify spawned runner birth')
        owned.append(identity)
        receipt.update(state='running', supervisor=identity)
        atomic_json(receipt_path, receipt)
        while child.poll() is None:
            now = time.monotonic()
            try:
                state = read_json(data / 'runnerstatus.json')
                receipt['run_id'] = check_state(state, source, root, receipt['run_id'], identity)
                check_cost_scheduler_state(state, source, root)
                receipt['last_synchronized_epochs'] = max(receipt['last_synchronized_epochs'],
                                                          synchronized_epochs(data, receipt['run_id']))
                receipt.pop('observation_error', None)
            except FileNotFoundError:
                pass  # Native first-update/error watchdogs remain in force.
            except (sqlite3.Error, json.JSONDecodeError) as exc:
                receipt['observation_error'] = repr(exc)
            stop_evidence = read_stop_evidence(data, receipt['run_id'], state) if receipt['run_id'] else None
            external_flag = stop_evidence['flag'] if stop_evidence else None
            runner_failed = state.get('state') in ('stopping_error', 'failed') and bool(state.get('error'))
            calendar = getattr(args, 'stop_control', None) is not None
            if calendar:
                from stop_control import due
                measurement_due = due(receipt,args,epochs=receipt['last_synchronized_epochs'],
                                      monotonic_now=now,job_id=job_id)
            else:
                measurement_due = False  # No automatic training duration in this deployment.
            cause = (('exact_user_stop_flag' if stop_evidence['kind'] == 'manual'
                      else stop_evidence['kind']) if stop_evidence else
                     'runner_error_state' if runner_failed else
                     ('user_authorized_calendar_stop' if calendar else 'user_authorized_measurement_complete') if measurement_due else
                     stop_cause(signal_number=received[0] if received else None))
            if cause and stop_at is None:
                stop_at = now
                receipt.update(stop_trigger={'reason': cause, 'at_ns': time.time_ns(),
                                             'elapsed_seconds': now - start,
                                             'synchronized_epochs': receipt['last_synchronized_epochs']},
                               state='stop_requested')
                atomic_json(receipt_path, receipt)
                log(stop_trigger=receipt['stop_trigger'])
            if stop_at is not None:
                stop_limits = stop_deadlines(stop_evidence, state, now_ns=time.time_ns(),
                    now_monotonic=now, post_stop_grace_seconds=args.post_stop_grace_seconds,
                    prior=stop_limits)
                receipt['stop_observation'] = stop_evidence
                receipt['stop_deadlines'] = stop_limits
                receipt['runner_error'] = state.get('error')
            if (stop_at is not None and not external_flag and not receipt.get('stop_ack') and receipt['run_id']
                    and not runner_failed and state.get('state') in ('starting', 'running') and (data / 'training').is_dir()
                    and now - last_stop_attempt >= 5):
                last_stop_attempt = now
                try:
                    receipt['stop_ack'] = request_stop(source, data, receipt['run_id'], env)
                    receipt.pop('stop_request_error', None)
                    atomic_json(receipt_path, receipt)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    receipt['stop_request_error'] = repr(exc)
            if stop_limits is not None and now >= stop_limits['supervisor_deadline']:
                raise RuntimeError('Runner exceeded bounded stop/teardown deadline')
            if now >= next_log:
                receipt.update(elapsed_seconds=now - start, updated_ns=time.time_ns(),
                               runner_state=state.get('state', 'awaiting_startup'))
                atomic_json(receipt_path, receipt)
                log(state=receipt['state'], runner_state=receipt['runner_state'],
                    elapsed_seconds=round(now - start, 1), run_id=receipt['run_id'],
                    synchronized_epochs=receipt['last_synchronized_epochs'])
                next_log = now + 60
            time.sleep(1)
    except BaseException as exc:
        failure = repr(exc)
        receipt['wrapper_error'] = failure
        # A controller fault must not abandon a live trainer. Try the exact run's
        # established boundary stop, then keep the foreground alive for its save.
        if child is not None and child.poll() is None and receipt['run_id']:
            try:
                state = read_json(data / 'runnerstatus.json')
                check_state(state, source, root, receipt['run_id'], receipt.get('supervisor'))
                stop_evidence = read_stop_evidence(data, receipt['run_id'], state)
                runner_failed = state.get('state') in ('stopping_error', 'failed') and bool(state.get('error'))
                # An error flag belongs to the runner's shutdown protocol. Do
                # not call the manual-stop CLI or overwrite its original cause.
                if not stop_evidence and not runner_failed and not receipt.get('stop_ack'):
                    try:
                        receipt['stop_ack'] = request_stop(source, data, receipt['run_id'], env)
                    except (OSError, ValueError, subprocess.SubprocessError) as stop_exc:
                        # The runner can publish its error flag between our
                        # read and the manual-stop CLI's exclusive check. Its
                        # existing evidence wins; no overwrite or early kill.
                        receipt['stop_request_error'] = repr(stop_exc)
                        stop_evidence = read_stop_evidence(data, receipt['run_id'], state)
                        if stop_evidence is None:
                            raise
                    stop_evidence = read_stop_evidence(data, receipt['run_id'], state)
                now = time.monotonic()
                stop_limits = stop_deadlines(stop_evidence, state, now_ns=time.time_ns(),
                    now_monotonic=now, post_stop_grace_seconds=args.post_stop_grace_seconds,
                    prior=stop_limits)
                receipt['stop_observation'] = stop_evidence
                receipt['stop_deadlines'] = stop_limits
                receipt['runner_error'] = state.get('error')
                remaining = max(0, stop_limits['supervisor_deadline'] - now)
                child.wait(timeout=remaining)
            except Exception as stop_exc:
                receipt['emergency_stop_error'] = repr(stop_exc)
    finally:
        # Normal completion has already cleaned the training cohort. Keep this
        # parent alive while guard/sample exit; abnormal leftovers get exact
        # marker/UID/PID-birth cleanup only, never a process-group signal.
        until = time.monotonic() + 30
        remaining = marked_children(job_id, owned)
        while remaining and time.monotonic() < until and failure is None:
            time.sleep(1)
            remaining = marked_children(job_id, remaining)
        if remaining:
            receipt['emergency_cleanup'] = scoped_cleanup(job_id, remaining)
            failure = failure or 'runner left owned descendants after exit'
            remaining = receipt['emergency_cleanup']['survivors']
        if child is not None:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                failure = failure or 'runner still alive after scoped cleanup'
        receipt.update(finished_ns=time.time_ns(), elapsed_seconds=time.monotonic() - start,
                       runner_exit_code=child.returncode if child is not None else None,
                       all_owned_exited=not remaining)
        try:
            receipt['final_source_validation'] = verify_git(source, args.git_commit)
            state = read_json(data / 'runnerstatus.json')
            check_state(state, source, root, receipt['run_id'], receipt.get('supervisor'))
            receipt['runner_final_state'] = state.get('state')
            receipt['runner_error'] = state.get('error')
            receipt['checkpoint_cpu_audit'] = state.get('checkpoint_cpu_audit')
            receipt['runner_cleanup'] = state.get('cleanup')
            report = read_json(data / 'reportstatus.json')
            stop_evidence = read_stop_evidence(data, receipt['run_id'], state)
            stop_flag = stop_evidence['flag'] if stop_evidence else None
            receipt['stop_evidence'] = stop_flag
            receipt['stop_observation'] = stop_evidence
            receipt['acceptance'] = acceptance(state, report, receipt['runner_exit_code'], remaining, stop_flag)
        except Exception as exc:
            receipt['acceptance'] = {'passed': False, 'reasons': ['final_audit_unavailable'], 'error': repr(exc)}
        if failure:
            receipt['wrapper_error'] = failure
            receipt['acceptance']['passed'] = False
        receipt['state'] = 'succeeded' if receipt['acceptance']['passed'] else 'failed'
        atomic_json(receipt_path, receipt)
        log(final_receipt=str(receipt_path), state=receipt['state'], acceptance=receipt['acceptance'])
    return 0 if receipt['state'] == 'succeeded' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', '--experiment-root', dest='root', type=Path, required=True)
    parser.add_argument('--git-commit', required=True)
    parser.add_argument('--post-stop-grace-seconds', type=int, default=1800)
    parser.add_argument('--native-gpu-preflight', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    try:
        validate_manual_contract(args.post_stop_grace_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        return run(args)
    except Exception as exc:
        log(state='failed_before_launch', error=repr(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
