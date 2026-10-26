"""GPU4-7 scope adapter; archived learning/process-safety sources stay frozen."""
import importlib.util
import json
import os
import sqlite3
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parent
MASTER = Path('/mnt/data/lcx1/dong/need_data/cuda_ipc_weight_sync_gpu47_20261022')
DEADLINE_UNIX = None  # Manual stop; no environment/legacy wall deadline.
GPU_IDS = (4, 5, 6, 7)
BASE = Path('/mnt/data/lcx1/dong/need_data/archive/serial_split_verified_run_20261016.py')
spec = importlib.util.spec_from_file_location('gpu47_archived_safety', BASE)
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
safe = legacy.safe


def __getattr__(name):
    return getattr(legacy, name)


def mps_receipt():
    receipt = json.loads((MASTER / 'mps.json').read_text())
    assert receipt['scope'] == list(GPU_IDS)
    return receipt


def control(command):
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=mps_receipt()['pipe'])
    return legacy.command_output(['nvidia-cuda-mps-control'], env=env, stdin=command + '\n')


def mps_clients(pid):
    raw = control(f'get_client_list {int(pid)}')
    if raw and not all(x.isdigit() for x in raw.split()):
        raise RuntimeError('Invalid MPS client response: ' + raw[:200])
    return [int(x) for x in raw.split()]


def preflight():
    if os.getuid() != 1004:
        raise RuntimeError('Wrong UID')
    receipt = mps_receipt()
    for role in ['server', 'control']:
        current = safe.proc_identity(receipt[role]['pid'])
        if not safe.same_process(receipt[role], current) or current['uid'] != 1004 or not safe.protected(current):
            raise RuntimeError('MPS identity changed: ' + role)
    if control('get_server_list').split() != [str(receipt['server']['pid'])]:
        raise RuntimeError('Unexpected MPS server list')
    if mps_clients(receipt['server']['pid']):
        raise RuntimeError('MPS has existing clients; no job was stopped')
    try:
        default_cap = float(control('get_default_active_thread_percentage').strip())
        server_cap = float(control(f"get_active_thread_percentage {receipt['server']['pid']}").strip())
    except (TypeError, ValueError):
        raise RuntimeError('Unverified MPS active-thread percentages')
    if default_cap != 100.0 or server_cap != 100.0:
        raise RuntimeError('Existing MPS must have default/server100%; no setting was changed')
    cards = gpu_snapshot()
    if [(c['index'], c['uuid']) for c in cards] != [(c['index'], c['uuid']) for c in receipt['cards']]:
        raise RuntimeError('Physical GPU UUID mapping changed')
    selected = {c['uuid'] for c in cards}
    raw = legacy.command_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    for line in raw.splitlines():
        gpu, pid = [x.strip() for x in line.split(',')]
        if gpu in selected and int(pid) != receipt['server']['pid']:
            raise RuntimeError('GPU4-7 has an existing task; no job was stopped')
    if any(c['used_mib'] > 1024 or c['utilization_percent'] > 0 for c in cards):
        raise RuntimeError('GPU4-7 is not idle')
    return {'cards': cards, 'mps_pipe': receipt['pipe'], 'mps_log': receipt['log'],
            'protected_mps_services': [receipt['server']], 'existing_clients': [],
            'deadline_unix': None, 'until_manual_stop': True}


def cpu_helper_environment():
    """CPU-only guard/sample with no inherited stop cap or profiler injection."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    for key in list(env):
        if key.startswith(('ACCERL_', 'ACCE_CAPTURE_', 'NSYS_', 'CUDA_INJECTION', 'NVTX_INJECTION')):
            env.pop(key, None)
    if 'LD_PRELOAD' in env:
        parts = [item for item in env['LD_PRELOAD'].replace(':', ' ').split()
                 if not any(tag in item.lower() for tag in ('nsight', 'nsys', 'toolsinjection'))]
        if parts:
            env['LD_PRELOAD'] = ':'.join(parts)
        else:
            env.pop('LD_PRELOAD', None)
    for key in ('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT',
                'CUDA_MPS_CLIENT_PRIORITY', 'CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING',
                'CUDA_MPS_SM_PARTITION', 'SOFT_STOP_AT', 'MEASURE_STOP_AFTER_EPOCHS', 'SHORT_TEST_DEADLINE'):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def start_sampler(data, state, log):
    env = cpu_helper_environment()
    proc = subprocess.Popen([sys.executable, str(SOURCE / 'runner.py'), '--mode', 'sample',
                             '--data', str(data), '--run-id', state['run_id']], env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state['sample'] = safe.proc_identity(proc.pid)
    if not state['sample']:
        raise RuntimeError('Sampler identity missing')
    return proc


def sample(data, run_id):
    import cpu_sampler
    while not (data / 'training').is_dir():
        state = legacy.load_status(data)
        if state.get('run_id') != run_id or state.get('state') in cpu_sampler.TERMINAL:
            return
        if not safe.same_process(state.get('supervisor'), safe.proc_identity(state.get('supervisor', {}).get('pid', -1))):
            return
        time.sleep(.25)
    cpu_sampler.load_status = lambda _: legacy.load_status(data)
    cpu_sampler.safe.placement_evidence = placement_evidence
    return cpu_sampler.sample(data / 'training', run_id)


def gpu_snapshot():
    raw = legacy.command_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'])
    result = []
    for line in raw.splitlines():
        columns = [part.strip() for part in line.split(',')]
        if int(columns[0]) in (4, 5, 6, 7):
            result.append({'index': int(columns[0]), 'uuid': columns[1],
                           'total_mib': float(columns[2]), 'used_mib': float(columns[3]),
                           'utilization_percent': float(columns[4])})
    result.sort(key=lambda item: item['index'])
    if [item['index'] for item in result] != [4, 5, 6, 7] or len({item['uuid'] for item in result}) != 4:
        raise RuntimeError('Could not identify exactly physical GPUs 4, 5, 6 and 7')
    return result


def placement_evidence(connection, run_id):
    """Initialization identities for twelve independent trainers + twelve infer actors."""
    roles = {'policy_trainer', 'ctrl_trainer', 'reward_trainer',
             'policy_infer', 'ctrl_infer', 'reward_infer'}
    keep = {}
    for row in safe.query_events(connection, run_id, 'placement', limit=2000):
        details = row['details']; role = details.get('role', row['actor'])
        if role not in roles:
            continue
        mapping = details.get('physical_gpus', [])
        physical = mapping[0].get('physical_index') if len(mapping) == 1 else None
        actual = str(details.get('physical_gpu_uuid', '')).removeprefix('GPU-').lower()
        mapped = str(mapping[0].get('uuid', '')).removeprefix('GPU-').lower() if len(mapping) == 1 else ''
        keep[role, row['actor_id']] = {
            'role': role, 'actor_id': row['actor_id'], 'pid': row['pid'],
            'physical_gpu': physical, 'uuid': details.get('physical_gpu_uuid'),
            'uuid_confirmed': bool(actual and actual == mapped),
            'cpu_threads': details.get('cpu_threads'), 'model_roles': 1}
    actors = sorted(keep.values(), key=lambda item: (str(item['physical_gpu']), item['role']))
    cards = {str(gpu): {'processes': [item for item in actors if item['physical_gpu'] == gpu]}
             for gpu in (4, 5, 6, 7)}
    for card in cards.values():
        card['gpu_process_count'] = len(card['processes'])
        card['model_role_count'] = card['gpu_process_count']
    complete = (len(actors) == 24 and len({item['pid'] for item in actors}) == 24
                and all(item['uuid_confirmed'] and item['cpu_threads'] == 1 for item in actors)
                and all(len(card['processes']) == 6 and {item['role'] for item in card['processes']} == roles
                        for card in cards.values()))
    return {'actors': actors, 'cards': cards, 'observed_gpu_processes': len(actors),
            'observed_model_roles': len(actors), 'complete_expected_placement': complete,
            'scope': 'recorded actor UUID placement; current MPS/aliveness verified separately'}


def audit_sampled_gpu_peaks(data, state):
    connection = safe.sqlite_reader(legacy.training_data(data))
    if connection is None: return {}
    try:
        deadline = time.monotonic() + 60
        connection.set_progress_handler(lambda: time.monotonic() > deadline, 10000)
        row = connection.execute('SELECT COUNT(*),MIN(ts_ns),MAX(ts_ns) FROM system_samples WHERE run_id=?',
                                 (state['run_id'],)).fetchone()
        result = {'samples': row[0], 'first_ns': row[1], 'last_ns': row[2], 'gpus': {},
                  'scope': 'sampled physical-card peak, not an unsampled instantaneous maximum; inference and training share each card'}
        for offset, physical in enumerate((4, 5, 6, 7)):
            item = connection.execute(f'''SELECT MAX(json_extract(payload_json,'$.gpus[{offset}].memory_used_bytes')),
                MAX(json_extract(payload_json,'$.gpus[{offset}].gpu_busy_percent')) FROM system_samples
                WHERE run_id=? AND json_extract(payload_json,'$.gpus[{offset}].physical_gpu')=?''',
                (state['run_id'], physical)).fetchone()
            result['gpus'][str(physical)] = {'sampled_peak_used_bytes': item[0], 'sampled_peak_gpu_busy_percent': item[1]}
        state['sampled_gpu_peaks'] = result
        return result
    except sqlite3.Error as error:
        result = {'status': 'unresolved', 'error': repr(error), 'scope': 'final CPU resource-sample aggregation'}
        state['sampled_gpu_peaks'] = result
        return result
    finally:
        connection.close()


def audit_final_weights(data, state, *, torch_module=None, persist=True):
    """CPU mmap inventory at explicit user stop, including zero-update fill stop."""
    result={'valid':False,'at':legacy.stamp(),'map_location':'cpu','weights_only':True,'mmap':True,
            'optimizer_or_replay_restore_executed':False,'cuda_initialized':False}
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        if torch_module is None:
            import torch as torch_module
        if torch_module.cuda.is_initialized():
            raise RuntimeError('CPU verifier has an initialized CUDA context')
        directory=data/'training/checkpoints'
        external=json.loads((directory/'manifest.json').read_text())
        path=directory/external.get('weights_file','latest_weights.pt')
        payload=torch_module.load(str(path),map_location='cpu',weights_only=True,mmap=True)
        manifest=payload['manifest']
        if manifest!={key:value for key,value in external.items() if key not in ('weights_file','weights_bytes')}:
            raise ValueError('External and embedded manifests differ')
        expected={model:state.get('report',{}).get('training',{}).get(model,{}).get('global_updates',0) for model in ('policy','ctrl','reward')}
        reason=manifest.get('reason')
        accepted_reasons={'final'} if expected['policy'] else {'final_before_first_update','final'}
        policy=manifest.get('metadata',{}).get('stop_policy',{})
        if (manifest.get('metadata',{}).get('run_id')!=state['run_id'] or manifest.get('world_size')!=4
            or reason not in accepted_reasons or manifest.get('steps')!=expected or manifest.get('weights_only') is not True
            or set(payload.get('models',{}))!={'policy','ctrl','reward'}
            or policy.get('until_manual_stop') is not True or policy.get('until_deadline') is not False
            or policy.get('deadline_unix_seconds') is not None or policy.get('stop_on_target') is not False):
            raise ValueError('Manual final run/world/steps/stop-policy identity differs')
        counts={}
        for model,saved in payload['models'].items():
            tensors=saved.get('state_dict',{})
            if (not tensors or any(not torch_module.is_tensor(value) or value.device.type!='cpu' for value in tensors.values())
                or not isinstance(saved.get('trainable_names'),list) or not saved['trainable_names']
                or not set(saved['trainable_names'])<=set(tensors) or saved.get('full_state') is not (model=='reward')
                or len(tensors)!=manifest.get('tensor_counts',{}).get(model)):
                raise ValueError('Incomplete CPU tensor inventory: '+model)
            counts[model]=len(tensors)
        if external.get('weights_bytes')!=path.stat().st_size:
            raise ValueError('Checkpoint byte size differs')
        result.update(valid=True,path=str(path),steps=expected,world_size=4,tensor_counts=counts,
                      bytes=path.stat().st_size,manifest_reason=reason,stopped_before_first_update=expected['policy']==0)
        del payload
    except Exception as exc:
        result['error']=repr(exc)
    state['checkpoint_cpu_audit']=result
    if persist and (data/'training/collection.sqlite').is_file():
        connection=sqlite3.connect(data/'training/collection.sqlite',timeout=5)
        try:
            with connection:
                connection.execute('INSERT INTO events(run_id,ts_ns,actor,actor_id,pid,kind,status,details_json) VALUES(?,?,?,?,?,?,?,?)',
                    (state['run_id'],time.time_ns(),'cpu_supervisor',str(os.getpid()),os.getpid(),'manual_final_weights_cpu_audit',
                     'ok' if result['valid'] else 'error',json.dumps(result)))
        except sqlite3.Error as exc:result['persistence_error']=repr(exc)
        finally:connection.close()
    return result
