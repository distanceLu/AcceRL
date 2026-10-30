"""Finite eight-process DDP/IPC qualification, never an automatic retry.

The GPU branch deliberately uses a tiny synthetic model with the production
native engine and IPC code. It checks arithmetic, ownership and publication;
it does not establish real WEAVER loss, six-service memory or convergence.
The parent and --plan-only mode do not import Torch or initialize CUDA.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
import json
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

from weaver_subprocess import WeaverProcessClient, process_identity, same_process, receive_frame, send_frame

GPU_UUIDS = (
    'GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c',
    'GPU-fe0edfe4-b193-1af7-6a98-cbff40214f74',
    'GPU-8c9bc264-7f09-989e-1ec7-cecd6cde782c',
    'GPU-07ae0bbc-8923-bd0c-199e-c726821e655d',
)
TORCH_VERSION = '2.7'


def validate_plan(plan):
    if plan.get('protocol') != 'weaver_online_tiny_smoke_v1':
        raise ValueError('Wrong smoke plan protocol')
    if plan.get('expected_uid') != 1004 or plan.get('hostname') != 'h20-xyff52ab-0001':
        raise ValueError('Smoke is restricted to the authorized UID and host')
    if tuple(plan.get('gpu_uuids', ())) != GPU_UUIDS:
        raise ValueError('Smoke only admits the exact physical GPU4-7 UUIDs')
    if not re.fullmatch(r'weaverintegrated_(?:smoke|preflight)_[a-zA-Z0-9_]+', str(plan.get('run_id', ''))):
        raise ValueError('Smoke needs its own explicit run marker')
    if not 90 <= plan.get('max_seconds', 0) <= 300:
        raise ValueError('Smoke requires a finite 90-300 second total budget')
    if not PurePosixPath(plan.get('python', '')).is_absolute() or not PurePosixPath(plan.get('output', '')).is_absolute():
        raise ValueError('Smoke Python and new output directory must be absolute')
    mps = plan.get('mps', {})
    if mps.get('pipe') != '/dev/shm/mps_lcx1_gpu47_pzero2_20261020/pipe':
        raise ValueError('Smoke must reuse the authorized existing MPS pipe')
    for key in ('server', 'control'):
        identity = mps.get(key, {})
        if identity.get('uid') != 1004 or not all(type(identity.get(field)) is int and identity[field] > 0
                                               for field in ('pid', 'birth')):
            raise ValueError('MPS exact UID/PID/birth is required')
    return {'validated': True, 'gpu_started': False, 'torch_imported': False,
            'world_size': 4, 'gpu_processes': 8, 'optimizer_updates': 2,
            'publications': [0, 1, 2], 'micro_batch': 1, 'gas': 2, 'global_batch': 8,
            'limits': 'Synthetic engine/IPC qualification; not native WEAVER or memory validation'}


def _control(plan, command, timeout=5):
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=plan['mps']['pipe'])
    result = subprocess.run(['nvidia-cuda-mps-control'], input=command + '\n', text=True,
                            capture_output=True, env=env, timeout=timeout, check=True)
    return result.stdout.strip()


def check_host(plan, *, require_idle=True,deadline=None):
    def budget():
        remaining=5. if deadline is None else min(5.,deadline-time.monotonic())
        if remaining<=0:
            raise TimeoutError('Read-only GPU/MPS verification reached the existing smoke deadline')
        return remaining
    if os.getuid() != plan['expected_uid'] or socket.gethostname() != plan['hostname']:
        raise RuntimeError('Smoke actual UID/host differs')
    for key in ('server', 'control'):
        if not same_process(plan['mps'][key]):
            raise RuntimeError('MPS ' + key + ' PID/birth/UID changed')
    server = plan['mps']['server']['pid']
    if str(server) not in _control(plan, 'get_server_list',budget()).split():
        raise RuntimeError('Authorized MPS server is absent')
    default = float(_control(plan, 'get_default_active_thread_percentage',budget()))
    active = float(_control(plan, 'get_active_thread_percentage %d' % server,budget()))
    if (default, active) != (100., 100.):
        raise RuntimeError('Existing MPS is not unlimited; smoke will not modify it')
    clients = _control(plan, 'get_client_list %d' % server,budget()).split()
    if require_idle and clients:
        raise RuntimeError('GPU training/evaluation clients remain; do not overlap the smoke')
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=budget(), check=True).stdout
    for row in query.splitlines():
        pid, gpu = (field.strip() for field in row.split(',', 1))
        if gpu in GPU_UUIDS and int(pid) != server:
            raise RuntimeError('Unknown compute process occupies a smoke GPU')
    return {'mps_server': process_identity(server), 'clients': clients, 'default_cap': default, 'server_cap': active}


def wait_postflight(plan,deadline):
    """MPS unregister is asynchronous; only bounded read-only polling follows."""
    consecutive=0
    observations=[]
    while time.monotonic()<deadline:
        try:
            observation=check_host(plan,deadline=deadline)
            consecutive+=1
            observations.append({'idle':True,'time_ns':time.time_ns()})
            if consecutive==2:
                return dict(observation,consecutive_idle_checks=2,observations=observations)
        except RuntimeError as error:
            if not str(error).startswith(('GPU training/evaluation clients remain','Unknown compute process occupies')):
                raise
            consecutive=0
            observations.append({'idle':False,'reason':str(error),'time_ns':time.time_ns()})
        time.sleep(min(.25,max(0.,deadline-time.monotonic())))
    raise TimeoutError('Owned children exited but MPS/GPU did not become idle inside the original smoke budget')


class MemorySpan:
    def __init__(self, writer, name, details):
        self.writer, self.name, self.details = writer, name, details
    def __enter__(self):
        return self
    def __exit__(self, kind, value, tb):
        self.writer.event('span', name=self.name, details=dict(self.details), error=str(value) if value else None)


class MemoryWriter:
    def __init__(self):
        self.events = []
    def span(self, name, **details):
        return MemorySpan(self, name, details)
    def event(self, kind, **values):
        self.events.append({'kind': kind, **values})
    def replay_used(self, replay, metadata, **values):
        self.event('replay_use', replay=replay, reservation_id=metadata['reservation_id'], **values)
    def flush(self, *args, **kwargs):
        pass


def tiny_backend(torch, device):
    class Reward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net=torch.nn.Linear(2,1)
            with torch.no_grad():
                self.net.weight.fill_(.125)
                self.net.bias.fill_(-.25)
        def forward(self,obs,actions,tasks):
            return self.net(obs['features']).squeeze(-1)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wm = torch.nn.Linear(2, 2)
            self.rm=Reward()
            with torch.no_grad():
                self.wm.weight.fill_(.25)
                self.wm.bias.fill_(.125)
        def forward(self, prepared):
            x, target,success = prepared
            predicted=self.wm(x).float()
            flow=(predicted-target.float()).square().mean()
            features={'features':torch.cat([x.detach(),predicted.detach()])}
            _,rm_loss=self.rm.compute_loss(features,None,None,torch.cat([success,success]))
            return flow+rm_loss,{'synthetic_flow_probe':flow,'native_half_brier_probe':rm_loss}
    class Backend:
        def __init__(self):
            self.model = Model().to(device)
            self.upstream_model=self.model
            from weaver_native_reward import install_native_reward_head
            self.native_reward_controller=install_native_reward_head(self.model,reward_specification())
        def prepare_train_batch(self, windows):
            return (torch.stack([torch.as_tensor(row['obs'], device=device).float() for row in windows]),
                    torch.stack([torch.as_tensor(row['target'], device=device).float() for row in windows]),
                    torch.tensor([row['success'] for row in windows],device=device,dtype=torch.float32))
        def publication_inventory(self):
            return {'parameter_names': sorted(name for name, _ in self.model.named_parameters()),
                    'buffer_names': [], 'backend_identity': {'backend': 'weaver', 'synthetic_probe': True},
                    'bundle_metadata': {'native_reward_head': reward_specification()}}
    return Backend()


def reward_specification():
    return {'mode':'libero_success_score_v1','link':'sigmoid','loss':'brier',
            'label_source':'libero_env_step','calibrated':False,'terminal':'horizon_only',
            'balance':'global_cohort_inverse_frequency','loss_coefficient':1.0}


def synthetic_reward_cohort(update):
    from weaver_native_reward import COHORT_PROTOCOL
    positive=1 if update==1 else 0
    return {'protocol':COHORT_PROTOCOL,'window_count':8,'frame_slice':[0,1],
            'label_source':'libero_env_step','positive_count':positive,'negative_count':8-positive,
            'positive_weight':7. if positive else 0.,'negative_weight':1.,'rm_active':bool(positive)}


def synthetic_row(rank, micro, update):
    return {'obs': [.25 * (rank + 1), .25 * (micro + 1)],
            'target': [.125 * (rank + micro + 1), .125 * update],
            'success':bool(update==1 and rank==0 and micro==1)}


def compare_reference(torch, engine, reference, optimizer, update):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = 1e-5 * update / 20
    rows = [synthetic_row(rank, micro, update) for rank in range(4) for micro in range(2)]
    reference.native_reward_controller.set_cohort(synthetic_reward_cohort(update))
    with torch.autocast(engine.device.type, dtype=torch.bfloat16):
        loss = reference.model(reference.prepare_train_batch(rows))[0]
    loss.backward()
    if not reference.native_reward_controller.active:
        for parameter in reference.native_reward_controller.parameters:
            parameter.grad=None
    norm = torch.nn.utils.clip_grad_norm_(reference.model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    errors = {'weight_max_abs': 0., 'adam_max_abs': 0.,
              'grad_norm_abs': abs(float(norm) - engine.last_result['grad_norm'])}
    names=[name for name,parameter in engine.module.named_parameters() if parameter.requires_grad]
    for name,actual, expected in zip(names,engine.parameters, reference.model.parameters()):
        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=1e-6)
        errors['weight_max_abs'] = max(errors['weight_max_abs'], float((actual - expected).detach().abs().max()))
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            value, wanted = engine.optimizer.state[actual][key], optimizer.state[expected][key]
            # BF16 linear backward rounds each rank/micro partial, whereas the
            # global reference rounds one whole matrix reduction. Native RM's
            # sigmoid gradients are not powers of two; retain tight WM checks
            # and explicitly bound this expected RM reduction difference.
            tolerance=({'atol':0.,'rtol':0.} if key=='step' else
                       {'atol':5e-5,'rtol':.02} if name.startswith('rm.') else
                       {'atol':2e-6,'rtol':2e-5})
            torch.testing.assert_close(value, wanted, **tolerance)
            errors['adam_max_abs'] = max(errors['adam_max_abs'], float((value - wanted).abs().max()))
    errors['grad_norm_tolerance']=max(2e-5,.01*float(norm))
    errors['rm_adam_bf16_tolerance']={'atol':5e-5,'rtol':.02}
    if errors['grad_norm_abs'] > errors['grad_norm_tolerance']:
        raise AssertionError('DDP accumulated gradient differs from the global-batch reference')
    from training_topology import _weight_hash
    errors['reference_parameter_hash'] = _weight_hash(sorted(reference.model.named_parameters()), 'ctrl')
    errors['actual_parameter_hash'] = _weight_hash(sorted(engine.module.named_parameters()), 'ctrl')
    return errors


def smoke_worker(role, rank):
    from weaver_gpu_worker import arm_parent_guard, canonical_uuid
    arm_parent_guard()
    run_id = os.environ['ACCERL_RUN_ID']
    if os.environ.get('WEAVER_GPU_WORKER_ID') != run_id + ':weaver:' + str(rank):
        raise RuntimeError('Smoke worker marker changed')
    output = sys.stdout.buffer
    sys.stdout = sys.stderr
    backend = engine = exporter = reference = optimizer = None
    writer = MemoryWriter()
    raw = None
    try:
        while True:
            try:
                message = receive_frame(sys.stdin.buffer)
            except EOFError:
                return
            try:
                method, values = message['method'], message['args']
                if method == 'load':
                    raw = values[0]
                    result = {'worker_identity': process_identity(os.getpid()), 'geometry': {}}
                elif method == 'setup':
                    import torch
                    if not str(torch.__version__).startswith(TORCH_VERSION + '.'):
                        raise RuntimeError('Smoke must exercise the actual WEAVER Torch2.7 environment')
                    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUIDS[rank] or torch.cuda.device_count() != 1:
                        raise RuntimeError('Smoke worker physical GPU mask changed')
                    torch.set_num_threads(1)
                    torch.cuda.set_device(0)
                    actual_uuid = canonical_uuid(torch.cuda.get_device_properties(0).uuid)
                    if actual_uuid != GPU_UUIDS[rank]:
                        raise RuntimeError('Smoke actual CUDA UUID changed')
                    backend = tiny_backend(torch, 'cuda:0')
                    if role == 'trainer':
                        from torch.nn.parallel import DistributedDataParallel
                        torch.distributed.init_process_group('nccl', init_method=values[0], rank=rank,
                            world_size=4, timeout=timedelta(seconds=60))
                        from weaver_online_engine import WeaverOnlineEngine
                        engine = WeaverOnlineEngine(backend, model=DistributedDataParallel(backend.model,
                            device_ids=[0], broadcast_buffers=False), rank=rank, world_size=4,
                            micro_batch=1, accumulation_steps=2)
                        reference = tiny_backend(torch, 'cuda:0')
                        optimizer = torch.optim.AdamW(reference.model.parameters(), lr=1e-5,
                            betas=(.9, .99), weight_decay=.01)
                        from cuda_ipc_weights import CudaIPCWeightExporter
                        exporter = CudaIPCWeightExporter(backend.model, 'ctrl', actual_uuid, run_id,
                            rank, world_backend=backend)
                        engine.publication_idle = lambda: not exporter.active
                    result = {'worker_identity': process_identity(os.getpid()), 'rank': rank,
                              'role': role, 'uuid': actual_uuid, 'torch': torch.__version__, 'run_id': run_id}
                elif method == 'update' and engine is not None:
                    update = values[0]
                    for policy in range(engine.policy_step + 1, update * 5 + 1):
                        engine.advance_policy(policy)
                    engine.begin_update(update * 5,synthetic_reward_cohort(update))
                    owner = SimpleNamespace(telemetry=writer, rank=rank, world_size=4,
                        training_role='ctrl', global_step=update * 5, capture_versions={'ctrl': update-1},
                        capture_wm_metadata={'reservation_id': 'smoke-%d' % update,
                                             'sample_ids': ['u%d-r%d-m%d' % (update, rank, m) for m in range(2)]})
                    from capture_training import micro_begin, micro_end
                    for micro in range(2):
                        token = micro_begin(owner, 'ctrl_train', engine, 1, micro, micro+1)
                        engine.micro([synthetic_row(rank, micro, update)])
                        micro_end(owner, token, engine)
                    comparisons = compare_reference(torch, engine, reference, optimizer, update)
                    result = {'state': engine.state(), 'comparisons': comparisons,
                              'events': list(writer.events)}
                elif method == 'export' and exporter is not None:
                    result = exporter.begin(engine.global_steps)
                elif method == 'apply' and role == 'inference':
                    from cuda_ipc_weights import apply_cuda_ipc_weights
                    result = apply_cuda_ipc_weights(backend.model, values[0], 'ctrl', GPU_UUIDS[rank],
                                                   run_id, rank, world_backend=backend)
                elif method == 'release' and exporter is not None:
                    result = exporter.finish(*values)
                    if exporter.active or not engine.state()['gradients_released']:
                        raise RuntimeError('Publication or gradients remain held')
                    torch.cuda.empty_cache()
                elif method == 'infer':
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        prediction = backend.model.wm(torch.tensor([[.25, .5]], device='cuda')).float().cpu().tolist()
                    from training_topology import _publication_state, _weight_hash
                    selected, publication = _publication_state(backend.model, 'ctrl', values[0], backend)
                    result = {'served_version': values[0], 'prediction': prediction,
                              'hash': _weight_hash(selected, 'ctrl', publication),
                              'raw_parameter_hash': _weight_hash(selected, 'ctrl'),
                              'rm_parameter_hash':_weight_hash([(name,value) for name,value in selected if name.startswith('rm.')],'ctrl')}
                else:
                    raise ValueError('Unknown smoke RPC: ' + str(method))
                send_frame(output, {'ok': True, 'result': result})
            except BaseException as error:
                send_frame(output, {'ok': False, 'error': type(error).__name__ + ': ' + str(error),
                                    'traceback': traceback.format_exc()})
                return
    finally:
        if engine is not None:
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


def run(plan):
    validate_plan(plan)
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    (output/'plan.json').write_text(json.dumps(plan, indent=2))
    started = time.time_ns()
    # Reserve cleanup and read-only postflight inside the absolute outer budget.
    absolute_deadline=time.monotonic()+plan['max_seconds']
    deadline = absolute_deadline-45
    clients = []
    report = {'started_ns': started, 'run_id': plan['run_id'], 'gpu_started': False,
              'scope': 'finite synthetic engine/DDP/IPC; not native full-WEAVER qualification'}
    report['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('weaver_online_smoke.py','weaver_online_engine.py','weaver_native_reward.py','cuda_ipc_weights.py',
                     'training_topology.py','weaver_subprocess.py')}
    old_run = os.environ.get('ACCERL_RUN_ID')
    os.environ['ACCERL_RUN_ID'] = plan['run_id']
    timer = None
    cleanup_lock = threading.Lock()
    cleanup_errors = []
    def close_one(client):
        try:
            client.close()
        except BaseException:
            cleanup_errors.append({'identity':client.identity,'traceback':traceback.format_exc()})
    def close_all():
        with cleanup_lock:
            with ThreadPoolExecutor(max_workers=max(1,len(clients))) as pool:
                list(pool.map(close_one,list(clients)))
    def remaining():
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise TimeoutError('Smoke absolute budget exhausted; no retry')
        return min(120., budget)
    def calls(targets, method, *values):
        with ThreadPoolExecutor(max_workers=len(targets)) as pool:
            futures = [pool.submit(client.call, method, *values, timeout=remaining()) for client in targets]
            return [future.result(timeout=remaining()+1) for future in futures]
    try:
        report['preflight'] = check_host(plan,deadline=deadline)
        timer = threading.Timer(max(.1, deadline-time.monotonic()),close_all)
        timer.daemon = True
        timer.start()
        for role in ('trainer', 'inference'):
            for rank, gpu in enumerate(GPU_UUIDS):
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, ACCERL_EXPECTED_GPU_UUID=gpu,
                    ACCERL_PHYSICAL_GPU=str(rank+4), ACCERL_USE_MPS='1',
                    CUDA_MPS_PIPE_DIRECTORY=plan['mps']['pipe'],
                    CUDA_MPS_LOG_DIRECTORY=str(Path(plan['mps']['pipe']).parent/'log'),
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                environment.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', None)
                client = WeaverProcessClient({'backend_options': {'inference_python': plan['python'],
                    'rpc_timeout_seconds': min(120, plan['max_seconds']), 'load_timeout_seconds':remaining()}}, rank,
                    command=[plan['python'], '-u', str(Path(__file__).resolve()), '--worker-role', role,
                             '--rank', str(rank)], environment=environment)
                clients.append(client)
        with socket.socket() as port_socket:
            port_socket.bind(('127.0.0.1', 0))
            address = 'tcp://127.0.0.1:%d' % port_socket.getsockname()[1]
        trainers, receivers = clients[:4], clients[4:]
        report['identities'] = calls(trainers, 'setup', address) + calls(receivers, 'setup', address)
        report['child_identities'] = [{**row['worker_identity'], 'rank': row['rank'], 'role': row['role'],
                                        'gpu_uuid': row['uuid']} for row in report['identities']]
        report['expected_children'] = 8
        report['gpu_started'] = True
        report['updates'], report['publications'] = [], []
        for version in range(3):
            if version:
                report['updates'].append(calls(trainers, 'update', version))
            packets = calls(trainers, 'export')
            if len({packet['inventory']['hash'] for packet in packets}) != 1:
                raise RuntimeError('Four World ranks have different complete weight hashes')
            acks = [client.call('apply', packet, timeout=remaining())
                    for client, packet in zip(receivers, packets)]
            served = calls(receivers, 'infer', version)
            releases = [client.call('release', packet['token'], ack, timeout=remaining())
                        for client, packet, ack in zip(trainers, packets, acks)]
            if any(row['hash'] != packets[0]['inventory']['hash'] for row in served):
                raise RuntimeError('Inference served weights differ from acknowledged source weights')
            report['publications'].append({'version': version, 'inventories': [p['inventory'] for p in packets],
                                           'acks': acks, 'released': releases, 'served': served})
        hashes = [row['served'][0]['raw_parameter_hash'] for row in report['publications']]
        if len(set(hashes)) != 3:
            raise RuntimeError('Synthetic optimizer did not change both successive publication weights')
        report['passed'] = True
        report['successful_updates_per_rank'] = 2
        if any(row['state']['rm_updates']!=1 or row['state']['rm_disabled_updates']!=1
               for row in report['updates'][-1]):
            raise RuntimeError('Synthetic missing-class cohort incorrectly counted a native head update')
        if (report['publications'][1]['served'][0]['rm_parameter_hash']!=
                report['publications'][2]['served'][0]['rm_parameter_hash']):
            raise RuntimeError('Native RM weights changed despite the missing-class head update skip')
        if any(first['rm_parameter_hash']==second['rm_parameter_hash'] for first,second in zip(
                report['publications'][0]['served'],report['publications'][1]['served'])):
            raise RuntimeError('Active RM update did not change and publish each rank-local native head')
        report['successful_rm_updates_per_rank']=1
        report['missing_class_kept_rm_weights']=True
        report['rm_weights_changed_on_active_update']=True
    except BaseException:
        report['passed'] = False
        report['failure'] = traceback.format_exc()
    finally:
        if timer is not None:
            timer.cancel()
        close_all()
        report['cleanup'] = {'all_owned_exited': all(not same_process(client.identity) for client in clients),
                             'identities': [client.identity for client in clients], 'errors':cleanup_errors}
        try:
            report['postflight'] = wait_postflight(plan,min(absolute_deadline,time.monotonic()+20))
        except BaseException:
            report['postflight_failure'] = traceback.format_exc()
            report['passed'] = False
        report['passed'] = bool(report.get('passed') and report['cleanup']['all_owned_exited'])
        report['ended_ns'] = time.time_ns()
        (output/'completed.json').write_text(json.dumps(report, indent=2, default=str))
        (output/'report.json').write_text(json.dumps(report, indent=2, default=str))
        if old_run is None:
            os.environ.pop('ACCERL_RUN_ID', None)
        else:
            os.environ['ACCERL_RUN_ID'] = old_run
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker-role', choices=['trainer', 'inference'])
    parser.add_argument('--rank', type=int)
    parser.add_argument('--output-root')
    parser.add_argument('--run-id')
    parser.add_argument('--max-seconds', type=int, default=180)
    args = parser.parse_args()
    if args.worker_role:
        smoke_worker(args.worker_role, args.rank)
        return
    plan = json.loads(Path(args.plan).read_text()) if args.plan else {
        'protocol': 'weaver_online_tiny_smoke_v1', 'run_id': args.run_id,
        'expected_uid': 1004, 'hostname': 'h20-xyff52ab-0001',
        'python': sys.executable, 'output': args.output_root,
        'max_seconds': args.max_seconds, 'gpu_uuids': list(GPU_UUIDS),
        'mps': {'pipe': '/dev/shm/mps_lcx1_gpu47_pzero2_20261020/pipe',
                'server': {'pid': 691086, 'birth': 361065122, 'uid': 1004},
                'control': {'pid': 690850, 'birth': 361065009, 'uid': 1004}}}
    result = validate_plan(plan) if args.plan_only else run(plan)
    print(json.dumps(result, indent=2))
    if not args.plan_only and not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
