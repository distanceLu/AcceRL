"""Queue-owned supervisor for the legacy trainer; no WEAVER runner contracts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

try:
    from .legacy_baseline import RAY_TEMP_BASE, make_record, match_record, match_job, require, validate_source
except ImportError:
    from legacy_baseline import RAY_TEMP_BASE, make_record, match_record, match_job, require, validate_source


def atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as handle:
        temporary = Path(handle.name)
        handle.write((json.dumps(value,indent=2)+'\n').encode())
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_bytes())


def audit_weight_tensors(directory):
    """Load only checkpoint weights on CPU and reject corrupt/non-finite tensors."""
    import torch
    summaries = []

    def inspect(value, name):
        if isinstance(value,torch.Tensor):
            require(value.device.type == 'cpu', 'Checkpoint tensor was not loaded onto CPU')
            require(value.numel() > 0, 'Empty checkpoint tensor: '+name)
            if value.is_floating_point() or value.is_complex():
                flat = value.reshape(-1)
                require(all(bool(torch.isfinite(part).all()) for part in flat.split(1048576)),
                        'Non-finite checkpoint tensor: '+name)
            summaries.append(dict(name=name,shape=list(value.shape),dtype=str(value.dtype)))
        elif isinstance(value,dict):
            for key, item in value.items():
                inspect(item,name+'.'+str(key))
        elif isinstance(value,(tuple,list)):
            for index, item in enumerate(value):
                inspect(item,name+'.'+str(index))

    directory = Path(directory)
    lora = directory/'agent_lora'
    if (lora/'adapter_model.safetensors').is_file():
        from safetensors import safe_open
        with safe_open(str(lora/'adapter_model.safetensors'),framework='pt',device='cpu') as handle:
            for key in handle.keys():
                inspect(handle.get_tensor(key),'lora.'+key)
    else:
        inspect(torch.load(lora/'adapter_model.bin',map_location='cpu',weights_only=True),'lora')
    require(summaries,'No LoRA tensors in checkpoint')
    lora_count = len(summaries)
    extra = torch.load(directory/'agent_extra_layers.pt',map_location='cpu',weights_only=True)
    require(isinstance(extra,dict) and {'value_head','attn_pool','lm_head'} <= set(extra),
            'Checkpoint extra-layer state is incomplete')
    inspect(extra,'extra')
    require(len(summaries) > lora_count,'No extra-layer tensors in checkpoint')
    return dict(device='cpu',weights_only=True,finite=True,tensor_count=len(summaries),tensors=summaries)


def audit_completion(record, *, weights_auditor=None):
    root = Path(record['root'])
    result = read(root/'training/driver_complete.json')
    require(result.get('schema_version') == 1 and result.get('status') == 'completed',
            'Legacy driver did not report completed')
    require(result.get('run_id') == record['run_id'] and result.get('git_commit') == record['git_commit'],
            'Legacy driver completion provenance differs')
    require(type(result.get('global_step')) is int and result['global_step'] > 0
            and type(result.get('completed_updates')) is int and result['completed_updates'] > 0,
            'Legacy driver has no completed update')
    require(result.get('reason') in ('deadline','stop_file','train_iters')
            and result.get('checkpoint_kind') == 'inference_weights_only'
            and result.get('exact_resume') is False,'Legacy final boundary/save contract differs')
    metadata = read(root/'training/driver_metadata.json')
    require(metadata.get('run_id') == record['run_id'] and metadata.get('git_commit') == record['git_commit']
            and metadata.get('global_batch_size') == 576,'Legacy metadata identity/global batch differs')
    placements = metadata.get('placements',[])
    require(len(placements) == 4 and {(p.get('role'),p.get('index')) for p in placements} ==
            {('trainer',0),('trainer',1),('trainer',2),('inference',0)}
            and len({p.get('pid') for p in placements}) == 4,
            'Legacy recorded actor placement differs')
    updates = [json.loads(line) for line in (root/'training/updates.jsonl').read_text().splitlines() if line]
    require(len(updates) == result['completed_updates'] and updates[-1].get('global_step') == result['global_step']
            and all(row.get('run_id') == record['run_id'] and row.get('git_commit') == record['git_commit']
                    and row.get('samples_per_update') == 576 for row in updates)
            and result.get('consumed_training_samples') == 576*len(updates),
            'Legacy update ledger disagrees with final boundary')
    directory = Path(result['checkpoint_dir'])
    expected = root/'training/checkpoints/latest'/('agent_checkpoint_epoch_'+str(result['global_step']))
    require(directory == expected and directory.resolve() == expected and directory.is_dir(),
            'Legacy checkpoint escaped the frozen output/step')
    paths = list(directory.rglob('*'))
    require(all(not path.is_symlink() and path.resolve().is_relative_to(directory) for path in paths),
            'Legacy checkpoint contains redirected files')
    files = [path for path in paths if path.is_file()]
    require(files and all(path.stat().st_size > 0 for path in files),'Empty legacy checkpoint artifact')
    require((directory/'agent_extra_layers.pt').is_file()
            and (directory/'agent_lora/adapter_config.json').is_file()
            and any((directory/'agent_lora'/name).is_file()
                    for name in ('adapter_model.safetensors','adapter_model.bin')),
            'Legacy checkpoint is missing LoRA/config/extra-layer weights')
    tensor_audit = (weights_auditor or audit_weight_tensors)(directory)
    audited = []
    for path in sorted(files):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(4*1024*1024),b''):
                digest.update(chunk)
        audited.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=digest.hexdigest()))
    return dict(passed=True,driver=result,files=audited,tensor_audit=tensor_audit,
                scope='CPU tensor finite audit and SHA256; weights-only, not exact resume')


def supervise(record, env, safety, *, popen=subprocess.Popen, clock=time.monotonic,
              sleep=time.sleep, startup_seconds=1800, grace_seconds=1200):
    """Injectable CPU lifecycle; signals only exact queue-marked UID/PID/birth owners."""
    root = Path(record['root'])
    job_id = env['EXPERIMENT_QUEUE_JOB_ID']
    receipt = dict(schema=1,state='starting',job_id=job_id,run_id=record['run_id'],
        git_commit=record['git_commit'],source_validation={'git_commit':record['git_commit']},
        started_ns=time.time_ns(),argv=record['driver_argv'],startup_limit_seconds=startup_seconds,
        training_budget_seconds=record['training_seconds'],boundary_save_grace_seconds=grace_seconds,
        all_owned_exited=False)
    child = None
    owned = []
    started = clock()
    training_started = None
    stop_requested = None
    received = []
    old_handlers = {}
    try:
        for signum in (signal.SIGTERM,signal.SIGINT):
            old_handlers[signum] = signal.signal(signum,lambda number,frame:received.append(number))
        atomic(root/'queue_receipt.json',receipt)
        with (root/'train.log').open('xb') as log:
            child = popen(record['driver_argv'],cwd=root,env=env,stdin=subprocess.DEVNULL,
                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity = safety.proc_identity(child.pid)
            uid_matches = identity and (not hasattr(os,'getuid') or identity['uid'] == os.getuid())
            require(uid_matches,
                    'Cannot identify legacy driver UID/PID/birth')
            owned.append(identity)
            receipt.update(supervisor=identity,state='running')
            atomic(root/'queue_receipt.json',receipt)
            while child.poll() is None:
                now = clock()
                try:
                    state = read(root/'training/driver_status.json')
                except FileNotFoundError:
                    state = {}
                if state:
                    require(state.get('run_id') == record['run_id'] and state.get('git_commit') == record['git_commit'],
                            'Legacy driver status provenance differs')
                if training_started is None and state.get('training_started_unix') is not None:
                    training_started = now
                    receipt['training_observed_ns'] = time.time_ns()
                    atomic(root/'queue_receipt.json',receipt)
                if received and stop_requested is None:
                    stop = root/'training/stop.flag'
                    stop.parent.mkdir(exist_ok=True)
                    stop.write_text('queue supervisor signal '+str(received[0])+'\n')
                    stop_requested = now
                    receipt['requested_stop_signal'] = received[0]
                    atomic(root/'queue_receipt.json',receipt)
                require(training_started is not None or now-started <= startup_seconds,
                        'Legacy startup/replay-fill exceeded its fault bound')
                require(training_started is None or now-training_started <= record['training_seconds']+grace_seconds,
                        'Legacy training boundary/final save exceeded its fault bound')
                require(stop_requested is None or now-stop_requested <= grace_seconds,
                        'Legacy requested boundary stop/final save exceeded its grace')
                sleep(1)
            code = child.wait(timeout=2)
            receipt['exit_code'] = code
            require(code == 0,'Legacy driver exited nonzero: '+str(code))
        # Ray normally exits during driver shutdown; collect only this queue's owners.
        until = clock()+15
        while safety.marked_children(job_id,owned) and clock() < until:
            sleep(.5)
        cleanup = safety.scoped_cleanup(job_id,owned)
        receipt['cleanup'] = cleanup
        require(not cleanup['survivors'],'Legacy owned processes survived cleanup')
        receipt['checkpoint_audit'] = audit_completion(record)
        receipt.update(state='succeeded',all_owned_exited=True)
    except BaseException as exc:
        receipt.update(state='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        try:
            if receipt['state'] != 'succeeded':
                receipt['cleanup'] = safety.scoped_cleanup(job_id,owned)
            survivors = safety.marked_children(job_id,owned)
            receipt['all_owned_exited'] = not survivors
            if survivors:
                receipt.update(state='failed',survivors=survivors)
                receipt.setdefault('error','Legacy owned processes remain after cleanup')
        except BaseException as exc:
            receipt.update(state='failed',all_owned_exited=False,cleanup_error=repr(exc))
        if child is not None and child.poll() is not None:
            child.wait(timeout=2)
        for signum, handler in old_handlers.items():
            signal.signal(signum,handler)
        receipt['finished_ns'] = time.time_ns()
        atomic(root/'queue_receipt.json',receipt)
    return receipt


def launch_environment(record, authorization):
    root = Path(record['root'])
    checkout = Path(record['checkout'])
    libero = os.environ.get('WEAVER_LIBERO_ROOT')
    env = {k:v for k,v in os.environ.items() if not k.startswith(
        ('ACCERL_','ACCE_','WEAVER_','CUDA_','NSYS_','PYTHON','RAY_')) and k != 'LD_PRELOAD'}
    paths = [str(checkout)]
    if libero:
        paths.append(libero)
    env.update(CUDA_VISIBLE_DEVICES=','.join(record['configuration']['cuda_visible_gpu_uuids']),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_MPS_PIPE_DIRECTORY=authorization['pipe'],
        CUDA_MPS_LOG_DIRECTORY=str(Path(authorization['pipe']).parent/'log'),
        MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(paths),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',RAY_DEDUP_LOGS='0',
        ACCERL_RUN_ID=record['run_id'],WEAVER_LEGACY_GIT_COMMIT=record['git_commit'],
        TMPDIR=str(root/'tmp'))
    return env


def run(args):
    """Production admission; called only inside its exact running queue reservation."""
    require(args.checkout.resolve() == args.checkout and args.root.resolve() == args.root,
            'Canonical legacy checkout/output required')
    require(args.checkout == Path(__file__).resolve().parents[2], 'Legacy wrapper source identity differs')
    expected = make_record(args.checkout,args.root,args.root.name,args.git_commit,args.training_seconds)
    match_record(read(args.root/'experiment.json'),expected)
    validation = validate_source(args.checkout,args.git_commit)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import cli
    row = cli.lookup(expected['job']['submission_id'])
    require(row and row['id'] == os.environ.get('EXPERIMENT_QUEUE_JOB_ID') and row['state'] == 'running',
            'Legacy entry requires its exact admitted queue identity')
    match_job(expected,row)
    for relative in ('training','tmp','train.log','queue_receipt.json'):
        require(not os.path.lexists(args.root/relative),'Legacy output already used: '+relative)
    ray_temp = Path(expected['ray_temp_dir'])
    require(ray_temp.parent == RAY_TEMP_BASE and ray_temp.resolve() == ray_temp
            and RAY_TEMP_BASE.resolve() == RAY_TEMP_BASE,
            'Legacy Ray temp path redirects outside the canonical data-disk base')
    require(not os.path.lexists(ray_temp),'Legacy Ray temp directory already exists; preserve and inspect it')
    sys.path.insert(0,str(args.checkout/'weaver_run/reference_site/queue_wrapper'))
    import training_entry as safety
    from launch import mps_preflight, gpu_inventory
    authorization = safety.mps_authorization()
    inventory = gpu_inventory()
    require(tuple(inventory[str(index)] for index in (4,5,6,7)) == safety.GPUS,
            'Legacy physical GPU UUID mapping differs')
    mps_preflight(authorization['pipe'],safety.GPUS)
    RAY_TEMP_BASE.mkdir(parents=True,exist_ok=True)
    require(RAY_TEMP_BASE.resolve() == RAY_TEMP_BASE and ray_temp.resolve() == ray_temp,
            'Legacy Ray temp base changed during admission')
    ray_temp.mkdir(mode=0o700)
    for relative in ('training','tmp'):
        (args.root/relative).mkdir(mode=0o700)
    result = supervise(expected,launch_environment(expected,authorization),safety)
    # Source must remain exact through training; a success is invalid if it moved.
    try:
        result['final_source_validation'] = validate_source(args.checkout,args.git_commit)
    except BaseException as exc:
        result.update(state='failed',source_error=repr(exc))
    result['source_validation'] = validation
    atomic(args.root/'queue_receipt.json',result)
    return 0 if result['state'] == 'succeeded' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--git-commit',required=True)
    parser.add_argument('--training-seconds',type=float,required=True)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except BaseException as exc:
        failure = dict(schema=1,state='failed',phase='queue_admission',error=repr(exc),
                       traceback=traceback.format_exc(),git_commit=args.git_commit,
                       job_id=os.environ.get('EXPERIMENT_QUEUE_JOB_ID'),finished_ns=time.time_ns(),
                       all_owned_exited=True,gpu_child_started=False)
        if args.root.is_dir() and args.root.resolve() == args.root:
            path = args.root/'queue_admission_failure.json'
            if not os.path.lexists(path):
                atomic(path,failure)
        print(json.dumps(failure),file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
