"""Bounded four-card P189 comparison on the original site's GPU lease."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO/'weaver_run/engine'
sys.path[:0] = [str(REPO/'weaver_run'),str(ENGINE)]
from git_version import BRANCH, revision
from launch import checksum, effective_pipeline, put


SOURCE_RUN = 'weaverintegrated_1119_001039_efef204a'
SOURCE_LABEL = 'model-free-autobatch-prune-20261118-a'
WEIGHTS_SHA256 = '9025fd14bd1e7e6bce8b284824a1a4eeb167d6c07b6074acdb00db4653f3e0fe'
MAX_SECONDS = 3*3600


def submit(cli,label,*,status=False,sampling=False,pulse=False,layout=False,critical=False,shapes=False,
           check_only=False,real_workers=None,normal_only=False,no_micro_cache_trim=False):
    cli.require(cli.LABEL.fullmatch(label),'Invalid benchmark label')
    root = cli.EXPERIMENTS/label
    if sum((sampling,pulse,layout,critical,shapes)) > 1:
        raise ValueError('Select one benchmark comparison')
    if check_only and (not shapes or status):
        raise ValueError('CPU admission is supported for the micro-shape comparison')
    if real_workers is None:
        real_workers = 8 if critical or shapes else 4
    if (critical or shapes) and real_workers != 8:
        raise ValueError('This paired comparison requires eight real workers')
    if (real_workers not in (4, 8) or (normal_only and not pulse)
            or (real_workers == 8 and not (normal_only or critical or shapes))):
        raise ValueError('Eight real workers require normal-only admission, critical JIT or micro shapes')
    if no_micro_cache_trim and not (critical or (pulse and normal_only)):
        raise ValueError('Micro cache trim comparison requires --normal-only')
    prefix = 'weaver-bench-'
    if shapes:
        prefix += 'shapes-'
    elif critical:
        prefix += 'critical-'
    elif layout:
        prefix += 'layout-'
    elif sampling:
        prefix += 'sampling-'
    elif pulse:
        prefix += 'pulse-'
    sid = prefix + label
    found = cli.lookup(sid)
    if status or (found and not check_only):
        if found and pulse and not status:
            prior = json.loads((root/'plan.json').read_bytes())
            if (prior['real_workers'], prior.get('normal_only', False), prior.get('no_micro_cache_trim', False)
                    ) != (real_workers, normal_only, no_micro_cache_trim):
                raise ValueError('Benchmark label already has different rollout settings')
        info = dict(label=label,job_id=found['id'] if found else None,
                    state=found['state'] if found else 'not_submitted',root=str(root))
        for file in ('benchmark_queue.json','submission.json'):
            if found and (root/file).is_file():
                info[file] = json.loads((root/file).read_bytes())
        return info
    commit = cli.published(cli.CHECKOUT, push=(critical or shapes) and not check_only)
    if pulse or layout:
        # 空闲队列复用现有运行目录；这个短测只新建实验数据。
        cli.ready()
        runtime = cli.RUNTIME
    else:
        # Jobs queued behind a live training must never repin its .git-runtime.
        runtime = cli.HOME/'library'/'jit-bench-code'/commit
    original = cli.EXPERIMENTS/SOURCE_LABEL
    weights = original/'formal_v3/training/checkpoints/latest_weights.pt'
    plan = dict(git_commit=commit,git_branch=BRANCH,label=label,output=str(root),
                checkpoint=str(weights),checkpoint_sha256=WEIGHTS_SHA256,
                checkpoint_policy_step=189,checkpoint_run_id=SOURCE_RUN,
                real_workers=real_workers,online_eval_workers=0,fixed_validation=False,
                global_batch=576,rank_size=144,train_micro=16,context_micro=2,
                source='fresh_real_rollouts_at_P189',gpu_ids=[4,5,6,7])
    if shapes:
        plan.update(comparison='micro_shapes', real_sampling='uniform_blocks',
                    no_micro_cache_trim=True, compile_action_inference=True, learning_rate=0.,
                    trainer_mps_percent=90, trainer_compile='on', train_shapes=[[16,9],[24,6]],
                    parity_pairs=1, warmup_pairs=1, measured_pairs=4, minimum_replay_chunks=2753,
                    protocol_seed=1127)
    elif critical:
        plan.update(comparison='critical_jit', real_sampling='uniform_blocks',
                    no_micro_cache_trim=True, compile_action_inference=False, learning_rate=0.,
                    warmup_updates=1, measured_blocks=6, noise_blocks=2, updates_per_arm=1,
                    minimum_replay_chunks=2753, high_next_chunks=1152, gate_ttl_s=5.,
                    protocol_seed=20261125)
    elif pulse:
        plan.update(comparison='admission_pulse', real_sampling='uniform_blocks',
                    warmup_updates=1, measured_blocks=3, updates_per_arm=3,
                    pulse_seconds=5.0, minimum_replay_chunks=2048)
        if normal_only:
            plan['normal_only'] = True
            plan['no_micro_cache_trim'] = no_micro_cache_trim
    elif layout:
        plan.update(comparison='trajectory_layout', real_sampling='uniform_blocks',
                    warmup_pairs=1, measured_pairs=8, minimum_replay_chunks=2753,
                    layout_pool_sizes=[581,2753], no_micro_cache_trim=True)
    else:
        plan.update(warmup_pairs=1, measured_pairs=6, minimum_replay_chunks=4096)
    if sampling:
        plan['comparison'] = 'episode_sampling'
        plan['sampling_modes'] = ['chunks','whole_episodes','uniform_blocks']
    comparison = 'three-sampler learner JIT' if sampling else 'same-cohort learner JIT'
    if shapes:
        comparison = 'paired compiled micro-batch shapes'
    elif critical:
        comparison = 'eight-rollout critical JIT admission'
    elif pulse:
        comparison = (f'normal colocation with {real_workers} real workers' if normal_only
                      else 'rollout and trainer admission pulses')
    elif layout:
        comparison = 'same-cohort trajectory layout'
    job = dict(name='WEAVER '+comparison+' '+label,submission_id=sid,cwd=str(runtime),
               argv=[cli.PYTHON,*(['-I'] if critical or shapes else []),'-B',str(runtime/'weaver_run/reference_site/jit_bench_queue.py'),
                     str(root/'plan.json')],manual=False,timeout_seconds=MAX_SECONDS+180)
    if shapes:
        admission = micro_shape_admission(cli,plan,job)
        if check_only:
            return admission
    runtime.parent.mkdir(parents=True,exist_ok=True)
    if (critical or shapes) and runtime.exists():
        cli.revision(runtime,commit)
    else:
        cli.pin_worktree(cli.CHECKOUT,runtime,commit)
    if root.exists():
        if not (root/'plan.json').is_file() or json.loads((root/'plan.json').read_bytes())!=plan:
            raise ValueError('Output is already bound to a different job')
        if (root/'formal_v3').exists():
            raise ValueError('Benchmark output already contains a previous execution')
    else:
        root.mkdir(mode=0o700)
        put(root/'plan.json',plan)
    put(root/'experiment.json',dict(profile='model_free_jit_bench',git_commit=commit,job=job))
    if shapes:
        put(root/'cpu_admission.json',admission)
    pipeline = effective_pipeline(True, real_workers=real_workers)
    put(root/'git-pipeline.json',pipeline)
    mps=(cli.PARENT/'mps.json').read_bytes()
    if hashlib.sha256(mps).hexdigest()!=cli.PROFILES['restore']['mps_receipt']:
        raise ValueError('Frozen GPU4-7 MPS receipt differs')
    put(root/'mps.json',json.loads(mps))
    put(root/'visualization_config.json',json.loads((runtime/'weaver_run/recipe/visualization.json').read_bytes()))
    cli.require(cli.published(cli.CHECKOUT)==commit,'Published branch changed during CPU admission')
    cli.revision(runtime,commit)
    item=cli.create(job,cli.QUEUE)
    submission=dict(job_id=item['id'],submission_id=sid,git_commit=commit,state=item['state'],root=str(root))
    put(root/'submission.json',submission)
    return submission


def benchmark_argv(plan):
    argv = ['bash',str(ENGINE/'run.sh'),'--model-free','--num-eval-workers','0',
            '--fixed-validation-interval-env-steps','0','--batch-autotune']
    if plan.get('comparison') in ('admission_pulse','critical_jit','micro_shapes'):
        # 最终驱动按命令行重建环境，JIT 和采样方式必须随同传入。
        argv.extend(['--trajectory-jit', '--real-sampling', plan['real_sampling']])
        if plan['comparison'] in ('critical_jit','micro_shapes'):
            argv.extend(['--real-workers', str(plan['real_workers']), '--no-micro-cache-trim',
                         '--policy-lr', str(plan['learning_rate']), '--value-lr', str(plan['learning_rate'])])
        if plan['comparison'] == 'micro_shapes':
            argv.extend(['--trainer-mps-percent',str(plan['trainer_mps_percent']),
                         '--trainer-compile',plan['trainer_compile'],'--compile-action-inference'])
    elif plan.get('comparison') == 'trajectory_layout':
        argv.append('--no-micro-cache-trim')
    return argv


def pulse_environment(plan):
    if plan.get('comparison') != 'admission_pulse':
        return {}
    return {'WEAVER_BENCH_REAL_WORKERS':str(plan.get('real_workers', 4)),
            'WEAVER_BENCH_NORMAL_ONLY':'1' if plan.get('normal_only') else '0',
            'WEAVER_BENCH_NO_MICRO_CACHE_TRIM':'1' if plan.get('no_micro_cache_trim') else '0',
            'ACCERL_TENSORBOARD':'0'}


def verified_checkpoint(path):
    manifest=json.loads((path.parent/'manifest.json').read_bytes())
    if (manifest['steps']!={'policy':189} or manifest['metadata']['run_id']!=SOURCE_RUN
            or manifest['weights_bytes']!=path.stat().st_size or checksum(path)!=WEIGHTS_SHA256):
        raise ValueError('Archived P189 checksum, run identity or Policy step mismatch')
    return manifest


def micro_shape_admission(cli,plan,job):
    verified_checkpoint(Path(plan['checkpoint']))
    cli.require(hashlib.sha256((cli.PARENT/'mps.json').read_bytes()).hexdigest() ==
                cli.PROFILES['restore']['mps_receipt'],'Frozen GPU4-7 MPS receipt differs')
    official = cli.offline_check('restore', model_free=True, num_eval_workers=0, batch_autotune=True,
                                 trajectory_jit=True, real_sampling=plan['real_sampling'], real_workers=plan['real_workers'],
                                 no_micro_cache_trim=True, trainer_mps_percent=plan['trainer_mps_percent'],
                                 trainer_compile=plan['trainer_compile'], compile_action_inference=True)
    # Exercise this benchmark command and its submission ID in a private workerless queue.
    from weaver_run.reference_site.dispatcher.store import QueueStore
    with tempfile.TemporaryDirectory(prefix='.shape-admission-',dir=cli.EXPERIMENTS) as private:
        store=QueueStore(private)
        first=store.create(**job)
        duplicate=store.create(**job)
        cli.require(first['id']==duplicate['id'] and first['state']=='queued' and
                    all(first[k]==v for k,v in job.items()),'Benchmark queue replay mismatch')
    return dict(official,profile='micro_shapes',benchmark_queue='passed',git_commit=plan['git_commit'],
                checkpoint_sha256=WEIGHTS_SHA256,submission_id=job['submission_id'],train_shapes=plan['train_shapes'])


def verified_micro_shape_hardware(db_path,run_id):
    """Read the CUDA-context and trainer receipts emitted by this run's live actors."""
    from training_topology import validate_trainer_mps_bindings
    with closing(sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True)) as db:
        def events(kind):
            rows=db.execute('SELECT details_json FROM events WHERE run_id=? AND kind=?',(run_id,kind))
            return [json.loads(row[0]) for row in rows]
        bindings=events('trainer_mps_verified')
        registration=events('policy_trainer_compiled')
    if len(bindings)!=1 or bindings[0].get('trainer_mps_percent')!=90:
        raise RuntimeError('Missing live MPS 90% receipt')
    trainers, inference=bindings[0]['trainers'],bindings[0]['policy_inference']
    validate_trainer_mps_bindings(trainers,inference,90)
    if sorted(int(row['physical_gpu']) for row in trainers)!=[4,5,6,7]:
        raise RuntimeError('Benchmark was not bound to physical GPUs 4-7')
    expected=dict(enabled=True, compiled=True, backend='inductor', tf32_enabled=True, dynamic=False,
                  llama_layers=32, vision_blocks=51, projector_compiled=True)
    if (sorted(row['rank'] for row in registration)!=[0,1,2,3]
            or any(any(row.get(k)!=v for k,v in expected.items()) for row in registration)):
        raise RuntimeError('Missing four complete Inductor/TF32 registration receipts')
    return dict(mps=bindings[0],trainer_registration=sorted(registration,key=lambda row:row['rank']))


def run(plan_file):
    plan = json.loads(plan_file.read_bytes())
    root = plan_file.parent
    if root!=Path(plan['output']) or plan['checkpoint_sha256']!=WEIGHTS_SHA256:
        raise ValueError('Unexpected benchmark plan identity')
    revision(REPO,plan['git_commit'])
    os.environ['ACCERL_EXPERIMENT_ROOT'] = str(root)
    import gpu47_safety as safe
    from runner import formal_environment
    path = Path(plan['checkpoint'])
    manifest=verified_checkpoint(path)
    pre = safe.preflight()
    data=root/'formal_v3'
    data.mkdir()
    state=dict(run_id='weaverintegrated_bench_'+plan['label'],started_ns=time.time_ns(),uid=os.getuid(),
               experiment_root=str(root),source_path=str(ENGINE),candidate='weaver_integrated_v1',
               pipeline_config_path=str(root/'git-pipeline.json'),model_free=True,batch_autotune=True,
               num_eval_workers=0,owned=[],supervisor=safe.proc_identity(os.getpid()))
    if plan.get('comparison') in ('admission_pulse','critical_jit','micro_shapes'):
        state.update(trajectory_jit=True, real_sampling=plan['real_sampling'],
                     initial_policy={'checkpoint': str(path)})
        if plan['comparison'] in ('critical_jit','micro_shapes'):
            state.update(real_workers=plan['real_workers'],no_micro_cache_trim=True)
        if plan['comparison'] == 'micro_shapes':
            state.update(trainer_mps_percent=plan['trainer_mps_percent'],trainer_compile=plan['trainer_compile'],
                         compile_action_inference=plan['compile_action_inference'])
    elif plan.get('comparison') == 'trajectory_layout':
        state['no_micro_cache_trim'] = True
    env=formal_environment(state,data,pre)
    env.update(pulse_environment(plan))
    env.update(ACCERL_RAY_TMP='/dev/shm/weaverintegrated_jb_'+str(os.getpid()),
               ACCERL_JIT_BENCH_PLAN=str(plan_file),ACCERL_JIT_BENCH_OUTPUT=str(root),
               ACCERL_JIT_BENCH_CHECKPOINT=str(path))
    put(root/'checkpoint_verified.json',dict(sha256=WEIGHTS_SHA256,policy_step=189,run_id=SOURCE_RUN,
            bytes=path.stat().st_size,tensors=manifest['tensor_counts']['policy']))
    argv=benchmark_argv(plan)
    def interrupted(signum, frame):
        raise InterruptedError('Queue supervisor received signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,interrupted)
    receipt=dict(state='running',git_commit=plan['git_commit'],run_id=state['run_id'],
                 started_ns=state['started_ns'],argv=argv,checkpoint_sha256=WEIGHTS_SHA256)
    put(root/'benchmark_queue.json',receipt)
    with (root/'driver.log').open('x',buffering=1) as log:
        child=subprocess.Popen(argv,env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            identity=safe.proc_identity(child.pid)
            if not identity:
                raise RuntimeError('Benchmark process exited before startup')
            state.update(child=identity,minimum_child_birth_ticks=identity['birth_ticks'])
            if child.wait(timeout=MAX_SECONDS)!=0:
                raise RuntimeError(f'Benchmark driver exited with code {child.returncode}')
            result=json.loads((root/'result.json').read_bytes())
            if result.get('status')!='completed':
                raise RuntimeError('Benchmark driver did not report completed')
            if plan.get('comparison')=='micro_shapes':
                if result.get('parity_passed') is not True or len(result.get('pairs',[]))!=plan['measured_pairs']:
                    raise RuntimeError('Missing micro-shape numerical parity or paired measurements')
                receipt['hardware'] = verified_micro_shape_hardware(data/'training/collection.sqlite',state['run_id'])
            for rank in range(4):
                restored=json.loads((root/f'checkpoint_rank_{rank}.json').read_bytes())
                if restored['policy_step']!=189 or restored['sha256']!=WEIGHTS_SHA256:
                    raise RuntimeError('Unverified trainer checkpoint for rank '+str(rank))
            receipt['state']='completed'
        except BaseException as exc:
            receipt.update(state='failed',error=repr(exc))
            raise
        finally:
            if 'minimum_child_birth_ticks' in state:
                try:
                    if not safe.cleanup(state,data):
                        raise RuntimeError('Some benchmark-owned processes survived cleanup')
                finally:
                    receipt['cleanup']=state.get('cleanup')
                    if not state.get('training_stopped'):
                        receipt['state']='cleanup_requires_attention'
            receipt['finished_ns']=time.time_ns()
            put(root/'benchmark_queue.json',receipt)


if __name__=='__main__':
    run(Path(sys.argv[1]).resolve(strict=True))
