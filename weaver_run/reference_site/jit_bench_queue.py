"""Bounded four-card P189 comparison on the original site's GPU lease."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO/'weaver_run/engine'
sys.path[:0] = [str(REPO/'weaver_run'),str(ENGINE)]
from git_version import BRANCH, revision
from launch import effective_pipeline, put


SOURCE_RUN = 'weaverintegrated_1119_001039_efef204a'
SOURCE_LABEL = 'model-free-autobatch-prune-20261118-a'
WEIGHTS_SHA256 = '9025fd14bd1e7e6bce8b284824a1a4eeb167d6c07b6074acdb00db4653f3e0fe'
MAX_SECONDS = 3*3600


def submit(cli,label,*,status=False):
    cli.require(cli.LABEL.fullmatch(label),'Invalid benchmark label')
    root = cli.EXPERIMENTS/label
    sid = 'weaver-bench-'+label
    found = cli.lookup(sid)
    if status or found:
        info = dict(label=label,job_id=found['id'] if found else None,
                    state=found['state'] if found else 'not_submitted',root=str(root))
        for file in ('benchmark_queue.json','submission.json'):
            if (root/file).is_file():
                info[file] = json.loads((root/file).read_bytes())
        return info
    commit = cli.published(cli.CHECKOUT)
    # Another queued training can repin .git-runtime. This job holds a separate
    # content-addressed checkout so both leases remain bound to their own commit.
    runtime = cli.HOME/'library'/'jit-bench-code'/commit
    runtime.parent.mkdir(parents=True,exist_ok=True)
    cli.pin_worktree(cli.CHECKOUT,runtime,commit)
    original = cli.EXPERIMENTS/SOURCE_LABEL
    weights = original/'formal_v3/training/checkpoints/latest_weights.pt'
    plan = dict(git_commit=commit,git_branch=BRANCH,label=label,output=str(root),
                checkpoint=str(weights),checkpoint_sha256=WEIGHTS_SHA256,
                checkpoint_policy_step=189,checkpoint_run_id=SOURCE_RUN,
                real_workers=4,online_eval_workers=0,fixed_validation=False,
                warmup_pairs=1,measured_pairs=6,minimum_replay_chunks=4096,
                global_batch=576,rank_size=144,train_micro=16,context_micro=2,
                source='fresh_real_rollouts_at_P189',gpu_ids=[4,5,6,7])
    job = dict(name='WEAVER same-cohort learner JIT '+label,submission_id=sid,cwd=str(runtime),
               argv=[cli.PYTHON,'-B',str(runtime/'weaver_run/reference_site/jit_bench_queue.py'),
                     str(root/'plan.json')],manual=False,timeout_seconds=MAX_SECONDS+180)
    if root.exists():
        if not (root/'plan.json').is_file() or json.loads((root/'plan.json').read_bytes())!=plan:
            raise ValueError('Output is already bound to a different job')
        if (root/'formal_v3').exists():
            raise ValueError('Benchmark output already contains a previous execution')
    else:
        root.mkdir(mode=0o700)
        put(root/'plan.json',plan)
    put(root/'experiment.json',dict(profile='model_free_jit_bench',git_commit=commit,job=job))
    put(root/'git-pipeline.json',effective_pipeline(True))
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


def run(plan_file):
    plan = json.loads(plan_file.read_bytes())
    root = plan_file.parent
    if root!=Path(plan['output']) or plan['checkpoint_sha256']!=WEIGHTS_SHA256:
        raise ValueError('Unexpected benchmark plan identity')
    revision(REPO,plan['git_commit'])
    os.environ['ACCERL_EXPERIMENT_ROOT'] = str(root)
    import gpu47_safety as safe
    from runner import formal_environment
    from launch import checksum
    path = Path(plan['checkpoint'])
    manifest=json.loads((path.parent/'manifest.json').read_bytes())
    if (manifest['steps']!={'policy':189} or manifest['metadata']['run_id']!=SOURCE_RUN
            or manifest['weights_bytes']!=path.stat().st_size or checksum(path)!=WEIGHTS_SHA256):
        raise ValueError('Archived P189 checksum, run identity or Policy step mismatch')
    pre = safe.preflight()
    data=root/'formal_v3'
    data.mkdir()
    state=dict(run_id='weaverintegrated_bench_'+plan['label'],started_ns=time.time_ns(),uid=os.getuid(),
               experiment_root=str(root),source_path=str(ENGINE),candidate='weaver_integrated_v1',
               pipeline_config_path=str(root/'git-pipeline.json'),model_free=True,batch_autotune=True,
               num_eval_workers=0,owned=[],supervisor=safe.proc_identity(os.getpid()))
    env=formal_environment(state,data,pre)
    env.update(ACCERL_RAY_TMP='/dev/shm/weaverintegrated_jb_'+str(os.getpid()),
               ACCERL_JIT_BENCH_PLAN=str(plan_file),ACCERL_JIT_BENCH_OUTPUT=str(root),
               ACCERL_JIT_BENCH_CHECKPOINT=str(path))
    put(root/'checkpoint_verified.json',dict(sha256=WEIGHTS_SHA256,policy_step=189,run_id=SOURCE_RUN,
            bytes=path.stat().st_size,tensors=manifest['tensor_counts']['policy']))
    argv=['bash',str(ENGINE/'run.sh'),'--model-free','--num-eval-workers','0',
          '--fixed-validation-interval-env-steps','0','--batch-autotune']
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
