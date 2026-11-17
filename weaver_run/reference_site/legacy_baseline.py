"""One isolated, versioned legacy GIPO speed run through the official queue."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1]
HOME = Path('/mnt/data/lcx1/dong/WEAVER')
EXPERIMENTS = HOME/'experiments'
PYTHON = '/mnt/data/lcx3/envs/merged-env/bin/python'
POLICY = '/mnt/data/lcx3/checkpoint/dsj/openvla-7b+libero_spatial_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt'
DISTILLED = '/mnt/data/lcx3/checkpoint/dsj/20251225_113851_distill_checkpoint_latest.pt'
PROFILE = 'legacy_speed_baseline'
RAY_TEMP_BASE = Path('/mnt/data/lcx1/ray')
GPUS = ('GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c',
        'GPU-fe0edfe4-b193-1af7-6a98-cbff40214f74',
        'GPU-8c9bc264-7f09-989e-1ec7-cecd6cde782c',
        'GPU-07ae0bbc-8923-bd0c-199e-c726821e655d')
LABEL = re.compile(r'[a-z0-9][a-z0-9_-]{0,39}\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_record(checkout, root, label, commit, training_seconds=7200):
    """Pure payload builder, also used to freeze the delayed handoff plan."""
    checkout, root = Path(checkout), Path(root)
    require(LABEL.fullmatch(label), 'Invalid legacy experiment label')
    require(re.fullmatch('[0-9a-f]{40}',commit), 'Full Git commit required')
    require(math.isfinite(training_seconds) and training_seconds > 0, 'Training seconds must be finite and positive')
    require(checkout.is_absolute() and root.is_absolute(), 'Absolute checkout/output required')
    seconds = float(training_seconds)
    training = root/'training'
    ray_temp = RAY_TEMP_BASE/hashlib.sha256((commit+'\0'+label+'\0'+str(root)).encode()).hexdigest()[:12]
    run_id = 'legacybaseline_'+label
    options = {'--cuda-visible-devices':','.join(GPUS),'--benchmark':'libero_spatial','--task-ids':'0',
        '--num-trainer-gpus':'3','--num-inference-actors':'1','--num-rollout-workers':'10',
        '--num-eval-workers':'0','--rollout-local-buf':'64','--inference-batch':'8',
        '--inference-timeout-ms':'300','--replay-capacity':'3000','--train-batch-size':'12',
        '--accumulation-steps':'16','--train-iters':'30000','--object-store-memory-gb':'256',
        '--ckpt-dir':str(training/'checkpoints/latest'),'--ckpt-every-steps':'2000000',
        '--gamma':'0.99','--lambda':'0.95','--clip-eps':'0.2','--vf-coef':'0.5',
        '--ent-coef':'0.0','--kl-coef':'0.1','--reward-scale':'1.0','--sigma':'1.0',
        '--value-lr':'0.0001','--policy-lr':'0.00001','--value-warmup-steps':'500',
        '--policy-warmup-steps':'500','--policy-train-start-step':'0',
        '--moving-avg-window':'1000','--log-interval-seconds':'10',
        '--broadcast-group-name':'trainer_to_inference_broadcast',
        '--num-images-in-input':'2','--pretrained-checkpoint':POLICY,'--checkpoint2':DISTILLED,
        '--clip-mode':'gipo','--exp-name':run_id,'--baseline-output-dir':str(training),
        '--max-train-seconds':str(seconds),'--ray-temp-dir':str(ray_temp),
        '--stop-file':str(training/'stop.flag')}
    driver = [PYTHON,'-B','-m','rl.ds_libero_ppo_discrete',
              *[part for pair in options.items() for part in pair],'--use-bf16']
    wrapper = checkout/'weaver_run/reference_site/legacy_queue_entry.py'
    argv = ['/usr/bin/env','PYTHONNOUSERSITE=1','PYTHONDONTWRITEBYTECODE=1',PYTHON,
            str(wrapper),'--checkout',str(checkout),'--root',str(root),
            '--git-commit',commit,'--training-seconds',str(seconds)]
    return dict(schema=1,profile=PROFILE,label=label,root=str(root),checkout=str(checkout),
        source=str(checkout),run_id=run_id,git_commit=commit,
        ray_temp_dir=str(ray_temp),
        model_free=True,num_eval_workers=0,batch_autotune=False,gpu_layout='legacy3+1',
        git_branch='codex/weaver-original95-runnable',training_seconds=seconds,
        configuration=dict(trainer_gpus=3,inference_actors=1,physical_gpus=[4,5,6,7],
            cuda_visible_gpu_uuids=list(GPUS),
            benchmark='libero_spatial',task_ids=[0],eval_workers=0,rollout_workers=10,
            images=2,use_proprio=False,global_batch=576,micro_batch=12,accumulation_steps=16,
            clip_mode='gipo',sigma=1.0,render_backend='osmesa',action_horizon=220,
            value_recomputation=False,checkpoint_kind='inference_weights_only',exact_resume=False),
        comparison_limits=['legacy replay and policy age semantics','old-value GAE',
                           'legacy action-chunk treatment','220-action horizon','OSMesa rendering',
                           'three trainer GPUs plus one inference GPU'],
        driver_argv=driver,
        job=dict(name='WEAVER legacy baseline '+label,submission_id='weaver-run-'+label,
                 cwd=str(checkout),argv=argv,manual=True,timeout_seconds=0))


def match_record(actual, expected):
    require(actual == expected, 'Existing legacy experiment source/configuration differs')


def match_job(record, row):
    wanted = record['job']
    require(json.loads(row['argv_json']) == wanted['argv'] and
            all(row.get(k) == wanted[k] for k in ('submission_id','name','cwd','manual','timeout_seconds')),
            'Existing legacy queue identity/payload differs')


def validate_source(checkout, commit):
    sys.path.insert(0,str(Path(checkout)/'weaver_run'))
    from git_version import revision
    revision(Path(checkout),commit)
    for relative in ('rl/ds_libero_ppo_discrete.py','rl/legacy_baseline_control.py',
                     'weaver_run/reference_site/legacy_baseline.py',
                     'weaver_run/reference_site/legacy_queue_entry.py'):
        path = Path(checkout)/relative
        require(not path.is_symlink(), 'Legacy source redirects: '+relative)
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    for path in (Path(PYTHON),Path(POLICY),Path(DISTILLED)):
        require(path.exists(), 'Missing legacy resource: '+str(path))
    return dict(git_commit=commit,gpu_access_performed=False,source=str(checkout),
                cpu_syntax='passed',resources='present')


def _main_unlocked(argv=None):
    parser = argparse.ArgumentParser(prog='./run.sh legacy-baseline')
    parser.add_argument('--id',required=True)
    parser.add_argument('--training-seconds',type=float,default=7200)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args(argv)
    # Import Linux/server helpers only at the actual entry point.
    sys.path.insert(0,str(HERE))
    sys.path.insert(0,str(CHECKOUT/'weaver_run'))
    import cli
    from git_version import published, revision, pin_worktree, git
    require(CHECKOUT == (HOME/'code').resolve(), 'Use the maintained WEAVER/code checkout')
    commit = published(CHECKOUT)
    runtime = HOME/'.legacy-runtime'/commit
    root = EXPERIMENTS/args.id
    record = make_record(runtime,root,args.id,commit,args.training_seconds)
    validation = validate_source(CHECKOUT,commit)
    if args.check_only:
        print(json.dumps(dict(valid=True,production_submission=False,validation=validation,record=record)))
        return 0
    if os.path.lexists(root):
        require(root.resolve() == root and not root.is_symlink(), 'Legacy output redirects')
        match_record(json.loads((root/'experiment.json').read_bytes()),record)
        previous = cli.lookup(record['job']['submission_id'])
        if previous:
            match_job(record,previous)
            cli.save_submission(root,previous)
            print(json.dumps(cli.summary(record,previous)))
            return 0
    else:
        require(cli.lookup(record['job']['submission_id']) is None,'Submission ID already used')
    cli.ready()
    if runtime.exists():
        require(not runtime.is_symlink(), 'Legacy runtime redirects')
        revision(runtime,commit)
        require(not git(runtime,'branch','--show-current'),'Legacy runtime must remain detached')
        common = ('rev-parse','--path-format=absolute','--git-common-dir')
        require(git(runtime,*common) == git(CHECKOUT,*common),'Legacy runtime belongs to another repository')
    else:
        runtime.parent.mkdir(parents=True,exist_ok=True)
        pin_worktree(CHECKOUT,runtime,commit)
    validate_source(runtime,commit)
    cli.ready()
    if not root.exists():
        with tempfile.TemporaryDirectory(dir=EXPERIMENTS,prefix='.legacy-staging-') as temp:
            stage = Path(temp)/'output'
            stage.mkdir(mode=0o700)
            cli.json_write(stage/'experiment.json',record)
            stage.rename(root)
    match_record(json.loads((root/'experiment.json').read_bytes()),record)
    require(published(CHECKOUT,push=True) == commit,'Published legacy branch moved')
    revision(runtime,commit)
    cli.ready()
    item = cli.create(record['job'],cli.QUEUE)
    cli.save_submission(root,item)
    print(json.dumps(cli.summary(record,item)))
    return 0


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--check-only' in arguments or '--help' in arguments or '-h' in arguments:
        return _main_unlocked(arguments)
    import fcntl
    with (EXPERIMENTS/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _main_unlocked(arguments)


if __name__ == '__main__':
    sys.exit(main())
