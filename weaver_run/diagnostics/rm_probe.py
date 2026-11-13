"""Finite, queue-owned RM validation on fresh frozen-policy real trajectories."""
import argparse
import gc
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'weaver_run'),str(ROOT/'weaver_run/engine')]
from weaver_run.diagnostics.panel_common import STATES, GPUS, save


def rm_prefix(actions, success):
    """Observed 150-action task outcome inside a complete 220-action episode."""
    if type(actions) is not int or not 1<=actions<=220 or type(success) is not bool:
        raise ValueError('Invalid complete episode outcome')
    if not success and actions!=220:
        raise ValueError('Unfinished episode is not an observed failure')
    n=min(actions,150); outcome=success and actions<=150
    return n,outcome,[(.99**(1/8))**(n-i) if outcome else 0. for i in range(1,n+1)]


def real_scoring_blocks(n):
    if type(n) is not int or not 1<=n<=150:raise ValueError('Invalid RM prefix length')
    return [(list(range(max(0,end-9),end+1)) if end>=9 else [0]*(9-end)+list(range(end+1)),
             list(range(start+1,end+1)))
            for start in range(0,n,8) for end in [min(start+8,n)]]


def probe_spec():
    base=Path('/mnt/data/lcx1/dong/WEAVER/experiments');models={}
    for name,label,run,p,w in (
        ('orange150','orange150terminal-20261110-a','weaverintegrated_1110_094754_4c411115',594,118),
        ('refill','inventory-refill-20261112-a','weaverintegrated_1112_180139_d5ee1e47',275,55)):
        root=base/label/'formal_v3'
        models[name]=dict(run_id=run,policy_step=p,world_step=w,
            policy=str(root/'training/checkpoints/latest_weights.pt'),
            world=str(root/f'training/checkpoints/checkpoints_world/world_p{p:08d}_w{w:08d}_final.pt'),
            config=str(root/'world_backend_config.json'))
    return dict(protocol='weaver_independent_rm_state_v1',models=models,initial_states=STATES,
        calibration_initial_states=STATES[::2],test_initial_states=STATES[1::2],
        collection_action_horizon=220,rm_action_horizon=150,max_seconds=14400,
        no_optimizer_steps=True,real_observations_only=True,
        independence='Fresh post-checkpoint episodes in an isolated process, never ingested into training Replay.',
        scope='Same task initial states, not unseen tasks; two frozen-policy panels cross-scored by both RMs.')


def admission(plan):
    import torch
    from frozen_policy_eval import load_snapshot, file_sha
    if torch.cuda.is_initialized():raise ValueError('Admission must be CPU only')
    if any(plan.get(k)!=v for k,v in probe_spec().items()):raise ValueError('Unexpected RM protocol')
    checked={};configs=[]
    for key,m in plan['models'].items():
        policy,receipt=load_snapshot(m['policy'],time.time()+300)
        if (receipt['metadata']['training_run_id'],receipt['metadata']['policy_step'])!=(m['run_id'],m['policy_step']):
            raise ValueError('Policy checkpoint identity mismatch')
        cfg=json.loads(Path(m['config']).read_bytes());configs.append(cfg)
        head=cfg.get('backend_options',{}).get('native_reward_head',{})
        if head.get('mode')!='libero_discounted_success_value_v1':raise ValueError('Wrong native RM target')
        world=torch.load(m['world'],map_location='cpu',mmap=True,weights_only=True)
        wm=world['manifest']
        if (wm['run_id'],wm['policy_step'],wm['model_version'])!=(m['run_id'],m['policy_step'],m['world_step']):
            raise ValueError('World checkpoint identity mismatch')
        checked[key]=dict(policy_sha256=receipt['sha256'],world_sha256=file_sha(m['world']),
                          world_tensor_count=len(world['model']))
        del world,policy
    if configs[0]!=configs[1]:raise ValueError('World configs differ; explicit review required')
    if torch.cuda.is_initialized():raise ValueError('Admission initialized CUDA')
    return dict(models=checked,cuda_initialized=False,protocol=plan['protocol'])


def execute(plan,out):
    import torch
    from frozen_policy_eval import restore_policy,load_snapshot,file_sha
    from weaver_run.diagnostics.policy_panel import Native,collect
    from weaver_run.diagnostics.rm_report import rm_state_report
    episodes={};rows=[];native=None
    # Collection is independent and finished before either model scores the panel.
    for key,m in plan['models'].items():
        payload,receipt=load_snapshot(m['policy'],time.time()+300)
        if receipt['sha256']!=plan['admission']['models'][key]['policy_sha256']:raise ValueError('Policy changed')
        model,cfg,dtype=restore_policy(payload,time.time()+1800)
        eps,segments=collect(model,cfg,dtype,plan,key,out)
        episodes[key]=eps
        del segments,model,payload
        gc.collect();torch.cuda.empty_cache()
    try:
        native=Native(out,plan,ROOT)
        for key,m in plan['models'].items():
            if file_sha(m['world'])!=plan['admission']['models'][key]['world_sha256']:raise ValueError('World changed')
            native.call(op='load',model=m)
            for policy,eps in episodes.items():
                for ep in eps:
                    for row in native.call(op='score_real',episode=ep)['rows']:
                        row.update(model=key,policy_panel=policy);rows.append(row)
                    save(out/'predictions.json',rows)
                    save(out/'progress.json',dict(stage='real_rm_scoring',model=key,policy_panel=policy,
                                                  episode_id=ep['episode_id'],rows=len(rows)))
        report=rm_state_report(rows,
            calibration_initial_states=plan['calibration_initial_states'],
            test_initial_states=plan['test_initial_states'])
        save(out/'rm_report.json',report)
        from hashlib import sha256
        manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest()) for p in out.glob('*-state*.pkl')}
        save(out/'real_panel_manifest.json',manifest)
        return dict(complete=True,rows=len(rows),report='rm_report.json',
            panels={k:dict(episodes=len(v),successes_220=sum(e['success'] for e in v),
                          successes_150=sum(e['success'] and len(e['actions'])<=150 for e in v)) for k,v in episodes.items()},
            limitations=['16 fixed initial states, stochastic once per policy; not a 50-state final success benchmark.',
                         'Calibration and test are split by initial state across both policies, not by frames.',
                         'Failure groups may have low support. RM scores are discounted outcomes, not calibrated binary probabilities.',
                         'No imaginary-observation scoring, no optimizer, no writes to training Replay.'])
    finally:
        if native is not None:native.close()


def submit(cli,label,status=False):
    label=label or 'rm-state-20261113-a';sid='weaver-rm-'+label
    cli.require(cli.LABEL.fullmatch(label) is not None and len(sid)<=64,'Invalid RM diagnostic ID')
    root=cli.EXPERIMENTS/label;found=cli.lookup(sid)
    if found or status:
        return dict(label=label,submission_id=sid,job_id=found['id'] if found else None,
            state=found['state'] if found else 'not_submitted',output=str(root),
            result=json.loads((root/'result.json').read_bytes()) if (root/'result.json').exists() else None)
    # The active training keeps its detached runtime. This finite job checks the
    # development commit before execution; later edits fail closed, never repin training.
    commit=cli.published(cli.CHECKOUT,push=False)
    cli.require(not root.exists(),'Output exists without queue receipt; inspect before retry')
    import shutil
    cli.require(shutil.disk_usage(cli.EXPERIMENTS).free>=20*1024**3,'Need20 GiB free diagnostic storage')
    root.mkdir(mode=0o700)
    plan=probe_spec();plan.update(git_commit=commit,output=str(root),submission_id=sid,
        native_python='/mnt/data/lcx1/dong/envs/weaver-libero-20261022/bin/python')
    argv=['/usr/bin/env','PYTHONNOUSERSITE=1','PYTHONDONTWRITEBYTECODE=1',cli.PYTHON,
          str(cli.CHECKOUT/'weaver_run/diagnostics/rm_probe.py'),'--plan',str(root/'plan.json')]
    plan['job_argv']=argv;save(root/'plan.json',plan)
    try:
        plan['admission']=json.loads(cli.cpu([*argv,'--validate-only'],cwd=cli.CHECKOUT,timeout=600).strip().splitlines()[-1])
        save(root/'plan.json',plan)
        cli.require(plan['admission']['cuda_initialized'] is False,'CPU admission used CUDA')
        cli.require(cli.published(cli.CHECKOUT)==commit,'Commit changed during admission')
        waiting=cli.active_jobs()
        job=dict(name='WEAVER independent RM '+label,submission_id=sid,cwd=str(cli.CHECKOUT),argv=argv,
                 manual=False,timeout_seconds=plan['max_seconds']+120)
        save(root/'experiment.json',dict(profile='rm_state_diagnostic',git_commit=commit,job=job,
                                         waiting_behind=waiting,no_training_stop_requested=True))
        item=cli.create(job,cli.QUEUE)
        receipt=dict(job_id=item['id'],submission_id=sid,git_commit=commit,state=item['state'],output=str(root))
        save(root/'submission.json',receipt);return receipt
    except BaseException as error:
        save(root/'submission_error.json',dict(error=str(error)));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();plan=json.loads(Path(args.plan).read_bytes());out=Path(plan['output'])
    if args.validate_only:print(json.dumps(admission(plan)));return
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('RM diagnostic time budget exceeded')))
    signal.alarm(probe_spec()['max_seconds'])
    try:
        from weaver_run.git_version import revision
        from weaver_run.diagnostics.panel_runner import verify_owner
        from weaver_run.launch import mps_preflight
        revision(ROOT,plan['git_commit']);identity=verify_owner(plan)
        mps_preflight(os.environ.get('CUDA_MPS_PIPE_DIRECTORY'),GPUS)
        os.environ.update(CUDA_VISIBLE_DEVICES=GPUS[0],MUJOCO_GL='osmesa',TOKENIZERS_PARALLELISM='false',
                          HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',OMP_NUM_THREADS='1')
        if admission(plan)!=plan['admission']:raise ValueError('Checkpoint admission changed')
        save(out/'started.json',dict(identity=identity,git_commit=plan['git_commit'],started_ns=time.time_ns()))
        save(out/'result.json',dict(execute(plan,out),finished_ns=time.time_ns()))
    except BaseException:
        save(out/'result.json',dict(complete=False,error=traceback.format_exc(),finished_ns=time.time_ns()));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
