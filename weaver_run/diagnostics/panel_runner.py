"""Finite queue-owned diagnostic. Frozen weights, fresh observations, no optimizer."""
import argparse,gc,json,math,os,random,signal,sqlite3,sys,time,traceback
from pathlib import Path
from contextlib import closing
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'weaver_run/engine')]
from weaver_run.diagnostics.panel_common import save,spec,GPUS,flatten_segments


def ema_after(model):
    with closing(sqlite3.connect(Path(model['db']).as_uri()+'?mode=ro',uri=True,timeout=10)) as c:
        c.execute('pragma query_only=on')
        rows=c.execute("select details_json from events where run_id=? and kind='jit_gae_optimizer_window' and json_extract(details_json,'$.policy_version')=?",
                       (model['run_id'],model['policy_step']-1)).fetchall()
    d=[json.loads(r[0]) for r in rows]
    if len(d)!=4 or {r['rank'] for r in d}!={0,1,2,3}:raise ValueError('Missing exact four-rank checkpoint EMA')
    keys=('ema_mean','ema_variance','ema_updates','ema_scale_floor','normalization','mode')
    if any(tuple(r[k] for k in keys)!=tuple(d[0][k] for k in keys) for r in d):raise ValueError('Rank EMA disagreement')
    r=d[0]
    if r['ema_updates']!=model['policy_step'] or r['normalization']!='ema_zscore' or r['mode']!='fused':raise ValueError('Checkpoint is not the frozen JIT/EMA recipe')
    if not all(math.isfinite(r[k]) for k in ('ema_mean','ema_variance','ema_scale_floor')) or r['ema_variance']<0 or r['ema_scale_floor']<=0:
        raise ValueError('Invalid checkpoint EMA values')
    return dict(mean=r['ema_mean'],variance=r['ema_variance'],scale=max(math.sqrt(r['ema_variance']),r['ema_scale_floor']),
                updates=r['ema_updates'],scope='EMA after successful checkpoint optimizer update')


def admission(plan):
    """CPU admission also runs again before any GPU model is loaded."""
    import torch
    from frozen_policy_eval import load_snapshot,file_sha
    if torch.cuda.is_initialized():raise ValueError('Admission must be CPU only')
    fixed=spec()
    if any(plan.get(k)!=v for k,v in fixed.items()):raise ValueError('Unexpected diagnostic protocol or checkpoints')
    receipts={};configs=[]
    for key,m in plan['models'].items():
        payload,r=load_snapshot(m['policy'],time.time()+300)
        if (r['metadata']['training_run_id'],r['metadata']['policy_step'])!=(m['run_id'],m['policy_step']):raise ValueError('Policy identity differs')
        args=payload['manifest']['metadata']['args']
        expected=dict(gamma=.99,lambda_=.95,vf_coef=.5,ent_coef=0.,kl_coef=.1,clip_mode='gipo',clip_eps=.2,train_batch_size=8,use_proprio=False)
        if any(args[k]!=v for k,v in expected.items()):raise ValueError('Frozen learning recipe differs')
        world=torch.load(m['world'],map_location='cpu',mmap=True,weights_only=True)
        wm=world['manifest']
        if (wm['run_id'],wm['policy_step'],wm['model_version'])!=(m['run_id'],m['policy_step'],m['world_step']):raise ValueError('World identity differs')
        configs.append(json.loads(Path(m['config']).read_bytes()))
        r.update(world_sha256=file_sha(m['world']),world_tensor_count=len(world['model']),ema=ema_after(m),loss_args=expected)
        receipts[key]=r
        del payload,world
    if configs[0]!=configs[1]:raise ValueError('Cross-model World configs differ; explicit review required')
    if torch.cuda.is_initialized():raise ValueError('CPU admission unexpectedly initialized CUDA')
    return dict(models=receipts,cuda_initialized=False,protocol=plan['protocol'])


def verify_owner(plan):
    from weaver_subprocess import process_identity
    pid=os.getpid();actual=process_identity(pid);job_id=os.environ.get('EXPERIMENT_QUEUE_JOB_ID')
    if not job_id or actual['uid']!=1004:raise ValueError('Diagnostic must own a queue job')
    path=Path('/mnt/data/lcx1/dong/need_data/experiment_queue_gpu47_20261023/queue.sqlite')
    for attempt in range(30):
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as c:
            c.row_factory=sqlite3.Row;c.execute('pragma query_only=on')
            rows=[dict(r) for r in c.execute("select * from jobs where state='running'")]
        if len(rows)==1 and rows[0]['id']==job_id and rows[0].get('identity_json'):
            row=rows[0];ident=json.loads(row['identity_json'])
            if ident.get('pid')==pid:break
        time.sleep(.1)
    else:raise ValueError('Queue has not granted exclusive diagnostic ownership')
    if (ident['uid'],ident['birth_ticks'])!=(actual['uid'],actual['birth']):raise ValueError('Queue process identity changed')
    if row['submission_id']!=plan['submission_id'] or json.loads(row['argv_json'])!=plan['job_argv'] or Path(row['cwd'])!=ROOT:
        raise ValueError('Queue job does not match this diagnostic plan')
    if tuple(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=GPUS:raise ValueError('Queue GPU allocation differs')
    return dict(job_id=job_id,pid=pid,uid=actual['uid'],birth_ticks=actual['birth'])


def selected_batch(real,imagined,real_count):
    rng=random.Random(20261109)
    selected=rng.sample(flatten_segments(real),real_count)
    if len(imagined)*4!=real_count*3 or any(len(s['rewards'])!=4 for s in imagined):raise ValueError('Incomplete imagined trajectory quota')
    selected += [(len(real)+j,i) for j,s in enumerate(imagined) for i in range(4)]
    rng.shuffle(selected)
    return selected


def execute(plan,out,receipt):
    import torch
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([],'GPU')
    except ImportError:pass
    from frozen_policy_eval import restore_policy,load_snapshot
    from weaver_run.diagnostics.policy_panel import Native,collect,imagined,gradients,dump
    all_eps={};metrics={};native=None
    try:
        native=Native(out,plan,ROOT)
        for key,m in plan['models'].items():
            payload,checked=load_snapshot(m['policy'],time.time()+300)
            if checked['sha256']!=receipt['models'][key]['sha256']:raise ValueError('Checkpoint changed after admission')
            model,cfg,dtype=restore_policy(payload,time.time()+1800)
            episodes,real=collect(model,cfg,dtype,plan,key,out);all_eps[key]=episodes
            count=min(plan['real_chunks'],len(flatten_segments(real))//4*4)
            if count<4:raise ValueError('Insufficient real samples for a diagnostic batch')
            native.call(op='load',model=m)
            imag=imagined(model,cfg,dtype,episodes,3*count//4,native,key,out)
            selected=selected_batch(real,imag,count);segments=real+imag
            dump(out/f'{key}-batch.pkl',dict(segments=segments,selected=selected))
            state=payload['models']['policy']
            names=set(state['trainable_names'])
            for name,p in model.named_parameters():p.requires_grad_(name in names)
            metrics[key]=gradients(model,dtype,segments,selected,receipt['models'][key]['ema'],payload['manifest']['metadata']['args'],key,out)
            # Check all stored trainable tensors and persistent buffers after measurement.
            current=model.state_dict()
            for name,value in state['state_dict'].items():
                if not torch.equal(current[name].detach().cpu(),value):raise ValueError('Diagnostic modified model state: '+name)
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Unexpected parameter.grad write')
            del model,current,state,payload,segments,real,imag
            gc.collect();torch.cuda.empty_cache()
        rows=[]
        for wm,m in plan['models'].items():
            native.call(op='load',model=m)
            for policy,episodes in all_eps.items():
                for ep in episodes:
                    for row in native.call(op='score',episode=ep)['rows']:
                        row.update(world_model=wm,policy_panel=policy);rows.append(row)
                    save(out/'paired_wm_rm.json',rows)
                    save(out/'progress.json',dict(stage='cross_model_scoring',world=wm,policy=policy,episode=ep['episode_id'],rows=len(rows)))
        counts={k:dict(episodes=len(v),successes=sum(e['success'] for e in v),actions=sum(len(e['actions']) for e in v)) for k,v in all_eps.items()}
        return dict(complete=True,panels=counts,paired_rows=len(rows),gradient_files=list(metrics),
            limitations=['Fresh frozen-policy panels, not historical Replay reconstruction or current live-training measurements.',
                'Sixteen initial states from training task 0, one stochastic episode each; not a learning benchmark.',
                'Recorded checkpoint EMA and fresh on-policy behavior logits; no stale-policy gradient reconstruction.',
                'Value suffix prepass uses full frozen-model forwards; graph/checkpoint execution differs from fused distributed training.',
                'RM generated-image scores versus real outcomes measure the combined WM/RM chain, not a simulator-observed imagined outcome.',
                'Gradient geometry is Euclidean, not Adam step geometry; no optimizer steps or parameter changes.'])
    finally:
        if native is not None:native.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--validate-only',action='store_true');args=p.parse_args()
    plan=json.loads(Path(args.plan).read_bytes());out=Path(plan['output'])
    if args.validate_only:
        print(json.dumps(admission(plan)));return
    from weaver_run.git_version import revision
    revision(ROOT,plan['git_commit']);identity=verify_owner(plan)
    # CPU parent has not imported Torch yet; check existing GPU service without changing it.
    from weaver_run.launch import mps_preflight
    mps_preflight(os.environ.get('CUDA_MPS_PIPE_DIRECTORY'),GPUS)
    os.environ.update(CUDA_VISIBLE_DEVICES=GPUS[0],MUJOCO_GL='osmesa',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',OMP_NUM_THREADS='1')
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Finite diagnostic time budget exceeded')))
    signal.alarm(plan['max_seconds'])
    try:
        receipt=admission(plan)
        if receipt!=plan['admission']:raise ValueError('Frozen checkpoint or EMA changed after queue submission')
        save(out/'started.json',dict(identity=identity,git_commit=plan['git_commit'],started_ns=time.time_ns()))
        result=execute(plan,out,receipt)
        save(out/'result.json',dict(result,finished_ns=time.time_ns()))
    except BaseException:
        save(out/'result.json',dict(complete=False,error=traceback.format_exc(),finished_ns=time.time_ns()));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
