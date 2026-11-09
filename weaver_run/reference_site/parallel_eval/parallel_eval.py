"""Four independent GPU replicas, one immutable paired plan and one deadline.

The managed Git queue wrapper invokes this entry after final training audit.
Runtime requires a stopped/audited run AND an existing four-GPU queue reservation.
Importing this coordinator initializes no CUDA, Torch, Ray or simulator.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

from plan_contract import (APPROVED_GPUS, load_base, make_shards, validate_shards,
                           merge_rows, choose_deadline)

RUN_MARKER='PARALLEL_POLICY_EVAL_RUN_TOKEN'


def same_identity(expected,actual,run_token):
    if not isinstance(expected,dict) or not isinstance(actual,dict):return False
    return (isinstance(run_token,str) and bool(run_token)
        and all(type(expected.get(k)) is int and type(actual.get(k)) is int
                and expected[k]==actual[k] for k in ('pid','uid','birth'))
        and expected['pid']>0 and expected['uid']==1004 and expected['birth']>=0
        and expected.get('run_token')==actual.get('run_token')==run_token
        and actual.get('state') not in ('Z','X'))


def signal_exact(expected,run_token,probe,send,signal_number):
    """Injectable identity gate; production cleanup uses the pinned pidfd adapter."""
    if not isinstance(expected,dict) or type(expected.get('pid')) is not int:return False
    if not same_identity(expected,probe(expected['pid']),run_token):return False
    send(expected['pid'],signal_number)
    return True


def completion_acceptance(plan,shards,worker_rows,worker_completions,exit_codes,cleanup,run_token,base):
    errors=[]
    rows=[]
    try:
        rows=merge_rows(plan,shards,worker_rows,run_token,base=base)
    except (ValueError,KeyError,TypeError) as exc:
        errors.append({'type':'merge_rejected','error':str(exc)})
    for index in range(4):
        item=worker_completions.get(index)
        worker=shards['workers'][index]
        expected=dict(plan_sha256=plan['sha256'],shards_sha256=shards['sha256'],run_token=run_token,
            worker_index=index,gpu_uuid=worker['gpu_uuid'],complete=True,errors=[],exit_code=0,
            expected_episode_records=worker['expected_episode_records'],
            completed_episode_records=worker['expected_episode_records'])
        if len(worker_rows.get(index,[]))!=worker['expected_episode_records']:
            errors.append({'type':'worker_ledger_count_differs','worker_index':index})
        if (not isinstance(item,dict) or any(item.get(k)!=v for k,v in expected.items())
                or type(item.get('worker_index')) is not int or item.get('complete') is not True
                or type(item.get('exit_code')) is not int):
            errors.append({'type':'worker_completion_rejected','worker_index':index})
        if type(exit_codes.get(index)) is not int or exit_codes[index]!=0:
            errors.append({'type':'worker_exit_not_zero','worker_index':index})
    for mapping in (worker_completions,exit_codes):
        if any(type(k) is not int for k in mapping) or set(mapping)!=set(range(4)):
            errors.append({'type':'worker_identity_set_differs'})
    if not isinstance(cleanup,dict) or cleanup.get('survivors')!=[] or cleanup.get('errors')!=[]:
        errors.append({'type':'owned_process_exit_unconfirmed'})
    try:summary=base.summarize(plan,rows,errors=errors)
    except (ValueError,KeyError,TypeError) as exc:
        errors.append({'type':'summary_rejected','error':str(exc)})
        summary=base.summarize(plan,[],errors=errors)
    summary.update(shards_sha256=shards['sha256'],run_token=run_token,cleanup=cleanup)
    return summary


def load_adapter(directory, source, git_commit):
    """Use the admission adapter from this same clean queued Git revision."""
    directory=Path(directory).resolve(strict=True)
    source=Path(source).resolve(strict=True)
    expected=Path(__file__).resolve().parent.parent/'queue_wrapper'
    if (directory!=expected or source!=expected.parent.parent/'engine'
            or (directory/'training_entry.py').is_symlink()):
        raise ValueError('Evaluation and queue adapter must use the same Git checkout')
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location('parallel_eval_queue_admission',directory/'training_entry.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_git(source,git_commit)
    return module


def stopped_admission(adapter,master,source,run_id,git_commit):
    if not hasattr(os,'getuid') or os.getuid()!=1004:raise ValueError('Runtime requires admitted UID1004')
    if not os.environ.get('EXPERIMENT_QUEUE_JOB_ID'):raise ValueError('No active queue reservation marker')
    receipt=adapter.read_json(master/'training_queue_receipt.json')
    if (receipt.get('run_id')!=run_id or receipt.get('state')!='succeeded'
            or receipt.get('job_id')!=os.environ['EXPERIMENT_QUEUE_JOB_ID']
            or receipt.get('runner_exit_code')!=0
            or receipt.get('final_source_validation',{}).get('git_commit')!=git_commit
            or receipt.get('all_owned_exited') is not True
            or receipt.get('acceptance',{}).get('passed') is not True
            or receipt.get('acceptance',{}).get('reasons')!=[]
            or receipt.get('checkpoint_cpu_audit',{}).get('valid') is not True):
        raise ValueError('This run has not completed its boundary save, audit and training cleanup')
    adapter.validate_mps_receipt(adapter.read_json(master/'mps.json'))
    # Fail if a prior training actor remains even if the saved receipt is stale.
    job=os.environ['EXPERIMENT_QUEUE_JOB_ID']
    ancestors={os.getpid()}
    current=adapter.proc_identity(os.getpid())
    while current and current.get('ppid',0)>1 and current['ppid'] not in ancestors:
        ancestors.add(current['ppid']);current=adapter.proc_identity(current['ppid'])
    if any(p['pid'] not in ancestors for p in adapter.marked_children(job)):
        raise ValueError('Queue-owned training/evaluation processes remain')
    env=adapter.launch_environment(master,source,os.environ)
    env['CUDA_VISIBLE_DEVICES']=','.join(APPROVED_GPUS)
    env['ACCERL_REPO_ROOT']=str(source.resolve().parent.parent)
    adapter.wait_idle(source,env,seconds=30)
    return env


def prepare_plan(base,args,deadline):
    if args.plan is not None:
        plan=json.loads(args.plan.read_text(encoding='utf-8'))
        if plan!=base.make_plan(plan['checkpoints'],initial_state_count=50,seed=plan['seed'],
                                modes=plan['modes'],repeats=plan['repeats']):
            raise ValueError('Supplied global plan differs from the reference protocol')
    else:
        checkpoints=[]
        for name,directory in [('baseline','checkpoints_initial'),('final','checkpoints')]:
            override=getattr(args,name+'_checkpoint',None)
            payload,item=base.load_snapshot(override or args.master/'formal_v3/training'/directory/'latest_weights.pt',deadline)
            checkpoints.append(dict(item,name=name))
            del payload
            gc.collect()
        plan=base.make_plan(checkpoints,modes=args.modes.split(','),seed=args.seed,repeats=args.repeats)
    recipes=[c['metadata']['args'] for c in plan['checkpoints']]
    if recipes[0]!=recipes[1] or recipes[0]['benchmark']!='libero_spatial':
        raise ValueError('Paired Policy recipes differ')
    if any(c['metadata']['training_run_id']!=args.run_id for c in plan['checkpoints']):
        raise ValueError('Checkpoint run identity differs from the stopped run')
    return plan


def collect(base,output,plan,shards,token,children,cleanup,errors):
    worker_rows={};completions={}
    can_repair=(cleanup.get('survivors')==[] and cleanup.get('errors')==[]
                and len(children)==4 and all(p.poll() is not None for p in children.values()))
    for i in range(4):
        directory=output/'workers'/('worker_'+str(i))
        # Owned workers have exited. Repair only an interrupted final JSONL tail,
        # preserving earlier fsynced records; nonzero exits still fail acceptance.
        try:worker_rows[i]=base.read_rows(directory/'episodes.jsonl',plan,repair_partial_tail=can_repair)
        except (OSError,ValueError) as exc:
            worker_rows[i]=[];errors.append({'type':'worker_ledger_error','worker_index':i,'error':str(exc)})
        try:completions[i]=json.loads((directory/'completed.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):completions[i]={}
    exit_codes={i:p.returncode for i,p in children.items()}
    summary=completion_acceptance(plan,shards,worker_rows,completions,exit_codes,cleanup,token,base)
    if errors:
        summary['errors'].extend(errors);summary['complete']=False
    try:rows=merge_rows(plan,shards,worker_rows,token,base=base)
    except (ValueError,KeyError,TypeError):rows=[]
    target=output/'episodes.jsonl';temporary=output/'episodes.jsonl.tmp'
    with temporary.open('w',encoding='utf-8',newline='\n') as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,target)
    base.atomic_json(output/'summary.json',summary)
    base.atomic_json(output/'completed.json',{**summary,'initial_state_count':50,'modes':plan['modes'],
        'checkpoint_sha256':{c['name']:c['sha256'] for c in plan['checkpoints']}})
    return summary


def run_workers(base,adapter,source,output,plan,shards,env,token,deadline,reference_source=None):
    children={};owned=[];logs=[];errors=[];cleanup={'survivors':[],'errors':[]}
    received=[]
    def interrupted(number,frame):received.append(number)
    prior_signals={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        for worker in shards['workers']:
            base.check_deadline(deadline,35)
            if received:raise InterruptedError('Coordinator interrupted before launch')
            index=worker['worker_index']
            worker_env=dict(env,CUDA_VISIBLE_DEVICES=worker['gpu_uuid'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                PARALLEL_POLICY_EVAL_RUN_TOKEN=token)
            log=(output/('worker_'+str(index)+'.log')).open('ab');logs.append(log)
            command=[sys.executable,str(Path(__file__).with_name('parallel_worker.py')),'--source',str(source),
                '--plan',str(output/'plan.json'),'--shards',str(output/'shards.json'),'--worker-index',str(index),
                '--output-root',str(output),'--run-token',token,'--deadline-unix',str(deadline-30)]
            if reference_source is not None:command+=['--reference-source',str(reference_source)]
            child=subprocess.Popen(command,cwd=source,env=worker_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            children[index]=child
            identity=adapter.proc_identity(child.pid)
            marker=(RUN_MARKER+'='+token).encode()
            if (not identity or identity['uid']!=1004 or
                    marker not in (Path('/proc')/str(child.pid)/'environ').read_bytes().split(b'\0')):
                raise RuntimeError('Worker PID/UID/birth/execution marker could not be verified')
            owned.append(identity)
        while any(child.poll() is None for child in children.values()):
            if received:raise InterruptedError('Coordinator received signal '+str(received[0]))
            if time.time()>=deadline-25:raise TimeoutError('Shared evaluation budget exhausted')
            if any(p.poll() not in (None,0) for p in children.values()):raise RuntimeError('An evaluation worker failed')
            time.sleep(.25)
    except BaseException as exc:
        errors.append({'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
    finally:
        try:
            remaining=adapter.marked_children(token,owned,marker_name=RUN_MARKER)
            if remaining:
                cleanup.update(adapter.scoped_cleanup(token,remaining,marker_name=RUN_MARKER))
            else:cleanup['survivors']=[]
            for child in children.values():
                try:child.wait(timeout=1)
                except subprocess.TimeoutExpired:cleanup['errors'].append('Worker wait timeout: '+str(child.pid))
        except BaseException as exc:cleanup['errors'].append(repr(exc))
        for log in logs:log.close()
        for sig,handler in prior_signals.items():signal.signal(sig,handler)
    return collect(base,output,plan,shards,token,children,cleanup,errors)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','queue-wrapper','master','output-root'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--git-commit',required=True)
    parser.add_argument('--reference-source',type=Path)
    parser.add_argument('--baseline-checkpoint',type=Path)
    parser.add_argument('--final-checkpoint',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--max-seconds',type=float,default=7200)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--seed',type=int,default=20261026)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--modes',default='greedy,stochastic')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    output=args.output_root
    # Refuse symlinked artifacts before any writes. Keep separate from policy_eval.
    if output.is_symlink():raise ValueError('Evaluation output cannot be a symlink')
    output.mkdir(parents=True,exist_ok=True)
    if any(path.is_symlink() for path in output.iterdir()):raise ValueError('Evaluation artifacts cannot be symlinks')
    if str(args.source) not in sys.path:sys.path.insert(0,str(args.source))
    base=load_base(args.reference_source or args.source)
    lock=output/'coordinator.lock'
    # A stale lock needs explicit operator review, never automatic take-over.
    with lock.open('x',encoding='utf-8') as handle:handle.write(str(os.getpid()))
    try:
        started_path=output/'started.json'
        prior=json.loads(started_path.read_text(encoding='utf-8')) if started_path.exists() else None
        if prior is not None and not args.resume:raise ValueError('Existing evaluation requires explicit --resume')
        if prior is None and any((output/name).exists() for name in ('workers','episodes.jsonl','completed.json')):
            raise ValueError('Existing results have no execution identity')
        deadline=choose_deadline(time.time(),args.max_seconds,args.deadline_unix,prior)
        adapter=load_adapter(args.queue_wrapper,args.source,args.git_commit)
        env=None if args.validate_only else stopped_admission(
            adapter,args.master,args.source,args.run_id,args.git_commit)
        plan=prepare_plan(base,args,deadline)
        shards=make_shards(plan)
        for name,value in [('plan.json',plan),('shards.json',shards)]:
            path=output/name
            if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=value:
                raise ValueError('Existing output belongs to another '+name)
            base.atomic_json(path,value)
        if args.validate_only:
            print(json.dumps({'validated':True,'scope':'CPU metadata/plan only; no GPU launch',
                'expected_episode_records':plan['expected_episode_records'],'plan_sha256':plan['sha256']}))
            return 0
        token=prior['run_token'] if prior else uuid.uuid4().hex
        if adapter.marked_children(token,marker_name=RUN_MARKER):
            raise ValueError('Prior evaluation workers are still alive; resume must not duplicate them')
        identity=dict(run_token=token,plan_sha256=plan['sha256'],shards_sha256=shards['sha256'],
            run_id=args.run_id,git_commit=args.git_commit,
            queue_job_id=os.environ['EXPERIMENT_QUEUE_JOB_ID'],deadline_unix=deadline)
        if prior and any(prior.get(k)!=identity[k] for k in ('run_id','git_commit','queue_job_id','plan_sha256','shards_sha256')):
            raise ValueError('Resume execution/queue identity differs')
        base.atomic_json(started_path,{**identity,'started_ns':prior['started_ns'] if prior else time.time_ns()})
        result=run_workers(base,adapter,args.source,output,plan,shards,env,token,deadline,args.reference_source)
        print(json.dumps({'complete':result['complete'],'episodes':result['completed_episode_records'],
                          'errors':result['errors']},ensure_ascii=False))
        return 0 if result['complete'] else 2
    finally:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
