"""UID1004 cooperative GPU4–7 queue. CPU control only; no HTTP listener remotely."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys,time

ROOT=Path('/mnt/data/lcx1/dong/need_data/experiment_queue_gpu47_20261023')
MPS_AUTHORIZATION=Path(__file__).resolve().with_name('mps_authorization.json')
TRAIN=Path('/mnt/data/lcx1/dong/need_data/weaver_libero_offline_20261022')
RUN='weaverpretrain_20261023_102831_f8c86281b8'
EVAL='weavereval_54db0c9d635cc45c'
GPUS=['GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c','GPU-fe0edfe4-b193-1af7-6a98-cbff40214f74','GPU-8c9bc264-7f09-989e-1ec7-cecd6cde782c','GPU-07ae0bbc-8923-bd0c-199e-c726821e655d']
PIPE='/dev/shm/mps_lcx1_gpu47_pzero2_20261020/pipe'
EVAL_ROOT=TRAIN/'posttraining_evaluation'
ACTIVE_STATES={'running','starting','stopping','waiting_training','waiting_gpu_idle','launch_claimed','evaluating'}
FAILED_STATES={'failed','evaluation_stopped','checker_stopped_without_evaluation','existing_launch_claim_no_retry','stopped_by_guard'}

def read(path):return json.loads(Path(path).read_text())
def atomic(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def identity(pid):
    try:
        root=Path('/proc')/str(pid);s=(root/'stat').read_text().rsplit(')',1)[1].split()
        uid=int(next(x.split()[1] for x in (root/'status').read_text().splitlines() if x.startswith('Uid:')))
        return {'pid':int(pid),'uid':uid,'birth_ticks':int(s[19]),'process_group':int(s[2]),'ppid':int(s[1]),'state':s[0]}
    except (OSError,ValueError,StopIteration):return None
def alive(expected):
    actual=identity(expected.get('pid',-1))
    return bool(actual and actual['state'] not in ('Z','X') and all(actual[k]==expected.get(k) for k in ('pid','uid','birth_ticks')))
def command(pid):
    try:return (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    except OSError:return []

def boot_identity():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def process_environment(pid):
    pairs=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
    return dict(pair.split(b'=',1) for pair in pairs if b'=' in pair)

def mps_authorization(path=None):
    """Use the explicitly authorized MPS births on this boot, never historical PIDs."""
    value=read(MPS_AUTHORIZATION if path is None else path)
    if (value.get('schema')!='mps_authorization_v1' or value.get('uid')!=1004
        or value.get('host')!='h20-xyff52ab-0001' or value.get('gpu_uuids')!=GPUS
        or value.get('pipe')!=PIPE or value.get('log')!=str(Path(PIPE).parent/'log')
        or value.get('default_cap')!=100 or value.get('server_cap')!=100
        or value.get('boot_id')!=boot_identity()):
        raise ValueError('MPS authorization scope/boot differs')
    cvd=','.join(GPUS).encode()
    for role,executable in (('control',b'nvidia-cuda-mps-control'),('server',b'nvidia-cuda-mps-server')):
        expected=value.get(role,{})
        if (not isinstance(expected,dict) or expected.get('uid')!=1004
            or any(type(expected.get(k)) is not int or expected[k]<=0 for k in ('pid','birth_ticks'))
            or not alive(expected)):
            raise ValueError('Authorized MPS '+role+' PID birth changed')
        argv=[item for item in command(expected['pid']) if item]
        if not argv or argv[0].rsplit(b'/',1)[-1]!=executable:
            raise ValueError('Authorized MPS '+role+' command differs')
        actual=identity(expected['pid'])
        if role=='server' and actual.get('ppid')!=value['control']['pid']:
            raise ValueError('Authorized MPS server parent differs')
        env=process_environment(expected['pid'])
        if env.get(b'CUDA_MPS_PIPE_DIRECTORY')!=PIPE.encode() or env.get(b'CUDA_VISIBLE_DEVICES')!=cvd:
            raise ValueError('Authorized MPS '+role+' environment differs')
    if value['server']['pid']==value['control']['pid']:
        raise ValueError('MPS server/control identities must differ')
    return value

def pipeline_gate(training,evaluation,completed,live=alive,expected_eval=EVAL,cohort_empty_verified=False):
    """A single reservation spans training, saving, CPU admission, and evaluation."""
    if training.get('run_id')!=RUN or evaluation.get('training_run_id')!=RUN or evaluation.get('evaluation_run_id')!=expected_eval:
        return {'ready':False,'reason':'预约的训练或评估身份不一致，等待诊断','label':'身份异常'}
    known=training.get('owned',[])+evaluation.get('owned',[])+[training.get('supervisor',{}),training.get('child',{}),evaluation.get('supervisor',{}),evaluation.get('child',{})]
    any_live=any(live(x) for x in known if x)
    if training.get('state') in ('running','starting','stopping'):
        healthy=live(training.get('supervisor',{}))
        if not any_live and cohort_empty_verified:
            return {'ready':True,'reason':'前项监督失效且任务进程已全退，记为失败并继续','label':'失效任务已跳过','outcome':'failed_lost_training'}
        return {'ready':False,'reason':'当前 WEAVER 训练及后续效果评估预约 GPU4–7' if healthy else '训练监督身份失效，保留预约等待诊断','label':'训练 → 评估预约','healthy':healthy}
    if training.get('state') in FAILED_STATES:
        if not any_live and cohort_empty_verified:
            return {'ready':True,'reason':'前项训练失败且已清退，继续后续队列','label':'前项失败，继续队列','outcome':'failed_training_cleaned'}
        return {'ready':False,'reason':'前项训练失败，等待相关进程清退','label':'等待清退'}
    if training.get('state')!='completed' or training.get('exit_code')!=0 or not training.get('cleanup',{}).get('all_owned_exited'):
        return {'ready':False,'reason':'训练尚未完成安全保存与清退，保留预约','label':'保存 / 诊断'}
    if any(live(x) for x in training.get('owned',[])+[training.get('supervisor',{})]):
        return {'ready':False,'reason':'等待训练相关进程完全退出','label':'等待清退'}
    if evaluation.get('state') in ACTIVE_STATES:
        if not any_live and cohort_empty_verified:
            return {'ready':True,'reason':'评估监督已失效且相关进程全退，记为失败并继续','label':'失效评估已跳过','outcome':'failed_lost_evaluation'}
        return {'ready':False,'reason':'自动评估尚未完成；交接期间也保留 GPU4–7','label':'等待 / 执行效果评估','healthy':live(evaluation.get('supervisor',{}))}
    if evaluation.get('state') in FAILED_STATES:
        if not any_live and (evaluation.get('cleanup',{}).get('all_owned_exited') or cohort_empty_verified):
            return {'ready':True,'reason':'前项评估失败或超时，相关进程已清退，继续后续队列','label':'前项失败，继续队列','outcome':'failed_evaluation_cleaned'}
        return {'ready':False,'reason':'前项评估失败，等待相关进程清退','label':'等待清退'}
    if evaluation.get('state') not in ('completed','partial') or not evaluation.get('cleanup',{}).get('all_owned_exited'):
        return {'ready':False,'reason':'评估状态异常或清退未确认，保留预约等待诊断','label':'等待诊断'}
    if completed.get('evaluation_run_id')!=expected_eval or completed.get('expected_training_run_id')!=RUN or type(completed.get('complete')) is not bool:
        if not any_live and cohort_empty_verified:
            return {'ready':True,'reason':'前项已退出但完成回执缺失或无效，记为失败并继续','label':'前项回执异常，继续队列','outcome':'failed_invalid_evaluation_receipt'}
        return {'ready':False,'reason':'评估结果回执尚未核实，保留预约','label':'等待评估回执'}
    if any_live:
        return {'ready':False,'reason':'等待评估进程退出','label':'等待清退'}
    return {'ready':True,'reason':'训练与评估预约已结束，继续核对实际 GPU 空闲','label':'预约完成'}

def evaluation_descriptor(root=EVAL_ROOT):
    """Follow the current hash-bound evaluation rather than a historical failed run."""
    pointer=root/'active_evaluation.json'
    value=read(pointer) if pointer.exists() else {'version':'v1','evaluation_run_id':EVAL,
        'plan':str(root/'evaluator_plan_v1.json'),'status':str(root/'checker_v1/status.json'),'output':str(root/'results_v1')}
    version=value.get('version','')
    if not re.fullmatch(r'v[1-9][0-9]{0,2}',version):raise ValueError('Invalid active evaluation version')
    expected={'plan':root/('evaluator_plan_'+version+'.json'),'status':root/('checker_'+version)/'status.json','output':root/('results_'+version)}
    for k,p in expected.items():
        if value.get(k)!=str(p) or p.resolve()!=p:raise ValueError('Evaluation pointer redirects: '+k)
    raw=expected['plan'].read_bytes();sha=hashlib.sha256(raw).hexdigest();plan=json.loads(raw)
    if (value.get('evaluation_run_id')!='weavereval_'+sha[:16] or plan.get('expected_training_run_id')!=RUN
        or plan.get('uid')!=1004 or plan.get('gpu_uuids')!=GPUS
        or plan.get('evaluation_output')!=value['output']):raise ValueError('Active evaluation plan identity differs')
    status=read(expected['status'])
    if (status.get('uid')!=1004 or status.get('evaluation_run_id')!=value['evaluation_run_id'] or status.get('training_run_id')!=RUN
        or status.get('plan_sha256')!=sha):raise ValueError('Active evaluation status identity differs')
    result=read(expected['output']/'completed.json') if (expected['output']/'completed.json').exists() else {}
    return value,status,result

def cohort_processes(markers,known):
    """Exact inherited marker or recorded birth/tree; no signals and no UID-only ownership."""
    identities={}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            actual=identity(int(p.name))
            if actual and actual['uid']==1004 and actual['state'] not in ('Z','X'):identities[actual['pid']]=actual
    owned={x['pid']:identities[x['pid']] for x in known if x and x.get('pid') in identities and alive(x)}
    wanted={('%s=%s'%pair).encode() for pair in markers}
    for pid,actual in identities.items():
        try:
            if wanted.intersection((Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')):owned[pid]=actual
        except OSError:pass # Unreadable, unproved unrelated session helpers are never owned.
    changed=True
    while changed:
        changed=False
        for pid,actual in identities.items():
            if pid not in owned and actual.get('ppid') in owned:owned[pid]=actual;changed=True
    return list(owned.values())

def resource_admission(gpus,mps,apps):
    if mps['clients'] or any(x['pid']!=mps.get('server') for x in apps):return {'ready':False,'reason':'等待前项或队列外 GPU 客户端退出','label':'GPU 正被占用'}
    if any(g['memory_total_mib']-g['memory_used_mib']<32768 for g in gpus):return {'ready':False,'reason':'GPU 可用显存不足 32 GiB，等待释放','label':'显存未释放'}
    return {'ready':True,'reason':'GPU4–7 客户端已清空且显存已释放','label':'资源已释放'}

class QuietAdmission:
    def __init__(self,seconds=5):self.seconds=seconds;self.first=None
    def reset(self):self.first=None
    def check(self,gate,now):
        if not gate.get('ready'):self.first=None;return gate
        if self.first is None:self.first=now
        if now-self.first<self.seconds:return {'ready':False,'reason':'前项已退出，等待第二次确认 GPU 清退','label':'确认资源清退'}
        return gate

def query(argv,input_text=None):
    p=subprocess.run(argv,input=input_text,text=True,capture_output=True,timeout=8,env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=PIPE))
    if p.returncode:raise RuntimeError(p.stderr.strip() or 'Resource query failed')
    return p.stdout
def gpu_snapshot():
    authorization=mps_authorization();server=authorization['server']['pid']
    raw=query(['nvidia-smi','--id='+','.join(GPUS),'--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'])
    gpus=[]
    for line in raw.splitlines():
        parts=[x.strip() for x in line.split(',')]
        gpus.append(dict(index=int(parts[0]),uuid=parts[1],memory_used_mib=float(parts[2]),memory_total_mib=float(parts[3]),utilization=float(parts[4])))
    if [(g['index'],g['uuid']) for g in gpus]!=list(zip(range(4,8),GPUS)):raise ValueError('GPU UUID/index mapping changed')
    mps={}
    for field,cmd in [('servers','get_server_list'),('clients','get_client_list '+str(server)),('default_cap','get_default_active_thread_percentage'),('server_cap','get_active_thread_percentage '+str(server))]:
        value=query(['nvidia-cuda-mps-control'],cmd+'\n').strip()
        mps[field]=[int(x) for x in value.split()] if field in ('servers','clients') else float(value)
    mps.update(server=server,control=authorization['control']['pid'],boot_id=authorization['boot_id'])
    if mps['servers']!=[server] or mps['default_cap']!=100 or mps['server_cap']!=100:raise ValueError('MPS identity/cap changed')
    apps=[]
    raw=query(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'])
    for line in raw.splitlines():
        if not line.strip():continue
        pid,gpu=[x.strip() for x in line.split(',')]
        if gpu in GPUS:apps.append({'pid':int(pid),'uuid':gpu})
    # Reject a race in which either daemon exited/restarted during the queries.
    if mps_authorization()!=authorization:raise ValueError('MPS authorization changed during observation')
    return gpus,mps,apps

def collect():
    train=read(TRAIN/'micro8_formal_v1/runstatus.json');progress=read(TRAIN/'micro8_formal_v1/progress.json')
    descriptor,evaluation,completed=evaluation_descriptor()
    known=train.get('owned',[])+[train.get('supervisor',{}),train.get('child',{})]
    markers=[('WEAVER_OFFLINE_RUN_ID',RUN)]
    for path in EVAL_ROOT.glob('checker_v*/status.json'):
        if not re.fullmatch(r'checker_v[1-9][0-9]{0,2}',path.parent.name):continue
        old=read(path)
        if old.get('training_run_id')!=RUN:raise ValueError('Historical evaluation training identity differs')
        known+=old.get('owned',[])+[old.get('supervisor',{}),old.get('child',{})]
        markers.append(('WEAVER_EVAL_RUN_ID',old['evaluation_run_id']))
    external=cohort_processes(markers,known)
    gate=pipeline_gate(train,evaluation,completed,expected_eval=descriptor['evaluation_run_id'],cohort_empty_verified=not external)
    now=time.time_ns();gpus,mps,apps=gpu_snapshot()
    if gate['ready']:
        if external:gate={'ready':False,'reason':'前项仍有相关 CPU/GPU 进程，等待全部退出','label':'等待清退'}
        else:
            physical=resource_admission(gpus,mps,apps)
            if not physical['ready']:gate=physical
    # Re-read the small pointer before admitting, so a newly registered v3 is never skipped.
    if evaluation_descriptor()[0]!=descriptor:gate={'ready':False,'reason':'当前评估预约已更新，重新核验','label':'重新核验预约'}
    from store import QueueStore
    active=QueueStore(ROOT).active()
    if active:
        gate={'ready':False,'reason':active[0]['name'],'label':'队列任务运行中' if active[0]['state']=='running' else '前项正在确认清退'}
    return {'observed_ns':now,'host':'h20-xyff52ab-0001','scope':'UID1004 · GPU4–7 · 整组独占预约',
        'training':{'run_id':train['run_id'],'state':train['state'],'steps':progress['successful_updates'],'target':3200,
            'loss':progress.get('loss'),'last_step_seconds':progress.get('update_wall_seconds'),'heartbeat_age_seconds':(now-train.get('heartbeat_ns',now))/1e9},
        'evaluation':{'state':evaluation['state'],'run_id':evaluation['evaluation_run_id'],'heartbeat_age_seconds':(now-evaluation.get('heartbeat_ns',now))/1e9,
            'version':descriptor['version'],'log_path':str(Path(descriptor['status']).parent/'evaluation.log')},
        'reservation':gate,'external_process_count':len(external),'gpus':gpus,'mps':mps,'compute_apps':apps,'worker':{}}

def supervised_snapshot(worker,cached,refresh,collector=collect):
    # Deadline/cleanup supervision is independent of NVML/MPS availability.
    state=worker.tick()
    if refresh:
        try:cached=collector()
        except Exception as exc:
            cached=dict(cached) if cached else {'observed_ns':time.time_ns()}
            cached.update(error=str(exc),reservation={'ready':False,'reason':str(exc),'label':'资源状态待核实'})
    cached['worker']=dict(state,heartbeat_ns=time.time_ns(),identity=identity(os.getpid()))
    return cached

def api(root,request):
    from store import QueueStore
    store=QueueStore(root);op=request.get('op')
    if op=='state':
        path=root/'state.json';snapshot=read(path) if path.exists() else None
        fresh=bool(snapshot and not snapshot.get('error') and time.time_ns()-snapshot['observed_ns']<90e9)
        return {'ok':fresh,'snapshot':snapshot,'jobs':store.list(),'read_only':False,'error':None if fresh else '调度守护进程状态缺失或超过 90 秒未更新'}
    if op=='create':
        value=request['job'];job=store.create(value['name'],value['argv'],value['cwd'],value['timeout_seconds'],submission_id=value['submission_id'],manual=value.get('manual',False));return {'ok':True,'job':job}
    if op=='cancel':return {'ok':True,'job':store.cancel(request['id'])}
    if op=='move':return {'ok':True,'job':store.reorder(request['id'],request['delta'])}
    if op=='log':
        key=request['id']
        if key=='external-training':path=TRAIN/'micro8_formal_v1/training.log';state='external'
        elif key=='external-evaluation':
            descriptor,status,_=evaluation_descriptor();path=Path(descriptor['status']).parent/'evaluation.log';state=status['state']
        else:
            job=store.get(key);path=root/'jobs'/job['id']/'output.log';state=job['state']
        if not path.exists():return {'ok':True,'text':'日志尚未生成。','truncated':False,'state':state}
        with path.open('rb') as f:
            size=f.seek(0,2);f.seek(max(0,size-65536));value=f.read().decode(errors='replace')
        return {'ok':True,'text':value,'truncated':size>65536,'state':state}
    raise ValueError('Unknown queue operation')

def watch(root):
    import fcntl
    from worker import QueueWorker
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'dispatcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(GPUS),CUDA_MPS_PIPE_DIRECTORY=PIPE,CUDA_MPS_LOG_DIRECTORY=str(Path(PIPE).parent/'log'),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for k in list(env):
        if k.startswith(('NSYS','CUDA_INJECTION','NVTX_INJECTION')) or k in ('LD_PRELOAD','WEAVER_OFFLINE_RUN_ID','WEAVER_EVAL_RUN_ID','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','CUDA_MPS_PINNED_DEVICE_MEM_LIMIT'):env.pop(k)
    cached={};last=0;quiet=QuietAdmission()
    def record_outcome(snapshot):
        gate=snapshot.get('reservation',{})
        if gate.get('outcome'):
            path=root/'external_outcomes.json';outcomes=read(path) if path.exists() else {}
            key=snapshot['evaluation']['run_id']+':'+gate['outcome']
            if key not in outcomes:
                outcomes[key]={'observed_ns':time.time_ns(),'evaluation':snapshot['evaluation'],'outcome':gate['outcome'],'reason':gate['reason']}
                atomic(path,outcomes)
    def gate():
        nonlocal cached,last
        if cached and not cached.get('reservation',{}).get('ready') and time.monotonic()-last<30:
            return quiet.check(cached['reservation'],time.monotonic())
        # Re-read receipts and physical clients immediately before a dispatch.
        last=time.monotonic()
        try:cached=collect()
        except Exception as exc:
            cached=dict(cached) if cached else {'observed_ns':time.time_ns()}
            cached.update(error=str(exc),reservation={'ready':False,'reason':str(exc),'label':'资源状态待核实'})
        record_outcome(cached)
        return quiet.check(cached['reservation'],time.monotonic())
    def recovery_gate():
        gpus,mps,apps=gpu_snapshot();return resource_admission(gpus,mps,apps)
    worker=QueueWorker(root,gate,env,recovery_gate_callback=recovery_gate)
    while True:
        try:
            refresh=not cached or time.monotonic()-last>=30
            if refresh:last=time.monotonic()
            cached=supervised_snapshot(worker,cached,refresh)
            if not cached.get('reservation',{}).get('ready') or cached['worker']['state'] in ('running','stopping','blocked'):
                quiet.reset()
            record_outcome(cached)
            atomic(root/'state.json',cached)
        except Exception as exc:
            value=dict(cached) if cached else {'observed_ns':time.time_ns()}
            value.update(error=str(exc),worker={'state':'blocked','error':str(exc),'identity':identity(os.getpid()),'heartbeat_ns':time.time_ns()},reservation={'ready':False,'reason':str(exc),'label':'等待诊断'})
            atomic(root/'state.json',value)
        time.sleep(5)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['api','watch']);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    if os.getuid()!=1004 or a.root.resolve()!=ROOT:raise ValueError('Approved UID1004 queue root required')
    if a.mode=='watch':watch(a.root)
    else:
        try:print(json.dumps(api(a.root,json.loads(sys.stdin.read(65537))),ensure_ascii=False,allow_nan=False))
        except Exception as exc:print(json.dumps({'ok':False,'error':str(exc)},ensure_ascii=False));sys.exit(1)

if __name__=='__main__':main()
