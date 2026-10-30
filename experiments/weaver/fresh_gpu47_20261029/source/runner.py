"""Trainable Policy and native WEAVER/Reward-head bundle until user stop."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid

ROOT = Path('/mnt/data/lcx1/dong/AcceRL-scheduler')
SOURCE = Path(__file__).resolve().parent
import gpu47_safety as safe
from deadline_guard import guard_decision
from world_backend_audit import validate_world_zero2_receipts
from policy_batch_receipts import validate_policy_batch_modes_from_db
DEADLINE_UNIX = None  # Compatibility metadata only; never a wall-clock stop.
TZ = dt.timezone(dt.timedelta(hours=8))
TERMINAL = {'completed', 'failed', 'stopped_by_user'}


def stamp():
    return dt.datetime.now(TZ).isoformat(timespec='seconds')


def load(data):
    from candidate_paths import validate_candidate_state
    return validate_candidate_state(data, json.loads((data / 'runnerstatus.json').read_text()), root=safe.MASTER)


def measurement_configuration(active_kinds=('policy', 'ctrl', 'reward')):
    return {'nsight_collection': False,
            'nsight_capture': {'nvtx_enabled': False, 'policy_steps': [],
                'state': 'disabled_for_ipc_validation', 'actual_enabled': False},
            'hardware_metrics': {'enabled': False, 'status': 'not_collected_permission_restricted',
                                 'sm_active': None, 'occupancy': None},
            'weight_transports': {kind: 'cpu_ray' if kind == 'reward' else 'cuda_ipc' for kind in active_kinds},
            'ray_object_store_bytes': 328 * 1024**3}



def _publication_version_from_sqlite(value):
    """Decode events.model_version TEXT without coercing malformed evidence.

    SQLite's declared TEXT affinity stores the writer's integer version as its
    canonical decimal string. Receipt validators continue to require an int;
    only this database read boundary accepts that exact storage representation.
    """
    if type(value) is int and value >= 0:
        return value
    if type(value) is str:
        import re
        if re.fullmatch(r'(?:0|[1-9][0-9]*)', value):
            return int(value)
    raise ValueError('SQLite publication model_version must be a canonical nonnegative integer')


def collect(data, state):
    from world_backends.runtime import service_plan_from_state
    plan = service_plan_from_state(state)
    result = {'run_id': state['run_id'], 'state': state['state'], 'updated': stamp(),
              'mode': state['mode'], 'training': {}, 'source': str(SOURCE),
              'until_manual_stop': True, 'deadline_unix': None,
              'stop_on_target': False, 'mps_active_thread_caps': None}
    result['policy_batch_schedule']={'ordinary_micro_GAS':[8,18],'joint_micro_GAS':[4,36],
                                    'joint_interval':5,'global_batch':576,'local_batch':144}
    result['warmup_steps']={kind: value for kind, value in {'policy':100,'value':100,'ctrl':20,'reward':20}.items()
                            if kind == 'value' or kind in plan.active_kinds}
    result['service_plan'] = plan.to_dict()
    result['world_backend_config'] = state.get('world_backend_config', {})
    result.update(measurement_configuration(plan.active_kinds))
    result['measurement_injection_verified'] = state.get('measurement_injection_verified', False)
    result['nsight_capture']['actual_enabled'] = result['measurement_injection_verified']
    if state.get('measurement_configuration_error'):
        result['measurement_configuration_error'] = state['measurement_configuration_error']
    db = data / 'training/collection.sqlite'
    if not db.exists():
        return result
    c = sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True, timeout=2)
    c.row_factory = sqlite3.Row
    c.execute('pragma query_only=on')
    if not c.execute("select 1 from sqlite_master where type='table' and name='events'").fetchone():
        c.close()
        return result  # NVML sampler creates its tables before model telemetry.
    c.set_progress_handler(lambda: int(time.monotonic() - begin > 12), 10000)
    begin = time.monotonic()
    try:
        result['training'] = safe.rank_updates(c, state['run_id'])
        for disabled in set(('policy', 'ctrl', 'reward')) - set(plan.active_kinds):
            inactive = result['training'].pop(disabled, {})
            if inactive.get('global_updates', 0) or any(
                    row.get('latest_ns') is not None or row.get('engine_global_steps', 0)
                    or row.get('successful_updates', 0) for row in inactive.get('ranks', {}).values()):
                raise RuntimeError('Disabled '+disabled+' emitted training receipts')
        result['placements'] = safe.placement_evidence(c, state['run_id'], service_plan=plan.to_dict())
        mode_rows = c.execute('SELECT run_id,ts_ns,actor,actor_id,pid,details_json FROM events '
            'WHERE run_id=? AND workload IS NULL AND kind=? ORDER BY ts_ns DESC,id DESC LIMIT 80',
            (state['run_id'], 'policy_batch_mode')).fetchall()
        result['policy_batch_modes_recent'] = [{key:row[key] for key in ('run_id','ts_ns','actor','actor_id','pid')} |
            {'details':json.loads(row['details_json'])} for row in mode_rows]
        result['policy_batch_modes_recent_scope'] = 'latest80 events only; final validation separately streams the complete run'
        for kind in ['fsdp_sharding_verified', 'synchronized_epoch_completed', 'evaluation_threshold',
                     'experiment_completed', 'split_wm_cohort_validated', 'reward_features_only_enabled',
                     'reward_checkpointing_verified', 'world_memory_policy_verified', 'policy_zero2_verified', 'world_zero2_verified',
                     'weaver_frozen_verified', 'weaver_online_trainer_verified', 'weaver_online_publication_released']:
            rows = c.execute('select ts_ns,pid,actor_id,details_json from events where run_id=? and workload is null and kind=? order by ts_ns desc limit ?',
                             (state['run_id'], kind, 8 if kind == 'fsdp_sharding_verified' else 4 if kind in {'reward_features_only_enabled', 'reward_checkpointing_verified', 'world_memory_policy_verified', 'policy_zero2_verified', 'world_zero2_verified', 'weaver_frozen_verified', 'weaver_online_trainer_verified', 'weaver_online_publication_released'} else 1)).fetchall()
            result[kind] = [{'ts_ns': ts, 'pid': pid, 'actor_id': actor_id, 'details': json.loads(j)} for ts, pid, actor_id, j in rows]
        result['world_publication_committed'] = []
        publications = (c.execute('select ts_ns,pid,actor_id,model_version,details_json from events '
            'where run_id=? and workload is null and kind=? order by ts_ns desc,id desc limit 16',
            (state['run_id'], 'weight_publication_committed')).fetchall()
            if plan.backend == 'weaver' and plan.world_trainable
                and state.get('world_backend_config',{}).get('backend_options',{}).get('transport') == 'isolated_online_ddp' else [])
        for ts, pid, actor_id, version, raw in publications:
            details = json.loads(raw)
            if details.get('kind') == 'ctrl':
                result['world_publication_committed'] = [{'ts_ns': ts, 'pid': pid, 'actor_id': actor_id,
                    'model_version': _publication_version_from_sqlite(version), 'details': details}]
                break
        result['replay'] = [tuple(row) for row in c.execute('select replay,count(*),sum(resident),sum(policy_uses),sum(ctrl_uses),sum(reward_uses) from replay_samples where run_id=? group by replay',
                                     (state['run_id'],))]
        intervals = {}
        for kind in ['split_serial_phase_begin', 'split_serial_phase_end']:
            for ts, j in c.execute('select ts_ns,details_json from events where run_id=? and workload is null and kind=? order by ts_ns desc limit 300',
                                   (state['run_id'], kind)):
                v = json.loads(j)
                key = (v['policy_step'], v['model'], v['rank'])
                intervals.setdefault(key, {})[kind.rsplit('_', 1)[1]] = ts
        overlaps = []
        for step in sorted({k[0] for k in intervals if k[0] % 5 == 0}):
            for rank in range(4):
                parts = [intervals.get((step, model, rank), {}) for model in plan.active_kinds]
                if all('begin' in p and 'end' in p for p in parts):
                    overlaps.append({'step': step, 'rank': rank,
                        ('host_phase_triple_overlap_seconds' if len(plan.active_kinds) == 3 else 'host_phase_overlap_seconds'):
                            max(0, min(p['end'] for p in parts) - max(p['begin'] for p in parts)) / 1e9})
        result['concurrent_host_phases'] = overlaps
    finally:
        c.close()
    manifest = data / 'training/checkpoints/manifest.json'
    if manifest.exists():
        m = json.loads(manifest.read_text())
        result['checkpoint'] = {k: m[k] for k in ['reason', 'steps', 'world_size', 'weights_only']}
        if 'online_world' in m.get('metadata', {}):
            result['checkpoint']['online_world'] = m['metadata']['online_world']
    return result


def report(data, state):
    from candidate_paths import validate_candidate_state
    validate_candidate_state(data, state, root=safe.MASTER)
    result = collect(data, state)
    safe.atomic_json(data/'reportstatus.json', result)
    lines = ['# GPU4–7 在线WEAVER及Policy/Reward长训', '',
             f'状态：{state["state"]}；更新：{stamp()}；run_id：`{state["run_id"]}`。',
             '全新Policy/Reward优化器及空Replay。Policy独立轮8/18、每第五轮4/36；Reward16/16；World冻结。',
             '只在用户明确要求后完整同步边界停止；没有固定时刻、15轮或85%停止。30000仅LR horizon，25完整epoch周期保存启用模型weights。',
             'Policy使用CUDA IPC，Reward保留CPU权重同步；保留两端CPU SHA校验。Ray Object Store容量328GiB。WEAVER固定3200步。',
             '预热：Policy100、Value100、Reward20，各按自身成功更新计数；峰值学习率及衰减尺度不变。World无在线优化器。',
             '记录请求ID/实际服务版本、真实episode及想象/GAE依赖DAG、Replay有序预留和实际micro曝光、成功更新及发布ACK。评估与训练数据分开；预留不等于消费，曝光不等于成功更新。',
             '本轮不采集Nsight或SM Active/Occupancy；阶段请求/批次/样本计数与端到端关联来自CPU遥测，不代表GPU资源份额。',
             '物理GPU4–7，既有MPS默认/server100%，无客户端计算配额。首更新120分钟、后续停滞60分钟、错误120秒；人工stop900秒保存宽限，不叠加窗口。',
             '失败或用户停止仅清退本轮UID/run_id/PID出生匹配进程；不自动重启，不改变MPS或其他GPU任务。', '',
             '```json', json.dumps(result, ensure_ascii=False, indent=2), '```']
    if result['service_plan']['backend'] != 'ctrl':
        plan = result['service_plan']
        external = 'reward' in plan['trainer_kinds']
        lines[0] = '# GPU4–7 World backend 候选运行记录'
        lines[3] = ('World backend='+plan['backend']+'；reward_provider='+plan['reward_provider']+
                    '；更新比例='+plan['logical_update_ratio']+'；GPU模型进程='+str(plan['expected_gpu_model_processes'])+'。')
        lines[4] = '仅在用户明确要求后完整同步边界停止；每25完整epoch保存实际启用模型的weights。'
        lines[5] = ('Policy和在线World/RM联合bundle使用CUDA IPC及四rank SHA/ACK/release；无外部Reward服务。'
            'WEAVER从已评估World微调3200步native权重初始化（该阶段RM冻结）；原生DDP4、micro4/GAS64/global1024，独立环境。')
        lines[6] = 'Policy100、Value100、World20预热。原生RM以真实LIBERO成功标签训练sigmoid/Brier分数，未经校准；想象仅按horizon终止。真实worker1、想象worker4；真实预算1000000步含reset settling。'
    for key in ('error', 'checkpoint_cpu_audit', 'sampled_gpu_peaks', 'cleanup'):
        if key in state:
            lines += ['', key+':', '```json', json.dumps(state[key], ensure_ascii=False, indent=2), '```']
    (data/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def stop(data, reason):
    directory = data/'training'
    if not directory.is_dir():
        return
    state = load(data)
    flag = directory/'stop.flag'
    if flag.exists():
        prior = json.loads(flag.read_text())
        if prior.get('run_id') != state['run_id'] or prior.get('uid') != 1004:
            raise RuntimeError('Existing stop flag identity differs; no foreign flag overwritten')
        return
    safe.atomic_json(flag, {'reason': reason, 'at': stamp(), 'requested_ns': time.time_ns(),
                            'run_id': state['run_id'], 'uid': 1004})


def guard(data, run_id):
    from candidate_paths import candidate_run_id, candidate_data
    candidate_run_id(run_id)
    candidate_data(data, root=safe.MASTER)
    failure_at = None
    while True:
        state = load(data)
        if state['run_id'] != run_id or state['state'] in TERMINAL:
            return
        observe_stop_request(data, state)
        current = safe.proc_identity(state['supervisor']['pid'])
        decision = guard_decision(state, time.time(), safe.same_process(state['supervisor'], current), failure_at)
        failure_at = decision['failure_at']
        if decision['stop_reason'] and not (data/'training/stop.flag').exists():
            stop(data, decision['stop_reason'])
        if decision['cleanup']:
            safe.cleanup(state, data, grace=15)
            state.update(state='failed', error='Independent guard: '+decision['stop_reason'])
            safe.atomic_json(data/'runnerstatus.json', state)
            return
        time.sleep(5)


def formal_environment(state, data, pre):
    env = safe.cpu_helper_environment()
    state['measurement_injection_verified'] = False
    env.update(ACCERL_RUN_ID=state['run_id'], ACCERL_RUN_STARTED_NS=str(state['started_ns']),
               ACCERL_DATA_DIR=str(data/'training'), ACCERL_SYSTEM_PROFILE='1',
               ACCERL_RUN_UNTIL_MANUAL_STOP='1', ACCERL_RUN_UNTIL_DEADLINE='0', ACCERL_STOP_ON_TARGET_85='0',
               ACCERL_SNAPSHOT_INTERVAL='25', ACCERL_CPU_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', ACCERL_NSIGHT_NVTX='0',
               ACCERL_NSIGHT_STEPS='', PYTHONUNBUFFERED='1', ACCERL_REPO_ROOT=str(ROOT),
               CUDA_VISIBLE_DEVICES=','.join(card['uuid'] for card in pre['cards']),
               CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_MPS_PIPE_DIRECTORY=pre['mps_pipe'],
               CUDA_MPS_LOG_DIRECTORY=pre['mps_log'])
    if state.get('world_backend_config_path'):
        env['ACCERL_WORLD_BACKEND_CONFIG'] = state['world_backend_config_path']
    if state.get('pipeline_config_path'):
        env['ACCERL_PIPELINE_CONFIG'] = state['pipeline_config_path']
    if state.get('experiment_root'):
        env['ACCERL_EXPERIMENT_ROOT'] = state['experiment_root']
    return env




class SaveGraceExpired(RuntimeError):
    """The fixed GPU save allowance is exhausted; do not append error grace."""


def error_wait_budget(state, now=None):
    now = time.time() if now is None else now
    limits = [120]
    if state.get('error_started_ns') is not None:
        limits.append(state['error_started_ns']/1e9+120-now)
    if state.get('stop_requested_ns') is not None:
        limits.append(state['stop_requested_ns']/1e9+900-now)
    return max(0, min(limits))


def enforce_stop_window(data, state, child, now=None):
    now = time.time() if now is None else now
    observe_stop_request(data, state, now)
    stopped = state.get('stop_requested_ns')
    if stopped is not None and now >= stopped/1e9+900 and child.poll() is None:
        raise SaveGraceExpired('Manual-stop GPU save grace of900seconds expired')


def run(data, mode, stop_on_target, world_backend_config=None, pipeline_config=None):
    if mode != 'manual' or stop_on_target:
        raise RuntimeError('This source only permits a fresh manual-stop formal run')
    if data.resolve() != (safe.MASTER/'formal_v3').resolve():
        raise RuntimeError('Manual data must be this new master/formal_v3 directory')
    from world_backends.runtime import load_backend_config, prepare_runtime
    from types import SimpleNamespace
    from candidate_paths import candidate_data, RUN_ID_PREFIX
    candidate_data(data, root=safe.MASTER)
    config_path = world_backend_config or os.environ.get('ACCERL_WORLD_BACKEND_CONFIG')
    config = load_backend_config(SimpleNamespace(world_backend_config=config_path))
    plan = prepare_runtime(config, world_size=4)
    if config.backend != 'weaver' or not plan.world_trainable or not plan.integrated_reward:
        raise ValueError('This candidate requires native trainable WEAVER and its integrated uncalibrated Reward head')
    from world_backend_audit import online_world_identity
    online_world_identity(config)
    from pipeline_budget import PipelineBudgetConfig
    budget = PipelineBudgetConfig.from_json(pipeline_config or os.environ.get('ACCERL_PIPELINE_CONFIG'))
    if os.getuid()!=1004:
        raise RuntimeError('Wrong UID')
    data.mkdir(parents=True,exist_ok=False)
    pre = safe.preflight()
    rid = RUN_ID_PREFIX+dt.datetime.now(TZ).strftime('%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    state = {'state':'starting','run_id':rid,'mode':'manual','uid':1004,'root':str(ROOT),
             'started_ns':time.time_ns(),'started':stamp(),'supervisor':safe.proc_identity(os.getpid()),
             'heartbeat_ns':time.time_ns(),'owned':[],'preflight':pre,'training_stopped':False,
             'stop_on_target':False,'deadline_unix':None,'until_manual_stop':True,'no_resume':True,
             'automatic_restart':False,'mps_active_thread_caps':None,'physical_gpus':[4,5,6,7],
             'gpu_save_grace_seconds':900,'error_grace_seconds':120,
             'base_frozen_archive_sha256':'3b030ada9172dac7856d7a6fe6ab2d7baf8a86967300049cc8d2c7243ab3a2e7',
             'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')}}
    state.update(candidate='weaver_integrated_v1', experiment_root=str(safe.MASTER), source_path=str(SOURCE))
    state['policy_batch_schedule']={'ordinary_micro_GAS':[8,18],'joint_micro_GAS':[4,36],
                                    'joint_interval':5,'global_batch':576,'local_batch':144}
    state['warmup_steps']={'policy':100,'value':100,'ctrl':20,'reward':20}
    state['world_backend_config'] = config.to_dict()
    state['service_plan'] = plan.to_dict()
    frozen_pipeline = data / 'pipeline_config.json'
    safe.atomic_json(frozen_pipeline, budget.to_dict())
    state['pipeline_config'] = budget.to_dict()
    state['pipeline_config_path'] = str(frozen_pipeline.resolve())
    if config_path or config.backend != 'ctrl' or config.reward_provider != 'external_vlm':
        frozen_config = data / 'world_backend_config.json'
        safe.atomic_json(frozen_config, config.to_dict())
        state['world_backend_config_path'] = str(frozen_config.resolve())
    state['warmup_steps'] = {kind: value for kind, value in state['warmup_steps'].items()
                             if kind == 'value' or kind in plan.active_kinds}
    state.update(measurement_configuration(plan.active_kinds))
    state['source_sha256']['run.sh']=hashlib.sha256((SOURCE/'run.sh').read_bytes()).hexdigest()
    safe.atomic_json(data/'runnerstatus.json',state)
    env = formal_environment(state,data,pre)
    processes=[]
    child=None
    with (data/'runner.log').open('a',buffering=1) as log:
        try:
            child=subprocess.Popen(['bash',str(SOURCE/'run.sh'),'--exp-name',
                'OpenVLA_IntegratedWEAVER_PolicyWorldRM_Real1_Imag4_Budget1M_Manual_gpu47_fresh'],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['child']=safe.proc_identity(child.pid)
            state['minimum_child_birth_ticks']=state['child']['birth_ticks']
            state['state']='running'
            safe.atomic_json(data/'runnerstatus.json',state)
            sampler=safe.start_sampler(data,state,log)
            guard_env = safe.cpu_helper_environment()
            guard_env['ACCERL_EXPERIMENT_ROOT'] = str(safe.MASTER)
            guard_process=subprocess.Popen([sys.executable,__file__,'--mode','guard','--data',str(data),'--run-id',rid],
                env=guard_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['guard']=safe.proc_identity(guard_process.pid)
            processes=[sampler,guard_process]
            report_error_since=None
            while True:
                safe.remember_owned(state)
                state['heartbeat_ns']=time.time_ns()
                enforce_stop_window(data,state,child)
                try:
                    result=report(data,state)
                    state.pop('report_error',None)
                    report_error_since=None
                except sqlite3.OperationalError as exc:
                    report_error_since=report_error_since or time.monotonic()
                    state['report_error']=repr(exc)
                    safe.atomic_json(data/'runnerstatus.json',state)
                    if time.monotonic()-report_error_since>120:
                        raise RuntimeError('CPU reporting unavailable for over120seconds') from exc
                    time.sleep(5)
                    continue
                safe.atomic_json(data/'runnerstatus.json',state)
                check_update_watchdogs(result,state)
                rc=child.poll()
                if rc is not None:
                    if rc!=0:
                        raise RuntimeError('Training application exit '+str(rc))
                    result=report(data,state)
                    flag=observe_stop_request(data,state)
                    result['policy_batch_mode_validation']=validate_policy_batch_modes_from_db(
                        data/'training/collection.sqlite', run_id=state['run_id'],
                        steps=result.get('training',{}).get('policy',{}).get('global_updates',0))
                    state['policy_batch_mode_validation']=result['policy_batch_mode_validation']
                    if plan.world_trainable:
                        if plan.backend == 'weaver' and state.get('world_backend_config',{}).get('backend_options',{}).get('transport') == 'isolated_online_ddp':
                            from world_backend_audit import validate_online_world_receipts
                            validate_online_world_receipts(result.get('weaver_online_trainer_verified', []), state['world_backend_config'])
                        else:
                            validate_world_zero2_receipts(result.get('world_zero2_verified', []))
                    state['manual_completion']=validate_manual_completion(result,state,flag or {})
                    state.update(return_code=0,state='stopped_by_user')
                    break
                time.sleep(8)
        except Exception as exc:
            state.update(state='stopping_error',error=repr(exc),error_started_ns=time.time_ns())
            stop(data,state['error'])
            observe_stop_request(data,state)
            safe.atomic_json(data/'runnerstatus.json',state)
            budget=0 if isinstance(exc,SaveGraceExpired) else error_wait_budget(state)
            if child is not None and budget>0:
                try:child.wait(timeout=budget)
                except Exception:pass
            state['state']='failed'
        finally:
            safe.cleanup(state,data,grace=15)
            if state['state']=='stopped_by_user':
                try:
                    state['report']=collect(data,state)
                    state['checkpoint_cpu_audit']=safe.audit_final_weights(data,state)
                    if not state['checkpoint_cpu_audit']['valid']:
                        raise RuntimeError(state['checkpoint_cpu_audit'])
                    state['sampled_gpu_peaks']=safe.audit_sampled_gpu_peaks(data,state)
                except Exception as exc:
                    state.update(state='failed',error='Final CPU audit failed: '+repr(exc))
            state['finished']=stamp()
            state['heartbeat_ns']=time.time_ns()
            safe.atomic_json(data/'runnerstatus.json',state)
            for process in processes:
                try:process.wait(timeout=18)
                except subprocess.TimeoutExpired:
                    expected=state.get('sample' if process is processes[0] else 'guard')
                    if expected and safe.same_process(expected,safe.proc_identity(process.pid)):
                        process.terminate()
                        process.wait(timeout=10)
            report(data,state)
            safe.atomic_json(data/'runnerstatus.json',state)


def observe_stop_request(data, state, now=None):
    now = time.time() if now is None else now
    flag = data/'training/stop.flag'
    if not flag.exists():
        return None
    value = json.loads(flag.read_text())
    if value.get('run_id') != state['run_id'] or value.get('uid') != 1004:
        raise RuntimeError('Stop flag UID/run_id identity does not match this manual run')
    requested = value.get('requested_ns')
    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0 or requested > now*1e9:
        requested = min(flag.stat().st_mtime_ns, int(now*1e9))
    state.setdefault('stop_requested_ns', requested)
    return value


def check_update_watchdogs(result, state, now_ns=None):
    from world_backends.runtime import service_plan_from_state
    if state.get('stop_requested_ns') is not None:
        return
    now_ns = time.time_ns() if now_ns is None else now_ns
    for model in service_plan_from_state(state).active_kinds:
        ranks = result.get('training', {}).get(model, {}).get('ranks', {})
        latest = min((row.get('last_success_ns') or state['started_ns'] for row in ranks.values()), default=state['started_ns'])
        cap = 3600 if len(ranks)==4 and all(row.get('successful_updates', 0)>0 for row in ranks.values()) else 7200
        if (now_ns-latest)/1e9 > cap:
            raise RuntimeError(model+' successful-update watchdog expired')


def validate_manual_completion(result, state, flag):
    from world_backends.runtime import service_plan_from_state
    plan = service_plan_from_state(state)
    if flag.get('reason') != 'user_requested_stop' or flag.get('run_id') != state['run_id'] or flag.get('uid') != 1004:
        raise RuntimeError('Exit0 without an explicit matching user stop is not completion')
    steps = {model:result.get('training', {}).get(model, {}).get('global_updates', 0) for model in plan.active_kinds}
    if (any(isinstance(value, bool) or not isinstance(value, int) or value<0 for value in steps.values())
            or steps != plan.expected_optimizer_steps(steps['policy'])):
        raise RuntimeError('Manual final updates do not match synchronized '+plan.logical_update_ratio+' boundaries')
    completed = result.get('experiment_completed', [])
    d = completed[0]['details'] if len(completed)==1 else {}
    policy = d.get('stop_policy', {})
    if (not completed or completed[0].get('ts_ns',0)<flag.get('requested_ns',0)
        or d.get('stop_reason') != 'scheduled_stop' or d.get('global_step') != steps['policy']
        or d.get('completed_epochs') != steps['policy'] or d.get('run_started_ns') != state['started_ns']
        or d.get('stop_on_target') is not False or d.get('until_deadline') is not False or d.get('deadline_unix_seconds') is not None
        or policy.get('until_manual_stop') is not True or policy.get('until_deadline') is not False
        or policy.get('stop_on_target') is not False or policy.get('deadline_unix_seconds') is not None):
        raise RuntimeError('Learning completion event does not confirm the manual user-stop contract')
    expected_reason = result.get('checkpoint', {}).get('reason')
    accepted_reasons = {'final'} if steps['policy'] else {'final_before_first_update', 'final'}
    if expected_reason not in accepted_reasons:
        raise RuntimeError('Expected final weights checkpoint is missing: '+str(expected_reason))
    sharding = result.get('fsdp_sharding_verified', [])
    if 'reward' in plan.active_kinds and (len(sharding)!=4 or {int(x['actor_id']) for x in sharding}!=set(range(4))
        or len({x['pid'] for x in sharding})!=4
        or any(x['details'].get('model')!='reward' or x['details']['local_parameter_elements']>=x['details']['global_parameter_elements']/2 for x in sharding)):
        raise RuntimeError('Actual Reward four-rank FSDP sharding not verified')
    policy_rows = result.get('policy_zero2_verified', [])
    if (len(policy_rows)!=4 or {x['details']['rank'] for x in policy_rows}!=set(range(4)) or len({x['pid'] for x in policy_rows})!=4
        or any((x['details']['backend'],x['details']['zero_stage'],x['details']['world_size'],x['details']['micro_batch_size'],
                x['details']['gradient_accumulation_steps'],x['details']['global_batch_size'])!=('deepspeed_zero2',2,4,8,18,576)
               or not x['details']['no_dtensor_parameters'] for x in policy_rows)):
        raise RuntimeError('Actual Policy8/18 four-rank ZeRO2 not verified')
    online_world = (plan.backend == 'weaver' and plan.world_trainable
        and state.get('world_backend_config',{}).get('backend_options',{}).get('transport') == 'isolated_online_ddp')
    if online_world:
        from world_backend_audit import validate_online_world_receipts, validate_online_publication, validate_online_descriptor, validate_joint_release_receipts
        validate_online_world_receipts(result.get('weaver_online_trainer_verified', []), state['world_backend_config'])
        validate_online_publication(result.get('world_publication_committed', []), version=steps['ctrl'])
        validate_online_descriptor(result.get('checkpoint', {}).get('online_world', {}), state['world_backend_config'],
            run_id=state['run_id'], policy_step=steps['policy'], reason=expected_reason)
        validate_joint_release_receipts(result.get('weaver_online_publication_released', []), result['checkpoint']['online_world'])
        if {row['pid'] for row in result['weaver_online_publication_released']} != {row['pid'] for row in result['weaver_online_trainer_verified']}:
            raise RuntimeError('Final integrated native release PID owners differ from the verified trainer children')
        if result.get('weaver_frozen_verified') or result.get('world_zero2_verified'):
            raise RuntimeError('Online WEAVER must not masquerade as frozen World or CTRL ZeRO2')
    elif plan.world_trainable:
        validate_world_zero2_receipts(result.get('world_zero2_verified', []))
    else:
        from world_backend_audit import validate_frozen_world_receipts
        validate_frozen_world_receipts(result.get('weaver_frozen_verified', []), state['world_backend_config'])
        if result.get('world_zero2_verified') or result.get('world_memory_policy_verified'):
            raise RuntimeError('Frozen WEAVER unexpectedly has a World trainer')
    if 'reward' in plan.active_kinds and (len(result.get('reward_features_only_enabled', []))!=4
        or not all(x['details'].get('skip_lm_head') for x in result['reward_features_only_enabled'])
        or len(result.get('reward_checkpointing_verified', []))!=4
        or not all(x['details'].get('vision_block_count',0)>0 for x in result['reward_checkpointing_verified'])):
        raise RuntimeError('Actual four-rank Reward memory configuration not verified')
    if ((plan.world_trainable and not online_world and len(result.get('world_memory_policy_verified', []))!=4)
        or not result.get('placements',{}).get('complete_expected_placement')):
        raise RuntimeError('Actual four-rank model/memory/placement configuration not verified')
    for model,count in steps.items():
        info = result.get('training',{}).get(model,{})
        ranks = info.get('ranks',{})
        if count and (not info.get('rank_counts_match') or {int(rank) for rank in ranks}!=set(range(4))
                      or any(row.get('skipped_steps',0) for row in ranks.values())):
            raise RuntimeError('Final rank counters mismatch or skipped update')
        if online_world and model == 'ctrl' and (not info.get('rank_counts_match')
                or {int(rank) for rank in ranks} != set(range(4))
                or any(row.get('successful_updates') != count or row.get('skipped_steps', 0)
                       for row in ranks.values())):
            raise RuntimeError('Native World final successful counters require all four real ranks and skip0')
    replay = {row[0]:row for row in result.get('replay',[])}
    if (replay.get('policy',[None,0,0,0,0,0])[3] != steps['policy']*576
        or replay.get('wm',[None,0,0,0,0,0])[4] != steps.get('ctrl',0)*1024
        or replay.get('wm',[None,0,0,0,0,0])[5] != steps.get('reward', 0)*1024):
        raise RuntimeError('Final Replay exposure ledger differs from real successful updates')
    if any(steps.get(kind,0) for kind in plan.wm_consumers) and not result.get('split_wm_cohort_validated'):
        raise RuntimeError('Shared World/Reward cohort validation is missing')
    if online_world:
        if d.get('service_plan') != plan.to_dict():
            raise RuntimeError('Completion service plan differs from the native online owners')
        if steps['ctrl']:
            cohort = result['split_wm_cohort_validated'][0].get('details', {})
            if cohort.get('wm_consumers') != ['ctrl'] or cohort.get('expected_wm_consumers') != 4:
                raise RuntimeError('Integrated WEAVER requires exactly four native WM/RM consumers')
        if any(result.get(kind) for kind in ('fsdp_sharding_verified','reward_features_only_enabled','reward_checkpointing_verified')):
            raise RuntimeError('Integrated WEAVER must not create external Reward training services')
    if not plan.world_trainable:
        if d.get('service_plan') != plan.to_dict():
            raise RuntimeError('Completion service plan differs from frozen WEAVER owners')
        if steps.get('reward',0):
            cohort = result['split_wm_cohort_validated'][0].get('details', {})
            if cohort.get('wm_consumers') != ['reward'] or cohort.get('expected_wm_consumers') != 4:
                raise RuntimeError('Frozen WEAVER cohort must confirm four Reward consumers')
    if plan.integrated_reward:
        if result.get('training', {}).get('reward', {}).get('global_updates', 0):
            raise RuntimeError('Disabled external Reward has unexpected optimizer updates')
        if d.get('service_plan') != plan.to_dict():
            raise RuntimeError('Completion service plan differs from enabled backend owners')
        if steps['ctrl']:
            cohort = result['split_wm_cohort_validated'][0].get('details', {})
            if cohort.get('wm_consumers') != ['ctrl'] or cohort.get('expected_wm_consumers') != 4:
                raise RuntimeError('Integrated World cohort must confirm exactly four consumers')
    dynamic=result.get('policy_batch_mode_validation',{})
    if (dynamic.get('valid') is not True or dynamic.get('run_id')!=state['run_id']
        or dynamic.get('steps')!=steps['policy'] or dynamic.get('mode_receipts')!=8*steps['policy']
        or dynamic.get('global_exposures')!=576*steps['policy']):
        raise RuntimeError('Complete dynamic Policy mode receipts do not match this manual run')
    return {'steps':steps,'checkpoint_reason':expected_reason,'stopped_before_first_update':steps['policy']==0}


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--mode', choices=['manual', 'guard', 'report', 'sample'], required=True)
    a.add_argument('--data', type=Path, required=True)
    a.add_argument('--run-id')
    a.add_argument('--stop-on-target', action='store_true')
    a.add_argument('--world-backend-config', type=Path)
    a.add_argument('--pipeline-config', type=Path)
    args = a.parse_args()
    if args.mode == 'guard':
        guard(args.data, args.run_id)
    elif args.mode == 'report':
        report(args.data, load(args.data))
    elif args.mode == 'sample':
        safe.sample(args.data, args.run_id)
    else:
        run(args.data, args.mode, args.stop_on_target, args.world_backend_config, args.pipeline_config)
