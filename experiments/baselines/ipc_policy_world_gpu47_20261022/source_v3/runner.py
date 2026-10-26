"""Fresh Policy ordinary8/18 joint4/36 W4/64 R16/16 GPU4-7 formal run until explicit user stop."""
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
    return json.loads((data / 'runnerstatus.json').read_text())


def measurement_configuration():
    return {'nsight_collection': False,
            'nsight_capture': {'nvtx_enabled': False, 'policy_steps': [],
                'state': 'disabled_for_ipc_validation', 'actual_enabled': False},
            'hardware_metrics': {'enabled': False, 'status': 'not_collected_permission_restricted',
                                 'sm_active': None, 'occupancy': None},
            'weight_transports': {'policy': 'cuda_ipc', 'ctrl': 'cuda_ipc', 'reward': 'cpu_ray'},
            'ray_object_store_bytes': 328 * 1024**3}



def collect(data, state):
    result = {'run_id': state['run_id'], 'state': state['state'], 'updated': stamp(),
              'mode': state['mode'], 'training': {}, 'source': str(SOURCE),
              'until_manual_stop': True, 'deadline_unix': None,
              'stop_on_target': False, 'mps_active_thread_caps': None}
    result['policy_batch_schedule']={'ordinary_micro_GAS':[8,18],'joint_micro_GAS':[4,36],
                                    'joint_interval':5,'global_batch':576,'local_batch':144}
    result['warmup_steps']={'policy':100,'value':100,'ctrl':20,'reward':20}
    result.update(measurement_configuration())
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
        result['placements'] = safe.placement_evidence(c, state['run_id'])
        mode_rows = c.execute('SELECT run_id,ts_ns,actor,actor_id,pid,details_json FROM events '
            'WHERE run_id=? AND workload IS NULL AND kind=? ORDER BY ts_ns DESC,id DESC LIMIT 80',
            (state['run_id'], 'policy_batch_mode')).fetchall()
        result['policy_batch_modes_recent'] = [{key:row[key] for key in ('run_id','ts_ns','actor','actor_id','pid')} |
            {'details':json.loads(row['details_json'])} for row in mode_rows]
        result['policy_batch_modes_recent_scope'] = 'latest80 events only; final validation separately streams the complete run'
        for kind in ['fsdp_sharding_verified', 'synchronized_epoch_completed', 'evaluation_threshold',
                     'experiment_completed', 'split_wm_cohort_validated', 'reward_features_only_enabled',
                     'reward_checkpointing_verified', 'world_memory_policy_verified', 'policy_zero2_verified', 'world_zero2_verified']:
            rows = c.execute('select ts_ns,pid,actor_id,details_json from events where run_id=? and workload is null and kind=? order by ts_ns desc limit ?',
                             (state['run_id'], kind, 8 if kind == 'fsdp_sharding_verified' else 4 if kind in {'reward_features_only_enabled', 'reward_checkpointing_verified', 'world_memory_policy_verified', 'policy_zero2_verified', 'world_zero2_verified'} else 1)).fetchall()
            result[kind] = [{'ts_ns': ts, 'pid': pid, 'actor_id': actor_id, 'details': json.loads(j)} for ts, pid, actor_id, j in rows]
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
                parts = [intervals.get((step, model, rank), {}) for model in ['policy', 'ctrl', 'reward']]
                if all('begin' in p and 'end' in p for p in parts):
                    overlaps.append({'step': step, 'rank': rank,
                        'host_phase_triple_overlap_seconds': max(0, min(p['end'] for p in parts) - max(p['begin'] for p in parts)) / 1e9})
        result['concurrent_host_phases'] = overlaps
    finally:
        c.close()
    manifest = data / 'training/checkpoints/manifest.json'
    if manifest.exists():
        m = json.loads(manifest.read_text())
        result['checkpoint'] = {k: m[k] for k in ['reason', 'steps', 'world_size', 'weights_only']}
    return result


def report(data, state):
    result = collect(data, state)
    safe.atomic_json(data/'reportstatus.json', result)
    lines = ['# GPU4–7 IPC权重同步、Ray328GiB及端到端依赖记录长训', '',
             f'状态：{state["state"]}；更新：{stamp()}；run_id：`{state["run_id"]}`。',
             '全新模型、优化器及空Replay，不续训。Policy独立轮8/18、三模型联合轮4/36、World4/64、Reward16/16；global576/1024/1024及5:1:1保持。',
             '只在用户明确要求后完整同步边界停止；没有固定时刻、15轮或85%停止。30000仅LR horizon，25完整epoch周期保存三模型weights。',
             'Policy/World使用CUDA IPC，Reward保留CPU权重同步；保留两端CPU SHA校验。Ray Object Store容量328GiB。',
             '用户新指定预热：Policy100、Value100、World20、Reward20，各按自身成功更新计数；峰值学习率及30000/6000/30000衰减尺度不变。',
             '记录请求ID/实际服务版本、真实episode及想象/GAE依赖DAG、Replay有序预留和实际micro曝光、成功更新及发布ACK。评估与训练数据分开；预留不等于消费，曝光不等于成功更新。',
             '本轮不采集Nsight或SM Active/Occupancy；阶段请求/批次/样本计数与端到端关联来自CPU遥测，不代表GPU资源份额。',
             '物理GPU4–7，既有MPS默认/server100%，无客户端计算配额。首更新120分钟、后续停滞60分钟、错误120秒；人工stop900秒保存宽限，不叠加窗口。',
             '失败或用户停止仅清退本轮UID/run_id/PID出生匹配进程；不自动重启，不改变MPS或其他GPU任务。', '',
             '```json', json.dumps(result, ensure_ascii=False, indent=2), '```']
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


def run(data, mode, stop_on_target):
    if mode != 'manual' or stop_on_target:
        raise RuntimeError('This source only permits a fresh manual-stop formal run')
    if data.resolve() != (safe.MASTER/'formal_v3').resolve():
        raise RuntimeError('Manual data must be this new master/formal_v3 directory')
    if os.getuid()!=1004:
        raise RuntimeError('Wrong UID')
    data.mkdir(parents=True,exist_ok=False)
    pre = safe.preflight()
    rid = 'ipcpwmanual_'+dt.datetime.now(TZ).strftime('%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
    state = {'state':'starting','run_id':rid,'mode':'manual','uid':1004,'root':str(ROOT),
             'started_ns':time.time_ns(),'started':stamp(),'supervisor':safe.proc_identity(os.getpid()),
             'heartbeat_ns':time.time_ns(),'owned':[],'preflight':pre,'training_stopped':False,
             'stop_on_target':False,'deadline_unix':None,'until_manual_stop':True,'no_resume':True,
             'automatic_restart':False,'mps_active_thread_caps':None,'physical_gpus':[4,5,6,7],
             'gpu_save_grace_seconds':900,'error_grace_seconds':120,
             'base_frozen_archive_sha256':'3b030ada9172dac7856d7a6fe6ab2d7baf8a86967300049cc8d2c7243ab3a2e7',
             'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')}}
    state['policy_batch_schedule']={'ordinary_micro_GAS':[8,18],'joint_micro_GAS':[4,36],
                                    'joint_interval':5,'global_batch':576,'local_batch':144}
    state['warmup_steps']={'policy':100,'value':100,'ctrl':20,'reward':20}
    state.update(measurement_configuration())
    state['source_sha256']['run.sh']=hashlib.sha256((SOURCE/'run.sh').read_bytes()).hexdigest()
    safe.atomic_json(data/'runnerstatus.json',state)
    env = formal_environment(state,data,pre)
    processes=[]
    child=None
    with (data/'runner.log').open('a',buffering=1) as log:
        try:
            child=subprocess.Popen(['bash',str(SOURCE/'run.sh'),'--exp-name',
                'OpenVLA_IPCPolicyWorld_Ray328GiB_JointPolicy4_World4_RewardFSDP2_gpu47_fresh'],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['child']=safe.proc_identity(child.pid)
            state['minimum_child_birth_ticks']=state['child']['birth_ticks']
            state['state']='running'
            safe.atomic_json(data/'runnerstatus.json',state)
            sampler=safe.start_sampler(data,state,log)
            guard_process=subprocess.Popen([sys.executable,__file__,'--mode','guard','--data',str(data),'--run-id',rid],
                env=safe.cpu_helper_environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
                    validate_world_zero2_receipts(result.get('world_zero2_verified',[]))
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
    if state.get('stop_requested_ns') is not None:
        return
    now_ns = time.time_ns() if now_ns is None else now_ns
    for model in ('policy', 'ctrl', 'reward'):
        ranks = result.get('training', {}).get(model, {}).get('ranks', {})
        latest = min((row.get('last_success_ns') or state['started_ns'] for row in ranks.values()), default=state['started_ns'])
        cap = 3600 if len(ranks)==4 and all(row.get('successful_updates', 0)>0 for row in ranks.values()) else 7200
        if (now_ns-latest)/1e9 > cap:
            raise RuntimeError(model+' successful-update watchdog expired')


def validate_manual_completion(result, state, flag):
    if flag.get('reason') != 'user_requested_stop' or flag.get('run_id') != state['run_id'] or flag.get('uid') != 1004:
        raise RuntimeError('Exit0 without an explicit matching user stop is not completion')
    steps = {model:result.get('training', {}).get(model, {}).get('global_updates', 0) for model in ('policy','ctrl','reward')}
    if any(not isinstance(value, int) or value<0 for value in steps.values()) or steps['ctrl'] != steps['policy']//5 or steps['reward'] != steps['policy']//5:
        raise RuntimeError('Manual final updates do not match synchronized5:1:1 boundaries')
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
    if (len(sharding)!=4 or {int(x['actor_id']) for x in sharding}!=set(range(4))
        or len({x['pid'] for x in sharding})!=4
        or any(x['details'].get('model')!='reward' or x['details']['local_parameter_elements']>=x['details']['global_parameter_elements']/2 for x in sharding)):
        raise RuntimeError('Actual Reward four-rank FSDP sharding not verified')
    policy_rows = result.get('policy_zero2_verified', [])
    if (len(policy_rows)!=4 or {x['details']['rank'] for x in policy_rows}!=set(range(4)) or len({x['pid'] for x in policy_rows})!=4
        or any((x['details']['backend'],x['details']['zero_stage'],x['details']['world_size'],x['details']['micro_batch_size'],
                x['details']['gradient_accumulation_steps'],x['details']['global_batch_size'])!=('deepspeed_zero2',2,4,8,18,576)
               or not x['details']['no_dtensor_parameters'] for x in policy_rows)):
        raise RuntimeError('Actual Policy8/18 four-rank ZeRO2 not verified')
    validate_world_zero2_receipts(result.get('world_zero2_verified', []))
    if (len(result.get('reward_features_only_enabled', []))!=4
        or not all(x['details'].get('skip_lm_head') for x in result['reward_features_only_enabled'])
        or len(result.get('reward_checkpointing_verified', []))!=4
        or not all(x['details'].get('vision_block_count',0)>0 for x in result['reward_checkpointing_verified'])
        or len(result.get('world_memory_policy_verified', []))!=4
        or not result.get('placements',{}).get('complete_expected_placement')):
        raise RuntimeError('Actual four-rank model/memory/placement configuration not verified')
    for model,count in steps.items():
        info = result.get('training',{}).get(model,{})
        ranks = info.get('ranks',{})
        if count and (not info.get('rank_counts_match') or {int(rank) for rank in ranks}!=set(range(4))
                      or any(row.get('skipped_steps',0) for row in ranks.values())):
            raise RuntimeError('Final rank counters mismatch or skipped update')
    replay = {row[0]:row for row in result.get('replay',[])}
    if (replay.get('policy',[None,0,0,0,0,0])[3] != steps['policy']*576
        or replay.get('wm',[None,0,0,0,0,0])[4] != steps['ctrl']*1024
        or replay.get('wm',[None,0,0,0,0,0])[5] != steps['reward']*1024):
        raise RuntimeError('Final Replay exposure ledger differs from real successful updates')
    if steps['ctrl'] and not result.get('split_wm_cohort_validated'):
        raise RuntimeError('Shared World/Reward cohort validation is missing')
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
    args = a.parse_args()
    if args.mode == 'guard':
        guard(args.data, args.run_id)
    elif args.mode == 'report':
        report(args.data, load(args.data))
    elif args.mode == 'sample':
        safe.sample(args.data, args.run_id)
    else:
        run(args.data, args.mode, args.stop_on_target)
