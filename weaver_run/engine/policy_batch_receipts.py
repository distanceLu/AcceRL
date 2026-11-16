"""CPU-only streaming validation of a complete manual run's Policy receipts."""
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from policy_joint_batch_mode import mode_for_step, expected_completed_microsteps


def validate_policy_batch_mode_receipts(rows, *, run_id, steps, model_free=False, autotune=False):
    """Consume per-rank chronological events once, using constant rank state.

    No fixed epoch limit, row limit, request join, CUDA operation or payload load.
    A zero-update fill stop must have no dynamic-mode events; the controller
    separately requires its original four-rank8/18 initialization receipts.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError('A nonnegative successful Policy-update count is required')
    if type(autotune) is not bool or (autotune and not model_free):
        raise ValueError('Autotuned receipts require model-free mode')
    cumulative = {rank:0 for rank in range(4)}
    last_modes = {rank:(8,18) for rank in range(4)}
    active = {}
    schedules = {rank:hashlib.sha256() for rank in range(4)}
    next_step={rank:1 for rank in range(4)}
    next_transition={rank:'begin' for rank in range(4)}
    pids={};count=0
    for row in rows:
        details=row.get('details')
        if not isinstance(details, dict):
            raise RuntimeError('Policy batch-mode event details are missing')
        step,rank,transition=details.get('policy_step'),details.get('rank'),details.get('transition')
        if (isinstance(rank,bool) or not isinstance(rank,int) or rank not in next_step
                or isinstance(step,bool) or not isinstance(step,int) or not 1<=step<=steps
                or row.get('run_id')!=run_id or row.get('actor')!='policy_trainer'
                or str(row.get('actor_id'))!=str(rank)
                or isinstance(row.get('pid'),bool) or not isinstance(row.get('pid'),int) or row['pid']<=0
                or step!=next_step[rank] or transition!=next_transition[rank]):
            raise RuntimeError('Missing, duplicate, out-of-order or foreign Policy batch-mode receipt')
        if autotune:
            micro,gas = details.get('micro'), details.get('gas')
            if (details.get('autotune') is not True or type(micro) is not int
                    or micro not in (8,16,24) or type(gas) is not int or gas != 144//micro
                    or cumulative[rank] % gas != 0):
                raise RuntimeError('Invalid autotuned Policy geometry or unsafe GAS boundary')
            old_micro,old_gas = last_modes[rank]
            before_micro = cumulative[rank]
        else:
            if details.get('autotune',False) is not False:
                raise RuntimeError('Autotuned Policy receipt appeared without launch opt-in')
            micro,gas=mode_for_step(step, model_free=model_free)
            old_micro,old_gas=mode_for_step(step-1, model_free=model_free) if step>1 else (8,18)
            before_micro=expected_completed_microsteps(step-1, model_free=model_free)
        wanted={'world_size':4,'global_batch':576,'local_batch':144,
                'joint_due':not model_free and step%5==0,'micro':micro,'gas':gas,
                'old_micro':old_micro,'old_gas':old_gas,
                'before_micro_steps':before_micro,
                'before_global_steps':step-1,'before_global_samples':576*(step-1)}
        if transition=='complete':
            wanted.update(after_micro_steps=before_micro+gas,
                          after_global_steps=step,after_global_samples=576*step)
        if any(details.get(name)!=value for name,value in wanted.items()):
            raise RuntimeError('Dynamic Policy batch/GAS or cumulative step receipt is invalid')
        if autotune:
            if any(type(details[name]) is not type(value) for name,value in wanted.items()):
                raise RuntimeError('Autotuned Policy counter/geometry types are invalid')
            if transition=='begin':
                active[rank] = dict(wanted)
            else:
                if any(details.get(name)!=value for name,value in active.pop(rank).items()):
                    raise RuntimeError('Autotuned begin/complete batch identity differs')
                schedules[rank].update(json.dumps([step,micro,gas,before_micro],
                    separators=(',',':')).encode('ascii'))
                cumulative[rank] = before_micro+gas
                last_modes[rank] = micro,gas
        if rank in pids and pids[rank]!=row['pid']:
            raise RuntimeError('Policy rank PID changed during candidate')
        pids[rank]=row['pid'];count+=1
        if transition=='begin':next_transition[rank]='complete'
        else:next_step[rank]+=1;next_transition[rank]='begin'
    if (any(next_step[rank]!=steps+1 or next_transition[rank]!='begin' for rank in range(4))
            or (steps and (len(pids)!=4 or len(set(pids.values()))!=4))):
        raise RuntimeError('Missing complete4-rank Policy begin/complete mode coverage')
    if autotune and len({digest.digest() for digest in schedules.values()}) != 1:
        raise RuntimeError('Autotuned Policy schedules disagree across ranks')
    microsteps = cumulative[0] if autotune else expected_completed_microsteps(steps, model_free=model_free)
    result = {'valid':True,'run_id':run_id,'steps':steps,'ranks':4,'observed_rank_pids':len(pids),
            'mode_receipts':count,'micro_steps_per_rank':microsteps,
            'successful_updates_per_rank':steps,'local_exposures_per_rank':144*steps,
            'global_exposures':576*steps,'rank_pids':pids,
            'joint_policy_micro_GAS':None if model_free else [4,36],
            'single_policy_micro_GAS':None if autotune else [8,18],
            'stopped_before_first_update':steps==0,'validation_scope':'all matching run events, no row cap'}
    if autotune:
        result.update(autotune=True, final_policy_micro_GAS=list(last_modes[0]),
                      schedule_sha256=schedules[0].hexdigest())
    return result



def validate_policy_batch_modes_from_db(db, *, run_id, steps, timeout_seconds=120, model_free=False, autotune=False):
    """Only call after the application has exited; live reports use a small tail."""
    path=Path(db)
    connection=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=3)
    connection.row_factory=sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    deadline=time.monotonic()+timeout_seconds
    connection.set_progress_handler(lambda:int(time.monotonic()>deadline),10000)
    try:
        cursor=connection.execute('SELECT run_id,actor,actor_id,pid,details_json FROM events '
            'WHERE run_id=? AND workload IS NULL AND kind=? ORDER BY ts_ns,id',
            (run_id,'policy_batch_mode'))
        def rows():
            for row in cursor:
                if time.monotonic()>deadline:
                    raise TimeoutError('Policy receipt CPU audit time budget expired')
                yield {'run_id':row['run_id'],'actor':row['actor'],'actor_id':row['actor_id'],
                       'pid':row['pid'],'details':json.loads(row['details_json'])}
        return validate_policy_batch_mode_receipts(rows(),run_id=run_id,steps=steps,model_free=model_free,autotune=autotune)
    finally:
        connection.close()
