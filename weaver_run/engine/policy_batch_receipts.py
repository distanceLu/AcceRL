"""CPU-only streaming validation of a complete manual run's Policy receipts."""
import json
from pathlib import Path
import sqlite3
import time

from policy_joint_batch_mode import mode_for_step, expected_completed_microsteps


def validate_policy_batch_mode_receipts(rows, *, run_id, steps):
    """Consume per-rank chronological events once, using constant rank state.

    No fixed epoch limit, row limit, request join, CUDA operation or payload load.
    A zero-update fill stop must have no dynamic-mode events; the controller
    separately requires its original four-rank8/18 initialization receipts.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError('A nonnegative successful Policy-update count is required')
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
        micro,gas=mode_for_step(step)
        old_micro,old_gas=mode_for_step(step-1) if step>1 else (8,18)
        wanted={'world_size':4,'global_batch':576,'local_batch':144,
                'joint_due':step%5==0,'micro':micro,'gas':gas,
                'old_micro':old_micro,'old_gas':old_gas,
                'before_micro_steps':expected_completed_microsteps(step-1),
                'before_global_steps':step-1,'before_global_samples':576*(step-1)}
        if transition=='complete':
            wanted.update(after_micro_steps=expected_completed_microsteps(step),
                          after_global_steps=step,after_global_samples=576*step)
        if any(details.get(name)!=value for name,value in wanted.items()):
            raise RuntimeError('Dynamic Policy batch/GAS or cumulative step receipt is invalid')
        if rank in pids and pids[rank]!=row['pid']:
            raise RuntimeError('Policy rank PID changed during candidate')
        pids[rank]=row['pid'];count+=1
        if transition=='begin':next_transition[rank]='complete'
        else:next_step[rank]+=1;next_transition[rank]='begin'
    if (any(next_step[rank]!=steps+1 or next_transition[rank]!='begin' for rank in range(4))
            or (steps and (len(pids)!=4 or len(set(pids.values()))!=4))):
        raise RuntimeError('Missing complete4-rank Policy begin/complete mode coverage')
    return {'valid':True,'run_id':run_id,'steps':steps,'ranks':4,'observed_rank_pids':len(pids),
            'mode_receipts':count,'micro_steps_per_rank':expected_completed_microsteps(steps),
            'successful_updates_per_rank':steps,'local_exposures_per_rank':144*steps,
            'global_exposures':576*steps,'rank_pids':pids,
            'joint_policy_micro_GAS':[4,36],'single_policy_micro_GAS':[8,18],
            'stopped_before_first_update':steps==0,'validation_scope':'all matching run events, no row cap'}


def validate_policy_batch_modes_from_db(db, *, run_id, steps, timeout_seconds=120):
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
        return validate_policy_batch_mode_receipts(rows(),run_id=run_id,steps=steps)
    finally:
        connection.close()
