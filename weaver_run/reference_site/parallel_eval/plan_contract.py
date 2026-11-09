"""CPU-only sharding and strict merge of the existing immutable paired plan."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys

BASE_EVALUATOR_SHA256='8fb85685fd50d0630428e499e6d6e09ca48084acf34e667e783018b63e71e051'
APPROVED_GPUS=(
    'GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c',
    'GPU-fe0edfe4-b193-1af7-6a98-cbff40214f74',
    'GPU-8c9bc264-7f09-989e-1ec7-cecd6cde782c',
    'GPU-07ae0bbc-8923-bd0c-199e-c726821e655d',
)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def load_base(source):
    source=Path(source).resolve(strict=True)
    path=source/'frozen_policy_eval.py'
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=BASE_EVALUATOR_SHA256:
        raise ValueError('The reviewed frozen evaluation implementation differs')
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('parallel_frozen_eval_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def check_plan(plan):
    raw={k:v for k,v in plan.items() if k!='sha256'}
    if (plan.get('sha256')!=digest(raw) or plan.get('protocol')!='accerl_frozen_policy_paired_v1'
            or plan.get('initial_state_count')!=50 or plan.get('task_id')!=0
            or plan.get('horizon_action_steps')!=220 or plan.get('reset_settling_steps')!=10
            or plan.get('inference_batch_size')!=1 or plan.get('policy_updates_during_evaluation') is not False
            or [c.get('name') for c in plan.get('checkpoints',[])]!=['baseline','final']):
        raise ValueError('Unexpected or modified global paired evaluation plan')
    cases=plan.get('cases',[])
    if len({c['pair_id'] for c in cases})!=len(cases) or plan.get('expected_episode_records')!=2*len(cases):
        raise ValueError('Global cases are duplicated or expected counts differ')
    expected={(repeat,initial,mode) for repeat in range(plan['repeats'])
              for initial in range(50) for mode in plan['modes']}
    actual={(c['repeat'],c['initial_state_id'],c['mode']) for c in cases}
    if len(actual)!=len(cases) or actual!=expected:
        raise ValueError('Global paired initial-state/mode/repeat coverage differs')
    if type(plan['seed']) is not int or not 0<=plan['seed']<2**31 or type(plan['repeats']) is not int or plan['repeats']<1:
        raise ValueError('Invalid seed or repeat count')
    if not plan['modes'] or len(set(plan['modes']))!=len(plan['modes']) or set(plan['modes'])-{'greedy','stochastic'}:
        raise ValueError('Invalid evaluation modes')
    canonical=[]
    for repeat in range(plan['repeats']):
        order=list(range(50));random.Random(plan['seed']+repeat).shuffle(order)
        for initial in order:
            for mode in plan['modes']:
                canonical.append(dict(pair_id=f'r{repeat}:s{initial}:{mode}',repeat=repeat,initial_state_id=initial,
                    env_seed=int(digest([plan['seed'],repeat,initial,'environment'])[:8],16)%2**31,
                    action_seed=int(digest([plan['seed'],repeat,initial,'actions'])[:8],16)%2**31,
                    mode=mode,deterministic=mode=='greedy'))
    if cases!=canonical or any(type(c.get('deterministic')) is not bool for c in cases):
        raise ValueError('Case order, seeds or mode differ from the original paired protocol')


def make_shards(plan,gpu_uuids=APPROVED_GPUS):
    check_plan(plan)
    gpu_uuids=tuple(gpu_uuids)
    if len(gpu_uuids)!=4 or len(set(gpu_uuids))!=4 or set(gpu_uuids)!=set(APPROVED_GPUS):
        raise ValueError('Four distinct explicitly approved GPU4-7 UUIDs are required')
    workers=[dict(worker_index=i,gpu_uuid=gpu,assigned_pair_ids=[],expected_episode_records=0)
             for i,gpu in enumerate(gpu_uuids)]
    groups={}
    for case in plan['cases']:
        key=(case['repeat'],case['initial_state_id'])
        if key not in groups:groups[key]=len(groups)%4
        workers[groups[key]]['assigned_pair_ids'].append(case['pair_id'])
    for worker in workers:worker['expected_episode_records']=2*len(worker['assigned_pair_ids'])
    result=dict(protocol='accerl_parallel_eval_shards_v1',plan_sha256=plan['sha256'],
                base_evaluator_sha256=BASE_EVALUATOR_SHA256,workers=workers)
    result['sha256']=digest(result)
    return result


def validate_shards(plan,shards):
    if not isinstance(shards,dict):raise ValueError('Missing shard contract')
    try:uuids=[w['gpu_uuid'] for w in shards['workers']]
    except (KeyError,TypeError):raise ValueError('Malformed worker/GPU assignments') from None
    if shards!=make_shards(plan,uuids):
        raise ValueError('Shard assignment differs from the immutable paired plan')
    return shards


def validate_worker_row(row,plan,shards,worker_index,run_token,base=None):
    validate_shards(plan,shards)
    if type(worker_index) is not int or not 0<=worker_index<4:
        raise ValueError('Invalid worker index')
    if base is None:raise ValueError('An explicitly reviewed evaluation module is required')
    if not isinstance(run_token,str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}',run_token) is None:
        raise ValueError('An explicit bounded execution token is required')
    base.validate_row(row,plan)
    worker=shards['workers'][worker_index]
    if (row['pair_id'] not in worker['assigned_pair_ids']
            or type(row.get('execution_worker_index')) is not int
            or row.get('execution_worker_index')!=worker_index
            or row.get('execution_gpu_uuid')!=worker['gpu_uuid']
            or row.get('execution_run_token')!=run_token):
        raise ValueError('Foreign shard, GPU or execution identity in episode row')
    return row


def merge_rows(plan,shards,worker_rows,run_token,base=None):
    validate_shards(plan,shards)
    if any(type(k) is not int for k in worker_rows) or set(worker_rows)!=set(range(4)):
        raise ValueError('All four worker ledgers are required, including partial ledgers')
    merged={}
    for index in range(4):
        for row in worker_rows[index]:
            validate_worker_row(row,plan,shards,index,run_token,base)
            key=(row['checkpoint'],row['pair_id'])
            if key in merged:raise ValueError('Duplicate completed episode across worker ledgers')
            merged[key]=row
    # Stable order is independent of process completion or append timing.
    return [merged[(checkpoint['name'],case['pair_id'])]
            for checkpoint in plan['checkpoints'] for case in plan['cases']
            if (checkpoint['name'],case['pair_id']) in merged]


def choose_deadline(now,max_seconds,deadline_unix=None,prior_started=None):
    for name,value in [('now',now),('max_seconds',max_seconds)]:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(name+' must be finite')
    if not 0<max_seconds<=7200:raise ValueError('The global evaluation budget must be in (0,7200] seconds')
    deadlines=[now+max_seconds]
    if deadline_unix is not None:
        if isinstance(deadline_unix,bool) or not isinstance(deadline_unix,(int,float)) or not math.isfinite(deadline_unix):
            raise ValueError('Explicit deadline must be finite')
        deadlines.append(deadline_unix)
    if prior_started is not None:
        value=prior_started.get('deadline_unix')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Original global deadline is missing')
        deadlines.append(value)
    deadline=min(deadlines)
    if deadline<=now:raise TimeoutError('The original global evaluation deadline has elapsed')
    return deadline
