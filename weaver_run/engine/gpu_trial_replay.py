"""Freeze real model inputs once and dispatch the same cohort to both timed GPU arms."""
import asyncio
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time

import numpy as np

from critic_refresh_plan import dependency_refresh_groups, dependency_refresh_receipt
from fresh_policy_replay import FreshReplayNotReady, SegmentReplay
from gpu_trial_placement import ordered_positions


SAMPLING_MODES = ('chunks', 'whole_episodes', 'uniform_blocks')
LAYOUT_ARMS = ('preserve', 'repack')


def _digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def _atomic_file(path, write):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, prefix='.writing-') as f:
        name = f.name
        try:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            f.close()
            Path(name).unlink(missing_ok=True)
            raise
    try:
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _persist(directory, frozen, cohort_id):
    import torch
    directory.mkdir(parents=True, exist_ok=True)
    segment_dir = directory / 'segments'
    segment_dir.mkdir(exist_ok=True)
    index = directory / (cohort_id + '.json')
    if index.exists():
        raise FileExistsError(f'Trial input manifest already exists: {index}')
    paths = {}
    added_bytes = 0
    for sid, segment in frozen['store'].segments.items():
        key = hashlib.sha256(sid.encode()).hexdigest()
        path = segment_dir / (key + '.pt')
        if not path.is_file():
            _atomic_file(path, lambda f: torch.save(vars(segment), f))
            added_bytes += path.stat().st_size
        paths[sid] = dict(file='segments/' + path.name, bytes=path.stat().st_size)
    ids = frozen['ids']
    manifest = dict(format='weaver_paired_real_rollout_inputs_v1', cohort_id=cohort_id,
                    sampled_versions=list(frozen['store'].versions),
                    sample_ids=ids, sample_ids_sha256=_digest(ids), segments=paths,
                    replay_admitted_at_draw=frozen['admitted'], replay_resident_at_draw=frozen['resident'],
                    description='Trusted torch.save plain dictionaries with real PolicySegment data, including K+1 preprocessed model observations, sampled action tokens, behaviour logits, masks and terminal/bootstrap metadata.')
    _atomic_file(index, lambda f: f.write((json.dumps(manifest, separators=(',', ':'))+'\n').encode()))
    return dict(manifest=str(index), new_segment_bytes=added_bytes,
                segment_bytes=sum(row['bytes'] for row in paths.values()))


def freeze_sampling_round(owner, count, cohort_id):
    """三次抽样在同一 actor 调用内完成，中间不让出执行权、不消耗样本。"""
    real = owner.fresh_store
    draws, receipts = {}, {}
    for mode in SAMPLING_MODES:
        started = time.perf_counter()
        try:
            token, ids = real.reserve(count, real_sampling=mode)
        except FreshReplayNotReady:
            return dict(ready=False, replay_resident=len(real.samples))
        try:
            sampling_s = time.perf_counter() - started
            sampling = dict(real.reservation_sampling.get(token, {'requested':mode,'actual':mode}))
            sampled = SegmentReplay(real.capacity, owner.fresh_config)
            sampled.set_versions(*real.versions)
            for sid in ids:
                sample = real.samples[sid]
                key = sample['segment_id']
                sampled.samples[sid] = dict(sample)
                sampled.segments[key] = real.segments[key]
        finally:
            real.cancel(token)
        receipt = dict(sample_ids=list(ids), sample_ids_sha256=_digest(ids), trajectories=len(sampled.segments),
                       sampling_seconds=sampling_s, sampling=sampling)
        draws[mode] = dict(key=cohort_id+'-'+mode, ids=ids, store=sampled, receipt=receipt,
                           preserve_rank_shards=mode!='chunks')
        receipts[mode] = receipt
    receipt = dict(ready=True, cohort_id=cohort_id, version=real.versions[0], n=count,
                   replay_resident_at_draw=len(real.samples), real_admitted_at_draw=owner.fresh_admitted_chunks['real'],
                   draws=receipts)
    # 保留三组样本的轨迹引用，采集端在测速期间可以继续逐出旧数据。
    owner._gpu_trial = dict(key=cohort_id, busy=False, receipt=receipt, draws=draws)
    return receipt


def freeze_layout_round(owner, count, key, pool_size, seed, *, exposures=None):
    from real_episode_sampling import _uniform_block

    real = owner.fresh_store
    eligible = [sid for sid, sample in real.samples.items() if real._eligible(sample)]
    if exposures is not None:
        def survives(sample):
            birth = real.segments[sample['segment_id']].versions[sample['index']]['policy']
            return (real.versions[0] - birth + exposures - 1 <= real.config.max_policy_age and
                    sample['uses'] == sample['reserved'] == 0 and exposures <= real.config.max_sample_uses)
        eligible = [sid for sid in eligible if survives(real.samples[sid])]
    if len(eligible) < pool_size:
        return dict(ready=False, eligible_before_limit=len(eligible), pool_size=pool_size)
    # 入库时整条轨迹按时间顺序写入，截取末尾即得到最新合格片段。
    pool = eligible[-pool_size:]
    groups = defaultdict(list)
    for sid in pool:
        segment = real.samples[sid]['segment_id']
        if real.segments[segment].source != 'real':
            raise ValueError('Non-real chunk in trajectory layout pool')
        groups[segment].append(sid)
    ids = _uniform_block(random.Random(seed), groups)
    sampled = SegmentReplay(real.capacity,owner.fresh_config)
    sampled.set_versions(*real.versions)
    for sid in ids:
        sample = real.samples[sid]
        segment = sample['segment_id']
        sampled.samples[sid] = dict(sample)
        sampled.segments[segment] = real.segments[segment]
    tokens = [int(sampled.segments[s['segment_id']].action_mask[s['index']].sum())
              for s in (sampled.samples[sid] for sid in ids)]
    receipt = dict(ready=True, cohort_id=key, n=count, pool_size=pool_size, seed=seed,
        pool_ids_sha256=_digest(pool), eligible_before_limit=len(eligible),
        sample_ids=ids, sample_ids_sha256=_digest(ids), version=real.versions[0],
        original_group_denominators=[sum(tokens[i:i+8]) for i in range(0,len(tokens),8)],
        trajectories=len(sampled.segments), replay_resident_at_draw=len(real.samples),
        real_admitted_at_draw=owner.fresh_admitted_chunks['real'])
    draws = {name:dict(key=key+'-'+name, ids=ids, store=sampled, receipt=receipt, layout_arm=name,
                       preserve_rank_shards=name=='preserve') for name in LAYOUT_ARMS}
    owner._gpu_trial = dict(key=key, busy=False, receipt=receipt, draws=draws)
    return receipt


def _pin_critical(store, key, version, ids):
    token = f'critical-jit:{key}:v{version}'
    if token in store.reservations or any(not store._eligible(store.samples[sid]) for sid in ids):
        raise ValueError('Cannot reserve the fixed critical cohort at the current policy')
    store.reservations[token] = set(ids)
    for sid in ids:
        store.samples[sid]['reserved'] += 1
    store._model_free_counts = None
    return token


def freeze_critical(owner, key, pool_size, seed, exposures):
    store = owner.fresh_store
    if (not owner.fresh_config.real_only or not owner.fresh_config.refresh_gae or
            not owner.fresh_config.replay_age_limits or owner.real_sampling != 'uniform_blocks' or
            len(owner.fresh_critics) != 4 or not owner.fresh_refresh_prune or
            getattr(owner,'_model_free_admission',None) or getattr(owner,'_gpu_trial',None) or
            exposures not in (1, 3, 4) or pool_size < 576):
        raise ValueError('Invalid fixed-cohort critical JIT admission')
    receipt = freeze_layout_round(owner, 576, key, pool_size, seed, exposures=exposures)
    if receipt['ready']:
        version = store.versions[0]
        ids = receipt['sample_ids']
        owner._gpu_trial['critical'] = dict(remaining=exposures, next_version=version,
            token=_pin_critical(store,key,version,ids), admitted_version=None)
        receipt.update(planned_charged_exposures=exposures, max_policy_age=store.config.max_policy_age,
                       max_sample_uses=store.config.max_sample_uses)
    return receipt


def critical_decision(owner, key, version, high_next):
    trial, store = owner._gpu_trial, owner.fresh_store
    state = trial['critical']
    ids = trial['receipt']['sample_ids']
    if (trial['key'] != key or version != state['next_version'] or version != store.versions[0] or
            state['remaining'] <= 0 or store.reservations.get(state['token']) != set(ids)):
        raise ValueError('Critical cohort reservation or publication differs')
    ages, uses = [], []
    for sid in ids:
        sample = store.samples[sid]
        birth = store.segments[sample['segment_id']].versions[sample['index']]['policy']
        ages.append(version - birth)
        uses.append(sample['uses'])
        if (sample['reserved'] != 1 or not store._eligible(dict(sample,reserved=0)) or
                ages[-1] + state['remaining'] - 1 > store.config.max_policy_age or
                uses[-1] + state['remaining'] > store.config.max_sample_uses):
            raise ValueError('Frozen exposure would exceed the live replay age or use limit')
    stock = store.model_free_inventory(576)
    ready = stock['eligible_next'] >= high_next and stock['next_two_versions_ready']
    state['admitted_version'] = version if ready else None
    return dict(ready=ready, observed_ns=time.time_ns(), stock=stock, reserved_ids=len(ids),
                min_age=min(ages), max_age=max(ages), min_uses_before=min(uses), max_uses_before=max(uses),
                exposures_remaining=state['remaining'], sample_ids_sha256=trial['receipt']['sample_ids_sha256'])


def commit_critical(owner, key, version, consumed):
    store, trial = owner.fresh_store, owner._gpu_trial
    state = trial['critical']
    ids = trial['receipt']['sample_ids']
    if (trial['key'] != key or state['admitted_version'] != version or
            version != store.versions[0] or version != state['next_version'] or
            len(consumed) != 576 or set(consumed) != set(ids)):
        raise ValueError('Cannot charge an unadmitted, stale or incomplete fixed cohort')
    store.commit(state['token'], consumed)
    state['remaining'] -= 1
    state['next_version'] += 1
    state['admitted_version'] = None
    state['token'] = _pin_critical(store,key,version+1,ids) if state['remaining'] else None
    return dict(charged=576, version=version, remaining=state['remaining'], charged_ns=time.time_ns(),
                sample_ids_sha256=trial['receipt']['sample_ids_sha256'])


async def freeze_trial(owner, count, cohort_id, *, sampling=False, layout_pool_size=None, layout_seed=None):
    output = os.environ.get('ACCERL_JIT_BENCH_OUTPUT')
    key = str(cohort_id)
    if (not output or not Path(output).is_absolute() or
            not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',key) or count != 576):
        raise ValueError('A paired GPU trial needs its absolute output directory, one cohort name and 576 actions')
    if (not owner.fresh_config.real_only or not owner.fresh_config.refresh_gae
            or len(owner.fresh_critics) != 4 or not owner.fresh_refresh_prune
            or getattr(owner,'_model_free_admission',None) is not None):
        raise RuntimeError('Trial requires independent real-only FRESH Replay with four current critics')
    if layout_pool_size is not None or layout_seed is not None:
        if sampling or layout_pool_size not in (581,2753) or type(layout_seed) is not int:
            raise ValueError('Invalid trajectory layout pool or seed')
    previous = getattr(owner,'_gpu_trial',None)
    if previous is not None:
        if previous['key'] == key and previous.get('receipt'):
            if (previous['receipt'].get('pool_size'),previous['receipt'].get('seed')) != (layout_pool_size,layout_seed):
                raise RuntimeError('Frozen cohort already has a different layout pool or seed')
            return previous['receipt']
        raise RuntimeError('A frozen trial already exists or is being written')
    if sampling:
        return freeze_sampling_round(owner, count, key)
    if layout_pool_size is not None:
        return freeze_layout_round(owner, count, key, layout_pool_size, layout_seed)
    real = owner.fresh_store
    try:
        token, ids = real.reserve(count)
    except FreshReplayNotReady:
        return dict(ready=False, replay_resident=len(real.samples))
    try:
        sampled = SegmentReplay(real.capacity, owner.fresh_config)
        sampled.set_versions(*real.versions)
        for sid in ids:
            sample = real.samples[sid]
            segment = real.segments[sample['segment_id']]
            if segment.source != 'real':
                raise ValueError('Imagined sample in a real-only paired cohort')
            sampled.samples[sid] = dict(sample)
            sampled.segments[segment.segment_id] = segment
    finally:
        # Independent timing runs do not consume training-data uses. Strong local
        # references now protect the copied draw from ongoing producer eviction.
        real.cancel(token)
    frozen = dict(key=key, ids=ids, store=sampled, busy=False, resident=len(real.samples),
                  admitted=owner.fresh_admitted_chunks['real'])
    owner._gpu_trial = frozen
    try:
        written = await asyncio.to_thread(_persist, Path(output)/'frozen', frozen, key)
        receipt = dict(ready=True, cohort_id=key, version=sampled.versions[0], n=len(ids),
                       trajectories=len(sampled.segments), sample_ids_sha256=_digest(ids),
                       replay_resident_at_draw=frozen['resident'],
                       real_admitted_at_draw=frozen['admitted'], **written)
        frozen['receipt'] = receipt
        return receipt
    except BaseException:
        owner._gpu_trial = None
        raise


def release_trial(owner, cohort_id):
    frozen = getattr(owner,'_gpu_trial',None)
    if frozen is None:
        return False
    if frozen['key'] != str(cohort_id) or frozen['busy'] or not frozen.get('receipt'):
        raise RuntimeError('Cannot release a different or active GPU cohort')
    token = frozen.get('critical', {}).get('token')
    if token:
        owner.fresh_store.cancel(token)
    owner._gpu_trial = None
    return True


async def trial_shards(owner, cohort_id, arm):
    frozen = getattr(owner,'_gpu_trial',None)
    if (frozen is None or frozen['key'] != str(cohort_id) or not frozen.get('receipt')
            or frozen['busy'] or arm not in frozen.get('draws', ('jit','pruned'))):
        raise RuntimeError('Invalid frozen GPU trial arm or concurrent replay of its inputs')
    frozen['busy'] = True
    try:
        if 'draws' in frozen:
            return await _shards(owner, frozen['draws'][arm], 'jit')
        return await _shards(owner, frozen, arm)
    finally:
        frozen['busy'] = False


async def _shards(owner, frozen, arm):
    store, ids = frozen['store'], frozen['ids']
    version = owner.fresh_store.versions[0]
    selected = defaultdict(list)
    transitions = []
    for sid in ids:
        entry = store.samples[sid]
        key, index = entry['segment_id'], entry['index']
        selected[key].append(index)
        transitions.append((store.segments[key], index))
    token_counts = [int(s.action_mask[i].sum()) for s,i in transitions]
    original_denominators = [sum(token_counts[start:start+8]) for start in range(0,len(ids),8)]
    receipt = dict(cohort_id=frozen['key'], version=version, freeze_version=store.versions[0],
                   sample_ids_sha256=frozen['receipt']['sample_ids_sha256'])
    if 'layout_arm' in frozen:
        receipt['layout_arm'] = frozen['layout_arm']
    if arm == 'pruned':
        groups = dependency_refresh_groups([store.segments[s] for s in selected], selected, 4, micro=2)
        live_groups = [(critic,group) for critic,group in zip(owner.fresh_critics,groups) if group.inputs]
        futures = [critic.autotuned_evaluate_values.remote(group.inputs,version,batch_sizes=group.batch_sizes)
                   for critic,group in live_groups]
        answers = await asyncio.gather(*futures)
        refreshed = {}
        for answer, (_, group) in zip(answers,live_groups):
            if answer['policy_version'] != version:
                raise RuntimeError('Mismatched shared critic version in paired GPU benchmark')
            refreshed.update(group.restore(answer['values']))
        actual, _ = store.materialize(ids, refreshed)
        obs, actions, adv, logits, targets = actual
        receipt['refresh_plan'] = dependency_refresh_receipt(groups)
        result = []
        for rank in range(4):
            sl = slice(144*rank,144*(rank+1))
            result.append(dict(receipt, rank=rank, sample_ids=ids[sl], observations=obs[sl],
                               actions=actions[sl], advantages=adv[sl], old_logits=logits[sl], targets=targets[sl],
                               original_denominators=[original_denominators[p//8] for p in range(sl.start,sl.stop)]))
    else:
        geometry = {key:(s.count,bool(s.terminated[-1])) for key,s in store.segments.items()}
        layout_start = time.perf_counter() if 'layout_arm' in frozen else None
        positions, counts = ordered_positions(ids,geometry,preserve_rank_shards=frozen.get('preserve_rank_shards',False))
        if layout_start is not None:
            receipt['layout_planning_s'] = time.perf_counter()-layout_start
        descriptors = {}
        for key, indices in selected.items():
            s = store.segments[key]
            first = min(indices)
            # Only the suffix is read by build_plan. Keep the original absolute
            # indices and length without moving irrelevant prefix images to Ray.
            descriptors[key] = dict(segment_id=key,count=s.count,source=s.source,
                                    observations=(None,)*first+s.observations[first:],rewards=s.rewards,
                                    terminated=s.terminated,truncated=s.truncated,gamma=s.gamma,lambda_=s.lambda_)
        result = []
        for rank, shard in enumerate(positions):
            obs, actions, logits = [], [], []
            for p in shard:
                s, i = transitions[p]
                obs.append(dict(s.observations[i], _jit_sample_id=ids[p], _jit_segment=descriptors[s.segment_id],
                                _jit_index=i, _jit_version=version, _fresh_action_mask=s.action_mask[i]))
                actions.append(s.action_tokens[i])
                logits.append(s.behaviour_logits[i])
            result.append(dict(receipt, rank=rank, sample_ids=[ids[p] for p in shard],observations=obs,
                               actions=np.stack(actions),old_logits=np.stack(logits),
                               original_denominators=[original_denominators[p//8] for p in shard],
                               bundle_context_counts=counts[rank]))
    if owner.fresh_store.versions[0] != version:
        raise RuntimeError('Critic publication changed inside a paired GPU benchmark arm')
    return result
