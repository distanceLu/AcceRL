"""Frozen P189 high-inventory pilot. One real publication per experimental decision."""
from collections import Counter, defaultdict
from itertools import permutations
import math
import os
import random
from statistics import median
import time


BATCH = 576
SHORT_NS, LONG_NS = 45_000_000, 168_000_000


def schedule(seed):
    rng = random.Random(seed)
    effects = list(permutations(('N', 'P-target', 'P-other')))
    rng.shuffle(effects)
    noise = [('N1', 'N2'), ('N2', 'N1')]
    rng.shuffle(noise)
    # Interleave one matched A/A block in each half of the crossover.
    return effects[:2] + noise[:1] + effects[2:5] + noise[1:] + effects[5:]


def calibrate(rows):
    if (sorted(r['rank'] for r in rows) != list(range(4)) or len({r['gpu_uuid'] for r in rows}) != 4 or
            any([b['bundle'] for b in r['bundles']] != list(range(9)) for r in rows)):
        raise ValueError('Calibration needs nine actual micro boundaries on four distinct GPUs')
    candidates = []
    for bundle in range(9):
        values = []
        for r in rows:
            b = r['bundles'][bundle]
            error = b.get('clock_error_ns',r.get('clock_error_ns'))
            if error is None or error < 0:
                raise ValueError('Missing calibration uncertainty')
            stamp = b['forward_ready_ns']
            lo, hi = b.get('forward_ready_lo_ns',stamp-error), b.get('forward_ready_hi_ns',stamp+error)
            if not lo<=stamp<=hi:
                raise ValueError('Invalid finite-forward completion time bounds')
            values.append((stamp, r['rank'], lo, hi, r['gpu_uuid']))
        values.sort()
        last, runner = values[-1], values[-2]
        gap = last[0] - runner[0]
        conservative = last[2] - max(row[3] for row in values[:-1])
        candidates.append(dict(target_rank=last[1], target_bundle=bundle, other_rank=values[0][1],
            target_uuid=last[4], other_uuid=values[0][4], gap_ns=gap,
            conservative_gap_ns=conservative, resolved_late_rank=conservative>0))
    return max(candidates, key=lambda c:(c['conservative_gap_ns'],c['gap_ns'],-c['target_bundle']))


def _spans(rows, start_kind, end_kind):
    spans = {}
    for row in rows:
        kind = row['kind']
        if kind not in (start_kind, end_kind):
            continue
        span = spans.setdefault(row['batch_id'], dict(requests=set()))
        key = 'start' if kind == start_kind else 'end'
        span[key] = min(row['ts_ns'],span.get(key,row['ts_ns']))
        span['requests'].add(row['request_id'])
    return spans


def _baseline(rows, rank, uuid, start, stop, start_kind, end_kind):
    reference = defaultdict(list)
    matched = [r for r in rows if r['rank']==rank and r.get('gpu_uuid')==uuid and start<=r['ts_ns']<=stop]
    for span in _spans(matched,start_kind,end_kind).values():
        if 'start' in span and 'end' in span and span['end'] > span['start']:
            reference[len(span['requests'])].append(span['end']-span['start'])
    return {n:int(median(x)) for n,x in reference.items()}


def competition_state(rows, rank, uuid, at_ns, gpu_baseline, short_ns, long_ns, service_baseline=None):
    matched = [r for r in rows if r['rank']==rank and r.get('gpu_uuid')==uuid]
    before = [r for r in matched if r['ts_ns'] <= at_ns]
    queued = {r['request_id'] for r in before if r['kind']=='enqueue'}
    queued.difference_update(r['request_id'] for r in before if r['kind']=='service_start')
    result = dict(cutoff_ns=at_ns,rank=rank,gpu_uuid=uuid,queued_requests=len(queued))
    for stage, baseline in [('gpu',gpu_baseline),('service',service_baseline or {})]:
        start, end = stage+'_start', stage+'_end'
        active = [(key,s) for key,s in _spans(before,start,end).items() if 'start' in s and 'end' not in s]
        if len(active)>1:
            raise ValueError('Overlapping inference batches on the same serial replica')
        age = delay = realized = None
        if active:
            batch, span = active[0]
            age = at_ns-span['start']
            typical = baseline.get(len(span['requests']))
            delay = None if typical is None else max(0,typical-age)
            finish = _spans(matched,start,end)[batch].get('end')
            realized = None if finish is None else finish-at_ns
        result.update({stage+'_active':bool(active),stage+'_batch_id':active[0][0] if active else None,
            stage+'_elapsed_ns':age,'projected_'+stage+'_residual_ns':delay,
            'realized_'+stage+'_residual_ns':realized})
    def bucket(active,duration):
        if not active:
            return 'no_service'
        if duration is None:
            return 'unknown'
        return 'short' if duration<short_ns else 'long' if duration>=long_ns else 'middle'
    for stage in ('service','gpu'):
        for source in ('projected','realized'):
            result[source+'_'+stage+'_state'] = bucket(result[stage+'_active'],result[source+'_'+stage+'_residual_ns'])
    result['state'] = result['projected_gpu_state']
    return result


def _workload(frozen, results, version):
    from paired_gpu_driver import verify_layout_exposures
    if (sorted(r['rank'] for r in results) != list(range(4)) or
        any(r['version']!=version or r['version_after']!=version+1 or r['learning_rate']!=0 or
            r['sample_ids_sha256']!=frozen['sample_ids_sha256'] or r['optimizer_delta']!=1 or
            r['skipped_delta']!=0 or r['train_micro']!=16 or r['gas']!=9 or not r['actor_loss_enabled'] or
            r['layout_arm']!='preserve' or r['cache_trim']!=dict(micro='0',idle='1') or
            r['initialization_prepass']!=(version==0) for r in results)):
        raise ValueError('Incomplete frozen 4-rank JIT optimizer update')
    verify_layout_exposures('preserve',frozen['sample_ids'],results,frozen['original_group_denominators'])
    return [dict(rank=r['rank'], ids_sha256=r['consumed_sample_ids_sha256'],
        context_observations=r['planned_bundle_context_counts'],
        context_calls=[[c['bundle'],c['count']] for c in r['context']['calls_detail']])
        for r in sorted(results,key=lambda r:r['rank'])]


def _publication(receipt, step, start):
    ranks = receipt['receiver_ranks']
    times = receipt['rank_ACK_observed_ns']
    if (receipt['version'] != step or ranks != list(range(4)) or len(times)!=4 or
            any(t<start for t in times) or max(times)>receipt['ack_completed_ns'] or
            any(i['version']!=step for i in receipt['inventories'])):
        raise ValueError('Missing four matching real IPC acknowledgement receipts')
    return max(times)


def _diagnose(block, actions):
    target = block['target']
    rank, uuid = target['target_rank'], target['target_uuid']
    cal = block['calibration']
    baseline = {s:_baseline(actions,rank,uuid,cal['intent_ns'],cal['last_ack_ns'],s+'_start',s+'_end')
                for s in ('service','gpu')}
    block['calibration_service_median_ns_by_batch_size'] = baseline
    response = {r['request_id']:r['ts_ns'] for r in actions if r['kind']=='response'}
    for arm in block['arms']:
        row = next(r for r in arm['ranks_probe'] if r['rank']==rank)
        b = row['bundles'][target['target_bundle']]
        gate_rank = arm['gate_rank']
        issued, closed, expires = b['gate_issued_ns'], b['gate_closed_ns'], b['gate_until_ns']
        physical = {r['rank']:(r['gpu_uuid'],int(r['physical_gpu'])) for r in arm['ranks_probe']}
        for rank_id, binding in physical.items():
            actual = {(r['gpu_uuid'],int(r['physical_gpu'])) for r in actions if r['rank']==rank_id and
                      r['kind']=='service_start' and arm['intent_ns']<=r['ts_ns']<=arm['last_ack_ns']}
            if actual != {binding}:
                raise ValueError(f"Missing matched inference service on {binding} in {arm['arm_id']}: {actual}")
        if (row['gpu_uuid']!=uuid or b['gate_rank']!=gate_rank or
                not b['forward_start_ns']<=issued<=closed or expires<=issued or
                bool(b['gate_effective']) != arm['label'].startswith('P-')):
            raise ValueError('Gate receipt differs from preregistered trigger, address or sham condition')
        state = competition_state(actions,rank,uuid,b['forward_start_ns'],baseline['gpu'],
                                  SHORT_NS,LONG_NS,baseline['service'])
        visible = b.get('gate_visible_by_ns',issued)
        end = min(closed,expires)
        held = [r for r in actions if r['kind']=='dispatch' and r.get('gate_arm_id')==arm['arm_id']
                and r.get('held_ns',0)>0]
        if any(r['rank']!=gate_rank for r in held) or held and not b['gate_effective']:
            raise ValueError('Request held on an unintended UUID or during sham')
        ready = [r for r in actions if r['kind']=='ready' and r['rank']==gate_rank and visible<=r['ts_ns']<end]
        addressed = {r['request_id'] for r in ready}
        inside = [r for r in actions if r['rank']==gate_rank and r['kind']=='dispatch' and issued<=r['ts_ns']<end]
        leaks = [r for r in inside if r['request_id'] in addressed and r.get('remote_returned_ns',r['ts_ns'])<end]
        edge = [r for r in inside if r['request_id'] in addressed and r.get('remote_returned_ns',r['ts_ns'])>=end]
        delayed = {r['request_id'] for r in held if r['request_id'] in addressed and r['ts_ns']>=end}
        bundles = {r['rank']:r['bundles'][target['target_bundle']] for r in arm['ranks_probe']}
        ranks = sorted((v['forward_ready_ns'],k) for k,v in bundles.items())
        others = [v for k,v in bundles.items() if k!=rank]
        last_vote = max(v['vote_done_ns'] for v in bundles.values())
        arm['four_rank_check'] = dict(last_vote_ns=last_vote,
            last_vote_lo_ns=max(v['vote_done_lo_ns'] for v in bundles.values()),
            last_vote_hi_ns=max(v['vote_done_hi_ns'] for v in bundles.values()),
            last_vote_from_intent_s=(last_vote-arm['intent_ns'])/1e9,
            last_vote_from_target_entry_s=(last_vote-b['forward_start_ns'])/1e9,
            target_forward_from_entry_s=(b['forward_ready_ns']-b['forward_start_ns'])/1e9,
            vote_spread_s=(last_vote-min(v['vote_done_ns'] for v in bundles.values()))/1e9)
        arm.update(competition_before_gate=state, measured_last_rank=ranks[-1][1],
            target_arrival_minus_latest_other_ns=b['forward_ready_ns']-max(t for t,r in ranks if r!=rank),
            target_arrival_minus_latest_other_lo_ns=b['forward_ready_lo_ns']-max(v['forward_ready_hi_ns'] for v in others),
            target_arrival_minus_latest_other_hi_ns=b['forward_ready_hi_ns']-max(v['forward_ready_lo_ns'] for v in others),
            gate=dict(rank=gate_rank,issued_ns=issued,visible_by_ns=visible,closed_ns=closed,expires_ns=expires,
                ttl_expired=expires<closed,new_ready_in_window=len(ready),dispatch_in_window=len(inside),
                new_ready_dispatched_before_reopen=len(leaks),new_ready_held_until_reopen=len(delayed),
                dispatch_straddled_reopen=len(edge),
                held_requests=len(held), held_request_ids=[r['request_id'] for r in held],
                held_total_s=sum(r['held_ns'] for r in held)/1e9,
                manipulated=bool(delayed) and not leaks and not edge))
        starts = [r for r in actions if r['kind']=='ready' and arm['intent_ns']<=r['ts_ns']<=arm['last_ack_ns']]
        for key, requests in [('all',starts),('target',[r for r in starts if r['rank']==rank]),
                              ('addressed_window',ready)]:
            elapsed = sorted((response[r['request_id']]-r['ts_ns'])/1e9 for r in requests
                             if r['request_id'] in response)
            arm.setdefault('ready_to_response',{})[key] = dict(ready=len(requests),complete=len(elapsed),
                censored=len(requests)-len(elapsed),p95_s=elapsed[math.ceil(.95*len(elapsed))-1] if elapsed else None)
        for event in ('ready','dispatch','response','gpu_start','gpu_end'):
            streamed = [r for r in actions if r['kind']==event and arm['intent_ns']<=r['ts_ns']<=arm['last_ack_ns']]
            arm.setdefault('action_counts',{})[event] = [sum(r['rank']==i for r in streamed) for i in range(4)]


def _summary(blocks):
    effects, noise, outcomes = [], [], Counter()
    for block in blocks:
        arms = {a['label']:a for a in block['arms']}
        if 'N1' in arms:
            noise.append(arms['N2']['t1_s']-arms['N1']['t1_s'])
            continue
        for a in arms.values():
            outcomes[(a['label'],a['competition_before_gate']['projected_service_state'])] += 1
        states = {k:v['competition_before_gate']['projected_service_state'] for k,v in arms.items()}
        uniform = len(set(states.values()))==1
        def difference(extract):
            return {k:(extract(arms[k])-extract(arms['N'])) if None not in (extract(arms[k]),extract(arms['N']))
                    else None for k in ('P-target','P-other')}
        effects.append(dict(cohort=block['cohort'], calibration_resolved=block['target']['resolved_late_rank'],
            target_gate_held=arms['P-target']['gate']['manipulated'],
            other_gate_held=arms['P-other']['gate']['manipulated'],
            sham_target_confirmed_late=arms['N']['target_arrival_minus_latest_other_lo_ns']>0,
            projected_service_states=states, same_state=states['N'] if uniform else None,
            all_rank_vote_minus_normal_s=difference(lambda a:a['four_rank_check']['last_vote_from_intent_s']),
            ready_to_response_p95_minus_normal_s=difference(lambda a:a['ready_to_response']['all']['p95_s']),
            target_minus_normal_s=arms['P-target']['t1_s']-arms['N']['t1_s'],
            other_minus_normal_s=arms['P-other']['t1_s']-arms['N']['t1_s'],
            target_minus_other_s=arms['P-target']['t1_s']-arms['P-other']['t1_s']))
    supported = Counter(r['same_state'] for r in effects if r['same_state'] in ('short','long') and
        r['calibration_resolved'] and r['sham_target_confirmed_late'] and r['target_gate_held'] and r['other_gate_held'])
    return dict(matched_contrasts=effects, noise_differences_s=noise,
        same_state_manipulated_blocks={s:supported[s] for s in ('short','long')},
        observed_pretreatment_state_counts=[dict(label=k[0],state=k[1],arms=n) for k,n in sorted(outcomes.items())],
        endpoint='T1 from common predispatch intent through fourth observed CUDA-IPC receiver ACK',
        inference='Randomized arm assignment. Competition state is observational. No T3 measured.')


def run_critical_jit_benchmark(*, ray, plan, output, replay, trainers, inference, collector,
                               workers, sync_policy, experiment, stop_requested):
    from critical_probe import configure_probe, read_actions, read_probe
    from frozen_policy_eval import atomic_json
    from paired_gpu_driver import common_layout_scalars

    required = dict(comparison='critical_jit', checkpoint_policy_step=189, real_workers=8,
        real_sampling='uniform_blocks', no_micro_cache_trim=True, compile_action_inference=False,learning_rate=0.,
        global_batch=576, rank_size=144, train_micro=16, context_micro=2, warmup_updates=1,
        measured_blocks=6, noise_blocks=2, updates_per_arm=1, minimum_replay_chunks=2753,
        high_next_chunks=1152, gate_ttl_s=5., protocol_seed=20261125)
    if (any(plan.get(k)!=v for k,v in required.items()) or len(workers)!=8 or len(trainers)!=4 or
            len(inference)!=4 or os.environ.get('ACCERL_TRAJECTORY_JIT')!='1' or
            os.environ.get('ACCERL_REAL_SAMPLING')!='uniform_blocks' or
            os.environ.get('ACCERL_COMPILE_ACTION_INFERENCE')=='1' or
            os.environ.get('ACCERL_ROLLOUT_INVENTORY_GATE')=='1'):
        raise ValueError('Unexpected critical JIT topology, arm protocol or inference mode')
    seed = plan['protocol_seed']
    orders = schedule(seed)
    report = dict(status='running', plan=plan, run_id=os.environ['ACCERL_RUN_ID'], git_commit=plan['git_commit'],
        learning_rate=0, order=orders, blocks=[], completed_updates=0, residual_bins_ns=[SHORT_NS,LONG_NS],
        fixed_inputs='Preserved rank membership/order and complete suffixes. Every exposure including calibration '
                     'reserves and charges one legal live use. Numeric P189 weights are frozen.',
        residual_protocol='On the calibrated UUID, classify active service at target forward entry. Project residual '
                     'from previous calibration-only same-size batch median minus elapsed service. Realized residual '
                     'uses the later completion and is descriptive only. No service is a separate state. GPU stage '
                     'is host entry into model-forward/output-copy through CPU-visible result, including stream waits.')
    version = 0
    worker_refs = []
    deadline = time.monotonic()+2*3600+55*60

    def progress(phase):
        report['phase'] = phase
        atomic_json(output/'progress.json',report)
        print('[critical JIT] '+phase,flush=True)

    def check():
        if stop_requested():
            raise InterruptedError('Critical JIT stop requested')
        if time.monotonic()>=deadline:
            raise TimeoutError('Critical JIT benchmark reached the bounded lease')
        if worker_refs:
            ready, _ = ray.wait(worker_refs,num_returns=1,timeout=0)
            if ready:
                ray.get(ready[0])
                raise RuntimeError('A real rollout worker exited during the crossover')

    def digests():
        rows = ray.get([t.paired_gpu_weight_digest.remote() for t in trainers],timeout=180)
        if sorted(r['rank'] for r in rows)!=list(range(4)) or len({r['sha256'] for r in rows})!=1:
            raise ValueError('Four P189 numerical weight digests differ')
        return rows

    def wait_ready(lookup, phase, seconds):
        limit = min(deadline,time.monotonic()+seconds)
        next_log = 0
        while True:
            check()
            value = ray.get(lookup(),timeout=90)
            if value['ready']:
                return value
            if time.monotonic()>=limit:
                raise TimeoutError(f'{phase}: {value}')
            if time.monotonic()>=next_log:
                report['waiting_for'] = value
                progress(phase)
                next_log = time.monotonic()+45
            time.sleep(1.)

    def freeze(cid, index, exposures):
        return wait_ready(lambda:replay.freeze_critical_jit.remote(cid,plan['minimum_replay_chunks'],
            seed+index,exposures),'waiting for legal fixed cohort '+cid,3600)

    def update(block, label, *, target=None):
        nonlocal version
        check()
        cid, frozen = block['cohort'], block['frozen']
        step = version+1
        arm_id = f'{cid}-{label}-v{step}'
        condition = label if label.startswith('P-') else 'N' if target else 'calibration'
        gate_rank = None if target is None else target['other_rank' if label=='P-other' else 'target_rank']
        spec = configure_probe(output,step=step,arm_id=arm_id,condition=condition,
            target_rank=target and target['target_rank'], target_bundle=target and target['target_bundle'],
            gate_rank=gate_rank, ttl_s=plan['gate_ttl_s'])
        progress('preparing '+arm_id)
        if version:
            first = 'input_ema' not in block
            ema = ray.get([t.prepare_layout_ema.remote(cid,version,first,repeated=True) for t in trainers],timeout=60)
            initial = common_layout_scalars(ema,version,('mean','variance','scale_floor'))
            if any(r['updates']!=version for r in ema) or block.setdefault('input_ema',initial)!=initial:
                raise ValueError('Fixed arms did not start from identical numeric advantage moments')
        ray.get([i.configure_batch_autotune.remote(step,2) for i in inference],timeout=45)
        ray.get([i.get_batch_autotune_stats.remote(step) for i in inference],timeout=45)
        inventory = wait_ready(lambda:replay.critical_jit_decision.remote(cid,version,plan['high_next_chunks']),
                               'waiting for high future stock '+arm_id,1200)
        intent = time.time_ns()
        shards = replay.gpu_trial_batch.options(num_returns=4).remote(cid,'preserve')
        pending = [t.run_paired_gpu_arm.remote('jit',s) for t,s in zip(trainers,shards)]
        # Leave time for queue cleanup and stop promptly on rollout failure during an update.
        rank_limit = min(deadline,time.monotonic()+900)
        active = list(pending)
        while active:
            finished,active = ray.wait(active,num_returns=1,timeout=10)
            if finished:
                ray.get(finished[0],timeout=30)
            check()
            if time.monotonic()>=rank_limit:
                raise TimeoutError('Trainer rank exceeded watchdog for '+arm_id)
        results = ray.get(pending,timeout=30)
        optimizer_done = time.time_ns()
        consumed = [sid for r in results for sid in r['consumed_sample_ids']]
        if any(r['version_after']!=step or r['optimizer_delta']!=1 or r['skipped_delta'] for r in results):
            raise ValueError('Cannot charge a skipped or incomplete critical optimizer update')
        charged = ray.get(replay.commit_critical_jit.remote(cid,version,consumed),timeout=60)
        ack = sync_policy()
        end = _publication(ack,step,intent)
        inventory_after = ray.get(replay.critical_jit_inventory.remote(),timeout=45)
        if inventory_after['policy_version']!=step:
            raise ValueError('Live replay did not advance after all four policy ACKs')
        arm = dict(arm_id=arm_id,label=label,condition=condition,gate_rank=gate_rank,stock_at_decision=inventory,
            inventory_after_publication=inventory_after,
            intent_ns=intent, optimizer_observed_ns=optimizer_done, last_ack_ns=end, t1_s=(end-intent)/1e9,
            input_version=version,published_version=step,charged=charged,publication=ack,
            ranks_probe=read_probe(output,step))
        if (len(arm['ranks_probe'])!=4 or
                any(any(r.get(k)!=v for k,v in spec.items()) for r in arm['ranks_probe'])):
            raise ValueError('Wrong step, arm or rank in critical JIT timing receipts')
        if {int(r['physical_gpu']) for r in arm['ranks_probe']}!={4,5,6,7}:
            raise ValueError('Rank trace does not cover physical GPUs 4–7')
        work = _workload(frozen,results,version)
        if work != block.setdefault('workload',work):
            raise ValueError('Fixed arms changed the consumed IDs, micro order or actual context call work')
        values = ray.get([t.read_layout_normalization.remote(version) for t in trainers],timeout=45)
        arm['loss_normalization'] = common_layout_scalars(values,version,('mean','scale'))
        if version:
            scale = max(math.sqrt(initial['variance']),initial['scale_floor'])
            if (arm['loss_normalization']['mean'] != initial['mean'] or not math.isclose(
                    arm['loss_normalization']['scale'],scale,rel_tol=1e-15,abs_tol=0)):
                raise ValueError('Fixed arms did not use the requested EMA scalars in the objective')
        traffic = ray.get([i.get_batch_autotune_stats.remote(step) for i in inference],timeout=45)
        arm['action_completions'] = [dict(rank=e['rank'],count=len(e['action_latencies'])) for e in traffic]
        if (sorted(r['rank'] for r in traffic)!=list(range(4)) or
                any(not r['action_latencies'] or r['refresh_observations'] for r in traffic)):
            raise ValueError('Expected live rollout traffic on four eager policy replicas and no critic refresh')
        version=step
        report['completed_updates'] = step
        return arm

    try:
        progress('starting fixed-weight pilot')
        report['initial_weights'] = digests()
        report['initial_inventory'] = ray.get(replay.critical_jit_inventory.remote(),timeout=45)
        ray.get([i.set_training_batch_limit.remote(2) for i in inference],timeout=45)
        worker_refs = [w.run.remote() for w in workers]
        for index, labels in enumerate([()] + orders):
            cid = 'critical-warmup' if index==0 else f'critical-block-{index:02d}'
            frozen = freeze(cid,index,1+len(labels))
            block = dict(cohort=cid,frozen=frozen,arms=[])
            if index:
                report['blocks'].append(block)
            else:
                report['warmup'] = block
            try:
                block['calibration'] = update(block,'calibration')
                if index:
                    block['target'] = calibrate(block['calibration']['ranks_probe'])
                    progress('locked calibration and order '+cid)
                for label in labels:
                    block['arms'].append(update(block,label,target=block['target']))
                    progress(f'{cid} {label} ACK P{version}')
                if index:
                    _diagnose(block,read_actions(output))
            finally:
                ray.get(replay.release_gpu_trial.remote(cid),timeout=60)
        check()
        report['final_weights']=digests()
        report['final_inventory'] = ray.get(replay.critical_jit_inventory.remote(),timeout=45)
        if report['final_weights'][0]['sha256']!=report['initial_weights'][0]['sha256']:
            raise ValueError('Numerical policy weights changed during critical JIT trials')
        # Re-read once to include completions that crossed the last measured boundary.
        actions=read_actions(output)
        for block in report['blocks']:
            _diagnose(block,actions)
        report['summary'] = _summary(report['blocks'])
        report.update(status='completed',phase='completed')
        experiment.event('critical_jit_benchmark_completed',details=dict(completed_updates=version))
        experiment.flush()
        atomic_json(output/'result.json',report)
        progress('completed')
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        progress('failed')
        raise
    finally:
        ray.get(collector.stop_pipeline.remote(),timeout=30)
