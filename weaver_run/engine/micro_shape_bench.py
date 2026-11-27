"""Frozen P189, four-rank paired micro-batch comparison under live rollout traffic."""
import json
import math
from statistics import mean
import time


SHAPE_PAIRS = ((16,24), (24,16), (24,16), (16,24))
TOLERANCES = dict(gae_atol=0.01, gae_rtol=0.01, gradient_relative_l2=0.02, gradient_cosine=0.999)
LATENCY_STAGES = {
    'inference_service_ms': ('policy_infer_gpu_forward_and_copy','service',4),
    'client_wait_ms': ('policy_infer','client_wait',8),
}


def shape_schedule():
    # Warm 24 then 16 after auditing 24. This avoids five redundant updates
    # before the first measured 16 -> 24 pair while retaining both warmups.
    requested = [('cold',16,None), ('audit',16,None), ('audit',24,None),
                 ('warmup',24,None), ('warmup',16,None)]
    requested.extend(('time',micro,pair) for pair,order in enumerate(SHAPE_PAIRS,1) for micro in order)
    steps, previous = 0, None
    for phase, micro, pair in requested:
        gas = 144 // micro
        # DeepSpeed indexes accumulation by lifetime micro_steps. Finish on a
        # common boundary using complete, explicitly unmeasured updates.
        while steps % gas:
            yield 'bridge', previous, None
            steps += 144 // previous
        yield phase, micro, pair
        steps += gas
        previous = micro


def _ranks(rows):
    rows = sorted(rows,key=lambda r:r['rank'])
    if [r['rank'] for r in rows] != list(range(4)):
        raise ValueError('Four distinct rank receipts are required')
    return rows


def compare_audits(reference, candidate, trainable_elements):
    base, alt = _ranks(reference), _ranks(candidate)
    norm = base[0]['normalization']
    if (not all(math.isfinite(norm[k]) for k in ('mean','scale')) or norm['scale'] <= 0 or
            any(r['normalization'] != norm for r in base+alt)):
        raise ValueError('GAE parity needs identical finite global normalization on all ranks')
    errors = {name: [] for name in ('advantage','value_target')}
    passed = True
    seen = set()
    for left,right in zip(base,alt):
        def by_id(row):
            values = dict(zip(row['consumed_sample_ids'],row['advantage_value_targets'],strict=True))
            if len(values)!=144 or any(len(v)!=2 or not all(map(math.isfinite,v)) for v in values.values()):
                raise ValueError('GAE parity needs 144 finite advantage/return pairs on each rank')
            return values
        a,b = by_id(left),by_id(right)
        if a.keys()!=b.keys() or seen.intersection(a):
            raise ValueError('GAE audit changed selected rank membership')
        seen.update(a)
        for sid,x in a.items():
            for index,name in enumerate(errors):
                diff = abs(x[index]-b[sid][index])
                passed &= diff <= TOLERANCES['gae_atol'] + TOLERANCES['gae_rtol']*abs(x[index])
                errors[name].append(diff)
    gae = {name:dict(max_abs=max(v), rms=math.sqrt(mean(x*x for x in v))) for name,v in errors.items()}
    gae.update(passed=passed,samples=len(seen))

    gradients = [row['shape_gradient'] for row in alt]
    if (len({g['schema_sha256'] for g in gradients})!=1 or
            sum(g['elements'] for g in gradients)!=trainable_elements or trainable_elements<=0):
        raise ValueError('Gradient partition coverage does not match the trainable checkpoint inventory')
    for left,right in zip(base,gradients):
        initial = left['shape_gradient']
        if (right['logical_elements']!=trainable_elements or right['elements']<=0 or
                any(initial[k]!=right[k] for k in ('schema_sha256','logical_elements','elements','reference_sq')) or
                not initial['finite'] or not right['finite'] or
                not all(math.isfinite(right[k]) for k in
                        ('candidate_sq','reference_sq','diff_sq','dot','max_abs_error')) or
                min(right[k] for k in ('candidate_sq','reference_sq','diff_sq','max_abs_error'))<0):
            raise ValueError('Invalid or mismatched ZeRO gradient partition receipt')
    def gradient_metrics(parts):
        left_sq,right_sq,delta_sq,dot = [math.fsum(g[key] for g in parts)
                                        for key in ('reference_sq','candidate_sq','diff_sq','dot')]
        relative = math.sqrt(delta_sq/left_sq) if left_sq else None
        cosine = dot/math.sqrt(left_sq*right_sq) if min(left_sq,right_sq)>0 else None
        ok = (left_sq==right_sq==delta_sq==0 if cosine is None else
              relative<=TOLERANCES['gradient_relative_l2'] and cosine>=TOLERANCES['gradient_cosine'])
        return dict(relative_l2=relative,cosine=cosine,max_abs_error=max(g['max_abs_error'] for g in parts),
                    reference_norm=math.sqrt(left_sq),candidate_norm=math.sqrt(right_sq),passed=ok)
    grad = gradient_metrics(gradients)
    if grad['reference_norm']<=0:
        raise ValueError('Gradient parity needs a nonzero full-model reference norm')
    # Detect a regression in a smaller optimizer group even if another group
    # dominates the norm over the entire checkpoint.
    groups = [gradient_metrics(parts) for parts in zip(*(g['groups'] for g in gradients),strict=True)]
    grad.update(elements=trainable_elements,schema_sha256=gradients[0]['schema_sha256'],groups=groups,
                passed=grad['passed'] and bool(groups) and all(g['passed'] for g in groups))
    return dict(passed=bool(gae['passed'] and grad['passed']),gae=gae,gradient=grad,normalization=norm)


def latency_samples(db, run_id, start, end):
    samples = {}
    for name,(workload,kind,actors) in LATENCY_STAGES.items():
        rows = db.execute("""select actor_id,duration_ns from events
            where run_id=? and workload=? and kind=? and end_ns>? and end_ns<=?
            and status='ok' and duration_ns>=0""",(run_id,workload,kind,start,end)).fetchall()
        if len({r[0] for r in rows})!=actors:
            raise ValueError(f'Missing live replicas or collectors for {workload}/{kind} in {start}..{end}')
        samples[name] = sorted(ns/1e6 for _,ns in rows)
    return samples


def distribution(values):
    sample = sorted(values)
    def quantile(q):
        i = (len(sample)-1)*q
        a,b = math.floor(i), math.ceil(i)
        return sample[a]+(sample[b]-sample[a])*(i-a)
    return dict(n=len(sample),p50=quantile(.5),p95=quantile(.95))


def _final_metrics(pairs, db, run_id, check):
    # Each writer drains its ordered WAL queue periodically while rollouts
    # continue. Wait for a later completion from each producer before reading.
    cutoff = max(a['timing_end_ns'] for p in pairs for a in p['arms'])
    deadline = time.monotonic()+90
    for workload,kind,n in LATENCY_STAGES.values():
        while db.execute("""select count(distinct actor_id) from events where run_id=? and workload=?
            and kind=? and end_ns>? and status='ok'""",(run_id,workload,kind,cutoff)).fetchone()[0]<n:
            check()
            if time.monotonic()>deadline:
                raise RuntimeError('Timed out waiting for post-measurement rollout ledger watermarks')
            time.sleep(.5)
    by_shape = {}
    for micro in (16,24):
        arms = [a for pair in pairs for a in pair['arms'] if a['micro']==micro]
        pooled = {key:[] for key in LATENCY_STAGES}
        for arm in arms:
            samples = latency_samples(db,run_id,arm['timing_start_ns'],arm['timing_end_ns'])
            arm['rollout_latency'] = {k:distribution(v) for k,v in samples.items()}
            for key,values in samples.items():
                pooled[key].extend(values)
        by_shape[str(micro)] = dict(measured_arms=len(arms),mean_optimizer_wall_s=mean(a['optimizer_wall_s'] for a in arms),
            mean_optimizer_plus_ipc_s=mean(a['optimizer_plus_ipc_s'] for a in arms),
            mean_ipc_s=mean(a['ipc_s'] for a in arms),
            pooled_rollout_latency={k:distribution(v) for k,v in pooled.items()},
            peak_by_rank=[{key:max(a['ranks'][rank][key] for a in arms)
                           for key in ('peak_allocated_bytes','peak_reserved_bytes','sampled_peak_used_bytes','estimated_peak_used_bytes')}
                          for rank in range(4)])
    paired = []
    for pair in pairs:
        a,b = sorted(pair['arms'],key=lambda row:row['micro'])
        paired.append(dict(pair=pair['pair'],optimizer_24_minus_16_s=b['optimizer_wall_s']-a['optimizer_wall_s'],
            service_p95_24_minus_16_ms=b['rollout_latency']['inference_service_ms']['p95']-a['rollout_latency']['inference_service_ms']['p95'],
            client_p95_24_minus_16_ms=b['rollout_latency']['client_wait_ms']['p95']-a['rollout_latency']['client_wait_ms']['p95']))
    return dict(by_micro=by_shape,paired_differences=paired)


def run_micro_shapes(*, ray, plan, output, replay, trainers, inference, collector, workers,
                     sync_policy, experiment, stop_requested):
    from critical_jit_bench import _publication
    from frozen_policy_eval import atomic_json
    from gpu_trial_replay import _digest
    from paired_gpu_driver import verify_layout_exposures
    from training_topology import validate_trainer_mps_bindings

    required = dict(train_shapes=[[16,9],[24,6]], real_workers=8, global_batch=576, rank_size=144,
                    real_sampling='uniform_blocks', trainer_mps_percent=90, trainer_compile='on',
                    compile_action_inference=True, no_micro_cache_trim=True, learning_rate=0.,
                    parity_pairs=1, warmup_pairs=1, measured_pairs=4, minimum_replay_chunks=2753,
                    protocol_seed=1127, checkpoint_policy_step=189)
    if any(plan.get(k)!=value for k,value in required.items()):
        raise ValueError('Unexpected paired shape plan or P189 source')
    cid = 'shape-cohort'
    report = dict(status='running', run_id=experiment.run_id, git_commit=plan['git_commit'], plan=plan,
        tolerances=TOLERANCES, parity_passed=False, setup=[], audits=[], pairs=[],
        protocol='Fixed P189 weights, one immutable uniform-block draw, preserved rank membership and '
                 'input order, identical frozen eight-sample denominators and numerical AdamW/EMA state '
                 'after the initial warmup. No live Replay uses charged. Logical publications advance.',
        timing='Four-rank optimizer wall spans earliest post-restore start through latest GPU completion. '
               'IPC duration is reported separately. Dispatch-to-last-ACK includes preparation and reporting. '
               'Allocator reservations use the shared warmed process. Sampled whole-device use is total minus '
               'free at micro boundaries; estimated use adds reservations released since the last peak reset. '
               'Latency uses successful service or client completions inside each optimizer window.')
    worker_refs, frozen = [], None
    db = experiment.connection

    def progress(phase):
        report['phase'] = phase
        atomic_json(output/'progress.json',report)
        print('[paired micro shapes] '+phase,flush=True)

    def check():
        if stop_requested():
            raise InterruptedError('Paired shape benchmark stop requested')
        if worker_refs:
            ready,_ = ray.wait(worker_refs, num_returns=1,timeout=0)
            if ready:
                ray.get(ready[0])
                raise RuntimeError('A real rollout worker exited during the paired shape benchmark')

    def digests():
        rows = _ranks(ray.get([t.paired_gpu_weight_digest.remote() for t in trainers],timeout=180))
        if len({(r['sha256'],r['elements']) for r in rows})!=1:
            raise ValueError('Fixed P189 numerical weights differ across ranks')
        return rows

    def graph_counters(step):
        reports = _ranks(ray.get([i.get_batch_autotune_stats.remote(step,include_graph=True) for i in inference],
                                 timeout=90))
        return [r['compiled_policy'] for r in reports]

    def arm_run(version,phase,micro):
        check()
        step = version+1
        method = 'shape_'+(phase if phase in ('audit','time') else 'warmup')+str(micro)
        progress(f'P{step} {phase} {micro}x{144//micro}')
        ray.get([i.configure_batch_autotune.remote(step,2) for i in inference],timeout=60)
        before_graph = graph_counters(step)
        dispatch = time.time_ns()
        shards = replay.gpu_trial_batch.options(num_returns=4).remote(cid,'preserve')
        results = _ranks(ray.get([t.run_paired_gpu_arm.remote(method,s) for t,s in zip(trainers,shards)],
                                 timeout=1800))
        if any(r['version']!=version or r['version_after']!=step or r['cohort_id']!=cid+'-preserve'
               or r['micro']!=micro or r['gas']!=144//micro or r['arm']!=method or r['learning_rate']!=0.
               or r['optimizer_steps']!=1 or r['skipped_steps']!=0 or r['cache_trim']['micro']!='0'
               or (r['shape_gradient'] is not None)!=(phase=='audit') for r in results):
            raise ValueError('Shape arm failed to perform the required single four-rank optimizer update')
        verify_layout_exposures('preserve',frozen['sample_ids'],results,frozen['original_group_denominators'])
        rank_orders = [r['input_order_sha256'] for r in results]
        if phase=='cold':
            ordered = [r['input_sample_ids'] for r in results]
            if ([_digest(ids) for ids in ordered]!=rank_orders or
                    any(set(ids)!=set(frozen['sample_ids'][rank*144:(rank+1)*144]) for rank,ids in enumerate(ordered))):
                raise ValueError('Initial rank input orders disagree with the immutable cohort')
            report['rank_input_orders'] = ordered
            report['rank_input_order_sha256'] = rank_orders
        if rank_orders!=report['rank_input_order_sha256']:
            raise ValueError('Shape arm changed the input order of a fixed rank shard')
        if phase not in ('cold','audit') and any(r['normalization']!=report['parity']['normalization'] for r in results):
            raise ValueError('Shape arm changed the audited normalization')
        for r in results:
            compile_state = r['trainer_compile']
            registered = compile_state['registered']
            if not registered['compiled'] or registered['backend']!='inductor':
                raise ValueError('Missing registered Inductor trainer')
            if (phase=='cold' or (phase=='audit' and micro==24)) and not (
                    compile_state['delta']['unique_graphs']>0 and
                    compile_state['delta']['fxgraph_cache_hit']+compile_state['delta']['fxgraph_cache_miss']>0):
                raise ValueError(f'Missing first-use Inductor graph evidence for micro={micro} rank={r["rank"]}')
            if phase=='time' and (any(compile_state['delta'].values()) or
                    min(r[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','sampled_peak_used_bytes'))<=0
                    or r['sampled_peak_used_bytes']>r['estimated_peak_used_bytes']):
                raise ValueError('Measured shape recompiled or omitted memory peaks')
        ack = sync_policy()
        last_ack = _publication(ack,step,dispatch)
        after_graph = graph_counters(step)
        graph = []
        for rank,(a,b) in enumerate(zip(before_graph,after_graph)):
            if not a or not b or b['backend']!='cuda_graph':
                raise ValueError('Missing colocated inference CUDA Graph receipt')
            hits = b['graph_replays']-a['graph_replays']
            if phase=='time' and hits<=0:
                raise ValueError(f'No CUDA Graph replays on measured policy replica {rank}')
            fallbacks = {k:v-a['fallback_counts'].get(k,0) for k,v in b['fallback_counts'].items()}
            graph.append(dict(rank=rank,graph_replays=hits,fallback_counts=fallbacks,
                              capture_failures=b['capture_failures']-a['capture_failures']))
        t0,t1 = min(r['timing_start_ns'] for r in results),max(r['timing_end_ns'] for r in results)
        if (not dispatch<=t0<t1<=ack['export_submitted_ns']<=last_ack or
                any(r['timing_end_ns']<=r['timing_start_ns'] for r in results)):
            raise ValueError('Invalid cross-rank optimizer/IPC time bounds')
        optimizer = (t1-t0)/1e9
        ipc = (ack['ack_completed_ns']-ack['export_submitted_ns'])/1e9
        return dict(phase=phase,micro=micro,gas=144//micro,version=version,published_version=step,
                    timing_start_ns=t0,timing_end_ns=t1,optimizer_wall_s=optimizer,ipc_s=ipc,
                    optimizer_plus_ipc_s=optimizer+ipc,dispatch_to_last_ack_s=(last_ack-dispatch)/1e9,
                    publication=ack,rollout_graph=graph,ranks=results)

    try:
        experiment.flush()
        rows = db.execute("select details_json from events where run_id=? and kind='trainer_mps_verified' order by id desc limit 1",
                          (experiment.run_id,)).fetchone()
        if not rows:
            raise ValueError('Missing live trainer MPS verification')
        binding = json.loads(rows[0])
        if binding['trainer_mps_percent']!=90:
            raise ValueError('Trainer client did not receive MPS 90%')
        validate_trainer_mps_bindings(binding['trainers'],binding['policy_inference'],90)
        report['gpu_bindings'] = binding
        report['initial_weights'] = digests()
        ray.get([i.set_training_batch_limit.remote(2) for i in inference],timeout=90)
        worker_refs = [w.run.remote() for w in workers]
        progress('collecting the fixed real P189 cohort')
        deadline = time.monotonic()+5400
        while True:
            check()
            if ray.get(replay.size.remote(),timeout=45)>=plan['minimum_replay_chunks']:
                draw = ray.get(replay.freeze_gpu_trial.remote(576,cid,layout_pool_size=plan['minimum_replay_chunks'],
                               layout_seed=plan['protocol_seed']),timeout=600)
                if draw['ready']:
                    frozen = report['frozen'] = draw
                    break
            if time.monotonic()>=deadline:
                raise TimeoutError('Could not freeze one eligible 576-sample P189 cohort')
            time.sleep(3)
        for version,(phase,micro,pair) in enumerate(shape_schedule()):
            if phase=='time' and not report['parity_passed']:
                raise RuntimeError('Measured shape admission requires a completed parity audit')
            arm = arm_run(version,phase,micro)
            if phase=='audit':
                report['audits'].append(arm)
                if micro==24:
                    a,b = report['audits']
                    report['parity'] = compare_audits(a['ranks'],b['ranks'],report['initial_weights'][0]['elements'])
                    report['parity_passed'] = report['parity']['passed']
                    if not report['parity_passed']:
                        raise RuntimeError('GAE or optimizer-input gradient parity exceeded the declared tolerance')
            elif phase=='time':
                if len(report['pairs'])<pair:
                    report['pairs'].append(dict(pair=pair,arms=[]))
                report['pairs'][-1]['arms'].append(arm)
            else:
                report['setup'].append(arm)
        report['final_weights'] = digests()
        if report['initial_weights']!=report['final_weights']:
            raise RuntimeError('P189 numerical weights changed during the comparison')
        report['summary'] = _final_metrics(report['pairs'],db,experiment.run_id,check)
        report['status'] = report['phase'] = 'completed'
        atomic_json(output/'result.json',report)
        experiment.event('paired_gpu_benchmark_completed',details=report['summary'])
        experiment.flush()
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        # A nonfinite diagnostic must survive allow_nan=False so the operator
        # can inspect the failed gate and the job still releases its cohort.
        def json_safe(value):
            if isinstance(value,float) and not math.isfinite(value):
                return str(value)
            if isinstance(value,dict):
                return {k:json_safe(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):
                return [json_safe(v) for v in value]
            return value
        report = json_safe(report)
        atomic_json(output/'result.json',report)
        raise
    finally:
        atomic_json(output/'progress.json',report)
        try:
            if frozen is not None:
                ray.get(replay.release_gpu_trial.remote(cid),timeout=60)
        finally:
            if worker_refs:
                ray.get(collector.stop_pipeline.remote(),timeout=45)
