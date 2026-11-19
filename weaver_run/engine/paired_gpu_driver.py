"""Bounded, frozen-weight A/B with four live real collectors and real IPC publications."""
import json
import math
import os
from pathlib import Path
import time


def summarize(pairs):
    measured = [p for p in pairs if not p['warmup']]
    arms = [{a['name']:a for a in p['arms']} for p in measured]
    baseline = sum(a['pruned']['seconds'] for a in arms)
    jit = sum(a['jit']['seconds'] for a in arms)
    context = [r['context'] for a in arms for r in a['jit']['trainers']]
    calls = sum(r['calls'] for r in context)
    if not calls or not baseline or not measured:
        raise ValueError('Measured pairs and real context calls are required')
    host = sum(r['host_service_sum_s'] for r in context)/calls
    cuda = sum(r['cuda_service_sum_s'] for r in context)/calls
    model = sum(r['cuda_model_sum_s'] for r in context)/calls
    return dict(measured_pairs=len(measured), pruned_mean_s=baseline/len(measured),
                jit_mean_s=jit/len(measured), whole_step_saving_fraction=1-jit/baseline,
                pairs_at_least_ten_percent_faster=sum(a['jit']['seconds'] <= .9*a['pruned']['seconds'] for a in arms),
                total_context_calls=calls, context_host_s_per_call=host,
                context_cuda_service_s_per_call=cuda, context_cuda_model_s_per_call=model,
                host_below_0269=host <= .269, cuda_service_below_0269=cuda <= .269,
                whole_step_at_least_ten_percent_faster=jit <= .9*baseline)


def run_benchmark(*, ray, args, replay, trainers, inference, collector, workers, sync_policy,
                  experiment, stop_requested):
    from frozen_policy_eval import atomic_json

    plan = json.loads(Path(os.environ['ACCERL_JIT_BENCH_PLAN']).read_text(encoding='utf8'))
    output = Path(os.environ['ACCERL_JIT_BENCH_OUTPUT'])
    if (len(workers) != 4 or len(trainers) != 4 or len(inference) != 4 or
            args.num_eval_workers != 0 or not args.batch_autotune):
        raise ValueError('GPU pairing requires four real collectors, four trainers and four inference replicas')
    warmup = plan['warmup_pairs']
    measured = plan['measured_pairs']
    minimum = plan['minimum_replay_chunks']
    if (warmup != 1 or measured != 6 or not 576 <= minimum <= args.replay_capacity):
        raise ValueError('Unsupported paired benchmark plan or Replay capacity')
    pairs = []
    report = dict(status='running', run_id=os.environ['ACCERL_RUN_ID'],
                  git_commit=plan.get('git_commit'), plan=plan, pairs=pairs,
                  checkpoint=str(os.environ['ACCERL_JIT_BENCH_CHECKPOINT']),
                  source='new real cohorts at frozen P189 weights', learning_rate=0,
                  timing='Replay arm dispatch through full optimizer update and four IPC acknowledgements; common cohort capture, configuration and reporting outside timing')

    def progress(phase):
        report['phase'] = phase
        atomic_json(output/'progress.json', report)
        print('[paired GPU] '+phase, flush=True)

    def digests():
        rows = ray.get([t.paired_gpu_weight_digest.remote() for t in trainers], timeout=180)
        hashes = {r['sha256'] for r in rows}
        if len(hashes) != 1 or sorted(r['rank'] for r in rows) != list(range(4)):
            raise RuntimeError('Four policy ranks disagree on fixed model weights')
        return rows

    ray.get([i.set_training_batch_limit.remote(2) for i in inference], timeout=90)
    report['initial_weights'] = digests()
    worker_refs = [w.run.remote() for w in workers]

    def check():
        if stop_requested():
            raise InterruptedError('Benchmark stop requested')
        ready,_ = ray.wait(worker_refs, timeout=0, num_returns=1)
        if ready:
            ray.get(ready[0])
            raise RuntimeError('A real rollout worker stopped during the GPU benchmark')

    def measure(arm, version, cohort):
        check()
        ray.get([i.configure_batch_autotune.remote(version+1,2) for i in inference], timeout=45)
        # Remove action latencies from time spent freezing/writing the cohort.
        ray.get([i.get_batch_autotune_stats.remote(version+1) for i in inference], timeout=45)
        wall_start_ns = time.time_ns()
        start = time.perf_counter()
        shards = replay.gpu_trial_batch.options(num_returns=4).remote(cohort,arm)
        ranks = [t.run_paired_gpu_arm.remote(arm,shard) for t,shard in zip(trainers,shards)]
        results = ray.get(ranks, timeout=1200)
        before_publish = time.perf_counter()
        sync_policy()
        end = time.perf_counter()
        wall_end_ns = time.time_ns()
        if (sorted(r['rank'] for r in results) != list(range(4)) or
                any(r['optimizer_delta'] != 1 or r['skipped_delta'] != 0 or
                    r['version'] != version or r['version_after'] != version+1 or
                    r['train_micro'] != 16 or r['gas'] != 9 or not r['actor_loss_enabled']
                    for r in results)):
            raise RuntimeError('An arm did not execute a complete four-rank optimizer update')
        stats = ray.get([i.get_batch_autotune_stats.remote(version+1) for i in inference], timeout=45)
        collectors = []
        for entry in stats:
            latencies = sorted(entry['action_latencies'])
            collectors.append(dict(rank=entry['rank'], real_action_requests=len(latencies),
                p95_s=latencies[max(0,math.ceil(.95*len(latencies))-1)] if latencies else None,
                refreshed=entry['refresh_observations'], refresh_s=entry['refresh_seconds']))
        if (sorted(x['rank'] for x in collectors)!=list(range(4)) or
                any(x['real_action_requests']==0 for x in collectors)):
            raise RuntimeError('Missing real rollout traffic on a measured replica')
        if any((x['refreshed']>0) != (arm=='pruned') for x in collectors):
            raise RuntimeError('Shared critic refresh traffic disagrees with measured arm')
        check()
        return dict(name=arm, input_version=version, published_version=version+1,
                    wall_start_ns=wall_start_ns, wall_end_ns=wall_end_ns,
                    seconds=end-start, pre_publication_s=before_publish-start, publication_s=end-before_publish,
                    trainers=results, rollout=collectors)

    version = 0
    try:
        deadline = time.monotonic()+5400
        last_report = 0
        while True:
            check()
            resident=ray.get(replay.size.remote(), timeout=45)
            if resident >= minimum:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Only {resident} of {minimum} real chunks collected')
            if time.monotonic() >= last_report+45:
                progress(f'collecting real samples: {resident}/{minimum}')
                last_report=time.monotonic()
            time.sleep(3)
        for pair_id in range(warmup+measured):
            check()
            cid=f'pair-{pair_id:02d}'
            progress('freezing '+cid)
            frozen=ray.get(replay.freeze_gpu_trial.remote(576,cid), timeout=600)
            if not frozen['ready']:
                raise RuntimeError(f'Replay could not freeze {cid}: {frozen}')
            pair=dict(cohort=cid, warmup=pair_id<warmup, frozen=frozen, arms=[])
            pairs.append(pair)
            try:
                names=('pruned','jit') if pair_id%2==0 else ('jit','pruned')
                for name in names:
                    progress(f'{cid} {name} starting')
                    arm=measure(name,version,cid)
                    if any(rank['sample_ids_sha256'] != frozen['sample_ids_sha256']
                           or (not pair['warmup'] and rank['initialization_prepass'])
                           for rank in arm['trainers']):
                        raise RuntimeError('Measured ranks did not consume the frozen steady-state cohort')
                    version+=1
                    pair['arms'].append(arm)
                    progress(f'{cid} {name} complete ({arm["seconds"]:.2f}s)')
            finally:
                ray.get(replay.release_gpu_trial.remote(cid),timeout=60)
        report['final_weights']=digests()
        if report['initial_weights'][0]['sha256']!=report['final_weights'][0]['sha256']:
            raise RuntimeError('Policy weights changed between paired trials')
        report['summary']=summarize(pairs)
        report['phase']='completed'
        report['status']='completed'
        atomic_json(output/'result.json',report)
        experiment.event('paired_gpu_benchmark_completed', details=report['summary'])
        experiment.flush()
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        atomic_json(output/'progress.json',report)
        raise
    finally:
        ray.get(collector.stop_pipeline.remote(),timeout=30)
