"""One live P189 learner, balanced three-condition crossover, and real Replay uses."""
import hashlib
import json
import math
import os
import random
import time


CONDITIONS = ('normal', 'rollout_yields', 'trainer_yields')
ORDER_SEED = 20261123
DEMAND_WINDOW_NS = 750_000_000
GLOBAL_BATCH = 576


def block_order(index):
    # 固定随机起点，三组轮转，使每个条件在每个顺序位置出现一次。
    order = list(CONDITIONS)
    random.Random(ORDER_SEED).shuffle(order)
    return order[index:] + order[:index]


def recent_demand(rows, now_ns):
    return any(row['last_request_ns'] > 0 and
               0 <= now_ns - row['last_request_ns'] <= DEMAND_WINDOW_NS for row in rows)


def replay_snapshot(owner):
    store = owner.fresh_store
    if not store.config.real_only or owner.real_sampling != 'uniform_blocks':
        raise RuntimeError('Admission pulses require the live real-only uniform-block replay')
    version = store.versions[0]
    eligible = projected = 0
    for sample in store.samples.values():
        if sample['uses'] + sample['reserved'] >= store.config.max_sample_uses:
            continue
        segment = store.segments[sample['segment_id']]
        age = version - segment.versions[sample['index']]['policy']
        eligible += int(0 <= age <= store.config.max_policy_age)
        projected += int(0 <= age + 1 <= store.config.max_policy_age)
    return dict(observed_ns=time.time_ns(), policy_version=version, resident=len(store.samples),
                admitted_real_chunks=owner.fresh_admitted_chunks['real'], eligible_now=eligible,
                eligible_after_next_publication=projected)


async def fixed_weight_epoch(owner, step):
    if (not os.environ.get('ACCERL_JIT_BENCH_PLAN') or not owner._trajectory_jit or
            not owner.cpu_replay_prefetch or os.environ.get('ACCERL_REAL_SAMPLING') != 'uniform_blocks'):
        raise ValueError('Frozen pulse epoch requires the isolated trajectory-JIT benchmark')
    if step != owner.global_step + 1 or owner.model.global_steps != owner.global_step:
        raise RuntimeError('Pulse benchmark and learner policy clocks diverged')
    owner.value_lr = owner.policy_lr = 0.
    owner.policy_train_start_step = 0
    skipped = int(owner.model.skipped_steps)
    start = time.time_ns()
    # 经过生产训练路径：真实抽批、四卡优化器、成功使用回执全部保留。
    result = await owner.run_training_epoch(step)
    end = time.time_ns()
    meta = owner.capture_policy_metadata
    ids = meta['sample_ids']
    if (int(owner.model.global_steps) != step or owner.model.skipped_steps != skipped
            or result[6] != step or result[5] != {'policy': 0., 'value': 0.}
            or not math.isfinite(float(result[0])) or meta['fresh_versions'][0] != step - 1
            or meta['real_sampling'] != 'uniform_blocks' or len(ids) != 144
            or {row['sample_id'] for row in owner._jit_receipts} != set(ids)):
        raise RuntimeError('Optimizer, versions or charged trajectory JIT cohort failed verification')
    return dict(rank=owner.rank, version=step, started_ns=start, finished_ns=end,
                sample_ids=ids, reservation=meta['fresh_reservation_token'])


def run_pulse_benchmark(*, ray, plan, output, replay, trainers, inference, collector,
                        workers, sync_policy, experiment, stop_requested):
    from bench_action_gate import write_rollout_pulse
    from frozen_policy_eval import atomic_json

    if ((plan['warmup_updates'], plan['measured_blocks'], plan['updates_per_arm'],
         plan['pulse_seconds'], plan['real_sampling']) != (1, 3, 3, 5., 'uniform_blocks')
            or not GLOBAL_BATCH <= plan['minimum_replay_chunks']
            or os.environ.get('ACCERL_TRAJECTORY_JIT') != '1'
            or os.environ.get('ACCERL_REAL_SAMPLING') != 'uniform_blocks'):
        raise ValueError('Invalid P189 continuous-update crossover plan')
    report = dict(status='running', run_id=os.environ['ACCERL_RUN_ID'], plan=plan,
        git_commit=plan['git_commit'], checkpoint=plan['checkpoint'], learning_rate=0,
        order_seed=ORDER_SEED, demand_window_ns=DEMAND_WINDOW_NS, blocks=[],
        protocol='One live replay and optimizer/EMA history. One warmup then three blocks of three arms. '
                 'Real successful uses and IPC publications advance logical versions at fixed numerical P189 weights. '
                 'Intent is after common readiness, before any pulse, reservation, training or publication.')

    def progress(phase):
        report['phase'] = phase
        atomic_json(output/'progress.json', report)
        print('[admission pulse] '+phase, flush=True)

    def digests():
        rows = ray.get([trainer.paired_gpu_weight_digest.remote() for trainer in trainers], timeout=180)
        if sorted(row['rank'] for row in rows) != list(range(4)) or len({r['sha256'] for r in rows}) != 1:
            raise RuntimeError('Frozen P189 replicas disagree')
        return rows

    def check_workers():
        if stop_requested():
            raise InterruptedError('Admission pulse benchmark stop requested')
        ready, _ = ray.wait(worker_refs, num_returns=1, timeout=0)
        if ready:
            ray.get(ready[0])
            raise RuntimeError('A real rollout worker exited during the admission pulse benchmark')

    def wait_for_arm_ready(minimum):
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            check_workers()
            snap, *rows = ray.get([replay.pulse_inventory.remote(),
                                  *[i.pulse_action_demand.remote() for i in inference]], timeout=30)
            if snap['eligible_now'] >= minimum and recent_demand(rows, time.time_ns()):
                return dict(inventory=snap, demand=rows)
            time.sleep(.2)
        raise TimeoutError('No concurrent trainable inventory and recent real action demand')

    def update(version):
        step = version + 1
        deadline = time.monotonic() + 300
        while True:
            check_workers()
            admission = ray.get(replay.admit_model_free_batch.remote(GLOBAL_BATCH), timeout=30)
            if admission['ready']:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Live Replay could not form the next legal pulse batch')
            time.sleep(.2)
        admitted_ns = time.time_ns()
        if admission['versions'][0] != version:
            raise RuntimeError('Pulse admission belongs to a different policy')
        ray.get([replay.configure_batch_autotune.remote(step, 2),
                 *[t.configure_batch_autotune.remote(step, 16) for t in trainers],
                 *[i.configure_batch_autotune.remote(step, 2) for i in inference]], timeout=30)
        experiment.event('model_free_update_admitted', ts_ns=admitted_ns,
                         details=dict(admission, policy_step=step))
        dispatch_ns = time.time_ns()
        ranks = ray.get([t.run_admission_pulse_epoch.remote(step) for t in trainers], timeout=600)
        ranks.sort(key=lambda row:row['rank'])
        ids = [sid for row in ranks for sid in row['sample_ids']]
        if (len(set(ids)) != GLOBAL_BATCH or [row['rank'] for row in ranks] != list(range(4))
                or any(row['version'] != step or row['reservation'] != admission['reservation_token']
                       for row in ranks)):
            raise RuntimeError('Four ranks did not consume a complete, disjoint, admitted live cohort')
        publish_started_ns = time.time_ns()
        sync_policy()  # waits for four IPC acknowledgements and replay version advancement
        published_ns = time.time_ns()
        return dict(input_version=version, published_version=step, admitted_ns=admitted_ns,
                    eligible_at_admission=admission['eligible']['real'], dispatch_ns=dispatch_ns,
                    rank_started_ns=[r['started_ns'] for r in ranks],
                    rank_finished_ns=[r['finished_ns'] for r in ranks],
                    publish_started_ns=publish_started_ns, published_ns=published_ns,
                    sample_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',',':')).encode()).hexdigest())

    worker_refs = []
    try:
        report['initial_weights'] = digests()
        ray.get([i.set_training_batch_limit.remote(2) for i in inference], timeout=30)
        worker_refs = [w.run.remote() for w in workers]
        progress('collecting initial inventory')
        wait_for_arm_ready(plan['minimum_replay_chunks'])
        report['warmup'] = update(0)
        version = 1
        for index in range(plan['measured_blocks']):
            block = dict(block=index+1, arms=[])
            report['blocks'].append(block)
            for condition in block_order(index):
                arm_id = f"block-{index+1}-{condition}"
                progress('preparing '+arm_id)
                ready = wait_for_arm_ready(GLOBAL_BATCH)
                # 各组共用同一计时起点，包含门控指令、等待、取批和发布。
                intent = time.time_ns()
                arm = dict(arm_id=arm_id, condition=condition, ready=ready,
                           intent_wall_ns=intent, pulse=None, updates=[])
                block['arms'].append(arm)
                if condition == 'rollout_yields':
                    arm['pulse'] = write_rollout_pulse(output, arm_id, plan['pulse_seconds'])
                elif condition == 'trainer_yields':
                    issued_ns = time.time_ns()
                    time.sleep(plan['pulse_seconds'])
                    arm['pulse'] = dict(arm_id=arm_id, issued_ns=issued_ns,
                        until_ns=issued_ns+int(plan['pulse_seconds']*1e9), released_ns=time.time_ns())
                for _ in range(plan['updates_per_arm']):
                    receipt = update(version)
                    arm['updates'].append(receipt)
                    version = receipt['published_version']
                arm['end_ns'] = arm['updates'][-1]['published_ns']
                arm['end_inventory'] = ray.get(replay.pulse_inventory.remote(), timeout=30)
                progress(f'{arm_id} complete through P{version}')
        # 让临近最后 ACK 的在途请求也能留下端到端回执，收尾不计入处理组。
        time.sleep(6.)
        check_workers()
        report['final_weights'] = digests()
        if report['final_weights'][0]['sha256'] != report['initial_weights'][0]['sha256']:
            raise RuntimeError('Numerical P189 policy weights changed in the pulse benchmark')
        report.update(status='completed', phase='completed', completed_updates=version)
        experiment.event('admission_pulse_benchmark_completed', details={'completed_updates': version})
        experiment.flush()
        atomic_json(output/'result.json', report)
        progress('completed')
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        progress('failed')
        raise
    finally:
        ray.get(collector.stop_pipeline.remote(), timeout=30)
