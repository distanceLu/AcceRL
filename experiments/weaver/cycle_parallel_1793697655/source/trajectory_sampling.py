"""CPU Replay admission for complete native four-chunk imagined segments.

Real sample selection stays uniform over eligible chunks. Real rank packing may
split a selected segment to preserve exact 36/108 shards; its full GAE suffix
is retained. No production credit, reward, age, or successful-use rule changes.
"""
from collections import defaultdict

UNIT = 4
RANKS = 4
LOCAL_REAL = 36
LOCAL_IMAGINED = 108
GLOBAL_BATCH = 576


def enabled(config):
    return getattr(config, 'policy_sampling', 'chunks') == 'imagined_trajectories'


def eligible_groups(store):
    for segment in store.segments.values():
        if segment.source != 'imagined':
            continue
        if segment.count != UNIT:
            raise ValueError('Trajectory mode requires native four-chunk imagined segments')
        ids = tuple(segment.sample_id(i) for i in range(UNIT))
        if all(store._eligible(store.samples[sid]) for sid in ids):
            yield segment, ids


def choose(store, count):
    from fresh_policy_replay import FreshReplayNotReady
    if count != GLOBAL_BATCH:
        raise ValueError('Trajectory mode requires global576 with four equal rank shards')
    real = [sid for sid, row in store.samples.items()
            if store.segments[row['segment_id']].source == 'real' and store._eligible(row)]
    groups = list(eligible_groups(store))
    if len(real) < LOCAL_REAL*RANKS or len(groups) < LOCAL_IMAGINED*RANKS//UNIT:
        raise FreshReplayNotReady('Complete fresh imagined trajectories/source quotas unavailable')
    # Keep the real draw distribution; only rank placement changes.
    real = store.rng.sample(real, LOCAL_REAL*RANKS)
    chosen = store.rng.sample(groups, LOCAL_IMAGINED*RANKS//UNIT)
    by_segment = defaultdict(list)
    for sid in real:
        by_segment[store.samples[sid]['segment_id']].append(sid)
    shards = [[] for _ in range(RANKS)]
    # Largest groups first: avoid a split when any shard has sufficient space.
    for group in sorted(by_segment.values(), key=len, reverse=True):
        while group:
            rank = min(range(RANKS), key=lambda i: len(shards[i]))
            take = min(len(group), LOCAL_REAL-len(shards[rank]))
            assert take > 0
            shards[rank].extend(group[:take]); group = group[take:]
    selected = []
    per_rank = LOCAL_IMAGINED//UNIT
    for rank, shard in enumerate(shards):
        assert len(shard) == LOCAL_REAL
        # Randomize group order, not group membership; each trajectory is
        # contiguous and can never straddle the 144-chunk rank boundary.
        units = [[sid] for sid in shard]
        units.extend(list(ids) for _, ids in chosen[rank*per_rank:(rank+1)*per_rank])
        store.rng.shuffle(units)
        selected.extend(sid for unit in units for sid in unit)
    return selected


def check_commit(store, sample_ids):
    selected = set(sample_ids)
    for sid in sample_ids:
        segment = store.segments[store.samples[sid]['segment_id']]
        if segment.source == 'imagined' and any(segment.sample_id(i) not in selected for i in range(UNIT)):
            raise ValueError('A successful rank commit must consume the entire imagined trajectory')


def forecast_groups(store, future):
    """Integral trajectory units, AND eligibility and min remaining member uses."""
    from collections import Counter
    groups, eligible = Counter(), 0
    cfg = store.config
    for segment, ids in eligible_groups(store):
        eligible += 1
        mask = tuple(all(0 <= p-v['policy'] <= cfg.max_policy_age and
            0 <= w-v.get('world_reward_bundle_version', v.get('world_bundle_version')) <= cfg.max_world_age
            for v in segment.versions) for p,w in future)
        uses = min(cfg.max_sample_uses-store.samples[sid]['uses']-store.samples[sid]['reserved'] for sid in ids)
        if any(mask):
            groups[(uses, mask)] += 1
    return groups, eligible


def inventory_buckets(store):
    """Conservative group projection for the macro controller, still chunk units.

    Each whole group uses its oldest member versions and tightest use budget.
    Future-born or currently ineligible groups are excluded until a new Replay
    snapshot. These projected use counters describe capacity, not consumption
    history. The authoritative per-ID history remains in Replay/telemetry.
    """
    from collections import Counter
    buckets = Counter()
    for segment, ids in eligible_groups(store):
        limiting = max((store.samples[sid] for sid in ids), key=lambda r:r['uses']+r['reserved'])
        key = (min(v['policy'] for v in segment.versions),
               min(v.get('world_reward_bundle_version', v.get('world_bundle_version')) for v in segment.versions),
               limiting['uses'], limiting['reserved'])
        buckets[key] += UNIT
    return [dict(bucket_id='trajectory:'+str(i), source='imagined', distinct_chunks=count,
        generation_policy_version=p, generation_world_version=w, successful_uses=uses,
        reserved_uses=reserved, sampling_unit_chunks=UNIT)
        for i, ((p,w,uses,reserved),count) in enumerate(buckets.items())]


def validate_rank(observations, rank, world_size):
    """Called inside the existing collective preparation failure vote."""
    if type(rank) is not int or not 0 <= rank < RANKS or world_size != RANKS or len(observations) != 144:
        raise ValueError('Trajectory JIT requires four 144-chunk rank shards')
    sources = [obs['_jit_segment'].get('source') for obs in observations]
    if sources.count('real') != LOCAL_REAL or sources.count('imagined') != LOCAL_IMAGINED:
        raise ValueError('Trajectory rank source quota must be 36 real / 108 imagined')
    if any(obs['_jit_segment'].get('sampling') != 'imagined_trajectories' for obs in observations):
        raise ValueError('Trajectory sampling transport/config mismatch')
