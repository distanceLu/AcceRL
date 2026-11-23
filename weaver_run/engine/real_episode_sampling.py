"""Real-only model-free cohorts. Selection leaves GAE boundaries and uses untouched."""
from collections import Counter


RANKS = 4
RANK_CHUNKS = 144
GLOBAL_CHUNKS = RANKS * RANK_CHUNKS


def _whole_rank(episodes, groups):
    """Find one exact rank in randomized episode order by 0–1 subset sum."""
    reachable = 1  # 第 i 位表示使用已经看过的回合可以恰好装 i 个 chunk。
    mask = (1 << (RANK_CHUNKS + 1)) - 1
    parents = {}
    for episode in episodes:
        length = len(groups[episode])
        added = ((reachable << length) & mask) & ~reachable
        reachable |= added
        while added:
            bit = added & -added
            total = bit.bit_length() - 1
            parents[total] = episode
            added ^= bit
        if RANK_CHUNKS in parents:
            result = []
            total = RANK_CHUNKS
            while total:
                episode = parents[total]
                result.append(episode)
                total -= len(groups[episode])
            return result
    return []


def _whole_pack(store, complete, groups):
    candidates = [episode for episode in complete if len(groups[episode]) <= RANK_CHUNKS]
    if sum(len(groups[episode]) for episode in candidates) < GLOBAL_CHUNKS:
        return None, 'insufficient_complete_chunks_that_fit_a_rank'
    # 不同的合法首卡组合会给末卡留下不同库存。限次重试，避免阻塞已有可用数据。
    for _ in range(4):
        remaining = candidates.copy()
        store.rng.shuffle(remaining)
        packed = []
        for _rank in range(RANKS):
            episodes = _whole_rank(remaining, groups)
            if not episodes:
                break
            for episode in episodes:
                packed.extend(groups[episode])
            used = set(episodes)
            remaining = [episode for episode in remaining if episode not in used]
        if len(packed) == GLOBAL_CHUNKS:
            return packed, None
    return None, 'four_rank_exact_pack_not_found'


def _uniform_block(rng, groups):
    # 任意固定回合排列中，每个合格 chunk 恰好被 576 个环起点覆盖。
    episodes = list(groups)
    rng.shuffle(episodes)
    ring = [sid for episode in episodes for sid in groups[episode]]
    start = rng.randrange(len(ring))
    stop = start + GLOBAL_CHUNKS
    selected = ring[start:stop]
    if stop > len(ring):
        selected.extend(ring[:stop - len(ring)])
    return selected


def choose(store, count, mode):
    from fresh_policy_replay import FreshReplayNotReady

    if mode not in ('uniform_blocks', 'whole_episodes'):
        raise ValueError(f'Unknown real Replay sampling mode: {mode!r}')
    if not store.config.real_only or count != GLOBAL_CHUNKS:
        raise ValueError('Real episode sampling requires a real-only batch of four 144-chunk ranks')

    groups = {}
    complete = []
    for episode, segment in store.segments.items():
        # 按真实时间顺序保留 ID；被版本或次数筛去的位置不会成为新的 GAE 终点。
        ids = []
        for index in range(segment.count):
            sid = segment.sample_id(index)
            if store._eligible(store.samples[sid]):
                ids.append(sid)
        if ids:
            groups[episode] = ids
            if len(ids) == segment.count and (segment.terminated[-1] or segment.truncated[-1]):
                complete.append(episode)
    eligible = sum(map(len, groups.values()))
    if eligible < count:
        raise FreshReplayNotReady('Fresh real cohort unavailable')

    selected = None
    fallback = None
    if mode == 'whole_episodes':
        selected, fallback = _whole_pack(store, complete, groups)
    if selected is None:
        selected = _uniform_block(store.rng, groups)

    visits = Counter()
    counts = Counter()
    for start in range(0, count, RANK_CHUNKS):
        shard = Counter(store.samples[sid]['segment_id'] for sid in selected[start:start + RANK_CHUNKS])
        visits.update(shard.keys())
        counts.update(shard)
    details = dict(requested=mode, actual='uniform_blocks' if fallback else mode,
                   fallback_reason=fallback, eligible_chunks=eligible, eligible_segments=len(groups),
                   complete_eligible_chunks=sum(len(groups[episode]) for episode in complete),
                   complete_eligible_segments=len(complete), selected_segments=len(counts),
                   complete_segments_selected=sum(counts[episode] == store.segments[episode].count
                                                  for episode in complete),
                   rank_split_segments=sum(value > 1 for value in visits.values()))
    return selected, details
