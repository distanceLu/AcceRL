"""Four-rank trajectory placement used by the paired real-rollout GPU benchmark.

The four shards own 144 previously drawn samples each. Costs only depend on
trajectory geometry; no benchmark timings enter placement or ordering.
"""
from collections import defaultdict



def select_by_segment(ids):
    groups = defaultdict(list)
    for pos,sample in enumerate(ids):
        s,idx=sample.rsplit(':chunk',1)
        groups[s].append((pos,int(idx)))
    return groups


def place_segments(ids, segments):
    """Exact selected capacity, then integer no-grad suffix load balancing.

    Pack groups with multiple selections first, leaving the singleton pool to
    fill residual exact capacities. Costs use only per-draw known geometry.
    An equal-size swap preserves each 144 constraint and strictly decreases
    the sum of squared context loads. Return original first-occurrence order
    within each resulting rank; this does not optimize micro-bundle order.
    """
    groups = select_by_segment(ids)
    size = {s:len(rows) for s,rows in groups.items()}
    cost = {s:segments[s][0]+int(not segments[s][1])-min(idx for _,idx in rows)-size[s]
            for s,rows in groups.items()}
    rank_segments = [[] for _ in range(4)]
    selected = [0]*4
    context = [0]*4
    for s in sorted(groups, key=lambda s:(-size[s],-cost[s],groups[s][0][0])):
        eligible = [r for r in range(4) if selected[r]+size[s]<=144]
        if not eligible:
            # P1/P2 have almost no singletons. Find exact two-dimensional bins:
            # 144 selections, plus floor/ceil of one quarter of total context.
            remaining=list(groups)
            rank_segments=[]
            total=sum(cost.values())
            for rank in range(3):
                target=total//4+int(rank<total%4)
                reached={(0,0):None}
                for block in remaining:
                    for prev in tuple(reached):
                        n,c=prev[0]+size[block],prev[1]+cost[block]
                        if n<=144 and c<=target and (n,c) not in reached:
                            reached[n,c]=(block,prev)
                    if (144,target) in reached: break
                key=(144,target)
                if key not in reached:
                    return None
                chosen=[]
                while key!=(0,0):
                    block,key=reached[key]; chosen.append(block)
                packed=set(chosen); rank_segments.append(chosen)
                remaining=[block for block in remaining if block not in packed]
            rank_segments.append(remaining)
            selected=[sum(size[x] for x in rank) for rank in rank_segments]
            context=[sum(cost[x] for x in rank) for rank in rank_segments]
            break
        r = min(eligible,key=lambda r:(context[r],selected[r],r))
        rank_segments[r].append(s)
        selected[r]+=size[s]
        context[r]+=cost[s]
    assert selected==[144]*4
    # A cost gap of at most one is the optimal integral max-minus-min spread.
    while max(context)-min(context)>1:
        best = None
        for a in range(4):
            for b in range(4):
                gap = context[a]-context[b]
                if gap<2: continue
                choices = {}
                for i,y in enumerate(rank_segments[b]):
                    choices.setdefault((size[y],cost[y]),i)
                for i,x in enumerate(rank_segments[a]):
                    for delta in range(1,min(gap,cost[x]+1)):
                        j = choices.get((size[x],cost[x]-delta))
                        if j is None: continue
                        gain=delta*(gap-delta)
                        if best is None or gain>best[0]: best=(gain,a,b,i,j,delta)
        if best is None: break
        _,a,b,i,j,delta=best
        rank_segments[a][i],rank_segments[b][j]=rank_segments[b][j],rank_segments[a][i]
        context[a]-=delta
        context[b]+=delta
    return [sorted(group,key=lambda s:groups[s][0][0]) for group in rank_segments]


def geometry(ids, segments):
    chosen = defaultdict(list)
    for sample in ids:
        sid, index = sample.rsplit(':chunk', 1)
        chosen[sid].append(int(index))
    costs = {}
    for sid, selected in chosen.items():
        length, terminal = segments[sid]
        end = length + int(not terminal)
        costs[sid] = []
        for index in sorted(selected, reverse=True):
            costs[sid].append(end - index - 1)
            end = index
    return chosen, costs


def buckets(order, costs, micro):
    flat = [k for sid in order for k in costs[sid]]
    assert len(flat) == 144 and len(flat) % micro == 0
    return [sum(flat[start:start+micro]) for start in range(0, 144, micro)]


def prefix_order(original, costs, micro):
    """Pick a trajectory whose accumulated context approaches the linear load target."""
    remaining = list(original)
    total = sum(sum(costs[sid]) for sid in original)
    selected = consumed = 0
    output = []
    while remaining:
        best = None
        for pos, sid in enumerate(remaining):
            running = consumed
            boundary_error = 0
            for j, gap in enumerate(costs[sid], selected + 1):
                running += gap
                if j % micro == 0:
                    boundary_error += abs(144*running - j*total)
            end_error = abs(144*running - j*total)
            # An imbalance at a closed bundle can no longer be repaired by the
            # next greedy choice. Pressure there is intentionally stronger.
            score = (end_error + 3*boundary_error, -len(costs[sid]), pos)
            if best is None or score < best:
                best = score
        sid = remaining.pop(best[2])
        selected += len(costs[sid])
        consumed += sum(costs[sid])
        output.append(sid)
    return output


def balance_with_swaps(order, costs, micro):
    """Pairwise swaps of singleton trajectories keep the remaining bundle slots fixed."""
    out = list(order)
    loads = buckets(order, costs, micro)
    # Prefer even contexts because the historical no-grad forward width is two.
    pairs, odd = divmod(sum(loads), 2)
    minimum, extra = divmod(pairs, len(loads))
    targets = [2*(minimum + (i < extra)) + int(odd and i == extra) for i in range(len(loads))]
    error = [a-b for a, b in zip(loads, targets)]
    singletons = []
    start = 0
    for pos, sid in enumerate(order):
        if len(costs[sid]) == 1:
            singletons.append((pos, start // micro))
        start += len(costs[sid])
    while True:
        best = 0
        swap = None
        for k, (i, left) in enumerate(singletons):
            for j, right in singletons[k+1:]:
                if left == right:
                    continue
                delta = costs[out[j]][0] - costs[out[i]][0]
                change = delta*(error[left] - error[right] + delta)
                if change < best:
                    best = change
                    swap = i, j, left, right, delta
        if swap is None:
            break
        i, j, left, right, delta = swap
        out[i], out[j] = out[j], out[i]
        error[left] += delta
        error[right] -= delta
    return out


def cross_card_cost(loads):
    micro_calls = [[(n+1)//2 for n in row] for row in loads]
    return sum(max(column) for column in zip(*micro_calls))


def _capacity_shards(ids):
    """Prioritize whole trajectories when the ideal context split cannot be packed."""
    groups = sorted(select_by_segment(ids).values(), key=len, reverse=True)
    bins = [[] for _ in range(4)]
    failed = set()

    def pack(index, space):
        if index == len(groups):
            return True
        state = (index, tuple(sorted(space)))
        if state in failed or len(failed) >= 20000:
            return False
        tried = set()
        block = groups[index]
        for rank in sorted(range(4), key=space.__getitem__):
            capacity = space[rank]
            if capacity < len(block) or capacity in tried:
                continue
            tried.add(capacity)
            space[rank] -= len(block)
            bins[rank].append(block)
            if pack(index+1, space):
                return True
            bins[rank].pop()
            space[rank] += len(block)
        failed.add(state)
        return False

    if pack(0, [144] * 4):
        return [[pos for block in rank for pos,_ in block] for rank in bins]
    # Preserve the draw even for a single group larger than a rank. Each of
    # the three boundaries can divide at most one trajectory.
    positions = [pos for block in groups for pos,_ in sorted(block, key=lambda row: row[1])]
    return [positions[start:start+144] for start in range(0,576,144)]


def ordered_positions(ids, segments):
    if len(set(ids)) != 576 or len(ids) != 576:
        raise ValueError('Paired GPU trial requires 576 distinct real actions')
    base = place_segments(ids, segments)
    if base is None:
        by_rank = [geometry([ids[pos] for pos in row], segments) for row in _capacity_shards(ids)]
        base = [list(selected) for selected,_ in by_rank]
    else:
        by_rank = [geometry(ids, segments)] * 4
    greedy = [prefix_order(rank, costs, 16) for rank,(_,costs) in zip(base, by_rank)]
    polished = [balance_with_swaps(rank, costs, 16) for rank,(_,costs) in zip(greedy, by_rank)]
    def loads(ranks):
        return [buckets(rank, costs, 16) for rank,(_,costs) in zip(ranks, by_rank)]
    order = min((base, greedy, polished),
                key=lambda ranks: cross_card_cost(loads(ranks)))
    original = {sid: ordinal for ordinal, sid in enumerate(ids)}
    positions = [[original[f'{sid}:chunk{index}'] for sid in rank for index in selected[sid]]
                 for rank,(selected,_) in zip(order, by_rank)]
    if list(map(len, positions)) != [144]*4 or sorted(sum(positions, [])) != list(range(576)):
        raise ValueError('Trajectory placement changed the sampled batch')
    return positions, loads(order)
