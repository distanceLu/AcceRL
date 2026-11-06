"""Validate actual trainer targets before committing any successful Replay uses."""
import math


def dependency_counts(store, sample_ids):
    first = {}
    for sid in sample_ids:
        sample = store.samples[sid]
        key = sample['segment_id']
        first[key] = min(first.get(key, sample['index']), sample['index'])
    return dict(jit_global_selected_chunks=len(sample_ids),
        jit_global_suffix_states=sum(store.segments[k].count-i+int(not store.segments[k].terminated[-1]) for k,i in first.items()),
        jit_full_refresh_states=sum(store.segments[k].count+1 for k in first))


def validate_receipts(records, sample_ids, receipts, version):
    if not isinstance(receipts, list) or len(receipts) != len(sample_ids):
        raise ValueError('JIT needs exactly one target receipt per selected chunk')
    mapped = {}
    for row in receipts:
        if set(row) != {'sample_id', 'advantage', 'value_target', 'refresh_policy_version'}:
            raise ValueError('JIT receipt may not relabel rewards or provenance')
        sid = row['sample_id']
        if sid in mapped or sid not in sample_ids or sid not in records:
            raise ValueError('JIT receipt differs from reserved shard')
        if type(row['refresh_policy_version']) is not int or row['refresh_policy_version'] != version:
            raise ValueError('JIT receipt has a different critic version')
        if any(type(row[k]) not in (int, float) or not math.isfinite(row[k])
               for k in ('advantage', 'value_target')):
            raise ValueError('JIT receipt has nonfinite targets')
        mapped[sid] = row
    if set(mapped) != set(sample_ids):
        raise ValueError('JIT receipt lacks selected samples')
    return mapped
