"""Admit explicitly synthetic reset-left padding for inference seeds only."""
from __future__ import annotations


def validate_reset_padding(sample, observations, states, actions, intervals, action_mask,
                           *, allowed, training):
    import numpy as np
    padding = sample.get('context_padding')
    if padding is None:
        return None
    if not allowed or training:
        raise ValueError('Reset history padding requires the FRESH inference-only switch')
    count = len(observations)
    if not isinstance(padding, dict):
        raise ValueError('Reset padding requires explicit primitive provenance')
    p = padding.get('count')
    if type(p) is not int or not 0 < p < count:
        raise ValueError('Reset padding must leave at least one genuine observation')
    anchor = count - p - 1
    expected = {'protocol': 'fresh_reset_left_repeat_v1', 'count': p,
                'context_start': -p, 'anchor_frame_index': anchor,
                'real_frame_start': 0, 'real_frame_end_exclusive': anchor + 1,
                'synthetic_history': True}
    if padding != expected or padding.get('synthetic_history') is not True or sample.get('frame_start') != 0:
        raise ValueError('Reset padding provenance differs from actual context geometry')
    indices = [max(i - p, 0) for i in range(count)]
    if sample.get('frame_indices') != indices:
        raise ValueError('Padded context must identify every actual source frame')
    valid = np.asarray(sample.get('valid_mask'))
    wanted_valid = np.arange(count) >= p
    wanted_actions = np.arange(count) > p
    if valid.dtype != np.bool_ or valid.shape != (count,) or not np.array_equal(valid, wanted_valid):
        raise ValueError('Synthetic history must never be labelled as real valid observations')
    if not np.array_equal(action_mask, wanted_actions):
        raise ValueError('Padding and real reset must have masked incoming actions')
    if (np.asarray(actions)[:p + 1] != 0).any() or (np.asarray(intervals)[:p + 1] != 0).any():
        raise ValueError('Synthetic/reset history must contain zero controls and zero intervals')
    if (not np.array_equal(observations[:p], np.repeat(observations[p:p + 1], p, axis=0))
            or not np.array_equal(states[:p], np.repeat(states[p:p + 1], p, axis=0))):
        raise ValueError('Synthetic history may repeat only its measured reset frame/state')
    return indices
