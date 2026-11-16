"""Preserve the legacy eight-sample masked objective when combining micros.

DeepSpeed scales each backward loss by GAS. Averaging the original group means
inside a 16/24-sample micro, then using GAS 9/6, gives each of the original
18 groups exactly the same weight. No cross-rank reduction or GPU sync is added.
"""


def token_mean_preserving_groups(value, action_mask, base_micro=8):
    if (type(base_micro) is not int or base_micro != 8 or value.ndim < 1
            or value.shape[0] not in (8, 16, 24)):
        raise ValueError('Batch loss requires reviewed 8/16/24 geometry and base groups of eight')
    if action_mask is None:
        return value.mean()
    if value.shape != action_mask.shape:
        raise ValueError('FRESH loss mask must match actual action-token geometry')
    groups = value.shape[0] // base_micro
    grouped_values = value.reshape(groups, -1)
    grouped_mask = action_mask.reshape(groups, -1)
    numerators = (grouped_values * grouped_mask.to(value.dtype)).sum(dim=1)
    denominators = grouped_mask.sum(dim=1).clamp_min(1)
    return (numerators / denominators).mean()
