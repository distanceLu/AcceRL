"""CPU contracts for observed-episode discounted-success regression.

Raw simulator success remains boolean. These separately named derived targets
use completed real outcomes only; a budget-censored suffix is never a failure.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


DENSE_MODE = 'libero_discounted_success_value_v1'
TARGET_PROTOCOL = 'weaver_discounted_success_targets_v1'
COHORT_PROTOCOL = 'weaver_discounted_success_value_cohort_v1'
DEFAULT_GAMMA_PER_ACTION = .99 ** (1. / 8.)


def dense_reward_specification(gamma_per_action=DEFAULT_GAMMA_PER_ACTION):
    if (isinstance(gamma_per_action, bool) or not isinstance(gamma_per_action, (int, float))
            or not math.isfinite(gamma_per_action) or not 0 < gamma_per_action <= 1):
        raise ValueError('Dense reward gamma_per_action must be finite in (0,1]')
    return {'mode': DENSE_MODE, 'link': 'sigmoid', 'loss': 'masked_half_mse',
            'label_source': 'libero_env_step', 'calibrated': False, 'terminal': 'horizon_only',
            'balance': 'none', 'loss_coefficient': 1.0, 'gamma_per_action': float(gamma_per_action),
            'discount_unit': 'action_steps', 'target_semantics': 'discounted_observed_episode_success'}


def validate_dense_specification(specification):
    if not isinstance(specification, Mapping):
        raise ValueError('Dense reward specification must be a mapping')
    expected = dense_reward_specification(specification.get('gamma_per_action'))
    if (dict(specification) != expected or specification.get('calibrated') is not False
            or isinstance(specification.get('loss_coefficient'), bool)):
        raise ValueError('Dense reward requires the explicit observed-episode masked regression contract')
    return expected


def derive_episode_reward_targets(success, terminated, truncated, metadata, specification):
    """Return frame-aligned targets/masks; frame T is the observed success frame.

    For a completed successful episode target[t]=gamma_action**(T-t), including
    the reset observation t=0. Complete failures have target zero. Incomplete or
    real-budget-censored episodes have no RM supervision; WM may still use them.
    """
    spec = validate_dense_specification(specification)
    if not isinstance(metadata, Mapping) or metadata.get('label_source') != 'libero_env_step':
        raise ValueError('Dense targets require genuine real-environment source metadata')
    if type(metadata.get('budget_censored')) is not bool:
        raise ValueError('Dense targets require explicit budget_censored metadata')
    arrays = [np.asarray(value) for value in (success, terminated, truncated)]
    count = len(arrays[0]) if arrays[0].ndim else 0
    if count < 1 or any(value.dtype != np.bool_ or value.shape != (count,) for value in arrays):
        raise ValueError('Dense targets require aligned boolean success/termination/truncation labels')
    success, terminated, truncated = arrays
    if success[0] or any(value[:-1].any() for value in (success, terminated, truncated)):
        raise ValueError('Real episodes must end at their first success/terminal/truncated frame')
    action_steps = count - 1
    if (type(metadata.get('action_steps')) is not int or metadata['action_steps'] != action_steps):
        raise ValueError('Dense target action_steps must match post-action frame alignment')
    reason = metadata.get('termination_reason')
    censored = metadata['budget_censored']
    ended = bool(terminated[-1] or truncated[-1])
    targets = np.zeros(count, dtype=np.float32)
    mask = np.zeros(count, dtype=bool)
    if success[-1]:
        if censored or not ended or reason != 'success':
            raise ValueError('Observed success needs a completed, non-censored success episode')
        outcome = 'success'
        targets[:] = np.power(spec['gamma_per_action'], np.arange(action_steps, -1, -1, dtype=np.float64))
        mask[:] = True
    elif censored:
        if reason != 'real_env_budget' or not truncated[-1] or terminated[-1]:
            raise ValueError('Budget-censored targets require the actual real_env_budget truncation')
        outcome = 'censored'
    elif not ended:
        if reason not in (None, 'incomplete', 'cancelled', 'producer_failure', 'user_requested_stop'):
            raise ValueError('Incomplete episode cannot declare an observed completed outcome')
        outcome = 'censored'
    else:
        if reason == 'task_horizon':
            if not truncated[-1]:
                raise ValueError('Observed task-horizon failure requires actual truncation')
        elif reason not in ('environment_terminal', 'env_terminated', 'terminated', 'failure') or not terminated[-1]:
            raise ValueError('Unknown or budget-only truncation cannot be labelled a failure')
        outcome = 'failure'
        mask[:] = True
    descriptor = {'protocol': TARGET_PROTOCOL, 'mode': DENSE_MODE,
        'gamma_per_action': spec['gamma_per_action'], 'discount_unit': 'action_steps',
        'label_source': 'libero_env_step', 'source': 'observed_episode_outcome',
        'outcome': outcome, 'terminal_frame_index': action_steps if ended else None,
        'action_steps': action_steps}
    return {'reward_targets': targets, 'reward_target_mask': mask, 'reward_target_metadata': descriptor}


def reward_target_payload(episode, start, end, specification):
    if any(type(v) is not int for v in (start, end)) or not 0 <= start < end <= len(episode.success):
        raise ValueError('Dense reward window must remain inside one observed episode')
    result = derive_episode_reward_targets(episode.success, episode.terminated,
                                           episode.truncated, episode.metadata, specification)
    descriptor = dict(result['reward_target_metadata'], episode_id=str(episode.episode_id),
                      frame_start=start, frame_end_exclusive=end)
    return {'reward_targets': result['reward_targets'][start:end].copy(),
            'reward_target_mask': result['reward_target_mask'][start:end].copy(),
            'reward_target_metadata': descriptor}


def validate_reward_target_payload(payload, specification, *, expected_count=None):
    """Validate derived payload values against their explicit full-episode index.

    A nonzero target cannot be smuggled in by a changed mask or reindexed window.
    The full-episode success evidence was checked at the episode owner; the
    native process checks this bounded descriptor and exact analytic target.
    """
    spec = validate_dense_specification(specification)
    descriptor = payload.get('reward_target_metadata')
    if not isinstance(descriptor, Mapping):
        raise ValueError('Dense reward payload lacks its derived target provenance')
    required = {'protocol': TARGET_PROTOCOL, 'mode': DENSE_MODE,
        'gamma_per_action': spec['gamma_per_action'], 'discount_unit': 'action_steps',
        'label_source': 'libero_env_step', 'source': 'observed_episode_outcome'}
    if any(descriptor.get(key) != value for key, value in required.items()):
        raise ValueError('Dense reward payload specification/provenance mismatch')
    for name in ('action_steps', 'frame_start', 'frame_end_exclusive'):
        if type(descriptor.get(name)) is not int:
            raise ValueError('Dense reward provenance frame identities must be actual integers')
    start, end, horizon = (descriptor[name] for name in ('frame_start','frame_end_exclusive','action_steps'))
    if not 0 <= start < end <= horizon + 1:
        raise ValueError('Dense reward provenance has invalid absolute frame identities')
    if descriptor.get('episode_id') != payload.get('episode_id'):
        raise ValueError('Dense target and measured window episode identities disagree')
    targets, mask = np.asarray(payload.get('reward_targets')), np.asarray(payload.get('reward_target_mask'))
    count = end - start
    if (expected_count is not None and count != expected_count) or targets.dtype != np.float32 or targets.shape != (count,):
        raise ValueError('Dense targets require exact float32 frame alignment')
    if mask.dtype != np.bool_ or mask.shape != (count,) or not np.isfinite(targets).all() or ((targets < 0)|(targets > 1)).any():
        raise ValueError('Dense target mask or finite bounded target values are invalid')
    outcome = descriptor.get('outcome')
    expected = np.zeros(count, dtype=np.float32)
    if outcome == 'success':
        if descriptor.get('terminal_frame_index') != horizon:
            raise ValueError('Dense success target terminal identity is invalid')
        expected[:] = np.power(spec['gamma_per_action'], horizon - np.arange(start, end, dtype=np.int64))
    elif outcome not in ('failure', 'censored'):
        raise ValueError('Dense reward payload outcome is invalid')
    if not np.array_equal(targets, expected) or not np.all(mask == (outcome != 'censored')):
        raise ValueError('Dense reward target differs from its observed-outcome formula or censor mask')
    if 'success' in payload:
        observed=np.asarray(payload['success'])
        expected_success=np.zeros(count,dtype=bool)
        if outcome=='success' and end==horizon+1:
            expected_success[-1]=True
        if observed.dtype != np.bool_ or observed.shape!=(count,) or not np.array_equal(observed,expected_success):
            raise ValueError('Dense target outcome contradicts measured success labels in this window')
    return targets, mask, dict(descriptor)


def dense_reward_cohort(records, specification, frame_slice=(30,40)):
    spec = validate_dense_specification(specification)
    start, end = frame_slice
    if (not records or any(type(v) is not int for v in (start,end)) or not 0 <= start < end):
        raise ValueError('Dense reward requires a nonempty aligned global cohort')
    valid = masked = success_count = failure_count = 0
    values = []
    for sample in records:
        if sample.length < end:
            raise ValueError('Dense reward selected frame slice exceeds its real window')
        payload = reward_target_payload(sample.episode, sample.start + start, sample.start + end, spec)
        targets, mask = payload['reward_targets'], payload['reward_target_mask']
        n = int(np.count_nonzero(mask)); valid += n; masked += len(mask)-n
        success_count += n if payload['reward_target_metadata']['outcome'] == 'success' else 0
        failure_count += n if payload['reward_target_metadata']['outcome'] == 'failure' else 0
        values.extend(targets[mask].astype(np.float64).tolist())
    return {'protocol': COHORT_PROTOCOL, 'mode': DENSE_MODE,
        'gamma_per_action': spec['gamma_per_action'], 'discount_unit': 'action_steps',
        'label_source':'libero_env_step','window_count':len(records),'frame_slice':[start,end],
        'label_frames_per_window':end-start,'global_selected_frames':len(records)*(end-start),
        'valid_count':valid,'masked_count':masked,'target_sum':math.fsum(values),
        'target_square_sum':math.fsum(v*v for v in values),
        'success_episode_valid_count':success_count,'failure_episode_valid_count':failure_count,
        'rm_active':valid>0,'normalization':'global_valid_frames','balance':'none'}


def validate_dense_reward_cohort(value, specification=None):
    if not isinstance(value, Mapping) or value.get('protocol') != COHORT_PROTOCOL:
        raise ValueError('Dense reward cohort has an unknown protocol')
    spec = validate_dense_specification(specification or dense_reward_specification(value.get('gamma_per_action')))
    for key in ('window_count','label_frames_per_window','global_selected_frames','valid_count','masked_count',
                'success_episode_valid_count','failure_episode_valid_count'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError('Dense reward cohort counts must be nonnegative integers')
    selected, valid = value['global_selected_frames'], value['valid_count']
    frame_slice = value.get('frame_slice')
    if (not isinstance(frame_slice,(list,tuple)) or len(frame_slice)!=2 or
            any(type(v) is not int for v in frame_slice) or not 0 <= frame_slice[0] < frame_slice[1]):
        raise ValueError('Dense reward cohort frame slice is invalid')
    if (selected < 1 or value['window_count']*(frame_slice[1]-frame_slice[0]) != selected
            or value['label_frames_per_window'] != frame_slice[1]-frame_slice[0]
            or valid+value['masked_count'] != selected
            or value['success_episode_valid_count']+value['failure_episode_valid_count'] != valid):
        raise ValueError('Dense reward cohort counts disagree with the fixed selected window geometry')
    for key in ('target_sum','target_square_sum'):
        number=value.get(key)
        if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) or not 0 <= number <= valid:
            raise ValueError('Dense reward cohort target moments must be finite and bounded')
    if value['target_square_sum'] > value['target_sum'] + 1e-10:
        raise ValueError('Dense reward squared target sum exceeds target sum in [0,1]')
    if any(value.get(key) != expected for key,expected in {
        'mode':DENSE_MODE,'gamma_per_action':spec['gamma_per_action'],'discount_unit':'action_steps',
        'label_source':'libero_env_step','normalization':'global_valid_frames','balance':'none'}.items()):
        raise ValueError('Dense reward cohort specification differs from the explicit target contract')
    if type(value.get('rm_active')) is not bool or value['rm_active'] != (valid>0):
        raise ValueError('Dense reward active flag must depend on actual valid coverage')
    return dict(value)
