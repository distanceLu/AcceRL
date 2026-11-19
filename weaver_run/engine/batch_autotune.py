"""Finite model-free batch calibration; no Torch, Ray or GPU side effects.

One driver feeds measurements from complete, real optimization updates. Every
candidate is held for a multiple of six updates, so GAS 18/9/6 can switch without
rebasing DeepSpeed's cumulative microstep counter. Memory forecasts are guards,
not a guarantee against CUDA OOM; this controller never retries an update.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from statistics import median


GIB = 1024 ** 3
MIN_IMPROVEMENT = .03
MAX_LATENCY_RATIO = 1.25


def _positive_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


@dataclass(frozen=True)
class BatchAutotuneConfig:
    enabled: bool = False
    reserve_fraction: float = .15
    reserve_bytes: int = 8 * GIB
    safety_factor: float = 1.25
    window_updates: int = 6
    train_candidates: tuple = (8, 16, 24)
    refresh_candidates: tuple = (2, 4, 8)

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('Batch autotune enabled must be bool')
        if not _positive_number(self.reserve_fraction) or self.reserve_fraction >= 1:
            raise ValueError('Memory reserve fraction must be finite and between zero and one')
        if type(self.reserve_bytes) is not int or self.reserve_bytes < 0:
            raise ValueError('Memory reserve bytes must be a nonnegative integer')
        if not _positive_number(self.safety_factor) or self.safety_factor < 1:
            raise ValueError('Activation safety factor must be finite and at least one')
        if type(self.window_updates) is not int or self.window_updates < 6 or self.window_updates % 6:
            raise ValueError('Calibration windows must contain a positive multiple of six updates')
        for field, allowed in (('train_candidates', (8, 16, 24)),
                               ('refresh_candidates', (2, 4, 8))):
            candidates = getattr(self, field)
            if (not isinstance(candidates, (list, tuple)) or not candidates
                    or any(type(n) is not int or n not in allowed for n in candidates)
                    or candidates[0] != allowed[0]
                    or list(candidates) != sorted(set(candidates))):
                raise ValueError(field + ' must be an ordered subset beginning with the baseline')
            object.__setattr__(self, field, tuple(candidates))

    def to_dict(self):
        result = asdict(self)
        result['train_candidates'] = list(self.train_candidates)
        result['refresh_candidates'] = list(self.refresh_candidates)
        return result


def _valid_cards(cards):
    if not isinstance(cards, (list, tuple)) or len(cards) != 4:
        return False
    ranks = []
    for card in cards:
        if not isinstance(card, dict):
            return False
        for name in ('rank', 'total_bytes', 'peak_used_bytes',
                     'train_activation_bytes', 'refresh_activation_bytes'):
            if type(card.get(name)) is not int:
                return False
        if (card['rank'] not in range(4) or card['total_bytes'] <= 0
                or not 0 < card['peak_used_bytes'] <= card['total_bytes']
                or not 0 <= card['train_activation_bytes'] <= card['peak_used_bytes']
                or not 0 <= card['refresh_activation_bytes'] <= card['peak_used_bytes']):
            return False
        ranks.append(card['rank'])
    return len(set(ranks)) == 4


def memory_guard(cards, *, stage, current_micro, candidate_micro, config):
    """Forecast incremental activation growth against the worst measured card.

    ``peak_used_bytes`` is whole-device usage, including other processes. The
    activation increment is process-local and measured separately for each role.
    Zero activation is treated as unavailable measurement, not free capacity.
    """
    if stage not in ('train', 'refresh'):
        raise ValueError('Unknown calibration stage')
    candidates = getattr(config, stage + '_candidates')
    if (type(current_micro) is not int or type(candidate_micro) is not int
            or current_micro not in candidates or candidate_micro not in candidates
            or candidate_micro <= current_micro):
        raise ValueError('Memory expansion guard requires two increasing supported candidates')
    result = dict(allowed=False, reason='missing_or_invalid_memory', predicted_by_rank=[])
    if not _valid_cards(cards):
        return result
    activation_key = stage + '_activation_bytes'
    if any(card[activation_key] <= 0 for card in cards):
        return result
    for card in sorted(cards, key=lambda item: item['rank']):
        reserve = max(math.ceil(config.reserve_fraction * card['total_bytes']), config.reserve_bytes)
        growth = math.ceil(config.safety_factor * card[activation_key]
                           * (candidate_micro / current_micro - 1))
        result['predicted_by_rank'].append(dict(
            rank=card['rank'], predicted_used_bytes=card['peak_used_bytes'] + growth,
            capacity_limit_bytes=max(0, card['total_bytes'] - reserve), reserve_bytes=reserve))
    result['allowed'] = all(item['predicted_used_bytes'] <= item['capacity_limit_bytes']
                            for item in result['predicted_by_rank'])
    result['reason'] = 'within_memory_reserve' if result['allowed'] else 'predicted_memory_limit'
    return result


def _card_peaks(samples):
    if not all(_valid_cards(item.get('cards')) for item in samples):
        return None
    result = []
    for rank in range(4):
        cards = [next(c for c in item['cards'] if c['rank'] == rank) for item in samples]
        if len({card['total_bytes'] for card in cards}) != 1:
            return None
        result.append(dict(rank=rank, total_bytes=cards[0]['total_bytes'],
                           **{key: max(card[key] for card in cards)
                              for key in ('peak_used_bytes', 'train_activation_bytes',
                                          'refresh_activation_bytes')}))
    return result


def _timing_valid(item):
    return (all(_positive_number(item.get(key)) for key in
                ('update_seconds', 'train_seconds', 'refresh_seconds', 'action_p95_seconds'))
            and type(item.get('refresh_observations')) is int
            and item['refresh_observations'] > 0)


class BatchAutotuner:
    """Refresh-first bounded search, then a fixed selection for the rest of a run.

    ``current`` always describes the next update to execute. An observe receipt
    contains both the configuration just measured (``current``) and the next one
    (``next``). ``phase`` is the state after the decision; ``observed_phase`` is
    the state in which the supplied update ran. Receipts are detached JSON data.
    """

    def __init__(self, config, *, fixed_train_micro=None):
        if not isinstance(config, BatchAutotuneConfig):
            raise ValueError('BatchAutotuner requires BatchAutotuneConfig')
        self.config = config
        self.completed_updates = 0
        self.phase = 'refresh' if config.enabled else 'disabled'
        self._current = dict(train_micro=8, refresh_micro=2)
        if fixed_train_micro is not None:
            if type(fixed_train_micro) is not int or fixed_train_micro not in config.train_candidates or not config.enabled:
                raise ValueError('Fixed training batch requires an enabled, supported microbatch')
            self._current['train_micro'] = fixed_train_micro
            self.phase = 'frozen'
        self._window = []
        self._stage_results = []
        self._history = []
        self._baseline_latency = None

    @property
    def current(self):
        return dict(self._current)

    @property
    def frozen(self):
        return self.phase in ('frozen', 'disabled')

    def summary(self):
        return deepcopy(dict(config=self.config.to_dict(), current=self.current,
                             phase=self.phase, frozen=self.frozen,
                             completed_updates=self.completed_updates, windows=self._history))

    def _summarize(self):
        measured = self._window[1:]
        # Timing warmup is excluded, but a missing metric or memory peak during
        # warmup still prevents optimistic expansion into unmeasured headroom.
        timing_valid = all(_timing_valid(item) for item in self._window)
        peaks = _card_peaks(self._window)
        result = dict(phase=self.phase, configuration=self.current,
                      first_step=self._window[0]['policy_step'],
                      last_step=self._window[-1]['policy_step'],
                      updates=len(self._window), measured_updates=len(measured),
                      valid=timing_valid and peaks is not None,
                      cards=peaks, update_seconds=None, train_seconds=None,
                      refresh_seconds=None, refresh_seconds_per_observation=None,
                      refresh_observations=None, action_p95_seconds=None, score=None)
        if timing_valid:
            for key in ('update_seconds', 'train_seconds', 'refresh_seconds',
                        'refresh_observations', 'action_p95_seconds'):
                result[key] = median(item[key] for item in measured)
            result['refresh_seconds_per_observation'] = median(
                item['refresh_seconds'] / item['refresh_observations'] for item in measured)
            result['score'] = result['train_seconds'] if self.phase == 'train' else result['refresh_seconds_per_observation']
        if not self._stage_results:
            self._baseline_latency = result['action_p95_seconds']
        result['latency_limit_seconds'] = (None if self._baseline_latency is None
                                           else self._baseline_latency * MAX_LATENCY_RATIO)
        result['within_reserve'] = bool(peaks is not None and all(
            card['peak_used_bytes'] <= card['total_bytes'] - max(
                math.ceil(self.config.reserve_fraction * card['total_bytes']), self.config.reserve_bytes)
            for card in peaks))
        result['eligible'] = bool(result['valid'] and result['within_reserve'] and self._baseline_latency is not None
                                  and result['action_p95_seconds'] <= result['latency_limit_seconds'])
        return result

    def _select_best(self):
        key = self.phase + '_micro'
        candidates = getattr(self.config, self.phase + '_candidates')
        best_micro, best_score = candidates[0], None
        for result in self._stage_results:
            if result['eligible']:
                if best_score is None or result['score'] <= best_score * (1 - MIN_IMPROVEMENT):
                    best_micro = result['configuration'][key]
                    best_score = result['score']
        return best_micro

    def _finish_stage(self):
        key = self.phase + '_micro'
        chosen = self._select_best()
        self._current[key] = chosen
        self.phase = 'train' if self.phase == 'refresh' else 'frozen'
        self._stage_results = []
        self._baseline_latency = None
        return chosen

    def observe(self, sample):
        if (not isinstance(sample, dict) or type(sample.get('policy_step')) is not int
                or sample['policy_step'] != self.completed_updates + 1):
            raise ValueError('Calibration needs every consecutive completed Policy update starting at one')
        before, observed_phase = self.current, self.phase
        self.completed_updates += 1
        receipt = dict(policy_step=self.completed_updates, current=before,
                       observed_phase=observed_phase, window=None, memory_guard=None)
        if self.frozen:
            reason = 'disabled' if self.phase == 'disabled' else 'selection_frozen'
        else:
            self._window.append(deepcopy(sample))
            reason = 'collecting_window'
            if len(self._window) == self.config.window_updates:
                result = self._summarize()
                self._stage_results.append(result)
                self._history.append(deepcopy(result))
                receipt['window'] = result
                self._window = []
                candidates = getattr(self.config, self.phase + '_candidates')
                key = self.phase + '_micro'
                index = candidates.index(self._current[key])
                guard = None
                if not result['valid']:
                    reason = 'missing_or_invalid_telemetry'
                elif not result['within_reserve']:
                    reason = 'observed_memory_limit'
                elif not result['eligible']:
                    reason = 'action_latency_limit'
                elif index + 1 == len(candidates):
                    reason = 'candidates_complete'
                else:
                    guard = memory_guard(result['cards'], stage=self.phase,
                                         current_micro=candidates[index], candidate_micro=candidates[index + 1],
                                         config=self.config)
                    receipt['memory_guard'] = guard
                    reason = 'try_next_candidate' if guard['allowed'] else guard['reason']
                if guard is not None and guard['allowed']:
                    self._current[key] = candidates[index + 1]
                else:
                    receipt['selected_micro'] = self._finish_stage()
        receipt.update(next=self.current, phase=self.phase, frozen=self.frozen,
                       reason=reason, window_progress=len(self._window))
        return deepcopy(receipt)
