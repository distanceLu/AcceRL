"""CPU-only behavior checks for finite, shared batch calibration."""
from copy import deepcopy
import json
import unittest

from batch_autotune import BatchAutotuneConfig, BatchAutotuner, memory_guard


GIB = 1024 ** 3


def sample(step, *, train=40., refresh=20., observations=1000, latency=.3,
           peak=70, activation=20, refresh_activation=2):
    return dict(policy_step=step, update_seconds=train + refresh + 5,
                train_seconds=train, refresh_seconds=refresh,
                refresh_observations=observations, action_p95_seconds=latency,
                cards=[dict(rank=rank, total_bytes=140 * GIB,
                            peak_used_bytes=peak * GIB,
                            train_activation_bytes=activation * GIB,
                            refresh_activation_bytes=refresh_activation * GIB)
                       for rank in range(4)])


def window(tuner, **kwargs):
    receipt = None
    for _ in range(tuner.config.window_updates):
        receipt = tuner.observe(sample(tuner.completed_updates + 1, **kwargs))
    return receipt


class BatchAutotuneTests(unittest.TestCase):
    def test_default_is_off_and_receipts_are_detached_finite_json(self):
        tuner = BatchAutotuner(BatchAutotuneConfig())
        result = window(tuner)
        self.assertTrue(result['frozen'])
        self.assertEqual(result['phase'], 'disabled')
        self.assertEqual(result['next'], {'train_micro': 8, 'refresh_micro': 2})
        result['next']['train_micro'] = 24
        self.assertEqual(tuner.current['train_micro'], 8)
        json.dumps(result, allow_nan=False)

    def test_refresh_first_search_then_train_and_freeze_best_not_largest(self):
        tuner = BatchAutotuner(BatchAutotuneConfig(enabled=True))
        self.assertEqual(window(tuner, refresh=30)['next']['refresh_micro'], 4)
        self.assertEqual(window(tuner, refresh=20)['next']['refresh_micro'], 8)
        end_refresh = window(tuner, refresh=21)
        self.assertEqual(end_refresh['next'], {'train_micro': 8, 'refresh_micro': 4})
        self.assertEqual(tuner.phase, 'train')
        self.assertEqual(window(tuner, train=40)['next']['train_micro'], 16)
        self.assertEqual(window(tuner, train=30)['next']['train_micro'], 24)
        final = window(tuner, train=29.5)
        self.assertTrue(final['frozen'])
        self.assertEqual(final['next'], {'train_micro': 16, 'refresh_micro': 4})
        for _ in range(12):
            self.assertEqual(tuner.observe(sample(tuner.completed_updates + 1, train=.1))['next'], final['next'])

    def test_only_window_boundaries_change_and_microstep_boundaries_stay_aligned(self):
        tuner = BatchAutotuner(BatchAutotuneConfig(enabled=True))
        microsteps = 0
        for step in range(1, 45):
            before = tuner.current
            gas = 144 // before['train_micro']
            self.assertEqual(microsteps % gas, 0)
            microsteps += gas
            result = tuner.observe(sample(step, train=50 / before['train_micro'],
                                          refresh=50 / before['refresh_micro']))
            if before != result['next']:
                self.assertEqual(step % 6, 0)
                self.assertEqual(microsteps % 18, 0)
                self.assertEqual(microsteps % (144 // result['next']['train_micro']), 0)

    def test_warmup_time_excluded_but_its_memory_peak_guards_expansion(self):
        tuner = BatchAutotuner(BatchAutotuneConfig(enabled=True))
        tuner.observe(sample(1, train=10000, refresh=10000, peak=118))
        for step in range(2, 7):
            result = tuner.observe(sample(step))
        self.assertEqual(result['window']['train_seconds'], 40.)
        self.assertEqual(result['window']['refresh_seconds_per_observation'], .02)
        self.assertFalse(result['memory_guard']['allowed'])
        self.assertEqual(result['next']['refresh_micro'], 2)

    def test_refresh_score_normalizes_observation_count(self):
        tuner = BatchAutotuner(BatchAutotuneConfig(enabled=True))
        window(tuner, refresh=20, observations=1000)
        window(tuner, refresh=30, observations=3000)
        result = window(tuner, refresh=19, observations=1000)
        self.assertEqual(result['next']['refresh_micro'], 4)

    def test_latency_regression_rejects_candidate_and_stops_expanding_it(self):
        tuner = BatchAutotuner(BatchAutotuneConfig(enabled=True))
        window(tuner, refresh=30, latency=.2)
        rejected = window(tuner, refresh=5, latency=.3)
        self.assertEqual(rejected['next']['refresh_micro'], 2)
        self.assertEqual(tuner.phase, 'train')
        self.assertEqual(rejected['window']['eligible'], False)

    def test_measured_reserve_violation_cannot_be_selected_as_fastest(self):
        tuner = BatchAutotuner(BatchAutotuneConfig(enabled=True))
        window(tuner, refresh=30)
        rejected = window(tuner, refresh=5, peak=125)
        self.assertEqual(rejected['next']['refresh_micro'], 2)
        self.assertEqual(rejected['reason'], 'observed_memory_limit')
        self.assertFalse(rejected['window']['eligible'])

    def test_missing_or_invalid_measurements_never_authorize_expansion(self):
        for field, value in [('cards', []), ('action_p95_seconds', None),
                             ('action_p95_seconds', float('nan')),
                             ('train_seconds', float('inf')),
                             ('refresh_observations', 0)]:
            with self.subTest(field=field):
                tuner = BatchAutotuner(BatchAutotuneConfig(enabled=True))
                for step in range(1, 13):
                    item = sample(step)
                    item[field] = value
                    result = tuner.observe(item)
                    self.assertEqual(result['next'], {'train_micro': 8, 'refresh_micro': 2})
                    json.dumps(result, allow_nan=False)
                self.assertTrue(result['frozen'])

    def test_duplicate_or_skipped_update_rejected_without_state_mutation(self):
        tuner = BatchAutotuner(BatchAutotuneConfig(enabled=True))
        tuner.observe(sample(1))
        for bad_step in (1, 3, True):
            with self.assertRaises(ValueError):
                tuner.observe(sample(bad_step))
            self.assertEqual(tuner.completed_updates, 1)

    def test_config_rejects_unsafe_geometry_and_nonfinite_values(self):
        for options in [dict(enabled=1), dict(window_updates=5),
                        dict(train_candidates=(8, 32)), dict(train_candidates=(16, 24)),
                        dict(refresh_candidates=(2, 8, 4)), dict(reserve_fraction=float('nan')),
                        dict(reserve_bytes=-1), dict(safety_factor=.9)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                BatchAutotuneConfig(**options)
        self.assertEqual(BatchAutotuneConfig(enabled=True).to_dict()['train_candidates'], [8, 16, 24])


class MemoryGuardTests(unittest.TestCase):
    def test_worst_card_and_reserve_are_applied_to_expansion(self):
        cfg = BatchAutotuneConfig(enabled=True)
        cards = sample(1)['cards']
        approved = memory_guard(cards, stage='train', current_micro=8, candidate_micro=16, config=cfg)
        self.assertTrue(approved['allowed'])
        self.assertEqual(approved['predicted_by_rank'][0]['predicted_used_bytes'], 95 * GIB)
        cards[3]['peak_used_bytes'] = 100 * GIB
        rejected = memory_guard(cards, stage='train', current_micro=8, candidate_micro=16, config=cfg)
        self.assertFalse(rejected['allowed'])
        self.assertEqual(rejected['predicted_by_rank'][3]['capacity_limit_bytes'], 119 * GIB)

    def test_zero_missing_or_duplicate_rank_activation_does_not_authorize_growth(self):
        for mutate in (lambda c: c[0].update(train_activation_bytes=0),
                       lambda c: c[0].pop('peak_used_bytes'),
                       lambda c: c[0].update(rank=1),
                       lambda c: c[0].update(total_bytes=True)):
            cards = deepcopy(sample(1)['cards'])
            mutate(cards)
            result = memory_guard(cards, stage='train', current_micro=8, candidate_micro=16,
                                  config=BatchAutotuneConfig(enabled=True))
            self.assertFalse(result['allowed'])


if __name__ == '__main__':
    unittest.main()
