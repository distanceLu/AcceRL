"""Execution options and seed lifetime must not alter the learning recipe."""
import copy
from pathlib import Path
import json
import types
import unittest
from unittest.mock import patch

from weaver_optimizations import DEFAULTS, optimization_options
import pipeline_imagination


class OptimizationOptionsTests(unittest.TestCase):
    def config(self):
        return {'reward_provider': 'world_head', 'reward_semantics': 'delta_native_score',
                'terminal_rule': 'horizon_only'}

    def test_legacy_is_off(self):
        options = optimization_options({})
        self.assertEqual(options, DEFAULTS)
        self.assertFalse(options['kv_cache'])
        self.assertEqual(options['decode_mode'], 'all')

    def test_strict_unknown_types_and_bounds(self):
        for options in ({'nfe': 4}, {'kv_cache': 1}, {'feature_cache_max_bytes': True},
                        {'resident_max_sessions': 0}, {'resident_max_bytes': 2**30},
                        {'task_cache_entries': 257}, {'safety_checks': 'none'},
                        {'selected_frame_encoding': True}, {'feature_cache_max_bytes': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                optimization_options({'optimizations': options})

    def test_sparse_only_for_native_horizon(self):
        for key, value in [('reward_provider', 'external_vlm'), ('reward_semantics', 'delta_success'),
                           ('terminal_rule', 'argmax')]:
            config = {**self.config(), key: value, 'optimizations': {'decode_mode': 'last'}}
            with self.subTest(config=config), self.assertRaises(ValueError):
                optimization_options(config)
        self.assertEqual(optimization_options({**self.config(),
            'optimizations': {'decode_mode': 'last'}})['decode_mode'], 'last')

    def test_optimized_example_preserves_nonexecution_recipe(self):
        root = Path(__file__).parent / 'examples'
        original = json.loads((root / 'weaver_integrated_internal.json').read_text())
        optimized = json.loads((root / 'weaver_integrated_optimized.json').read_text())
        options = optimized['backend_options'].pop('optimizations')
        self.assertEqual(optimized, original)
        self.assertEqual(options['decode_mode'], 'last')
        self.assertTrue(options['kv_cache'])
        self.assertTrue(options['selected_frame_encoding'])


class SeedReleaseTests(unittest.TestCase):
    def execute(self, inner, release_error=None):
        calls = []
        def rpc(*args, **kwargs):
            calls.append((args, kwargs))
            if release_error:
                raise release_error
        hooks = types.SimpleNamespace(profile_rpc=rpc)
        worker = types.SimpleNamespace(ctrl_world_infer=object())
        with patch.dict('sys.modules', {'capture_hooks': hooks}), patch.object(
                pipeline_imagination, '_imagine_weaver_seed_owned', inner):
            return pipeline_imagination.imagine_weaver_seed(worker, {}), calls

    def test_releases_latest_generation_on_normal_return(self):
        latest = {'_resident': {'generation': 2}}
        def inner(worker, job, owner):
            owner[0] = {'_resident': {'generation': 0}}
            owner[0] = latest
            return 7
        result, calls = self.execute(inner)
        self.assertEqual(result, 7)
        self.assertEqual(calls[0][0][3], 'release_rollout')
        self.assertIs(calls[0][0][4], latest)

    def test_release_also_runs_on_failure(self):
        calls = []
        def inner(worker, job, owner):
            owner[0] = {'_resident': {'generation': 3}}
            raise RuntimeError('original prediction failure')
        hooks = types.SimpleNamespace(profile_rpc=lambda *a, **k: calls.append(a))
        with patch.dict('sys.modules', {'capture_hooks': hooks}), patch.object(
                pipeline_imagination, '_imagine_weaver_seed_owned', inner):
            with self.assertRaisesRegex(RuntimeError, 'original prediction failure'):
                pipeline_imagination.imagine_weaver_seed(types.SimpleNamespace(ctrl_world_infer=1), {})
        self.assertEqual(len(calls), 1)

    def test_expected_publication_stale_cleanup_is_tolerated(self):
        def inner(worker, job, owner):
            owner[0] = {'_resident': {'generation': 0}}
        self.execute(inner, ValueError('Stale WEAVER rollout/cache version; publication'))

    def test_other_release_failure_is_visible(self):
        def inner(worker, job, owner):
            owner[0] = {'_resident': {'generation': 0}}
        with self.assertRaisesRegex(RuntimeError, 'wire failed'):
            self.execute(inner, RuntimeError('wire failed'))

    def test_legacy_state_has_no_release_rpc(self):
        def inner(worker, job, owner):
            owner[0] = {'latents': 'legacy CPU'}
        _, calls = self.execute(inner)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
