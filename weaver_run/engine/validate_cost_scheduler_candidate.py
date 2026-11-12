"""One reproducible CPU acceptance run for the isolated cost-scheduler candidate.

Run once after candidate edits finish. This runner never connects to a server,
launches Ray, initializes CUDA, submits a job, or changes a production recipe.
CPU fixtures and AST extraction do not certify real Ray/DDP/ZeRO execution.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import py_compile
import sys
import time
import traceback
import unittest


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NEW_MODULES = (
    'test_cost_scheduler_cpu',
    'test_collective_phase_gate_cpu',
    'test_cycle_service_controller_cpu',
    'test_policy_service_dispatch_cpu',
    'test_cost_scheduler_runtime_cpu',
    'test_scheduler_backend_adapters_cpu',
    'test_scheduler_world_dispatch_cpu',
    'test_scheduler_review_regressions_cpu',
    'test_probe_transaction_cpu',
    'test_critic_refresh_plan_cpu',
    'test_critic_dependency_cpu',
    'test_inventory_refill_cpu',
)
FULL_FRESH_RULES = {
    'dense_reward': True, 'signal_telemetry': True, 'refresh_gae': True,
    'replay_age_limits': True, 'max_policy_age': 16, 'max_world_age': 3,
    'max_sample_uses': 4, 'seed_balancing': True, 'early_seed_context': True,
    'real_policy_replay': True, 'real_fraction': .25,
}


def source_hashes(root=ROOT):
    files = {}
    excluded = {'source_manifest.json', 'candidate_manifest.json'}
    for path in sorted(root.rglob('*')):
        if (path.is_file() and 'validation' not in path.parts and '__pycache__' not in path.parts
                and path.name not in excluded and path.suffix in ('.py', '.sh', '.json', '.md')):
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return files, digest


def modules_to_run():
    # Importing the parent runner does not execute its main or run its suite.
    from validate_fresh_candidate import LEGACY_MODULES
    names = [p.stem for p in sorted(ROOT.glob('test_fresh*_cpu.py'))]
    return list(dict.fromkeys(names + list(LEGACY_MODULES) + list(NEW_MODULES)))


def validate_profiles():
    from fresh_config import FreshConfig
    from fresh_reward_targets import dense_reward_specification
    from world_backends import WorldBackendConfig
    from cost_scheduler_runtime import load_settings

    fresh_profiles = []
    for path in sorted((ROOT / 'examples').glob('fresh_*.json')):
        config = WorldBackendConfig.from_mapping(json.loads(path.read_text('utf8')))
        config.validate_admission()
        fresh = FreshConfig.from_world_config(config)
        fresh.validate_reward_binding(config.backend_options['native_reward_head'])
        fresh_profiles.append({'profile': str(path.resolve()), 'switches': fresh.to_dict()})
    switches = ('dense_reward', 'signal_telemetry', 'refresh_gae', 'replay_age_limits',
                'seed_balancing', 'early_seed_context', 'real_policy_replay')
    for name in switches:
        raw = json.loads((ROOT / 'examples/fresh_00_baseline.json').read_text('utf8'))
        raw['backend_options']['fresh'][name] = True
        if name == 'dense_reward':
            raw['backend_options']['native_reward_head'] = dense_reward_specification()
        WorldBackendConfig.from_mapping(raw).validate_admission()

    full_path = ROOT / 'examples/fresh_04_real.json'
    full_config = WorldBackendConfig.from_mapping(json.loads(full_path.read_text('utf8')))
    full_config.validate_admission()
    full_fresh = FreshConfig.from_world_config(full_config)
    for name, expected in FULL_FRESH_RULES.items():
        actual = getattr(full_fresh, name)
        if actual != expected or type(actual) is not type(expected):
            raise AssertionError(f'Full FRESH learning-rule drift: {name}={actual!r}')
    if full_config.backend_options['native_reward_head'] != dense_reward_specification():
        raise AssertionError('The original dense observed-success target contract changed')

    costs = []
    expected_costs = ['cost_00_disabled.json', 'cost_01_fixed.json',
                      'cost_02_measured.json', 'cost_03_adaptive.json']
    found = sorted(p.name for p in (ROOT / 'examples').glob('cost_*.json'))
    if found != expected_costs:
        raise AssertionError(f'Expected exactly four cost profiles; got {found}')
    for name in expected_costs:
        path = ROOT / 'examples' / name
        loaded = load_settings(str(path))
        costs.append({'profile': str(path.resolve()), 'settings': loaded,
                      'learning_profile': str(full_path.resolve()),
                      'learning_rules': dict(FULL_FRESH_RULES)})
    expected_modes = [(False, False, False), (True, False, False),
                      (True, True, False), (True, True, True)]
    actual_modes = [(row['settings']['enabled'], row['settings']['macro_enabled'],
                     row['settings']['allow_probe']) for row in costs]
    if actual_modes != expected_modes:
        raise AssertionError('Cost profile off/fixed/measured/adaptive staging changed')
    return fresh_profiles, list(switches), costs


def validate_runtime_wiring():
    """Check actual candidate call sites; this is static evidence, not Ray execution."""
    def tree(name):
        return ast.parse((ROOT / name).read_text('utf8'), filename=str(ROOT / name))
    def function(parent, name):
        matches = [node for node in ast.walk(parent)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
        if len(matches) != 1:
            raise AssertionError(f'Expected exactly one {name} method in reviewed scope')
        return matches[0]
    def calls(parent):
        return {ast.unparse(node.func) for node in ast.walk(parent) if isinstance(node, ast.Call)}
    train = tree('train.py')
    main = function(train, 'main')
    classes = {node.name: node for node in train.body if isinstance(node, ast.ClassDef)}
    trainer = classes['TrainerActor']
    infer = classes['InferenceActor']
    replay = classes['ReplayBufferActor']
    bridge = tree('weaver_online_bridge.py')
    world_train = function(function(bridge, 'make_weaver_online_trainer'), 'run_training_epoch')
    world_infer = function(bridge, 'make_weaver_online_inference')
    engine = tree('weaver_online_engine.py')
    policy_train = function(trainer, '_train_policy')
    checks = {
        'main_loads_opt_in_settings': 'load_settings' in calls(main),
        'main_builds_new_source_identity': 'source_fingerprint' in calls(main),
        'main_configures_actual_actor_pools': 'actor.configure_cost_scheduler.remote' in calls(main),
        'policy_optimizer_body_uses_group_adapter': 'scheduled_policy_update' in calls(policy_train),
        'policy_backward_step_remain_in_reviewed_body': {'self.model.backward', 'self.model.step'} <= calls(policy_train),
        'policy_loop_uses_bounded_dispatch': 'self._scheduled_policy_step' in calls(function(infer, '_loop')),
        'policy_refresh_has_async_endpoint': isinstance(function(infer, 'scheduled_evaluate_values'), ast.AsyncFunctionDef),
        'replay_uses_scheduled_current_critic': 'critic.scheduled_evaluate_values.remote' in calls(function(replay, 'sample')),
        'native_world_update_has_begin_and_end_adapter': {'scheduled_world_begin', 'scheduled_world_end'} <= calls(world_train),
        'native_world_loop_still_runs_complete_micro_calls': 'self.worker_client.call' in calls(world_train),
        'native_world_receipt_uses_own_stream': 'self.torch.cuda.current_stream' in calls(function(engine, 'scheduler_boundary')),
        'native_world_generation_is_metered': {'acquire_world_request', 'complete_world_request'} <= calls(function(world_infer, '_loop')),
        'native_context_initialization_is_metered': {'acquire_world_request', 'complete_world_request'} <= calls(function(world_infer, 'init_rollout')),
    }
    if not all(checks.values()):
        raise AssertionError(f'Incomplete runtime wiring: {[key for key, value in checks.items() if not value]}')
    return {'checks': checks, 'scope': 'Static AST call-site evidence; actual Ray/GPU runtime not exercised'}


class RuntimeGuards(importlib.abc.MetaPathFinder):
    """Forbid actual Ray imports and Torch CUDA initialization during this run.

    Test-owned fake modules installed in sys.modules remain usable. They do not
    import a real Ray runtime and cannot serve as evidence of Ray admission.
    """
    def __init__(self, torch):
        self.torch = torch
        self.cuda_attempts = []
        self.ray_attempts = []
        self.original_lazy_init = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ray' or fullname.startswith('ray.'):
            self.ray_attempts.append(fullname)
            raise RuntimeError('Actual Ray import is forbidden in the CPU acceptance runner')
        return None

    def forbid_cuda(self, *args, **kwargs):
        self.cuda_attempts.append('torch.cuda._lazy_init')
        raise RuntimeError('CUDA initialization is forbidden in the CPU acceptance runner')

    def __enter__(self):
        self.original_lazy_init = self.torch.cuda._lazy_init
        self.torch.cuda._lazy_init = self.forbid_cuda
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *args):
        self.torch.cuda._lazy_init = self.original_lazy_init
        if self in sys.meta_path:
            sys.meta_path.remove(self)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-only', action='store_true',
                        help='Print selected module names without executing tests or creating reports')
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    upstream = ROOT.parent / 'analysis/weaver_integration_20261022/upstream'
    if upstream.is_dir():
        os.environ.setdefault('WEAVER_TEST_UPSTREAM', str(upstream))
    modules = modules_to_run()
    if args.list_only:
        print(json.dumps({'candidate': str(ROOT), 'modules': modules}, ensure_ascii=True, indent=2))
        return 0

    started = time.time_ns()
    output = io.StringIO()
    failures = []
    profiles, switches, costs = [], [], []
    wiring = {}
    result = None
    torch = None
    guards = None
    files_before, source_before = source_hashes()
    compiled = 0
    try:
        import torch
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError('Fresh CPU runner already has CUDA initialized')
        if 'ray' in sys.modules:
            raise RuntimeError('CPU runner must start without an actual Ray module loaded')
        with RuntimeGuards(torch) as guards:
            profiles, switches, costs = validate_profiles()
            wiring = validate_runtime_wiring()
            for path in sorted(ROOT.rglob('*.py')):
                if 'validation' not in path.parts and '__pycache__' not in path.parts:
                    py_compile.compile(str(path), doraise=True)
                    compiled += 1
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromNames(modules)
            result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
            if loader.errors:
                failures.extend(loader.errors)
    except BaseException:
        failures.append(traceback.format_exc())
        output.write('\nVALIDATOR EXCEPTION\n' + failures[-1])

    files, source_sha = source_hashes()
    changed = sorted(set(files_before) ^ set(files) |
                     {name for name in files_before.keys() & files.keys() if files_before[name] != files[name]})
    cuda_initialized = None if torch is None else bool(torch.cuda.is_initialized())
    cuda_attempts = [] if guards is None else list(guards.cuda_attempts)
    ray_attempts = [] if guards is None else list(guards.ray_attempts)
    passed = (not failures and result is not None and result.wasSuccessful()
              and not result.skipped and cuda_initialized is False
              and not cuda_attempts and not ray_attempts and not changed)
    directory = ROOT / 'validation'
    directory.mkdir(exist_ok=True)
    report_path = (directory / f'cost_cpu_validation_{started}.json').resolve()
    log_path = (directory / f'cost_cpu_validation_{started}.log').resolve()
    report = {
        'protocol': 'weaver_cost_scheduler_cpu_acceptance_v1',
        'started_ns': started, 'completed_ns': time.time_ns(), 'candidate': str(ROOT),
        'source_sha256': source_sha, 'source_sha256_before': source_before,
        'source_file_sha256': files, 'source_changed_during_validation': changed,
        'python': sys.executable, 'python_version': sys.version,
        'torch_version': None if torch is None else torch.__version__,
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'cuda_initialized': cuda_initialized, 'cuda_initialization_attempts': cuda_attempts,
        'real_ray_import_attempts': ray_attempts,
        'actual_ray_runtime_executed': False,
        'modules': modules, 'compiled_python_files': compiled,
        'fresh_profiles': profiles, 'independent_fresh_switch_admissions': switches,
        'cost_profiles': costs, 'learning_rules': dict(FULL_FRESH_RULES),
        'runtime_wiring': wiring,
        'tests_run': 0 if result is None else result.testsRun,
        'failures': 0 if result is None else len(result.failures),
        'errors': 0 if result is None else len(result.errors),
        'skipped': 0 if result is None else len(result.skipped),
        'validator_errors': failures, 'passed': passed,
        'scope': 'CPU scheduling/retention/fairness, asyncio fake broker, actual-method AST fixtures, tiny CPU native models and serialization; no real Ray/DDP/ZeRO/TP/GPU validation',
        'production_gpu_verified': False, 'actual_ray_ddp_verified': False,
        'training_deployed': False, 'training_submitted': False,
        'production_modified': False, 'log': str(log_path), 'report': str(report_path),
    }
    log_path.write_text(output.getvalue(), encoding='utf8')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (directory / 'cost_cpu_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    manifest = {
        'protocol': report['protocol'], 'candidate': str(ROOT),
        'source_sha256': source_sha, 'files': files, 'validation_passed': passed,
        'validation_report': str(report_path), 'validation_log': str(log_path),
        'tests_run': report['tests_run'], 'cuda_initialized': cuda_initialized,
        'production_gpu_verified': False, 'actual_ray_ddp_verified': False,
        'training_deployed': False, 'training_submitted': False,
    }
    (directory / 'candidate_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({key: report[key] for key in (
        'candidate', 'passed', 'source_sha256', 'tests_run', 'failures', 'errors', 'skipped',
        'cuda_initialized', 'torch_version', 'compiled_python_files', 'report', 'log')},
        ensure_ascii=True, indent=2))
    if failures:
        print(json.dumps({'validator_errors': failures}, ensure_ascii=True))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
