"""Run the bounded local CPU acceptance suite and preserve an exact report."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import py_compile
import time
import unittest


LEGACY_MODULES = (
    'test_pipeline_budget_cpu', 'test_pipeline_integration_cpu',
    'test_initial_state_pipeline_cpu', 'test_episode_frame_store_cpu',
    'test_replay_projection_cpu', 'test_pipeline_producers_cpu',
    'test_weaver_integrated_reward_cpu', 'test_weaver_online_engine_cpu',
    'test_weaver_online_snapshot_cpu', 'test_weaver_online_bridge_cpu',
    'test_online_world_stream_cpu', 'test_online_world_controller_cpu',
    'test_online_controller_acceptance_cpu')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    root = Path(__file__).resolve().parent
    upstream = root.parent / 'analysis/weaver_integration_20261022/upstream'
    if upstream.is_dir(): os.environ.setdefault('WEAVER_TEST_UPSTREAM', str(upstream))
    started = time.time_ns()
    from fresh_config import FreshConfig
    from fresh_reward_targets import dense_reward_specification
    from world_backends import WorldBackendConfig
    profiles = []
    for path in sorted((root / 'examples').glob('fresh_*.json')):
        config = WorldBackendConfig.from_mapping(json.loads(path.read_text()))
        config.validate_admission()
        fresh = FreshConfig.from_world_config(config)
        fresh.validate_reward_binding(config.backend_options['native_reward_head'])
        profiles.append({'profile': path.name, 'switches': fresh.to_dict()})
    switches = ('dense_reward', 'signal_telemetry', 'refresh_gae', 'replay_age_limits',
                'seed_balancing', 'early_seed_context', 'real_policy_replay')
    for name in switches:
        raw = json.loads((root / 'examples/fresh_00_baseline.json').read_text())
        raw['backend_options']['fresh'][name] = True
        if name == 'dense_reward': raw['backend_options']['native_reward_head'] = dense_reward_specification()
        WorldBackendConfig.from_mapping(raw).validate_admission()
    for path in root.rglob('*.py'): py_compile.compile(str(path), doraise=True)
    modules = [p.stem for p in sorted(root.glob('test_fresh*_cpu.py'))] + list(LEGACY_MODULES)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames(modules))
    import torch
    files = {}
    for path in sorted(root.rglob('*')):
        if (path.is_file() and '__pycache__' not in path.parts and 'validation' not in path.parts
                and path.suffix in ('.py', '.sh', '.json', '.md')):
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    directory = root / 'validation'; directory.mkdir(exist_ok=True)
    report_path = directory / ('cpu_validation_%d.json' % started)
    log_path = directory / ('cpu_validation_%d.log' % started)
    report = {'protocol': 'weaver_fresh_cpu_acceptance_v1', 'started_ns': started,
        'completed_ns': time.time_ns(), 'candidate': str(root), 'source_sha256': source_sha,
        'source_file_sha256': files, 'profiles': profiles, 'independent_switch_admissions': list(switches),
        'modules': modules, 'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'skipped': len(result.skipped),
        'cuda_initialized': torch.cuda.is_initialized(), 'torch_version': torch.__version__,
        'passed': result.wasSuccessful() and not result.skipped and not torch.cuda.is_initialized(),
        'scope': 'local CPU helpers, actual-method AST fixtures, tiny native models and CPU serialization',
        'production_gpu_verified': False, 'actual_ray_ddp_verified': False,
        'training_deployed': False, 'training_submitted': False, 'log': str(log_path)}
    log_path.write_text(output.getvalue(), encoding='utf8')
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    (directory / 'cpu_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_file_sha256', 'profiles', 'modules')}, indent=2))
    if not report['passed']:
        print(output.getvalue())
        raise SystemExit(1)


if __name__ == '__main__': main()
