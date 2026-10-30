"""Reproducible targeted validation with explicit dependency/scope receipts."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import unittest

HERE = Path(__file__).resolve().parent
NEW = ['test_episode_frame_store_cpu','test_pipeline_budget_cpu','test_replay_projection_cpu',
       'test_pipeline_integration_cpu','test_pipeline_producers_cpu',
       'test_initial_state_scheduler_cpu', 'test_initial_state_pipeline_cpu',
       'test_frozen_policy_eval_cpu', 'test_checkpoint_evaluation_cpu']
NEW += ['test_frozen_controller_cpu', 'test_frozen_weaver_admission_cpu',
        'test_frozen_weaver_runtime_cpu', 'test_weaver_process_cpu', 'test_weaver_backend_cpu']
LOCAL = NEW + ['test_main_startup_cpu','test_pluggable_main_cpu','test_backend_controller_cpu',
    'test_manual_controller_cpu','test_world_contracts_cpu','test_world_runtime_cpu',
    'test_world_rollout_cpu','test_inspect_world_backend_cpu','test_manual_guard_cpu',
    'test_lineage_trace_cpu','test_policy_backend_audit_cpu','test_world_backend_audit_cpu',
    'test_policy_batch_receipts_cpu','test_policy_joint_batch_mode_cpu','test_observed_cpu_prefetch']
TORCH = NEW + ['test_backend_topology_cpu','test_world_data_training_cpu','test_snapshot_cpu',
               'test_world_rollout_cpu']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=('local','torch'), default='local')
    args = parser.parse_args()
    modules = LOCAL if args.suite == 'local' else TORCH
    started = time.time_ns()
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(
            unittest.TestLoader().loadTestsFromNames(modules))
    syntax_errors = []
    for path in HERE.rglob('*.py'):
        try:
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        except (SyntaxError,UnicodeError) as error:
            syntax_errors.append({'path':str(path.relative_to(HERE)), 'error':str(error)})
    base = json.loads((HERE/'baseline_manifest.json').read_text(encoding='utf-8'))
    original = Path(base['source'])
    mismatches = []
    if args.suite == 'local':
        for name,digest in base['files'].items():
            path = original/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                mismatches.append(name)
    bash = shutil.which('bash')
    if platform.system() == 'Windows':
        bash = 'C:/Program Files/Git/bin/bash.exe'
    check = subprocess.run([bash,'-n',str(HERE/'run.sh')],capture_output=True,text=True,timeout=15)
    success = result.wasSuccessful() and not syntax_errors and not mismatches and check.returncode == 0
    report = {'status':'passed' if success else 'failed', 'suite':args.suite,
        'tests_run':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'failures':len(result.failures),'errors':len(result.errors),
        'skipped':[(test.id(),reason) for test,reason in result.skipped],
        'modules':modules,'python':platform.python_version(),'started_ns':started,'ended_ns':time.time_ns(),
        'syntax_errors':syntax_errors,'bash_syntax_returncode':check.returncode,
        'baseline_files_checked':len(base['files']) if args.suite=='local' else 0,
        'baseline_modified':mismatches,'gpu_tested':False,'real_models_loaded':False,
        'scope':'targeted source/CPU tests; Ray RPC validation is a separate receipt'}
    prefix = 'validation_cpu' if args.suite=='local' else 'validation_torch_cpu'
    (HERE/(prefix+'.log')).write_text(stream.getvalue(),encoding='utf-8')
    (HERE/(prefix+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not success:
        print(stream.getvalue()[-12000:])
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
