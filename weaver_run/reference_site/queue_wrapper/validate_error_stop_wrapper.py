"""Run the portable no-GPU wrapper suite and preserve its exact outcome."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'weaver_integrated_v1'
OUTPUT = HERE.parents[1]


def main():
    os.environ['COST_CANDIDATE_SOURCE'] = str(SOURCE)
    os.environ['FRESH_CANDIDATE_SOURCE'] = str(SOURCE)
    modules = ['test_manual_wrapper', 'test_optimized_wrapper_cpu', 'test_fresh_wrapper_cpu',
               'test_cost_deployment_cpu', 'test_error_stop_wrapper_cpu']
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    stream = io.StringIO(); started = time.monotonic()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (OUTPUT / 'wrapper_cpu_validation.log').write_text(stream.getvalue(), encoding='utf-8')
    value = {'tests_run': result.testsRun, 'failures': len(result.failures),
             'errors': len(result.errors), 'skipped': len(result.skipped),
             'passed': result.wasSuccessful() and not result.skipped,
             'torch_imported': 'torch' in sys.modules, 'gpu_used': False,
             'elapsed_seconds': time.monotonic() - started,
             'training_entry_sha256': hashlib.sha256((HERE / 'training_entry.py').read_bytes()).hexdigest(),
             'new_regression_tests': 16,
             'scope': 'Actual wrapper functions/files and simulated child/clock integration; not real Linux processes or GPU'}
    (OUTPUT / 'wrapper_cpu_validation.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value))
    if not value['passed']: print(stream.getvalue())
    return 0 if value['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
