"""Manual online WEAVER training -> final audit -> bounded 200-episode paired evaluation.

The queue reservation has no training deadline. A user boundary stop ends the
training phase. Only then does the separately bounded evaluation phase start.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import training_entry as training

DEFAULT_EVAL_SECONDS = 2 * 3600
FINAL_CLEANUP_SECONDS = 1800


def validate_budgets(args):
    budgets = training.validate_manual_contract(args.post_stop_grace_seconds)
    if getattr(args,'stop_control',None) is not None or getattr(args,'measurement_seconds',None) is not None:
        raise ValueError('JIT deployment is manual-stop only')
    budgets.update(stop_at_ns=None, four_hour_cutoff_enabled=False)
    if (type(args.evaluation_seconds) is not int or args.evaluation_seconds < 60
            or args.evaluation_seconds > DEFAULT_EVAL_SECONDS):
        raise ValueError('Evaluation phase must be independently bounded to 60..7200 seconds')
    budgets.update(evaluation_seconds=args.evaluation_seconds,
                   final_cleanup_reserve_seconds=FINAL_CLEANUP_SECONDS,
                   ordered_phases=['training', 'final_checkpoint_cpu_audit',
                                   'frozen_policy_evaluation', 'cleanup'])
    return budgets


def evaluation_command(args, seconds, *, run_id='weaverintegrated_plan_only', deadline_unix=None):
    command = [sys.executable, str(Path(__file__).resolve().parent.parent / 'parallel_eval/parallel_eval.py'),
               '--source', str(args.source), '--queue-wrapper', str(Path(__file__).resolve().parent),
               '--master', str(args.root), '--run-id', run_id, '--git-commit', args.git_commit,
               '--output-root', str(args.root / 'policy_eval'),
               '--max-seconds', str(seconds),
               '--modes', args.evaluation_modes, '--seed', str(args.evaluation_seed),
               '--repeats', str(args.evaluation_repeats)]
    if deadline_unix is not None:
        command += ['--deadline-unix', str(deadline_unix)]
    if args.baseline_checkpoint is not None:
        command += ['--baseline-checkpoint', str(args.baseline_checkpoint)]
    return command


def evaluation_acceptance(root, exit_code, remaining, *, modes='greedy,stochastic', repeats=1):
    path = root / 'policy_eval/completed.json'
    reasons = []
    if exit_code != 0:
        reasons.append('evaluator_exit_nonzero')
    if remaining:
        reasons.append('evaluation_owned_processes_remain')
    try:
        complete = training.read_json(path)
        if complete.get('complete') is not True:
            reasons.append('paired_full_pool_evaluation_incomplete')
        if (complete.get('errors') != [] or complete.get('cleanup',{}).get('survivors') != []
                or complete.get('cleanup',{}).get('errors') != []):
            reasons.append('parallel_worker_errors_or_cleanup_unconfirmed')
        expected = 2 * 50 * len(modes.split(',')) * repeats
        if (complete.get('initial_state_count') != 50
                or complete.get('modes') != modes.split(',')
                or complete.get('expected_episode_records') != expected
                or complete.get('completed_episode_records') != expected):
            reasons.append('paired_pool_or_episode_counts_mismatch')
        hashes = complete.get('checkpoint_sha256', {})
        if (set(hashes) != {'baseline', 'final'}
                or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
                       for value in hashes.values())):
            reasons.append('frozen_checkpoint_fingerprints_missing')
    except (OSError, ValueError) as exc:
        complete = {'read_error': repr(exc)}
        reasons.append('evaluation_completion_receipt_unavailable')
    for name in ('summary.json', 'episodes.jsonl', 'plan.json'):
        if not (root / 'policy_eval' / name).is_file():
            reasons.append('evaluation_artifact_missing:' + name)
    try:
        parallel_dir = str(Path(__file__).resolve().parent.parent / 'parallel_eval')
        if parallel_dir not in sys.path:
            sys.path.insert(0, parallel_dir)
        import plan_contract
        plan = training.read_json(root / 'policy_eval/plan.json')
        shards = training.read_json(root / 'policy_eval/shards.json')
        plan_contract.validate_shards(plan, shards)
        base = plan_contract.load_base(Path(__file__).resolve().parents[2] / 'engine')
        rows = base.read_rows(root / 'policy_eval/episodes.jsonl', plan)
        by_worker = {i: [r for r in rows if r.get('execution_worker_index') == i] for i in range(4)}
        checked_rows = plan_contract.merge_rows(plan, shards, by_worker, complete.get('run_token'), base)
        reconstructed = base.summarize(plan, checked_rows)
        if (len(checked_rows) != len(rows) or not reconstructed['complete']
                or any(complete.get(key) != reconstructed[key] for key in ('groups', 'paired'))):
            reasons.append('parallel_episode_ledger_coverage_or_summary_mismatch')
        workers = shards.get('workers', [])
        if (plan.get('horizon_action_steps') != 220 or plan.get('task_id') != 0
                or len(workers) != 4 or [w.get('gpu_uuid') for w in workers] != list(training.GPUS)
                or complete.get('shards_sha256') != shards.get('sha256')):
            reasons.append('four_gpu_220_step_protocol_mismatch')
        for index, worker in enumerate(workers):
            result = training.read_json(root / f'policy_eval/workers/worker_{index}/completed.json')
            if (result.get('complete') is not True or result.get('errors') != []
                    or result.get('exit_code') != 0 or result.get('worker_index') != index
                    or result.get('run_token') != complete.get('run_token')
                    or result.get('plan_sha256') != plan.get('sha256')
                    or result.get('shards_sha256') != shards.get('sha256')
                    or result.get('gpu_uuid') != worker.get('gpu_uuid')
                    or result.get('completed_episode_records') != worker.get('expected_episode_records')):
                reasons.append('parallel_worker_result_mismatch:' + str(index))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        reasons.append('parallel_evaluation_evidence_unavailable:' + str(exc))
    return {'passed': not reasons, 'reasons': reasons,
            'completion_receipt': complete, 'exit_code': exit_code,
            'scope': 'frozen checkpoints, paired complete 50-state pool; '
                     'not an assertion of model improvement'}


def run(args):
    budgets = validate_budgets(args)
    if getattr(args, 'native_gpu_preflight', False) is not True:
        raise ValueError('Online WEAVER requires one explicit --native-gpu-preflight after queue admission')
    for relative in ('policy_eval', 'training_queue_receipt.json'):
        if os.path.lexists(args.root / relative):
            raise ValueError('Existing evaluation output cannot be reused: ' + relative)
    if args.validate_only:
        verified = training.validate(args.source, args.root, args.git_commit)
        training.log(validation=verified, budgets=budgets,
                     evaluation_argv=evaluation_command(args, args.evaluation_seconds),
                     native_preflight_argv=training.native_preflight_command(args, 'weaverintegrated_preflight_plan_only', plan_only=True),
                     native_preflight_plan=training.native_preflight_plan(args, 'weaverintegrated_preflight_plan_only'),
                     native_gpu_preflight_executed=False,
                     valid=True, outputs_created=False)
        return 0
    started = time.monotonic()
    training_exit = training.run(args)
    receipt_path = args.root / 'queue_receipt.json'
    receipt = training.read_json(receipt_path)
    training.atomic_json(args.root / 'training_queue_receipt.json', receipt)
    receipt.update(schema=3, ordered_phases=budgets['ordered_phases'],
                   training_acceptance=receipt.get('acceptance'),
                   evaluation_budget_seconds=args.evaluation_seconds,
                   final_cleanup_reserve_seconds=FINAL_CLEANUP_SECONDS)
    if training_exit != 0:
        receipt.update(state='failed', evaluation={'skipped': True,
                       'reason': 'training_or_final_checkpoint_audit_failed'})
        training.atomic_json(receipt_path, receipt)
        return 1

    if getattr(args, 'skip_final_evaluation', False):
        receipt.update(state='succeeded', all_owned_exited=True,
            evaluation={'skipped':True,'reason':'user_requested_no_final_evaluation_for_infra_trials',
                        'expected_episode_records':200,'completed_episode_records':0})
        training.atomic_json(receipt_path, receipt)
        return 0

    # Training_entry has confirmed all owned descendants exited and the final
    # checkpoint's CPU audit passed. Recheck admission before starting CUDA.
    seconds = args.evaluation_seconds
    child = None
    owned = []
    received = []
    failure = None
    remaining = []
    job_id = os.environ['EXPERIMENT_QUEUE_JOB_ID']
    env = training.launch_environment(args.root, args.source, os.environ)
    # Coordinator is CPU-only; workers independently bind the four reserved UUIDs.
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['ACCERL_RUN_ID'] = str(receipt['run_id']) + '_frozen_eval'
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda number, frame: received.append(number))
        # Admission checks inspect the full reservation, using the original
        # four-device environment; each independent worker receives one UUID.
        admission_env = dict(env, CUDA_VISIBLE_DEVICES=','.join(training.GPUS))
        training.wait_idle(args.source, admission_env)
        # Its own budget begins after training cleanup and the idle admission.
        # No wall-clock duration from training is subtracted or converted into
        # a hidden morning deadline.
        evaluation_started_ns = time.time_ns()
        deadline = time.monotonic() + seconds
        command = evaluation_command(args, seconds, run_id=receipt['run_id'],
                                     deadline_unix=evaluation_started_ns / 1e9 + seconds)
        receipt.update(state='evaluating', evaluation={'argv': command,
                       'started_ns': evaluation_started_ns, 'budget_seconds': seconds,
                       'cuda_visible_devices': env['CUDA_VISIBLE_DEVICES'],
                       'worker_gpu_uuids': list(training.GPUS), 'worker_count': 4,
                       'horizon_action_steps': 220,
                       'checkpoint_selection': 'final and explicit baseline or initial snapshot'})
        training.atomic_json(receipt_path, receipt)
        child = subprocess.Popen(command, cwd=args.source, env=env,
                                 stdin=subprocess.DEVNULL)
        identity = training.proc_identity(child.pid)
        if identity is None or identity['uid'] != 1004:
            raise RuntimeError('Cannot identify evaluator PID/UID/birth')
        owned.append(identity)
        receipt['evaluation']['supervisor'] = identity
        training.atomic_json(receipt_path, receipt)
        stop_sent_at = None
        while child.poll() is None:
            now = time.monotonic()
            if (received or now >= deadline) and stop_sent_at is None:
                receipt['evaluation']['stop_cause'] = (
                    'external_signal_' + str(received[0]) if received
                    else 'evaluation_time_budget_elapsed')
                receipt['evaluation']['stop_signals'] = training.signal_verified(
                    owned, signal.SIGTERM)
                stop_sent_at = now
                training.atomic_json(receipt_path, receipt)
            if stop_sent_at is not None and now - stop_sent_at >= 60:
                raise RuntimeError('Evaluator exceeded graceful partial-result save allowance')
            time.sleep(1)
        child.wait(timeout=2)
    except BaseException as exc:
        failure = repr(exc)
    finally:
        remaining = training.marked_children(job_id, owned)
        if remaining:
            receipt['evaluation_emergency_cleanup'] = training.scoped_cleanup(job_id, remaining)
            remaining = receipt['evaluation_emergency_cleanup']['survivors']
        if child is not None:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                failure = failure or 'evaluator still alive after cleanup'
        evaluation = evaluation_acceptance(args.root,
                     child.returncode if child is not None else None, remaining,
                     modes=args.evaluation_modes, repeats=args.evaluation_repeats)
        if failure:
            evaluation['passed'] = False
            evaluation['wrapper_error'] = failure
        receipt['evaluation_acceptance'] = evaluation
        receipt['all_owned_exited'] = not remaining
        receipt.update(finished_ns=time.time_ns(), elapsed_seconds=time.monotonic() - started,
                       state='succeeded' if evaluation['passed'] else 'failed')
        receipt['acceptance'] = {'passed': evaluation['passed'],
                                'training': receipt['training_acceptance'],
                                'evaluation': evaluation}
        training.atomic_json(receipt_path, receipt)
        training.log(final_receipt=str(receipt_path), state=receipt['state'],
                     acceptance=receipt['acceptance'])
    return 0 if receipt['state'] == 'succeeded' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-final-evaluation',action='store_true')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', '--experiment-root', dest='root', type=Path, required=True)
    parser.add_argument('--git-commit', required=True)
    parser.add_argument('--post-stop-grace-seconds', type=int, default=1800)
    parser.add_argument('--native-gpu-preflight', action='store_true')
    parser.add_argument('--evaluation-seconds', type=int, default=DEFAULT_EVAL_SECONDS)
    parser.add_argument('--evaluation-modes', choices=('greedy', 'greedy,stochastic'), default='greedy,stochastic')
    parser.add_argument('--evaluation-seed', type=int, default=20261026)
    parser.add_argument('--evaluation-repeats', type=int, default=1)
    parser.add_argument('--baseline-checkpoint', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.evaluation_repeats < 1:
        parser.error('evaluation repeats must be positive')
    try:
        return run(args)
    except Exception as exc:
        training.log(state='failed_before_launch', error=repr(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
