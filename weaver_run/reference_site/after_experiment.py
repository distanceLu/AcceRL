"""CPU-only, exactly-once continuation after training, final evaluation and cleanup."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import tempfile
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path):
    try:
        return json.loads(Path(path).read_bytes())
    except FileNotFoundError:
        return None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write((json.dumps(value, indent=2)+'\n').encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def queue_rows(plan):
    started = time.monotonic()
    with closing(sqlite3.connect(Path(plan['queue_db']).as_uri()+'?mode=ro', uri=True, timeout=3)) as con:
        con.row_factory = sqlite3.Row
        con.execute('pragma query_only=on')
        con.set_progress_handler(lambda: int(time.monotonic()-started > 5), 1000)
        wanted = [plan[key]['submission_id'] for key in ('training', 'target')]
        return [dict(row) for row in con.execute("select * from jobs where submission_id in (?,?) "
            "or state in ('queued','running','blocked')", wanted)]


def same_job(row, expected):
    return (json.loads(row['argv_json']) == expected['argv'] and
        all(row.get(key) == value for key, value in expected.items() if key != 'argv'))


def evaluate(plan, rows, now):
    target = plan['target']
    existing = [r for r in rows if r['submission_id'] == target['submission_id']]
    require(len(existing) <= 1, 'Duplicate target submission identity')
    if existing:
        record = load(Path(target['root'])/'experiment.json')
        require(record is not None and record.get('git_commit') == target['git_commit']
            and record.get('label') == target['label'] and record.get('root') == target['root']
            and record.get('job') == target['job'] and same_job(existing[0], target['job']),
            'Existing continuation job does not match the fixed commit/payload')
        return dict(state='submitted', job_id=existing[0]['id'], queue_state=existing[0]['state'])
    for path, digest in plan.get('protected_files', {}).items():
        require(sha(path) == digest, 'Protected file changed: '+path)
    jobs = {}
    for key in ('training',):
        dep = plan[key]
        found = [r for r in rows if r['submission_id'] == dep['submission_id']]
        require(len(found) == 1, 'Missing or duplicate predecessor: '+key)
        row = jobs[key] = found[0]
        require(row['id'] == dep['job_id'] and same_job(row, {k: dep[k] for k in ('cwd','argv')}),
            'Predecessor identity/payload changed: '+key)
        require(row['state'] in ('queued','running','succeeded'), 'Predecessor failed/blocked: '+key)
        if row['state'] == 'succeeded':
            require(row.get('exitcode') == 0, 'Nonzero predecessor exit: '+key)
    for key in ('training',):
        if jobs[key]['state'] != 'succeeded':
            return dict(state='waiting_'+key)
    training = plan['training']
    receipt = load(Path(training['root'])/'queue_receipt.json')
    if receipt is None:
        return dict(state='waiting_training_receipt')
    require(receipt.get('job_id') == training['job_id'] and receipt.get('run_id') == training['run_id']
        and receipt.get('source_validation', {}).get('git_commit') == training['git_commit'],
        'Training final receipt identity changed')
    require(receipt.get('state') != 'failed', 'Training final receipt failed')
    for key in ('training_acceptance','evaluation_acceptance'):
        require(receipt.get(key, {}).get('passed') is not False, 'Training/final evaluation rejected')
    if not (receipt.get('state') == 'succeeded' and receipt.get('all_owned_exited') is True
            and all(receipt.get(k, {}).get('passed') is True
                    for k in ('training_acceptance','evaluation_acceptance'))):
        return dict(state='waiting_training_receipt')
    require(jobs['training']['started_ns'] <= receipt['finished_ns'] <= jobs['training']['finished_ns'] <= now,
        'Training final receipt time invalid')
    if any(r['state'] in ('queued','running','blocked') for r in rows):
        return dict(state='waiting_idle')
    try:
        state = load(plan['queue_state'])
    except (OSError, json.JSONDecodeError):
        state = None
    if not (isinstance(state, dict) and not state.get('error')
            and type(state.get('observed_ns')) is int and 0 <= now-state['observed_ns'] < 90_000_000_000
            and isinstance(state.get('mps', {}).get('clients'), list)
            and type(state.get('external_process_count')) is int):
        return dict(state='waiting_telemetry')
    return dict(state='waiting_idle' if state['external_process_count'] or state['mps']['clients'] else 'ready')


def run(argv, *, cwd=None, timeout=180):
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('ACCERL_', 'ACCE_', 'WEAVER_', 'CUDA_', 'NSYS_', 'PYTHON')) and k != 'LD_PRELOAD'}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', GIT_TERMINAL_PROMPT='0')
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    require(result.returncode == 0, 'Command failed: '+repr(argv)+'\n'+(result.stderr+result.stdout)[-4000:])
    return result.stdout.strip()


def git(repo, *args):
    return run(['git', '-C', repo, *args])


def prepare(plan):
    target = plan['target']
    repo, runtime, commit = target['checkout'], target['runtime'], target['git_commit']
    require(evaluate(plan, queue_rows(plan), time.time_ns())['state'] == 'ready', 'Dependencies no longer ready')
    for path in (repo, runtime):
        require(Path(git(path, 'rev-parse', '--show-toplevel')).resolve() == Path(path).resolve(), 'Wrong repository root')
        require(not git(path, 'status', '--porcelain=v1', '--untracked-files=normal'), 'Repository not clean: '+path)
    require(git(runtime, 'rev-parse', 'HEAD') == plan['training']['git_commit']
        and not git(runtime, 'branch', '--show-current'), 'Old runtime changed before handoff')
    require(git(repo, 'rev-parse', 'HEAD') == commit, 'Unexpected development HEAD')
    require(git(repo, 'branch', '--show-current') == target['branch'], 'Unexpected development branch')
    ref = 'refs/heads/'+target['branch']
    require(git(repo, 'rev-parse', ref) == commit, 'Maintenance branch tip moved')
    require(git(repo, 'remote', 'get-url', 'origin') in
        ('git@github.com:distanceLu/AcceRL.git', 'https://github.com/distanceLu/AcceRL.git'), 'Unexpected Git origin')
    require(git(repo, 'ls-remote', '--exit-code', '--heads', 'origin', ref).split() == [commit, ref], 'Published branch moved')
    require(sha(target['run_sh']) == target['run_sh_sha256'], 'Server run.sh changed')
    require(evaluate(plan, queue_rows(plan), time.time_ns())['state'] == 'ready', 'Dependencies changed before switch')
    require(git(repo, 'rev-parse', 'HEAD') == commit and not git(repo, 'status', '--porcelain=v1', '--untracked-files=normal'),
        'Restored development branch differs')
    require(sha(target['run_sh']) == target['run_sh_sha256'], 'Server run.sh changed after switch')


def submit_once(plan, previous, save_status):
    require(previous.get('attempted_ns') is None and previous.get('state') not in ('submitting','failed','submitted'),
        'A launch was already attempted; inspect the durable receipt, never retry blindly')
    prepare(plan)
    intent = dict(state='submitting', attempted_ns=time.time_ns())
    save_status(intent)
    target = plan['target']
    raw = run(target['launch_argv'], cwd=target['checkout'], timeout=900)
    result = json.loads(raw)
    require(result.get('id') == target['label'] and result.get('git_commit') == target['git_commit'] and result.get('job'),
        'run.sh returned an unexpected submission')
    return dict(intent, state='submitted', job_id=result['job'], queue_state=result['state'], response=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    path = Path(args.plan).resolve()
    plan = load(path)
    require(plan['schema'] == 1 and (socket.gethostname(), os.getuid()) == (plan['hostname'], plan['uid']), 'Wrong host/owner/schema')
    if args.check_only:
        print(json.dumps(evaluate(plan, queue_rows(plan), time.time_ns())))
        return
    import fcntl
    directory = Path(plan['status_dir'])
    require(directory.is_absolute() and directory.resolve() != Path(plan['target']['root']).resolve(), 'Invalid watcher status directory')
    directory.mkdir(parents=True, exist_ok=True)
    digest = sha(path)
    with (directory/'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = load(directory/'status.json') or {}
        require(not previous or previous.get('plan_sha256') == digest, 'Watcher plan changed')
        def save_status(value):
            nonlocal previous
            previous = dict(value, plan_sha256=digest, observed_ns=time.time_ns(), pid=os.getpid(), cpu_only=True)
            atomic(directory/'status.json', previous)
        try:
            while True:
                require(sha(path) == digest, 'Plan changed while waiting')
                state = evaluate(plan, queue_rows(plan), time.time_ns())
                if state['state'] == 'submitted':
                    save_status(state)
                    return
                require(previous.get('state') not in ('submitting','failed','submitted') and previous.get('attempted_ns') is None,
                    'Previous launch attempt needs inspection; no automatic retry')
                if state['state'] == 'ready':
                    save_status(submit_once(plan, previous, save_status))
                    return
                save_status(state)
                time.sleep(30)
        except BaseException as error:
            save_status(dict(state='failed', error=repr(error), attempted_ns=previous.get('attempted_ns')))
            raise


if __name__ == '__main__':
    main()
