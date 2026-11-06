"""Run-bound calendar stop policy; reloadable CPU control, no GPU work."""
import hashlib
import json
import time


def load(path, *, submission_id, job_id=None, run_id=None):
    if path.is_symlink() or path.resolve() != path:
        raise ValueError('Stop control path must be canonical')
    raw = path.read_bytes()
    d = json.loads(raw)
    if (d.get('schema') != 1 or d.get('submission_id') != submission_id
            or type(d.get('revision')) is not int or d['revision'] < 1
            or type(d.get('stop_at_ns')) is not int or type(d.get('authorized_ns')) is not int
            or not 0 < d['stop_at_ns']-d['authorized_ns'] <= 48*3600*10**9
            or d.get('timezone') != 'Asia/Shanghai'
            or not d.get('authorization')):
        raise ValueError('Invalid explicit calendar stop authorization')
    for field, actual in [('job_id',job_id),('run_id',run_id)]:
        if d.get(field) is not None and actual is not None and d[field] != actual:
            raise ValueError('Stop control '+field+' mismatch')
    return d, hashlib.sha256(raw).hexdigest()


def due(receipt, args, *, epochs, monotonic_now, wall_ns=None, job_id=None):
    d, digest = load(args.stop_control,submission_id=args.stop_submission,
                     job_id=job_id,run_id=receipt.get('run_id'))
    old = receipt.get('calendar_stop')
    if old is not None and (d['revision'] < old['revision']
            or (d['revision'] == old['revision'] and digest != old['sha256'])):
        raise ValueError('Stop control revision must be monotonic and immutable')
    history = receipt.setdefault('calendar_stop_history',[])
    if old is None or d['revision'] != old['revision']:
        history.append(dict(revision=d['revision'],stop_at_ns=d['stop_at_ns'],sha256=digest))
        if len(history)>32:raise ValueError('Excessive unreviewed deadline revisions')
    receipt['calendar_stop'] = dict(d,sha256=digest)
    measurement=receipt.setdefault('measurement',{})
    measurement.update(anchor='absolute_user_stop_time',stop_at_ns=d['stop_at_ns'],
        stop_method='user_authorized_boundary_request',hard_process_deadline=False,
        four_hour_cutoff_enabled=False)
    if epochs>0 and 'first_success_ns' not in measurement:
        measurement.update(first_success_ns=time.time_ns() if wall_ns is None else wall_ns,
                           first_success_monotonic=monotonic_now)
    return (time.time_ns() if wall_ns is None else wall_ns) >= d['stop_at_ns']
