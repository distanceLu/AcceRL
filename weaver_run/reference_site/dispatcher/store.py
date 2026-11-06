"""Durable cooperative GPU4-7 queue. A launch claim is never automatically retried."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath
import sqlite3
import time
import uuid
from contextlib import contextmanager


STATES = ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'blocked')
DEFAULT_CWD_ROOT = '/mnt/data/lcx1/dong'
SUBMISSION_ID_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,63}')


class QueueError(ValueError):
    pass


class QueueStore:
    def __init__(self, root, *, allowed_cwd_root=DEFAULT_CWD_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'queue.sqlite'
        self.allowed_cwd_root = str(allowed_cwd_root)
        with self.connect() as con:
            con.execute('PRAGMA journal_mode=WAL')
            con.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, submission_id TEXT, name TEXT NOT NULL, argv_json TEXT NOT NULL,
                    cwd TEXT NOT NULL, timeout_seconds INTEGER NOT NULL,
                    position INTEGER NOT NULL, state TEXT NOT NULL,
                    created_ns INTEGER NOT NULL, claimed_ns INTEGER, started_ns INTEGER,
                    finished_ns INTEGER, deadline_ns INTEGER, launch_claim TEXT UNIQUE,
                    identity_json TEXT, owned_json TEXT NOT NULL DEFAULT '[]',
                    logpath TEXT, exitcode INTEGER, reason TEXT, heartbeat_ns INTEGER
                );
                CREATE INDEX IF NOT EXISTS jobs_queue_order ON jobs(state,position,created_ns);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    at_ns INTEGER NOT NULL, kind TEXT NOT NULL, details_json TEXT NOT NULL
                );
            ''')
            # Existing queues predate submission IDs. Migrate under the write lock
            # so simultaneous client/daemon construction cannot both add the column.
            con.execute('BEGIN IMMEDIATE')
            columns = {row['name'] for row in con.execute('PRAGMA table_info(jobs)')}
            if 'submission_id' not in columns:
                con.execute('ALTER TABLE jobs ADD COLUMN submission_id TEXT')
            if 'manual' not in columns:
                con.execute('ALTER TABLE jobs ADD COLUMN manual INTEGER NOT NULL DEFAULT 0')
            con.execute('CREATE UNIQUE INDEX IF NOT EXISTS jobs_submission_id ON jobs(submission_id)')
            con.commit()

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute('PRAGMA busy_timeout=10000')
        con.execute('PRAGMA synchronous=FULL')
        try:
            with con:
                yield con
        finally:
            con.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        item = dict(row)
        item['manual'] = bool(item.get('manual', 0))
        item['argv'] = json.loads(item.pop('argv_json'))
        item['identity'] = json.loads(item.pop('identity_json') or 'null')
        item['owned'] = json.loads(item.pop('owned_json'))
        return item

    @staticmethod
    def event(con, job_id, kind, details):
        con.execute('INSERT INTO events(job_id,at_ns,kind,details_json) VALUES(?,?,?,?)',
                    (job_id, time.time_ns(), kind, json.dumps(details, sort_keys=True)))

    def validate_submission(self, name, argv, cwd, timeout_seconds, manual=False):
        if not isinstance(name, str) or not name.strip() or len(name) > 160:
            raise QueueError('name must contain 1-160 characters')
        if (not isinstance(argv, list) or not argv or len(argv) > 256
                or any(not isinstance(x, str) or '\0' in x or len(x) > 32768 for x in argv)):
            raise QueueError('argv must be a nonempty JSON list of bounded strings without NUL')
        if not (Path(argv[0]).is_absolute() or PurePosixPath(argv[0]).is_absolute()):
            raise QueueError('argv[0] must be an absolute executable path')
        protected_assignments = ('CUDA_VISIBLE_DEVICES=', 'CUDA_MPS_PIPE_DIRECTORY=',
                                 'CUDA_MPS_LOG_DIRECTORY=', 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=',
                                 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=')
        if any(value.startswith(protected_assignments) for value in argv):
            raise QueueError('GPU/MPS environment assignments belong to the queue, not the submitted argv')
        if type(manual) is not bool:
            raise QueueError('manual must be a boolean')
        if type(timeout_seconds) is not int or (
                (manual and timeout_seconds != 0)
                or (not manual and not 60 <= timeout_seconds <= 259200)):
            raise QueueError('manual jobs require timeout_seconds=0; bounded jobs require 60..259200')
        if not isinstance(cwd, str) or '\0' in cwd:
            raise QueueError('cwd must be an absolute directory')
        # Production runs on Linux. Tests may explicitly use their temporary root.
        if self.allowed_cwd_root == DEFAULT_CWD_ROOT:
            p = PurePosixPath(cwd)
            if not p.is_absolute() or '..' in p.parts or str(p) != cwd.rstrip('/'):
                raise QueueError('cwd must be canonical and absolute')
            try:
                p.relative_to(PurePosixPath(self.allowed_cwd_root))
            except ValueError:
                raise QueueError('cwd must remain under ' + self.allowed_cwd_root) from None
        else:
            try:
                Path(cwd).resolve().relative_to(Path(self.allowed_cwd_root).resolve())
            except ValueError:
                raise QueueError('cwd escapes the configured root') from None

    def create(self, name, argv, cwd, timeout_seconds, submission_id=None, manual=False):
        self.validate_submission(name, argv, cwd, timeout_seconds, manual)
        if submission_id is not None and (not isinstance(submission_id, str)
                                          or not SUBMISSION_ID_RE.fullmatch(submission_id)):
            raise QueueError('submission_id must contain 1-64 safe identifier characters')
        job_id = uuid.uuid4().hex
        now = time.time_ns()
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            if submission_id is not None:
                row = con.execute('SELECT * FROM jobs WHERE submission_id=?', (submission_id,)).fetchone()
                if row is not None:
                    existing = self.decode(row)
                    same = (existing['name'] == name.strip() and existing['argv'] == argv
                            and existing['cwd'] == cwd and existing['timeout_seconds'] == timeout_seconds
                            and existing['manual'] == manual)
                    if not same:
                        raise QueueError('submission_id already belongs to a different job payload')
                    con.commit()
                    return existing
            position = con.execute('SELECT COALESCE(MAX(position),0)+1 FROM jobs').fetchone()[0]
            con.execute('INSERT INTO jobs(id,submission_id,name,argv_json,cwd,timeout_seconds,position,state,created_ns,manual) '
                        'VALUES(?,?,?,?,?,?,?,?,?,?)',
                        (job_id, submission_id, name.strip(), json.dumps(argv), cwd, timeout_seconds, position, 'queued', now, int(manual)))
            self.event(con, job_id, 'submitted', {'gpu_bundle': 'gpu4-7', 'automatic_retry': False,
                                                'submission_id': submission_id, 'manual': manual,
                                                'timeout_seconds': timeout_seconds})
            con.commit()
        return self.get(job_id)

    def list(self):
        with self.connect() as con:
            return [self.decode(x) for x in con.execute('SELECT * FROM jobs ORDER BY position,created_ns')]

    def get(self, job_id):
        with self.connect() as con:
            return self.decode(con.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())

    def active(self):
        with self.connect() as con:
            return [self.decode(x) for x in con.execute("SELECT * FROM jobs WHERE state IN ('running','blocked') ORDER BY claimed_ns")]

    def cancel(self, job_id):
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row or row['state'] != 'queued':
                raise QueueError('Only a queued job can be cancelled')
            con.execute("UPDATE jobs SET state='cancelled',finished_ns=?,reason=? WHERE id=?",
                        (time.time_ns(), 'cancelled_before_launch', job_id))
            self.event(con, job_id, 'cancelled', {})
            con.commit()
        return self.get(job_id)

    def reorder(self, job_id, delta):
        if type(delta) is not int or delta not in (-1, 1):
            raise QueueError('delta must be -1 or +1')
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            rows = con.execute("SELECT id FROM jobs WHERE state='queued' ORDER BY position,created_ns").fetchall()
            ids = [x['id'] for x in rows]
            if job_id not in ids:
                raise QueueError('Only a queued job can be reordered')
            index = ids.index(job_id)
            target = max(0, min(len(ids)-1, index+delta))
            ids[index], ids[target] = ids[target], ids[index]
            for position, value in enumerate(ids, 1):
                con.execute('UPDATE jobs SET position=? WHERE id=?', (position, value))
            self.event(con, job_id, 'reordered', {'delta': delta})
            con.commit()
        return self.get(job_id)

    def claim_next(self):
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            if con.execute("SELECT 1 FROM jobs WHERE state IN ('running','blocked') LIMIT 1").fetchone():
                con.commit()
                return None
            row = con.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY position,created_ns LIMIT 1").fetchone()
            if row is None:
                con.commit()
                return None
            now, claim = time.time_ns(), uuid.uuid4().hex
            con.execute("UPDATE jobs SET state='running',claimed_ns=?,launch_claim=?,heartbeat_ns=? WHERE id=? AND state='queued'",
                        (now, claim, now, row['id']))
            self.event(con, row['id'], 'launch_claimed', {'claim': claim, 'automatic_retry': False})
            con.commit()
        return self.get(row['id'])

    def record_started(self, job_id, identity, logpath, deadline_ns):
        now = time.time_ns()
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute('SELECT state,launch_claim,identity_json FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row or row['state'] != 'running' or not row['launch_claim'] or row['identity_json']:
                raise QueueError('Job has no unused launch claim')
            con.execute('UPDATE jobs SET identity_json=?,owned_json=?,logpath=?,deadline_ns=?,started_ns=?,heartbeat_ns=? WHERE id=?',
                        (json.dumps(identity), json.dumps([identity]), str(logpath), deadline_ns, now, now, job_id))
            self.event(con, job_id, 'started', {'identity': identity, 'deadline_ns': deadline_ns})
            con.commit()
        return self.get(job_id)

    def observe(self, job_id, owned):
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute('SELECT state,owned_json FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row or row['state'] not in ('running', 'blocked'):
                raise QueueError('Job is not active')
            # Retain every observed birth. A reparented child which clears its
            # environment remains owned after the parent has exited.
            identities = {(item['pid'], item['uid'], item['birth_ticks']): item
                          for item in json.loads(row['owned_json'])}
            identities.update({(item['pid'], item['uid'], item['birth_ticks']): item for item in owned})
            con.execute('UPDATE jobs SET owned_json=?,heartbeat_ns=? WHERE id=?',
                        (json.dumps(list(identities.values())), time.time_ns(), job_id))
            con.commit()

    def record_recovery_observation(self, job_id, details):
        """Append proof for releasing an uncertain claim without retrying it."""
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row or row['state'] not in ('running', 'blocked'):
                raise QueueError('Job is not active')
            self.event(con, job_id, 'recovery_quiet_observation', details)
            con.commit()

    def finish(self, job_id, state, reason, exitcode=None):
        if state not in ('succeeded', 'failed', 'blocked'):
            raise QueueError('Invalid worker terminal state')
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row or row['state'] not in ('running', 'blocked'):
                raise QueueError('Job is not active')
            now = time.time_ns()
            con.execute('UPDATE jobs SET state=?,reason=?,exitcode=?,finished_ns=?,heartbeat_ns=? WHERE id=?',
                        (state, str(reason), exitcode, now, now, job_id))
            self.event(con, job_id, state, {'reason': str(reason), 'exitcode': exitcode})
            con.commit()
        return self.get(job_id)
