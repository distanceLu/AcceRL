"""Low-overhead, multi-process capture for AcceRL scheduler experiments.

All processes write one SQLite WAL database. Replay reservation and actual
training consumption are distinct, and identities are never inferred from batch
sizes. No torch dependency is imported until a CUDA span is explicitly requested.
"""
from __future__ import annotations

import atexit
import contextlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path


SCHEMA_VERSION = 1
WORKLOADS = ("policy_infer", "ctrl_infer", "reward_infer", "policy_train", "ctrl_train", "reward_train")


def now_ns():
    return time.time_ns()


def new_id(prefix="sample"):
    return "%s:%s" % (prefix, uuid.uuid4().hex)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
 run_id TEXT PRIMARY KEY, started_ns INTEGER NOT NULL, finished_ns INTEGER,
 status TEXT, config_json TEXT, schema_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, ts_ns INTEGER NOT NULL,
 actor TEXT NOT NULL, actor_id TEXT NOT NULL, pid INTEGER NOT NULL,
 kind TEXT NOT NULL, workload TEXT, request_id TEXT, submitted_ns INTEGER,
 start_ns INTEGER, end_ns INTEGER, actual_batch INTEGER, duration_ns INTEGER,
 gpu_ms REAL, gpu_device TEXT, allocated_bytes INTEGER, peak_bytes INTEGER,
 reserved_bytes INTEGER, free_bytes INTEGER, total_bytes INTEGER,
 queue_mean_ns REAL, queue_p95_ns REAL, queue_max_ns INTEGER,
 cpu_seconds REAL, rss_bytes INTEGER, model_version TEXT, status TEXT,
 details_json TEXT
);
CREATE INDEX IF NOT EXISTS events_run_workload ON events(run_id, workload, kind, ts_ns);
CREATE TABLE IF NOT EXISTS replay_samples (
 run_id TEXT NOT NULL, replay TEXT NOT NULL, sample_id TEXT NOT NULL,
 episode_id TEXT, sample_index INTEGER, added_ns INTEGER, resident INTEGER NOT NULL DEFAULT 0,
 evicted_ns INTEGER, eviction_reason TEXT, versions_json TEXT,
 reserved_count INTEGER NOT NULL DEFAULT 0, first_reserved_ns INTEGER,
 last_reserved_ns INTEGER, use_count INTEGER NOT NULL DEFAULT 0,
 policy_uses INTEGER NOT NULL DEFAULT 0, ctrl_uses INTEGER NOT NULL DEFAULT 0,
 reward_uses INTEGER NOT NULL DEFAULT 0, eval_uses INTEGER NOT NULL DEFAULT 0,
 first_used_ns INTEGER, last_used_ns INTEGER,
 PRIMARY KEY (run_id, replay, sample_id)
);
CREATE INDEX IF NOT EXISTS replay_run ON replay_samples(run_id, replay, resident);
CREATE TABLE IF NOT EXISTS replay_snapshots (
 id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, ts_ns INTEGER NOT NULL,
 replay TEXT NOT NULL, actor_id TEXT NOT NULL, reason TEXT NOT NULL,
 resident_count INTEGER NOT NULL, capacity INTEGER NOT NULL, details_json TEXT
);
"""


class Telemetry:
    def __init__(self, actor, actor_id="", *, db_path=None, run_id=None,
                 flush_records=256, flush_seconds=2.0, trace_every=None):
        self.actor = str(actor)
        self.actor_id = str(actor_id)
        self.run_id = str(run_id or os.environ.get("ACCERL_RUN_ID") or os.environ.get("ACCE_CAPTURE_RUN_ID", "manual"))
        self.db_path = str(db_path or os.environ.get("ACCERL_TELEMETRY_DB") or os.environ.get("ACCE_CAPTURE_DB", "/mnt/data/lcx1/dong/need_data/collection.sqlite"))
        self.flush_records = int(flush_records)
        self.flush_seconds = float(flush_seconds)
        self.trace_every = int(trace_every if trace_every is not None else os.environ.get("ACCE_CAPTURE_TRACE_EVERY", "100"))
        self.pid = os.getpid()
        self._lock = threading.RLock()
        self._pending = []
        self._gpu_pending = []
        self._span_counts = {}
        self._gpu_span_depth = 0
        self._closed = False
        self._last_error = None
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, timeout=60, check_same_thread=False)
        self.connection.execute("PRAGMA busy_timeout=60000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(_SCHEMA)
        self.connection.execute(
            "INSERT OR IGNORE INTO runs(run_id,started_ns,status,schema_version) VALUES(?,?,?,?)",
            (self.run_id, now_ns(), "running", SCHEMA_VERSION),
        )
        self.connection.commit()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._periodic_flush, name="capture-sqlite", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def _periodic_flush(self):
        while not self._stop.wait(self.flush_seconds):
            try:
                self.flush()
                self._last_error = None
            except Exception as exc:
                if self._last_error is None:
                    print("[capture] SQLite flush failed; retained for retry: %r" % exc, flush=True)
                self._last_error = exc

    def _queue(self, sql, parameters):
        with self._lock:
            if self._closed:
                raise RuntimeError("Telemetry writer is closed")
            self._pending.append((sql, parameters))
            if len(self._pending) >= self.flush_records:
                self.flush()

    def _queue_many(self, sql, rows):
        # One operation represents a whole batch, so 1024 samples do not incur
        # 1024 Python lock acquisitions or transactions.
        rows = list(rows)
        if rows:
            self._queue(("many", sql), rows)

    def event(self, kind, *, workload=None, request_id=None, **fields):
        columns = ["run_id", "ts_ns", "actor", "actor_id", "pid", "kind", "workload", "request_id"]
        values = [self.run_id, fields.pop("ts_ns", now_ns()), self.actor, self.actor_id, self.pid, kind, workload, request_id]
        known = {"submitted_ns", "start_ns", "end_ns", "actual_batch", "duration_ns", "gpu_ms", "gpu_device", "allocated_bytes", "peak_bytes", "reserved_bytes", "free_bytes", "total_bytes", "queue_mean_ns", "queue_p95_ns", "queue_max_ns", "cpu_seconds", "rss_bytes", "model_version", "status"}
        extra = fields.pop("details", {}) or {}
        for key, value in fields.items():
            if key in known:
                columns.append(key)
                values.append(str(value) if key == "model_version" and value is not None else value)
            else:
                extra[key] = value
        columns.append("details_json")
        values.append(_json(extra))
        sql = "INSERT INTO events(%s) VALUES(%s)" % (",".join(columns), ",".join("?" for _ in columns))
        self._queue(sql, values)

    def span(self, workload, *, actual_batch=1, model_version=None, request_ids=None,
             submitted_ns=None, cuda=False, **details):
        return _Span(self, workload, actual_batch, model_version, request_ids, submitted_ns, cuda, details)

    def replay_used(self, replay, metadata, *, workload, sample_ids=None,
                    step=None, model_version=None, use_id=None):
        """Call after successful backward+step of the corresponding microbatch.

        Count actual sample exposures; optimizer update boundaries are recorded
        separately by the caller. A WM sample can contribute to BOTH models.
        """
        sample_ids = list(metadata["sample_ids"] if sample_ids is None else sample_ids)
        column = {"policy_train": "policy_uses", "ctrl_train": "ctrl_uses",
                  "reward_train": "reward_uses", "ctrl_eval": "eval_uses"}.get(workload)
        if column is None:
            raise ValueError("Unrecognized replay consumer: %s" % workload)
        ts = now_ns()
        training = int(workload != "ctrl_eval")
        sql = """INSERT INTO replay_samples(run_id,replay,sample_id,use_count,%s,first_used_ns,last_used_ns)
                 VALUES(?,?,?,?,?,?,?) ON CONFLICT(run_id,replay,sample_id) DO UPDATE SET
                 use_count=use_count+excluded.use_count,%s=%s+excluded.%s,
                 first_used_ns=CASE WHEN excluded.first_used_ns IS NULL THEN first_used_ns
                     WHEN first_used_ns IS NULL THEN excluded.first_used_ns
                     ELSE MIN(first_used_ns,excluded.first_used_ns) END,
                 last_used_ns=CASE WHEN excluded.last_used_ns IS NULL THEN last_used_ns
                     WHEN last_used_ns IS NULL THEN excluded.last_used_ns
                     ELSE MAX(last_used_ns,excluded.last_used_ns) END""" % (column, column, column, column)
        # Eval use does not make an otherwise-unused training sample "used".
        self._queue_many(sql, [(self.run_id, replay, str(s), training, 1, ts if training else None, ts if training else None) for s in sample_ids])
        self.event("replay_use", workload=workload, actual_batch=len(sample_ids), model_version=model_version,
                   details={"replay": replay, "reservation_id": metadata.get("reservation_id"),
                            "step": step, "use_id": use_id or new_id("use")})

    def config(self, config):
        self._queue("UPDATE runs SET config_json=? WHERE run_id=?", (_json(config), self.run_id))

    def finish(self, status="completed"):
        self._queue("UPDATE runs SET status=?,finished_ns=? WHERE run_id=?", (status, now_ns(), self.run_id))
        self.flush()

    def _resolve_gpu(self, force=False):
        ready, waiting = [], []
        for start, end, span_fields in self._gpu_pending:
            if force:
                end.synchronize()
            if force or end.query():
                span_fields["gpu_ms"] = float(start.elapsed_time(end))
                ready.append(span_fields)
            else:
                waiting.append((start, end, span_fields))
        self._gpu_pending = waiting
        for fields in ready:
            self.event("service", **fields)

    def flush(self, force_gpu=False):
        with self._lock:
            if self._closed:
                return
            self._resolve_gpu(force_gpu)
            if not self._pending:
                return
            pending = self._pending
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                for sql, parameters in pending:
                    if isinstance(sql, tuple):
                        self.connection.executemany(sql[1], parameters)
                    else:
                        self.connection.execute(sql, parameters)
                self.connection.commit()
                del self._pending[:len(pending)]
            except Exception:
                self.connection.rollback()
                raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._stop.set()
            self.flush(force_gpu=True)
            self.connection.close()
            self._closed = True


class _Span:
    def __init__(self, writer, workload, actual_batch, version, ids, submitted, cuda, details):
        self.writer, self.workload = writer, workload
        self.batch, self.version = int(actual_batch), version
        self.ids, self.submitted = list(ids or []), submitted
        self.cuda, self.details = cuda, dict(details)
        self.gpu_start = self.gpu_end = self.torch = None
        self.peak_scope = "unavailable"

    def __enter__(self):
        from nsight_capture import nvtx_begin
        self.nvtx_handle = nvtx_begin("%s/%s/%s batch=%s" % (
            self.writer.actor, self.writer.actor_id, self.workload, self.batch))
        counts = self.writer._span_counts
        counts[self.workload] = counts.get(self.workload, 0) + 1
        self.representative = counts[self.workload] == 1 or (
            self.writer.trace_every > 0 and counts[self.workload] % self.writer.trace_every == 0)
        if self.cuda:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                self.device = torch.cuda.current_device()
                nested = self.writer._gpu_span_depth > 0
                self.writer._gpu_span_depth += 1
                if self.representative:
                    torch.cuda.synchronize(self.device)
                    if not nested:
                        torch.cuda.reset_peak_memory_stats(self.device)
                self.peak_scope = ("parent_span" if nested else "span" if self.representative
                                   else "process_since_last_representative_span")
                self.gpu_start = torch.cuda.Event(enable_timing=True)
                self.gpu_end = torch.cuda.Event(enable_timing=True)
                self.gpu_start.record()
        self.start_wall = now_ns()
        self.start_perf = time.perf_counter_ns()
        times = os.times()
        self.start_cpu = times.user + times.system
        return self

    def __exit__(self, exc_type, exc, tb):
        from nsight_capture import nvtx_end
        nvtx_end(getattr(self, "nvtx_handle", None))
        if self.gpu_end is not None:
            self.gpu_end.record()
            if self.representative:
                self.gpu_end.synchronize()
        end = now_ns()
        elapsed = time.perf_counter_ns() - self.start_perf
        times = os.times()
        cpu_seconds = times.user + times.system - self.start_cpu
        details = dict(self.details)
        details.update(request_ids=self.ids, representative=self.representative,
                       timing="cuda_events" if self.gpu_end is not None else "cpu_wall",
                       synchronized=self.representative and self.gpu_end is not None,
                       peak_scope=self.peak_scope)
        if exc is not None:
            details["error"] = repr(exc)
        submitted = self.submitted
        if submitted is None:
            submitted = []
        elif isinstance(submitted, int):
            submitted = [submitted]
        else:
            submitted = [int(s) for s in submitted if s is not None]
        delays = [max(0, self.start_wall - s) for s in submitted]
        fields = dict(workload=self.workload, start_ns=self.start_wall, end_ns=end,
                      duration_ns=elapsed, actual_batch=self.batch, model_version=self.version,
                      submitted_ns=min(submitted) if submitted else None,
                      queue_mean_ns=sum(delays) / len(delays) if delays else None,
                      queue_p95_ns=_percentile(delays, .95), queue_max_ns=max(delays) if delays else None,
                      cpu_seconds=cpu_seconds, status="error" if exc else "ok", details=details)
        try:
            import psutil
            fields["rss_bytes"] = psutil.Process().memory_info().rss
        except ImportError:
            pass
        if self.torch is not None:
            cuda = self.torch.cuda
            fields.update(gpu_device=str(self.device), allocated_bytes=cuda.memory_allocated(self.device),
                          peak_bytes=cuda.max_memory_allocated(self.device), reserved_bytes=cuda.memory_reserved(self.device))
            if self.representative:
                free, total = cuda.mem_get_info(self.device)
                fields.update(free_bytes=free, total_bytes=total)
            with self.writer._lock:
                self.writer._gpu_pending.append((self.gpu_start, self.gpu_end, fields))
                self.writer._gpu_span_depth = max(0, self.writer._gpu_span_depth - 1)
                # Ready events are harvested without a synchronization on ordinary batches.
                self.writer._resolve_gpu()
        else:
            self.writer.event("service", **fields)
        return False


class ReplayLedger:
    """Replay actor-side identity/reservation ledger; training commits go to DB.

    Keep this object's resident IDs aligned with the actual deque. Capacity
    eviction and shutdown disposal are different reasons and never conflated.
    """
    def __init__(self, writer, replay, capacity):
        self.writer, self.replay, self.capacity = writer, str(replay), int(capacity)
        self.resident = {}

    def add(self, sample_id, episode_id=None, sample_index=None, *, model_versions=None):
        sample_id = str(sample_id)
        if sample_id in self.resident:
            raise ValueError("Duplicate resident sample identity %s" % sample_id)
        ts = now_ns()
        self.resident[sample_id] = ts
        self.writer._queue("""INSERT INTO replay_samples(run_id,replay,sample_id,episode_id,sample_index,added_ns,resident,versions_json)
            VALUES(?,?,?,?,?,?,1,?) ON CONFLICT(run_id,replay,sample_id) DO UPDATE SET
            episode_id=excluded.episode_id,sample_index=excluded.sample_index,added_ns=excluded.added_ns,
            resident=1,versions_json=excluded.versions_json""",
            (self.writer.run_id, self.replay, sample_id, episode_id, sample_index, ts, _json(model_versions or {})))

    def evict(self, sample_id, reason="capacity"):
        sample_id = str(sample_id)
        if sample_id not in self.resident:
            raise KeyError("Eviction does not match resident ledger: %s" % sample_id)
        del self.resident[sample_id]
        self.writer._queue("UPDATE replay_samples SET resident=0,evicted_ns=?,eviction_reason=? WHERE run_id=? AND replay=? AND sample_id=?",
                           (now_ns(), reason, self.writer.run_id, self.replay, sample_id))

    def reserve(self, sample_ids, *, consumer="training"):
        ids = [str(s) for s in sample_ids]
        unknown = set(ids) - self.resident.keys()
        if unknown:
            raise KeyError("Reservation contains nonresident samples: %s" % list(unknown)[:3])
        ts = now_ns()
        self.writer._queue_many("""UPDATE replay_samples SET reserved_count=reserved_count+1,
            first_reserved_ns=COALESCE(first_reserved_ns,?),last_reserved_ns=?
            WHERE run_id=? AND replay=? AND sample_id=?""",
            [(ts, ts, self.writer.run_id, self.replay, sid) for sid in ids])
        result = {"reservation_id": new_id("reservation"), "sample_ids": ids, "reserved_ns": ts,
                  "replay": self.replay, "consumer": consumer}
        self.writer.event("replay_reserve", actual_batch=len(ids), details={
            "replay": self.replay, "reservation_id": result["reservation_id"], "consumer": consumer})
        return result

    def snapshot(self, reason="periodic"):
        self.writer._queue("INSERT INTO replay_snapshots(run_id,ts_ns,replay,actor_id,reason,resident_count,capacity,details_json) VALUES(?,?,?,?,?,?,?,?)",
                           (self.writer.run_id, now_ns(), self.replay, self.writer.actor_id, reason,
                            len(self.resident), self.capacity, "{}"))
        self.writer.flush()
        return replay_summary(self.writer.connection, self.writer.run_id, self.replay)

    def shutdown(self):
        self.snapshot("before_shutdown")
        for sample_id in list(self.resident):
            self.evict(sample_id, "shutdown")
        return self.snapshot("after_shutdown")


def replay_summary(connection, run_id, replay):
    """After all writers flush, these quantities reconcile the physical pool.

    added = resident + capacity_evicted + shutdown_discarded + other_discarded.
    Reservation count is not a training-use count. Uses include reuse; unique
    counts do not. Unused-at-eviction tests timestamps so a prefetched sample
    trained after eviction remains correctly unused *when it was evicted*.
    """
    names = ("added", "resident", "capacity_evicted", "shutdown_discarded", "other_discarded",
             "reserved", "reserved_unique", "actual_uses", "used_unique", "reused",
             "policy_uses", "ctrl_uses", "reward_uses", "eval_uses",
             "unused_resident", "unused_capacity_evicted", "unused_shutdown_discarded",
             "reserved_not_used_resident", "ctrl_used_unique", "reward_used_unique",
             "policy_used_unique", "ctrl_reused", "reward_reused", "policy_reused")
    query = """SELECT COUNT(*),COALESCE(SUM(resident),0),
      COALESCE(SUM(eviction_reason='capacity'),0),COALESCE(SUM(eviction_reason='shutdown'),0),
      COALESCE(SUM(resident=0 AND eviction_reason IS NOT NULL AND eviction_reason NOT IN ('capacity','shutdown')),0),
      COALESCE(SUM(reserved_count),0),COALESCE(SUM(reserved_count>0),0),COALESCE(SUM(use_count),0),
      COALESCE(SUM(use_count>0),0),COALESCE(SUM(MAX(use_count-1,0)),0),
      COALESCE(SUM(policy_uses),0),COALESCE(SUM(ctrl_uses),0),COALESCE(SUM(reward_uses),0),COALESCE(SUM(eval_uses),0),
      COALESCE(SUM(resident=1 AND use_count=0),0),
      COALESCE(SUM(eviction_reason='capacity' AND (first_used_ns IS NULL OR first_used_ns>evicted_ns)),0),
      COALESCE(SUM(eviction_reason='shutdown' AND (first_used_ns IS NULL OR first_used_ns>evicted_ns)),0),
      COALESCE(SUM(resident=1 AND reserved_count>0 AND use_count=0),0),
      COALESCE(SUM(ctrl_uses>0),0),COALESCE(SUM(reward_uses>0),0),COALESCE(SUM(policy_uses>0),0),
      COALESCE(SUM(MAX(ctrl_uses-1,0)),0),COALESCE(SUM(MAX(reward_uses-1,0)),0),COALESCE(SUM(MAX(policy_uses-1,0)),0)
      FROM replay_samples WHERE run_id=? AND replay=? AND added_ns IS NOT NULL"""
    result = dict(zip(names, connection.execute(query, (run_id, replay)).fetchone()))
    result["reconciles"] = result["added"] == sum(result[k] for k in ("resident", "capacity_evicted", "shutdown_discarded", "other_discarded"))
    return result
