#!/usr/bin/env python3
"""One bounded, CPU-observed, real four-rank ZeRO-2 + existing-MPS run.

This module never imports torch, Ray or a CUDA runtime. NVML is read-only.
The reviewed mps_target_run / real_fill_experiment helpers provide strict
UID + run marker + PID birth cleanup. Existing MPS services are protected.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid


ROOT = Path('/mnt/data/lcx1/dong/AcceRL-scheduler')
DATA = Path('/mnt/data/lcx1/dong/need_data/mps_profile_20261015')
BASE = Path('/mnt/data/lcx1/dong/need_data/archive/mps_target_run_20261014.py')
TERMINAL = {'preflight_passed', 'completed', 'failed', 'stopped_by_user'}
MODEL_NAMES = ('policy', 'ctrl', 'reward')
TZ = dt.timezone(dt.timedelta(hours=8))


def load_base():
    path = BASE if BASE.is_file() else Path(__file__).with_name('mps_target_run.py')
    spec = importlib.util.spec_from_file_location('accerl_mps_profile_base', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
safe = base.safe
atomic_json = safe.atomic_json


def stamp(unix=None):
    return dt.datetime.fromtimestamp(time.time() if unix is None else unix, TZ).isoformat(timespec='seconds')


def summary(values):
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return {'n': 0, 'mean': None, 'p50': None, 'p95': None, 'max': None, 'min': None}
    return {'n': len(clean), 'mean': sum(clean) / len(clean),
            'p50': clean[int((len(clean) - 1) * .5)], 'p95': clean[int((len(clean) - 1) * .95)],
            'max': clean[-1], 'min': clean[0]}


def load_status(data):
    try:
        return json.loads((data / 'runnerstatus.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


def build_environment(data, run_id, started_ns, cards, epochs):
    env = base.build_environment(data, run_id, started_ns, cards)
    for key in ('ACCERL_EVAL_TARGET', 'ACCERL_EVAL_CONSECUTIVE', 'ACCERL_EVAL_MIN_EPISODES',
                'ACCERL_SOFT_STOP_AT', 'ACCERL_RESUME', 'ACCERL_RESUME_CHECKPOINT'):
        env.pop(key, None)
    env.update({'ACCERL_MEASURE_STOP_AFTER_EPOCHS': str(epochs),
                'ACCERL_SYSTEM_PROFILE': '1', 'ACCERL_PROFILE_INTERVAL_SEC': '1',
                'ACCERL_RAY_TMP': f'/dev/shm/aprof_{os.getuid()}_{run_id[-8:]}',
                # The final snapshot covers epoch 25; avoid a duplicate periodic
                # snapshot at the same epoch while collecting system timings.
                'ACCERL_SNAPSHOT_INTERVAL': str(epochs + 1),
                'ACCE_CAPTURE_TRACE_EVERY': '10'})
    return env


def build_command(root):
    command = base.build_command(root)
    command[command.index('--exp-name') + 1] = 'OpenVLA_DS_gipo_DISCRETE_task0_ctrl_wm_four_zero2_mps_profile'
    return command


def create_resource_schema(connection):
    connection.executescript('''
    CREATE TABLE IF NOT EXISTS system_samples (
      run_id TEXT NOT NULL, ts_ns INTEGER NOT NULL, elapsed_seconds REAL NOT NULL,
      sample_duration_seconds REAL NOT NULL, late_seconds REAL NOT NULL,
      missed_ticks INTEGER NOT NULL, payload_json TEXT NOT NULL,
      PRIMARY KEY(run_id, ts_ns));
    CREATE TABLE IF NOT EXISTS process_samples (
      run_id TEXT NOT NULL, ts_ns INTEGER NOT NULL, pid INTEGER NOT NULL,
      birth_ticks INTEGER NOT NULL, role TEXT NOT NULL, physical_gpu INTEGER,
      elapsed_seconds REAL, cpu_cores REAL, rss_bytes INTEGER, threads INTEGER,
      read_bytes_delta INTEGER, write_bytes_delta INTEGER,
      voluntary_ctx_delta INTEGER, involuntary_ctx_delta INTEGER,
      payload_json TEXT NOT NULL, PRIMARY KEY(run_id, ts_ns, pid, birth_ticks));
    CREATE INDEX IF NOT EXISTS process_samples_run_role_time ON process_samples(run_id,role,ts_ns);
    CREATE TABLE IF NOT EXISTS sampler_health (
      run_id TEXT PRIMARY KEY, ts_ns INTEGER NOT NULL, payload_json TEXT NOT NULL);
    ''')
    connection.commit()


def pressure_snapshot():
    result = {}
    for name in ('cpu', 'memory', 'io'):
        try:
            lines = (Path('/proc/pressure') / name).read_text().splitlines()
            result[name] = {line.split()[0]: {key: float(value) for key, value in
                           (part.split('=', 1) for part in line.split()[1:])} for line in lines}
        except (OSError, ValueError):
            result[name] = None
    return result


def cgroup_snapshot():
    """Read the inherited cgroup quota/counters, without changing limits."""
    try:
        groups = Path('/proc/self/cgroup').read_text().splitlines()
        unified = next((line.split(':', 2)[2] for line in groups if line.startswith('0::')), None)
        if unified is None:
            return {'available': False, 'reason': 'no unified cgroupv2 entry'}
        root = Path('/sys/fs/cgroup').resolve()
        directory = (root / unified.lstrip('/')).resolve()
        if directory != root and root not in directory.parents:
            return {'available': False, 'reason': 'cgroup path outside sysfs root'}
        result = {'available': True, 'path': unified,
                  'scope': 'sampler inherited cgroup; may contain additional processes/users'}
        for name in ('cpu.max', 'cpu.stat', 'cpu.pressure', 'memory.max', 'memory.current',
                     'cpuset.cpus.effective'):
            try:
                result[name] = (directory / name).read_text().strip()
            except OSError:
                result[name] = None
        return result
    except (OSError, ValueError):
        return {'available': False, 'reason': 'cgroup proc/sysfs data inaccessible'}


def delta_counter(current, previous, key):
    if previous is None or key not in current or key not in previous:
        return None
    return max(0, current[key] - previous[key])


class ResourceCollector:
    """Injectable readers make the delta/cadence logic independently CPU-testable."""
    def __init__(self, state, *, psutil_module=None, nvml_module=None):
        if psutil_module is None:
            import psutil as psutil_module
        if nvml_module is None:
            import pynvml as nvml_module
        self.ps = psutil_module
        self.nv = nvml_module
        self.state = state
        self.nv.nvmlInit()
        self.handles = {int(card['index']): self.nv.nvmlDeviceGetHandleByUUID(card['uuid'])
                        for card in state['preflight']['cards']}
        if sorted(self.handles) != [4, 5, 6, 7]:
            raise RuntimeError('Sampler may only open physical GPU4-7 UUID handles')
        self.known = {}
        self.roles = {}
        self.previous_process = {}
        self.previous_disk = None
        self.previous_host_monotonic = None
        self.self_process = self.ps.Process(os.getpid())
        self.ps.cpu_times_percent(interval=None)
        self.ps.cpu_percent(interval=None, percpu=True)

    def close(self):
        self.nv.nvmlShutdown()

    def nv_read(self, function, *args):
        try:
            return getattr(self.nv, function)(*args)
        except Exception as error:
            # Unsupported counters remain null, rather than invented zeros.
            return {'unavailable': type(error).__name__}

    @staticmethod
    def available(value):
        return None if isinstance(value, dict) and 'unavailable' in value else value

    def gpu(self, physical, handle):
        values, unavailable = {'physical_gpu': physical}, {}
        def read(key, method, *args, scale=1):
            item = self.nv_read(method, handle, *args)
            if isinstance(item, dict) and 'unavailable' in item:
                unavailable[key] = item['unavailable']; values[key] = None
            else:
                values[key] = item * scale
        util = self.nv_read('nvmlDeviceGetUtilizationRates', handle)
        if self.available(util) is None:
            values.update(gpu_busy_percent=None, memory_controller_busy_percent=None)
            unavailable['utilization'] = util['unavailable']
        else:
            values.update(gpu_busy_percent=util.gpu, memory_controller_busy_percent=util.memory)
        memory = self.nv_read('nvmlDeviceGetMemoryInfo', handle)
        if self.available(memory) is None:
            values.update(memory_used_bytes=None, memory_total_bytes=None)
            unavailable['memory'] = memory['unavailable']
        else:
            values.update(memory_used_bytes=memory.used, memory_total_bytes=memory.total)
        read('power_watts', 'nvmlDeviceGetPowerUsage', scale=.001)
        read('sm_clock_mhz', 'nvmlDeviceGetClockInfo', getattr(self.nv, 'NVML_CLOCK_SM', 1))
        read('memory_clock_mhz', 'nvmlDeviceGetClockInfo', getattr(self.nv, 'NVML_CLOCK_MEM', 2))
        read('pcie_tx_bytes_per_second', 'nvmlDeviceGetPcieThroughput',
             getattr(self.nv, 'NVML_PCIE_UTIL_TX_BYTES', 0), scale=1024)
        read('pcie_rx_bytes_per_second', 'nvmlDeviceGetPcieThroughput',
             getattr(self.nv, 'NVML_PCIE_UTIL_RX_BYTES', 1), scale=1024)
        values['unavailable'] = unavailable
        return values

    def host(self, monotonic):
        cpu = self.ps.cpu_times_percent(interval=None)._asdict()
        per_cpu = self.ps.cpu_percent(interval=None, percpu=True)
        virtual, swap = self.ps.virtual_memory(), self.ps.swap_memory()
        disk_raw = self.ps.disk_io_counters()
        disk = disk_raw._asdict() if disk_raw is not None else {}
        wall = monotonic - self.previous_host_monotonic if self.previous_host_monotonic is not None else None
        disk_rates = {key + '_per_second': (delta_counter(disk, self.previous_disk, key) / wall
                      if wall and self.previous_disk is not None and key in disk else None)
                      for key in ('read_bytes', 'write_bytes', 'read_count', 'write_count',
                                  'read_time', 'write_time', 'busy_time')}
        self.previous_disk, self.previous_host_monotonic = disk, monotonic
        try:
            load = list(os.getloadavg())
        except (OSError, AttributeError):
            load = None
        return {'cpu_times_percent': cpu, 'logical_cpus': self.ps.cpu_count(logical=True),
                'per_cpu_busy_percent': per_cpu,
                'busy_logical_core_equivalents': sum(per_cpu) / 100,
                'memory': virtual._asdict(), 'swap': swap._asdict(), 'load_average': load,
                'pressure': pressure_snapshot(), 'disk_io_host_aggregate': disk_rates,
                'cgroup': cgroup_snapshot(),
                'collector_cpu_seconds': sum(self.self_process.cpu_times()[:2])}

    def refresh_identities(self, data):
        # Full /proc discovery every ten seconds; already verified identities
        # are then rechecked at each two-second process sample.
        self.known = {item['pid']: item for item in safe.scan_owned(self.state)}
        connection = safe.sqlite_reader(data)
        if connection is None:
            return
        try:
            placement = safe.placement_evidence(connection, self.state['run_id'])
            self.roles = {item['pid']: {'role': item['role'], 'physical_gpu': item['physical_gpu']}
                          for item in placement['actors']}
            # The safety helper deliberately recognizes only the sixteen GPU
            # actors. CPU actors still need attribution for system profiling.
            for row in safe.query_events(connection, self.state['run_id'], 'placement', limit=2000):
                pid = row.get('pid')
                if pid is not None and pid not in self.roles:
                    self.roles[pid] = {'role': row['details'].get('role') or row.get('actor') or 'other_owned',
                                       'physical_gpu': None}
        except sqlite3.OperationalError as error:
            if 'no such table' not in str(error):
                raise
        finally:
            connection.close()

    def processes(self, ts_ns, monotonic):
        rows = []
        for pid, expected in list(self.known.items()):
            current = safe.proc_identity(pid)
            if not (safe.same_process(expected, current, run_id=self.state['run_id'])
                    and current['uid'] == self.state['uid']
                    and current['birth_ticks'] >= self.state['minimum_child_birth_ticks']
                    and not safe.protected(current)):
                self.known.pop(pid, None)
                continue
            try:
                process = self.ps.Process(pid)
                with process.oneshot():
                    cpu = process.cpu_times()
                    counters = {'cpu_seconds': cpu.user + cpu.system,
                                'rss_bytes': process.memory_info().rss, 'threads': process.num_threads(),
                                'status': process.status(), 'cpu_affinity': process.cpu_affinity(),
                                'user_cpu_seconds': cpu.user, 'system_cpu_seconds': cpu.system}
                    for name, reader in (('io', process.io_counters), ('context_switches', process.num_ctx_switches)):
                        try:
                            counters[name] = reader()._asdict()
                        except (self.ps.AccessDenied, self.ps.NoSuchProcess):
                            counters[name] = {}
                # A PID cannot be silently reused during psutil collection.
                if not safe.same_process(expected, safe.proc_identity(pid), run_id=self.state['run_id']):
                    continue
                identity_key = (pid, expected['birth_ticks'])
                previous = self.previous_process.get(identity_key)
                wall = monotonic - previous['monotonic'] if previous else None
                cpu_delta = counters['cpu_seconds'] - previous['cpu_seconds'] if previous else None
                role = self.roles.get(pid, {'role': current['comm'] or 'other_owned', 'physical_gpu': None})
                io_prev = previous.get('io', {}) if previous else None
                ctx_prev = previous.get('context_switches', {}) if previous else None
                row = {'run_id': self.state['run_id'], 'ts_ns': ts_ns, 'pid': pid,
                       'birth_ticks': expected['birth_ticks'], **role, 'elapsed_seconds': wall,
                       'cpu_cores': max(0, cpu_delta) / wall if wall else None,
                       'rss_bytes': counters['rss_bytes'], 'threads': counters['threads'],
                       'read_bytes_delta': delta_counter(counters['io'], io_prev, 'read_bytes'),
                       'write_bytes_delta': delta_counter(counters['io'], io_prev, 'write_bytes'),
                       'voluntary_ctx_delta': delta_counter(counters['context_switches'], ctx_prev, 'voluntary'),
                       'involuntary_ctx_delta': delta_counter(counters['context_switches'], ctx_prev, 'involuntary'),
                       'payload_json': json.dumps(counters, separators=(',', ':'))}
                rows.append(row)
                self.previous_process[identity_key] = dict(counters, monotonic=monotonic)
            except (self.ps.NoSuchProcess, self.ps.AccessDenied, self.ps.ZombieProcess):
                continue
        return rows


PROCESS_COLUMNS = ('run_id', 'ts_ns', 'pid', 'birth_ticks', 'role', 'physical_gpu', 'elapsed_seconds',
                   'cpu_cores', 'rss_bytes', 'threads', 'read_bytes_delta', 'write_bytes_delta',
                   'voluntary_ctx_delta', 'involuntary_ctx_delta', 'payload_json')


def sample(data, run_id):
    state = load_status(data)
    if state.get('run_id') != run_id:
        raise RuntimeError('Sampler run identity does not match runnerstatus.json')
    with safe.exclusive(data / '.sampler.lock'):
        connection = sqlite3.connect(data / 'collection.sqlite', timeout=.2)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA busy_timeout=200')
        create_resource_schema(connection)
        collector = ResourceCollector(state)
        health = {'samples': 0, 'process_rows': 0, 'sqlite_dropped_samples': 0,
                  'missed_ticks': 0, 'errors': 0, 'last_error': None, 'interval_seconds': 1,
                  'process_interval_seconds': 2, 'identity_refresh_seconds': 10,
                  'pid': os.getpid(), 'started_ns': time.time_ns(), 'previous_write_seconds': None}
        next_tick = next_process = next_discovery = time.monotonic()
        try:
            while True:
                current = load_status(data)
                if current.get('run_id') != run_id or current.get('state') in TERMINAL:
                    break
                if not safe.same_process(current.get('supervisor'),
                                         safe.proc_identity(current.get('supervisor', {}).get('pid', -1))):
                    break
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(min(.25, next_tick - now)); continue
                ts_ns = time.time_ns()
                late = max(0, now - next_tick)
                missed = int(late)
                health['missed_ticks'] += missed
                next_tick += (missed + 1)
                capture_start = time.monotonic()
                try:
                    if now >= next_discovery:
                        collector.refresh_identities(data); next_discovery = now + 10
                    payload = {'host': collector.host(now),
                               'gpus': [collector.gpu(index, handle) for index, handle in collector.handles.items()],
                               'previous_write_seconds': health['previous_write_seconds']}
                    process_rows = collector.processes(ts_ns, now) if now >= next_process else []
                    if now >= next_process:
                        next_process = now + 2
                    duration = time.monotonic() - capture_start
                    write_start = time.monotonic()
                    with connection:
                        connection.execute('INSERT INTO system_samples VALUES (?,?,?,?,?,?,?)',
                                           (run_id, ts_ns, (ts_ns - state['started_ns']) / 1e9,
                                            duration, late, missed, json.dumps(payload, separators=(',', ':'))))
                        connection.executemany('INSERT INTO process_samples (' + ','.join(PROCESS_COLUMNS)
                                               + ') VALUES (' + ','.join('?' for _ in PROCESS_COLUMNS) + ')',
                                               [tuple(row[key] for key in PROCESS_COLUMNS) for row in process_rows])
                        written_health = dict(health, last_sample_ns=ts_ns, samples=health['samples'] + 1,
                                              process_rows=health['process_rows'] + len(process_rows))
                        connection.execute('INSERT OR REPLACE INTO sampler_health VALUES (?,?,?)',
                                           (run_id, ts_ns, json.dumps(written_health, separators=(',', ':'))))
                    health.update(samples=health['samples'] + 1,
                                  process_rows=health['process_rows'] + len(process_rows),
                                  last_sample_ns=ts_ns,
                                  previous_write_seconds=time.monotonic() - write_start)
                except sqlite3.OperationalError as error:
                    connection.rollback(); health['sqlite_dropped_samples'] += 1
                    health['last_error'] = repr(error)
                except Exception as error:
                    health['errors'] += 1; health['last_error'] = repr(error)
                    if health['errors'] >= 10:
                        raise
        finally:
            health.update(stopped_ns=time.time_ns())
            try:
                with connection:
                    connection.execute('INSERT OR REPLACE INTO sampler_health VALUES (?,?,?)',
                                       (run_id, time.time_ns(), json.dumps(health)))
            finally:
                collector.close(); connection.close()
    return 0


