"""CPU-only checks for the requested Ray object-store allocation."""
from pathlib import Path
import os

OBJECT_STORE_GIB = 328
OBJECT_STORE_BYTES = OBJECT_STORE_GIB * 1024**3


def validate_object_store_budget(size_bytes=OBJECT_STORE_BYTES):
    if size_bytes != OBJECT_STORE_BYTES:
        raise ValueError("This candidate requires exactly 328 GiB of Ray object storage")
    meminfo = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        meminfo[key] = int(value.split()[0]) * 1024
    shared = os.statvfs('/dev/shm')
    shm_available = shared.f_bavail * shared.f_frsize
    if shm_available < size_bytes + 8 * 1024**3:
        raise RuntimeError('Insufficient /dev/shm headroom for the 328 GiB object store')
    available = meminfo['MemAvailable']
    if available < size_bytes + 64 * 1024**3:
        raise RuntimeError('Insufficient host RAM headroom for the 328 GiB object store')
    cgroup = Path('/sys/fs/cgroup')
    maximum, current = cgroup/'memory.max', cgroup/'memory.current'
    cgroup_remaining = None
    if maximum.exists() and current.exists():
        limit = maximum.read_text().strip()
        if limit != 'max':
            cgroup_remaining = int(limit) - int(current.read_text())
            if cgroup_remaining < size_bytes + 64 * 1024**3:
                raise RuntimeError('Cgroup memory headroom cannot support the 328 GiB object store')
    return {'object_store_gib': OBJECT_STORE_GIB, 'object_store_bytes': size_bytes,
            'host_mem_available_bytes': available, 'shm_available_bytes': shm_available,
            'cgroup_remaining_bytes': cgroup_remaining}
