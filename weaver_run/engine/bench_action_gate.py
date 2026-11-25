"""Single-host, expiring admission pulse for new real actions in the P189 benchmark."""
import json
import math
from pathlib import Path
import time

from frozen_policy_eval import atomic_json


def write_rollout_pulse(output, arm_id, seconds):
    if not arm_id or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('A rollout pulse needs an arm identifier and a positive duration')
    duration_ns = round(seconds * 1e9)
    issued_ns = time.time_ns()
    pulse = dict(arm_id=arm_id, issued_ns=issued_ns, until_ns=issued_ns + duration_ns,
                 until_monotonic_ns=time.monotonic_ns() + duration_ns)
    atomic_json(Path(output) / 'rollout_pulse.json', pulse)
    return pulse


class RolloutPulseGate:
    def __init__(self, output):
        self.path = Path(output) / 'rollout_pulse.json'

    @classmethod
    def from_environment(cls, env, worker_id=None):
        plan_path = env.get('ACCERL_JIT_BENCH_PLAN')
        if plan_path:
            comparison = json.loads(Path(plan_path).read_bytes()).get('comparison')
            if comparison == 'admission_pulse':
                return cls(env['ACCERL_JIT_BENCH_OUTPUT'])
            if comparison == 'critical_jit':
                from critical_probe import TargetActionGate
                return TargetActionGate(env['ACCERL_JIT_BENCH_OUTPUT'], worker_id)
        return None

    def wait(self):
        try:
            pulse = json.loads(self.path.read_bytes())
        except FileNotFoundError:
            return None, 0
        # 四个采集进程共用本机单调时钟；只等同一截止点，驱动退出也能自行放行。
        start = time.monotonic_ns()
        remaining = pulse['until_monotonic_ns'] - start
        if remaining <= 0:
            return None, 0
        time.sleep(remaining / 1e9)
        return pulse['arm_id'], time.monotonic_ns() - start
