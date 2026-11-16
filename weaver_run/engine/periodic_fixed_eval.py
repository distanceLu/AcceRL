"""Finite validation through the existing, frozen Policy inference service.

The caller drains Real collection, pauses updates/publication, and temporarily
sets inference's training_batch_limit to one. Online Eval workers stay at zero. This
worker owns only a CPU simulator with EGL rendering and a processor: no model,
Replay handle, or training/evaluation statistics handle. Results are returned
only to the coordinator, which decides whether to persist a complete round.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid

from frozen_policy_eval import PROTOCOL as REFERENCE_PROTOCOL, digest, make_plan


PROTOCOL = 'accerl_periodic_fixed_policy_v1'


def fixed_plan():
    # make_plan owns the canonical state order and both seed derivations. These
    # internal descriptors merely satisfy its paired-snapshot API; they are
    # never exposed as checkpoint provenance or used to load any weights.
    descriptors = [dict(name=name, path='unused-case-template', sha256='0'*64, metadata={})
                   for name in ('baseline', 'final')]
    reference = make_plan(descriptors, modes=('greedy',))
    plan = {key: reference[key] for key in ('seed', 'initial_state_count', 'task_id',
             'modes', 'repeats', 'cases', 'horizon_action_steps', 'reset_settling_steps',
             'policy_updates_during_evaluation', 'inference_batch_size')}
    plan.update(protocol=PROTOCOL, reference_protocol=REFERENCE_PROTOCOL,
                benchmark='libero_spatial', action_seed_semantics='greedy_actions_do_not_depend_on_rng')
    plan['sha256'] = digest(plan)
    return plan


def _result(policy_version):
    if type(policy_version) is not int or policy_version < 0:
        raise ValueError('Fixed validation requires an explicit nonnegative Policy version')
    plan = fixed_plan()
    return dict(protocol=PROTOCOL, protocol_sha256=plan['sha256'], plan=plan,
                policy_version=policy_version, complete=False, num_cases=50,
                completed_cases=0, success_count=None, success_rate=None, action_steps=0,
                rows=[], errors=[], initial_state_pool=None, started_ns=time.time_ns())


def _remaining(deadline, clock, should_stop):
    if should_stop():
        raise InterruptedError('Fixed validation stopped before completing all 50 cases')
    remaining = deadline - clock()
    if remaining <= 0:
        raise TimeoutError('Fixed validation deadline reached')
    return remaining


def _timeout(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Validation timeout must be finite and positive')
    return float(value)


def _record_error(result, error):
    result.update(complete=False, success_count=None, success_rate=None)
    result['errors'].append({'type': type(error).__name__, 'error': str(error)})


def run_fixed_evaluation(env, request_actions, *, policy_version, timeout_seconds=3600,
                         expected_pool=None, should_stop=lambda: False, clock=time.monotonic):
    """Run all reference cases or return an explicitly incomplete result.

    request_actions(observation, case, seconds_remaining) returns the existing
    capture envelope with a tuple of five inference outputs. Every served
    Policy version is verified before applying any action. Partial rows remain
    diagnostic only: incomplete rounds have no success count or success rate.
    The caller owns the environment lifecycle and must enforce BATCH1.
    """
    import numpy as np
    from initial_state_env import initial_state_pool_metadata, reset_with_initial_state

    result = _result(policy_version)
    deadline = clock() + _timeout(timeout_seconds)
    try:
        _remaining(deadline, clock, should_stop)
        if (env.benchmark_name != 'libero_spatial' or env.task_id != 0
                or env.max_episode_steps != 220 or env.num_steps_wait != 10):
            raise ValueError('Environment differs from the fixed task0/220-action/10-reset protocol')
        pool = initial_state_pool_metadata(env)
        result['initial_state_pool'] = pool
        if pool['count'] != 50:
            raise ValueError('Fixed validation requires the full 50-state initial-state pool')
        if expected_pool is not None and any(expected_pool.get(key) != pool[key]
                                              for key in ('count', 'shape', 'sha256')):
            raise ValueError('Validation initial-state pool differs from training')
        for case in result['plan']['cases']:
            _remaining(deadline, clock, should_stop)
            observation, reset_info = reset_with_initial_state(env, case['initial_state_id'], case['env_seed'])
            started_ns = time.time_ns()
            steps, chunks, total_return, done = 0, 0, 0.0, False
            token_hash = hashlib.sha256()
            while not done:
                remaining = _remaining(deadline, clock, should_stop)
                answer = request_actions(observation, dict(case), remaining)
                _remaining(deadline, clock, should_stop)
                if not isinstance(answer, dict):
                    raise ValueError('Inference response lacks its served Policy version envelope')
                version = answer.get('versions', {}).get('policy')
                if type(version) is not int or version != policy_version:
                    raise ValueError('Served Policy version differs from the frozen validation version')
                payload = answer.get('_capture_payload')
                if not isinstance(payload, (list, tuple)) or len(payload) != 5:
                    raise ValueError('Malformed Policy action response')
                actions = np.asarray(payload[1])
                if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
                    raise ValueError('Policy must return a nonempty finite action chunk of shape [N,7]')
                token_hash.update(np.ascontiguousarray(payload[2]).tobytes())
                chunks += 1
                for action in actions:
                    _remaining(deadline, clock, should_stop)
                    if steps >= 220:
                        raise ValueError('Environment did not terminate at the 220-action horizon')
                    observation, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    result['action_steps'] += 1
                    total_return += float(reward)
                    if not math.isfinite(total_return):
                        raise ValueError('Environment returned a nonfinite episode return')
                    _remaining(deadline, clock, should_stop)
                    done = bool(terminated or truncated)
                    if done:
                        success = bool(info.get('is_success', False))
                        termination = 'success' if success else 'horizon' if truncated else 'terminated_unsuccessful'
                        if truncated and not terminated and steps != 220:
                            raise ValueError('Environment truncated before the full 220-action task horizon')
                        result['rows'].append(dict(case, policy_version=policy_version,
                            initial_state_hash=reset_info['initial_state_hash'], reset_env_steps=10,
                            steps=steps, success=success, episode_return=total_return, termination=termination,
                            action_chunks=chunks, action_tokens_sha256=token_hash.hexdigest(),
                            started_ns=started_ns, completed_ns=time.time_ns()))
                        result['completed_cases'] = len(result['rows'])
                        break
                    if steps == 220:
                        raise ValueError('Environment did not terminate at the 220-action horizon')
        _remaining(deadline, clock, should_stop)
        result.update(complete=True, success_count=sum(row['success'] for row in result['rows']))
        result['success_rate'] = result['success_count'] / 50
    except Exception as error:
        _record_error(result, error)
    result['finished_ns'] = time.time_ns()
    return result


class PeriodicFixedEvalWorker:
    """Single-use simulator actor; wrap with ray.remote(max_concurrency=2).

    Construct with existing render_worker_options(..., 'eval-1') (or another
    unused numeric label). run and stop may execute concurrently. Each remote
    wait polls once per second so stop is responsive even if inference hangs.
    """
    def __init__(self, infer, cfg, benchmark_name, torch_dtype):
        self.infer, self.cfg = infer, cfg
        self.benchmark_name, self.torch_dtype = benchmark_name, torch_dtype
        self._stop = threading.Event()
        self._run_lock = threading.Lock()
        self._used = False
        self.env = None

    def stop(self):
        self._stop.set()

    def _request(self, observation, case, remaining):
        import ray
        from rl.utils import prepare_one_obs

        deadline = time.monotonic() + remaining
        request_id = 'fixed-eval-' + uuid.uuid4().hex
        inputs = prepare_one_obs(self.cfg, self.processor, observation,
                                 self.env.task_description, self.torch_dtype)
        metadata = dict(request_id=request_id, submitted_ns=time.time_ns(), workflow='evaluation',
                        caller_role='periodic_fixed_evaluation', episode_id=case['pair_id'],
                        initial_state_id=case['initial_state_id'], deterministic=True)
        ref = self.infer.request.remote(inputs, deterministic=True, telemetry_meta=metadata)
        try:
            while True:
                timeout = min(1.0, _remaining(deadline, time.monotonic, self._stop.is_set))
                try:
                    answer = ray.get(ref, timeout=timeout)
                    break
                except ray.exceptions.GetTimeoutError:
                    continue
            if not isinstance(answer, dict) or answer.get('request_id') != request_id:
                raise ValueError('Inference receipt does not match the fixed-validation request')
            return answer
        except BaseException:
            # Cancel queued async action work on interruption; actor-side capture
            # hooks retain ownership of an action already being served.
            try:
                ray.cancel(ref, force=False)
            except Exception:
                pass
            raise

    def run(self, policy_version, timeout_seconds=3600, expected_pool=None):
        result = _result(policy_version)
        deadline = time.monotonic() + _timeout(timeout_seconds)
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError('Fixed evaluator already has an active round')
        try:
            if self._used:
                raise RuntimeError('Fixed evaluator actors are single-use')
            self._used = True
            _remaining(deadline, time.monotonic, self._stop.is_set)
            from environment_rendering import prepare_environment_renderer, verify_environment_renderer
            if prepare_environment_renderer() != 'egl':
                raise ValueError('Periodic fixed validation requires the allocated EGL renderer')
            from rl.libero_env import LiberoEnvWrapper
            from experiments.robot.sole_utils import get_processor
            self.env = LiberoEnvWrapper(benchmark_name=self.benchmark_name, task_id=0,
                                       image_size=224, render_mode='rgb_array')
            receipt = verify_environment_renderer()
            if receipt.get('backend') != 'egl' or receipt.get('cuda_visible_devices') != '':
                raise ValueError('Fixed evaluator must use verified EGL with CUDA hidden')
            result['renderer'] = receipt
            self.processor = get_processor(self.cfg)
            remaining = _remaining(deadline, time.monotonic, self._stop.is_set)
            result = run_fixed_evaluation(self.env, self._request, policy_version=policy_version,
                timeout_seconds=remaining, expected_pool=expected_pool, should_stop=self._stop.is_set)
            result['renderer'] = receipt
        except Exception as error:
            _record_error(result, error)
        finally:
            try:
                if self.env is not None:
                    self.env.close()
            except Exception as error:
                _record_error(result, error)
            finally:
                self.env = None
                self._run_lock.release()
        result['finished_ns'] = time.time_ns()
        return result

    def close(self):
        """Signal an active run; its finally block owns simulator destruction."""
        self.stop()
        return self.env is None
