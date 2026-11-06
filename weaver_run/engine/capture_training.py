"""Identity-preserving replay/training hooks; no changes to learning recipe."""
from __future__ import annotations

import functools
import random
import time
from collections import deque

try:
    from capture_telemetry import Telemetry, ReplayLedger, new_id, now_ns
except ImportError:
    from telemetry import Telemetry, ReplayLedger, new_id, now_ns


def _writer(actor, role="trainer"):
    for name in ("capture_telemetry", "telemetry", "capture"):
        candidate = getattr(actor, name, None)
        if candidate is not None and hasattr(candidate, "span") and hasattr(candidate, "event"):
            return candidate
    writer = Telemetry(role, getattr(actor, "rank", getattr(actor, "actor_id", "")))
    actor.capture_telemetry = writer
    return writer


def _identity(sample, prefix):
    sample_id = getattr(sample, "sample_id", None)
    if not sample_id:
        sample_id = new_id(prefix)
        sample.sample_id = sample_id
    return str(sample_id)


class TrackedDeque(deque):
    def __init__(self, iterable=(), maxlen=None, *, ledger):
        super().__init__((), maxlen=maxlen)
        self.ledger = ledger
        self.additions = 0
        self.extend(iterable)

    def append(self, sample):
        if self.maxlen is not None and len(self) >= self.maxlen:
            self.ledger.evict(_identity(self[0], self.ledger.replay), "capacity")
        sample_id = _identity(sample, self.ledger.replay)
        self.ledger.add(sample_id, getattr(sample, "episode_id", None),
                        getattr(sample, "sample_index", None),
                        model_versions=getattr(sample, "model_versions", None))
        super().append(sample)
        self.additions += 1

    def extend(self, samples):
        for sample in samples:
            self.append(sample)


class TrackedList(list):
    def __init__(self, iterable=(), *, ledger):
        super().__init__()
        self.ledger = ledger
        self.additions = 0
        self.extend(iterable)

    def append(self, sample):
        self.ledger.add(_identity(sample, self.ledger.replay), getattr(sample, "episode_id", None),
                        getattr(sample, "sample_index", None),
                        model_versions=getattr(sample, "model_versions", None))
        super().append(sample)
        self.additions += 1

    def extend(self, samples):
        for sample in samples:
            self.append(sample)


def _modified_class(ray_actor):
    metadata = getattr(ray_actor, "__ray_metadata__", None)
    return metadata.modified_class if metadata is not None else ray_actor


def _install_replay_hooks(namespace):
    np = namespace["np"]
    policy_cls = _modified_class(namespace["ReplayBufferActor"])
    wm_cls = _modified_class(namespace["WMReplayBufferActor"])
    if getattr(policy_cls, "_capture_replay_hooks_installed", False):
        return
    policy_init = policy_cls.__init__

    @functools.wraps(policy_init)
    def init_policy(self, capacity, *args, **kwargs):
        policy_init(self, capacity, *args, **kwargs)
        writer = _writer(self, "policy_replay")
        self.capture_ledger = ReplayLedger(writer, "policy", capacity)
        self.buffer = TrackedDeque(self.buffer, maxlen=capacity, ledger=self.capture_ledger)
        self.capture_add_calls = self.capture_sample_calls = 0

    policy_cls.__init__ = init_policy
    original_policy_add = policy_cls.add_batch

    @functools.wraps(original_policy_add)
    def add_policy(self, batch):
        count = len(batch)
        before = self.buffer.additions
        result = original_policy_add(self, batch)
        self.capture_add_calls += 1
        _writer(self).event("replay_add", actual_batch=self.buffer.additions - before,
                            replay="policy", offered_count=count)
        if self.capture_add_calls % 50 == 0:
            self.capture_ledger.snapshot()
        return result

    policy_cls.add_batch = add_policy
    original_policy_sample = policy_cls.sample

    async def sample_policy(self, batch_size):
        if getattr(self, 'fresh_enabled', False):
            # The FRESH implementation owns ordered segments, age/use leases,
            # current-critic inference and source quotas. Do not replace it with
            # the legacy uniform deque draw below.
            return await original_policy_sample(self, batch_size)
        with _writer(self).span("policy_replay_sample", actual_batch=batch_size):
            batch = random.sample(self.buffer, batch_size)
            result = ([b.obs for b in batch], np.stack([b.action_token for b in batch]),
                      np.asarray([b.advantage for b in batch], np.float32),
                      np.stack([b.behaviour_logits for b in batch]),
                      np.asarray([b.value_target for b in batch], np.float32))
            metadata = self.capture_ledger.reserve([b.sample_id for b in batch])
        self.capture_sample_calls += 1
        if self.capture_sample_calls % 20 == 0:
            self.capture_ledger.snapshot()
        return (*result, metadata)

    policy_cls.sample = sample_policy
    policy_cls._capture_replay_hooks_installed = True

    wm_init = wm_cls.__init__

    @functools.wraps(wm_init)
    def init_wm(self, capacity, *args, **kwargs):
        wm_init(self, capacity, *args, **kwargs)
        writer = _writer(self, "wm_replay")
        self.capture_ledger = ReplayLedger(writer, "wm", capacity)
        self.capture_eval_ledger = ReplayLedger(writer, "wm_eval", self.ctrl_eval_capacity)
        self.buffer = TrackedDeque(self.buffer, maxlen=capacity, ledger=self.capture_ledger)
        self.eval_buffer = TrackedList(self.eval_buffer, ledger=self.capture_eval_ledger)
        self.capture_add_calls = self.capture_sample_calls = 0

    wm_cls.__init__ = init_wm
    original_wm_add = wm_cls.add_batch

    @functools.wraps(original_wm_add)
    def add_wm(self, batch):
        before_train, before_eval = self.buffer.additions, self.eval_buffer.additions
        result = original_wm_add(self, batch)
        train_added = self.buffer.additions - before_train
        eval_added = self.eval_buffer.additions - before_eval
        self.capture_add_calls += 1
        _writer(self).event("replay_add", actual_batch=train_added, replay="wm",
                            offered_count=len(batch), train_added=train_added,
                            eval_added=eval_added, holdout_not_retained=len(batch) - train_added - eval_added)
        if self.capture_add_calls % 50 == 0:
            self.capture_ledger.snapshot()
            self.capture_eval_ledger.snapshot()
        return result

    wm_cls.add_batch = add_wm

    def sample_wm(self, batch_size):
        with _writer(self).span("wm_replay_sample", actual_batch=batch_size):
            batch = random.sample(self.buffer, batch_size)
            obs = np.stack([b.obs for b in batch])
            act = np.stack([b.act for b in batch])
            rew = np.asarray([b.rew for b in batch], np.float32)
            instructions = [b.instruction for b in batch]
            sampled_ids = {str(b.episode_id) for b in batch}
            new_ids = sampled_ids - self.sampled_episode_ids
            self.sampled_episode_ids.update(sampled_ids)
            stats = {"unique_episodes": len(sampled_ids), "fresh_episode_ratio": len(new_ids) / max(len(sampled_ids), 1),
                     "capture_meta": self.capture_ledger.reserve([b.sample_id for b in batch])}
            backend_samples = [getattr(b, "backend_payload", {}) for b in batch]
            if any(not isinstance(payload, dict) for payload in backend_samples):
                raise ValueError("WM backend payloads must be dictionaries")
            if any(backend_samples):
                # A reservation carries backend targets but does not record use.
                # Actual consumption remains recorded by micro_end/replay_used.
                stats["backend_samples"] = backend_samples
        self.capture_sample_calls += 1
        if self.capture_sample_calls % 20 == 0:
            self.capture_ledger.snapshot()
        return obs, act, rew, instructions, stats

    wm_cls.sample = sample_wm
    original_eval_sample = wm_cls.sample_eval

    @functools.wraps(original_eval_sample)
    def sample_eval(self, batch_size):
        result = original_eval_sample(self, batch_size)
        self.capture_eval_ledger.reserve([b.sample_id for b in self.eval_buffer[:batch_size]], consumer="ctrl_eval")
        # Sampling a holdout is a reservation, not proof that evaluation ran.
        return result

    wm_cls.sample_eval = sample_eval
    wm_cls._capture_replay_hooks_installed = True


def _install_trainer_hooks(namespace):
    trainer_cls = _modified_class(namespace["TrainerActor"])
    if getattr(trainer_cls, "_capture_training_hooks_installed", False):
        return
    original_setup = trainer_cls.setup_deepspeed_group

    @functools.wraps(original_setup)
    def setup(self, *args, **kwargs):
        _writer(self)
        if not hasattr(self, "capture_versions"):
            self.capture_versions = {kind: 0 for kind in getattr(self, "active_kinds", ("policy", "ctrl", "reward"))}
        result = original_setup(self, *args, **kwargs)
        if getattr(self, "training_role", None) == "policy":
            from policy_backend_audit import policy_zero2_report
            _writer(self).event("policy_zero2_verified",
                                details=policy_zero2_report(self.model, self.rank))
            _writer(self).flush()
        if getattr(self, "training_role", None) == "ctrl":
            from world_backend_audit import world_zero2_report
            _writer(self).event("world_zero2_verified",
                                details=world_zero2_report(self.ctrl_world_engine, self.rank))
            _writer(self).flush()
        owned_engines = [name for name, attr in (("policy", "model"),
                         ("ctrl", "ctrl_world_engine"), ("reward", "reward_engine"))
                         if getattr(self, attr, None) is not None]
        _writer(self).event("trainer_ready", model_version=self.capture_versions,
                           details={"training_role": getattr(self, "training_role", "combined"),
                                    "owned_engines": owned_engines})
        return result

    trainer_cls.setup_deepspeed_group = setup
    for name, workload, key, count_name, interval_name in (
        ("_train_policy", "policy_train", "policy", "super_batch_size", None),
        ("_train_ctrl_world", "ctrl_train", "ctrl", "ctrl_super_batch_size", "ctrl_train_interval"),
        ("_train_reward_model", "reward_train", "reward", "reward_super_batch_size", "reward_train_interval"),
    ):
        original = getattr(trainer_cls, name)

        def make_wrapper(original, workload, key, count_name, interval_name):
            @functools.wraps(original)
            def train(self, *args, **kwargs):
                scheduled = interval_name is None or self.global_step % getattr(self, interval_name) == 0
                if not scheduled:
                    return original(self, *args, **kwargs)
                versions = getattr(self, "capture_versions", {})
                training_role = getattr(self, "training_role", "combined")
                with _writer(self).span(workload + "_update", actual_batch=getattr(self, count_name),
                                        model_version=versions.get(key, 0), cuda=False,
                                        policy_step=getattr(self, "global_step", 0),
                                        training_role=training_role,
                                        step_scope=("logical_policy_boundary_in_progress"
                                                    if training_role in ("ctrl", "reward")
                                                    else "completed_policy_updates_before_update")):
                    return original(self, *args, **kwargs)
            return train

        setattr(trainer_cls, name, make_wrapper(original, workload, key, count_name, interval_name))
    trainer_cls._capture_training_hooks_installed = True


def install_training_hooks(namespace):
    """Invoke after Ray actor definitions and before main creates actors."""
    _install_replay_hooks(namespace)
    _install_trainer_hooks(namespace)


def micro_begin(self, workload, engine, actual_batch, start, end, sample_indices=None):
    writer = _writer(self)
    key = {"policy_train": "policy", "ctrl_train": "ctrl", "reward_train": "reward"}[workload]
    if not hasattr(self, "capture_versions"):
        self.capture_versions = {kind: 0 for kind in getattr(self, "active_kinds", ("policy", "ctrl", "reward"))}
    metadata = getattr(self, "capture_policy_metadata" if key == "policy" else "capture_wm_metadata", None)
    if not metadata or len(metadata.get("sample_ids", [])) < end:
        raise RuntimeError("%s microbatch has no complete replay identity metadata" % workload)
    indices = list(range(start, end)) if sample_indices is None else list(sample_indices)
    if (len(indices) != actual_batch or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(metadata['sample_ids']) for i in indices)):
        raise ValueError('Invalid reordered microbatch identities')
    policy_step = int(getattr(self, "global_step", 0))
    training_role = getattr(self, "training_role", "combined")
    step_scope = ("logical_policy_boundary_in_progress" if training_role in ("ctrl", "reward")
                  else "completed_policy_updates_before_microbatch")
    span = writer.span(workload, actual_batch=actual_batch, model_version=self.capture_versions[key], cuda=True,
                       policy_step=policy_step, training_role=training_role,
                       step_scope=step_scope,
                       microbatch_start=start, microbatch_end=end,
                        reordered_sample_indices=indices if sample_indices is not None else None,
                       reservation_id=metadata.get("reservation_id"))
    token = {"span": span, "writer": writer, "workload": workload, "key": key,
             "replay": "policy" if key == "policy" else "wm", "metadata": metadata,
             "sample_ids": [metadata['sample_ids'][i] for i in indices],
             "before_steps": int(getattr(engine, "global_steps", 0)),
             "before_skips": int(getattr(engine, "skipped_steps", 0)),
             "micro_id": new_id("micro"),
             "engine_micro_steps_before": int(getattr(engine, "micro_steps", 0)),
             "boundary_before": bool(engine.is_gradient_accumulation_boundary()),
             "policy_step": policy_step, "training_role": training_role, "step_scope": step_scope,
             "actual_batch": actual_batch}
    span.__enter__()
    return token


def micro_end(self, token, engine, error=None):
    writer, key = token["writer"], token["key"]
    global_delta = max(0, int(getattr(engine, "global_steps", token["before_steps"])) - token["before_steps"])
    skipped_delta = max(0, int(getattr(engine, "skipped_steps", token["before_skips"])) - token["before_skips"])
    success_delta = max(0, global_delta - skipped_delta)
    rank = int(getattr(self, "rank", 0))
    world_size = int(getattr(self, "world_size", 1))
    logical_delta = success_delta if rank == 0 else 0
    token["span"].details.update(optimizer_steps=global_delta, skipped_steps=skipped_delta,
                                successful_optimizer_steps=success_delta,
                                logical_optimizer_steps=logical_delta,
                                data_parallel_rank=rank, data_parallel_world_size=world_size,
                                accumulation_boundary=token["boundary_before"])
    token["span"].__exit__(type(error) if error else None, error, getattr(error, "__traceback__", None))
    if error is not None:
        return
    writer.replay_used(token["replay"], token["metadata"], workload=token["workload"],
                       sample_ids=token["sample_ids"], step=getattr(self, "global_step", 0),
                       model_version=self.capture_versions[key], use_id=token["micro_id"],
                       lineage_details={"data_parallel_rank": rank,
                           "engine_update_attempt": token["before_steps"] + 1,
                           "engine_micro_steps_before": token["engine_micro_steps_before"],
                           "target_model_version_if_success": self.capture_versions[key] + 1})
    writer.event("microstep", workload=token["workload"], actual_batch=token["actual_batch"],
                 model_version=self.capture_versions[key], engine_global_steps=getattr(engine, "global_steps", None),
                 policy_step=token["policy_step"], training_role=token["training_role"],
                 step_scope=token["step_scope"],
                 micro_id=token["micro_id"], engine_update_attempt=token["before_steps"] + 1,
                 optimizer_steps=global_delta, successful_optimizer_steps=success_delta,
                 logical_optimizer_steps=logical_delta,
                 data_parallel_rank=rank, data_parallel_world_size=world_size,
                 skipped_steps=skipped_delta, boundary_before=token["boundary_before"])
    if global_delta:
        self.capture_versions[key] += success_delta
        # train.py advances the Policy owner's counter immediately after this
        # hook. Update events label the completed update; microbatch records
        # retain the counter at their start. WM updates do not advance Policy.
        completed_policy_step = int(getattr(self, "global_step", 0)) + (success_delta if key == "policy" else 0)
        # Replicated rank updates are real, but describe the same global update.
        # Keep rank 0 as optimizer_update so counting that event cannot double it.
        writer.event("optimizer_update" if rank == 0 else "optimizer_update_replica", workload=token["workload"],
                     actual_batch=int(getattr(engine, "train_batch_size", lambda: token["actual_batch"])()),
                     model_version=self.capture_versions[key], step=completed_policy_step,
                     training_role=token["training_role"],
                     step_scope=("logical_policy_boundary_in_progress"
                                 if token["training_role"] in ("ctrl", "reward")
                                 else "completed_policy_updates"),
                     engine_global_steps=getattr(engine, "global_steps", None),
                     engine_update_attempt=token["before_steps"] + 1,
                     data_parallel_rank=rank, data_parallel_world_size=world_size,
                     logical_optimizer_steps=logical_delta,
                     local_batch_size=int(getattr(engine, "train_batch_size", lambda: token["actual_batch"])()) // world_size,
                     successful_optimizer_steps=success_delta, skipped_steps=skipped_delta,
                     status="skipped" if success_delta == 0 else "ok")
        # Completion of every real update is durable before unattended stop.
        writer.flush()
