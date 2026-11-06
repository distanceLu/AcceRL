"""Application admission, all-rank microstep grants and priority publication.

The pure async engine is CPU-testable. Ray wraps the same engine in one CPU
actor. No SQLite call is on the permit critical path; its sink owns a thread.
"""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
import math
import os
import queue
import statistics
import threading
import time
import uuid

from resource_controller import ResourceController
from resource_state import CostTable, FlowWindow, ReplayReader, SERVICES


class EventSink:
    def __init__(self):
        self.queue = queue.SimpleQueue()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="governor-events", daemon=True)
        self.thread.start()

    def _run(self):
        from capture_telemetry import Telemetry
        writer = Telemetry("resource_governor", "global", flush_records=512)
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                kind, service, details = item
                writer.event(kind, workload=service, details=details)
            writer.close()
        except BaseException as exc:
            self.error = repr(exc)

    def __call__(self, kind, service=None, **details):
        self.queue.put((kind, service, details))


class GovernorEngine:
    def __init__(self, cards, *, clock=time.monotonic, emit=None):
        self.cards = list(cards)
        self.n = len(cards)
        if self.n not in (4, 8) or len(set(cards)) != self.n:
            raise ValueError("The fixed DP group must contain 4 or 8 distinct cards")
        self.clock = clock
        self.emit = emit or (lambda *a, **k: None)
        self.costs = CostTable()
        self.flows = FlowWindow()
        self.controller = ResourceController()
        self.plan = self.controller.decide(ready=False, costs={}, flows={}, pressure={}, publication_lag=False, now=clock())
        self.balances = {card: {s: 0. for s in ("training", *SERVICES[3:])} for card in cards}
        self.debts = {card: dict.fromkeys(self.balances[card], 0.) for card in cards}
        self.last_credit = clock()
        self.pending = {}
        self.active = {}
        self.training_pending = None
        self.training_groups = {}
        self.training_active = None
        self.paused = set()
        self.free_bytes = {card: None for card in cards}
        self.reserve_bytes = 6 * 2**30
        self.ready = False
        self.calibration_started = None
        self.calibration_execution = dict.fromkeys(cards, 0.)
        self.credit_rate = 1.
        self.quota_enforced = False
        self.failure = None
        self.checkpoint_paused = False
        self.sequence = 0
        self.memory_generation = 0
        self.full_cost_accumulator = defaultdict(float)
        self.statistics = Counter()

    def _check(self):
        if self.failure:
            raise RuntimeError("Governor stopped: " + self.failure)

    def pressure(self):
        result = dict.fromkeys(SERVICES[3:], 0.)
        for request in self.pending.values():
            age_pressure = 1 + min(4., max(0., self.clock() - request["submitted"]) / 5)
            result[request["service"]] += request["cost"] * max(1, request["backlog"] / request["batch"]) * age_pressure
        result["training"] = float(bool(self.training_groups) or bool(self.training_active))
        return result

    def decide(self):
        self.costs.freeze()
        self.plan = self.controller.decide(ready=self.ready, costs=self.costs.update_costs(), flows=self.flows.rates(),
                                           pressure=self.pressure(), publication_lag=bool(self.paused), now=self.clock())
        if self.plan["stage"] == "adaptive" and not self.quota_enforced:
            elapsed = self.clock() - self.calibration_started if self.calibration_started is not None else 0
            if elapsed > 0:
                self.credit_rate = min(4., max(1., statistics.median(self.calibration_execution.values()) / elapsed))
                self.quota_enforced = True
        self.plan.update(quota_enforced=self.quota_enforced, reference_seconds_per_wall_second=self.credit_rate,
                         capacity_scope="same-run initial completed actor execution seconds / wall; concurrency proxy, not SM occupancy")
        self.emit("resource_plan", **self.plan, cost_table_version=self.costs.version,
                  reference_costs=self.costs.frozen)

    def _credit(self):
        now = self.clock(); elapsed = max(0., now - self.last_credit); self.last_credit = now
        rates = {"training": self.plan["alpha"], **{s: self.plan["weights"][s] for s in SERVICES[3:]}}
        for card in self.cards:
            for key, rate in rates.items():
                amount = elapsed * rate * self.credit_rate
                repaid = min(self.debts[card][key], amount)
                self.debts[card][key] -= repaid
                costs = [g["cost"] for g in self.training_groups.values()] if key == "training" else [r["cost"] for r in self.pending.values() if r["service"] == key]
                cap = max(12., 2 * max(costs, default=0))
                self.balances[card][key] = min(cap, self.balances[card][key] + amount - repaid)

    def _demanded(self, card):
        wanted = {r["service"] for r in self.pending.values() if r["card"] == card and r["service"] not in self.paused}
        wanted.update(r["service"] for r in self.active.values() if r["card"] == card)
        if self.training_groups or self.training_active:
            wanted.add("training")
        return wanted

    def _payment(self, card, key, cost):
        if not self.quota_enforced:
            return (0., [])
        balance = self.balances[card]
        gap = max(0., cost - balance[key])
        donors = [k for k in balance if k != key and k not in self._demanded(card)]
        available = sum(balance[k] for k in donors)
        if gap > available + 1e-8 or self.debts[card][key] + gap > max(4., 2 * cost):
            return None
        return gap, donors

    def _pay(self, card, key, cost, payment):
        if not self.quota_enforced:
            return 0.
        gap, donors = payment
        self.balances[card][key] = max(0., self.balances[card][key] - cost)
        self.debts[card][key] += gap
        remaining = gap
        for donor in donors:
            borrowed = min(remaining, self.balances[card][donor])
            self.balances[card][donor] -= borrowed; remaining -= borrowed
        return gap

    def _memory_ok(self, card, estimate):
        free = self.free_bytes[card]
        # Reserve expected increments between memory samples. A fresh NVML
        # sample already contains allocations of previously granted contexts.
        reserved = sum(r["memory_bytes"] for r in self.active.values()
                       if r["card"] == card and r["memory_generation"] == self.memory_generation)
        if self.training_active and self.training_active["memory_generation"] == self.memory_generation:
            reserved += self.training_active["memory_bytes"]
        return free is not None and free >= self.reserve_bytes + estimate + reserved

    def _grant(self, request, borrowed):
        token = {k: v for k, v in request.items() if k != "future"}
        token.update(token=uuid.uuid4().hex, granted=self.clock(), plan_version=self.plan["version"],
                     cost_table_version=request["reference_cost_version"], borrowed=borrowed, memory_generation=self.memory_generation)
        self.active[token["token"]] = token
        self.emit("resource_grant", **token, wait_seconds=token["granted"] - token["submitted"])
        request["future"].set_result({k: v for k, v in token.items() if k != "service"})

    def pump(self):
        self._check(); self._credit()
        if self.checkpoint_paused:
            return
        ready = [g for g in self.training_groups.values() if len(g["requests"]) == self.n]
        group = min(ready, key=lambda g: g["submitted"]) if ready else None
        if group and not self.training_active:
            payments = [self._payment(card, "training", group["cost"]) for card in self.cards]
            if all(p is not None for p in payments) and all(self._memory_ok(card, group["memory_bytes"]) for card in self.cards):
                borrowed = [self._pay(card, "training", group["cost"], p) for card, p in zip(self.cards, payments)]
                group.update(granted=self.clock(), token=uuid.uuid4().hex, finished={}, borrowed=borrowed,
                             plan_version=self.plan["version"], cost_table_version=group["reference_cost_version"],
                             memory_generation=self.memory_generation)
                self.training_active = group; del self.training_groups[group["key"]]
                self.emit("resource_training_joint_grant", group["service"], key=group["key"], token=group["token"],
                          ranks=self.n, plan_version=group["plan_version"], estimated_cost=group["cost"], borrowed=borrowed,
                          memory_estimated_increment_bytes=group["memory_bytes"], memory_generation=self.memory_generation,
                          rank_waits={str(rank): group["granted"] - r["submitted"] for rank, r in group["requests"].items()})
                for rank, request in group["requests"].items():
                    request["future"].set_result({k: group[k] for k in ("token", "cost", "granted", "plan_version", "cost_table_version")})
        for owner, request in sorted(list(self.pending.items()), key=lambda x: x[1]["submitted"]):
            if request["service"] in self.paused:
                continue
            payment = self._payment(request["card"], request["service"], request["cost"])
            if payment is not None and self._memory_ok(request["card"], request["memory_bytes"]):
                borrowed = self._pay(request["card"], request["service"], request["cost"], payment)
                self._grant(request, borrowed); del self.pending[owner]

    async def acquire_inference(self, service, rank, operation, batch, backlog=1, sources=None):
        self._check()
        if service not in SERVICES[3:] or not 0 <= int(rank) < self.n:
            raise ValueError("Invalid inference service/rank")
        owner = (service, int(rank))
        if owner in self.pending or any((r["service"], r["rank"]) == owner for r in self.active.values()):
            raise RuntimeError("An inference actor may have only one in-flight GPU unit")
        request = dict(service=service, rank=int(rank), card=self.cards[int(rank)], operation=operation,
                       batch=int(batch), backlog=int(backlog), sources=sources or {}, submitted=self.clock(),
                       cost=self.costs.estimate(service, operation, batch), memory_bytes=3 * 2**30,
                       reference_cost_version=self.costs.version,
                       future=asyncio.get_running_loop().create_future())
        self.pending[owner] = request
        self.emit("resource_request", service, rank=rank, operation=operation, actual_batch=batch,
                  backlog=backlog, sources=sources, estimated_cost=request["cost"])
        try:
            return await self._wait(request["future"])
        except BaseException:
            self.pending.pop(owner, None)
            self.abort("inference admission timed out/cancelled")
            raise

    async def acquire_training(self, service, rank, key, batch, last_micro=False, operation="microstep"):
        self._check()
        if service not in SERVICES[:3] or not 0 <= int(rank) < self.n:
            raise ValueError("Invalid training service/rank")
        key = tuple(key)
        if key not in self.training_groups:
            estimate = 4 if operation == "prepare" else 1 if operation in ("adv_stats", "verify") else (40 if service == "ctrl_train" else 36 if service == "policy_train" else 12)
            self.training_groups[key] = dict(service=service, key=key, batch=int(batch), last_micro=bool(last_micro),
                                         operation=operation, cost=self.costs.estimate(service, operation, batch),
                                         reference_cost_version=self.costs.version,
                                         memory_bytes=estimate * 2**30, submitted=self.clock(), requests={})
        group = self.training_groups[key]
        if (group["service"], group["key"], group["batch"], group["last_micro"]) != (service, key, int(batch), bool(last_micro)):
            self.abort("rank microstep identity mismatch")
            raise RuntimeError(self.failure)
        if int(rank) in group["requests"]:
            raise RuntimeError("Duplicate rank in joint grant")
        request = {"submitted": self.clock(), "future": asyncio.get_running_loop().create_future()}
        group["requests"][int(rank)] = request
        try:
            return await self._wait(request["future"])
        except BaseException:
            self.abort("training joint admission timed out/cancelled")
            raise

    async def _wait(self, future):
        elapsed = 0.; previous = self.clock()
        while not future.done():
            self._check()
            await asyncio.sleep(.01)
            now = self.clock()
            if not self.checkpoint_paused:
                elapsed += max(0., now - previous)
            previous = now
            if elapsed > 180:
                raise TimeoutError("GPU admission exceeded 180 active seconds")
        return await future

    async def begin_checkpoint(self):
        self._check(); self.checkpoint_paused = True
        started = self.clock()
        while self.active or self.training_active:
            self._check()
            if self.clock() - started > 180:
                raise TimeoutError("GPU units failed to drain for recovery checkpoint")
            await asyncio.sleep(.01)
        return self.snapshot()

    def end_checkpoint(self):
        self.checkpoint_paused = False
        return True

    def complete_inference(self, token, seconds, model_version, error=None):
        self._check()
        request = self.active.pop(token)
        if error:
            self.abort("inference GPU unit failed: " + error)
        else:
            self.costs.observe(request["service"], request["operation"], request["batch"], seconds)
            if self.calibration_started is not None and request["granted"] >= self.calibration_started and not self.quota_enforced:
                self.calibration_execution[request["card"]] += seconds
        self.statistics[request["service"] + "/completed"] += 1
        self.statistics[request["service"] + "/samples"] += request["batch"]
        self.emit("resource_complete", request["service"], token=token, seconds=seconds,
                  operation=request["operation"], actual_batch=request["batch"], rank=request["rank"],
                  model_version=model_version, reference_cost=request["cost"], plan_version=request["plan_version"], error=error)

    def complete_training(self, token, rank, seconds, success, skipped, error=None):
        self._check(); group = self.training_active
        if not group or token != group["token"] or int(rank) in group["finished"]:
            raise RuntimeError("Unknown/duplicate training completion")
        group["finished"][int(rank)] = dict(seconds=seconds, success=int(success), skipped=int(skipped))
        if error or skipped:
            self.abort("training failure or skipped optimizer update: " + str(error))
            raise RuntimeError(self.failure)
        if len(group["finished"]) == self.n:
            if len({(v["success"], v["skipped"]) for v in group["finished"].values()}) != 1:
                self.abort("optimizer update disagreement across ranks")
                raise RuntimeError(self.failure)
            execution = max(v["seconds"] for v in group["finished"].values())
            if self.calibration_started is not None and group["granted"] >= self.calibration_started and not self.quota_enforced:
                for rank, result in group["finished"].items():
                    self.calibration_execution[self.cards[rank]] += result["seconds"]
            self.costs.observe(group["service"], group["operation"], group["batch"], execution)
            if group["operation"] == "microstep":
                self.full_cost_accumulator[group["service"]] += execution
            if group["last_micro"]:
                if success != 1:
                    self.abort("final accumulation microstep did not update optimizer")
                    raise RuntimeError(self.failure)
                self.costs.full_updates[group["service"]].append(self.full_cost_accumulator.pop(group["service"]))
            self.emit("resource_training_complete", group["service"], token=token, key=group["key"],
                      ranks=group["finished"], execution_seconds=execution, actual_global_batch=group["batch"] * self.n,
                      consumes_training_samples=(group["operation"] == "microstep"),
                      reference_cost=group["cost"], last_micro=group["last_micro"], operation=group["operation"], plan_version=group["plan_version"])
            self.statistics[group["service"] + "/" + group["operation"]] += 1
            self.training_active = None

    async def begin_sync(self, service):
        self._check()
        if service not in SERVICES[3:]:
            raise ValueError("Invalid publication service")
        self.paused.add(service)
        start = self.clock()
        while any(r["service"] == service for r in self.active.values()):
            self._check()
            if self.clock() - start > 180:
                raise TimeoutError("Inference drain before publication timed out")
            await asyncio.sleep(.01)
        self.emit("resource_sync_drained", service, seconds=self.clock() - start)

    def end_sync(self, service, versions):
        if len(versions) != self.n or len(set(versions)) != 1:
            self.abort("publication version ACK mismatch")
            raise RuntimeError(self.failure)
        self.emit("resource_sync_ack", service, versions=versions)
        self.paused.remove(service)

    def abort(self, reason):
        if self.failure:
            return
        self.failure = reason
        requests = list(self.pending.values()) + [r for g in self.training_groups.values() for r in g["requests"].values()]
        for request in requests:
            if not request["future"].done():
                request["future"].set_exception(RuntimeError(reason))
        self.emit("resource_failure", reason=reason)

    def snapshot(self):
        return {"plan": self.plan, "controller": self.controller.state_dict(), "costs": self.costs.state_dict(),
                "balances": {k: dict(v) for k, v in self.balances.items()}, "borrow_debts": {k: dict(v) for k, v in self.debts.items()}, "pressure": self.pressure(),
                "quota_enforced": self.quota_enforced, "credit_rate": self.credit_rate,
                "free_bytes": dict(self.free_bytes), "gpu_admission_reserve_bytes": self.reserve_bytes,
                "memory_scope": "whole-card NVML free, expected increments reserved until next sample; bounded actor concurrency; not per-model attribution",
                "statistics": dict(self.statistics), "pending_inference": len(self.pending),
                "training_pending_ranks": {str(k): len(g["requests"]) for k, g in self.training_groups.items()},
                "training_in_flight": bool(self.training_active), "paused": sorted(self.paused), "failure": self.failure}

    def restore_snapshot(self, state):
        if self.active or self.training_active or self.pending or self.training_groups:
            raise RuntimeError("Restore scheduler before admitting new work")
        if set(state["balances"]) != set(self.cards):
            raise ValueError("Recovery GPU/card count differs")
        for name, values in state["costs"]["samples"].items(): self.costs.samples[name] = deque(values, maxlen=128)
        for name, values in state["costs"]["full_updates"].items(): self.costs.full_updates[name] = deque(values, maxlen=16)
        self.costs.frozen = state["costs"]["frozen"]; self.costs.version = state["costs"]["version"]
        for name, value in state["controller"].items(): setattr(self.controller, name, value)
        self.controller.last_change = self.clock()  # Rebase the hold interval to this process clock.
        self.plan = state["plan"]; self.balances = state["balances"]; self.debts = state["borrow_debts"]
        self.statistics = Counter(state["statistics"]); self.ready = True
        self.quota_enforced = state.get("quota_enforced", True); self.credit_rate = state.get("credit_rate", 1.)
        self.last_credit = self.clock(); self.calibration_started = self.clock()
        self.emit("resource_scheduler_restored", plan_version=self.plan["version"], prefetched_work_discarded=True)
        return True


class ResourceGovernor(GovernorEngine):
    """One async Ray CPU actor; owns no CUDA context and selects only allowed UUIDs."""
    def __init__(self, bindings):
        self.sink = EventSink()
        super().__init__([b["index"] for b in bindings], emit=self.sink)
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handles = {b["index"]: pynvml.nvmlDeviceGetHandleByUUID(b["uuid"]) for b in bindings}
        self.reader = ReplayReader(os.environ["ACCERL_TELEMETRY_DB"], os.environ["ACCERL_RUN_ID"])
        self.last_read = self.last_decision = -1e30
        self.task = asyncio.get_event_loop().create_task(self._loop())

    def _read_memory(self):
        return {card: int(self.nvml.nvmlDeviceGetMemoryInfo(handle).free) for card, handle in self.handles.items()}

    async def _loop(self):
        try:
            last_memory = -1e30
            while not self.failure:
                now = self.clock()
                if self.sink.error:
                    raise RuntimeError(self.sink.error)
                if now - last_memory >= 1:
                    self.free_bytes = await asyncio.to_thread(self._read_memory)
                    self.memory_generation += 1
                    last_memory = now
                if now - self.last_read >= 5:
                    observation = await asyncio.to_thread(self.reader.read)
                    self.flows.add(observation)
                    self.emit("resource_state", observation=observation, state=self.snapshot())
                    self.last_read = now
                if now - self.last_decision >= 60:
                    self.decide(); self.last_decision = now
                self.pump()
                await asyncio.sleep(.01)
        except BaseException as exc:
            self.abort(repr(exc))

    def mark_ready(self):
        self.ready = True
        self.calibration_started = self.clock()
        self.calibration_execution = dict.fromkeys(self.cards, 0.)
        self.flows = FlowWindow()
        self.decide(); self.last_decision = self.clock()
        return self.plan

    async def health(self):
        self._check()
        return self.snapshot()


class ResourceClientMixin:
    def configure_resource_governor(self, governor):
        self.resource_governor = governor
        self._resource_gpu_lock = asyncio.Lock()
        return True

    @asynccontextmanager
    async def resource_unit(self, service, operation, batch, **kwargs):
        if getattr(self, "resource_governor", None) is None:
            yield
            return
        # Inference batch, latent init and fixed WM validation share one local
        # permit. Trainers use all-rank cohorts; a local lock would invert the
        # order of background prefetch and foreground collective cohorts.
        if service.endswith("_infer"):
            async with self._resource_gpu_lock:
                async with self._resource_unit(service, operation, batch, **kwargs):
                    yield
        else:
            async with self._resource_unit(service, operation, batch, **kwargs):
                yield

    @asynccontextmanager
    async def _resource_unit(self, service, operation, batch, *, micro_index=None, engine=None, last_micro=False, unit_key=None):
        governor = getattr(self, "resource_governor", None)
        if governor is None:
            yield
            return
        rank = int(getattr(self, "rank", getattr(self, "actor_id", 0)))
        before = (int(engine.global_steps), int(engine.skipped_steps)) if engine is not None else None
        if engine is not None:
            key = (service, operation, str(unit_key)) if unit_key is not None else (service, operation, int(self.global_step), before[0], int(micro_index))
            grant = await governor.acquire_training.remote(service, rank, key, int(batch), bool(last_micro), operation)
            before = (int(engine.global_steps), int(engine.skipped_steps))
        else:
            pending = list(getattr(self, "capture_pending_meta", []))[:int(batch)]
            sources = (dict(Counter(str(x.get("caller_role", "unknown")) for x in pending)) if operation == "predict"
                       else {"rollout": int(batch)} if operation == "init_latent_state"
                       else {"wm_fixed_evaluation": int(batch)})
            grant = await governor.acquire_inference.remote(service, rank, operation, int(batch),
                                                            len(getattr(self, "requests", [])) + int(batch), sources)
        start = time.perf_counter()
        error = None
        try:
            yield
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            # Every client context's submitted compute/copies/collectives finish
            # before its next grant. We preserve DS forward/backward/step calls.
            import torch
            try:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if engine is not None:
                    skipped = int(engine.skipped_steps) - before[1]
                    success = int(engine.global_steps) - before[0] - skipped
                    await governor.complete_training.remote(grant["token"], rank, elapsed, success, skipped, error)
                else:
                    await governor.complete_inference.remote(grant["token"], elapsed,
                                                              int(getattr(self, "capture_model_version", -1)), error)
            except BaseException:
                governor.abort.remote("GPU completion/ACK failed")
                raise
