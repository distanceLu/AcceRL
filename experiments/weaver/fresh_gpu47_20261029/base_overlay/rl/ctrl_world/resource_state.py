"""CPU-only observations and frozen reference-cost tables for the v0 governor.

Reference costs are wall-time proxies for completed GPU work, excluding permit
waits. They are not exclusive SM seconds. SQLite is read by the background
collector; admission never queries or writes the database synchronously.
"""
from __future__ import annotations

from collections import defaultdict, deque
import json
import math
import sqlite3
import statistics
import time

SERVICES = ("policy_train", "ctrl_train", "reward_train", "policy_infer", "ctrl_infer", "reward_infer")
DEFAULT_COSTS = {"policy_train": 3.5, "ctrl_train": 3.7, "reward_train": 2.0,
                 "policy_infer": .65, "ctrl_infer": 2.0, "reward_infer": .15}
TRAIN_BATCH = {"policy_train": 8, "ctrl_train": 4, "reward_train": 32}


def cost_key(service, operation, batch):
    if service not in SERVICES or int(batch) < 1:
        raise ValueError("Unknown service or invalid actual batch")
    return "%s/%s/%d" % (service, operation, int(batch))


class CostTable:
    def __init__(self):
        self.samples = defaultdict(lambda: deque(maxlen=128))
        self.frozen = {}
        self.version = 0
        self.full_updates = defaultdict(lambda: deque(maxlen=16))

    def estimate(self, service, operation, batch):
        key = cost_key(service, operation, batch)
        if key in self.frozen:
            return self.frozen[key]
        if service.endswith("_train") and operation != "microstep":
            return {"prepare": 13. if service == "ctrl_train" else .15,
                    "adv_stats": .01, "verify": .05}.get(operation, 1.)
        scale = max(1.0, int(batch) / TRAIN_BATCH.get(service, 1))
        return DEFAULT_COSTS[service] * scale

    def observe(self, service, operation, batch, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Execution cost must be finite and positive")
        self.samples[cost_key(service, operation, batch)].append(float(seconds))

    def freeze(self):
        self.frozen = {k: max(.001, statistics.median(v)) for k, v in self.samples.items() if v}
        self.version += 1

    def update_costs(self):
        # Actual sums over a completed optimizer update, never microstep median * GAS.
        return {k: statistics.median(v) for k, v in self.full_updates.items() if v}

    def state_dict(self):
        return {"samples": {k: list(v) for k, v in self.samples.items()}, "frozen": self.frozen,
                "version": self.version, "full_updates": {k: list(v) for k, v in self.full_updates.items()}}


class ReplayReader:
    def __init__(self, path, run_id):
        self.path, self.run_id = str(path), run_id

    def read(self):
        result = {"observed_monotonic": time.monotonic(), "replays": {}, "errors": []}
        try:
            with sqlite3.connect("file:" + self.path + "?mode=ro", uri=True, timeout=.2) as c:
                for name in ("policy", "wm"):
                    row = c.execute("SELECT COUNT(*),SUM(resident),SUM(use_count),SUM(policy_uses),"
                                    "SUM(ctrl_uses),SUM(reward_uses),SUM(CASE WHEN use_count=0 AND "
                                    "eviction_reason='capacity' THEN 1 ELSE 0 END) FROM replay_samples "
                                    "WHERE run_id=? AND replay=?", (self.run_id, name)).fetchone()
                    result["replays"][name] = dict(zip(
                        ("added", "resident", "uses", "policy_uses", "ctrl_uses", "reward_uses", "unused_evicted"),
                        (int(v or 0) for v in row)))
                    age = c.execute("SELECT AVG((last_used_ns-added_ns)/1e9),COUNT(*) FROM replay_samples "
                                    "WHERE run_id=? AND replay=? AND last_used_ns>?",
                                    (self.run_id, name, time.time_ns() - 180_000_000_000)).fetchone()
                    result["replays"][name]["insertion_age_seconds"] = age[0]
                    result["replays"][name]["age_scope"] = "last-use minus Replay insertion; not exact generation/version age"
                    episodes = c.execute("SELECT COUNT(DISTINCT episode_id) FROM replay_samples WHERE run_id=? AND replay=?",
                                         (self.run_id, name)).fetchone()[0]
                    result["replays"][name]["episodes"] = int(episodes)
                latest = c.execute("SELECT model_version FROM events WHERE run_id=? AND workload='policy_train' "
                                   "AND kind='optimizer_update' ORDER BY ts_ns DESC LIMIT 1", (self.run_id,)).fetchone()
                if latest:
                    current = int(latest[0])
                    ages = []
                    for versions, in c.execute("SELECT versions_json FROM replay_samples WHERE run_id=? AND replay='policy' "
                                               "AND last_used_ns>?", (self.run_id, time.time_ns() - 180_000_000_000)):
                        version = json.loads(versions or "{}").get("policy")
                        if isinstance(version, int) and 0 <= version <= current:
                            ages.append(current - version)
                    result["replays"]["policy"]["policy_version_age"] = statistics.mean(ages) if ages else None
                    result["replays"]["policy"]["version_age_scope"] = "recently-used identities, exact behavior Policy version; unweighted by repeated exposure"
                result["wm_window_version_age"] = "unresolved; episode-end versions are not exact window lineage"
        except sqlite3.Error as exc:
            result["errors"].append(repr(exc))
        return result


class FlowWindow:
    def __init__(self):
        self.history = deque(maxlen=80)

    def add(self, observation):
        if not observation.get("errors") and len(observation.get("replays", {})) == 2:
            self.history.append(observation)

    def rates(self):
        if len(self.history) < 2:
            return {}
        last = self.history[-1]
        candidates = [x for x in self.history if last["observed_monotonic"] - x["observed_monotonic"] <= 180]
        first = candidates[0]
        elapsed = last["observed_monotonic"] - first["observed_monotonic"]
        if elapsed < 30:
            return {}
        result = {"seconds": elapsed}
        for name in ("policy", "wm"):
            old, new = first["replays"][name], last["replays"][name]
            added = max(0, new["added"] - old["added"])
            uses = max(0, new["uses"] - old["uses"])
            result[name] = {"new_per_second": added / elapsed, "uses_per_second": uses / elapsed,
                            "reuse_intensity": uses / max(added, 1),
                            "new_episodes_per_second": max(0, new.get("episodes", 0) - old.get("episodes", 0)) / elapsed,
                            "policy_version_age": new.get("policy_version_age"),
                            "insertion_age_seconds": new.get("insertion_age_seconds"),
                            "resident": new["resident"]}
        return result
