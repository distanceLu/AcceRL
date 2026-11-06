"""Explainable 60-second feedback controller; no success-rate-driven budget edits."""
from __future__ import annotations

import time
from resource_state import SERVICES


class ResourceController:
    def __init__(self):
        self.alpha = .60
        self.version = 0
        self.stage = "fill"
        self.baseline = None
        self.last_change = -1e30
        self.direction = None
        self.streak = 0
        self.last_plan = None

    def decide(self, *, ready, costs, flows, pressure, publication_lag, now=None):
        now = time.monotonic() if now is None else now
        previous_stage = self.stage
        calibrated = all(k in costs for k in ("policy_train", "ctrl_train", "reward_train"))
        flowing = all(flows.get(name, {}).get("new_per_second", 0) > 0
                      and flows.get(name, {}).get("uses_per_second", 0) > 0 for name in ("policy", "wm"))
        self.stage = "adaptive" if ready and calibrated and (self.baseline is not None or flowing) and flows.get("seconds", 0) >= 150 else ("online_initialization" if ready else "fill")
        reason = "collect same-run execution/flow reference"
        if self.stage == "adaptive":
            if self.baseline is None:
                self.baseline = {name: dict(flows[name]) for name in ("policy", "wm")}
                reason = "established same-run stable flow reference"
            else:
                stressed = any(flows[name]["new_per_second"] < .8 * self.baseline[name]["new_per_second"]
                               and flows[name]["reuse_intensity"] > 1.25 * max(self.baseline[name]["reuse_intensity"], 1)
                               for name in ("policy", "wm"))
                age = flows["policy"].get("policy_version_age")
                reference_age = self.baseline["policy"].get("policy_version_age")
                if age is not None and reference_age is not None:
                    stressed = stressed or (age > max(reference_age + 2, 1.25 * reference_age)
                                            and flows["policy"]["reuse_intensity"] > 1.25 * max(self.baseline["policy"]["reuse_intensity"], 1))
                supplied = all(flows[name]["new_per_second"] >= .9 * self.baseline[name]["new_per_second"]
                               and flows[name]["reuse_intensity"] <= 1.1 * max(self.baseline[name]["reuse_intensity"], 1)
                               for name in ("policy", "wm"))
                direction = "decrease" if stressed else ("increase" if supplied and pressure.get("training", 0) else "hold")
                self.streak = self.streak + 1 if direction == self.direction else 1
                self.direction = direction
                reason = direction + ": flow/reuse; exact WM version-age unresolved and excluded"
                if publication_lag:
                    reason = "publication priority active; hold alpha until version ACK"
                elif self.streak >= 2 and now - self.last_change >= 120 and direction != "hold":
                    self.alpha = min(.85, max(.25, self.alpha + (.05 if direction == "increase" else -.05)))
                    self.last_change = now
                    self.streak = 0
        if previous_stage != self.stage:
            self.streak = 0
        # A shared training envelope enforces these cost-weighted recipe shares.
        raw = {"policy_train": 5 * costs.get("policy_train", 66),
               "ctrl_train": costs.get("ctrl_train", 200), "reward_train": costs.get("reward_train", 15)}
        train_sum = sum(raw.values())
        alpha = self.alpha if ready else 0.0
        weights = {k: alpha * v / train_sum for k, v in raw.items()}
        infer_services = SERVICES[3:]
        demand = {k: max(0, pressure.get(k, 0)) for k in infer_services}
        total_demand = sum(demand.values())
        defaults = {"policy_infer": .25, "ctrl_infer": .625, "reward_infer": .125}
        # Keep nonzero floors for every dependency, with smoothed pressure allocation.
        shares = {k: .5 * defaults[k] + .5 * (demand[k] / total_demand if total_demand else defaults[k]) for k in demand}
        for k in infer_services:
            weights[k] = (1 - alpha) * (.05 + .85 * shares[k])
        assert abs(sum(weights.values()) - 1) < 1e-9
        self.version += 1
        self.last_plan = {"version": self.version, "stage": self.stage, "alpha": alpha, "weights": weights,
                          "reason": reason, "flows": flows, "baseline": self.baseline,
                          "full_update_costs": costs, "pressure": pressure, "publication_lag": publication_lag,
                          "unit": "completed execution wall-time proxy; not exclusive GPU/SM share"}
        return self.last_plan

    def state_dict(self):
        return dict(vars(self))
