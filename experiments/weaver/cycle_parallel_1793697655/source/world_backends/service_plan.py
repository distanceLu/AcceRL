"""Derive active owners and replay consumers before actor construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import WorldBackendConfig


@dataclass(frozen=True)
class ServicePlan:
    backend: str
    reward_provider: str
    world_size: int
    trainer_kinds: tuple[str, ...]
    inference_kinds: tuple[str, ...]
    wm_consumers: tuple[str, ...]
    world_trainable: bool = True

    @property
    def active_kinds(self) -> tuple[str, ...]:
        return self.trainer_kinds

    @property
    def integrated_reward(self) -> bool:
        return self.reward_provider == "world_head"

    @property
    def gpu_model_processes_per_card(self) -> int:
        return len(self.trainer_kinds) + len(self.inference_kinds)

    @property
    def expected_gpu_model_processes(self) -> int:
        return self.world_size * self.gpu_model_processes_per_card

    @property
    def expected_wm_consumers(self) -> int:
        return self.world_size * len(self.wm_consumers)

    @property
    def logical_update_ratio(self) -> str:
        return "5:0:1" if not self.world_trainable else ("5:1" if self.integrated_reward else "5:1:1")

    @property
    def checkpoint_kinds(self) -> tuple[str, ...]:
        return self.trainer_kinds

    @property
    def publication_kinds(self) -> tuple[str, ...]:
        return self.trainer_kinds

    def expected_optimizer_steps(self, policy_steps: int) -> dict[str, int]:
        if isinstance(policy_steps, bool) or not isinstance(policy_steps, int) or policy_steps < 0:
            raise ValueError("policy_steps must be a nonnegative integer")
        return {kind: policy_steps if kind == "policy" else policy_steps // 5 for kind in self.trainer_kinds}

    def due_kinds(self, policy_step: int) -> tuple[str, ...]:
        if isinstance(policy_step, bool) or not isinstance(policy_step, int) or policy_step < 1:
            raise ValueError("policy_step must be a positive, one-based step")
        return self.trainer_kinds if policy_step % 5 == 0 else ("policy",)

    def validate_runtime(self, world_size: int | None = None,
                         role_receipts: Sequence[Mapping[str, Any]] | None = None) -> "ServicePlan":
        if self.world_size != 4 or (world_size is not None and world_size != self.world_size):
            raise ValueError("The runtime service topology requires exactly four ranks")
        if role_receipts is not None:
            expected = {(phase, kind, rank) for phase, kinds in
                        (("trainer", self.trainer_kinds), ("inference", self.inference_kinds))
                        for kind in kinds for rank in range(self.world_size)}
            observed = [(receipt.get("phase"), receipt.get("kind"), receipt.get("rank"))
                        for receipt in role_receipts]
            if len(observed) != len(set(observed)) or set(observed) != expected:
                raise ValueError("Role receipts must contain each active phase/kind/rank exactly once")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "reward_provider": self.reward_provider,
                "world_trainable": self.world_trainable,
                "world_kind": "ctrl", "world_size": self.world_size,
                "trainer_kinds": list(self.trainer_kinds), "inference_kinds": list(self.inference_kinds),
                "wm_consumers": list(self.wm_consumers), "expected_wm_consumers": self.expected_wm_consumers,
                "gpu_model_processes_per_card": self.gpu_model_processes_per_card,
                "expected_gpu_model_processes": self.expected_gpu_model_processes,
                "logical_update_ratio": self.logical_update_ratio,
                "checkpoint_kinds": list(self.checkpoint_kinds), "publication_kinds": list(self.publication_kinds)}


def build_service_plan(config: WorldBackendConfig | Mapping[str, Any] | None = None,
                       world_size: int = 4) -> ServicePlan:
    """Planning is CPU-only; call config.validate_admission() before runtime.

    The world owner keeps the legacy ``ctrl`` kind to preserve receipts and
    telemetry. A hypothetical integrated plan can be inspected without claiming
    that a compatible checkpoint or calibrated reward head exists.
    """
    if not isinstance(config, WorldBackendConfig):
        config = WorldBackendConfig.from_mapping(config)
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    inference = ("policy", "ctrl") if config.reward_provider == "world_head" else ("policy", "ctrl", "reward")
    kinds = inference if config.world_trainable else ("policy", "reward")
    consumers = (("ctrl",) if config.reward_provider == "world_head" else ("ctrl", "reward")) if config.world_trainable else ("reward",)
    return ServicePlan(config.backend, config.reward_provider, world_size, kinds, inference, consumers,
                       config.world_trainable)
