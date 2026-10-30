"""Policy and Reward inference owners in one CUDA process.

The owners keep their original request queues, batching loops, model versions,
and capture hooks. Only their Ray process and physical GPU are shared, so an
NCCL communicator never has two different processes using that GPU.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

import ray


def _owner_class(cls: Any) -> Type:
    """Accept a plain class or a Ray actor class with its installed hooks."""
    metadata = getattr(cls, "__ray_metadata__", None)
    return getattr(metadata, "modified_class", cls)


@ray.remote(num_gpus=1)
class PolicyRewardInferenceActor:
    def __init__(
        self,
        policy_cls: Type,
        reward_cls: Type,
        policy_kwargs: Dict[str, Any],
        reward_kwargs: Dict[str, Any],
        placement_guard: Optional[Callable[[], Any]] = None,
    ):
        if placement_guard is not None:
            placement_guard()
        self.policy = _owner_class(policy_cls)(**dict(policy_kwargs))
        self.reward = _owner_class(reward_cls)(**dict(reward_kwargs))
        if self.policy.requests is self.reward.requests:
            raise RuntimeError("Policy and Reward must keep independent request queues")
        if self.policy.promises is self.reward.promises:
            raise RuntimeError("Policy and Reward must keep independent promise queues")

    def _broadcast_owner(self, group_name: str):
        if group_name.startswith("broadcast_actor"):
            return self.policy
        if group_name.startswith("broadcast_reward"):
            return self.reward
        raise ValueError(f"Shared inference does not own broadcast group {group_name!r}")

    async def request(self, *args, **kwargs):
        return await self.policy.request(*args, **kwargs)

    async def reward_request(self, *args, **kwargs):
        return await self.reward.request(*args, **kwargs)

    def setup_broadcast_group(
        self, master_addr, master_port, group_name, group_world_size,
        my_rank_in_group,
    ):
        return self._broadcast_owner(group_name).setup_broadcast_group(
            master_addr, master_port, group_name, group_world_size,
            my_rank_in_group,
        )

    def receive_and_update_weights(self, group_name):
        return self._broadcast_owner(group_name).receive_and_update_weights(group_name)

    def broadcast_sanity_check(self, group_name, dtype_name="float32", numel=8):
        return self._broadcast_owner(group_name).broadcast_sanity_check(
            group_name, dtype_name=dtype_name, numel=numel,
        )

    def evaluate_fixed_holdout(self, *args, **kwargs):
        return self.reward.evaluate_fixed_holdout(*args, **kwargs)

    def get_model_keys(self):
        return self.policy.get_model_keys()

    def forward_test(self):
        return self.policy.forward_test()

    def get_inference_owners(self):
        """Small serializable audit record; the model objects stay in the actor."""
        return {
            name: {
                "role": getattr(owner, "capture_role", name),
                "actor_id": getattr(owner, "actor_id", None),
                "model_version": getattr(owner, "capture_model_version", -1),
                "queued_requests": len(owner.requests),
                "queued_promises": len(owner.promises),
            }
            for name, owner in (("policy", self.policy), ("reward", self.reward))
        }


class RewardInferenceHandle:
    """Serializable reward endpoint retaining the original .request.remote API.

    Explicit properties avoid dynamic attribute lookup while Ray/cloudpickle
    reconstructs this facade inside rollout workers.
    """

    __slots__ = ("_pair",)

    def __init__(self, pair_actor):
        self._pair = pair_actor

    @property
    def request(self):
        return self._pair.reward_request

    @property
    def setup_broadcast_group(self):
        return self._pair.setup_broadcast_group

    @property
    def receive_and_update_weights(self):
        return self._pair.receive_and_update_weights

    @property
    def broadcast_sanity_check(self):
        return self._pair.broadcast_sanity_check

    @property
    def evaluate_fixed_holdout(self):
        return self._pair.evaluate_fixed_holdout

    def __getstate__(self):
        return {"pair": self._pair}

    def __setstate__(self, state):
        self._pair = state["pair"]
