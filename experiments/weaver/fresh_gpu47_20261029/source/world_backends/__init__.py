"""World-model contracts with lazy, backend-specific ML dependencies."""
from .contracts import BackendCapabilities, RolloutResult, WorldBackend, WorldBackendConfig, validate_schema
from .data import EpisodeWindow, validate_episode_window
from .reward import DeltaSuccessRewardProvider, RewardSequence, build_reward_provider
from .service_plan import ServicePlan, build_service_plan

__all__ = ["BackendCapabilities", "RolloutResult", "WorldBackend", "WorldBackendConfig",
           "validate_schema", "EpisodeWindow", "validate_episode_window", "DeltaSuccessRewardProvider",
           "RewardSequence", "build_reward_provider", "ServicePlan", "build_service_plan"]
