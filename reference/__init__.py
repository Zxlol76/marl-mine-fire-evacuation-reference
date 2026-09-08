"""Reference components for graph-based mine-fire evacuation research."""

from .agents import PPOConfig, ReferenceIPPO, ReferenceMAPPO
from .buffer import MultiAgentRolloutBuffer, RolloutBatch
from .event_replanning import (
    AgentPlanningState,
    EventReplanningManager,
    PlanningObservation,
    ReplanDecision,
    ReplanReason,
    ReplanningConfig,
)
from .mpnn import CentralizedCritic, IndependentCritic, MPNNEncoder, SharedActor
from .personnel import Evacuee, PersonnelMovementModel, RouteSegment
from .physical_models import (
    WorkerState,
    advance_worker_state,
    gradient_adjusted_speed,
    integrated_worker_speed,
    smoke_spread_velocity,
    stamina_after_step,
)
from .ppo import (
    PPOLoss,
    clipped_ppo_loss,
    generalized_advantage_estimation,
    normalize_advantages,
)
from .refuge_capacity import CapacityStatus, RefugeCapacityManager
from .smoke_network import DirectionalSmokeNetwork, Roadway, SmokeState

__all__ = [
    "AgentPlanningState",
    "CapacityStatus",
    "CentralizedCritic",
    "DirectionalSmokeNetwork",
    "Evacuee",
    "EventReplanningManager",
    "IndependentCritic",
    "MPNNEncoder",
    "MultiAgentRolloutBuffer",
    "PPOConfig",
    "PPOLoss",
    "PersonnelMovementModel",
    "PlanningObservation",
    "ReferenceIPPO",
    "ReferenceMAPPO",
    "ReplanDecision",
    "ReplanReason",
    "ReplanningConfig",
    "RefugeCapacityManager",
    "Roadway",
    "RolloutBatch",
    "RouteSegment",
    "SharedActor",
    "SmokeState",
    "WorkerState",
    "advance_worker_state",
    "clipped_ppo_loss",
    "generalized_advantage_estimation",
    "gradient_adjusted_speed",
    "integrated_worker_speed",
    "normalize_advantages",
    "smoke_spread_velocity",
    "stamina_after_step",
]
