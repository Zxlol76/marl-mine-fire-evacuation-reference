"""Tensor rollout storage for graph-based multi-agent PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .ppo import generalized_advantage_estimation


@dataclass
class RolloutBatch:
    node_features: torch.Tensor
    edge_features: torch.Tensor
    edge_mask: torch.Tensor
    action_mask: torch.Tensor
    agent_node_index: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    alive_mask: torch.Tensor

    @property
    def sample_count(self) -> int:
        return int(self.actions.shape[0])

    def select(self, indices: torch.Tensor) -> "RolloutBatch":
        return RolloutBatch(
            node_features=self.node_features[indices],
            edge_features=self.edge_features[indices],
            edge_mask=self.edge_mask[indices],
            action_mask=self.action_mask[indices],
            agent_node_index=self.agent_node_index[indices],
            actions=self.actions[indices],
            old_log_probabilities=self.old_log_probabilities[indices],
            advantages=self.advantages[indices],
            returns=self.returns[indices],
            alive_mask=self.alive_mask[indices],
        )

    def to(self, device: torch.device | str) -> "RolloutBatch":
        values = {
            name: getattr(self, name).to(device)
            for name in self.__dataclass_fields__
        }
        return RolloutBatch(**values)


class MultiAgentRolloutBuffer:
    """Store graph observations and construct PPO training batches.

    Each call to :meth:`add` stores one time step for a vectorized group of
    environments. Tensors use the following leading dimensions:

    - graph observations: ``[environments, agents, ...]``;
    - actions and scalar agent values: ``[environments, agents]``;
    - edge data: ``[environments, edges, ...]``.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.9997,
        gae_lambda: float = 0.97,
    ) -> None:
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clear()

    def clear(self) -> None:
        self.node_features: list[torch.Tensor] = []
        self.edge_features: list[torch.Tensor] = []
        self.edge_masks: list[torch.Tensor] = []
        self.action_masks: list[torch.Tensor] = []
        self.agent_node_indices: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probabilities: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.terminated: list[torch.Tensor] = []
        self.alive_masks: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        *,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        action_mask: torch.Tensor,
        agent_node_index: torch.Tensor,
        actions: torch.Tensor,
        log_probabilities: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        terminated: torch.Tensor,
        alive_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if alive_mask is None:
            alive_mask = torch.ones_like(rewards, dtype=torch.bool)
        self.node_features.append(node_features.detach().cpu())
        self.edge_features.append(edge_features.detach().cpu())
        self.edge_masks.append(edge_mask.detach().cpu())
        self.action_masks.append(action_mask.detach().cpu())
        self.agent_node_indices.append(agent_node_index.detach().cpu())
        self.actions.append(actions.detach().cpu())
        self.log_probabilities.append(log_probabilities.detach().cpu())
        self.rewards.append(rewards.detach().cpu())
        self.values.append(values.detach().cpu())
        self.terminated.append(terminated.detach().cpu())
        self.alive_masks.append(alive_mask.detach().cpu())

    def build_batch(
        self,
        next_values: torch.Tensor,
        *,
        device: torch.device | str = "cpu",
    ) -> RolloutBatch:
        if not self.rewards:
            raise RuntimeError("Cannot build a batch from an empty rollout buffer.")

        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        terminated = torch.stack(self.terminated)
        shifted_values = torch.cat(
            (values[1:], next_values.detach().cpu().unsqueeze(0)), dim=0
        )
        advantages, returns = generalized_advantage_estimation(
            rewards,
            values,
            shifted_values,
            terminated,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        time_steps, environments, agents = rewards.shape

        def flatten_samples(items: list[torch.Tensor]) -> torch.Tensor:
            tensor = torch.stack(items)
            return tensor.reshape(time_steps * environments, *tensor.shape[2:])

        batch = RolloutBatch(
            node_features=flatten_samples(self.node_features),
            edge_features=flatten_samples(self.edge_features),
            edge_mask=flatten_samples(self.edge_masks),
            action_mask=flatten_samples(self.action_masks),
            agent_node_index=flatten_samples(self.agent_node_indices),
            actions=flatten_samples(self.actions).reshape(-1, agents),
            old_log_probabilities=flatten_samples(
                self.log_probabilities
            ).reshape(-1, agents),
            advantages=advantages.reshape(-1, agents),
            returns=returns.reshape(-1, agents),
            alive_mask=flatten_samples(self.alive_masks).reshape(-1, agents),
        )
        return batch.to(device)
