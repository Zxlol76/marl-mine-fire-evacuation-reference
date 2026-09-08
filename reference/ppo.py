"""Clipped-PPO utilities shared by the IPPO and MAPPO implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.distributions import Categorical


@dataclass
class PPOLoss:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


def normalize_advantages(
    advantages: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    if advantages.ndim < 2:
        mask = torch.ones_like(advantages, dtype=torch.bool)
        if valid_mask is not None:
            mask = valid_mask.bool()
        selected = advantages[mask]
        normalized = torch.zeros_like(advantages)
        if selected.numel() > 0:
            normalized[mask] = (selected - selected.mean()) / (
                selected.std(unbiased=False) + 1e-8
            )
        return normalized

    mask = (
        torch.ones_like(advantages, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.bool()
    )
    normalized = torch.zeros_like(advantages)
    for agent_index in range(advantages.shape[-1]):
        agent_values = advantages[..., agent_index]
        agent_mask = mask[..., agent_index]
        selected = agent_values[agent_mask]
        if selected.numel() > 0:
            normalized[..., agent_index][agent_mask] = (
                selected - selected.mean()
            ) / (selected.std(unbiased=False) + 1e-8)
    return normalized


def generalized_advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    *,
    gamma: float = 0.9997,
    gae_lambda: float = 0.97,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute reverse-time GAE for tensors shaped ``[time, batch, agents]``."""

    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros_like(rewards[-1])
    for step in reversed(range(rewards.shape[0])):
        not_terminal = 1.0 - terminated[step].to(rewards.dtype)
        delta = (
            rewards[step]
            + gamma * next_values[step] * not_terminal
            - values[step]
        )
        running_advantage = (
            delta
            + gamma * gae_lambda * not_terminal * running_advantage
        )
        advantages[step] = running_advantage
    returns = advantages + values
    return advantages, returns


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def clipped_ppo_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.08,
) -> PPOLoss:

    distribution = Categorical(logits=logits)
    log_probabilities = distribution.log_prob(actions)
    probability_ratio = (log_probabilities - old_log_probabilities).exp()

    mask = (
        torch.ones_like(advantages, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.bool()
    )
    normalized_advantages = normalize_advantages(advantages, mask)
    unclipped = probability_ratio * normalized_advantages
    clipped = probability_ratio.clamp(
        1.0 - clip_epsilon, 1.0 + clip_epsilon
    ) * normalized_advantages

    policy_loss = -_masked_mean(torch.minimum(unclipped, clipped), mask)
    value_loss = _masked_mean((values - returns).square(), mask)
    entropy = _masked_mean(distribution.entropy(), mask)
    total = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy
    )
    return PPOLoss(total=total, policy=policy_loss, value=value_loss, entropy=entropy)
