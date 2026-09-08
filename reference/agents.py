"""Reference IPPO and MAPPO agents with graph observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Type

import torch
from torch import nn
from torch.distributions import Categorical

from .buffer import RolloutBatch
from .mpnn import CentralizedCritic, IndependentCritic, MPNNEncoder, SharedActor
from .ppo import clipped_ppo_loss


@dataclass
class PPOConfig:
    node_feature_dim: int
    edge_feature_dim: int
    action_dim: int
    agent_count: int
    mpnn_hidden_dim: int = 64
    mpnn_layers: int = 2
    policy_hidden_dim: int = 128
    value_hidden_dim: int = 128
    agent_id_dim: int = 16
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.08
    ppo_epochs: int = 3
    mini_batch_size: int = 512
    max_gradient_norm: float = 0.8


@dataclass
class AgentStep:
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    values: torch.Tensor


@dataclass
class UpdateMetrics:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    updates: int


class GraphPPOAgent:

    critic_type: Type[nn.Module]

    def __init__(
        self,
        config: PPOConfig,
        edge_index: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.edge_index = edge_index.long().to(self.device)

        actor_encoder = self._make_encoder()
        critic_encoder = self._make_encoder()
        self.actor = SharedActor(
            actor_encoder,
            action_dim=config.action_dim,
            policy_hidden_dim=config.policy_hidden_dim,
        ).to(self.device)
        self.critic = self.critic_type(
            critic_encoder,
            config.agent_count,
            value_hidden_dim=config.value_hidden_dim,
            agent_id_dim=config.agent_id_dim,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )

    def _make_encoder(self) -> MPNNEncoder:
        return MPNNEncoder(
            node_feature_dim=self.config.node_feature_dim,
            edge_feature_dim=self.config.edge_feature_dim,
            hidden_dim=self.config.mpnn_hidden_dim,
            layers=self.config.mpnn_layers,
        )

    def _critic_values(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        agent_node_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic(
            node_features,
            self.edge_index,
            edge_features,
            edge_mask=edge_mask,
            agent_node_index=agent_node_index,
        )

    @torch.no_grad()
    def act(
        self,
        *,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        action_mask: torch.Tensor,
        agent_node_index: torch.Tensor,
        deterministic: bool = False,
    ) -> AgentStep:
        node_features = node_features.to(self.device)
        edge_features = edge_features.to(self.device)
        edge_mask = edge_mask.to(self.device)
        action_mask = action_mask.to(self.device)
        agent_node_index = agent_node_index.to(self.device)

        logits, _ = self.actor(
            node_features,
            self.edge_index,
            edge_features,
            action_mask,
            edge_mask=edge_mask,
            agent_node_index=agent_node_index,
        )
        distribution = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        values = self._critic_values(
            node_features, edge_features, edge_mask, agent_node_index
        )
        return AgentStep(
            actions=actions,
            log_probabilities=distribution.log_prob(actions),
            values=values,
        )

    def evaluate_actions(
        self,
        batch: RolloutBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, _ = self.actor(
            batch.node_features,
            self.edge_index,
            batch.edge_features,
            batch.action_mask,
            edge_mask=batch.edge_mask,
            agent_node_index=batch.agent_node_index,
        )
        values = self._critic_values(
            batch.node_features,
            batch.edge_features,
            batch.edge_mask,
            batch.agent_node_index,
        )
        return logits, values

    def update(self, batch: RolloutBatch) -> UpdateMetrics:
        batch = batch.to(self.device)
        sample_count = batch.sample_count
        mini_batch_size = max(
            1, min(int(self.config.mini_batch_size), sample_count)
        )
        totals = {
            "total": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "entropy": 0.0,
        }
        update_count = 0

        for _ in range(max(1, int(self.config.ppo_epochs))):
            order = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, mini_batch_size):
                indices = order[start : start + mini_batch_size]
                mini_batch = batch.select(indices)
                logits, values = self.evaluate_actions(mini_batch)
                loss = clipped_ppo_loss(
                    logits,
                    values,
                    mini_batch.actions,
                    mini_batch.old_log_probabilities,
                    mini_batch.advantages,
                    mini_batch.returns,
                    valid_mask=mini_batch.alive_mask,
                    clip_epsilon=self.config.clip_epsilon,
                    value_coefficient=self.config.value_coefficient,
                    entropy_coefficient=self.config.entropy_coefficient,
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_gradient_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_gradient_norm
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                totals["total"] += float(loss.total.detach())
                totals["policy"] += float(loss.policy.detach())
                totals["value"] += float(loss.value.detach())
                totals["entropy"] += float(loss.entropy.detach())
                update_count += 1

        denominator = max(1, update_count)
        return UpdateMetrics(
            total_loss=totals["total"] / denominator,
            policy_loss=totals["policy"] / denominator,
            value_loss=totals["value"] / denominator,
            entropy=totals["entropy"] / denominator,
            updates=update_count,
        )

    def save(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), output / "actor.pt")
        torch.save(self.critic.state_dict(), output / "critic.pt")
        torch.save(self.actor_optimizer.state_dict(), output / "actor_optimizer.pt")
        torch.save(
            self.critic_optimizer.state_dict(), output / "critic_optimizer.pt"
        )
        torch.save(asdict(self.config), output / "config.pt")

    def load(self, directory: str | Path, *, load_optimizers: bool = False) -> None:
        source = Path(directory)
        self.actor.load_state_dict(
            torch.load(source / "actor.pt", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(source / "critic.pt", map_location=self.device)
        )
        if load_optimizers:
            self.actor_optimizer.load_state_dict(
                torch.load(
                    source / "actor_optimizer.pt", map_location=self.device
                )
            )
            self.critic_optimizer.load_state_dict(
                torch.load(
                    source / "critic_optimizer.pt", map_location=self.device
                )
            )


class ReferenceIPPO(GraphPPOAgent):

    critic_type = IndependentCritic


class ReferenceMAPPO(GraphPPOAgent):
    
    critic_type = CentralizedCritic
