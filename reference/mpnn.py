"""MPNN actor/critic reference for graph-based multi-agent learning."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class EdgeMessageLayer(nn.Module):

    def __init__(self, hidden_dim: int, edge_feature_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        source, target = edge_index.long()
        source_states = nodes[:, source, :]
        messages = self.message(torch.cat((source_states, edge_features), dim=-1))

        if edge_mask is not None:
            messages = messages * edge_mask.to(messages.dtype).unsqueeze(-1)

        aggregated = torch.zeros_like(nodes)
        scatter_index = target.view(1, -1, 1).expand(
            nodes.shape[0], -1, nodes.shape[-1]
        )
        aggregated.scatter_add_(1, scatter_index, messages)
        return self.update(torch.cat((nodes, aggregated), dim=-1))


class MPNNEncoder(nn.Module):
    """Encode each agent's graph observation into one fixed-size vector."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            EdgeMessageLayer(hidden_dim, edge_feature_dim)
            for _ in range(max(1, layers))
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
        agent_node_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations shaped ``[batch, agents, nodes, features]``."""

        batch_size, agent_count, node_count, feature_dim = node_features.shape
        flat_nodes = node_features.reshape(
            batch_size * agent_count, node_count, feature_dim
        )
        hidden = self.node_encoder(flat_nodes)

        flat_edges = (
            edge_features[:, None, :, :]
            .expand(-1, agent_count, -1, -1)
            .reshape(batch_size * agent_count, edge_features.shape[1], -1)
        )
        flat_mask = None
        if edge_mask is not None:
            flat_mask = (
                edge_mask[:, None, :]
                .expand(-1, agent_count, -1)
                .reshape(batch_size * agent_count, edge_mask.shape[1])
            )

        for layer in self.layers:
            hidden = torch.relu(
                hidden + layer(hidden, edge_index, flat_edges, flat_mask)
            )

        if agent_node_index is None:
            pooled = hidden.mean(dim=1)
        else:
            flat_index = agent_node_index.reshape(-1).long()
            pooled = hidden[
                torch.arange(hidden.shape[0], device=hidden.device), flat_index
            ]
        return pooled.reshape(batch_size, agent_count, self.hidden_dim)


class SharedActor(nn.Module):
    """Parameter-shared masked actor used by both reference variants."""

    def __init__(
        self,
        encoder: MPNNEncoder,
        action_dim: int,
        policy_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.policy_head = nn.Sequential(
            nn.Linear(encoder.hidden_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(policy_hidden_dim, action_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        action_mask: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
        agent_node_index: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(
            node_features,
            edge_index,
            edge_features,
            edge_mask=edge_mask,
            agent_node_index=agent_node_index,
        )
        logits = self.policy_head(embeddings)
        logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits, embeddings


class IndependentCritic(nn.Module):
    """IPPO-style critic with a separate MPNN and agent-ID embedding."""

    def __init__(
        self,
        encoder: MPNNEncoder,
        agent_count: int,
        value_hidden_dim: int = 128,
        agent_id_dim: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.agent_embedding = nn.Embedding(agent_count, agent_id_dim)
        self.value_head = nn.Sequential(
            nn.Linear(encoder.hidden_dim + agent_id_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
        agent_node_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        local_embeddings = self.encoder(
            node_features,
            edge_index,
            edge_features,
            edge_mask=edge_mask,
            agent_node_index=agent_node_index,
        )
        batch_size, agent_count, _ = local_embeddings.shape
        agent_ids = torch.arange(agent_count, device=local_embeddings.device)
        id_embeddings = self.agent_embedding(agent_ids)
        id_embeddings = id_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        critic_input = torch.cat((local_embeddings, id_embeddings), dim=-1)
        return self.value_head(critic_input).squeeze(-1)


class CentralizedCritic(nn.Module):
    """MAPPO-style critic with local, global, and agent-ID features."""

    def __init__(
        self,
        encoder: MPNNEncoder,
        agent_count: int,
        value_hidden_dim: int = 128,
        agent_id_dim: int = 16,
        summed_feature_indices: tuple[int, ...] = (4, 9, 10),
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.summed_feature_indices = tuple(
            int(index) for index in summed_feature_indices
        )
        self.agent_embedding = nn.Embedding(agent_count, agent_id_dim)
        self.value_head = nn.Sequential(
            nn.Linear(2 * encoder.hidden_dim + agent_id_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
        agent_node_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, agent_count, _, feature_dim = node_features.shape
        global_node_features = node_features.mean(dim=1)
        for feature_index in self.summed_feature_indices:
            if 0 <= feature_index < feature_dim:
                global_node_features[..., feature_index] = node_features[
                    ..., feature_index
                ].sum(dim=1)

        global_agent_views = global_node_features[:, None, :, :].expand(
            -1, agent_count, -1, -1
        )
        local_embeddings = self.encoder(
            global_agent_views,
            edge_index,
            edge_features,
            edge_mask=edge_mask,
            agent_node_index=agent_node_index,
        )

        global_embeddings = self.encoder(
            global_node_features[:, None, :, :],
            edge_index,
            edge_features,
            edge_mask=edge_mask,
        )
        global_context = global_embeddings.expand(
            -1, local_embeddings.shape[1], -1
        )

        agent_ids = torch.arange(agent_count, device=local_embeddings.device)
        id_embeddings = self.agent_embedding(agent_ids)
        id_embeddings = id_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        critic_input = torch.cat(
            (local_embeddings, global_context, id_embeddings), dim=-1
        )
        return self.value_head(critic_input).squeeze(-1)
