"""Sparse GCN actor-critic and PPO update for the 1,584-node v3 graph."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .observation import (
    ACTION_FEATURE_DIM,
    BASE_NODE_FEATURE_DIM,
    GRAPH_EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    PolicyObservation,
)


@dataclass(frozen=True)
class PPOConfig:
    architecture: str = "gcn"
    hidden_dim: int = 128
    gcn_layers: int = 3
    learning_rate: float = 3.0e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    value_clip: float = 1.0
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    rollout_steps: int = 1024
    target_kl: float = 0.03
    route_history: bool = False
    current_route_state: bool = True

    def __post_init__(self) -> None:
        if self.architecture not in {"gcn", "mlp"}:
            raise ValueError("architecture must be 'gcn' or 'mlp'")
        if self.hidden_dim < 1 or self.gcn_layers < 1 or self.minibatch_size < 1 or self.rollout_steps < 1:
            raise ValueError("network and batch sizes must be positive")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("learning rate and gradient norm must be positive")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and GAE lambda must be in [0, 1]")
        if not 0.0 < self.clip_ratio < 1.0 or self.value_clip <= 0.0:
            raise ValueError("PPO clipping parameters must be positive")
        if self.update_epochs < 1 or self.target_kl <= 0.0:
            raise ValueError("update epochs and target KL must be positive")


class SparseGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_linear = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        message = self.neighbor_linear(x[source]) + self.edge_linear(edge_features)
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, target, message)
        degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
        aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
        update = F.relu(self.norm(self.self_linear(x) + aggregate))
        return x + update


class SparseGCNActorCritic(nn.Module):
    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,
        graph_edge_dim: int = GRAPH_EDGE_FEATURE_DIM,
        action_dim: int = ACTION_FEATURE_DIM,
        hidden_dim: int = 64,
        architecture: str = "gcn",
        gcn_layers: int = 2,
        route_history: bool = False,
    ):
        super().__init__()
        if architecture not in {"gcn", "mlp"}:
            raise ValueError("architecture must be 'gcn' or 'mlp'")
        self.architecture = architecture
        self.route_history = bool(route_history)
        self.input = nn.Linear(node_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            SparseGraphBlock(hidden_dim, graph_edge_dim) for _ in range(gcn_layers)
        )
        actor_hidden_inputs = 7 if self.route_history else 4
        critic_hidden_inputs = 6 if self.route_history else 3
        self.actor = nn.Sequential(
            nn.Linear(actor_hidden_inputs * hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_hidden_inputs * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        action_features: torch.Tensor,
        current: torch.Tensor,
        egress: torch.Tensor,
        start: torch.Tensor,
        previous: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, node_count, _ = node_features.shape
        flat = F.relu(self.input(node_features.reshape(batch_size * node_count, -1)))
        if self.architecture == "gcn":
            for block in self.blocks:
                flat = block(flat, edge_index, edge_features)
        encoded = flat.reshape(batch_size, node_count, -1)
        batch = torch.arange(batch_size, device=encoded.device)
        h_current = encoded[batch, current]
        h_egress = encoded[batch, egress]
        current_all = h_current[:, None, :].expand_as(encoded)
        egress_all = h_egress[:, None, :].expand_as(encoded)
        actor_parts = [current_all, encoded, encoded - current_all, egress_all]
        critic_parts = [h_current, h_egress, encoded.mean(dim=1)]
        if self.route_history:
            h_start = encoded[batch, start]
            h_previous = encoded[batch, previous]
            visited = node_features[:, :, 10].clamp(0.0, 1.0)
            h_path = (encoded * visited.unsqueeze(-1)).sum(dim=1) / visited.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            actor_parts.extend(
                [
                    h_start[:, None, :].expand_as(encoded),
                    h_previous[:, None, :].expand_as(encoded),
                    h_path[:, None, :].expand_as(encoded),
                ]
            )
            critic_parts.extend([h_start, h_previous, h_path])
        actor_input = torch.cat([*actor_parts, action_features], dim=-1)
        logits = self.actor(actor_input).squeeze(-1)
        value = self.critic(torch.cat(critic_parts, dim=-1)).squeeze(-1)
        return logits, value


def collate_observations(
    observations: Sequence[PolicyObservation], device: torch.device
) -> tuple[torch.Tensor, ...]:
    if not observations:
        raise ValueError("cannot collate an empty observation batch")
    node_count = observations[0].node_features.shape[0]
    node = torch.as_tensor(
        np.stack([item.node_features for item in observations]),
        dtype=torch.float32,
        device=device,
    )
    action = torch.as_tensor(
        np.stack([item.action_features for item in observations]),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.as_tensor(
        np.stack([item.action_mask for item in observations]),
        dtype=torch.bool,
        device=device,
    )
    edge_indices = []
    edge_features = []
    for batch_index, item in enumerate(observations):
        edge_indices.append(item.edge_index + batch_index * node_count)
        edge_features.append(item.edge_features)
    edge_index = torch.as_tensor(
        np.concatenate(edge_indices, axis=1), dtype=torch.long, device=device
    )
    edge = torch.as_tensor(
        np.concatenate(edge_features, axis=0), dtype=torch.float32, device=device
    )
    current = torch.as_tensor(
        [item.current for item in observations], dtype=torch.long, device=device
    )
    egress = torch.as_tensor(
        [item.egress for item in observations], dtype=torch.long, device=device
    )
    start = torch.as_tensor(
        [item.start for item in observations], dtype=torch.long, device=device
    )
    previous = torch.as_tensor(
        [item.previous for item in observations], dtype=torch.long, device=device
    )
    return node, edge_index, edge, action, mask, current, egress, start, previous


class PPOAgent:
    def __init__(self, config: PPOConfig = PPOConfig(), device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.network = SparseGCNActorCritic(
            node_dim=(NODE_FEATURE_DIM if config.route_history else BASE_NODE_FEATURE_DIM),
            hidden_dim=config.hidden_dim,
            architecture=config.architecture,
            gcn_layers=config.gcn_layers,
            route_history=config.route_history,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )

    def _distribution_value(
        self, observations: Sequence[PolicyObservation]
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        node, edge_index, edge, action, mask, current, egress, start, previous = collate_observations(
            observations, self.device
        )
        logits, value = self.network(
            node, edge_index, edge, action, current, egress, start, previous
        )
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("policy observation has no valid action")
        logits = logits.masked_fill(~mask, -1.0e9)
        return torch.distributions.Categorical(logits=logits), value

    @torch.no_grad()
    def act(
        self, observation: PolicyObservation, greedy: bool = False
    ) -> tuple[int, float, float]:
        distribution, value = self._distribution_value([observation])
        action = (
            torch.argmax(distribution.logits, dim=-1)
            if greedy
            else distribution.sample()
        )
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(value.item()),
        )

    @torch.no_grad()
    def value_of(self, observation: PolicyObservation) -> float:
        return float(self._distribution_value([observation])[1].item())

    def update(self, transitions: Sequence[dict]) -> dict[str, float]:
        if not transitions:
            return {}
        cfg = self.config
        advantages = torch.as_tensor(
            [item["advantage"] for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1.0e-8
        )
        returns = torch.as_tensor(
            [item["return"] for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        old_log_prob = torch.as_tensor(
            [item["log_prob"] for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        old_value = torch.as_tensor(
            [item["value"] for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item["action"] for item in transitions],
            dtype=torch.long,
            device=self.device,
        )
        logs: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
            "clip_fraction": [],
        }
        count = len(transitions)
        stop = False
        for _ in range(cfg.update_epochs):
            if stop:
                break
            for indices in torch.randperm(count).split(cfg.minibatch_size):
                ids = indices.tolist()
                observations = [transitions[index]["observation"] for index in ids]
                distribution, value = self._distribution_value(observations)
                chosen = actions[indices]
                log_prob = distribution.log_prob(chosen)
                ratio = torch.exp(log_prob - old_log_prob[indices])
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio
                ) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                clipped_value = old_value[indices] + torch.clamp(
                    value - old_value[indices], -cfg.value_clip, cfg.value_clip
                )
                value_loss = 0.5 * torch.maximum(
                    (value - returns[indices]).pow(2),
                    (clipped_value - returns[indices]).pow(2),
                ).mean()
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + cfg.value_coefficient * value_loss
                    - cfg.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                approximate_kl = (old_log_prob[indices] - log_prob).mean()
                logs["policy_loss"].append(float(policy_loss.item()))
                logs["value_loss"].append(float(value_loss.item()))
                logs["entropy"].append(float(entropy.item()))
                logs["kl"].append(float(approximate_kl.item()))
                logs["clip_fraction"].append(
                    float(((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item())
                )
                if approximate_kl.item() > 1.5 * cfg.target_kl:
                    stop = True
                    break
        return {key: float(np.mean(value)) for key, value in logs.items() if value}

    def set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    def save(self, path: str, **metadata: object) -> None:
        torch.save(
            {
                "config": asdict(self.config),
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metadata": metadata,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = False) -> dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return dict(checkpoint.get("metadata", {}))
