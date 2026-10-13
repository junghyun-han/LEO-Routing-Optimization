"""Active reward configuration and local relay reward for v3 PPO-GCN."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Single reward configuration shared by Cost-Greedy and PPO-GCN."""

    relay_progress_weight: float = 1.0
    relay_link_weight: float = 0.05
    relay_lifetime_shortfall_weight: float = 3.0
    relay_pat_weight: float = 0.15
    route_success_reward: float = 5.0
    route_failure_penalty: float = 5.0
    route_service_reward: float = 5.0
    route_interruption_penalty: float = 5.0
    session_completion_reward: float = 10.0
    session_incomplete_penalty: float = 5.0
    session_setup_failure_penalty: float = 10.0

    def __post_init__(self) -> None:
        if (
            self.session_completion_reward <= 0.0
            or self.session_incomplete_penalty <= 0.0
            or self.session_setup_failure_penalty <= self.session_incomplete_penalty
        ):
            raise ValueError(
                "session rewards require positive completion/incomplete values "
                "and setup failure must be the larger penalty"
            )
        relay_weights = (
            self.relay_progress_weight,
            self.relay_link_weight,
            self.relay_lifetime_shortfall_weight,
            self.relay_pat_weight,
        )
        if any(weight < 0.0 for weight in relay_weights):
            raise ValueError("relay reward weights must be non-negative")
        if sum(relay_weights) <= 0.0:
            raise ValueError("relay reward needs at least one routing term")
        if (
            self.route_success_reward <= 0.0
            or self.route_failure_penalty < 0.0
            or self.route_service_reward < 0.0
            or self.route_interruption_penalty < 0.0
        ):
            raise ValueError("route success reward must be positive and route costs non-negative")


@dataclass(frozen=True)
class RelayRewardBreakdown:
    """The official one-step reward used by both greedy and learned policies."""

    geometric_progress_reward: float
    link_length_cost: float
    lifetime_shortfall_cost: float
    pat_setup_cost: float
    total_cost: float
    reward: float


def relay_penalty_terms(
    *,
    is_dynamic: bool,
    is_new_dynamic: bool,
    remaining_s: float,
    required_s: float,
    pat_s: float,
    pat_reference_s: float = 20.0,
) -> tuple[float, float]:
    """Return local ``(q_X, q_P)`` without any graph search."""
    if min(remaining_s, required_s, pat_s) < 0.0 or pat_reference_s <= 0.0:
        raise ValueError("relay lifetime/PAT inputs must be non-negative")
    if not is_dynamic:
        return 0.0, 0.0
    setup = float(pat_s) if is_new_dynamic else 0.0
    usable = max(0.0, float(remaining_s) - setup)
    denominator = max(float(required_s), float(pat_s), 1.0)
    shortfall = min(
        1.0,
        max(0.0, float(required_s) - usable) / denominator,
    )
    return shortfall, setup / float(pat_reference_s)


def relay_reward(
    *,
    normalized_progress: float,
    normalized_link_length: float,
    lifetime_shortfall: float,
    normalized_pat_cost: float,
    config: RewardConfig = RewardConfig(),
) -> RelayRewardBreakdown:
    """Evaluate the common v3 relay transition cost and its exact negative.

    Cost-Greedy maximizes this one-step reward. PPO-GCN receives the same
    reward at the same transition and maximizes its cumulative return.
    """
    if not 0.0 <= lifetime_shortfall <= 1.0:
        raise ValueError("lifetime_shortfall must be in [0, 1]")
    if normalized_pat_cost < 0.0:
        raise ValueError("normalized_pat_cost must be non-negative")
    if normalized_link_length < 0.0:
        raise ValueError("normalized_link_length must be non-negative")
    geometric = config.relay_progress_weight * float(normalized_progress)
    link = config.relay_link_weight * float(normalized_link_length)
    lifetime = config.relay_lifetime_shortfall_weight * float(lifetime_shortfall)
    pat = config.relay_pat_weight * float(normalized_pat_cost)
    cost = -geometric + link + lifetime + pat
    return RelayRewardBreakdown(
        geometric_progress_reward=geometric,
        link_length_cost=link,
        lifetime_shortfall_cost=lifetime,
        pat_setup_cost=pat,
        total_cost=cost,
        reward=-cost,
    )
