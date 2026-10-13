"""Sparse graph observations for the v3 sequential relay policy."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from leo_session.routing import RelayPolicyDecision, RoutePlanner
from leo_session.topology import canonical_edge


BASE_NODE_FEATURE_DIM = 13
NODE_FEATURE_DIM = 16
VISIT_ORDER_INDEX = 13
PREVIOUS_NODE_INDEX = 14
HOP_FRACTION_INDEX = 15
GRAPH_EDGE_FEATURE_DIM = 8
ACTION_FEATURE_DIM = 8


@dataclass(frozen=True)
class PolicyObservation:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    current: int
    egress: int
    start: int
    previous: int


def build_policy_observation(
    planner: RoutePlanner,
    decision: RelayPolicyDecision,
    *,
    include_remaining_contact: bool = True,
    include_route_history: bool = True,
    include_current_route_state: bool = True,
    static_observation: PolicyObservation | None = None,
) -> PolicyObservation:
    """Convert one policy decision into normalized sparse graph arrays."""
    config = planner.config
    n = config.total_satellites
    time_s = decision.time_s
    required_s = decision.required_s
    current = decision.state.current
    egress = decision.destination_satellite
    visited = np.zeros(n, dtype=np.float32)
    visited[list(decision.state.satellites)] = 1.0
    node_dim = NODE_FEATURE_DIM if include_route_history else BASE_NODE_FEATURE_DIM
    if static_observation is not None:
        if static_observation.node_features.shape != (n, node_dim):
            raise ValueError("cached observation has incompatible node shape")
        node = static_observation.node_features.copy()
        node[:, 7] = 0.0
        node[current, 7] = 1.0
        node[:, 10] = visited
        edge_index = static_observation.edge_index
        edge_features = static_observation.edge_features
    else:
        positions = planner.constellation.positions(time_s).astype(np.float32)
        position_scale = max(config.orbital_radius_km, 1.0)
        destination_position = positions[egress]
        delta = destination_position[None, :] - positions
        distance = np.linalg.norm(delta, axis=1)
        unit = delta / np.maximum(distance[:, None], 1.0e-6)
        occupied = np.zeros(n, dtype=np.float32)
        for u, entries in decision.graph.items():
            if any(is_dynamic and is_active for _v, is_dynamic, is_active, _r in entries):
                occupied[int(u)] = 1.0
        node = np.zeros((n, node_dim), dtype=np.float32)
        node[:, 0:3] = positions / position_scale
        node[:, 3:6] = unit
        node[:, 6] = distance / max(decision.source_egress_km, 1.0)
        node[current, 7] = 1.0
        node[decision.state.satellites[0], 8] = 1.0
        node[egress, 9] = 1.0
        node[:, 10] = visited
        node[:, 11] = occupied
        node[:, 12] = min(
            1.0, required_s / max(planner.constellation.period_s, 1.0)
        )
    if include_route_history:
        node[:, VISIT_ORDER_INDEX: HOP_FRACTION_INDEX + 1] = 0.0
        max_actions = max(1, planner.max_relay_actions)
        for selection_index, satellite in enumerate(decision.state.satellites, 1):
            node[int(satellite), VISIT_ORDER_INDEX] = min(
                1.0, selection_index / max_actions
            )
        if len(decision.state.satellites) >= 2:
            node[int(decision.state.satellites[-2]), PREVIOUS_NODE_INDEX] = 1.0
        node[:, HOP_FRACTION_INDEX] = min(
            1.0, (len(decision.state.satellites) - 1) / max_actions
        )

    norm_time = max(required_s, config.dynamic_setup_s, 1.0)
    current_route_edges = set(decision.current_route_edges)
    current_dynamic_nodes = {
        node_id
        for edge in current_route_edges
        if edge not in planner.persistent
        for node_id in edge
    }

    def breaks_current_route(u: int, v: int, is_dynamic: bool, is_active: bool) -> float:
        if not include_current_route_state or not is_dynamic or is_active:
            return 0.0
        return float(int(u) in current_dynamic_nodes or int(v) in current_dynamic_nodes)

    if static_observation is None:
        senders: list[int] = []
        receivers: list[int] = []
        graph_features: list[list[float]] = []
        for u, entries in decision.graph.items():
            for v, is_dynamic, is_active, remaining in entries:
                is_new = bool(is_dynamic and not is_active)
                usable = math.inf if not is_dynamic else max(
                    0.0, remaining - (config.dynamic_setup_s if is_new else 0.0)
                )
                x_feasible = float(not is_dynamic or usable + 1.0e-9 >= required_s)
                remaining_norm = (
                    1.0 if not is_dynamic else min(1.0, remaining / norm_time)
                )
                margin = 1.0 if not is_dynamic else max(
                    -1.0, min(1.0, (usable - required_s) / norm_time)
                )
                if not include_remaining_contact:
                    remaining_norm = margin = x_feasible = 0.0
                senders.append(int(u))
                receivers.append(int(v))
                if include_current_route_state:
                    state_features = [
                        float(is_dynamic),
                        float(is_active),
                        float(canonical_edge(int(u), int(v)) in current_route_edges),
                        breaks_current_route(
                            int(u), int(v), bool(is_dynamic), bool(is_active)
                        ),
                    ]
                else:
                    state_features = [
                        float(not is_dynamic),
                        float(is_dynamic),
                        float(is_active),
                        float(is_new),
                    ]
                graph_features.append(
                    state_features
                    + [
                        float(np.linalg.norm(positions[int(u)] - positions[int(v)]))
                        / config.lisl_range_km,
                        remaining_norm,
                        x_feasible,
                        margin,
                    ]
                )
        edge_index = np.asarray([senders, receivers], dtype=np.int64)
        edge_features = np.asarray(graph_features, dtype=np.float32)
    action_features = np.zeros((n, ACTION_FEATURE_DIM), dtype=np.float32)
    action_mask = np.zeros(n, dtype=np.bool_)
    for candidate in decision.candidates:
        usable = math.inf if not candidate.is_dynamic else max(
            0.0,
            candidate.remaining_s
            - (config.dynamic_setup_s if candidate.is_new_dynamic else 0.0),
        )
        remaining_norm = (
            1.0
            if not candidate.is_dynamic
            else min(1.0, candidate.remaining_s / norm_time)
        )
        margin = (
            1.0
            if not candidate.is_dynamic
            else max(-1.0, min(1.0, (usable - required_s) / norm_time))
        )
        if not include_remaining_contact:
            remaining_norm = margin = 0.0
        if include_current_route_state:
            state_features = [
                float(candidate.is_dynamic),
                float(candidate.is_active),
                float(candidate.edge in current_route_edges),
                breaks_current_route(
                    decision.state.current,
                    candidate.neighbor,
                    candidate.is_dynamic,
                    candidate.is_active,
                ),
            ]
        else:
            state_features = [
                float(not candidate.is_dynamic),
                float(candidate.is_dynamic),
                float(candidate.is_active),
                float(candidate.is_new_dynamic),
            ]
        action_features[candidate.neighbor] = np.asarray(
            state_features
            + [
                candidate.link_length_km / config.lisl_range_km,
                remaining_norm,
                margin,
                candidate.normalized_progress,
            ],
            dtype=np.float32,
        )
        action_mask[candidate.neighbor] = True

    return PolicyObservation(
        node_features=node,
        edge_index=edge_index,
        edge_features=edge_features,
        action_features=action_features,
        action_mask=action_mask,
        current=int(current),
        egress=int(egress),
        start=int(decision.state.satellites[0]),
        previous=int(
            decision.state.satellites[-2]
            if len(decision.state.satellites) >= 2
            else current
        ),
    )
