"""Initial ACTIVE topology required before a session arrives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import DEFAULT_CONFIG, EnvironmentConfig
from .contacts import ContactTrace
from .orbit import WalkerDeltaConstellation


Edge = Tuple[int, int]


def canonical_edge(u: int, v: int) -> Edge:
    return (min(int(u), int(v)), max(int(u), int(v)))


@dataclass
class ActiveTopology:
    persistent: Set[Edge]
    dynamic: Set[Edge]
    dynamic_partner: Dict[int, int]

    @classmethod
    def from_edges(cls, persistent: Iterable[Edge], dynamic: Iterable[Edge]) -> "ActiveTopology":
        pset = {canonical_edge(*edge) for edge in persistent}
        dset = {canonical_edge(*edge) for edge in dynamic}
        partner: Dict[int, int] = {}
        for u, v in dset:
            if u in partner or v in partner:
                raise ValueError("dynamic terminal conflict in initial topology")
            partner[u] = v
            partner[v] = u
        return cls(pset, dset, partner)

    def active_edges(self) -> Set[Edge]:
        return set(self.persistent) | set(self.dynamic)

    def clone(self) -> "ActiveTopology":
        """Independent mutable copy for one routing method/session run."""
        return ActiveTopology(set(self.persistent), set(self.dynamic), dict(self.dynamic_partner))

    def max_degree(self, node_count: int) -> int:
        degree = np.zeros(node_count, dtype=np.int16)
        for u, v in self.active_edges():
            degree[u] += 1
            degree[v] += 1
        return int(degree.max(initial=0))


def build_initial_topology(
    constellation: WalkerDeltaConstellation,
    trace: ContactTrace,
    request_time_s: float,
    config: EnvironmentConfig = DEFAULT_CONFIG,
) -> ActiveTopology:
    """Build a neutral, session-independent initial ACTIVE topology.

    Each plane sends even-indexed satellites to odd-indexed satellites in the
    next plane.  This partitions terminals without conflicts and balances all
    adjacent plane interfaces.  On each interface, a Hungarian assignment
    maximizes feasible cardinality first, then minimizes current link length.
    A candidate must have been continuously feasible for the preceding setup
    time, so declaring it ACTIVE at session arrival is physically consistent.
    """
    qn = config.satellites_per_plane
    persistent = [tuple(map(int, edge)) for edge in constellation.persistent_edges()]
    dynamic: List[Edge] = []
    infeasible_cost = 1.0e12

    for p in range(config.orbital_planes):
        pn = (p + 1) % config.orbital_planes
        left = [constellation.node_id(p, q) for q in range(0, qn, 2)]
        right = [constellation.node_id(pn, q) for q in range(1, qn, 2)]
        cost = np.full((len(left), len(right)), infeasible_cost, dtype=np.float64)
        for i, u in enumerate(left):
            for j, v in enumerate(right):
                if trace.age_s(u, v, request_time_s) + 1.0e-9 < config.dynamic_setup_s:
                    continue
                distance, _ = constellation.link_geometry(u, v, request_time_s)
                cost[i, j] = distance
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] < infeasible_cost:
                dynamic.append(canonical_edge(left[i], right[j]))

    topology = ActiveTopology.from_edges(persistent, dynamic)
    if topology.max_degree(config.total_satellites) > 3:
        raise AssertionError("initial topology violates the maximum degree 3 model")
    return topology
