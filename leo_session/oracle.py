"""A deliberately relaxed physical upper bound for early v3 experiments.

This is not the final constrained time-expanded oracle.  It gives the
controller every physically feasible inter-plane contact immediately and
ignores terminal occupancy and PAT.  Its feasible set therefore contains the
feasible set of every implemented online method.  It is useful as a sanity
ceiling, but it must be reported as ``Physical-Upper`` rather than ``Oracle``.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Set

from .config import DEFAULT_CONFIG, EnvironmentConfig
from .contacts import ContactTrace
from .ground import GroundContactTrace
from .topology import Edge, canonical_edge


class PhysicalUpperBound:
    def __init__(
        self,
        isl_trace: ContactTrace,
        ground_trace: GroundContactTrace,
        persistent_edges: Iterable[Edge],
        config: EnvironmentConfig = DEFAULT_CONFIG,
    ):
        self.isl_trace = isl_trace
        self.ground_trace = ground_trace
        self.persistent = {canonical_edge(*edge) for edge in persistent_edges}
        self.config = config

    def connected(self, source_station: str, destination_station: str, time_s: float) -> bool:
        sources = self.ground_trace.visible(source_station, time_s)
        destinations = set(self.ground_trace.visible(destination_station, time_s))
        if not sources or not destinations:
            return False
        if destinations.intersection(sources):
            return True

        adjacency: Dict[int, List[int]] = {}
        for u, v in self.persistent:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)
        for u, v, _end in self.isl_trace.active_edges(time_s):
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)

        queue = deque(sources)
        visited: Set[int] = set(sources)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, ()):
                if neighbor in visited:
                    continue
                if neighbor in destinations:
                    return True
                visited.add(neighbor)
                queue.append(neighbor)
        return False

    def session_feasible(
        self,
        source_station: str,
        destination_station: str,
        start_s: float,
        duration_s: float,
    ) -> bool:
        step = self.config.oracle_sample_step_s
        end = float(start_s) + float(duration_s)
        times = [float(start_s)]
        now = float(start_s) + step
        while now < end - 1.0e-9:
            times.append(now)
            now += step
        times.append(end)
        return all(self.connected(source_station, destination_station, t) for t in times)
