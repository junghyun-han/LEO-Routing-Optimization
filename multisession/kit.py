"""Building blocks for constructive heuristics.

The baselines and every LLM-written heuristic use the same blocks, so any
difference in results comes from the strategy (session order, link costs,
when to reconfigure, rip-up and reroute), not from the plumbing.  Whatever a
heuristic does, its final ``Solution`` is checked by ``model.verify``.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from .model import Edge, Instance, Session, Solution, TOL, canonical_edge


@dataclass(frozen=True)
class EdgeInfo:
    u: int
    v: int
    kind: str  # "persistent", "dynamic" (ACTIVE dynamic link) or "new" (switch two free terminals on)
    remaining_s: float  # remaining contact time (inf for persistent links)
    load: int  # sessions already on the link
    capacity: int


EdgeCost = Callable[[EdgeInfo], float]


def default_cost(info: EdgeInfo) -> float:
    """Fewest new links first, then fewest hops."""
    return 1.0e6 if info.kind == "new" else 1.0


class State:
    """Loads, dynamic links, and terminal partners during one decision."""

    def __init__(self, instance: Instance):
        snap = instance.snapshot
        self.instance = instance
        self.load: Dict[Edge, int] = {}
        self.matching: Set[Edge] = set(snap.initial_dynamic)
        self.created: Set[Edge] = set()  # switched on in this decision: usable only by delayed sessions
        self.partner: Dict[int, int] = {}
        for u, v in self.matching:
            self.partner[u] = v
            self.partner[v] = u

    def has_room(self, edge: Edge) -> bool:
        return self.load.get(edge, 0) < self.instance.capacity

    def terminal_free(self, x: int) -> bool:
        p = self.partner.get(x)
        return p is None or self.load.get(canonical_edge(x, p), 0) == 0

    def search(self, sources: Dict[int, float], goals: Dict[int, float], need_s: float, allow_new: bool,
               edge_cost: EdgeCost = default_cost) -> Optional[Tuple[List[int], Set[Edge]]]:
        """Cheapest simple path from any source to any goal satellite.

        Only links with spare capacity and remaining contact >= ``need_s`` are
        considered.  ``allow_new`` marks a delayed session (setup + X): it may
        switch free terminals on and use links switched on earlier in this
        decision.  State = (satellite, entered through a dynamic link): a
        satellite has one dynamic terminal, so a route never enters and leaves
        it through dynamic links.  A neighbour already on the search-tree path
        is skipped, so routes are simple; with this path-dependent rule the
        search is a heuristic.  ``edge_cost`` must return a non-negative number
        (``math.inf`` forbids the link).
        """
        snap = self.instance.snapshot
        cap = self.instance.capacity
        dist: Dict[Tuple[int, bool], float] = {}
        parent: Dict[Tuple[int, bool], Tuple[Tuple[int, bool], bool]] = {}
        heap: List[Tuple[float, int, bool]] = []
        for s in sources:
            dist[(s, False)] = 0.0
            heapq.heappush(heap, (0.0, s, False))
        goal = None
        while heap:
            cost, u, entered_dyn = heapq.heappop(heap)
            state = (u, entered_dyn)
            if dist.get(state) != cost:
                continue
            if u in goals:
                goal = state
                break
            on_path = {u}
            walk = state
            while walk in parent:
                walk = parent[walk][0]
                on_path.add(walk[0])
            steps = []
            for v in snap.adjacency_persistent.get(u, ()):
                e = canonical_edge(u, v)
                if self.has_room(e):
                    steps.append((v, False, False, EdgeInfo(u, v, "persistent", math.inf, self.load.get(e, 0), cap)))
            if not entered_dyn:
                p = self.partner.get(u)
                if p is not None:
                    e = canonical_edge(u, p)
                    ready = allow_new or e not in self.created
                    r = snap.remaining(e)
                    if ready and self.has_room(e) and r + TOL >= need_s:
                        steps.append((p, True, False, EdgeInfo(u, p, "dynamic", r, self.load.get(e, 0), cap)))
                if allow_new and self.terminal_free(u):
                    for v in snap.adjacency_dynamic.get(u, ()):
                        e = canonical_edge(u, v)
                        if e in self.matching or not self.terminal_free(v):
                            continue
                        r = snap.remaining(e)
                        if r + TOL >= need_s:
                            steps.append((v, True, True, EdgeInfo(u, v, "new", r, 0, cap)))
            for v, via_dyn, is_new, info in steps:
                if v in on_path:
                    continue
                c = float(edge_cost(info))
                if not c >= 0.0 or math.isinf(c):  # negative, NaN, or forbidden
                    continue
                nxt = (v, via_dyn)
                cand = cost + c
                if cand < dist.get(nxt, math.inf):
                    dist[nxt] = cand
                    parent[nxt] = (state, is_new)
                    heapq.heappush(heap, (cand, v, via_dyn))
        if goal is None:
            return None
        path = [goal[0]]
        new_edges: Set[Edge] = set()
        state = goal
        while state in parent:
            prev, is_new = parent[state]
            if is_new:
                new_edges.add(canonical_edge(prev[0], state[0]))
            path.append(prev[0])
            state = prev
        path.reverse()
        return path, new_edges

    def admit(self, path: List[int], new_edges: Set[Edge]) -> None:
        """Commit a route found by ``search`` (switches its new links on)."""
        for u, v in new_edges:
            for x in (u, v):
                p = self.partner.pop(x, None)
                if p is not None:
                    self.partner.pop(p, None)
                    self.matching.discard(canonical_edge(x, p))
            self.matching.add((u, v))
            self.created.add((u, v))
            self.partner[u] = v
            self.partner[v] = u
        for i in range(len(path) - 1):
            e = canonical_edge(path[i], path[i + 1])
            self.load[e] = self.load.get(e, 0) + 1

    def remove(self, path: List[int]) -> None:
        """Release an admitted route's capacity (its links stay ACTIVE)."""
        for i in range(len(path) - 1):
            e = canonical_edge(path[i], path[i + 1])
            self.load[e] -= 1


def route_session(state: State, session: Session, edge_cost: EdgeCost = default_cost,
                  allow_reconfiguration: bool = True) -> Optional[Tuple[List[int], Set[Edge]]]:
    """Immediate route if one exists, otherwise (optionally) a delayed route with new links."""
    inst = state.instance
    need = session.duration_s
    found = state.search(inst.usable_access(session.source, need),
                         inst.usable_access(session.destination, need), need, False, edge_cost)
    if found is None and allow_reconfiguration:
        need = inst.snapshot.setup_s + session.duration_s
        found = state.search(inst.usable_access(session.source, need),
                             inst.usable_access(session.destination, need), need, True, edge_cost)
    return found


def finish(state: State, routes: Dict[int, List[int]], start_time: Optional[float] = None) -> Solution:
    runtime = 0.0 if start_time is None else time.perf_counter() - start_time
    return Solution({k: list(p) for k, p in routes.items()}, set(state.matching), runtime)
