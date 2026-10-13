"""Sequential admission baselines for the multi-session problem.

Both baselines admit sessions one at a time in the given order and never
revisit an earlier decision.

* ``fixed_topology`` keeps the initial dynamic links (no reconfiguration) and
  gives each session a minimum-hop path on links with spare capacity that stay
  in contact for the whole session.
* ``greedy_reconfiguration`` first tries the same thing; if no such path
  exists it may point free dynamic terminals at new partners (20 s setup,
  break-before-make).  A terminal is free when its current dynamic link carries
  no admitted session.  Among paths it minimises (new links, hops).
"""
from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple

from .model import Edge, Instance, Solution, TOL, canonical_edge


class _State:
    def __init__(self, instance: Instance):
        snap = instance.snapshot
        self.instance = instance
        self.load: Dict[Edge, int] = {}
        self.matching: Set[Edge] = set(snap.initial_dynamic)
        self.created: Set[Edge] = set()  # links switched on in this decision: usable only after setup
        self.partner: Dict[int, int] = {}
        for u, v in self.matching:
            self.partner[u] = v
            self.partner[v] = u

    def has_room(self, edge: Edge) -> bool:
        return self.load.get(edge, 0) < self.instance.capacity

    def terminal_free(self, x: int) -> bool:
        p = self.partner.get(x)
        return p is None or self.load.get(canonical_edge(x, p), 0) == 0

    def search(self, sources: Dict[int, float], goals: Dict[int, float], need_s: float,
               allow_new: bool) -> Optional[Tuple[List[int], Set[Edge]]]:
        """Lexicographic (new links, hops) shortest path.

        State = (satellite, entered through a dynamic link).  A satellite has one
        dynamic terminal, so a path may not enter and leave it through dynamic
        links (same rule as the v3 route search).  Routes must be simple, so a
        neighbour already on the search-tree path is skipped; with this
        path-dependent rule the search is a heuristic, not an exact
        constrained shortest path.  ``allow_new`` also marks a
        delayed session (setup + X): only then may it use links switched on
        earlier in this decision, since those become ACTIVE after the setup.
        """
        snap = self.instance.snapshot
        dist: Dict[Tuple[int, bool], Tuple[int, int]] = {}
        parent: Dict[Tuple[int, bool], Tuple[Tuple[int, bool], bool]] = {}
        heap: List[Tuple[int, int, int, bool]] = []
        for s in sources:
            dist[(s, False)] = (0, 0)
            heapq.heappush(heap, (0, 0, s, False))
        goal = None
        while heap:
            new_count, hops, u, entered_dyn = heapq.heappop(heap)
            state = (u, entered_dyn)
            if dist.get(state) != (new_count, hops):
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
                    steps.append((v, False, False))
            if not entered_dyn:
                p = self.partner.get(u)
                if p is not None:
                    e = canonical_edge(u, p)
                    ready = allow_new or e not in self.created
                    if ready and self.has_room(e) and snap.remaining(e) + TOL >= need_s:
                        steps.append((p, True, False))
                if allow_new and self.terminal_free(u):
                    for v in snap.adjacency_dynamic.get(u, ()):
                        e = canonical_edge(u, v)
                        if e in self.matching or not self.terminal_free(v):
                            continue
                        if snap.remaining(e) + TOL >= need_s:
                            steps.append((v, True, True))
            for v, via_dyn, is_new in steps:
                if v in on_path:  # routes are simple paths (no satellite visited twice)
                    continue
                nxt = (v, via_dyn)
                cand = (new_count + int(is_new), hops + 1)
                if cand < dist.get(nxt, (1 << 30, 1 << 30)):
                    dist[nxt] = cand
                    parent[nxt] = (state, is_new)
                    heapq.heappush(heap, (cand[0], cand[1], v, via_dyn))
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


def _run(instance: Instance, allow_reconfiguration: bool) -> Solution:
    start = time.perf_counter()
    state = _State(instance)
    routes: Dict[int, List[int]] = {}
    setup = instance.snapshot.setup_s
    for session in instance.sessions:
        need = session.duration_s
        found = state.search(instance.usable_access(session.source, need),
                             instance.usable_access(session.destination, need), need, allow_new=False)
        if found is None and allow_reconfiguration:
            need = setup + session.duration_s
            found = state.search(instance.usable_access(session.source, need),
                                 instance.usable_access(session.destination, need), need, allow_new=True)
        if found is None:
            continue
        path, new_edges = found
        state.admit(path, new_edges)
        routes[session.index] = path
    return Solution(routes, set(state.matching), time.perf_counter() - start)


def fixed_topology(instance: Instance) -> Solution:
    return _run(instance, allow_reconfiguration=False)


def greedy_reconfiguration(instance: Instance) -> Solution:
    return _run(instance, allow_reconfiguration=True)
