"""Multi-session terminal assignment and routing at one decision time.

The definitions follow docs/PROBLEM.md.  Everything here works on a
``Snapshot`` (pure data), so the verifier and the baselines can be tested on
small synthetic graphs as well as on the v3 constellation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

Edge = Tuple[int, int]
INF = math.inf
TOL = 1.0e-9


def canonical_edge(u: int, v: int) -> Edge:
    return (min(int(u), int(v)), max(int(u), int(v)))


@dataclass(frozen=True)
class Session:
    index: int
    source: str
    destination: str
    duration_s: float


@dataclass
class Snapshot:
    """Network state at decision time ``time_s``, independent of the sessions.

    ``persistent``: intra-plane links, always ACTIVE, infinite remaining contact.
    ``dynamic_remaining``: inter-plane pairs in contact at ``time_s`` (candidate
    dynamic links) and their remaining contact time.
    ``initial_dynamic``: ACTIVE dynamic links before the batch arrives
    (at most one per satellite).
    ``access``: per ground station, visible satellites with remaining GSL time,
    latest-ending first (the first entry is the v3 access satellite).
    """

    time_s: float
    persistent: Tuple[Edge, ...]
    dynamic_remaining: Dict[Edge, float]
    initial_dynamic: Tuple[Edge, ...]
    access: Dict[str, Tuple[Tuple[int, float], ...]]
    setup_s: float = 20.0
    _persistent_set: Set[Edge] = field(init=False, repr=False)
    adjacency_persistent: Dict[int, List[int]] = field(init=False, repr=False)
    adjacency_dynamic: Dict[int, List[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.persistent = tuple(canonical_edge(*e) for e in self.persistent)
        self.dynamic_remaining = {canonical_edge(*e): float(r) for e, r in self.dynamic_remaining.items()}
        self.initial_dynamic = tuple(canonical_edge(*e) for e in self.initial_dynamic)
        self._persistent_set = set(self.persistent)
        self.adjacency_persistent = {}
        for u, v in self.persistent:
            self.adjacency_persistent.setdefault(u, []).append(v)
            self.adjacency_persistent.setdefault(v, []).append(u)
        self.adjacency_dynamic = {}
        for u, v in self.dynamic_remaining:
            self.adjacency_dynamic.setdefault(u, []).append(v)
            self.adjacency_dynamic.setdefault(v, []).append(u)
        seen: Set[int] = set()
        for u, v in self.initial_dynamic:
            if (u, v) not in self.dynamic_remaining:
                raise ValueError(f"initial dynamic link {(u, v)} is not in contact")
            if u in seen or v in seen:
                raise ValueError("initial dynamic links share a terminal")
            seen.update((u, v))

    def is_persistent(self, edge: Edge) -> bool:
        return edge in self._persistent_set

    def remaining(self, edge: Edge) -> float:
        edge = canonical_edge(*edge)
        if edge in self._persistent_set:
            return INF
        return self.dynamic_remaining.get(edge, 0.0)


@dataclass(frozen=True)
class Instance:
    snapshot: Snapshot
    sessions: Tuple[Session, ...]
    antennas: int  # A: satellites each ground station tracks at once
    capacity: int  # kappa: sessions one ISL carries at once

    def antenna_sats(self, station: str) -> Tuple[Tuple[int, float], ...]:
        """The A latest-ending visible satellites (A=1 is the v3 access rule)."""
        return self.snapshot.access.get(station, ())[: self.antennas]

    def usable_access(self, station: str, required_s: float) -> Dict[int, float]:
        return {s: g for s, g in self.antenna_sats(station) if g + TOL >= required_s}


@dataclass
class Solution:
    routes: Dict[int, List[int]]  # session index -> satellites from ingress to egress
    final_dynamic: Set[Edge]  # M: ACTIVE dynamic links after the decision
    runtime_s: float = 0.0


def route_edges(path: Sequence[int]) -> List[Edge]:
    return [canonical_edge(path[i], path[i + 1]) for i in range(len(path) - 1)]


def verify(instance: Instance, solution: Solution) -> List[str]:
    """Every constraint of docs/PROBLEM.md. An empty list means feasible."""
    snap = instance.snapshot
    problems: List[str] = []
    final = {canonical_edge(*e) for e in solution.final_dynamic}
    new_links = final - set(snap.initial_dynamic)

    terminal_use: Dict[int, Edge] = {}
    for e in final:
        if e not in snap.dynamic_remaining:
            problems.append(f"dynamic link {e} is not in contact at decision time")
        for x in e:
            if x in terminal_use:
                problems.append(f"satellite {x} uses two dynamic links {terminal_use[x]} and {e}")
            terminal_use[x] = e

    sessions = {s.index: s for s in instance.sessions}
    load: Dict[Edge, int] = {}
    for k, path in solution.routes.items():
        if k not in sessions:
            problems.append(f"route for unknown session {k}")
            continue
        session = sessions[k]
        if not path:
            problems.append(f"session {k}: empty route")
            continue
        if len(set(path)) != len(path):
            problems.append(f"session {k}: route revisits a satellite")
        edges = route_edges(path)
        uses_new = any(e in new_links for e in edges)
        need = (snap.setup_s if uses_new else 0.0) + session.duration_s
        ingress = dict(instance.antenna_sats(session.source))
        egress = dict(instance.antenna_sats(session.destination))
        if path[0] not in ingress:
            problems.append(f"session {k}: ingress {path[0]} is not an antenna satellite of {session.source}")
        elif ingress[path[0]] + TOL < need:
            problems.append(f"session {k}: ingress GSL ends before the session ({ingress[path[0]]:.1f} < {need:.1f} s)")
        if path[-1] not in egress:
            problems.append(f"session {k}: egress {path[-1]} is not an antenna satellite of {session.destination}")
        elif egress[path[-1]] + TOL < need:
            problems.append(f"session {k}: egress GSL ends before the session ({egress[path[-1]]:.1f} < {need:.1f} s)")
        for e in edges:
            if not (snap.is_persistent(e) or e in final):
                problems.append(f"session {k}: link {e} is not ACTIVE")
                continue
            if snap.remaining(e) + TOL < need:
                problems.append(f"session {k}: link {e} ends before the session ({snap.remaining(e):.1f} < {need:.1f} s)")
            load[e] = load.get(e, 0) + 1
    for e, count in load.items():
        if count > instance.capacity:
            problems.append(f"link {e} carries {count} sessions > capacity {instance.capacity}")
    return problems


def evaluate(instance: Instance, solution: Solution) -> Dict[str, float]:
    snap = instance.snapshot
    final = {canonical_edge(*e) for e in solution.final_dynamic}
    new_links = final - set(snap.initial_dynamic)
    used: Set[Edge] = set()
    hops = 0
    delayed = 0
    for path in solution.routes.values():
        edges = route_edges(path)
        used.update(edges)
        hops += len(edges)
        delayed += int(any(e in new_links for e in edges))
    admitted = len(solution.routes)
    return {
        "sessions": len(instance.sessions),
        "admitted": admitted,
        "admitted_frac": admitted / max(1, len(instance.sessions)),
        "new_links": len(new_links),
        "new_links_used": len(new_links & used),
        "delayed_sessions": delayed,
        "mean_hops": hops / admitted if admitted else float("nan"),
        "total_hops": hops,
        "runtime_s": solution.runtime_s,
    }


def cut_upper_bound(instance: Instance) -> int:
    """Valid upper bound on admitted sessions from degree cuts.

    Every satellite has at most three ACTIVE ISLs (two persistent, one dynamic),
    each carrying at most ``capacity`` sessions.  Sessions leaving the antenna
    satellites of a station must cross the cut around them, so a station with
    usable antenna set S serves at most 3 * capacity * |S| sessions on either
    side (zero-hop sessions excepted, which need a shared antenna satellite).
    """
    by_source: Dict[str, List[Session]] = {}
    by_dest: Dict[str, List[Session]] = {}
    for s in instance.sessions:
        by_source.setdefault(s.source, []).append(s)
        by_dest.setdefault(s.destination, []).append(s)

    def side_bound(groups: Mapping[str, List[Session]], station_of) -> int:
        total = 0
        for station, items in groups.items():
            shortest = min(s.duration_s for s in items)
            usable = instance.usable_access(station, shortest)
            shared = 0
            for s in items:
                other = instance.usable_access(station_of(s), s.duration_s)
                if set(usable) & set(other):
                    shared += 1
            total += min(len(items), 3 * instance.capacity * len(usable) + shared)
        return total

    return min(
        len(instance.sessions),
        side_bound(by_source, lambda s: s.destination),
        side_bound(by_dest, lambda s: s.source),
    )
