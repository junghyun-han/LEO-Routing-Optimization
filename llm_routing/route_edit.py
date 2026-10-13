"""Dijkstra route + its neighbourhood as text, and validation of an edited route.

At every route decision (initial and proactive reroute, exactly as in the WCNC
protocol) the LLM receives the Dijkstra route, the satellites within ``hops`` of
it, and every link among them with its kind and remaining contact time.  It
returns a full ingress-to-egress satellite list.  ``to_route`` accepts the list
only if the v3 route rules hold: it starts at the common ingress and ends at
the common egress satellite, consecutive satellites share a currently feasible
link, no satellite repeats, no satellite enters and leaves through dynamic
links.  These are the structural rules the Dijkstra baseline also satisfies.
Whether a link lasts long enough (for example a new link whose contact ends
before its 20 s setup) is not checked here: the simulator decides the outcome,
exactly as for Dijkstra, and ``risky_links`` reports such links as warnings.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from leo_session.routing import Route
from leo_session.topology import canonical_edge

Graph = Dict[int, Sequence[Tuple[int, bool, bool, float]]]  # u -> (v, is_dynamic, is_active, remaining_s)

LEGEND = (
    "Each line: satellite | lat,lon | links to other listed satellites as id:kind:remaining_s:km. "
    "kind P = persistent in-plane link (never expires), A = active inter-plane link (usable now), "
    "N = inter-plane link that must be switched on (20 s setup; the route then starts 20 s later). "
    "remaining_s = seconds until the link's contact ends. A satellite has one inter-plane terminal, "
    "so a route may not enter and leave a satellite through inter-plane (A or N) links."
)


def neighbourhood(graph: Graph, route: Sequence[int], hops: int = 1) -> List[int]:
    nodes = set(route)
    frontier = set(route)
    for _ in range(hops):
        frontier = {v for u in frontier for v, *_ in graph.get(u, ())} - nodes
        nodes |= frontier
    return sorted(nodes)


def describe(graph: Graph, nodes: Sequence[int], positions: np.ndarray, latlon: np.ndarray) -> str:
    keep = set(nodes)
    lines = []
    for u in nodes:
        links = []
        for v, dynamic, active, remaining in graph.get(u, ()):
            if v not in keep:
                continue
            kind = "P" if not dynamic else ("A" if active else "N")
            rem = "inf" if math.isinf(remaining) else str(int(remaining))
            links.append(f"{v}:{kind}:{rem}:{int(np.linalg.norm(positions[u] - positions[v]))}")
        lines.append(f"{u} | {latlon[u][0]:.0f},{latlon[u][1]:.0f} | " + " ".join(links))
    return "\n".join(lines)


def risky_links(planner, graph: Graph, satellites: Sequence[int], required_s: float) -> List[str]:
    """Links on the route that end before the session would (warning, not a rejection)."""
    setup = planner.config.dynamic_setup_s
    out = []
    for u, v in zip(satellites, satellites[1:]):
        hit = next(((d, a, r) for n, d, a, r in graph.get(u, ()) if n == v), None)
        if hit is None or not hit[0]:
            continue
        dynamic, active, remaining = hit
        if not active and remaining + 1.0e-9 < setup:
            out.append(f"{u}-{v} ends before its setup")
        elif remaining + 1.0e-9 < required_s:
            out.append(f"{u}-{v} ends {required_s - remaining:.0f} s early")
    return out


def to_route(planner, graph: Graph, satellites: Sequence[int], source: str, destination: str,
             time_s: float, active_dynamic) -> Tuple[Optional[Route], str]:
    sats = [int(s) for s in satellites]
    ingress = planner.ground_trace.selected_access(source, time_s)
    egress = planner.ground_trace.selected_access(destination, time_s)
    if not sats or sats[0] != ingress or sats[-1] != egress:
        return None, f"route must start at ingress {ingress} and end at egress {egress}"
    if len(set(sats)) != len(sats):
        return None, "a satellite appears twice"
    links = []
    for u, v in zip(sats, sats[1:]):
        hit = next(((d, a, r) for n, d, a, r in graph.get(u, ()) if n == v), None)
        if hit is None:
            return None, f"no feasible link {u}-{v} at this time"
        links.append(((u, v), *hit))
    for (_, d_in, _, _), (_, d_out, _, _) in zip(links, links[1:]):
        if d_in and d_out:
            return None, "a satellite enters and leaves through inter-plane links"
    new_edges = tuple(canonical_edge(u, v) for (u, v), d, a, _ in links
                      if d and canonical_edge(u, v) not in active_dynamic)
    remaining = [planner.ground_trace.remaining_s(source, sats[0], time_s),
                 planner.ground_trace.remaining_s(destination, sats[-1], time_s)]
    remaining += [r for (u, v), d, a, r in links if d]
    return Route(tuple(sats), sats[0], sats[-1], new_edges, min(remaining, default=math.inf)), ""
