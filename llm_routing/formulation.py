"""Multi-constraint session routing as a CP-SAT model (docs/LLM_ROUTING.md, section 15).

``decision_data`` is the data an LLM-written model receives at every route
decision; ``reference_route`` is the author's model on the same data.  With
several side constraints at once (forbidden region, delay bound, new-link
budget, link lifetime) the problem is a resource-constrained shortest path,
NP-hard in general, so it is solved exactly with CP-SAT.

Model (directed arc variables x[u, v] on usable links):
* flow: one unit leaves the ingress and enters the egress; every satellite has
  at most one incoming and one outgoing route arc (no branching);
* terminal rule: at most one inter-plane arc (kind A or N) touches a satellite;
* lifetime: a link is usable only if its remaining contact covers the session
  time (plus the 20 s setup for a link that must be switched on);
* forbidden region: relays over the region at any time of the session are excluded;
* delay: (ground links + route length) / c <= bound;
* new links: switched-on links on the route <= remaining budget;
* objective: fewest hops.
Cycles disjoint from the path only add hops, so an optimal solution has none.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

import time

from leo_session.ground import GROUND_STATIONS
from leo_session.routing import RoutePlanner
from leo_session.topology import canonical_edge
from llm_routing.regions import KO_REGION
from llm_routing.regions import REGIONS, satellite_latlon

from .route_edit import to_route

C_KM_S = 299792.458


def decision_data(planner, graph, source: str, destination: str, time_s: float, required_s: float,
                  regions: Optional[List[str]] = None, new_links_used: int = 0) -> Dict:
    """Network data at one decision.  ``regions=None`` gives every region box (what an LLM-written model gets)."""
    regions = sorted(REGIONS) if regions is None else list(regions)
    ingress = planner.ground_trace.selected_access(source, time_s)
    egress = planner.ground_trace.selected_access(destination, time_s)
    pos = planner.constellation.positions(time_s)
    radius = planner.constellation.config.earth_radius_km
    links, seen = [], set()
    for u, nbrs in graph.items():
        for v, dynamic, active, remaining in nbrs:
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            links.append({"u": int(key[0]), "v": int(key[1]),
                          "kind": "P" if not dynamic else ("A" if active else "N"),
                          "remaining_s": float(remaining) if not math.isinf(remaining) else None,
                          "km": round(float(np.linalg.norm(pos[u] - pos[v])), 1)})
    window_end = time_s + required_s + planner.config.dynamic_setup_s
    over = {r: np.zeros(len(pos), dtype=bool) for r in regions}
    for t in np.arange(time_s, window_end + 1e-9, 10.0):  # same 10 s sampling as the tools and compliance
        ll = np.asarray(satellite_latlon(planner.constellation, float(t)))
        for r in regions:
            lat_lo, lat_hi, lon_lo, lon_hi = REGIONS[r]
            over[r] |= (ll[:, 0] >= lat_lo) & (ll[:, 0] <= lat_hi) & (ll[:, 1] >= lon_lo) & (ll[:, 1] <= lon_hi)
    region_sats = {r: sorted(int(s) for s in np.flatnonzero(m) if s not in (ingress, egress)) for r, m in over.items()}
    return {
        "ingress": int(ingress), "egress": int(egress), "required_s": float(required_s),
        "setup_s": float(planner.config.dynamic_setup_s),
        "ground_link_km": round(float(np.linalg.norm(pos[ingress] - GROUND_STATIONS[source].eci(time_s, radius)))
                                + float(np.linalg.norm(pos[egress] - GROUND_STATIONS[destination].eci(time_s, radius))), 1),
        "speed_of_light_km_s": C_KM_S,
        "links": links,
        "region_satellites": region_sats,
        "region_names": {r: KO_REGION[r] for r in regions},
        "n_new_links_so_far": int(new_links_used),
    }


DATA_DOC = """data (a dict) describes the network at this decision. Fields and types:
- data["ingress"], data["egress"] (int): the route must start at the ingress satellite and end at the egress satellite.
- data["links"] (list of dict): every usable undirected link, {"u": int, "v": int, "kind": "P" | "A" | "N",
  "remaining_s": float or None, "km": float}. kind "P" = in-plane link (never expires; remaining_s is None); "A" = active
  inter-plane link; "N" = inter-plane link that must be switched on (a PAT re-pointing with data["setup_s"] = 20 s
  setup), so an N link must last setup_s longer than required_s. remaining_s = seconds until the link's contact ends;
  km = current link length.
- data["required_s"] (float): seconds the route must stay connected.
- data["ground_link_km"] (float): length of the two ground links; one-way delay in ms = (ground_link_km + sum of the
  route's link km) / data["speed_of_light_km_s"] * 1000.
- data["region_satellites"] (dict: region key -> list of int): satellites that are over that region at some time during
  the session (ingress and egress excluded; they are fixed by the ground stations). data["region_names"] (dict: region
  key -> Korean name).
- data["n_new_links_so_far"] (int): how many links earlier route decisions of this session switched on (0 at the start).
Routing rules: a route is a list of satellite ids (int) from ingress to egress; consecutive satellites are joined by a
link in data["links"]; no satellite appears twice. Each satellite has one inter-plane terminal: at every satellite of the
route, at most one of the route links touching it may be inter-plane (A or N); other satellites may each use their own."""


def reference_route(data: Dict, avoid_region: Optional[str] = None, max_delay_ms: Optional[float] = None,
                    max_new_links: Optional[int] = None, require_lifetime: bool = True, objective: str = "hops",
                    time_limit_s: float = 20.0, hint_route: Optional[List[int]] = None) -> Dict:
    from ortools.sat.python import cp_model
    req, setup = data["required_s"], data["setup_s"]
    banned = set(data["region_satellites"].get(avoid_region, [])) if avoid_region else set()
    model = cp_model.CpModel()
    x, info = {}, {}
    for link in data["links"]:
        u, v = link["u"], link["v"]
        if u in banned or v in banned:
            continue
        need = (req if require_lifetime else 0.0) + (setup if link["kind"] == "N" else 0.0)
        if link["remaining_s"] is not None and link["remaining_s"] + 1e-9 < need:
            continue
        for a, b in ((u, v), (v, u)):
            x[(a, b)] = model.NewBoolVar("")
            info[(a, b)] = link
    out_arcs, in_arcs, inter = {}, {}, {}
    for (a, b), var in x.items():
        out_arcs.setdefault(a, []).append(var)
        in_arcs.setdefault(b, []).append(var)
        if info[(a, b)]["kind"] != "P":
            inter.setdefault(a, []).append(var)
            inter.setdefault(b, []).append(var)
    s, t = data["ingress"], data["egress"]
    for node in set(out_arcs) | set(in_arcs) | {s, t}:
        o, i = sum(out_arcs.get(node, [])), sum(in_arcs.get(node, []))
        rhs = (1 if node == s else 0) - (1 if node == t else 0)
        model.Add(o - i == rhs)
        model.Add(o <= 1)
        model.Add(i <= 1)
    for node, vars_ in inter.items():
        model.Add(sum(vars_) <= 1)
    if max_new_links is not None:
        budget = max_new_links - data["n_new_links_so_far"]
        model.Add(sum(v for k, v in x.items() if info[k]["kind"] == "N") <= max(0, budget))
    if max_delay_ms is not None:
        km_cap = max_delay_ms / 1000.0 * data["speed_of_light_km_s"] - data["ground_link_km"]
        model.Add(sum(int(round(info[k]["km"] * 10)) * v for k, v in x.items()) <= int(math.floor(km_cap * 10)))
    if objective == "km":
        model.Minimize(sum(int(round(info[k]["km"] * 10)) * v for k, v in x.items()))
    elif objective == "new_links":
        model.Minimize(1000 * sum(v for k, v in x.items() if info[k]["kind"] == "N") + sum(x.values()))
    else:
        model.Minimize(sum(x.values()))
    if hint_route:
        used = set(zip(hint_route, hint_route[1:]))
        for k, v in x.items():
            model.AddHint(v, int(k in used))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 8
    code = solver.Solve(model)
    status = solver.StatusName(code)
    if code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": status, "route": None}
    succ = {a: b for (a, b), v in x.items() if solver.Value(v)}
    route, node = [s], s
    while node != t and node in succ and len(route) <= len(succ) + 1:
        node = succ[node]
        route.append(node)
    ok = route[-1] == t
    km = sum(info[(a, b)]["km"] for a, b in zip(route, route[1:])) if ok else None
    return {"status": status, "route": route if ok else None, "hops": len(route) - 1,
            "new_links": sum(info[(a, b)]["kind"] == "N" for a, b in zip(route, route[1:])) if ok else None,
            "delay_ms": (data["ground_link_km"] + km) / data["speed_of_light_km_s"] * 1000 if ok else None}


def route_facts(data: Dict, route: List[int]) -> Optional[Dict]:
    """What a route does on this decision's data (None if it uses a link that is not in data["links"])."""
    link = {(l["u"], l["v"]): l for l in data["links"]}
    hops = [link.get((min(a, b), max(a, b))) for a, b in zip(route, route[1:])]
    if any(h is None for h in hops):
        return None
    need = lambda h: data["required_s"] + (data["setup_s"] if h["kind"] == "N" else 0.0)
    km = sum(h["km"] for h in hops)
    return {"hops": len(hops), "new_links": sum(h["kind"] == "N" for h in hops),
            "delay_ms": round((data["ground_link_km"] + km) / data["speed_of_light_km_s"] * 1000, 2),
            "links_ending_early": [[h["u"], h["v"]] for h in hops if h["remaining_s"] is not None and h["remaining_s"] + 1e-9 < need(h)],
            "relays_over": {r: sorted(set(route[1:-1]) & set(s)) for r, s in data["region_satellites"].items()
                            if set(route[1:-1]) & set(s)}}


def conditions_met(data: Dict, route: List[int], cond: Dict) -> Dict:
    """Per-condition check at one decision (the grader; never shown to the model)."""
    f = route_facts(data, route)
    if f is None:
        return {"valid_links": False}
    out = {"valid_links": True, "lifetime": not f["links_ending_early"]}
    if cond.get("avoid_region"):
        out["avoid"] = cond["avoid_region"] not in f["relays_over"]
    if cond.get("max_delay_ms") is not None:
        out["delay"] = f["delay_ms"] <= cond["max_delay_ms"] + 1e-9
    if cond.get("max_new_links") is not None:
        out["budget"] = data["n_new_links_so_far"] + f["new_links"] <= cond["max_new_links"]
    return out


class FormulationPlanner(RoutePlanner):
    """Calls ``solve(data) -> (route list or None, error)`` at every route decision of a session.

    ``solve`` is the author's reference model or an LLM-written model run in a
    sandbox.  An invalid or missing answer falls back to Dijkstra, as for the
    route-editing agents; every decision is logged with the grader's check.
    """

    def __init__(self, base: RoutePlanner, solve, regions: Optional[List[str]] = None):
        self.__dict__.update(base.__dict__)
        self.solve, self.regions = solve, regions
        self.decisions: List[Dict] = []
        self.request_id: Optional[int] = None
        self.condition: Dict = {}
        self.session_new_links = 0

    def plan(self, method, source_station, destination_station, time_s, required_s, topology):
        if method != "llm":
            return super().plan(method, source_station, destination_station, time_s, required_s, topology)
        dijkstra = super().plan("raw", source_station, destination_station, time_s, required_s, topology)
        if dijkstra is None:
            return None
        graph = self._graph(time_s, topology)
        data = decision_data(self, graph, source_station, destination_station, time_s, required_s,
                             self.regions, self.session_new_links)
        tick = time.perf_counter()
        proposal, error = self.solve(data)
        seconds = time.perf_counter() - tick
        route = None
        if proposal is not None and not error:
            route, error = to_route(self, graph, proposal, source_station, destination_station, time_s,
                                    {canonical_edge(*e) for e in topology.dynamic})
        elif proposal is None and not error:
            error = "solve returned None"
        chosen = route if route is not None else dijkstra
        self.session_new_links += len(chosen.new_dynamic_edges)
        self.decisions.append({
            "request_id": self.request_id, "time_s": time_s, "required_s": required_s, "valid": route is not None,
            "error": error or "", "proposal": proposal, "route": list(chosen.satellites), "dijkstra": list(dijkstra.satellites),
            "met": conditions_met(data, list(chosen.satellites), self.condition), "facts": route_facts(data, list(chosen.satellites)),
            "solve_seconds": round(seconds, 2)})
        return chosen
