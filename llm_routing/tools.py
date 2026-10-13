"""Python tools for the route-editing agent (condition 3 of docs/LLM_ROUTING.md).

The tools only see network data; they never read the operator text.  Turning
the text into tool arguments (which satellites to avoid, how long links must
last) is the model's job.  All routes follow the v3 rules used by ``to_route``.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence

from leo_session.topology import canonical_edge
from multisession.conditions import REGIONS

from .compliance import in_region
from multisession.conditions import satellite_latlon

from .route_edit import hop_table, risky_links, to_route

SCHEMAS = [
    {"type": "function", "function": {
        "name": "check_route",
        "description": "Check a full route against the routing rules; list problems, each hop's link kind and "
                       "remaining contact, links that end before the session needs them, and avoided satellites on it.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": ["route"],
                       "properties": {"route": {"type": "array", "items": {"type": "integer"}},
                                      "avoid_satellites": {"type": "array", "items": {"type": "integer"}},
                                      "regions": {"type": "array", "items": {"type": "string", "enum": sorted(REGIONS)},
                                                  "description": "report route satellites over these regions during the session"}}}}},
    {"type": "function", "function": {
        "name": "find_route",
        "description": "Optional arguments: omit any the operator gave no condition for. "
                       "Compute a rule-abiding route from the ingress to the egress satellite over the whole current "
                       "network, never using avoid_satellites or satellites over avoid_regions during the session. If "
                       "min_remaining_s is given, inter-plane links must last at least that long (new links 20 s longer). "
                       "objective: fewest_hops, longest_contact (maximize the shortest remaining contact), fewest_new_links "
                       "(fewest links to switch on), or shortest_distance (lowest propagation delay).",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": ["objective"],
                       "properties": {"avoid_satellites": {"type": "array", "items": {"type": "integer"}},
                                      "avoid_regions": {"type": "array", "items": {"type": "string", "enum": sorted(REGIONS)},
                                                        "description": "never use satellites that are over these regions at any time during the session"},
                                      "prefer_avoid_regions": {"type": "array", "items": {"type": "string", "enum": sorted(REGIONS)},
                                                               "description": "pass over these regions as little as possible (fewest relay satellites there first, then the objective); not with longest_contact"},
                                      "min_remaining_s": {"type": ["number", "null"]},
                                      "objective": {"type": "string", "enum": ["fewest_hops", "longest_contact",
                                                                               "fewest_new_links", "shortest_distance"]}}}}},
    {"type": "function", "function": {
        "name": "submit_route",
        "description": "Submit the final route (ingress ... egress). Rejected with the reason if it breaks a rule.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["route", "reason"],
                       "properties": {"route": {"type": "array", "items": {"type": "integer"}},
                                      "reason": {"type": "string"}}}}},
]


class Toolbox:
    def __init__(self, planner, graph, source: str, destination: str, time_s: float, required_s: float, active_dynamic):
        self.p, self.g = planner, graph
        self.source, self.destination, self.time_s, self.required_s = source, destination, time_s, required_s
        self.active = active_dynamic
        self.ingress = planner.ground_trace.selected_access(source, time_s)
        self.egress = planner.ground_trace.selected_access(destination, time_s)
        self.submitted = None

    def check_route(self, route: List[int], avoid_satellites: List[int] = (), regions: List[str] = ()) -> Dict:
        avoid_satellites, regions = avoid_satellites or [], regions or []
        ok, error = to_route(self.p, self.g, route, self.source, self.destination, self.time_s, self.active)
        out = {"valid": ok is not None, "problem": error or None,
               "avoided_satellites_on_route": sorted(set(route) & set(avoid_satellites))}
        if ok is not None:
            out["hops"] = hop_table(self.g, route)
            out["links_ending_early"] = risky_links(self.p, self.g, route, self.required_s)
            out["new_links"] = len(ok.new_dynamic_edges)
            out["delay_ms_now"] = round(self._delay_ms(route), 1)
            for r in regions:
                out[f"satellites_over_{r}_during_session"] = sorted(set(route) & self._region_sats(r))
        return out

    def _window(self):
        end = self.time_s + self.required_s + self.p.config.dynamic_setup_s
        return [self.time_s + k * 10.0 for k in range(int((end - self.time_s) // 10.0) + 1)]

    def _region_sats(self, region: str) -> set:
        if region not in REGIONS:
            raise TypeError(f"unknown region {region}")
        cache = self.__dict__.setdefault("_region_cache", {})
        if region not in cache:
            sats = set()
            for t in self._window():
                ll = satellite_latlon(self.p.constellation, t)
                sats |= set(in_region(ll, range(len(ll)), region))
            cache[region] = sats
        return cache[region]

    def _delay_ms(self, route: List[int]) -> float:
        import numpy as np
        from leo_session.ground import GROUND_STATIONS
        pos = self.p.constellation.positions(self.time_s)
        radius = self.p.constellation.config.earth_radius_km
        km = float(np.linalg.norm(pos[route[0]] - GROUND_STATIONS[self.source].eci(self.time_s, radius)))
        km += float(np.linalg.norm(pos[route[-1]] - GROUND_STATIONS[self.destination].eci(self.time_s, radius)))
        km += sum(float(np.linalg.norm(pos[a] - pos[b])) for a, b in zip(route, route[1:]))
        return km / 299792.458 * 1000.0

    def find_route(self, objective: str = "fewest_hops", avoid_satellites: List[int] = (), min_remaining_s: Optional[float] = None,
                   avoid_regions: List[str] = (), prefer_avoid_regions: List[str] = ()) -> Dict:
        avoid_satellites, avoid_regions, prefer_avoid_regions = avoid_satellites or [], avoid_regions or [], prefer_avoid_regions or []
        banned = set(avoid_satellites)
        disliked = set()
        for r in prefer_avoid_regions:
            disliked |= self._region_sats(r)
        disliked -= {self.ingress, self.egress}
        for r in avoid_regions:  # the ingress and egress are fixed for every method, so regions never ban them
            banned |= self._region_sats(r) - {self.ingress, self.egress}
        positions = self.p.constellation.positions(self.time_s)
        if self.ingress in banned or self.egress in banned:
            return {"found": False, "problem": "the ingress or egress satellite is in avoid_satellites"}
        setup = self.p.config.dynamic_setup_s

        def usable(remaining, dynamic, active):
            if min_remaining_s is None or not dynamic:
                return True
            return remaining + 1.0e-9 >= min_remaining_s + (0.0 if active else setup)

        # state = (satellite, entered through an inter-plane link); v3 terminal rule
        start = (self.ingress, False)
        if objective == "longest_contact":
            best = {start: (math.inf, 0)}  # (bottleneck, -hops) maximized
            heap = [(-math.inf, 0, self.ingress, False)]
        else:
            best = {start: (0, 0)}
            heap = [(0, 0, self.ingress, False)]
        parent = {}
        goal = None
        while heap:
            a, b, u, dyn = heapq.heappop(heap)
            state = (u, dyn)
            key = (-a, -b) if objective == "longest_contact" else (a, b)
            if best.get(state) != key:
                continue
            if u == self.egress:
                goal = state
                break
            for v, d, act, rem in self.g.get(u, ()):
                if v in banned or (dyn and d) or not usable(rem, d, act):
                    continue
                if any(s[0] == v for s in self._path(parent, state)):
                    continue
                nxt = (v, d)
                if objective == "longest_contact":
                    cand = (min(key[0], rem if d else math.inf), key[1] - 1)
                    better = nxt not in best or cand > best[nxt]
                    push = (-cand[0], -cand[1], v, d)
                else:
                    new = int(d and not act)
                    if objective == "fewest_new_links":
                        step = (new, 1)
                    elif objective == "shortest_distance":
                        import numpy as np
                        step = (float(np.linalg.norm(positions[u] - positions[v])), new)
                    else:
                        step = (1, new)
                    if v in disliked:  # lexicographic: fewer relays over preferred-avoid regions first
                        step = (step[0] + 1.0e9, step[1])
                    cand = (key[0] + step[0], key[1] + step[1])
                    better = nxt not in best or cand < best[nxt]
                    push = (cand[0], cand[1], v, d)
                if better:
                    best[nxt] = cand
                    parent[nxt] = state
                    heapq.heappush(heap, push)
        if goal is None:
            hint = (" No route avoids these regions for the whole session; prefer_avoid_regions passes over them as "
                    "little as possible." if avoid_regions else "")
            return {"found": False, "problem": "no route satisfies these conditions." + hint}
        route = [s[0] for s in self._path(parent, goal)]
        ok, _ = to_route(self.p, self.g, route, self.source, self.destination, self.time_s, self.active)
        return {"found": True, "route": route, "hops": hop_table(self.g, route),
                "links_ending_early": risky_links(self.p, self.g, route, self.required_s),
                "new_links": len(ok.new_dynamic_edges) if ok else None, "delay_ms_now": round(self._delay_ms(route), 1)}

    @staticmethod
    def _path(parent, state) -> List:
        path = [state]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        return path[::-1]

    def submit_route(self, route: List[int], reason: str) -> Dict:
        ok, error = to_route(self.p, self.g, route, self.source, self.destination, self.time_s, self.active)
        if ok is None:
            return {"accepted": False, "problem": error}
        self.submitted = ok
        return {"accepted": True}

    def call(self, name: str, args: Dict) -> Dict:
        if name not in ("check_route", "find_route", "submit_route"):
            return {"error": f"unknown tool {name}"}
        try:
            return getattr(self, name)(**args)
        except TypeError as exc:
            return {"error": f"bad arguments: {exc}"}
