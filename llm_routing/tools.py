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

from .route_edit import hop_table, risky_links, to_route

SCHEMAS = [
    {"type": "function", "function": {
        "name": "check_route",
        "description": "Check a full route against the routing rules; list problems, each hop's link kind and "
                       "remaining contact, links that end before the session needs them, and avoided satellites on it.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["route", "avoid_satellites"],
                       "properties": {"route": {"type": "array", "items": {"type": "integer"}},
                                      "avoid_satellites": {"type": "array", "items": {"type": "integer"}}}}}},
    {"type": "function", "function": {
        "name": "find_route",
        "description": "Compute a rule-abiding route from the ingress to the egress satellite over the whole current "
                       "network, never using avoid_satellites. If min_remaining_s is given, inter-plane links must last at "
                       "least that long (new links 20 s longer). objective: fewest_hops or longest_contact "
                       "(maximize the shortest remaining contact on the route).",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": ["avoid_satellites", "min_remaining_s", "objective"],
                       "properties": {"avoid_satellites": {"type": "array", "items": {"type": "integer"}},
                                      "min_remaining_s": {"type": ["number", "null"]},
                                      "objective": {"type": "string", "enum": ["fewest_hops", "longest_contact"]}}}}},
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

    def check_route(self, route: List[int], avoid_satellites: List[int]) -> Dict:
        ok, error = to_route(self.p, self.g, route, self.source, self.destination, self.time_s, self.active)
        out = {"valid": ok is not None, "problem": error or None,
               "avoided_satellites_on_route": sorted(set(route) & set(avoid_satellites))}
        if ok is not None:
            out["hops"] = hop_table(self.g, route)
            out["links_ending_early"] = risky_links(self.p, self.g, route, self.required_s)
        return out

    def find_route(self, avoid_satellites: List[int], min_remaining_s: Optional[float], objective: str) -> Dict:
        banned = set(avoid_satellites)
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
                    cand = (key[0] + 1, key[1] + int(d and not act))
                    better = nxt not in best or cand < best[nxt]
                    push = (cand[0], cand[1], v, d)
                if better:
                    best[nxt] = cand
                    parent[nxt] = state
                    heapq.heappush(heap, push)
        if goal is None:
            return {"found": False, "problem": "no route satisfies these conditions"}
        route = [s[0] for s in self._path(parent, goal)]
        return {"found": True, "route": route, "hops": hop_table(self.g, route),
                "links_ending_early": risky_links(self.p, self.g, route, self.required_s)}

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
