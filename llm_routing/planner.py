"""Route planner whose "llm" method lets a language model edit the Dijkstra route.

``SessionEnvironment.run(method, request)`` asks its planner for a route at the
request time and at every proactive reroute.  Replacing the planner with
``LLMPlanner`` and calling ``run("llm", request)`` therefore uses exactly the
WCNC event loop, timing, and metrics; only the route choice changes.  An
invalid edit falls back to the Dijkstra route and is logged.
"""
from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional

from leo_session.routing import Route, RoutePlanner
from leo_session.topology import canonical_edge

from .route_edit import LEGEND, describe, neighbourhood, risky_links, to_route

SYSTEM = ("You route sessions in a LEO satellite network with optical inter-satellite links. A session must stay "
          "connected for its whole holding time; links whose contact ends earlier will break it. Edit the given "
          "Dijkstra route into the route you recommend. Answer with JSON only: "
          '{"route": [satellite ids from ingress to egress], "reason": "one sentence"}')


class LLMPlanner(RoutePlanner):
    def __init__(self, base: RoutePlanner, model, latlon_of, operator_text: str = "", hops: int = 1):
        self.__dict__.update(base.__dict__)
        self.model, self.latlon_of, self.operator_text, self.hops = model, latlon_of, operator_text, hops
        self.decisions: List[Dict] = []
        self.request_id: Optional[int] = None

    def plan(self, method, source_station, destination_station, time_s, required_s, topology) -> Optional[Route]:
        if method != "llm":
            return super().plan(method, source_station, destination_station, time_s, required_s, topology)
        dijkstra = super().plan("raw", source_station, destination_station, time_s, required_s, topology)
        if dijkstra is None:
            return None
        graph = self._graph(time_s, topology)
        nodes = neighbourhood(graph, dijkstra.satellites, self.hops)
        positions = self.constellation.positions(time_s)
        prompt = (
            (f"Operator conditions:\n{self.operator_text}\n\n" if self.operator_text else "")
            + f"Session {source_station} -> {destination_station}. Decision time {time_s:.0f} s. The route must stay "
            f"connected for the next {required_s:.0f} s (add 20 s if it switches any link on).\n"
            f"Ingress satellite {dijkstra.ingress}, egress satellite {dijkstra.egress}.\n"
            f"Dijkstra route: {list(dijkstra.satellites)}\n\n{LEGEND}\n{describe(graph, nodes, positions, self.latlon_of(time_s))}"
        )
        tick = time.perf_counter()
        error, usage, text, route = "", {}, "", None
        try:
            msg, usage = self.model.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            text = msg["content"]
            match = re.search(r"\{.*\}", text, re.S)
            proposal = json.loads(match.group(0))["route"] if match else None
            if proposal is None:
                error = "no JSON route in the answer"
            else:
                active = {canonical_edge(*e) for e in topology.dynamic}
                route, error = to_route(self, graph, proposal, source_station, destination_station, time_s, active)
        except Exception as exc:  # model or parsing failure: keep the Dijkstra route
            error = f"{type(exc).__name__}: {exc}"
        chosen = route if route is not None else dijkstra
        self.decisions.append({
            "request_id": self.request_id, "time_s": time_s, "required_s": required_s, "valid": route is not None,
            "error": error, "changed": chosen.satellites != dijkstra.satellites, "dijkstra": list(dijkstra.satellites),
            "route": list(chosen.satellites), "warnings": risky_links(self, graph, chosen.satellites, required_s),
            "prompt_chars": len(prompt), "seconds": time.perf_counter() - tick,
            **{k: usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens")},
        })
        return chosen
