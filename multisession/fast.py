"""Fast condition-aware heuristic (the agent's quick solver; the exact one is ``exact.solve_exact``).

Sequential admission with the kit search, adapted to the conditions:
session order follows the priorities and minimums, link costs follow the
objective order (hotspot visits, new links, hops), a route must pass the
required region, and new links stop at the limit.  Hard removals are already in
``prepared.instance``.
"""
from __future__ import annotations

import time
from typing import Dict, List

from .conditions import Prepared
from .kit import EdgeInfo, State, finish
from .model import Solution

WEIGHTS = (1.0e8, 1.0e4, 1.0)  # by rank among hotspot / new_links / hops in the objective order


def _order(prepared: Prepared) -> List:
    inst, cond = prepared.instance, prepared.conditions
    sessions = list(inst.sessions)
    first: List = []
    for dest, n in cond.min_admitted:  # guarantee the minimums before anything else
        first += [s for s in sessions if s.destination == dest][:n]
    for dest in cond.priority:
        first += [s for s in sessions if s.destination == dest and s not in first]
    return first + [s for s in sessions if s not in first]


def solve_fast(prepared: Prepared) -> Solution:
    start = time.perf_counter()
    inst, cond = prepared.instance, prepared.conditions
    rank = [o for o in cond.objective if o != "admitted"]
    w = {name: WEIGHTS[i] for i, name in enumerate(rank)}
    hot = prepared.hotspot
    reconfigure = cond.objective.index("new_links") > cond.objective.index("admitted")

    def cost(info: EdgeInfo) -> float:
        c = w["hops"]
        if info.kind == "new":
            c += w["new_links"]
        if info.v in hot:
            c += w["hotspot"]
        return c

    state = State(inst)
    routes: Dict[int, List[int]] = {}
    new_used = 0
    for s in _order(prepared):
        via = prepared.via_for(s.destination)
        need = s.duration_s
        found = state.search(inst.usable_access(s.source, need), inst.usable_access(s.destination, need),
                             need, False, cost, via)
        budget = None if cond.max_new_links is None else cond.max_new_links - new_used
        if found is None and reconfigure and (budget is None or budget > 0):
            need = inst.snapshot.setup_s + s.duration_s
            found = state.search(inst.usable_access(s.source, need), inst.usable_access(s.destination, need),
                                 need, True, cost, via)
            if found is not None and budget is not None and len(found[1]) > budget:
                found = None
        if found is None:
            continue
        state.admit(*found)
        new_used += len(found[1])
        routes[s.index] = found[0]
    return finish(state, routes, start)
