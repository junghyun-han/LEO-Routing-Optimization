"""Exact CP-SAT model of docs/PROBLEM.md (section 5), optionally with operator conditions.

Sessions with the same (source, destination, X) are interchangeable, so the
model uses one integer flow per commodity and setup class instead of one path
per session:

* immediate class (no setup): persistent links and kept initial dynamic links
  with remaining contact >= X;
* delayed class (20 s setup): every ACTIVE link with remaining contact >= 20 + X.

Conditions (multisession.conditions): hard ones are applied to the instance
before this model is built; a required region is modelled by two copies of the
network, the flow entering the second copy only at satellites in the region
(the standard layered construction for "pass through a node set"); a hotspot
adds the objective "fewest satellite visits in the hotspot"; limits on new links
and per-destination minimums are linear constraints.

Objectives are solved lexicographically, each stage fixing the value of the
previous one.  Flows are decomposed into unit walks and returned as an ordinary
``Solution`` so the independent verifier can check them.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from .model import Edge, Instance, Solution, TOL


@dataclass
class ExactResult:
    solution: Solution
    status: List[str] = field(default_factory=list)  # per stage: OPTIMAL / FEASIBLE / INFEASIBLE / ...
    admitted_bound: float = 0.0  # proven upper bound on admitted sessions (total-admission stage)
    optimal: bool = False  # every stage proven optimal
    stage_seconds: List[float] = field(default_factory=list)
    hops_bound: float = 0.0  # proven lower bound on total hops
    stages: List[str] = field(default_factory=list)


def solve_exact(instance: Instance, time_limit_s: float = 60.0, workers: int = 8,
                hint: Optional[Solution] = None, hops_time_limit_s: Optional[float] = None,
                prepared=None) -> ExactResult:
    """``prepared`` is a ``conditions.Prepared``; its instance replaces ``instance``."""
    start = time.perf_counter()
    if prepared is not None:
        instance = prepared.instance
    cond = prepared.conditions if prepared is not None else None
    snap = instance.snapshot
    kappa = instance.capacity
    initial = set(snap.initial_dynamic)
    dynamic = list(snap.dynamic_remaining)

    groups: Dict[Tuple[str, str, float], List[int]] = defaultdict(list)
    for s in instance.sessions:
        groups[(s.source, s.destination, s.duration_s)].append(s.index)

    model = cp_model.CpModel()
    y = {e: model.NewBoolVar("") for e in dynamic}
    by_node: Dict[int, List[cp_model.IntVar]] = defaultdict(list)
    for e, var in y.items():
        by_node[e[0]].append(var)
        by_node[e[1]].append(var)
    for vars_ in by_node.values():  # (C1) one reconfigurable terminal
        model.AddAtMostOne(vars_)

    hotspot = prepared.hotspot if prepared is not None else set()
    edge_flow: Dict[Edge, List[cp_model.IntVar]] = defaultdict(list)
    classes = []  # (key, delayed, arcs{(u, v, layer)}, transfer{v}, src{s}, snk{z})
    admitted_by_dest: Dict[str, list] = defaultdict(list)
    hop_terms, hotspot_terms = [], []
    for key, members in groups.items():
        source, destination, duration = key
        via = prepared.via_for(destination) if prepared is not None else None
        layers = (0, 1) if via is not None else (1,)
        per_key = []
        for delayed in (False, True):
            need = duration + (snap.setup_s if delayed else 0.0)
            usable = [e for e in snap.persistent]
            usable += [e for e in dynamic
                       if (delayed or e in initial) and snap.remaining(e) + TOL >= need]  # (C2), (C3)
            arcs: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
            for e in usable:
                for layer in layers:
                    for u, v in (e, (e[1], e[0])):
                        var = model.NewIntVar(0, kappa, "")
                        arcs[(u, v, layer)] = var
                        edge_flow[e].append(var)
                        hop_terms.append(var)
                        if v in hotspot:
                            hotspot_terms.append(var)
            transfer = {v: model.NewIntVar(0, len(members), "") for v in (via or ())}
            src = {s: model.NewIntVar(0, len(members), "") for s in instance.usable_access(source, need)}
            snk = {z: model.NewIntVar(0, len(members), "") for z in instance.usable_access(destination, need)}
            hotspot_terms += [var for s, var in src.items() if s in hotspot]
            flow_in: Dict[Tuple[int, int], list] = defaultdict(list)
            flow_out: Dict[Tuple[int, int], list] = defaultdict(list)
            for (u, v, layer), var in arcs.items():
                flow_out[(u, layer)].append(var)
                flow_in[(v, layer)].append(var)
            for s, var in src.items():
                flow_in[(s, layers[0])].append(var)
            for z, var in snk.items():
                flow_out[(z, layers[-1])].append(var)
            for v, var in transfer.items():
                flow_out[(v, 0)].append(var)
                flow_in[(v, 1)].append(var)
            for node in set(flow_in) | set(flow_out):
                if flow_in[node] or flow_out[node]:
                    model.Add(sum(flow_in[node]) == sum(flow_out[node]))
            per_key.append(sum(src.values()) if src else 0)
            classes.append((key, delayed, arcs, transfer, src, snk))
        if any(not isinstance(t, int) for t in per_key):
            model.Add(sum(per_key) <= len(members))
        admitted_by_dest[destination].extend(per_key)

    for e, flows in edge_flow.items():  # (C4) capacity, only on ACTIVE links
        if snap.is_persistent(e):
            model.Add(sum(flows) <= kappa)
        else:
            model.Add(sum(flows) <= kappa * y[e])

    new_links = sum(y[e] for e in dynamic if e not in initial)
    if cond is not None and cond.max_new_links is not None:
        model.Add(new_links <= int(cond.max_new_links))
    if cond is not None:
        for dest, n in cond.min_admitted:
            model.Add(sum(admitted_by_dest.get(dest, [])) >= int(n))
    _add_hint(model, instance, y, classes, hint)

    admitted = sum(t for terms in admitted_by_dest.values() for t in terms)
    stages: List[Tuple[str, object, bool]] = []
    for dest in (cond.priority if cond is not None else ()):
        stages.append((f"admitted_{dest}", sum(admitted_by_dest.get(dest, [])), True))
    order = cond.objective if cond is not None else ("admitted", "hotspot", "new_links", "hops")
    exprs = {"admitted": (admitted, True), "hotspot": (sum(hotspot_terms), False),
             "new_links": (new_links, False), "hops": (sum(hop_terms), False)}
    for name in order:
        if name == "hotspot" and not hotspot_terms:
            continue
        stages.append((name, *exprs[name]))

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = workers
    statuses, seconds = [], []
    bound, hops_bound, last = float(len(instance.sessions)), 0.0, None
    for name, expr, maximize in stages:
        (model.Maximize if maximize else model.Minimize)(expr)
        limit = hops_time_limit_s if (name == "hops" and hops_time_limit_s is not None) else time_limit_s
        solver.parameters.max_time_in_seconds = limit
        tick = time.perf_counter()
        code = solver.Solve(model)
        seconds.append(time.perf_counter() - tick)
        statuses.append(solver.StatusName(code))
        if code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        if name == "admitted":
            bound = solver.BestObjectiveBound()
        if name == "hops":
            hops_bound = solver.BestObjectiveBound()
        value = int(round(solver.ObjectiveValue()))
        model.Add(expr >= value if maximize else expr <= value)
        model.ClearHints()
        for var in y.values():
            model.AddHint(var, solver.Value(var))
        last = {"y": {e: solver.Value(v) for e, v in y.items()}, "classes": []}
        for key, _, arcs, transfer, src, snk in classes:
            for var in list(arcs.values()) + list(transfer.values()) + list(src.values()) + list(snk.values()):
                model.AddHint(var, solver.Value(var))
            last["classes"].append((key, {a: solver.Value(v) for a, v in arcs.items() if solver.Value(v)},
                                    {v: solver.Value(t) for v, t in transfer.items() if solver.Value(t)},
                                    {s: solver.Value(v) for s, v in src.items() if solver.Value(v)},
                                    {z: solver.Value(v) for z, v in snk.items() if solver.Value(v)},
                                    bool(transfer) or prepared is not None and prepared.via_for(key[1]) is not None))

    names = [s[0] for s in stages]
    if last is None:
        return ExactResult(Solution({}, set(initial), time.perf_counter() - start), statuses, bound, False,
                           seconds, hops_bound, names)
    routes: Dict[int, List[int]] = {}
    queues = {key: list(members) for key, members in groups.items()}
    for key, arc_flow, transfer, src_flow, snk_flow, layered in last["classes"]:
        for walk in _decompose(arc_flow, transfer, src_flow, snk_flow, layered):
            routes[queues[key].pop(0)] = walk
    final = {e for e, v in last["y"].items() if v}
    optimal = len(statuses) == len(stages) and all(s == "OPTIMAL" for s in statuses)
    return ExactResult(Solution(routes, final, time.perf_counter() - start), statuses, bound, optimal,
                       seconds, hops_bound, names)


def _add_hint(model, instance: Instance, y, classes, hint: Optional[Solution]) -> None:
    """Start the search from a known feasible solution (e.g. a baseline); single-layer classes only."""
    initial = set(instance.snapshot.initial_dynamic)
    if hint is None:
        for e in initial:
            if e in y:
                model.AddHint(y[e], 1)
        return
    final = set(hint.final_dynamic)
    for e, var in y.items():
        model.AddHint(var, int(e in final))
    new = final - initial
    sessions = {s.index: s for s in instance.sessions}
    counts = defaultdict(int)
    for k, path in hint.routes.items():
        s = sessions[k]
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        delayed = any((min(u, v), max(u, v)) in new for u, v in edges)
        cls = ((s.source, s.destination, s.duration_s), delayed)
        for u, v in edges:
            counts[(cls, "arc", (u, v, 1))] += 1
        counts[(cls, "src", path[0])] += 1
        counts[(cls, "snk", path[-1])] += 1
    for key, delayed, arcs, transfer, src, snk in classes:
        if transfer:
            continue
        cls = (key, delayed)
        for a, var in arcs.items():
            model.AddHint(var, counts.get((cls, "arc", a), 0))
        for s_, var in src.items():
            model.AddHint(var, counts.get((cls, "src", s_), 0))
        for z, var in snk.items():
            model.AddHint(var, counts.get((cls, "snk", z), 0))


def _decompose(arc_flow, transfer, src_flow, snk_flow, layered) -> List[List[int]]:
    """Split an integer flow on the (layered) network into unit walks, dropping cycles."""
    first, last_layer = (0, 1) if layered else (1, 1)
    flow = dict(arc_flow)
    for v, t in transfer.items():
        flow[(v, v, "T")] = t  # transfer from layer 0 to layer 1 at v
    out = defaultdict(list)
    for (u, v, layer) in arc_flow:
        out[(u, layer)].append(((u, v, layer), (v, layer)))
    for v in transfer:
        out[(v, 0)].append(((v, v, "T"), (v, 1)))
    src, snk = dict(src_flow), dict(snk_flow)
    walks = []
    while src:
        s = next(iter(src))
        states = [(s, first)]
        position = {(s, first): 0}
        while True:
            node = states[-1]
            if node[1] == last_layer and snk.get(node[0], 0) > 0:
                break
            arc, nxt = next((a, n) for a, n in out[node] if flow.get(a, 0) > 0)
            flow[arc] -= 1
            if nxt in position:  # cycle: drop it
                cut = position[nxt]
                for st in states[cut + 1:]:
                    position.pop(st, None)
                states = states[: cut + 1]
                continue
            position[nxt] = len(states)
            states.append(nxt)
        src[s] -= 1
        if src[s] == 0:
            del src[s]
        z = states[-1][0]
        snk[z] -= 1
        if snk[z] == 0:
            del snk[z]
        walk = [states[0][0]]
        for v, _ in states[1:]:
            if v != walk[-1]:  # a layer transfer stays at the same satellite
                walk.append(v)
        walks.append(walk)
    return walks
