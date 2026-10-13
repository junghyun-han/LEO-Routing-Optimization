"""Exact CP-SAT model of docs/PROBLEM.md (section 5).

Sessions with the same (source, destination, X) are interchangeable, so the
model uses one integer flow per commodity and setup class instead of one path
per session:

* immediate class (no setup): persistent links and kept initial dynamic links
  with remaining contact >= X;
* delayed class (20 s setup): every ACTIVE link with remaining contact >= 20 + X.

The three objectives are solved lexicographically (admitted sessions, then new
links, then ISL hops), each stage fixing the optimum of the previous one.  The
flows are decomposed into unit paths and returned as an ordinary ``Solution``
so the independent verifier can check them.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from .model import Edge, Instance, Solution, TOL

SRC, SNK = -1, -2


@dataclass
class ExactResult:
    solution: Solution
    status: List[str] = field(default_factory=list)  # per stage: OPTIMAL / FEASIBLE / ...
    admitted_bound: float = 0.0  # proven upper bound on admitted sessions
    optimal: bool = False  # all three stages proven optimal
    stage_seconds: List[float] = field(default_factory=list)
    hops_bound: float = 0.0  # proven lower bound on total hops (stage 3)


def _status_name(code: int) -> str:
    return cp_model.CpSolverStatus.Name(code) if hasattr(cp_model.CpSolverStatus, "Name") else str(code)


def solve_exact(instance: Instance, time_limit_s: float = 60.0, workers: int = 8,
                hint: Optional[Solution] = None, hops_time_limit_s: Optional[float] = None) -> ExactResult:
    start = time.perf_counter()
    snap = instance.snapshot
    kappa = instance.capacity
    initial = set(snap.initial_dynamic)
    dynamic = list(snap.dynamic_remaining)

    groups: Dict[Tuple[str, str, float], List[int]] = defaultdict(list)
    for s in instance.sessions:
        groups[(s.source, s.destination, s.duration_s)].append(s.index)

    model = cp_model.CpModel()
    y = {e: model.NewBoolVar(f"y{e}") for e in dynamic}
    by_node: Dict[int, List[cp_model.IntVar]] = defaultdict(list)
    for e, var in y.items():
        by_node[e[0]].append(var)
        by_node[e[1]].append(var)
    for vars_ in by_node.values():  # (C1) one reconfigurable terminal
        model.AddAtMostOne(vars_)

    edge_flow: Dict[Edge, List[cp_model.IntVar]] = defaultdict(list)
    classes = []  # (key, delayed, arc vars, source vars, sink vars)
    admitted_terms = []
    hop_terms = []
    for key, members in groups.items():
        source, destination, duration = key
        per_key = []
        for delayed in (False, True):
            need = duration + (snap.setup_s if delayed else 0.0)
            arcs: Dict[Tuple[int, int], cp_model.IntVar] = {}
            for e in snap.persistent:
                for a in (e, (e[1], e[0])):
                    arcs[a] = model.NewIntVar(0, kappa, "")
            for e in dynamic:
                if not delayed and e not in initial:
                    continue  # links switched on now are usable only after the setup
                if snap.remaining(e) + TOL < need:
                    continue  # (C3) lifetime
                for a in (e, (e[1], e[0])):
                    arcs[a] = model.NewIntVar(0, kappa, "")
            for (u, v), var in arcs.items():
                edge_flow[(min(u, v), max(u, v))].append(var)
                hop_terms.append(var)
            src = {s: model.NewIntVar(0, len(members), "") for s in instance.usable_access(source, need)}
            snk = {z: model.NewIntVar(0, len(members), "") for z in instance.usable_access(destination, need)}
            out_of: Dict[int, list] = defaultdict(list)
            into: Dict[int, list] = defaultdict(list)
            for (u, v), var in arcs.items():
                out_of[u].append(var)
                into[v].append(var)
            for v in set(out_of) | set(into) | set(src) | set(snk):
                inflow = sum(into[v]) + (src[v] if v in src else 0)
                outflow = sum(out_of[v]) + (snk[v] if v in snk else 0)
                model.Add(inflow == outflow)
            per_key.append(sum(src.values()) if src else 0)
            classes.append((key, delayed, arcs, src, snk))
        if any(not isinstance(t, int) for t in per_key):
            model.Add(sum(per_key) <= len(members))
        admitted_terms.extend(per_key)

    for e, flows in edge_flow.items():  # (C4) capacity, only on ACTIVE links
        if snap.is_persistent(e):
            model.Add(sum(flows) <= kappa)
        else:
            model.Add(sum(flows) <= kappa * y[e])

    _add_hint(model, instance, y, classes, hint)
    admitted = sum(admitted_terms)
    new_links = sum(y[e] for e in dynamic if e not in initial)
    hops = sum(hop_terms)

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = workers
    statuses: List[str] = []
    bound = float(len(instance.sessions))
    seconds: List[float] = []
    hops_bound = 0.0
    for stage, (expr, maximize) in enumerate(((admitted, True), (new_links, False), (hops, False))):
        if maximize:
            model.Maximize(expr)
        else:
            model.Minimize(expr)
        limit = hops_time_limit_s if (stage == 2 and hops_time_limit_s is not None) else time_limit_s
        solver.parameters.max_time_in_seconds = limit
        tick = time.perf_counter()
        code = solver.Solve(model)
        seconds.append(time.perf_counter() - tick)
        statuses.append(solver.StatusName(code))
        if code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        if stage == 0:
            bound = solver.BestObjectiveBound()
        if stage == 2:
            hops_bound = solver.BestObjectiveBound()
        value = int(round(solver.ObjectiveValue()))
        # keep the stage value (fix it if proven, otherwise keep it as a floor/ceiling)
        if maximize:
            model.Add(expr >= value)
        else:
            model.Add(expr <= value)
        model.ClearHints()
        for var in list(y.values()):
            model.AddHint(var, solver.Value(var))
        for _, _, arcs, src, snk in classes:
            for var in list(arcs.values()) + list(src.values()) + list(snk.values()):
                model.AddHint(var, solver.Value(var))
        last = {
            "y": {e: solver.Value(v) for e, v in y.items()},
            "classes": [(key, {a: solver.Value(v) for a, v in arcs.items() if solver.Value(v)},
                         {s: solver.Value(v) for s, v in src.items() if solver.Value(v)},
                         {z: solver.Value(v) for z, v in snk.items() if solver.Value(v)})
                        for key, _, arcs, src, snk in classes],
        }

    if not statuses or statuses[0] not in ("OPTIMAL", "FEASIBLE"):
        return ExactResult(Solution({}, set(initial), time.perf_counter() - start), statuses, bound, False, seconds)

    routes: Dict[int, List[int]] = {}
    queues = {key: list(members) for key, members in groups.items()}
    for key, arc_flow, src_flow, snk_flow in last["classes"]:
        for path in _decompose(arc_flow, src_flow, snk_flow):
            routes[queues[key].pop(0)] = path
    final = {e for e, v in last["y"].items() if v}
    solution = Solution(routes, final, time.perf_counter() - start)
    return ExactResult(solution, statuses, bound, all(s == "OPTIMAL" for s in statuses) and len(statuses) == 3,
                       seconds, hops_bound)


def _add_hint(model, instance: Instance, y, classes, hint: Optional[Solution]) -> None:
    """Start the search from a known feasible solution (e.g. a baseline)."""
    initial = set(instance.snapshot.initial_dynamic)
    if hint is None:
        for e in initial:
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
        for a in edges:
            counts[(cls, "arc", a)] += 1
        counts[(cls, "src", path[0])] += 1
        counts[(cls, "snk", path[-1])] += 1
    for key, delayed, arcs, src, snk in classes:
        cls = (key, delayed)
        for a, var in arcs.items():
            model.AddHint(var, counts.get((cls, "arc", a), 0))
        for s_, var in src.items():
            model.AddHint(var, counts.get((cls, "src", s_), 0))
        for z, var in snk.items():
            model.AddHint(var, counts.get((cls, "snk", z), 0))


def _decompose(arc_flow, src_flow, snk_flow) -> List[List[int]]:
    """Split an integer flow into unit paths, dropping any cycles."""
    flow = dict(arc_flow)
    src = dict(src_flow)
    snk = dict(snk_flow)
    out = defaultdict(list)
    for (u, v), f in flow.items():
        out[u].append(v)
    paths = []
    while src:
        s = next(iter(src))
        path = [s]
        position = {s: 0}
        while True:
            u = path[-1]
            if snk.get(u, 0) > 0:
                break
            nxt = next(v for v in out[u] if flow.get((u, v), 0) > 0)
            flow[(u, nxt)] -= 1
            if nxt in position:  # cycle: drop it and continue from nxt
                cut = position[nxt]
                for node in path[cut + 1:]:
                    position.pop(node, None)
                path = path[: cut + 1]
                continue
            position[nxt] = len(path)
            path.append(nxt)
        src[s] -= 1
        if src[s] == 0:
            del src[s]
        snk[path[-1]] -= 1
        if snk[path[-1]] == 0:
            del snk[path[-1]]
        paths.append(path)
    return paths
