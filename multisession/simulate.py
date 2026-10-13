"""Replay a solution in the v3 simulator and measure what an operator would look at.

The true world state of a scenario (its ground-truth conditions) is applied:
failed satellites are down for the whole session, and a ground station only has
the antennas it really has.  Each admitted session is replayed second by second
from its start (t0, or t0 + 20 s if it uses a link switched on now) for X
seconds: it completes only if every ISL is in contact, no satellite on it has
failed, and both ground links are up throughout.  Path length and one-way
propagation delay are averaged over the session.  Policy conditions (regions,
hotspots, limits, minimums) are reported as compliance numbers, not as outages.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from leo_session.ground import GROUND_STATIONS

from .conditions import Prepared
from .model import Solution, canonical_edge, route_edges

C_KM_S = 299792.458


def replay(world: Prepared, solution: Solution, constellation, isl_trace, ground_trace,
           step_s: float = 1.0, latency_step_s: float = 10.0) -> Dict:
    inst = world.instance
    snap = inst.snapshot
    t0 = snap.time_s
    cond = world.conditions
    failed = set(cond.failed_satellites)
    initial = set(snap.initial_dynamic)
    final = {canonical_edge(*e) for e in solution.final_dynamic}
    new_links = final - initial
    sessions = {s.index: s for s in inst.sessions}
    radius = constellation.config.earth_radius_km
    cache: Dict[float, np.ndarray] = {}

    def positions(t):
        key = round(t, 3)
        if key not in cache:
            cache[key] = constellation.positions(t)
        return cache[key]

    per_session = []
    for k, path in solution.routes.items():
        s = sessions[k]
        edges = route_edges(path)
        delayed = any(e in new_links for e in edges)
        start = t0 + (snap.setup_s if delayed else 0.0)
        times = np.arange(start, start + s.duration_s + 1e-9, step_s)
        ingress_ok = path[0] in dict(inst.antenna_sats(s.source))
        egress_ok = path[-1] in dict(inst.antenna_sats(s.destination))
        down = 0
        for t in times:
            up = ingress_ok and egress_ok and not (failed & set(path))
            up = up and ground_trace.contact_window(s.source, path[0], t) is not None
            up = up and ground_trace.contact_window(s.destination, path[-1], t) is not None
            if up:
                for u, v in edges:
                    if not snap.is_persistent((u, v)) and not isl_trace.is_feasible(u, v, t):
                        up = False
                        break
            down += 0 if up else 1
        lat_times = np.arange(start, start + s.duration_s + 1e-9, latency_step_s)
        km = []
        for t in lat_times:
            pos = positions(t)
            d = float(np.linalg.norm(pos[path[0]] - GROUND_STATIONS[s.source].eci(t, radius)))
            d += float(np.linalg.norm(pos[path[-1]] - GROUND_STATIONS[s.destination].eci(t, radius)))
            d += sum(float(np.linalg.norm(pos[u] - pos[v])) for u, v in edges)
            km.append(d)
        per_session.append({
            "session": k, "destination": s.destination, "completed": down == 0,
            "interrupted_s": down * step_s, "hops": len(edges), "path_km": float(np.mean(km)),
            "delay_ms": float(np.mean(km)) / C_KM_S * 1000.0, "delayed_start": delayed,
            "failed_satellite": bool(failed & set(path)), "avoided_region": bool(world.avoid & set(path)),
            "hotspot_visits": sum(1 for v in path if v in world.hotspot),
            "via_ok": (world.via_for(s.destination) is None) or bool(world.via_for(s.destination) & set(path)),
        })
    return summarize(world, solution, per_session, len(new_links))


def summarize(world: Prepared, solution: Solution, rows: List[Dict], new_links: int) -> Dict:
    inst, cond = world.instance, world.conditions
    done = [r for r in rows if r["completed"]]
    by_dest = {}
    for s in inst.sessions:
        by_dest.setdefault(s.destination, 0)
    for r in done:
        by_dest[r["destination"]] += 1
    min_ok = all(by_dest.get(d, 0) >= n for d, n in cond.min_admitted)
    return {
        "requested": len(inst.sessions),
        "admitted": len(rows),
        "completed": len(done),
        "completion_rate": len(done) / max(1, len(inst.sessions)),
        "interrupted_s": sum(r["interrupted_s"] for r in rows),
        "mean_delay_ms": float(np.mean([r["delay_ms"] for r in done])) if done else float("nan"),
        "mean_path_km": float(np.mean([r["path_km"] for r in done])) if done else float("nan"),
        "mean_hops": float(np.mean([r["hops"] for r in done])) if done else float("nan"),
        "new_links": new_links,
        "new_link_limit_ok": cond.max_new_links is None or new_links <= cond.max_new_links,
        "sessions_on_failed_satellite": sum(r["failed_satellite"] for r in rows),
        "sessions_in_avoided_region": sum(r["avoided_region"] for r in rows),
        "via_violations": sum(not r["via_ok"] for r in rows),
        "hotspot_visits": sum(r["hotspot_visits"] for r in rows),
        "sessions_through_hotspot": sum(r["hotspot_visits"] > 0 for r in rows),
        "min_guarantee_ok": min_ok,
        "completed_by_destination": by_dest,
        "sessions": rows,
    }
