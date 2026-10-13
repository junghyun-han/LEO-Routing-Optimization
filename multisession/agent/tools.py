"""Tools (skills) the routing agent can call, with JSON schemas and validated handlers.

Every condition tool checks its arguments against the live network (satellite
ids, region and station names, antenna counts) and answers with a short error
and the valid options when something is wrong.  ``solve`` runs the fast or the
exact solver on the agent's current conditions and replays the result in the
simulator under those conditions; ``diagnose_conflicts`` relaxes one condition
at a time to show which one limits admissions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..conditions import OBJECTIVES, REGIONS, Conditions, prepare, region_satellites
from ..exact import solve_exact
from ..fast import solve_fast
from ..model import Instance, Solution
from ..simulate import replay

STATIONS = ("new_york", "london", "istanbul", "hanoi")
DESTINATIONS = ("london", "istanbul", "hanoi")


def empty_spec() -> Dict:
    return {"failed_satellites": [], "avoid_regions": [], "hotspot_regions": [], "via": {}, "antennas": {},
            "max_new_links": None, "min_admitted": {}, "priority": [], "objective": list(OBJECTIVES)}


@dataclass
class Workspace:
    instance: Instance
    latlon: object
    env: tuple  # (constellation, isl_trace, ground_trace)
    spec: Dict = field(default_factory=empty_spec)
    last_solution: Optional[Solution] = None
    finished: bool = False
    explanation: str = ""

    def conditions(self) -> Conditions:
        return Conditions.from_dict(self.spec)


def network_state(ws: Workspace) -> Dict:
    inst = ws.instance
    sessions: Dict[str, int] = {}
    for s in inst.sessions:
        sessions[s.destination] = sessions.get(s.destination, 0) + 1
    return {
        "decision_time_s": round(inst.snapshot.time_s, 1),
        "sessions_from_new_york": sessions,
        "session_duration_s": inst.sessions[0].duration_s,
        "link_capacity_sessions": inst.capacity,
        "antenna_satellites": {st: [s for s, _ in inst.antenna_sats(st)] for st in STATIONS},
        "regions_with_satellite_counts": {r: len(region_satellites(ws.latlon, r)) for r in REGIONS},
        "satellite_ids": "0-1583",
        "link_setup_s": inst.snapshot.setup_s,
        "objectives": list(OBJECTIVES),
    }


def _err(msg: str, **valid) -> Dict:
    return {"ok": False, "error": msg, **({"valid": valid} if valid else {})}


def _ok(ws: Workspace, **extra) -> Dict:
    return {"ok": True, "current_conditions": ws.spec, **extra}


# ------------------------------------------------------------------ handlers
def get_network_state(ws: Workspace) -> Dict:
    return {"ok": True, **network_state(ws)}


def lookup_satellite(ws: Workspace, satellite_id: int) -> Dict:
    if not isinstance(satellite_id, int) or not 0 <= satellite_id < len(ws.latlon):
        return _err(f"satellite {satellite_id!r} does not exist", satellite_ids="0-1583")
    lat, lon = ws.latlon[satellite_id]
    regions = [r for r in REGIONS if satellite_id in region_satellites(ws.latlon, r)]
    antenna_of = [st for st in STATIONS if satellite_id in dict(ws.instance.antenna_sats(st))]
    return {"ok": True, "satellite": satellite_id, "lat": round(float(lat), 1), "lon": round(float(lon), 1),
            "regions": regions, "antenna_satellite_of": antenna_of}


def set_failed_satellites(ws: Workspace, satellite_ids: List[int]) -> Dict:
    bad = [s for s in satellite_ids if not isinstance(s, int) or not 0 <= s < len(ws.latlon)]
    if bad:
        return _err(f"unknown satellite ids {bad}", satellite_ids="0-1583")
    ws.spec["failed_satellites"] = sorted(set(satellite_ids))
    return _ok(ws)


def set_region_policy(ws: Workspace, region: str, policy: str) -> Dict:
    if region not in REGIONS:
        return _err(f"unknown region {region!r}", regions=list(REGIONS))
    if policy not in ("avoid", "hotspot", "none"):
        return _err(f"unknown policy {policy!r}", policy=["avoid (never pass)", "hotspot (pass as little as possible)", "none"])
    for key in ("avoid_regions", "hotspot_regions"):
        if region in ws.spec[key]:
            ws.spec[key].remove(region)
    if policy == "avoid":
        ws.spec["avoid_regions"].append(region)
    elif policy == "hotspot":
        ws.spec["hotspot_regions"].append(region)
    return _ok(ws)


def set_required_region(ws: Workspace, destination: str, region: Optional[str]) -> Dict:
    if destination not in DESTINATIONS + ("all",):
        return _err(f"unknown destination {destination!r}", destination=list(DESTINATIONS) + ["all"])
    if region is None:
        ws.spec["via"].pop(destination, None)
        return _ok(ws)
    if region not in REGIONS:
        return _err(f"unknown region {region!r}", regions=list(REGIONS))
    ws.spec["via"][destination] = region
    return _ok(ws)


def set_antenna_count(ws: Workspace, station: str, count: int) -> Dict:
    if station not in STATIONS:
        return _err(f"unknown station {station!r}", station=list(STATIONS))
    if not isinstance(count, int) or not 0 <= count <= ws.instance.antennas:
        return _err(f"count must be an integer between 0 and {ws.instance.antennas}")
    if count == ws.instance.antennas:
        ws.spec["antennas"].pop(station, None)
    else:
        ws.spec["antennas"][station] = count
    return _ok(ws)


def set_new_link_limit(ws: Workspace, limit: Optional[int]) -> Dict:
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        return _err("limit must be a non-negative integer or null")
    ws.spec["max_new_links"] = limit
    return _ok(ws)


def set_minimum_sessions(ws: Workspace, destination: str, count: int) -> Dict:
    if destination not in DESTINATIONS:
        return _err(f"unknown destination {destination!r}", destination=list(DESTINATIONS))
    if not isinstance(count, int) or count < 0:
        return _err("count must be a non-negative integer")
    if count == 0:
        ws.spec["min_admitted"].pop(destination, None)
    else:
        ws.spec["min_admitted"][destination] = count
    return _ok(ws)


def set_priority(ws: Workspace, destinations: List[str]) -> Dict:
    bad = [d for d in destinations if d not in DESTINATIONS]
    if bad:
        return _err(f"unknown destinations {bad}", destination=list(DESTINATIONS))
    ws.spec["priority"] = list(dict.fromkeys(destinations))
    return _ok(ws)


def set_objective_order(ws: Workspace, order: List[str]) -> Dict:
    if sorted(order) != sorted(OBJECTIVES):
        return _err("order must contain each objective exactly once", objectives=list(OBJECTIVES))
    ws.spec["objective"] = list(order)
    return _ok(ws)


def _summary(ws: Workspace, prepared, solution: Solution) -> Dict:
    r = replay(prepared, solution, *ws.env)
    keep = ("requested", "admitted", "completed", "mean_delay_ms", "new_links", "new_link_limit_ok",
            "via_violations", "hotspot_visits", "sessions_through_hotspot", "min_guarantee_ok", "completed_by_destination")
    return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k in keep}


def solve(ws: Workspace, method: str = "fast") -> Dict:
    if method not in ("fast", "exact"):
        return _err(f"unknown method {method!r}", method=["fast (under 1 s)", "exact (CP-SAT, up to about a minute)"])
    try:
        prepared = prepare(ws.instance, ws.conditions(), ws.latlon)
    except ValueError as exc:
        return _err(str(exc))
    if method == "exact":
        result = solve_exact(ws.instance, time_limit_s=30.0, hops_time_limit_s=5.0, prepared=prepared)
        if result.status and result.status[0] == "INFEASIBLE":
            return _err("the conditions cannot all hold together (CP-SAT proved infeasibility)")
        solution = result.solution
    else:
        solution = solve_fast(prepared)
    ws.last_solution = solution
    return {"ok": True, "method": method, **_summary(ws, prepared, solution)}


def diagnose_conflicts(ws: Workspace) -> Dict:
    base = solve_fast(prepare(ws.instance, ws.conditions(), ws.latlon))
    report = {"admitted_with_all_conditions": len(base.routes), "admitted_if_relaxed": {}}
    for key, empty in (("failed_satellites", []), ("avoid_regions", []), ("via", {}), ("antennas", {}),
                       ("max_new_links", None), ("min_admitted", {})):
        if ws.spec[key] in ([], {}, None):
            continue
        spec = json.loads(json.dumps(ws.spec))
        spec[key] = empty
        sol = solve_fast(prepare(ws.instance, Conditions.from_dict(spec), ws.latlon))
        report["admitted_if_relaxed"][key] = len(sol.routes)
    return {"ok": True, **report}


def finish(ws: Workspace, explanation: str = "") -> Dict:
    ws.finished, ws.explanation = True, explanation
    return {"ok": True, "final_conditions": ws.spec}


def _schema(name: str, description: str, properties: Dict, required: List[str]) -> Dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


REGION = {"type": "string", "enum": list(REGIONS)}
DEST = {"type": "string", "enum": list(DESTINATIONS)}
TOOLS: Dict[str, tuple] = {
    "get_network_state": (get_network_state, _schema("get_network_state", "Sessions, capacity, antenna satellites per station, regions.", {}, [])),
    "lookup_satellite": (lookup_satellite, _schema("lookup_satellite", "Position, regions, and antenna role of one satellite.",
                                                   {"satellite_id": {"type": "integer"}}, ["satellite_id"])),
    "set_failed_satellites": (set_failed_satellites, _schema("set_failed_satellites", "Satellites that are down (replaces the list).",
                                                             {"satellite_ids": {"type": "array", "items": {"type": "integer"}}}, ["satellite_ids"])),
    "set_region_policy": (set_region_policy, _schema("set_region_policy", "avoid = never route over the region; hotspot = route over it as little as possible; none = no rule.",
                                                     {"region": REGION, "policy": {"type": "string", "enum": ["avoid", "hotspot", "none"]}}, ["region", "policy"])),
    "set_required_region": (set_required_region, _schema("set_required_region", "Sessions to a destination (or all) must pass over a region; region null removes the rule.",
                                                         {"destination": {"type": "string", "enum": list(DESTINATIONS) + ["all"]}, "region": {"type": ["string", "null"], "enum": list(REGIONS) + [None]}}, ["destination", "region"])),
    "set_antenna_count": (set_antenna_count, _schema("set_antenna_count", "Antennas currently usable at a ground station.",
                                                     {"station": {"type": "string", "enum": list(STATIONS)}, "count": {"type": "integer"}}, ["station", "count"])),
    "set_new_link_limit": (set_new_link_limit, _schema("set_new_link_limit", "Maximum number of laser terminal re-pointings (new links); null = no limit.",
                                                       {"limit": {"type": ["integer", "null"]}}, ["limit"])),
    "set_minimum_sessions": (set_minimum_sessions, _schema("set_minimum_sessions", "Guarantee at least this many sessions to a destination (0 removes).",
                                                           {"destination": DEST, "count": {"type": "integer"}}, ["destination", "count"])),
    "set_priority": (set_priority, _schema("set_priority", "Destinations whose sessions are admitted first, in order.",
                                           {"destinations": {"type": "array", "items": DEST}}, ["destinations"])),
    "set_objective_order": (set_objective_order, _schema("set_objective_order", "Order of objectives: admitted (more sessions), hotspot (fewer hotspot visits), new_links (fewer re-pointings), hops (shorter routes).",
                                                         {"order": {"type": "array", "items": {"type": "string", "enum": list(OBJECTIVES)}}}, ["order"])),
    "solve": (solve, _schema("solve", "Route the sessions under the current conditions and replay them in the simulator.",
                             {"method": {"type": "string", "enum": ["fast", "exact"]}}, ["method"])),
    "diagnose_conflicts": (diagnose_conflicts, _schema("diagnose_conflicts", "Show how many sessions would be admitted if each condition were relaxed.", {}, [])),
    "finish": (finish, _schema("finish", "Finish with a short explanation for the operator.",
                               {"explanation": {"type": "string"}}, ["explanation"])),
}
TOOL_SCHEMAS = [schema for _, schema in TOOLS.values()]


def call_tool(ws: Workspace, name: str, arguments: str) -> Dict:
    if name not in TOOLS:
        return _err(f"unknown tool {name!r}", tools=list(TOOLS))
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return _err(f"arguments are not valid JSON: {exc}")
    try:
        return TOOLS[name][0](ws, **args)
    except TypeError as exc:
        return _err(f"bad arguments: {exc}")
