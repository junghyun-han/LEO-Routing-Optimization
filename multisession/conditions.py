"""Operator conditions given in text, after the LLM turns them into this structure.

Hard conditions change the instance (failed satellites, regions to avoid, antenna
outages) or constrain the solution (pass through a region, at most m new links,
at least n sessions per destination).  Soft conditions change the objective
(avoid a hotspot if possible, serve some destinations first, put fewer
reconfigurations above more sessions).  ``check`` verifies a solution against
the conditions independently of how it was produced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .model import Instance, Snapshot, Solution, canonical_edge, route_edges

EARTH_ROTATION_RAD_S = 7.2921150e-5  # same constant as leo_session.ground

# Approximate latitude/longitude boxes over the sub-satellite point (degrees).
# They are operating areas, not political borders.  Longitudes run west to east.
REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "north_america_east": (25.0, 50.0, -90.0, -60.0),
    "north_atlantic": (30.0, 60.0, -60.0, -10.0),
    "europe": (35.0, 60.0, -10.0, 40.0),
    "mediterranean": (30.0, 45.0, -5.0, 36.0),
    "africa": (-35.0, 30.0, -18.0, 52.0),
    "middle_east": (12.0, 42.0, 34.0, 63.0),
    "south_asia": (5.0, 35.0, 63.0, 92.0),
    "indian_ocean": (-40.0, 5.0, 50.0, 100.0),
    "southeast_asia": (-10.0, 25.0, 92.0, 130.0),
    "east_asia": (20.0, 50.0, 100.0, 145.0),
}

OBJECTIVES = ("admitted", "hotspot", "new_links", "hops")


@dataclass(frozen=True)
class Conditions:
    failed_satellites: Tuple[int, ...] = ()
    avoid_regions: Tuple[str, ...] = ()  # hard: never route through satellites over these regions
    hotspot_regions: Tuple[str, ...] = ()  # soft: pass through as little as possible
    via: Tuple[Tuple[str, str], ...] = ()  # (destination or "all", region) the route must pass through
    antennas: Tuple[Tuple[str, int], ...] = ()  # (station, antenna count) overrides
    max_new_links: Optional[int] = None
    min_admitted: Tuple[Tuple[str, int], ...] = ()  # (destination, minimum sessions)
    priority: Tuple[str, ...] = ()  # destinations whose admissions are maximized first, in order
    objective: Tuple[str, ...] = OBJECTIVES  # order of the remaining objectives

    def __post_init__(self):
        for r in self.avoid_regions + self.hotspot_regions + tuple(r for _, r in self.via):
            if r not in REGIONS:
                raise ValueError(f"unknown region {r!r}; known: {sorted(REGIONS)}")
        if sorted(self.objective) != sorted(OBJECTIVES):
            raise ValueError(f"objective must order exactly {OBJECTIVES}")

    @classmethod
    def from_dict(cls, data: Dict) -> "Conditions":
        def pairs(value, cast=int):
            return tuple(sorted((str(k), cast(v)) for k, v in (value or {}).items()))
        return cls(
            failed_satellites=tuple(sorted(int(s) for s in data.get("failed_satellites", []))),
            avoid_regions=tuple(sorted(data.get("avoid_regions", []))),
            hotspot_regions=tuple(sorted(data.get("hotspot_regions", []))),
            via=pairs(data.get("via"), str),
            antennas=pairs(data.get("antennas")),
            max_new_links=data.get("max_new_links"),
            min_admitted=pairs(data.get("min_admitted")),
            priority=tuple(data.get("priority", [])),
            objective=tuple(data.get("objective", OBJECTIVES)),
        )

    def to_dict(self) -> Dict:
        return {"failed_satellites": list(self.failed_satellites), "avoid_regions": list(self.avoid_regions),
                "hotspot_regions": list(self.hotspot_regions), "via": dict(self.via),
                "antennas": dict(self.antennas), "max_new_links": self.max_new_links,
                "min_admitted": dict(self.min_admitted), "priority": list(self.priority),
                "objective": list(self.objective)}


def satellite_latlon(constellation, time_s: float) -> np.ndarray:
    """Sub-satellite latitude/longitude in degrees (spherical Earth), shape (N, 2)."""
    pos = constellation.positions(time_s)
    r = np.linalg.norm(pos, axis=1)
    lat = np.degrees(np.arcsin(pos[:, 2] / r))
    lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]) - EARTH_ROTATION_RAD_S * float(time_s))
    lon = (lon + 180.0) % 360.0 - 180.0
    return np.stack([lat, lon], axis=1)


def region_satellites(latlon: np.ndarray, region: str) -> Set[int]:
    lat_lo, lat_hi, lon_lo, lon_hi = REGIONS[region]
    lat, lon = latlon[:, 0], latlon[:, 1]
    mask = (lat >= lat_lo) & (lat <= lat_hi) & (lon >= lon_lo) & (lon <= lon_hi)
    return {int(i) for i in np.nonzero(mask)[0]}


@dataclass
class Prepared:
    """Instance with hard conditions applied, plus the satellite sets the rest needs."""
    instance: Instance
    conditions: Conditions
    hotspot: Set[int] = field(default_factory=set)
    banned: Set[int] = field(default_factory=set)  # failed satellites and satellites over avoided regions
    avoid: Set[int] = field(default_factory=set)  # satellites over avoided regions only
    via: Dict[str, Set[int]] = field(default_factory=dict)  # destination -> satellites to pass through

    def via_for(self, destination: str) -> Optional[Set[int]]:
        return self.via.get(destination)


def prepare(instance: Instance, conditions: Conditions, latlon: np.ndarray) -> Prepared:
    snap = instance.snapshot
    avoid: Set[int] = set()
    for region in conditions.avoid_regions:
        avoid |= region_satellites(latlon, region)
    banned = set(conditions.failed_satellites) | avoid

    def keep(e):
        return e[0] not in banned and e[1] not in banned

    reduced = Snapshot(
        snap.time_s,
        tuple(e for e in snap.persistent if keep(e)),
        {e: r for e, r in snap.dynamic_remaining.items() if keep(e)},
        tuple(e for e in snap.initial_dynamic if keep(e)),
        {st: tuple((s, g) for s, g in sats if s not in banned) for st, sats in snap.access.items()},
        snap.setup_s,
    )
    inst = replace(instance, snapshot=reduced, antenna_overrides=dict(conditions.antennas))
    hotspot = set().union(*(region_satellites(latlon, r) for r in conditions.hotspot_regions)) \
        if conditions.hotspot_regions else set()
    via = {}
    destinations = {s.destination for s in instance.sessions}
    for dest, region in conditions.via:
        for d in (destinations if dest == "all" else {dest}):
            via[d] = region_satellites(latlon, region) - banned
    return Prepared(inst, conditions, hotspot, banned, avoid, via)


def hotspot_visits(prepared: Prepared, solution: Solution) -> int:
    return sum(sum(1 for v in path if v in prepared.hotspot) for path in solution.routes.values())


def check(prepared: Prepared, solution: Solution) -> List[str]:
    """Hard-condition violations (the base constraints are checked by model.verify)."""
    inst, cond = prepared.instance, prepared.conditions
    problems: List[str] = []
    dest_of = {s.index: s.destination for s in inst.sessions}
    for k, path in solution.routes.items():
        if prepared.banned & set(path):
            problems.append(f"session {k} uses a failed satellite or an avoided region")
        need = prepared.via_for(dest_of[k])
        if need is not None and not (need & set(path)):
            problems.append(f"session {k} to {dest_of[k]} does not pass through the required region")
    if cond.max_new_links is not None:
        new = set(map(lambda e: canonical_edge(*e), solution.final_dynamic)) - set(inst.snapshot.initial_dynamic)
        if len(new) > cond.max_new_links:
            problems.append(f"{len(new)} new links > limit {cond.max_new_links}")
    for dest, n in cond.min_admitted:
        got = sum(1 for k in solution.routes if dest_of[k] == dest)
        if got < n:
            problems.append(f"{got} sessions to {dest} < required {n}")
    return problems
