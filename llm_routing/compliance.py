"""Constraint compliance over the whole session, from the logged routes.

Each established route is followed from its start to the next route's start
(the last one to the end of the session), sampled every ``step_s`` seconds.
Satellite positions come from the v3 constellation, so a satellite that drifts
into a region during the session counts.  Region boxes are
``multisession.conditions.REGIONS`` (the same boxes the prompt and the tools use).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from leo_session.ground import GROUND_STATIONS
from multisession.conditions import REGIONS, satellite_latlon

C_KM_S = 299792.458


def in_region(latlon: np.ndarray, sats: List[int], region: str) -> List[int]:
    lat_lo, lat_hi, lon_lo, lon_hi = REGIONS[region]
    return [s for s in sats if lat_lo <= latlon[s][0] <= lat_hi and lon_lo <= latlon[s][1] <= lon_hi]


def samples(entry: Dict, step_s: float = 10.0):
    """(time, satellites) along the session for one logged request."""
    routes = entry.get("routes") or []
    if not routes:
        return []
    end = entry["established_time_s"] + entry["duration_s"]
    out = []
    for i, r in enumerate(routes):
        stop = routes[i + 1]["time_s"] if i + 1 < len(routes) else end
        for t in np.arange(r["time_s"], max(r["time_s"], stop), step_s):
            out.append((float(t), r["satellites"]))
    return out


def evaluate(entry: Dict, condition: Dict, constellation, step_s: float = 10.0) -> Dict:
    pts = samples(entry, step_s)
    out: Dict[str, Optional[float]] = {"sampled_s": len(pts) * step_s}
    if not pts:
        return out
    radius = constellation.config.earth_radius_km
    delays, hot, avoid = [], [], []
    for t, sats in pts:
        pos = constellation.positions(t)
        km = float(np.linalg.norm(pos[sats[0]] - GROUND_STATIONS[entry["source"]].eci(t, radius)))
        km += float(np.linalg.norm(pos[sats[-1]] - GROUND_STATIONS[entry["destination"]].eci(t, radius)))
        km += sum(float(np.linalg.norm(pos[a] - pos[b])) for a, b in zip(sats, sats[1:]))
        delays.append(km / C_KM_S * 1000.0)
        if condition.get("hotspot_region") or condition.get("avoid_region"):
            ll = satellite_latlon(constellation, t)
            if condition.get("hotspot_region"):  # relays only: ingress and egress are fixed for every method
                hot.append(len(in_region(ll, sats[1:-1], condition["hotspot_region"])))
            if condition.get("avoid_region"):  # relays only, as for hotspots
                avoid.append(len(in_region(ll, sats[1:-1], condition["avoid_region"])) > 0)
    out["mean_delay_ms"] = float(np.mean(delays))
    if "max_delay_ms" in condition:
        out["delay_violation_share"] = float(np.mean([d > condition["max_delay_ms"] for d in delays]))
    if hot:
        out["hotspot_time_share"] = float(np.mean([h > 0 for h in hot]))
        out["hotspot_satellites_mean"] = float(np.mean(hot))
    if avoid:
        out["avoid_violation"] = float(any(avoid))
        out["avoid_time_share"] = float(np.mean(avoid))
    return out
