"""Region boxes for the region conditions, and sub-satellite latitude/longitude.

A satellite is over a region when its sub-satellite point lies inside the
region's latitude/longitude box.  The prompts, the tools, the CP-SAT data, and
the compliance measurement all use these boxes.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

EARTH_ROTATION_RAD_S = 7.2921150e-5  # same constant as leo_session.ground

# (lat_min, lat_max, lon_min, lon_max) in degrees
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

# Korean names used in the operator text
KO_REGION = {"north_america_east": "북미 동부", "north_atlantic": "북대서양", "europe": "유럽", "mediterranean": "지중해",
             "africa": "아프리카", "middle_east": "중동", "south_asia": "남아시아", "indian_ocean": "인도양",
             "southeast_asia": "동남아시아", "east_asia": "동아시아"}


def satellite_latlon(constellation, time_s: float) -> np.ndarray:
    """Sub-satellite latitude/longitude in degrees (spherical Earth), shape (N, 2)."""
    pos = constellation.positions(time_s)
    r = np.linalg.norm(pos, axis=1)
    lat = np.degrees(np.arcsin(pos[:, 2] / r))
    lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]) - EARTH_ROTATION_RAD_S * float(time_s))
    lon = (lon + 180.0) % 360.0 - 180.0
    return np.stack([lat, lon], axis=1)
