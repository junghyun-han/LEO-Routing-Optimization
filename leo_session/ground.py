"""Ground endpoints and ground-to-satellite contact intervals for v3.

The model is intentionally modest: a spherical rotating Earth, fixed ground
coordinates, and a literature-based maximum GSL slant range.  This is enough
to make ingress and egress satellites change with time without pretending to
be a link-budget simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from .config import DEFAULT_CONFIG, EnvironmentConfig
from .orbit import WalkerDeltaConstellation


# IERS conventional mean angular velocity, rad/s.  The spherical-Earth model
# uses this only to rotate fixed ground coordinates into ECI.
EARTH_ROTATION_RAD_S = 7.2921150e-5


@dataclass(frozen=True)
class GroundStation:
    key: str
    name: str
    latitude_deg: float
    longitude_deg: float

    def eci(self, time_s: float, earth_radius_km: float) -> np.ndarray:
        lat = math.radians(self.latitude_deg)
        lon = math.radians(self.longitude_deg) + EARTH_ROTATION_RAD_S * float(time_s)
        return earth_radius_km * np.asarray(
            [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
            dtype=np.float64,
        )


# Coordinates are ordinary city-centre coordinates.  They locate the
# simulation endpoints; they are not claimed to be operational gateway sites.
GROUND_STATIONS: Dict[str, GroundStation] = {
    "new_york": GroundStation("new_york", "New York", 40.71427, -74.00597),
    "london": GroundStation("london", "London", 51.50853, -0.12574),
    "istanbul": GroundStation("istanbul", "Istanbul", 41.01384, 28.94966),
    "hanoi": GroundStation("hanoi", "Hanoi", 21.02450, 105.84117),
}

# Bhattacharjee et al. (2023) evaluate all three pairs.  The later 2024 TAES
# paper evaluates the short and long pair only.
GROUND_PAIRS: Dict[str, Tuple[str, str]] = {
    "short": ("new_york", "london"),
    "medium": ("new_york", "istanbul"),
    "long": ("new_york", "hanoi"),
}


@dataclass(frozen=True, order=True)
class GroundContact:
    station: str
    satellite: int
    start_s: float
    end_s: float


class GroundContactTrace:
    """Non-periodic GSL contacts over one explicitly generated experiment window."""

    def __init__(self, horizon_s: float, contacts: Sequence[GroundContact]):
        self.horizon_s = float(horizon_s)
        by_station: Dict[str, Dict[int, List[Tuple[float, float]]]] = {}
        for item in contacts:
            by_station.setdefault(item.station, {}).setdefault(item.satellite, []).append(
                (item.start_s, item.end_s)
            )
        self._by_station = {
            name: {sat: tuple(sorted(windows)) for sat, windows in sats.items()}
            for name, sats in by_station.items()
        }

    def contact_window(self, station: str, satellite: int, time_s: float) -> Optional[Tuple[float, float]]:
        for start, end in self._by_station.get(station, {}).get(int(satellite), ()):
            if start - 1.0e-9 <= time_s <= end + 1.0e-9:
                return start, end
        return None

    def visible(self, station: str, time_s: float) -> List[int]:
        return sorted(
            sat
            for sat in self._by_station.get(station, {})
            if self.contact_window(station, sat, time_s) is not None
        )

    def selected_access(self, station: str, time_s: float) -> Optional[int]:
        """Common access satellite used by every routing method.

        Choosing the visible contact with the latest end time keeps GSL
        selection outside the ISL-routing comparison and avoids an imminent
        access handover when another currently visible satellite is available.
        """
        visible = self.visible(station, time_s)
        if not visible:
            return None
        return max(
            visible,
            key=lambda sat: (self.remaining_s(station, sat, time_s), -int(sat)),
        )

    def remaining_s(self, station: str, satellite: int, time_s: float) -> float:
        hit = self.contact_window(station, satellite, time_s)
        return 0.0 if hit is None else max(0.0, hit[1] - float(time_s))

    def duration_samples(self) -> np.ndarray:
        return np.asarray(
            [end - start for sats in self._by_station.values() for windows in sats.values() for start, end in windows],
            dtype=np.float64,
        )


class GroundContactGenerator:
    def __init__(
        self,
        constellation: WalkerDeltaConstellation,
        stations: Mapping[str, GroundStation] = GROUND_STATIONS,
        config: EnvironmentConfig = DEFAULT_CONFIG,
        gsl_range_km: Optional[float] = None,
        scan_step_s: Optional[float] = None,
    ):
        self.constellation = constellation
        self.stations = dict(stations)
        self.config = config
        self.gsl_range_km = float(config.gsl_range_km if gsl_range_km is None else gsl_range_km)
        self.scan_step_s = float(
            config.contact_scan_step_s if scan_step_s is None else scan_step_s
        )

    def _margins(self, station: GroundStation, time_s: float) -> np.ndarray:
        ground = station.eci(time_s, self.config.earth_radius_km)
        slant = np.linalg.norm(self.constellation.positions(time_s) - ground[None, :], axis=1)
        # At h=550 km, the adopted 1000 km slant-range cap is well inside the
        # geometric horizon.  A separate elevation mask is therefore unnecessary
        # for this deliberately simple reference model.
        return self.gsl_range_km - slant

    def _margin_one(self, station: GroundStation, satellite: int, time_s: float) -> float:
        ground = station.eci(time_s, self.config.earth_radius_km)
        slant = np.linalg.norm(self.constellation.position(satellite, time_s) - ground)
        return self.gsl_range_km - float(slant)

    def generate(self, horizon_s: float) -> GroundContactTrace:
        horizon_s = float(horizon_s)
        times = list(np.arange(0.0, horizon_s, self.scan_step_s, dtype=np.float64))
        if not times or times[-1] < horizon_s:
            times.append(horizon_s)
        contacts: List[GroundContact] = []

        for key, station in self.stations.items():
            previous_t = float(times[0])
            previous_margin = self._margins(station, previous_t)
            active = previous_margin >= 0.0
            starts = np.where(active, 0.0, np.nan)
            for current_t in times[1:]:
                current_t = float(current_t)
                margin = self._margins(station, current_t)
                current = margin >= 0.0
                for satellite in np.where(~active & current)[0]:
                    starts[satellite] = brentq(
                        lambda t, s=int(satellite): self._margin_one(station, s, t),
                        previous_t,
                        current_t,
                    )
                for satellite in np.where(active & ~current)[0]:
                    end = brentq(
                        lambda t, s=int(satellite): self._margin_one(station, s, t),
                        previous_t,
                        current_t,
                    )
                    contacts.append(GroundContact(key, int(satellite), float(starts[satellite]), float(end)))
                    starts[satellite] = np.nan
                active = current
                previous_margin = margin
                previous_t = current_t
            for satellite in np.where(active)[0]:
                contacts.append(GroundContact(key, int(satellite), float(starts[satellite]), horizon_s))

        return GroundContactTrace(horizon_s, contacts)
