"""Analytic circular Walker-Delta orbit model in ECI coordinates."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .config import DEFAULT_CONFIG, EnvironmentConfig


@dataclass(frozen=True)
class SatelliteAddress:
    plane: int
    slot: int


class WalkerDeltaConstellation:
    """Reference Walker T/P/F constellation with circular two-body motion.

    Node id is ``plane * satellites_per_plane + slot``.  The logical node
    indexing is 2-D, but every contact decision is generated from 3-D ECI
    coordinates and an Earth line-of-sight check.
    """

    def __init__(self, config: EnvironmentConfig = DEFAULT_CONFIG):
        self.config = config
        self._inclination = math.radians(config.inclination_deg)
        self._radius = config.orbital_radius_km
        self._mean_motion = math.sqrt(config.earth_mu_km3_s2 / self._radius**3)
        self._period = 2.0 * math.pi / self._mean_motion

        planes = np.repeat(np.arange(config.orbital_planes), config.satellites_per_plane)
        slots = np.tile(np.arange(config.satellites_per_plane), config.orbital_planes)
        self._raan = 2.0 * math.pi * planes / config.orbital_planes
        self._u0 = (
            2.0 * math.pi * slots / config.satellites_per_plane
            + 2.0 * math.pi * config.walker_phasing * planes / config.total_satellites
        )

    @property
    def period_s(self) -> float:
        return self._period

    @property
    def mean_motion_rad_s(self) -> float:
        return self._mean_motion

    def node_id(self, plane: int, slot: int) -> int:
        p = int(plane) % self.config.orbital_planes
        q = int(slot) % self.config.satellites_per_plane
        return p * self.config.satellites_per_plane + q

    def address(self, node: int) -> SatelliteAddress:
        p, q = divmod(int(node), self.config.satellites_per_plane)
        if not 0 <= p < self.config.orbital_planes:
            raise IndexError(node)
        return SatelliteAddress(p, q)

    def positions(self, time_s: float) -> np.ndarray:
        u = self._u0 + self._mean_motion * float(time_s)
        cu, su = np.cos(u), np.sin(u)
        co, so = np.cos(self._raan), np.sin(self._raan)
        ci, si = math.cos(self._inclination), math.sin(self._inclination)
        out = np.empty((self.config.total_satellites, 3), dtype=np.float64)
        out[:, 0] = self._radius * (co * cu - so * su * ci)
        out[:, 1] = self._radius * (so * cu + co * su * ci)
        out[:, 2] = self._radius * (su * si)
        return out

    def position(self, node: int, time_s: float) -> np.ndarray:
        idx = int(node)
        u = self._u0[idx] + self._mean_motion * float(time_s)
        cu, su = math.cos(u), math.sin(u)
        co, so = math.cos(self._raan[idx]), math.sin(self._raan[idx])
        ci, si = math.cos(self._inclination), math.sin(self._inclination)
        return self._radius * np.array(
            [co * cu - so * su * ci, so * cu + co * su * ci, su * si],
            dtype=np.float64,
        )

    def persistent_edges(self) -> np.ndarray:
        edges = []
        for p in range(self.config.orbital_planes):
            for q in range(self.config.satellites_per_plane):
                u = self.node_id(p, q)
                v = self.node_id(p, q + 1)
                edges.append((min(u, v), max(u, v)))
        return np.asarray(sorted(set(edges)), dtype=np.int32)

    def adjacent_plane_candidate_pairs(self) -> np.ndarray:
        """All cross-plane pairs on adjacent plane interfaces.

        Range and LOS are applied by the contact generator.  Listing all 66x66
        pairs per interface avoids silently excluding a short-lived contact by
        a nearest-neighbour shortcut.
        """
        pairs = []
        qn = self.config.satellites_per_plane
        for p in range(self.config.orbital_planes):
            pn = (p + 1) % self.config.orbital_planes
            left = np.arange(p * qn, (p + 1) * qn, dtype=np.int32)
            right = np.arange(pn * qn, (pn + 1) * qn, dtype=np.int32)
            uu = np.repeat(left, qn)
            vv = np.tile(right, qn)
            block = np.column_stack([np.minimum(uu, vv), np.maximum(uu, vv)])
            pairs.append(block)
        return np.unique(np.concatenate(pairs, axis=0), axis=0)

    def link_geometry(self, u: int, v: int, time_s: float) -> Tuple[float, float]:
        a = self.position(u, time_s)
        b = self.position(v, time_s)
        distance = float(np.linalg.norm(a - b))
        closest_to_earth = float(np.linalg.norm(0.5 * (a + b)))
        return distance, closest_to_earth

    def feasible(self, u: int, v: int, time_s: float) -> bool:
        distance, closest = self.link_geometry(u, v, time_s)
        return (
            distance <= self.config.lisl_range_km
            and closest >= self.config.earth_radius_km
        )
