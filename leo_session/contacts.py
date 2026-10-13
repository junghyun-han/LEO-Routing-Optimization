"""Offline generation and querying of periodic inter-plane contact intervals."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from .config import DEFAULT_CONFIG, EnvironmentConfig
from .orbit import WalkerDeltaConstellation


@dataclass(frozen=True, order=True)
class ContactInterval:
    u: int
    v: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class ContactTrace:
    def __init__(self, period_s: float, intervals: Sequence[ContactInterval]):
        self.period_s = float(period_s)
        self.intervals = tuple(sorted(intervals))
        by_pair: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        for item in self.intervals:
            key = (min(item.u, item.v), max(item.u, item.v))
            by_pair.setdefault(key, []).append((item.start_s, item.end_s))
        self._by_pair = {key: tuple(value) for key, value in by_pair.items()}

    @property
    def pairs(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(self._by_pair)

    def _absolute_window(self, start: float, end: float, time_s: float) -> Optional[Tuple[float, float]]:
        k = math.floor((float(time_s) - start) / self.period_s)
        a = start + k * self.period_s
        b = end + k * self.period_s
        eps = 1.0e-9
        if a - eps <= time_s <= b + eps:
            return a, b
        return None

    def contact_window(self, u: int, v: int, time_s: float) -> Optional[Tuple[float, float]]:
        key = (min(int(u), int(v)), max(int(u), int(v)))
        for start, end in self._by_pair.get(key, ()): 
            hit = self._absolute_window(start, end, float(time_s))
            if hit is not None:
                return hit
        return None

    def is_feasible(self, u: int, v: int, time_s: float) -> bool:
        return self.contact_window(u, v, time_s) is not None

    def remaining_s(self, u: int, v: int, time_s: float) -> float:
        hit = self.contact_window(u, v, time_s)
        return 0.0 if hit is None else max(0.0, hit[1] - float(time_s))

    def age_s(self, u: int, v: int, time_s: float) -> float:
        hit = self.contact_window(u, v, time_s)
        return 0.0 if hit is None else max(0.0, float(time_s) - hit[0])

    def active_edges(self, time_s: float) -> List[Tuple[int, int, float]]:
        out = []
        for u, v in self._by_pair:
            hit = self.contact_window(u, v, time_s)
            if hit is not None:
                out.append((u, v, hit[1]))
        return out

    def duration_samples(self) -> np.ndarray:
        return np.asarray([x.duration_s for x in self.intervals], dtype=np.float64)

    def save(self, path: Path, metadata: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(
            [(x.u, x.v, x.start_s, x.end_s) for x in self.intervals], dtype=np.float64
        )
        np.savez_compressed(
            path,
            period_s=np.asarray([self.period_s], dtype=np.float64),
            intervals=arr,
            metadata=np.asarray([json.dumps(metadata or {}, sort_keys=True)]),
        )

    @classmethod
    def load(cls, path: Path) -> "ContactTrace":
        data = np.load(Path(path), allow_pickle=False)
        rows = data["intervals"]
        intervals = [
            ContactInterval(int(row[0]), int(row[1]), float(row[2]), float(row[3]))
            for row in rows
        ]
        return cls(float(data["period_s"][0]), intervals)


class ContactTraceGenerator:
    def __init__(
        self,
        constellation: WalkerDeltaConstellation,
        config: EnvironmentConfig = DEFAULT_CONFIG,
    ):
        self.constellation = constellation
        self.config = config

    def _margin(self, u: int, v: int, time_s: float) -> float:
        distance, closest = self.constellation.link_geometry(u, v, time_s)
        return min(
            self.config.lisl_range_km - distance,
            closest - self.config.earth_radius_km,
        )

    def _refine_crossing(
        self, u: int, v: int, t0: float, t1: float, m0: float, m1: float
    ) -> float:
        if abs(m0) < 1.0e-12:
            return t0
        if abs(m1) < 1.0e-12:
            return t1
        try:
            return float(
                brentq(
                    lambda t: self._margin(u, v, t),
                    t0,
                    t1,
                    xtol=self.config.root_tolerance_s,
                    rtol=1.0e-12,
                )
            )
        except ValueError:
            # Defensive interpolation fallback for a grazing numerical contact.
            frac = abs(m0) / max(1.0e-15, abs(m0) + abs(m1))
            return t0 + frac * (t1 - t0)

    def _vector_margins(self, pairs: np.ndarray, time_s: float) -> np.ndarray:
        pos = self.constellation.positions(time_s)
        a = pos[pairs[:, 0]]
        b = pos[pairs[:, 1]]
        distance = np.linalg.norm(a - b, axis=1)
        closest = np.linalg.norm(0.5 * (a + b), axis=1)
        return np.minimum(
            self.config.lisl_range_km - distance,
            closest - self.config.earth_radius_km,
        )

    def generate(self, pairs: Optional[np.ndarray] = None) -> ContactTrace:
        pairs = (
            self.constellation.adjacent_plane_candidate_pairs()
            if pairs is None
            else np.asarray(pairs, dtype=np.int32)
        )
        period = self.constellation.period_s
        step = self.config.contact_scan_step_s
        times = list(np.arange(0.0, period, step, dtype=np.float64))
        if not times or times[-1] < period:
            times.append(period)

        prev_t = float(times[0])
        prev_margin = self._vector_margins(pairs, prev_t)
        active = prev_margin >= 0.0
        starts = np.where(active, 0.0, np.nan)
        raw: Dict[int, List[Tuple[float, float]]] = {}

        for now_t in times[1:]:
            now_t = float(now_t)
            margin = self._vector_margins(pairs, now_t)
            current = margin >= 0.0
            opened = np.where(~active & current)[0]
            closed = np.where(active & ~current)[0]
            for idx in opened:
                u, v = map(int, pairs[idx])
                starts[idx] = self._refine_crossing(
                    u, v, prev_t, now_t, float(prev_margin[idx]), float(margin[idx])
                )
            for idx in closed:
                u, v = map(int, pairs[idx])
                end = self._refine_crossing(
                    u, v, prev_t, now_t, float(prev_margin[idx]), float(margin[idx])
                )
                raw.setdefault(int(idx), []).append((float(starts[idx]), end))
                starts[idx] = np.nan
            active = current
            prev_margin = margin
            prev_t = now_t

        for idx in np.where(active)[0]:
            raw.setdefault(int(idx), []).append((float(starts[idx]), period))

        intervals: List[ContactInterval] = []
        for idx, windows in raw.items():
            windows.sort()
            if len(windows) >= 2 and windows[0][0] <= 1.0e-8 and windows[-1][1] >= period - 1.0e-8:
                merged = (windows[-1][0] - period, windows[0][1])
                windows = [merged] + windows[1:-1]
            u, v = map(int, pairs[idx])
            for start, end in windows:
                if end > start + self.config.root_tolerance_s:
                    intervals.append(ContactInterval(u, v, start, end))
        return ContactTrace(period, intervals)
