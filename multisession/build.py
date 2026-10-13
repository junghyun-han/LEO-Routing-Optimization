"""Build multi-session instances from the v3 constellation, contacts, and ground access."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from leo_session import ContactTrace, GroundContactGenerator, WalkerDeltaConstellation
from leo_session.config import DEFAULT_CONFIG, EnvironmentConfig
from leo_session.ground import GROUND_PAIRS
from leo_session.topology import build_initial_topology

from .model import Instance, Session, Snapshot, canonical_edge

PAIR_ORDER = ("short", "medium", "long")


def load_environment(trace_path: Path, horizon_s: float, config: EnvironmentConfig = DEFAULT_CONFIG):
    constellation = WalkerDeltaConstellation(config)
    isl_trace = ContactTrace.load(trace_path)
    ground_trace = GroundContactGenerator(constellation, config=config).generate(horizon_s)
    return constellation, isl_trace, ground_trace


def build_snapshot(constellation, isl_trace, ground_trace, time_s: float,
                   stations: Iterable[str], config: EnvironmentConfig = DEFAULT_CONFIG) -> Snapshot:
    persistent = [tuple(map(int, e)) for e in constellation.persistent_edges()]
    dynamic = {canonical_edge(u, v): float(end) - float(time_s) for u, v, end in isl_trace.active_edges(time_s)}
    initial = build_initial_topology(constellation, isl_trace, time_s, config)
    access = {}
    for station in stations:
        ranked = sorted(
            ((int(s), ground_trace.remaining_s(station, s, time_s)) for s in ground_trace.visible(station, time_s)),
            key=lambda item: (-item[1], item[0]),  # same order as v3 selected_access
        )
        access[station] = tuple(ranked)
    return Snapshot(float(time_s), tuple(persistent), dynamic, tuple(initial.dynamic), access,
                    config.dynamic_setup_s)


def make_sessions(per_destination: int, duration_s: float, pairs: Sequence[str] = PAIR_ORDER) -> Tuple[Session, ...]:
    """Round-robin over the v3 ground pairs so no destination is served first."""
    sessions: List[Session] = []
    for _ in range(per_destination):
        for name in pairs:
            source, destination = GROUND_PAIRS[name]
            sessions.append(Session(len(sessions), source, destination, float(duration_s)))
    return tuple(sessions)


def make_instance(snapshot: Snapshot, per_destination: int, duration_s: float, antennas: int,
                  capacity: int) -> Instance:
    return Instance(snapshot, make_sessions(per_destination, duration_s), int(antennas), int(capacity))


def decision_times(request_file: Path, count: int = 40) -> List[Tuple[float, str]]:
    """Evenly spaced v3 request times; even positions = validation, odd = test.

    Uses the 240 frozen offsets of pair 'short' at X = 20 s, which span one
    orbital period.  The split is fixed before any method is designed.
    """
    with open(request_file, newline="") as handle:
        times = sorted(
            float(row["request_time_s"])
            for row in csv.DictReader(handle)
            if row["pair"] == "short" and float(row["duration_s"]) == 20.0
        )
    step = len(times) / count
    picked = [times[int(i * step)] for i in range(count)]
    return [(t, "validation" if i % 2 == 0 else "test") for i, t in enumerate(picked)]
