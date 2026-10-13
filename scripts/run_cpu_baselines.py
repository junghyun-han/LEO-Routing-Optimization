#!/usr/bin/env python3
"""Run paired, CPU-only v3 session experiments.

The defaults are intentionally small enough for a smoke experiment.  Increase
``--sessions-per-pair`` only after the validation tests pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leo_session import (  # noqa: E402
    GROUND_PAIRS,
    ContactTrace,
    GroundContactGenerator,
    SessionEnvironment,
    SessionRequest,
    RewardConfig,
    WalkerDeltaConstellation,
)
from leo_session.config import EnvironmentConfig, PARAMETER_MANIFEST  # noqa: E402


METHODS = (
    "raw",
    "lifetime",
    "prp",
    "remaining",
    "cost_greedy",
    "constrained_oracle",
    "physical_upper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=ROOT / "cache/contact_trace_1584_24_11.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_cpu")
    parser.add_argument("--sessions-per-pair", type=int, default=4)
    parser.add_argument("--durations", type=float, nargs="+", default=[20.0, 60.0, 300.0])
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--setup-delay", type=float, default=20.0)
    parser.add_argument("--gsl-range", type=float, default=1000.0)
    parser.add_argument("--reward-relay-progress-weight", type=float, default=1.0)
    parser.add_argument("--reward-relay-link-weight", type=float, default=0.05)
    parser.add_argument("--reward-relay-lifetime-shortfall-weight", type=float, default=3.0)
    parser.add_argument("--reward-relay-pat-weight", type=float, default=0.15)
    parser.add_argument(
        "--max-relay-actions",
        type=int,
        default=128,
        help="loop-free rollout safety cap (not simulated time)",
    )
    return parser.parse_args()


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return center - half, center + half


def main() -> None:
    args = parse_args()
    if args.sessions_per_pair < 1 or any(x <= 0 for x in args.durations):
        raise SystemExit("sessions and durations must be positive")

    config = EnvironmentConfig(dynamic_setup_s=args.setup_delay, gsl_range_km=args.gsl_range)
    constellation = WalkerDeltaConstellation(config)
    isl_trace = ContactTrace.load(args.trace)
    if args.request_file is not None:
        with args.request_file.open(newline="", encoding="utf-8") as stream:
            request_specs = list(csv.DictReader(stream))
        if not request_specs:
            raise SystemExit("request file is empty")
        max_request_end = max(
            float(row["request_time_s"]) + float(row["duration_s"])
            for row in request_specs
        )
        horizon = max_request_end + 2.0 * constellation.config.dynamic_setup_s
    else:
        max_duration = max(args.durations)
        horizon = constellation.period_s + max_duration + 2.0 * constellation.config.dynamic_setup_s
    ground_trace = GroundContactGenerator(constellation, config=config).generate(horizon)
    reward_config = RewardConfig(
        relay_progress_weight=args.reward_relay_progress_weight,
        relay_link_weight=args.reward_relay_link_weight,
        relay_lifetime_shortfall_weight=args.reward_relay_lifetime_shortfall_weight,
        relay_pat_weight=args.reward_relay_pat_weight,
    )
    environment = SessionEnvironment(
        constellation,
        isl_trace,
        ground_trace,
        config,
        reward_config,
        args.max_relay_actions,
    )

    if args.request_file is None:
        # Mid-bin offsets cover the orbit without putting all methods on
        # different network states. Every method receives the same request list.
        offsets = [
            constellation.period_s * (i + 0.5) / args.sessions_per_pair
            for i in range(args.sessions_per_pair)
        ]
        request_specs = []
        generated_id = 0
        for pair_name, (source, destination) in GROUND_PAIRS.items():
            for duration in args.durations:
                for t0 in offsets:
                    request_specs.append(
                        {
                            "request_id": generated_id,
                            "pair": pair_name,
                            "source": source,
                            "destination": destination,
                            "request_time_s": t0,
                            "duration_s": duration,
                        }
                    )
                    generated_id += 1
    rows = []
    total_requests = len(request_specs)
    started = time.monotonic()
    for request_index, spec in enumerate(request_specs, start=1):
        pair_name = str(spec["pair"])
        source = str(spec["source"])
        destination = str(spec["destination"])
        t0 = float(spec["request_time_s"])
        duration = float(spec["duration_s"])
        request = SessionRequest(source, destination, t0, duration)
        for method in args.methods:
            result = environment.run(method, request)
            row = asdict(result)
            row.update(
                request_id=int(spec["request_id"]),
                pair=pair_name,
                source=source,
                destination=destination,
                request_time_s=t0,
                duration_s=duration,
            )
            row["availability"] = (
                max(0.0, 1.0 - result.interruption_s / duration)
                if result.setup_success else 0.0
            )
            rows.append(row)
        if request_index % 25 == 0 or request_index == total_requests:
            elapsed = time.monotonic() - started
            rate = request_index / max(elapsed, 1.0e-9)
            remaining = (total_requests - request_index) / max(rate, 1.0e-9)
            print(
                f"progress {request_index}/{total_requests} requests "
                f"({100*request_index/total_requests:.1f}%), ETA {remaining/60:.1f} min",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "sessions.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["pair"], row["duration_s"])].append(row)
    summary = []
    for (method, pair, duration), items in sorted(groups.items()):
        setup_delays = [x["setup_delay_s"] for x in items if not math.isnan(x["setup_delay_s"])]
        setup_successes = sum(bool(x["setup_success"]) for x in items)
        completions = sum(bool(x["completion"]) for x in items)
        setup_lo, setup_hi = wilson(setup_successes, len(items))
        comp_lo, comp_hi = wilson(completions, len(items))
        established_items = [x for x in items if x["setup_success"]]
        summary.append(
            {
                "method": method,
                "pair": pair,
                "duration_s": duration,
                "n": len(items),
                "setup_success_rate": fmean(x["setup_success"] for x in items),
                "setup_success_ci95_low": setup_lo,
                "setup_success_ci95_high": setup_hi,
                "completion_rate": fmean(x["completion"] for x in items),
                "completion_ci95_low": comp_lo,
                "completion_ci95_high": comp_hi,
                "mean_setup_delay_s": fmean(setup_delays) if setup_delays else math.nan,
                "mean_interruption_s": fmean(x["interruption_s"] for x in items),
                "mean_availability": fmean(x["availability"] for x in items),
                "mean_reconfigurations": fmean(x["reconfigurations"] for x in items),
                "mean_dynamic_setups": fmean(x["dynamic_setups"] for x in items),
                "mean_total_pat_time_s": fmean(x["total_pat_time_s"] for x in items),
                "mean_initial_hops": (
                    fmean(x["initial_hops"] for x in established_items)
                    if established_items else math.nan
                ),
                "mean_initial_path_km": (
                    fmean(x["initial_path_km"] for x in established_items)
                    if established_items else math.nan
                ),
                "mean_initial_bottleneck_s": (
                    fmean(x["initial_bottleneck_s"] for x in established_items)
                    if established_items else math.nan
                ),
                "mean_route_hops": (
                    fmean(x["mean_hops"] for x in established_items)
                    if established_items else math.nan
                ),
                "mean_max_hops": (
                    fmean(x["max_hops"] for x in established_items)
                    if established_items else math.nan
                ),
                "mean_route_path_km": (
                    fmean(x["mean_path_km"] for x in established_items)
                    if established_items else math.nan
                ),
                "mean_max_path_km": (
                    fmean(x["max_path_km"] for x in established_items)
                    if established_items else math.nan
                ),
                "mean_path_stretch_excess": (
                    fmean(x["mean_path_stretch_excess"] for x in established_items)
                    if established_items else math.nan
                ),
                "mean_planned_bottleneck_s": (
                    fmean(x["mean_planned_bottleneck_s"] for x in established_items)
                    if established_items else math.nan
                ),
                "mean_min_planned_bottleneck_s": (
                    fmean(x["min_planned_bottleneck_s"] for x in established_items)
                    if established_items else math.nan
                ),
            }
        )
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    counts_per_pair_duration = defaultdict(int)
    for spec in request_specs:
        counts_per_pair_duration[(str(spec["pair"]), float(spec["duration_s"]))] += 1
    unique_counts = set(counts_per_pair_duration.values())
    sessions_per_pair_duration = (
        next(iter(unique_counts)) if len(unique_counts) == 1 else None
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trace": str(args.trace.resolve()),
        "methods": args.methods,
        "pairs": GROUND_PAIRS,
        "durations_s": sorted({float(row["duration_s"]) for row in request_specs}),
        "request_count": len(request_specs),
        "sessions_per_pair_duration": sessions_per_pair_duration,
        "request_time_rule": (
            f"exact rows from {args.request_file.resolve()}"
            if args.request_file is not None
            else "deterministic midpoints of equal bins over one orbit"
        ),
        "orbit_period_s": constellation.period_s,
        "setup_delay_s": args.setup_delay,
        "gsl_range_km": args.gsl_range,
        "shared_reward": asdict(reward_config),
        "max_relay_actions": args.max_relay_actions,
        "parameter_manifest": {
            key: {
                "value": record.value,
                "unit": record.unit,
                "category": record.category,
                "reference": record.reference,
                "rationale": record.rationale,
            }
            for key, record in PARAMETER_MANIFEST.items()
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"wrote {len(rows)} paired method-runs to {raw_path}")
    print(f"wrote {len(summary)} aggregate rows to {summary_path}")


if __name__ == "__main__":
    main()
