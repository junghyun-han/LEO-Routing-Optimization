"""Run multi-session methods over the factor grid of docs/PROBLEM.md.

Example (validation split only, used for design decisions):
    .venv/bin/python scripts/run_multisession.py --split validation --out results_multisession/pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from leo_session.ground import GROUND_STATIONS  # noqa: E402
from multisession import cut_upper_bound, evaluate, fixed_topology, greedy_reconfiguration, verify  # noqa: E402
from multisession.build import build_snapshot, decision_times, load_environment, make_instance  # noqa: E402

METHODS = {"fixed": fixed_topology, "greedy": greedy_reconfiguration}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request-file", type=Path, default=ROOT / "results_paper/paper_artifacts/evaluation_requests.csv")
    ap.add_argument("--trace", type=Path, default=ROOT / "cache/contact_trace_1584_24_11.npz")
    ap.add_argument("--split", choices=["validation", "test", "all"], default="validation")
    ap.add_argument("--antennas", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--capacity", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--per-destination", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--durations", type=float, nargs="+", default=[20.0, 60.0, 180.0])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    times = [(t, s) for t, s in decision_times(args.request_file) if args.split in ("all", s)]
    horizon = max(t for t, _ in times) + max(args.durations) + 60.0
    constellation, isl_trace, ground_trace = load_environment(args.trace, horizon)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.perf_counter()
    for t0, split in times:
        snapshot = build_snapshot(constellation, isl_trace, ground_trace, t0, GROUND_STATIONS)
        for a in args.antennas:
            for kappa in args.capacity:
                for k in args.per_destination:
                    for x in args.durations:
                        instance = make_instance(snapshot, k, x, a, kappa)
                        bound = cut_upper_bound(instance)
                        for name in args.methods:
                            solution = METHODS[name](instance)
                            problems = verify(instance, solution)
                            if problems:
                                raise AssertionError(f"{name} infeasible at t0={t0}: {problems[:3]}")
                            rows.append({"t0": t0, "split": split, "antennas": a, "capacity": kappa,
                                         "per_destination": k, "duration_s": x, "method": name,
                                         "cut_bound": bound, **evaluate(instance, solution)})
    frame = pd.DataFrame(rows)
    frame["gap_to_bound"] = frame["cut_bound"] - frame["admitted"]
    frame.to_csv(args.out / "instances.csv", index=False)
    keys = ["method", "antennas", "capacity", "per_destination", "duration_s"]
    summary = frame.groupby(keys).agg(
        instances=("t0", "count"), admitted_frac=("admitted_frac", "mean"),
        at_bound=("gap_to_bound", lambda g: float((g == 0).mean())), mean_gap=("gap_to_bound", "mean"),
        new_links=("new_links", "mean"), mean_hops=("mean_hops", "mean"), runtime_s=("runtime_s", "mean"),
    ).reset_index()
    summary.to_csv(args.out / "summary.csv", index=False)
    meta = {"split": args.split, "decision_times": [t for t, _ in times], "factors": {
        "antennas": args.antennas, "capacity": args.capacity, "per_destination": args.per_destination,
        "durations": args.durations}, "methods": args.methods, "wall_s": time.perf_counter() - start}
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(frame)} rows, {meta['wall_s']:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
