"""Exact CP-SAT runs on the contested factor levels (docs/PROBLEM.md, section 7).

Writes one row per instance and method, appending as it goes so a long run can
be resumed.  Example:
    .venv/bin/python scripts/run_exact.py --split validation --out results_multisession/exact_validation
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leo_session.ground import GROUND_STATIONS  # noqa: E402
from multisession import cut_upper_bound, evaluate, fixed_topology, greedy_reconfiguration, verify  # noqa: E402
from multisession.build import build_snapshot, decision_times, ground_horizon, load_environment, make_instance  # noqa: E402
from multisession.exact import solve_exact  # noqa: E402

FIELDS = ["t0", "split", "antennas", "capacity", "per_destination", "duration_s", "method", "cut_bound",
          "sessions", "admitted", "admitted_frac", "new_links", "new_links_used", "delayed_sessions",
          "mean_hops", "total_hops", "runtime_s", "status", "admitted_bound", "hops_bound", "optimal"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request-file", type=Path, default=ROOT / "results_paper/paper_artifacts/evaluation_requests.csv")
    ap.add_argument("--trace", type=Path, default=ROOT / "cache/contact_trace_1584_24_11.npz")
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--antennas", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--capacity", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--per-destination", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--durations", type=float, nargs="+", default=[20.0])
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--hops-time-limit", type=float, default=20.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    times = [(t, s) for t, s in decision_times(args.request_file) if s == args.split]
    constellation, isl_trace, ground_trace = load_environment(args.trace, ground_horizon(t for t, _ in times))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "instances.csv"
    done = set()
    if path.exists():
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                done.add((float(row["t0"]), int(row["antennas"]), int(row["capacity"]),
                          int(row["per_destination"]), float(row["duration_s"])))
    new_file = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for t0, split in times:
            snapshot = build_snapshot(constellation, isl_trace, ground_trace, t0, GROUND_STATIONS)
            for a in args.antennas:
                for kappa in args.capacity:
                    for k in args.per_destination:
                        for x in args.durations:
                            if (t0, a, kappa, k, x) in done:
                                continue
                            inst = make_instance(snapshot, k, x, a, kappa)
                            base = {"t0": t0, "split": split, "antennas": a, "capacity": kappa,
                                    "per_destination": k, "duration_s": x, "cut_bound": cut_upper_bound(inst)}
                            greedy = greedy_reconfiguration(inst)
                            results = [("fixed", fixed_topology(inst), {}), ("greedy", greedy, {})]
                            exact = solve_exact(inst, args.time_limit, hint=greedy, hops_time_limit_s=args.hops_time_limit)
                            results.append(("exact", exact.solution, {
                                "status": "/".join(exact.status), "admitted_bound": exact.admitted_bound,
                                "hops_bound": exact.hops_bound, "optimal": exact.optimal}))
                            sol_dir = args.out / "solutions"
                            sol_dir.mkdir(exist_ok=True)
                            name_key = f"t{t0:.3f}_A{a}_k{kappa}_K{k}_X{x:.0f}"
                            (sol_dir / f"{name_key}.json").write_text(json.dumps({
                                "routes": {str(i): p for i, p in exact.solution.routes.items()},
                                "final_dynamic": sorted(map(list, exact.solution.final_dynamic)),
                                "status": exact.status}))
                            for name, sol, extra in results:
                                problems = verify(inst, sol)
                                if problems:
                                    raise AssertionError(f"{name} infeasible at t0={t0}: {problems[:3]}")
                                writer.writerow({**base, "method": name, **evaluate(inst, sol), **extra})
                            handle.flush()
                            print(f"t0={t0:.0f} A={a} k={kappa} K={k} X={x:.0f} exact={exact.status}", flush=True)


if __name__ == "__main__":
    main()
