"""Run one heuristic (Python source defining ``solve(instance, kit)``) on a set of instances.

Executed in a child process by the agent so a broken or slow candidate cannot
take the experiment down.  Every returned solution is checked by
``model.verify``; an infeasible solution counts as zero admitted sessions and
is reported as an error.

    python -m multisession.llm.candidate --code h.py --keys keys.json --out result.json
"""
from __future__ import annotations

import argparse
import json
import signal
import time
import traceback
from pathlib import Path
from typing import Dict, List

from leo_session.ground import GROUND_STATIONS

from .. import kit
from ..build import build_snapshot, ground_horizon, load_environment, make_instance
from ..model import evaluate, verify

ROOT = Path(__file__).resolve().parents[2]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # pragma: no cover - signal handler
    raise _Timeout()


def load_instances(keys: List[Dict]) -> List:
    env = load_environment(ROOT / "cache/contact_trace_1584_24_11.npz", ground_horizon(k["t0"] for k in keys))
    snaps = {}
    out = []
    for k in keys:
        if k["t0"] not in snaps:
            snaps[k["t0"]] = build_snapshot(*env, k["t0"], GROUND_STATIONS)
        out.append(make_instance(snaps[k["t0"]], k["per_destination"], k["duration_s"], k["antennas"], k["capacity"]))
    return out


def run(code: str, keys: List[Dict], per_instance_s: float) -> Dict:
    namespace: Dict = {}
    try:
        exec(compile(code, "<candidate>", "exec"), namespace)  # noqa: S102 - trusted local experiment
        solve = namespace["solve"]
    except Exception:
        return {"error": "code does not load:\n" + traceback.format_exc(limit=3), "rows": []}
    instances = load_instances(keys)
    signal.signal(signal.SIGALRM, _alarm)
    rows, errors = [], []
    for key, inst in zip(keys, instances):
        row = dict(key)
        start = time.perf_counter()
        try:
            signal.setitimer(signal.ITIMER_REAL, per_instance_s)
            solution = solve(inst, kit)
            signal.setitimer(signal.ITIMER_REAL, 0)
            solution.runtime_s = time.perf_counter() - start
            problems = verify(inst, solution)
            if problems:
                errors.append(f"infeasible solution: {problems[0]}")
                row.update(admitted=0, error="infeasible")
            else:
                row.update(evaluate(inst, solution))
                row["routes"] = {str(i): p for i, p in solution.routes.items()}
                row["final_dynamic"] = sorted(map(list, solution.final_dynamic))
        except _Timeout:
            errors.append(f"time limit {per_instance_s:.0f} s exceeded")
            row.update(admitted=0, error="timeout")
        except Exception:
            signal.setitimer(signal.ITIMER_REAL, 0)
            errors.append("runtime error:\n" + traceback.format_exc(limit=3))
            row.update(admitted=0, error="exception")
        rows.append(row)
    return {"error": None, "errors": errors[:3], "n_errors": len(errors), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--keys", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-instance-s", type=float, default=2.0)
    args = ap.parse_args()
    result = run(args.code.read_text(), json.loads(args.keys.read_text()), args.per_instance_s)
    args.out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
