"""Run the LLM route editor through the WCNC session protocol on the 180-request subset.

The subset is 3 ground pairs x 5 holding times x 12 arrival times taken from the
frozen 3,600-request test set (arrival-time indices 10, 30, ..., 230).  The
environment is built exactly as for the frozen baseline run (same ground-contact
horizon), so rows are comparable request by request with PPO-GCN and the baselines.

    python scripts/run_llm_routing.py --mock identity --out results_llm/identity
    python scripts/run_llm_routing.py --provider openai --model gpt-4o-mini --out results_llm/gpt-4o-mini_none
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from leo_session import ContactTrace, GroundContactGenerator, SessionEnvironment, SessionRequest, WalkerDeltaConstellation  # noqa: E402
from leo_session.config import DEFAULT_CONFIG  # noqa: E402
from leo_session.reward import RewardConfig  # noqa: E402
from llm_routing.planner import LLMPlanner  # noqa: E402
from multisession.conditions import satellite_latlon  # noqa: E402

REQUESTS = ROOT / "results_paper/paper_artifacts/evaluation_requests.csv"


def subset_requests():
    rows = list(csv.DictReader(open(REQUESTS, newline="")))
    times = np.sort(np.unique([float(r["request_time_s"]) for r in rows]))
    picked = set(times[10::20].round(6))
    return rows, [r for r in rows if round(float(r["request_time_s"]), 6) in picked]


def build_environment(all_rows):
    con = WalkerDeltaConstellation(DEFAULT_CONFIG)
    isl = ContactTrace.load(ROOT / "cache/contact_trace_1584_24_11.npz")
    horizon = max(float(r["request_time_s"]) + float(r["duration_s"]) for r in all_rows) + 2.0 * DEFAULT_CONFIG.dynamic_setup_s
    gtr = GroundContactGenerator(con, config=DEFAULT_CONFIG).generate(horizon)
    return con, SessionEnvironment(con, isl, gtr, DEFAULT_CONFIG, RewardConfig(), 128)


class IdentityModel:
    """Returns the Dijkstra route unchanged (protocol check)."""
    name = "identity"

    def chat(self, messages, tools=None):
        route = re.search(r"Dijkstra route: (\[[^\]]*\])", messages[-1]["content"]).group(1)
        return {"role": "assistant", "content": json.dumps({"route": json.loads(route), "reason": "keep"})}, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", choices=["identity"])
    ap.add_argument("--provider", choices=["openai", "ollama"])
    ap.add_argument("--model")
    ap.add_argument("--operator-text", default="")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.mock:
        model = IdentityModel()
    else:
        from multisession.agent.llm import ChatModel
        model = ChatModel(args.provider, args.model, max_tokens=1024)
    all_rows, subset = subset_requests()
    con, env = build_environment(all_rows)
    planner = LLMPlanner(env.planner, model, lambda t: satellite_latlon(con, t), args.operator_text, args.hops)
    env.planner = planner
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in subset[: args.limit]:
        planner.request_id = int(spec["request_id"])
        request = SessionRequest(spec["source"], spec["destination"], float(spec["request_time_s"]), float(spec["duration_s"]))
        result = env.run("llm", request)
        row = asdict(result)
        row.update(request_id=int(spec["request_id"]), pair=spec["pair"], source=spec["source"],
                   destination=spec["destination"], request_time_s=float(spec["request_time_s"]),
                   duration_s=float(spec["duration_s"]))
        row["availability"] = max(0.0, 1.0 - result.interruption_s / row["duration_s"]) if result.setup_success else 0.0
        rows.append(row)
    with open(args.out / "sessions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out / "decisions.jsonl", "w") as handle:
        for d in planner.decisions:
            handle.write(json.dumps(d) + "\n")
    valid = sum(d["valid"] for d in planner.decisions)
    print(f"{len(rows)} requests, {len(planner.decisions)} decisions, valid edits {valid}, "
          f"completion {np.mean([r['completion'] for r in rows]):.4f} -> {args.out}")


if __name__ == "__main__":
    main()
