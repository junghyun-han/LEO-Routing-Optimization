"""Constraint scenarios for the 180 evaluation requests (and the disjoint dev requests).

maintenance: an unannounced (to the routing methods) PAT maintenance outage.
For each request, the satellite in the middle of the request-time Dijkstra
route (never the ingress or egress) goes down at t0 + X/2, so the outage falls
inside the session.  The operator text announces it; only the LLM reads text.

    python scripts/make_scenarios.py maintenance --split eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_llm_routing import build_environment, subset_requests  # noqa: E402

MAINTENANCE_TEXT = [
    "위성 {m}는 시각 {t:.0f}초부터 PAT 장비 정비로 운용이 중단돼.",
    "{t:.0f}초에 위성 {m}의 광 터미널 정비가 시작돼서 그때부터 이 위성은 쓸 수 없어.",
    "정비 공지: 위성 {m}, 시각 {t:.0f}초부터 서비스 중단.",
]


def maintenance(split: str) -> dict:
    all_rows, subset = subset_requests(split)
    _, env = build_environment(all_rows)
    out = {}
    for i, spec in enumerate(subset):
        t0, x = float(spec["request_time_s"]), float(spec["duration_s"])
        topo = env._initial_topology(t0)
        route = env.planner.plan("raw", spec["source"], spec["destination"], t0, x, topo)
        if route is None or len(route.satellites) < 3:
            continue
        interior = route.satellites[1:-1]
        m = int(interior[len(interior) // 2])
        t = t0 + x / 2.0
        out[spec["request_id"]] = {"outages": {str(m): t}, "satellite": m, "time_s": t,
                                   "text": MAINTENANCE_TEXT[i % len(MAINTENANCE_TEXT)].format(m=m, t=t)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["maintenance"])
    ap.add_argument("--split", choices=["eval", "dev"], default="eval")
    args = ap.parse_args()
    data = {"maintenance": maintenance}[args.kind](args.split)
    path = ROOT / "results_llm/scenarios" / f"{args.kind}{'' if args.split == 'eval' else '_dev'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"{len(data)} scenarios -> {path}")


if __name__ == "__main__":
    main()
