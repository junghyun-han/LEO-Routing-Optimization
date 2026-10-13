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

from leo_session.topology import canonical_edge  # noqa: E402
from llm_routing.compliance import in_region  # noqa: E402
from llm_routing.tools import Toolbox  # noqa: E402
from multisession.agent.scenarios import KO_REGION  # noqa: E402
from multisession.conditions import REGIONS, satellite_latlon  # noqa: E402

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


HOTSPOT_TEXT = [
    "{r} 상공 구간에 트래픽이 몰려 혼잡해. 가능하면 피해서 경로를 잡아 줘.",
    "지금 {r} 쪽이 핫스팟이야. 꼭 필요한 경우가 아니면 지나가지 마.",
    "{r} 구간 부하가 높아. 되도록 덜 거치게 해 줘.",
]


def region_note(region: str) -> str:
    lat_lo, lat_hi, lon_lo, lon_hi = REGIONS[region]
    return (f"Region box used by operations ({region}, '{KO_REGION[region]}'): sub-satellite latitude {lat_lo:g} to "
            f"{lat_hi:g}, longitude {lon_lo:g} to {lon_hi:g}.")


def hotspot(split: str) -> dict:
    """The region the request-time Dijkstra route crosses most (soft condition: avoid where possible).

    Only relay (non-endpoint) satellites count, since the ingress and egress are
    fixed for every method.  Whether the region can be avoided for the whole
    session is recorded, not required.
    """
    all_rows, subset = subset_requests(split)
    con, env = build_environment(all_rows)
    out = {}
    for i, spec in enumerate(subset):
        t0, x = float(spec["request_time_s"]), float(spec["duration_s"])
        topo = env._initial_topology(t0)
        route = env.planner.plan("raw", spec["source"], spec["destination"], t0, x, topo)
        if route is None:
            continue
        ll = satellite_latlon(con, t0)
        relays = list(route.satellites[1:-1])
        ranked = [r for r in sorted(REGIONS, key=lambda r: -len(in_region(ll, relays, r))) if in_region(ll, relays, r)]
        if not ranked:
            continue
        r = ranked[0]
        graph = env.planner._graph(t0, topo)
        box = Toolbox(env.planner, graph, spec["source"], spec["destination"], t0, x, {canonical_edge(*e) for e in topo.dynamic})
        out[spec["request_id"]] = {"hotspot_region": r, "note": region_note(r),
                                   "avoidable_whole_session": bool(box.find_route([], None, "fewest_hops", [r]).get("found")),
                                   "text": HOTSPOT_TEXT[i % len(HOTSPOT_TEXT)].format(r=KO_REGION[r])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["maintenance", "hotspot"])
    ap.add_argument("--split", choices=["eval", "dev"], default="eval")
    args = ap.parse_args()
    data = {"maintenance": maintenance, "hotspot": hotspot}[args.kind](args.split)
    path = ROOT / "results_llm/scenarios" / f"{args.kind}{'' if args.split == 'eval' else '_dev'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"{len(data)} scenarios -> {path}")


if __name__ == "__main__":
    main()
