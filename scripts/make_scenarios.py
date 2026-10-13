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
from llm_routing.regions import KO_REGION  # noqa: E402
from llm_routing.regions import REGIONS, satellite_latlon  # noqa: E402

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
                                   "avoidable_whole_session": bool(box.find_route(objective="fewest_hops", avoid_regions=[r]).get("found")),
                                   "text": HOTSPOT_TEXT[i % len(HOTSPOT_TEXT)].format(r=KO_REGION[r])}
    return out


AVOID_TEXT = [
    "규제 때문에 {r} 상공으로는 절대 보내면 안 돼.",
    "데이터 주권 정책상 {r} 위로는 이 세션 트래픽이 지나가면 안 돼.",
    "보안 정책: {r} 상공 통과 금지.",
]
BUDGET_TEXT = [
    "이 세션에서 PAT 재지향(새 링크 설정)은 총 {k}번까지만 허용돼.",
    "광 터미널 재지향 작업 여유가 없어. 이 세션은 새 링크를 {k}개까지만 켜.",
    "운용 지침: 이 세션의 새 링크 설정은 최대 {k}회.",
]
LATENCY_TEXT = [
    "이 세션은 지연에 민감해. 단방향 지연을 {L:.0f} ms 이하로 유지해 줘.",
    "실시간 서비스라서 종단 간 지연이 {L:.0f} ms를 넘으면 안 돼.",
    "QoS 조건: 단방향 지연 {L:.0f} ms 이하.",
]


def _requests_with_tools(split):
    all_rows, subset = subset_requests(split)
    con, env = build_environment(all_rows)
    for i, spec in enumerate(subset):
        t0, x = float(spec["request_time_s"]), float(spec["duration_s"])
        topo = env._initial_topology(t0)
        route = env.planner.plan("raw", spec["source"], spec["destination"], t0, x, topo)
        if route is None:
            continue
        box = Toolbox(env.planner, env.planner._graph(t0, topo), spec["source"], spec["destination"], t0, x,
                      {canonical_edge(*e) for e in topo.dynamic})
        yield i, spec, con, t0, x, route, box


def avoid(split: str) -> dict:
    """Hard: the region Dijkstra's relays cross most among those avoidable for the whole session."""
    out = {}
    for i, spec, con, t0, x, route, box in _requests_with_tools(split):
        ll = satellite_latlon(con, t0)
        relays = list(route.satellites[1:-1])
        for r in sorted(REGIONS, key=lambda r: -len(in_region(ll, relays, r))):
            if not in_region(ll, relays, r):
                break
            if box.find_route(objective="fewest_hops", avoid_regions=[r]).get("found"):
                out[spec["request_id"]] = {"avoid_region": r, "note": region_note(r),
                                           "text": AVOID_TEXT[i % len(AVOID_TEXT)].format(r=KO_REGION[r])}
                break
    return out


def budget(split: str) -> dict:
    """At most k new links over the session: k = fewest new links of a route lasting X at t0, plus one."""
    out = {}
    for i, spec, con, t0, x, route, box in _requests_with_tools(split):
        f = box.find_route(objective="fewest_new_links", min_remaining_s=x) or {}
        if not f.get("found"):
            f = box.find_route(objective="fewest_new_links")
        k = int(f["new_links"]) + 1
        out[spec["request_id"]] = {"max_new_links": k, "text": BUDGET_TEXT[i % len(BUDGET_TEXT)].format(k=k)}
    return out


def latency(split: str) -> dict:
    """One-way delay bound: 1.15 x the shortest-distance route at t0, rounded up to 5 ms."""
    import math
    out = {}
    for i, spec, con, t0, x, route, box in _requests_with_tools(split):
        f = box.find_route(objective="shortest_distance")
        bound = 5.0 * math.ceil(f["delay_ms_now"] * 1.15 / 5.0)
        out[spec["request_id"]] = {"max_delay_ms": bound, "text": LATENCY_TEXT[i % len(LATENCY_TEXT)].format(L=bound)}
    return out


MULTI_TEXT = {
    "lifetime": "세션이 끝날 때까지 끊기지 않는 링크만 써 줘.",
    "avoid": "{r} 상공은 규제 때문에 통과 금지야.",
    "delay": "단방향 지연은 {L:.0f} ms 이하여야 해.",
    "budget": "PAT 재지향(새 링크 설정)은 이 세션에서 {k}번까지만.",
}


def multi(split: str) -> dict:
    """Several conditions at once, made satisfiable at the request time (docs/LLM_ROUTING.md, section 15).

    Every 4th request of the subset, 40 (eval) or 8 (dev).  Always: links must last the session.  Then 2-3 of {forbidden region (if the
    request has one), delay bound = 1.15 x the shortest-distance route meeting the
    earlier conditions, new-link budget = fewest new links meeting them + 1}.
    """
    import itertools
    import math
    from llm_routing.formulation import decision_data, reference_route
    avoid_of = json.loads((ROOT / "results_llm/scenarios" / f"avoid{'' if split == 'eval' else '_dev'}.json").read_text())
    all_rows, subset = subset_requests(split)
    con, env = build_environment(all_rows)
    out = {}
    combos = itertools.cycle([("avoid", "delay"), ("delay", "budget"), ("avoid", "budget"), ("avoid", "delay", "budget")])
    for i, spec in enumerate(subset[::4][: 40 if split == "eval" else 8]):
        t0, x = float(spec["request_time_s"]), float(spec["duration_s"])
        topo = env._initial_topology(t0)
        graph = env.planner._graph(t0, topo)
        region = avoid_of.get(spec["request_id"], {}).get("avoid_region")
        want = next(combos)
        if "avoid" in want and not region:
            want = ("delay", "budget")
        data = decision_data(env.planner, graph, spec["source"], spec["destination"], t0, x, [region] if region else [])
        cond = {"avoid_region": region if "avoid" in want else None}
        base = reference_route(data, cond["avoid_region"], objective="km")
        if base["route"] is None:
            continue
        if "delay" in want:
            cond["max_delay_ms"] = 5.0 * math.ceil(base["delay_ms"] * 1.15 / 5.0)
        if "budget" in want:
            fewest = reference_route(data, cond["avoid_region"], cond.get("max_delay_ms"), objective="new_links",
                                     hint_route=base["route"])
            cond["max_new_links"] = int(fewest["new_links"] if fewest["route"] else base["new_links"]) + 1
        ref = reference_route(data, cond["avoid_region"], cond.get("max_delay_ms"), cond.get("max_new_links"),
                              hint_route=base["route"], time_limit_s=60.0)
        if ref["route"] is None:
            continue
        parts = [MULTI_TEXT["lifetime"]]
        if cond.get("avoid_region"):
            parts.append(MULTI_TEXT["avoid"].format(r=KO_REGION[cond["avoid_region"]]))
        if "max_delay_ms" in cond:
            parts.append(MULTI_TEXT["delay"].format(L=cond["max_delay_ms"]))
        if "max_new_links" in cond:
            parts.append(MULTI_TEXT["budget"].format(k=cond["max_new_links"]))
        entry = {k: v for k, v in cond.items() if v is not None}
        entry.update(text=" ".join(parts), reference_hops_t0=ref["hops"], conditions=sorted(["lifetime"] + list(want)))
        if cond.get("avoid_region"):
            entry["note"] = region_note(cond["avoid_region"])
        out[spec["request_id"]] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["maintenance", "hotspot", "avoid", "budget", "latency", "multi"])
    ap.add_argument("--split", choices=["eval", "dev"], default="eval")
    args = ap.parse_args()
    data = {"maintenance": maintenance, "hotspot": hotspot, "avoid": avoid, "budget": budget, "latency": latency, "multi": multi}[args.kind](args.split)
    path = ROOT / "results_llm/scenarios" / f"{args.kind}{'' if args.split == 'eval' else '_dev'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"{len(data)} scenarios -> {path}")


if __name__ == "__main__":
    main()
