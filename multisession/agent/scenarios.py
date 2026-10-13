"""Synthetic operator scenarios: Korean text plus the ground-truth conditions it states.

Six condition categories (docs/CONDITIONS.md) are spread evenly: each appears in
the same number of scenarios, two to four per scenario.  Satellites and regions
are chosen so that the condition matters at that decision time (a failed
satellite lies on a route the unconstrained solver would use, a hotspot or an
avoided region is crossed by such routes, a required region is not).  Text and
ground truth are generated together, so they cannot disagree.  These are
author-made scenarios, not operator logs.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List

from ..conditions import OBJECTIVES, REGIONS, Conditions, prepare, region_satellites
from ..fast import solve_fast

KO_REGION = {"north_america_east": "북미 동부", "north_atlantic": "북대서양", "europe": "유럽", "mediterranean": "지중해",
             "africa": "아프리카", "middle_east": "중동", "south_asia": "남아시아", "indian_ocean": "인도양",
             "southeast_asia": "동남아시아", "east_asia": "동아시아"}
KO_PLACE = {"new_york": "뉴욕", "london": "런던", "istanbul": "이스탄불", "hanoi": "하노이"}
CATEGORIES = ("failure", "hotspot", "region", "ground", "reconfig", "traffic")
# Regions a New York route to each destination could reasonably be asked to pass (gateway or
# jurisdiction on the way), so "via" never asks for an absurd detour.
VIA_PLAUSIBLE = {"london": ["north_atlantic", "europe", "mediterranean"],
                 "istanbul": ["north_atlantic", "europe", "mediterranean", "africa", "middle_east"],
                 "hanoi": ["europe", "mediterranean", "middle_east", "south_asia", "indian_ocean", "southeast_asia", "east_asia"]}

T = {
    "failure": ["위성 {ids}가 고장 났어.", "{ids}번 위성이 응답이 없어서 쓸 수 없어.", "지금 {ids} 위성은 장애 상태야."],
    "hotspot": ["{r} 상공은 트래픽이 몰려서 혼잡하니까 가능하면 피해 줘.", "{r} 쪽이 지금 핫스팟이야. 꼭 필요할 때만 지나가.",
                "{r} 구간 부하가 높아. 되도록 덜 거치게 해 줘."],
    "avoid": ["규제 때문에 {r} 상공으로는 절대 보내면 안 돼.", "{r} 위로는 어떤 세션도 지나가지 않게 해.",
              "보안 정책상 {r} 상공 통과는 금지야."],
    "via": ["{d}행 세션은 {r} 상공을 반드시 거쳐서 보내야 해.", "{d}로 가는 트래픽은 {r} 구간을 경유하도록 해 줘.",
            "{d} 세션은 {r} 위를 지나가는 경로만 허용돼."],
    "ground": ["{s} 지상국 안테나가 강우 때문에 {n}개만 쓸 수 있어.", "{s} 게이트웨이는 지금 안테나 {n}개만 가동 중이야.",
               "{s} 쪽 안테나 일부가 다운돼서 {n}개만 남았어."],
    "limit": ["링크 재설정은 {m}번까지만 허용돼.", "레이저 터미널 재지향 작업은 최대 {m}건으로 제한해 줘.",
              "새 링크는 {m}개 넘게 만들지 마."],
    "limit0": ["지금은 링크 재설정을 하지 마.", "레이저 터미널은 하나도 다시 맞추지 말고 지금 링크만 써."],
    "objective": ["세션을 조금 덜 받더라도 링크 재설정은 최소로 해.", "재설정을 줄이는 게 세션을 더 받는 것보다 중요해."],
    "priority": ["{d} 세션을 최우선으로 받아 줘.", "{d}행을 제일 먼저 처리해 줘."],
    "priority2": ["{d1} 세션을 먼저, 그다음 {d2} 순서로 우선해 줘."],
    "minimum": ["{d}에는 최소 {n}개 세션은 꼭 보장해 줘.", "{d}행은 적어도 {n}개는 받아야 해."],
}
OPENERS = ["지금 뉴욕에서 런던, 이스탄불, 하노이로 가는 세션 요청이 들어와 있어.",
           "뉴욕발 세션들(런던, 이스탄불, 하노이행)을 배정해야 해.", "새 세션 요청 묶음이 들어왔어."]
CLOSERS = ["이 조건으로 라우팅해 줘.", "조건 맞춰서 경로 잡아 줘.", "반영해서 배정해 줘."]


def _join_ids(ids: List[int]) -> str:
    return ", ".join(str(i) for i in ids)


def _route_regions(latlon, routes) -> Counter:
    count: Counter = Counter()
    for path in routes:
        for r in REGIONS:
            sats = region_satellites(latlon, r)
            count[r] += sum(1 for v in path if v in sats)
    return count


def make_scenarios(instances: List, latlons: List, n: int = 30, seed: int = 0) -> List[Dict]:
    """``instances[i]``/``latlons[i]`` give the base problem of scenario i (cycled)."""
    rng = random.Random(seed)
    slots = [c for c in CATEGORIES for _ in range(n // 2)]  # each category in n/2 scenarios
    rng.shuffle(slots)
    plans: List[List[str]] = [[] for _ in range(n)]
    sizes = [rng.choice([2, 3, 4]) for _ in range(n)]
    for c in slots:  # fill the scenario that still needs the most categories and lacks this one
        order = sorted(range(n), key=lambda i: (len(plans[i]) - sizes[i], rng.random()))
        target = next((i for i in order if c not in plans[i]), None)
        if target is not None:
            plans[target].append(c)
    scenarios = []
    for i, cats in enumerate(plans):
        inst, ll = instances[i % len(instances)], latlons[i % len(latlons)]
        base = solve_fast(prepare(inst, Conditions(), ll))
        routes = list(base.routes.values())
        antenna = {s for st in KO_PLACE for s, _ in inst.antenna_sats(st)}
        crossed = _route_regions(ll, routes)
        gt: Dict = {"failed_satellites": [], "avoid_regions": [], "hotspot_regions": [], "via": {}, "antennas": {},
                    "max_new_links": None, "min_admitted": {}, "priority": [], "objective": list(OBJECTIVES)}
        lines: List[str] = []
        used_regions = set()
        applied: List[str] = []
        for c in rng.sample(cats, len(cats)):
            before = len(lines)
            if c == "failure":
                interior = sorted({v for p in routes for v in p[1:-1]} - antenna)
                ids = sorted(rng.sample(interior, min(len(interior), rng.choice([1, 2, 3]))))
                gt["failed_satellites"] = ids
                lines.append(rng.choice(T["failure"]).format(ids=_join_ids(ids)))
            elif c == "hotspot":
                cands = [r for r, k in crossed.most_common() if k > 0 and r not in used_regions]
                if not cands:
                    continue
                r = cands[0]
                used_regions.add(r)
                gt["hotspot_regions"] = [r]
                lines.append(rng.choice(T["hotspot"]).format(r=KO_REGION[r]))
            elif c == "region":
                avoid = [r for r, k in crossed.most_common() if k > 0 and r not in used_regions
                         and not (region_satellites(ll, r) & antenna)]
                if avoid and rng.random() < 0.5:
                    r = avoid[0]
                    used_regions.add(r)
                    gt["avoid_regions"] = [r]
                    lines.append(rng.choice(T["avoid"]).format(r=KO_REGION[r]))
                else:
                    d = rng.choice(["london", "istanbul", "hanoi"])
                    d_routes = [p for k, p in base.routes.items() if inst.sessions[k].destination == d]
                    on = _route_regions(ll, d_routes)
                    plausible = [r for r in VIA_PLAUSIBLE[d] if r not in used_regions and region_satellites(ll, r)]
                    cands = [r for r in plausible if on[r] == 0] or plausible
                    if not cands:
                        continue
                    r = rng.choice(cands)
                    used_regions.add(r)
                    gt["via"] = {d: r}
                    lines.append(rng.choice(T["via"]).format(d=KO_PLACE[d], r=KO_REGION[r]))
            elif c == "ground":
                st = rng.choice(list(KO_PLACE))
                k = max(1, inst.antennas - rng.choice([1, 2])) if inst.antennas > 1 else 0
                gt["antennas"] = {st: k}
                lines.append(rng.choice(T["ground"]).format(s=KO_PLACE[st], n=k))
            elif c == "reconfig":
                if rng.random() < 0.3:
                    gt["objective"] = ["new_links", "admitted", "hotspot", "hops"]
                    lines.append(rng.choice(T["objective"]))
                else:
                    m = rng.choice([0, 1, 2, 3])
                    gt["max_new_links"] = m
                    lines.append(rng.choice(T["limit0"] if m == 0 else T["limit"]).format(m=m))
            elif c == "traffic":
                if rng.random() < 0.5:
                    ds = rng.sample(["london", "istanbul", "hanoi"], rng.choice([1, 2]))
                    gt["priority"] = ds
                    tpl = rng.choice(T["priority"] if len(ds) == 1 else T["priority2"])
                    lines.append(tpl.format(d=KO_PLACE[ds[0]], d1=KO_PLACE[ds[0]], d2=KO_PLACE[ds[-1]]))
                else:
                    d = rng.choice(["london", "istanbul", "hanoi"])
                    k = rng.choice([2, 3, 4])
                    gt["min_admitted"] = {d: k}
                    lines.append(rng.choice(T["minimum"]).format(d=KO_PLACE[d], n=k))
            if len(lines) > before:
                applied.append(c)
        text = " ".join([rng.choice(OPENERS)] + lines + [rng.choice(CLOSERS)])
        truth = Conditions.from_dict(gt)  # validates
        scenarios.append({"id": i, "t0": inst.snapshot.time_s, "antennas": inst.antennas, "capacity": inst.capacity,
                          "per_destination": len(inst.sessions) // 3, "duration_s": inst.sessions[0].duration_s,
                          "categories": sorted(applied), "text": text, "truth": truth.to_dict()})
    return scenarios
