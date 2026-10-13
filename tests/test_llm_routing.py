import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_llm_routing import IdentityModel, build_environment, subset_requests  # noqa: E402

from leo_session import SessionRequest  # noqa: E402
from leo_session.topology import canonical_edge  # noqa: E402
from llm_routing.planner import LLMPlanner  # noqa: E402
from llm_routing.route_edit import to_route  # noqa: E402
from llm_routing.regions import satellite_latlon  # noqa: E402


def test_subset_is_3_pairs_5_durations_12_times():
    _, subset = subset_requests()
    assert len(subset) == 180
    assert len({float(r["request_time_s"]) for r in subset}) == 12


def test_identity_model_reproduces_frozen_dijkstra_rows():
    all_rows, subset = subset_requests()
    con, env = build_environment(all_rows)
    env.planner = LLMPlanner(env.planner, IdentityModel(), lambda t: satellite_latlon(con, t))
    sample = subset[::15]  # 12 requests across pairs and durations
    ref = pd.read_csv(ROOT / "results_paper/main_240_metrics/sessions.csv")
    ref = ref[ref.method == "raw"].set_index("request_id")
    for spec in sample:
        res = asdict(env.run("llm", SessionRequest(spec["source"], spec["destination"],
                                                    float(spec["request_time_s"]), float(spec["duration_s"]))))
        row = ref.loc[int(spec["request_id"])]
        for key in ("setup_success", "completion", "interruption_s", "reconfigurations", "dynamic_setups", "mean_path_km"):
            assert np.isclose(float(res[key]), float(row[key]), equal_nan=True), (spec["request_id"], key)


def test_structural_validation_rejects_broken_routes():
    all_rows, subset = subset_requests()
    con, env = build_environment(all_rows)
    spec = subset[0]
    t0 = float(spec["request_time_s"])
    topo = env._initial_topology(t0)
    route = env.planner.plan("raw", spec["source"], spec["destination"], t0, float(spec["duration_s"]), topo)
    graph = env.planner._graph(t0, topo)
    active = {canonical_edge(*e) for e in topo.dynamic}
    sats = list(route.satellites)
    ok, err = to_route(env.planner, graph, sats, spec["source"], spec["destination"], t0, active)
    assert ok is not None and err == ""
    assert to_route(env.planner, graph, sats[1:], spec["source"], spec["destination"], t0, active)[0] is None  # wrong ingress
    assert to_route(env.planner, graph, sats[:2] + sats[3:], spec["source"], spec["destination"], t0, active)[0] is None  # gap
    assert to_route(env.planner, graph, sats[:2] + sats[:2] + sats[2:], spec["source"], spec["destination"], t0, active)[0] is None  # repeat


def _toy_data():
    # 0 -P- 1 -P- 3 is short but 1 is over a forbidden region; 0 -P- 2 -N- 3 avoids it with a new link.
    return {"ingress": 0, "egress": 3, "required_s": 100.0, "setup_s": 20.0, "ground_link_km": 1000.0,
            "speed_of_light_km_s": 299792.458, "n_new_links_so_far": 0,
            "links": [{"u": 0, "v": 1, "kind": "P", "remaining_s": None, "km": 900.0},
                      {"u": 1, "v": 3, "kind": "P", "remaining_s": None, "km": 900.0},
                      {"u": 0, "v": 2, "kind": "P", "remaining_s": None, "km": 1500.0},
                      {"u": 2, "v": 3, "kind": "N", "remaining_s": 110.0, "km": 1500.0}],
            "region_satellites": {"r": [1]}, "region_names": {"r": "r"}}


def test_reference_formulation_conditions():
    from llm_routing.formulation import conditions_met, reference_route
    data = _toy_data()
    assert reference_route(data)["route"] == [0, 1, 3]
    # avoiding the region needs 2-3, whose remaining contact (110 s) is shorter than 100 s + 20 s setup
    assert reference_route(data, avoid_region="r")["route"] is None
    data["links"][3]["remaining_s"] = 130.0
    r = reference_route(data, avoid_region="r")
    assert r["route"] == [0, 2, 3] and r["new_links"] == 1
    assert reference_route(data, avoid_region="r", max_new_links=0)["route"] is None
    assert reference_route(data, avoid_region="r", max_delay_ms=12.0)["route"] is None  # (1000 + 3000) km / c = 13.3 ms
    assert conditions_met(data, [0, 1, 3], {"avoid_region": "r", "max_delay_ms": 12.0}) == \
        {"valid_links": True, "lifetime": True, "avoid": False, "delay": True}


def test_sandbox_rejects_unsafe_code_and_runs_safe_code():
    from llm_routing.sandbox import check_code, run_code
    assert "not allowed" in check_code("import os\ndef solve(data): return None")
    assert "not allowed" in check_code("def solve(data): return open('x')")
    assert "not allowed" in check_code("def solve(data): return ().__class__")
    assert "solve(data)" in check_code("def other(data): return None")
    route, err, _ = run_code("def solve(data):\n    return [data['ingress'], data['egress']]", {"ingress": 4, "egress": 7})
    assert route == [4, 7] and err == ""
    route, err, _ = run_code("def solve(data):\n    raise ValueError('boom')", {})
    assert route is None and "boom" in err
