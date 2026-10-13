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
from multisession.conditions import satellite_latlon  # noqa: E402


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
