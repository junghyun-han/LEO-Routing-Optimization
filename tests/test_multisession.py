from pathlib import Path

import pytest

from multisession import (
    Instance, Session, Snapshot, Solution, cut_upper_bound, evaluate, fixed_topology,
    greedy_reconfiguration, verify,
)

ROOT = Path(__file__).resolve().parents[1]


def toy_snapshot(gsl_t=900.0):
    # Two planes: 0-1-2-3 and 4-5-6-7. Only (0,4) is ACTIVE before the batch.
    return Snapshot(
        time_s=0.0,
        persistent=((0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)),
        dynamic_remaining={(0, 4): 1000.0, (3, 7): 1000.0, (1, 5): 1000.0, (3, 5): 1000.0, (2, 6): 40.0},
        initial_dynamic=((0, 4),),
        access={"S": ((0, 900.0),), "T": ((7, gsl_t),)},
    )


def toy_instance(n_sessions, capacity=1, gsl_t=900.0, duration=30.0):
    sessions = tuple(Session(i, "S", "T", duration) for i in range(n_sessions))
    return Instance(toy_snapshot(gsl_t), sessions, antennas=1, capacity=capacity)


def test_fixed_topology_uses_initial_links_only():
    inst = toy_instance(2)
    sol = fixed_topology(inst)
    assert verify(inst, sol) == []
    assert sol.routes == {0: [0, 4, 5, 6, 7]}  # second session blocked by capacity
    assert evaluate(inst, sol)["new_links"] == 0


def test_greedy_switches_a_free_terminal_and_pays_setup():
    inst = toy_instance(2)
    sol = greedy_reconfiguration(inst)
    assert verify(inst, sol) == []
    assert sol.routes[1] == [0, 1, 2, 3, 7]
    metrics = evaluate(inst, sol)
    assert metrics["admitted"] == 2 and metrics["new_links"] == 1 and metrics["delayed_sessions"] == 1


def test_new_link_needs_setup_plus_duration_on_ground_link():
    # Egress GSL lasts 40 s: enough for X=30 without setup, not with the 20 s setup.
    inst = toy_instance(2, gsl_t=40.0)
    sol = greedy_reconfiguration(inst)
    assert verify(inst, sol) == []
    assert list(sol.routes) == [0]


def test_verifier_catches_each_violation():
    inst = toy_instance(2, capacity=1)
    assert any("capacity" in p for p in verify(inst, Solution({0: [0, 4, 5, 6, 7], 1: [0, 4, 5, 6, 7]}, {(0, 4)})))
    assert any("not ACTIVE" in p for p in verify(inst, Solution({0: [0, 1, 2, 3, 7]}, {(0, 4)})))
    assert any("two dynamic links" in p for p in verify(inst, Solution({}, {(0, 4), (1, 5), (3, 5)})))
    assert any("revisits" in p for p in verify(inst, Solution({0: [0, 1, 0, 4, 5, 6, 7]}, {(0, 4)})))
    late = toy_instance(1, gsl_t=40.0)
    assert any("ends before" in p for p in verify(late, Solution({0: [0, 1, 2, 3, 7]}, {(0, 4), (3, 7)})))
    short = toy_instance(1, duration=30.0)
    assert any("ends before" in p for p in verify(short, Solution({0: [0, 1, 2, 6, 7]}, {(0, 4), (2, 6)})))


def test_cut_bound_is_valid_on_toy():
    inst = toy_instance(5)
    assert cut_upper_bound(inst) == 3  # one antenna satellite, three ISLs, capacity 1
    assert evaluate(inst, greedy_reconfiguration(inst))["admitted"] <= cut_upper_bound(inst)


@pytest.fixture(scope="module")
def v3_snapshot():
    from leo_session.ground import GROUND_STATIONS
    from multisession.build import build_snapshot, decision_times, ground_horizon, load_environment
    t0, _ = decision_times(ROOT / "results_paper/paper_artifacts/evaluation_requests.csv")[0]
    con, isl, gtr = load_environment(ROOT / "cache/contact_trace_1584_24_11.npz", ground_horizon([t0]))
    return build_snapshot(con, isl, gtr, t0, GROUND_STATIONS), gtr, t0


def test_one_antenna_is_the_v3_access_rule(v3_snapshot):
    snap, gtr, t0 = v3_snapshot
    for station in ("new_york", "london", "istanbul", "hanoi"):
        assert snap.access[station][0][0] == gtr.selected_access(station, t0)


@pytest.mark.parametrize("antennas,capacity,per_dest,duration", [(1, 1, 4, 60.0), (2, 1, 8, 20.0), (4, 2, 16, 180.0)])
def test_baselines_feasible_on_v3(v3_snapshot, antennas, capacity, per_dest, duration):
    from multisession.build import make_instance
    inst = make_instance(v3_snapshot[0], per_dest, duration, antennas, capacity)
    bound = cut_upper_bound(inst)
    for method in (fixed_topology, greedy_reconfiguration):
        sol = method(inst)
        assert verify(inst, sol) == []
        assert evaluate(inst, sol)["admitted"] <= bound


def test_exact_model_on_toy_is_optimal_and_verified():
    from multisession.exact import solve_exact
    inst = toy_instance(3, capacity=1)  # node 0 has two usable ISLs, so at most 2 sessions
    result = solve_exact(inst, time_limit_s=10, workers=1)
    assert verify(inst, result.solution) == []
    metrics = evaluate(inst, result.solution)
    assert result.optimal and metrics["admitted"] == 2 and metrics["new_links"] == 1
    assert result.admitted_bound == 2
