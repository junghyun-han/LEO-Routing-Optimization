from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from leo_session import (
    GROUND_PAIRS,
    ContactTrace,
    GroundContactGenerator,
    SessionEnvironment,
    SessionRequest,
    RelayCandidate,
    RewardConfig,
    WalkerDeltaConstellation,
    build_initial_topology,
)
from ppo_gcn.observation import (
    ACTION_FEATURE_DIM,
    GRAPH_EDGE_FEATURE_DIM,
    HOP_FRACTION_INDEX,
    NODE_FEATURE_DIM,
    PREVIOUS_NODE_INDEX,
    VISIT_ORDER_INDEX,
    build_policy_observation,
)
from ppo_gcn.model import PPOAgent, PPOConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    constellation = WalkerDeltaConstellation()
    trace = ContactTrace.load(ROOT / "cache/contact_trace_1584_24_11.npz")
    ground = GroundContactGenerator(constellation).generate(1000.0)
    return constellation, trace, ground, SessionEnvironment(constellation, trace, ground)


def test_reference_ground_pairs_include_three_distances():
    assert GROUND_PAIRS == {
        "short": ("new_york", "london"),
        "medium": ("new_york", "istanbul"),
        "long": ("new_york", "hanoi"),
    }


def test_initial_topology_exists_before_request_and_obeys_degree(model):
    constellation, trace, _ground, _environment = model
    topology = build_initial_topology(constellation, trace, 300.0)
    assert topology.dynamic
    assert topology.max_degree(constellation.config.total_satellites) <= 3
    assert all(trace.age_s(u, v, 300.0) >= 20.0 - 1.0e-8 for u, v in topology.dynamic)


def test_ground_ingress_changes_as_constellation_moves(model):
    _constellation, _trace, ground, _environment = model
    early = set(ground.visible("new_york", 100.0))
    later = set(ground.visible("new_york", 500.0))
    assert early
    assert later
    assert early != later


def test_all_methods_receive_same_ground_access(model):
    constellation, trace, ground, environment = model
    topology = build_initial_topology(constellation, trace, 300.0)
    expected_source = ground.selected_access("new_york", 300.0)
    expected_destination = ground.selected_access("london", 300.0)
    for method in ("raw", "lifetime", "prp", "remaining"):
        route = environment.planner.plan(
            method, "new_york", "london", 300.0, 60.0, topology
        )
        assert route is not None
        assert route.ingress == expected_source
        assert route.egress == expected_destination


@pytest.mark.parametrize(
    "method",
    [
        "raw",
        "lifetime",
        "prp",
        "remaining",
        "cost_greedy",
        "constrained_oracle",
        "physical_upper",
    ],
)
def test_all_cpu_methods_execute(model, method):
    _constellation, _trace, _ground, environment = model
    result = environment.run(method, SessionRequest("new_york", "london", 300.0, 20.0))
    assert result.method == method
    assert result.reason
    if result.setup_success:
        assert result.setup_delay_s in (0.0, 20.0)


def test_planned_route_respects_one_dynamic_terminal_per_satellite(model):
    constellation, trace, _ground, environment = model
    topology = build_initial_topology(constellation, trace, 300.0)
    route = environment.planner.plan("remaining", "new_york", "hanoi", 300.0, 60.0, topology)
    assert route is not None
    dynamic = environment._dynamic_edges(route)
    incidence = {}
    for u, v in dynamic:
        incidence[u] = incidence.get(u, 0) + 1
        incidence[v] = incidence.get(v, 0) + 1
    assert max(incidence.values(), default=0) <= 1


def test_session_records_bounded_route_quality(model):
    _constellation, _trace, _ground, environment = model
    result = environment.run(
        "lifetime", SessionRequest("new_york", "london", 300.0, 60.0)
    )
    assert result.setup_success
    assert result.route_decisions >= 1
    assert result.mean_path_km >= 0.0
    assert result.mean_hops >= 2.0
    assert result.max_hops >= result.mean_hops
    assert result.max_path_km >= result.mean_path_km
    assert 0.0 <= result.mean_path_inefficiency <= 1.0
    assert result.mean_path_stretch_excess >= 0.0
    assert result.total_pat_time_s >= result.setup_delay_s


def test_cost_greedy_builds_relays_without_advancing_clock(model):
    method = "cost_greedy"
    _constellation, _trace, _ground, environment = model
    request = SessionRequest("new_york", "london", 300.0, 60.0)
    topology = environment._initial_topology(request.request_time_s)
    route = environment.planner.plan(
        method,
        request.source_station,
        request.destination_station,
        request.request_time_s,
        request.duration_s,
        topology,
    )
    assert route is not None
    assert len(route.satellites) == len(set(route.satellites))
    result = environment.run(method, request)
    if result.setup_success:
        assert result.established_time_s - request.request_time_s == result.setup_delay_s


def test_cost_greedy_candidate_uses_the_shared_reward():
    candidate = RelayCandidate(
        neighbor=1,
        edge=(0, 1),
        is_dynamic=True,
        is_active=False,
        is_new_dynamic=True,
        remaining_s=80.0,
        link_length_km=100.0,
        destination_progress_km=50.0,
        normalized_progress=0.05,
        normalized_link_length=100.0 / 1500.0,
        lifetime_shortfall=0.25,
        normalized_pat_cost=1.0,
    )
    config = RewardConfig()
    breakdown = candidate.reward_breakdown(config)
    assert candidate.immediate_reward(config) == breakdown.reward
    assert breakdown.reward == -breakdown.total_cost
    assert breakdown.geometric_progress_reward == pytest.approx(0.05)
    assert breakdown.link_length_cost == pytest.approx(0.05 * 100.0 / 1500.0)
    assert breakdown.lifetime_shortfall_cost == pytest.approx(0.75)
    assert breakdown.pat_setup_cost == pytest.approx(0.15)


def test_external_policy_path_matches_cost_greedy(model):
    _constellation, _trace, _ground, environment = model
    request = SessionRequest("new_york", "london", 300.0, 60.0)

    def greedy_selector(decision):
        return max(
            decision.candidates,
            key=lambda item: (
                item.immediate_reward(environment.reward_config),
                -int(item.is_new_dynamic),
                -item.neighbor,
            ),
        ).neighbor

    expected = environment.run("cost_greedy", request)
    actual = environment.run_policy(request, greedy_selector, method="cost_greedy")
    assert actual == expected


def test_policy_observation_shapes_and_mask(model):
    constellation, _trace, _ground, environment = model
    request = SessionRequest("new_york", "london", 300.0, 60.0)
    seen = []

    def selector(decision):
        observation = build_policy_observation(environment.planner, decision)
        seen.append(observation)
        return max(
            decision.candidates,
            key=lambda item: item.immediate_reward(environment.reward_config),
        ).neighbor

    environment.run_policy(request, selector)
    assert seen
    observation = seen[0]
    assert observation.node_features.shape == (
        constellation.config.total_satellites,
        NODE_FEATURE_DIM,
    )
    assert observation.edge_index.shape[0] == 2
    assert observation.edge_features.shape[1] == GRAPH_EDGE_FEATURE_DIM
    assert observation.action_features.shape == (
        constellation.config.total_satellites,
        ACTION_FEATURE_DIM,
    )
    assert observation.action_mask.sum() > 0
    assert observation.start == observation.current
    assert observation.previous == observation.current
    assert observation.node_features[observation.current, VISIT_ORDER_INDEX] > 0.0
    assert observation.node_features[:, PREVIOUS_NODE_INDEX].sum() == 0.0
    assert (observation.node_features[:, HOP_FRACTION_INDEX] == 0.0).all()

    later = next((item for item in seen if item.node_features[:, 10].sum() >= 2), None)
    if later is not None:
        orders = later.node_features[:, VISIT_ORDER_INDEX]
        selected_orders = sorted(orders[orders > 0.0])
        assert len(selected_orders) == int(later.node_features[:, 10].sum())
        assert all(a < b for a, b in zip(selected_orders, selected_orders[1:]))
        assert orders[later.current] == max(selected_orders)
        assert later.node_features[later.previous, PREVIOUS_NODE_INDEX] == 1.0
        assert later.node_features[:, PREVIOUS_NODE_INDEX].sum() == 1.0
        assert later.node_features[0, HOP_FRACTION_INDEX] > 0.0


def test_route_static_observation_cache_is_exact(model):
    _constellation, _trace, _ground, environment = model
    request = SessionRequest("new_york", "london", 300.0, 60.0)
    static = None
    comparisons = 0

    def selector(decision):
        nonlocal static, comparisons
        if len(decision.state.satellites) == 1:
            static = None
        fresh = build_policy_observation(
            environment.planner,
            decision,
            include_route_history=False,
            include_current_route_state=True,
        )
        cached = build_policy_observation(
            environment.planner,
            decision,
            include_route_history=False,
            include_current_route_state=True,
            static_observation=static,
        )
        if static is None:
            static = fresh
        else:
            comparisons += 1
        np.testing.assert_array_equal(cached.node_features, fresh.node_features)
        np.testing.assert_array_equal(cached.edge_index, fresh.edge_index)
        np.testing.assert_array_equal(cached.edge_features, fresh.edge_features)
        np.testing.assert_array_equal(cached.action_features, fresh.action_features)
        np.testing.assert_array_equal(cached.action_mask, fresh.action_mask)
        return max(
            decision.candidates,
            key=lambda item: item.immediate_reward(environment.reward_config),
        ).neighbor

    environment.run_policy(request, selector)
    assert comparisons > 0


def test_no_rct_observation_removes_lifetime_derived_inputs(model):
    _constellation, _trace, _ground, environment = model
    request = SessionRequest("new_york", "london", 300.0, 60.0)
    seen = []

    def selector(decision):
        seen.append(
            build_policy_observation(
                environment.planner,
                decision,
                include_remaining_contact=False,
            )
        )
        return decision.candidates[0].neighbor

    environment.run_policy(request, selector)
    assert seen
    observation = seen[0]
    assert (observation.edge_features[:, 5:8] == 0.0).all()
    assert (observation.action_features[:, 5:7] == 0.0).all()
    assert (observation.edge_features[:, 4] > 0.0).all()
    assert (observation.action_features[observation.action_mask, 4] > 0.0).all()


@pytest.mark.parametrize("architecture", ["gcn", "mlp"])
def test_ppo_forward_update_and_checkpoint(model, tmp_path, architecture):
    _constellation, _trace, _ground, environment = model
    request = SessionRequest("new_york", "london", 300.0, 60.0)
    decisions = []

    def selector(decision):
        decisions.append(decision)
        return decision.candidates[0].neighbor

    environment.run_policy(request, selector)
    assert decisions
    observation = build_policy_observation(environment.planner, decisions[0])
    agent = PPOAgent(
        PPOConfig(
            architecture=architecture,
            hidden_dim=16,
            minibatch_size=2,
            update_epochs=1,
            route_history=True,
        )
    )
    action, log_prob, value = agent.act(observation)
    assert observation.action_mask[action]
    transitions = [
        {
            "observation": observation,
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "advantage": advantage,
            "return": value + advantage,
        }
        for advantage in (0.5, -0.25)
    ]
    logs = agent.update(transitions)
    assert {"policy_loss", "value_loss", "entropy", "kl", "clip_fraction"} <= set(logs)
    assert all(torch.isfinite(torch.tensor(number)) for number in logs.values())

    checkpoint = tmp_path / f"{architecture}.pt"
    agent.save(str(checkpoint), seed=7)
    restored = PPOAgent(agent.config)
    assert restored.load(str(checkpoint)) == {"seed": 7}
