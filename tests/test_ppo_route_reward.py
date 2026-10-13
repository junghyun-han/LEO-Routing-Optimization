from types import SimpleNamespace

import pytest

from leo_session import SessionResult
from ppo_gcn.trainer import finish_trajectory


def transition(route_id, reward=0.0, progress=0.0, reached=True):
    return {
        "reward": reward,
        "progress_reward": progress,
        "value": 0.0,
        "route_id": route_id,
        "route_reached_egress": reached,
    }


def finish(transitions, outcomes, *, setup=True, completion=True):
    result = SessionResult(method="test")
    result.setup_success = setup
    result.completion = completion
    agent = SimpleNamespace(config=SimpleNamespace(gamma=1.0, gae_lambda=0.95))
    finish_trajectory(
        transitions,
        agent,
        route_success_reward=5.0,
        route_failure_penalty=5.0,
        route_service_reward=5.0,
        route_interruption_penalty=5.0,
        session_completion_reward=10.0,
        session_incomplete_penalty=5.0,
        session_setup_failure_penalty=10.0,
        session_result=result,
        request_duration_s=300.0,
        route_outcomes=outcomes,
    )


def test_each_reroute_keeps_full_v2_progress_and_completion_credit():
    transitions = [
        transition(1, reward=1.0, progress=1.0),
        transition(2, reward=1.0, progress=1.0),
    ]
    outcomes = {
        1: {"reached_egress": True, "activation_success": True},
        2: {"reached_egress": True, "activation_success": True},
    }
    finish(transitions, outcomes)
    # Each route keeps its full +1 progress and +5 completion credit; the
    # session completion bonus is applied once on the final route.
    assert transitions[0]["reward"] == pytest.approx(6.0)
    assert transitions[1]["reward"] == pytest.approx(16.0)
    assert sum(item["reward"] for item in transitions) == pytest.approx(22.0)


def test_four_term_relay_reward_is_not_scaled():
    transitions = [transition(1, reward=-3.25, progress=0.4)]
    finish(
        transitions,
        {1: {"reached_egress": True, "activation_success": True}},
    )
    assert transitions[0]["reward"] == pytest.approx(-3.25 + 5.0 + 10.0)


def test_session_outcome_is_applied_exactly_once():
    complete = [transition(1), transition(2)]
    incomplete = [transition(1), transition(2)]
    outcomes = {
        1: {"reached_egress": True, "activation_success": True},
        2: {"reached_egress": True, "activation_success": True},
    }
    finish(complete, outcomes, completion=True)
    finish(incomplete, outcomes, completion=False)
    assert sum(x["reward"] for x in complete) - sum(x["reward"] for x in incomplete) == pytest.approx(15.0)


def test_gae_resets_at_route_boundary():
    transitions = [transition(1), transition(2)]
    outcomes = {
        1: {"reached_egress": True, "activation_success": True},
        2: {"reached_egress": False, "activation_success": False},
    }
    finish(transitions, outcomes, setup=True, completion=False)
    assert transitions[0]["advantage"] == pytest.approx(transitions[0]["reward"])
    assert transitions[1]["advantage"] == pytest.approx(transitions[1]["reward"])
