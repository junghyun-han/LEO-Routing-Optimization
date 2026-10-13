from __future__ import annotations

import pytest

from leo_session.reward import RewardConfig, relay_penalty_terms, relay_reward


def test_relay_reward_is_exact_negative_common_cost():
    config = RewardConfig(
        relay_progress_weight=2.0,
        relay_lifetime_shortfall_weight=3.0,
        relay_pat_weight=4.0,
    )
    breakdown = relay_reward(
        normalized_progress=0.1,
        normalized_link_length=0.5,
        lifetime_shortfall=0.25,
        normalized_pat_cost=1.0,
        config=config,
    )
    expected_cost = -2.0 * 0.1 + 0.05 * 0.5 + 3.0 * 0.25 + 4.0
    assert breakdown.total_cost == pytest.approx(expected_cost)
    assert breakdown.reward == pytest.approx(-expected_cost)


def test_relay_reward_has_four_local_components():
    breakdown = relay_reward(
        normalized_progress=0.1,
        normalized_link_length=0.5,
        lifetime_shortfall=0.05,
        normalized_pat_cost=1.0,
    )
    assert set(breakdown.__dataclass_fields__) == {
        "geometric_progress_reward",
        "link_length_cost",
        "lifetime_shortfall_cost",
        "pat_setup_cost",
        "total_cost",
        "reward",
    }


def test_relay_reward_prefers_progress_and_penalizes_short_lived_new_links():
    stable = relay_reward(
        normalized_progress=0.1,
        normalized_link_length=0.5,
        lifetime_shortfall=0.0,
        normalized_pat_cost=0.0,
    )
    short_new = relay_reward(
        normalized_progress=0.1,
        normalized_link_length=0.5,
        lifetime_shortfall=0.5,
        normalized_pat_cost=1.0,
    )
    assert stable.reward > short_new.reward


def test_local_lifetime_and_pat_terms_do_not_need_a_route_solver():
    persistent = relay_penalty_terms(
        is_dynamic=False,
        is_new_dynamic=False,
        remaining_s=0.0,
        required_s=100.0,
        pat_s=20.0,
    )
    active_dynamic = relay_penalty_terms(
        is_dynamic=True,
        is_new_dynamic=False,
        remaining_s=80.0,
        required_s=100.0,
        pat_s=20.0,
    )
    new_dynamic = relay_penalty_terms(
        is_dynamic=True,
        is_new_dynamic=True,
        remaining_s=80.0,
        required_s=100.0,
        pat_s=20.0,
    )
    assert persistent == (0.0, 0.0)
    assert active_dynamic == pytest.approx((0.2, 0.0))
    assert new_dynamic == pytest.approx((0.4, 1.0))
