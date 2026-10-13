"""Trajectory collection, GAE, training, and evaluation for v3 PPO-GCN."""
from __future__ import annotations

import copy
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from leo_session import SessionEnvironment, SessionRequest
from leo_session.routing import RelayPolicyDecision

from .model import PPOAgent
from .observation import build_policy_observation


@dataclass(frozen=True)
class TrainingConfig:
    total_relay_steps: int = 1_000_000
    durations_s: tuple[float, ...] = (20.0, 60.0, 180.0, 300.0, 600.0)
    snapshot_interval_updates: int = 45
    checkpoint_interval_updates: int = 45
    validation_offsets: int = 0


class PolicyTrajectoryCollector:
    def __init__(
        self,
        environment: SessionEnvironment,
        agent: PPOAgent,
        *,
        greedy: bool,
        include_remaining_contact: bool,
        record: bool,
    ):
        self.environment = environment
        self.agent = agent
        self.greedy = greedy
        self.include_remaining_contact = include_remaining_contact
        self.record = record
        self.transitions: list[dict] = []
        self.route_id = 0
        self.route_bottleneck_s = float("inf")
        self.route_has_new_dynamic = False
        self.route_outcomes: dict[int, dict[str, object]] = {}
        self._route_static_observation = None

    def record_route_outcome(self, route_id: int, **outcome: object) -> None:
        self.route_outcomes[int(route_id)] = dict(outcome)

    def __call__(self, decision: RelayPolicyDecision) -> int:
        if len(decision.state.satellites) == 1:
            self.route_id += 1
            self._route_static_observation = None
            source = decision.state.satellites[0]
            self.route_bottleneck_s = min(
                self.environment.ground_trace.remaining_s(
                    decision.source_station, source, decision.time_s
                ),
                self.environment.ground_trace.remaining_s(
                    decision.destination_station,
                    decision.destination_satellite,
                    decision.time_s,
                ),
            )
            self.route_has_new_dynamic = False
        observation = build_policy_observation(
            self.environment.planner,
            decision,
            include_remaining_contact=self.include_remaining_contact,
            include_route_history=self.agent.config.route_history,
            include_current_route_state=self.agent.config.current_route_state,
            static_observation=self._route_static_observation,
        )
        if self._route_static_observation is None:
            self._route_static_observation = observation
        action, log_prob, value = self.agent.act(observation, greedy=self.greedy)
        if self.record:
            by_neighbor = {item.neighbor: item for item in decision.candidates}
            chosen = by_neighbor[action]
            reward = chosen.immediate_reward(self.environment.reward_config)
            if chosen.is_dynamic:
                self.route_bottleneck_s = min(
                    self.route_bottleneck_s, chosen.remaining_s
                )
            self.route_has_new_dynamic = (
                self.route_has_new_dynamic or chosen.is_new_dynamic
            )
            self.transitions.append(
                {
                    "observation": observation,
                    "action": int(action),
                    "log_prob": float(log_prob),
                    "value": float(value),
                    "reward": float(reward),
                    "route_id": self.route_id,
                    "route_reached_egress": bool(action == decision.destination_satellite),
                    "route_bottleneck_s": float(self.route_bottleneck_s),
                    "route_has_new_dynamic": self.route_has_new_dynamic,
                    "route_required_s": float(decision.required_s),
                }
            )
        return int(action)


def finish_trajectory(
    transitions: list[dict], agent: PPOAgent,
    *, route_success_reward: float, route_failure_penalty: float,
    route_service_reward: float, route_interruption_penalty: float,
    session_completion_reward: float, session_incomplete_penalty: float,
    session_setup_failure_penalty: float, session_result: object,
    request_duration_s: float,
    route_outcomes: dict[int, dict[str, object]],
) -> None:
    if not transitions:
        return

    route_groups: list[list[dict]] = []
    group: list[dict] = []
    for index, item in enumerate(transitions):
        group.append(item)
        is_route_end = (
            index == len(transitions) - 1
            or transitions[index + 1]["route_id"] != item["route_id"]
        )
        if is_route_end:
            route_groups.append(group)
            group = []

    duration = max(1.0, float(request_duration_s))
    for route in route_groups:
        end = route[-1]
        route_id = int(end["route_id"])
        outcome = route_outcomes.get(route_id, {})
        reached = bool(
            outcome.get("reached_egress", end.get("route_reached_egress", False))
        )
        activated = bool(outcome.get("activation_success", True))
        if reached and activated:
            # Exactly the v2 contract: one full completion bonus for the
            # ingress-to-egress route episode. Never divide it by reroute count.
            end["reward"] += float(route_success_reward)
        else:
            end["reward"] -= float(route_failure_penalty)
        service_s = max(0.0, float(outcome.get("service_s", 0.0)))
        end["reward"] += float(route_service_reward) * min(1.0, service_s / duration)
        if float(outcome.get("interruption_s", 0.0)) > 1.0e-9:
            end["reward"] -= float(route_interruption_penalty)

    # Exactly one session outcome. It cannot be multiplied by rerouting.
    setup = bool(getattr(session_result, "setup_success"))
    completion = bool(getattr(session_result, "completion"))
    if completion:
        route_groups[-1][-1]["reward"] += float(session_completion_reward)
    elif setup:
        route_groups[-1][-1]["reward"] -= float(session_incomplete_penalty)
    else:
        route_groups[-1][-1]["reward"] -= float(session_setup_failure_penalty)

    cfg = agent.config
    # Every ingress-to-egress construction is one PPO episode. GAE is reset at
    # every route boundary, so a later reroute cannot relabel an earlier route.
    for route in route_groups:
        gae = 0.0
        next_value = 0.0
        for item in reversed(route):
            delta = item["reward"] + cfg.gamma * next_value - item["value"]
            gae = delta + cfg.gamma * cfg.gae_lambda * gae
            item["advantage"] = float(gae)
            item["return"] = float(gae + item["value"])
            next_value = item["value"]


def collect_episode(
    environment: SessionEnvironment,
    agent: PPOAgent,
    request: SessionRequest,
    *,
    greedy: bool = False,
    include_remaining_contact: bool = True,
    record: bool = True,
) -> tuple[object, list[dict], float]:
    collector = PolicyTrajectoryCollector(
        environment,
        agent,
        greedy=greedy,
        include_remaining_contact=include_remaining_contact,
        record=record,
    )
    result = environment.run_policy(request, collector)
    if record:
        finish_trajectory(
            collector.transitions,
            agent,
            route_success_reward=environment.reward_config.route_success_reward,
            route_failure_penalty=environment.reward_config.route_failure_penalty,
            route_service_reward=environment.reward_config.route_service_reward,
            route_interruption_penalty=environment.reward_config.route_interruption_penalty,
            session_completion_reward=environment.reward_config.session_completion_reward,
            session_incomplete_penalty=environment.reward_config.session_incomplete_penalty,
            session_setup_failure_penalty=environment.reward_config.session_setup_failure_penalty,
            session_result=result,
            request_duration_s=request.duration_s,
            route_outcomes=collector.route_outcomes,
        )
    total_reward = sum(item["reward"] for item in collector.transitions)
    return result, collector.transitions, float(total_reward)


def validation_requests(
    environment: SessionEnvironment,
    ground_pairs: dict[str, tuple[str, str]],
    durations_s: Sequence[float],
    offsets: int,
) -> list[SessionRequest]:
    period = environment.constellation.period_s
    requests = []
    for source, destination in ground_pairs.values():
        for duration in durations_s:
            for index in range(offsets):
                # The 20-bin reward-validation and 240-bin frozen-test
                # midpoints are avoided by this independent offset.
                fraction = (index + 0.371) / offsets
                requests.append(
                    SessionRequest(source, destination, period * fraction, duration)
                )
    return requests


def evaluate_requests(
    environment: SessionEnvironment,
    agent: PPOAgent,
    requests: Iterable[SessionRequest],
    *,
    include_remaining_contact: bool = True,
) -> dict[str, float]:
    requests = list(requests)
    results = []
    for request in requests:
        result, _transitions, _reward = collect_episode(
            environment,
            agent,
            request,
            greedy=True,
            include_remaining_contact=include_remaining_contact,
            record=False,
        )
        results.append(result)
    if not results:
        return {"setup_rate": 0.0, "completion_rate": 0.0, "availability": 0.0}
    return {
        "setup_rate": float(np.mean([item.setup_success for item in results])),
        "completion_rate": float(np.mean([item.completion for item in results])),
        "availability": float(
            np.mean(
                [
                    0.0
                    if not item.setup_success
                    else max(0.0, 1.0 - item.interruption_s / request.duration_s)
                    for item, request in zip(results, requests)
                ]
            )
        ),
    }


def train_agent(
    environment: SessionEnvironment,
    agent: PPOAgent,
    ground_pairs: dict[str, tuple[str, str]],
    output_dir: Path,
    seed: int,
    config: TrainingConfig = TrainingConfig(),
    *,
    include_remaining_contact: bool = True,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"best_seed{seed}.pt"
    latest_path = output_dir / f"latest_seed{seed}.pt"
    curve_path = output_dir / f"training_seed{seed}.csv"
    validation = (
        validation_requests(
            environment,
            ground_pairs,
            config.durations_s,
            config.validation_offsets,
        )
        if config.validation_offsets > 0
        else []
    )
    transitions: list[dict] = []
    curve: list[dict] = []
    relay_steps = updates = episodes = 0
    best_completion = -1.0
    best_state = None
    period = environment.constellation.period_s
    pair_values = list(ground_pairs.values())

    while relay_steps < config.total_relay_steps:
        source, destination = random.choice(pair_values)
        duration = float(random.choice(config.durations_s))
        request = SessionRequest(
            source,
            destination,
            random.random() * period,
            duration,
        )
        result, episode, total_reward = collect_episode(
            environment,
            agent,
            request,
            greedy=False,
            include_remaining_contact=include_remaining_contact,
            record=True,
        )
        episodes += 1
        if not episode:
            continue
        transitions.extend(episode)
        relay_steps += len(episode)
        route_returns: dict[int, float] = {}
        for item in episode:
            route_id = int(item["route_id"])
            route_returns[route_id] = route_returns.get(route_id, 0.0) + float(
                item["reward"]
            )
        mean_route_reward = float(np.mean(list(route_returns.values())))
        curve.append(
            {
                "episode": episodes,
                "relay_steps": relay_steps,
                # PPO/GAE resets at each route, so this is the comparable
                # v2-style episode return. Keep the session sum separately.
                "reward": mean_route_reward,
                "session_reward_sum": total_reward,
                "route_episodes": len(route_returns),
                "setup": int(result.setup_success),
                "completion": int(result.completion),
                "interruption_s": result.interruption_s,
                "reconfigurations": result.reconfigurations,
            }
        )
        if len(transitions) < agent.config.rollout_steps:
            continue

        fraction = min(1.0, relay_steps / config.total_relay_steps)
        agent.set_learning_rate(
            agent.config.learning_rate * max(0.05, 1.0 - fraction)
        )
        update_log = agent.update(transitions)
        transitions.clear()
        updates += 1

        if validation and updates % config.snapshot_interval_updates == 0:
            metrics = evaluate_requests(
                environment,
                agent,
                validation,
                include_remaining_contact=include_remaining_contact,
            )
            if metrics["completion_rate"] > best_completion:
                best_completion = metrics["completion_rate"]
                best_state = copy.deepcopy(agent.network.state_dict())
                agent.save(
                    str(best_path),
                    seed=seed,
                    relay_steps=relay_steps,
                    validation=metrics,
                    include_remaining_contact=include_remaining_contact,
                )
            curve[-1].update(
                {f"validation_{key}": value for key, value in metrics.items()}
            )
        curve[-1].update({f"ppo_{key}": value for key, value in update_log.items()})
        # Persist the learning curve independently of heavyweight checkpoint
        # retention so long runs remain observable while keeping only final
        # model files.
        if updates % config.snapshot_interval_updates == 0:
            _write_curve(curve_path, curve)
        if updates % config.checkpoint_interval_updates == 0:
            milestone_path = output_dir / (
                f"checkpoint_update{updates:04d}_steps{relay_steps}_seed{seed}.pt"
            )
            agent.save(
                str(milestone_path),
                seed=seed,
                relay_steps=relay_steps,
                updates=updates,
                validation={
                    key.removeprefix("validation_"): value
                    for key, value in curve[-1].items()
                    if key.startswith("validation_")
                },
                include_remaining_contact=include_remaining_contact,
            )
            agent.save(
                str(latest_path),
                seed=seed,
                relay_steps=relay_steps,
                updates=updates,
                include_remaining_contact=include_remaining_contact,
            )
            _write_curve(curve_path, curve)

    if transitions:
        agent.update(transitions)
    if best_state is not None:
        agent.network.load_state_dict(best_state)
    agent.save(
        str(best_path),
        seed=seed,
        relay_steps=relay_steps,
        best_validation_completion=best_completion,
        include_remaining_contact=include_remaining_contact,
    )
    _write_curve(curve_path, curve)
    return {
        "seed": seed,
        "relay_steps": relay_steps,
        "episodes": episodes,
        "updates": updates,
        "best_validation_completion": best_completion,
        "model": str(best_path),
        "curve": str(curve_path),
        "training_config": asdict(config),
    }


def _write_curve(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
