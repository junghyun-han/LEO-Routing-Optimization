"""Event-oriented single-session simulator for the v3 CPU experiments.

This is intentionally a network-level abstraction, not a packet simulator.
Route calculation consumes no physical time.  Only optical ISL setup and
contact/hand-over events move the clock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import DEFAULT_CONFIG, EnvironmentConfig
from .contacts import ContactTrace
from .ground import GROUND_STATIONS, GroundContactTrace
from .orbit import WalkerDeltaConstellation
from .oracle import PhysicalUpperBound
from .reward import RewardConfig
from .routing import RelaySelector, Route, RoutePlanner
from .topology import ActiveTopology, Edge, build_initial_topology, canonical_edge


@dataclass(frozen=True)
class SessionRequest:
    source_station: str
    destination_station: str
    request_time_s: float
    duration_s: float


@dataclass
class SessionResult:
    method: str
    setup_success: bool = False
    setup_delay_s: float = math.nan
    completion: bool = False
    interruption_s: float = 0.0
    reconfigurations: int = 0
    dynamic_setups: int = 0
    initial_hops: int = 0
    initial_path_km: float = math.nan
    initial_bottleneck_s: float = math.nan
    route_decisions: int = 0
    mean_hops: float = math.nan
    max_hops: int = 0
    mean_path_km: float = math.nan
    max_path_km: float = math.nan
    mean_path_inefficiency: float = math.nan
    mean_path_stretch_excess: float = math.nan
    mean_planned_bottleneck_s: float = math.nan
    min_planned_bottleneck_s: float = math.nan
    total_pat_time_s: float = 0.0
    established_time_s: float = math.nan
    reason: str = ""
    outage_hits: int = 0  # unannounced satellite outages that broke the active route


class SessionEnvironment:
    def __init__(
        self,
        constellation: WalkerDeltaConstellation,
        isl_trace: ContactTrace,
        ground_trace: GroundContactTrace,
        config: EnvironmentConfig = DEFAULT_CONFIG,
        reward_config: RewardConfig = RewardConfig(),
        max_relay_actions: Optional[int] = 128,
    ):
        self.constellation = constellation
        self.isl_trace = isl_trace
        self.ground_trace = ground_trace
        self.config = config
        self.reward_config = reward_config
        self.persistent = {canonical_edge(*edge) for edge in constellation.persistent_edges()}
        self.planner = RoutePlanner(
            constellation,
            isl_trace,
            ground_trace,
            self.persistent,
            config,
            reward_config,
            max_relay_actions,
        )
        self.physical_upper = PhysicalUpperBound(
            isl_trace, ground_trace, self.persistent, config
        )
        self._topology_cache: Dict[float, ActiveTopology] = {}
        # Unannounced outages (see set_outages).  Shared with the planner.
        self.outages: Dict[int, float] = {}
        self.planner.outages = self.outages
        self.outage_detection_s = 1.0

    def set_outages(self, outages: Dict[int, float]) -> None:
        """Satellites that go down at the given times without prior notice.

        Routing methods see the satellite as normal until its outage time and
        without it afterwards.  The environment does not schedule a proactive
        reroute for it: a route through the satellite breaks at that time, the
        break is detected after ``outage_detection_s`` (counted as interruption),
        and a replacement is planned then.  Predictable v3 events (contact ends)
        are unchanged.  An empty dict reproduces v3 exactly.
        """
        self.outages.clear()
        self.outages.update({int(k): float(v) for k, v in outages.items()})
        self.planner.outages = self.outages

    def _outage_break(self, route: Route) -> float:
        return min((self.outages[s] for s in route.satellites if s in self.outages), default=math.inf)

    def _initial_topology(self, time_s: float) -> ActiveTopology:
        key = round(float(time_s), 9)
        if key not in self._topology_cache:
            self._topology_cache[key] = build_initial_topology(
                self.constellation, self.isl_trace, float(time_s), self.config
            )
        return self._topology_cache[key].clone()

    def _path_length_km(self, request: SessionRequest, route: Route, time_s: float) -> float:
        source = GROUND_STATIONS[request.source_station].eci(
            time_s, self.config.earth_radius_km
        )
        destination = GROUND_STATIONS[request.destination_station].eci(
            time_s, self.config.earth_radius_km
        )
        ingress = self.constellation.position(route.ingress, time_s)
        egress = self.constellation.position(route.egress, time_s)
        total = float(math.dist(source, ingress) + math.dist(egress, destination))
        total += sum(
            self.constellation.link_geometry(u, v, time_s)[0]
            for u, v in zip(route.satellites[:-1], route.satellites[1:])
        )
        return total

    def _record_route_quality(
        self,
        result: SessionResult,
        request: SessionRequest,
        route: Route,
        time_s: float,
    ) -> None:
        """Record a cap-free path cost against the physical straight-line lower bound."""
        length_km = self._path_length_km(request, route, time_s)
        source = GROUND_STATIONS[request.source_station].eci(
            time_s, self.config.earth_radius_km
        )
        destination = GROUND_STATIONS[request.destination_station].eci(
            time_s, self.config.earth_radius_km
        )
        lower_bound_km = float(math.dist(source, destination))
        inefficiency = min(1.0, max(0.0, 1.0 - lower_bound_km / max(length_km, 1.0e-12)))
        stretch_excess = max(0.0, length_km / max(lower_bound_km, 1.0e-12) - 1.0)
        old_count = result.route_decisions
        new_count = old_count + 1
        if old_count == 0:
            result.mean_hops = float(route.hops)
            result.max_hops = int(route.hops)
            result.mean_path_km = length_km
            result.max_path_km = length_km
            result.mean_path_inefficiency = inefficiency
            result.mean_path_stretch_excess = stretch_excess
            result.mean_planned_bottleneck_s = route.bottleneck_remaining_s
            result.min_planned_bottleneck_s = route.bottleneck_remaining_s
        else:
            result.mean_hops += (route.hops - result.mean_hops) / new_count
            result.max_hops = max(result.max_hops, route.hops)
            result.mean_path_km += (length_km - result.mean_path_km) / new_count
            result.max_path_km = max(result.max_path_km, length_km)
            result.mean_path_inefficiency += (
                inefficiency - result.mean_path_inefficiency
            ) / new_count
            result.mean_path_stretch_excess += (
                stretch_excess - result.mean_path_stretch_excess
            ) / new_count
            result.mean_planned_bottleneck_s += (
                route.bottleneck_remaining_s - result.mean_planned_bottleneck_s
            ) / new_count
            result.min_planned_bottleneck_s = min(
                result.min_planned_bottleneck_s, route.bottleneck_remaining_s
            )
        result.route_decisions = new_count

    def _dynamic_edges(self, route: Route) -> Set[Edge]:
        return {
            canonical_edge(u, v)
            for u, v in zip(route.satellites[:-1], route.satellites[1:])
            if canonical_edge(u, v) not in self.persistent
        }

    def _route_expiry(
        self, request: SessionRequest, route: Route, time_s: float
    ) -> float:
        remaining = [
            self.ground_trace.remaining_s(request.source_station, route.ingress, time_s),
            self.ground_trace.remaining_s(request.destination_station, route.egress, time_s),
        ]
        remaining.extend(
            self.isl_trace.remaining_s(*edge, time_s) for edge in self._dynamic_edges(route)
        )
        return float(time_s) + min(remaining, default=math.inf)

    def _route_alive(self, request: SessionRequest, route: Route, time_s: float) -> bool:
        if self._outage_break(route) <= time_s:
            return False
        if self.ground_trace.remaining_s(request.source_station, route.ingress, time_s) <= 0.0:
            return False
        if self.ground_trace.remaining_s(request.destination_station, route.egress, time_s) <= 0.0:
            return False
        return all(self.isl_trace.is_feasible(*edge, time_s) for edge in self._dynamic_edges(route))

    def _activate(
        self,
        request: SessionRequest,
        topology: ActiveTopology,
        route: Route,
        time_s: float,
        previous_route: Optional[Route],
    ) -> Tuple[bool, float, bool, int]:
        """Activate route; return success, delay, old-route continuity, setup count."""
        new_edges = {canonical_edge(*edge) for edge in route.new_dynamic_edges}
        if len({node for edge in new_edges for node in edge}) != 2 * len(new_edges):
            return False, 0.0, False, 0
        delay = self.config.dynamic_setup_s if new_edges else 0.0
        finish = float(time_s) + delay
        if finish > self.ground_trace.horizon_s + 1.0e-9:
            return False, delay, False, len(new_edges)
        if self.ground_trace.remaining_s(request.source_station, route.ingress, time_s) + 1.0e-9 < delay:
            return False, delay, False, len(new_edges)
        if self.ground_trace.remaining_s(request.destination_station, route.egress, time_s) + 1.0e-9 < delay:
            return False, delay, False, len(new_edges)
        if any(self.isl_trace.remaining_s(*edge, time_s) + 1.0e-9 < delay for edge in new_edges):
            return False, delay, False, len(new_edges)
        if any(self.outages.get(node, math.inf) < finish - 1.0e-9 for edge in new_edges for node in edge):
            return False, delay, False, len(new_edges)

        released: Set[Edge] = set()
        for u, v in new_edges:
            for node in (u, v):
                old = topology.dynamic_partner.get(node)
                if old is not None:
                    released.add(canonical_edge(node, old))

        continuity = previous_route is not None and self._route_alive(request, previous_route, time_s)
        if previous_route is not None and released & self._dynamic_edges(previous_route):
            continuity = False
        if previous_route is not None and self._route_expiry(request, previous_route, time_s) < finish - 1.0e-9:
            continuity = False
        if previous_route is not None and self._outage_break(previous_route) < finish - 1.0e-9:
            continuity = False

        # break-before-make: release occupied terminals at acquisition start.
        topology.dynamic.difference_update(released)
        for u, v in released:
            topology.dynamic_partner.pop(u, None)
            topology.dynamic_partner.pop(v, None)

        # All setups on distinct dynamic terminals proceed in parallel.
        for edge in new_edges:
            topology.dynamic.add(edge)
            u, v = edge
            topology.dynamic_partner[u] = v
            topology.dynamic_partner[v] = u

        active_dynamic = {canonical_edge(*edge) for edge in topology.dynamic}
        if not self._dynamic_edges(route).issubset(active_dynamic):
            return False, delay, continuity, len(new_edges)
        return True, delay, continuity, len(new_edges)

    def _outage_limit(self, expiry: float, current_time: float, target_end: float) -> float:
        """An outage matters if it hits before the route ends or is proactively replaced."""
        if expiry >= target_end - 1.0e-9:
            return target_end
        return max(current_time, expiry - self.config.dynamic_setup_s)

    def _handle_outage(self, result, request, route, topology, broken_at, target_end, plan_fn):
        """The route broke at ``broken_at`` (unannounced outage): detect, replan, re-establish.

        Returns (new route, time it carries the session) or None when the session ends here.
        """
        result.outage_hits += 1
        detect = broken_at + self.outage_detection_s
        result.interruption_s += min(self.outage_detection_s, max(0.0, target_end - broken_at))
        if detect >= target_end - 1.0e-9:
            result.reason = "outage near session end"
            return None
        candidate = plan_fn(detect, target_end - detect)
        if candidate is None:
            result.interruption_s += target_end - detect
            result.reason = "no route after outage"
            return None
        ok, setup_delay, _continuity, setups = self._activate(request, topology, candidate, detect, route)
        result.dynamic_setups += setups
        result.total_pat_time_s += setup_delay
        if not ok:
            result.interruption_s += target_end - detect
            result.reason = "setup after outage failed"
            return None
        result.interruption_s += setup_delay  # the broken route carries nothing meanwhile
        result.reconfigurations += 1
        current = detect + setup_delay
        self._record_route_quality(result, request, candidate, current)
        return candidate, current

    def run(self, method: str, request: SessionRequest) -> SessionResult:
        result = SessionResult(method=method)
        t0 = float(request.request_time_s)
        if t0 < 0.0 or t0 + request.duration_s + self.config.dynamic_setup_s > self.ground_trace.horizon_s:
            result.reason = "request outside generated ground-contact window"
            return result

        if method == "physical_upper":
            result.setup_success = self.physical_upper.connected(
                request.source_station, request.destination_station, t0
            )
            if not result.setup_success:
                result.reason = "physically disconnected at request time"
                return result
            result.setup_delay_s = 0.0
            result.established_time_s = t0
            result.completion = self.physical_upper.session_feasible(
                request.source_station,
                request.destination_station,
                t0,
                request.duration_s,
            )
            result.reason = (
                "relaxed physical upper bound completed"
                if result.completion else "relaxed physical upper bound disconnected"
            )
            return result

        topology = self._initial_topology(t0)
        route = self.planner.plan(
            method, request.source_station, request.destination_station,
            t0, request.duration_s, topology,
        )
        if route is None:
            result.reason = "no route at request time"
            return result
        ok, delay, _continuity, setups = self._activate(request, topology, route, t0, None)
        result.dynamic_setups += setups
        result.total_pat_time_s += delay
        if not ok:
            result.reason = "initial route setup failed"
            return result

        established = t0 + delay
        if not self._route_alive(request, route, established):
            result.reason = "route unavailable when setup completed"
            return result
        result.setup_success = True
        result.setup_delay_s = delay
        result.established_time_s = established
        result.initial_hops = route.hops
        result.initial_path_km = self._path_length_km(request, route, established)
        result.initial_bottleneck_s = route.bottleneck_remaining_s
        self._record_route_quality(result, request, route, established)
        target_end = established + request.duration_s
        current_time = established

        # Replan one setup interval before the selected route expires.  This is
        # the smallest event model that can distinguish reactive shortest-path
        # routing from remaining-contact-aware routing.
        for _ in range(128):
            expiry = self._route_expiry(request, route, current_time)
            replacement_method = "raw" if method == "prp" else method
            broken = self._outage_break(route)
            if broken < self._outage_limit(expiry, current_time, target_end) - 1.0e-9:
                handled = self._handle_outage(
                    result, request, route, topology, max(current_time, broken), target_end,
                    lambda t, req: self.planner.plan(
                        replacement_method, request.source_station, request.destination_station, t, req, topology),
                )
                if handled is None:
                    return result
                route, current_time = handled
                continue
            if expiry >= target_end - 1.0e-9:
                result.completion = result.interruption_s <= 1.0e-9
                result.reason = "completed" if result.completion else "completed with interruption"
                return result

            trigger = max(current_time, expiry - self.config.dynamic_setup_s)
            required = max(0.0, target_end - trigger)
            replacement_method = "raw" if method == "prp" else method
            candidate = self.planner.plan(
                replacement_method, request.source_station, request.destination_station,
                trigger, required, topology,
            )
            if candidate is None:
                result.interruption_s += max(0.0, target_end - expiry)
                result.reason = "no replacement route"
                return result

            if candidate.satellites == route.satellites:
                # A reactive method may keep the expiring shortest route.  Let
                # it expire and force a new decision with the changed graph.
                current_time = min(target_end, expiry + 1.0e-6)
                if current_time >= target_end:
                    result.completion = result.interruption_s <= 1.0e-9
                    result.reason = "completed"
                    return result
                candidate = self.planner.plan(
                    replacement_method, request.source_station, request.destination_station,
                    current_time, max(0.0, target_end - current_time), topology,
                )
                if candidate is None:
                    result.interruption_s += max(0.0, target_end - expiry)
                    result.reason = "route expired without replacement"
                    return result
                trigger = current_time

            ok, setup_delay, continuity, setups = self._activate(
                request, topology, candidate, trigger, route
            )
            result.dynamic_setups += setups
            result.total_pat_time_s += setup_delay
            if not ok:
                # The old route can carry the session only until its known
                # expiry.  With no activated replacement, the rest of the
                # requested interval is interrupted.
                result.interruption_s += max(0.0, target_end - expiry)
                result.reason = "replacement setup failed"
                return result
            if setup_delay > 0.0 and not continuity:
                result.interruption_s += setup_delay
            current_time = trigger + setup_delay
            route = candidate
            result.reconfigurations += 1
            self._record_route_quality(result, request, route, current_time)

        result.reason = "reconfiguration limit exceeded"
        return result

    def run_policy(
        self,
        request: SessionRequest,
        selector: RelaySelector,
        method: str = "PPO-GCN",
    ) -> SessionResult:
        """Run the same session events while an external policy builds routes.

        The selector is invoked once per zero-time relay action, including
        later proactive reroutes.  All physical timing and outcome accounting
        are identical to :meth:`run`.
        """
        result = SessionResult(method=method)
        t0 = float(request.request_time_s)
        if t0 < 0.0 or t0 + request.duration_s + self.config.dynamic_setup_s > self.ground_trace.horizon_s:
            result.reason = "request outside generated ground-contact window"
            return result

        topology = self._initial_topology(t0)

        def plan_at(
            time_s: float,
            required_s: float,
            previous_route: Optional[Route],
        ) -> Tuple[Optional[Route], Optional[int]]:
            before = getattr(selector, "route_id", None)
            planned = self.planner.plan_policy(
                request.source_station,
                request.destination_station,
                time_s,
                required_s,
                topology,
                selector,
                previous_route=previous_route,
            )
            after = getattr(selector, "route_id", None)
            token = int(after) if after is not None and after != before else None
            return planned, token

        def route_outcome(token: Optional[int], **outcome: object) -> None:
            callback = getattr(selector, "record_route_outcome", None)
            if token is not None and callable(callback):
                callback(token, **outcome)

        route, route_token = plan_at(t0, request.duration_s, None)
        if route is None:
            route_outcome(
                route_token,
                reached_egress=False,
                activation_success=False,
                service_s=0.0,
                interruption_s=0.0,
                setup_failure=True,
            )
            result.reason = "no route at request time"
            return result
        ok, delay, _continuity, setups = self._activate(
            request, topology, route, t0, None
        )
        result.dynamic_setups += setups
        result.total_pat_time_s += delay
        if not ok:
            route_outcome(
                route_token,
                reached_egress=True,
                activation_success=False,
                service_s=0.0,
                interruption_s=0.0,
                setup_failure=True,
                new_dynamic_links=setups,
            )
            result.reason = "initial route setup failed"
            return result

        established = t0 + delay
        if not self._route_alive(request, route, established):
            route_outcome(
                route_token,
                reached_egress=True,
                activation_success=False,
                service_s=0.0,
                interruption_s=0.0,
                setup_failure=True,
                new_dynamic_links=setups,
            )
            result.reason = "route unavailable when setup completed"
            return result
        result.setup_success = True
        result.setup_delay_s = delay
        result.established_time_s = established
        result.initial_hops = route.hops
        result.initial_path_km = self._path_length_km(request, route, established)
        result.initial_bottleneck_s = route.bottleneck_remaining_s
        self._record_route_quality(result, request, route, established)
        target_end = established + request.duration_s
        current_time = established
        active_token = route_token
        active_start = established
        active_interruption = 0.0
        active_new_links = setups

        for _ in range(128):
            expiry = self._route_expiry(request, route, current_time)
            broken = self._outage_break(route)
            if broken < self._outage_limit(expiry, current_time, target_end) - 1.0e-9:
                previous = route
                handled = self._handle_outage(
                    result, request, route, topology, max(current_time, broken), target_end,
                    lambda t, req: plan_at(t, req, previous)[0],
                )
                if handled is None:
                    return result
                route, current_time = handled
                active_token, active_start, active_interruption, active_new_links = None, current_time, 0.0, 0
                continue
            if expiry >= target_end - 1.0e-9:
                result.completion = result.interruption_s <= 1.0e-9
                route_outcome(
                    active_token,
                    reached_egress=True,
                    activation_success=True,
                    service_s=max(0.0, target_end - active_start),
                    interruption_s=active_interruption,
                    session_complete=result.completion,
                    new_dynamic_links=active_new_links,
                )
                result.reason = "completed" if result.completion else "completed with interruption"
                return result

            trigger = max(current_time, expiry - self.config.dynamic_setup_s)
            candidate, candidate_token = plan_at(
                trigger, max(0.0, target_end - trigger), route
            )
            if candidate is None:
                route_outcome(
                    active_token,
                    reached_egress=True,
                    activation_success=True,
                    service_s=max(0.0, expiry - active_start),
                    interruption_s=active_interruption,
                    new_dynamic_links=active_new_links,
                )
                route_outcome(
                    candidate_token,
                    reached_egress=False,
                    activation_success=False,
                    service_s=0.0,
                    interruption_s=max(0.0, target_end - expiry),
                )
                result.interruption_s += max(0.0, target_end - expiry)
                result.reason = "no replacement route"
                return result

            if candidate.satellites == route.satellites:
                route_outcome(
                    candidate_token,
                    reached_egress=True,
                    activation_success=False,
                    service_s=0.0,
                    interruption_s=0.0,
                )
                current_time = min(target_end, expiry + 1.0e-6)
                if current_time >= target_end:
                    result.completion = result.interruption_s <= 1.0e-9
                    route_outcome(
                        active_token,
                        reached_egress=True,
                        activation_success=True,
                        service_s=max(0.0, target_end - active_start),
                        interruption_s=active_interruption,
                        session_complete=result.completion,
                        new_dynamic_links=active_new_links,
                    )
                    result.reason = "completed"
                    return result
                candidate, candidate_token = plan_at(
                    current_time,
                    max(0.0, target_end - current_time),
                    route,
                )
                if candidate is None:
                    route_outcome(
                        active_token,
                        reached_egress=True,
                        activation_success=True,
                        service_s=max(0.0, expiry - active_start),
                        interruption_s=active_interruption,
                        new_dynamic_links=active_new_links,
                    )
                    route_outcome(
                        candidate_token,
                        reached_egress=False,
                        activation_success=False,
                        service_s=0.0,
                        interruption_s=max(0.0, target_end - expiry),
                    )
                    result.interruption_s += max(0.0, target_end - expiry)
                    result.reason = "route expired without replacement"
                    return result
                trigger = current_time

            ok, setup_delay, continuity, setups = self._activate(
                request, topology, candidate, trigger, route
            )
            result.dynamic_setups += setups
            result.total_pat_time_s += setup_delay
            if not ok:
                route_outcome(
                    active_token,
                    reached_egress=True,
                    activation_success=True,
                    service_s=max(0.0, expiry - active_start),
                    interruption_s=active_interruption,
                    new_dynamic_links=active_new_links,
                )
                route_outcome(
                    candidate_token,
                    reached_egress=True,
                    activation_success=False,
                    service_s=0.0,
                    interruption_s=max(0.0, target_end - expiry),
                    new_dynamic_links=setups,
                )
                result.interruption_s += max(0.0, target_end - expiry)
                result.reason = "replacement setup failed"
                return result
            if setup_delay > 0.0 and not continuity:
                result.interruption_s += setup_delay
            old_end = trigger + setup_delay if continuity else trigger
            route_outcome(
                active_token,
                reached_egress=True,
                activation_success=True,
                service_s=max(0.0, old_end - active_start),
                interruption_s=active_interruption,
                new_dynamic_links=active_new_links,
            )
            current_time = trigger + setup_delay
            route = candidate
            active_token = candidate_token
            active_start = current_time
            active_interruption = setup_delay if setup_delay > 0.0 and not continuity else 0.0
            active_new_links = setups
            result.reconfigurations += 1
            self._record_route_quality(result, request, route, current_time)

        route_outcome(
            active_token,
            reached_egress=True,
            activation_success=True,
            service_s=max(0.0, current_time - active_start),
            interruption_s=active_interruption,
            new_dynamic_links=active_new_links,
        )
        result.reason = "reconfiguration limit exceeded"
        return result
