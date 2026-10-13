"""Small, CPU-only route planners used by the v3 session simulator."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .config import DEFAULT_CONFIG, EnvironmentConfig
from .contacts import ContactTrace
from .ground import GROUND_STATIONS, GroundContactTrace
from .orbit import WalkerDeltaConstellation
from .reward import (
    RelayRewardBreakdown,
    RewardConfig,
    relay_penalty_terms,
    relay_reward,
)
from .topology import ActiveTopology, Edge, canonical_edge


@dataclass(frozen=True)
class Route:
    satellites: Tuple[int, ...]
    ingress: int
    egress: int
    new_dynamic_edges: Tuple[Edge, ...]
    bottleneck_remaining_s: float

    @property
    def hops(self) -> int:
        # Includes the two GSL hops for reporting.
        return max(0, len(self.satellites) - 1) + 2


@dataclass(frozen=True)
class RelayCandidate:
    neighbor: int
    edge: Edge
    is_dynamic: bool
    is_active: bool
    is_new_dynamic: bool
    remaining_s: float
    link_length_km: float
    destination_progress_km: float
    normalized_progress: float
    normalized_link_length: float
    lifetime_shortfall: float
    normalized_pat_cost: float

    def reward_breakdown(self, config: RewardConfig) -> RelayRewardBreakdown:
        return relay_reward(
            normalized_progress=self.normalized_progress,
            normalized_link_length=self.normalized_link_length,
            lifetime_shortfall=self.lifetime_shortfall,
            normalized_pat_cost=self.normalized_pat_cost,
            config=config,
        )

    def immediate_reward(self, config: RewardConfig) -> float:
        return self.reward_breakdown(config).reward


@dataclass(frozen=True)
class RelayDecisionState:
    satellites: Tuple[int, ...]
    incoming_dynamic: bool

    @property
    def current(self) -> int:
        return self.satellites[-1]


@dataclass(frozen=True)
class RelayPolicyDecision:
    """One zero-time next-relay decision exposed to Cost-Greedy and PPO."""

    graph: Mapping[int, Sequence[Tuple[int, bool, bool, float]]]
    state: RelayDecisionState
    candidates: Tuple[RelayCandidate, ...]
    source_station: str
    destination_station: str
    destination_satellite: int
    time_s: float
    required_s: float
    source_egress_km: float
    current_route_edges: Tuple[Edge, ...] = ()


RelaySelector = Callable[[RelayPolicyDecision], int]


class RoutePlanner:
    """Plan on ACTIVE + physically FEASIBLE ISLs.

    The expanded search state records whether the route entered the current
    satellite through its dynamic terminal.  It therefore rejects a second
    inter-plane edge at the same satellite, which would require two dynamic
    terminals simultaneously.
    """

    def __init__(
        self,
        constellation: WalkerDeltaConstellation,
        trace: ContactTrace,
        ground_trace: GroundContactTrace,
        persistent_edges: Iterable[Edge],
        config: EnvironmentConfig = DEFAULT_CONFIG,
        reward_config: RewardConfig = RewardConfig(),
        max_relay_actions: Optional[int] = 128,
    ):
        self.constellation = constellation
        self.trace = trace
        self.ground_trace = ground_trace
        self.persistent = {canonical_edge(*edge) for edge in persistent_edges}
        self.config = config
        self.reward_config = reward_config
        self.max_relay_actions = 128 if max_relay_actions is None else int(max_relay_actions)
        if self.max_relay_actions < 1:
            raise ValueError("max_relay_actions must be positive")
        # Unannounced satellite outages {satellite: time}.  From that time the
        # satellite's links disappear from the graph; before it, every routing
        # method sees the satellite as normal.  Empty means v3 behaviour.
        self.outages: Dict[int, float] = {}
        self._persistent_adj: Dict[int, List[int]] = {}
        for u, v in self.persistent:
            self._persistent_adj.setdefault(u, []).append(v)
            self._persistent_adj.setdefault(v, []).append(u)

    def _graph(
        self, time_s: float, topology: ActiveTopology
    ) -> Dict[int, List[Tuple[int, bool, bool, float]]]:
        """neighbor -> (node, is_dynamic, is_active, remaining_s)."""
        graph: Dict[int, List[Tuple[int, bool, bool, float]]] = {}
        down = {s for s, at in self.outages.items() if at <= time_s}
        for u, neighbors in self._persistent_adj.items():
            if u in down:
                continue
            for v in neighbors:
                if v not in down:
                    graph.setdefault(u, []).append((v, False, True, math.inf))
        active_dynamic = {canonical_edge(*edge) for edge in topology.dynamic}

        for u, v, end_s in self.trace.active_edges(time_s):
            if u in down or v in down:
                continue
            edge = canonical_edge(u, v)
            active = edge in active_dynamic
            remaining = max(0.0, float(end_s) - float(time_s))
            graph.setdefault(u, []).append((v, True, active, remaining))
            graph.setdefault(v, []).append((u, True, active, remaining))
        return graph

    def plan(
        self,
        method: str,
        source_station: str,
        destination_station: str,
        time_s: float,
        required_s: float,
        topology: ActiveTopology,
    ) -> Optional[Route]:
        source_access = self.ground_trace.selected_access(source_station, time_s)
        destination_access = self.ground_trace.selected_access(destination_station, time_s)
        if source_access is None or destination_access is None:
            return None
        if self.outages.get(source_access, math.inf) <= time_s or self.outages.get(destination_access, math.inf) <= time_s:
            return None
        sources = [source_access]
        destinations = {destination_access}
        graph = self._graph(time_s, topology)
        active_dynamic = {canonical_edge(*edge) for edge in topology.dynamic}

        if method == "constrained_oracle":
            # Constructive feasibility witness with exact contact-end
            # knowledge.  Unlike PhysicalUpperBound, the returned route is
            # activated by SessionEnvironment and therefore pays the same PAT,
            # terminal-occupancy, and break-before-make costs as every online
            # method.
            return self._widest(
                graph,
                sources,
                destinations,
                source_station,
                destination_station,
                time_s,
                topology,
                active_dynamic,
            )

        if method == "cost_greedy":
            return self._cost_greedy(
                graph,
                source_access,
                destination_access,
                source_station,
                destination_station,
                time_s,
                required_s,
                active_dynamic,
            )

        if method == "remaining":
            return self._remaining_cost(
                graph, sources, destinations, source_station, destination_station,
                time_s, active_dynamic
            )

        if method not in {"raw", "lifetime", "prp"}:
            raise ValueError(f"unknown routing method: {method}")

        # Additive Dijkstra/BFS family. State=(satellite, entered_via_dynamic).
        distance: Dict[Tuple[int, bool], Tuple[int, int]] = {}
        parent: Dict[Tuple[int, bool], Tuple[Tuple[int, bool], Edge, bool]] = {}
        heap: List[Tuple[int, int, int, bool]] = []
        for satellite in sources:
            state = (satellite, False)
            distance[state] = (0, 0)
            heapq.heappush(heap, (0, 0, satellite, False))

        goal: Optional[Tuple[int, bool]] = None
        while heap:
            hops, new_count, node, incoming_dynamic = heapq.heappop(heap)
            state = (node, incoming_dynamic)
            if distance.get(state) != (hops, new_count):
                continue
            if node in destinations:
                goal = state
                break
            for neighbor, is_dynamic, is_active, remaining in graph.get(node, ()):
                if incoming_dynamic and is_dynamic:
                    continue
                setup = self.config.dynamic_setup_s if is_dynamic and not is_active else 0.0
                if method == "lifetime" and remaining + 1.0e-9 < required_s + setup:
                    continue
                if (
                    method == "prp"
                    and is_dynamic
                    and remaining - setup + 1.0e-9 < self.config.prp_route_hold_s
                ):
                    continue
                next_state = (neighbor, is_dynamic)
                candidate = (hops + 1, new_count + int(is_dynamic and not is_active))
                if candidate < distance.get(next_state, (10**9, 10**9)):
                    distance[next_state] = candidate
                    parent[next_state] = (state, canonical_edge(node, neighbor), is_dynamic)
                    heapq.heappush(heap, (*candidate, neighbor, is_dynamic))

        if goal is None:
            return None
        return self._restore(goal, parent, source_station, destination_station, time_s, active_dynamic)

    def plan_policy(
        self,
        source_station: str,
        destination_station: str,
        time_s: float,
        required_s: float,
        topology: ActiveTopology,
        selector: RelaySelector,
        previous_route: Optional[Route] = None,
    ) -> Optional[Route]:
        """Build a route with an external policy under the Cost-Greedy contract."""
        source = self.ground_trace.selected_access(source_station, time_s)
        destination = self.ground_trace.selected_access(destination_station, time_s)
        if source is None or destination is None:
            return None
        if self.outages.get(source, math.inf) <= time_s or self.outages.get(destination, math.inf) <= time_s:
            return None
        graph = self._graph(time_s, topology)
        active_dynamic = {canonical_edge(*edge) for edge in topology.dynamic}
        return self._sequential_route(
            graph,
            source,
            destination,
            source_station,
            destination_station,
            time_s,
            required_s,
            active_dynamic,
            selector,
            current_route_edges=(
                tuple(
                    canonical_edge(u, v)
                    for u, v in zip(
                        previous_route.satellites[:-1],
                        previous_route.satellites[1:],
                    )
                )
                if previous_route is not None
                else ()
            ),
        )

    def relay_candidates(
        self,
        graph: Mapping[int, Sequence[Tuple[int, bool, bool, float]]],
        state: RelayDecisionState,
        destination_station: str,
        destination_satellite: int,
        time_s: float,
        required_s: float,
        source_egress_km: float,
    ) -> Tuple[RelayCandidate, ...]:
        """Return the valid next-relay action set without advancing time."""
        # The relay policy must reach the already selected egress satellite.
        # Progress toward the ground station can create a false local optimum
        # at a different visible satellite, so rho_d uses the actual graph
        # target while GSL selection remains common to every method.
        destination_position = self.constellation.position(destination_satellite, time_s)
        current_position = self.constellation.position(state.current, time_s)
        current_to_destination = float(math.dist(current_position, destination_position))
        visited = set(state.satellites)
        candidates: List[RelayCandidate] = []
        for neighbor, is_dynamic, is_active, remaining in graph.get(state.current, ()):
            if neighbor in visited:
                continue
            if state.incoming_dynamic and is_dynamic:
                continue
            is_new = bool(is_dynamic and not is_active)
            if is_new and remaining + 1.0e-9 < self.config.dynamic_setup_s:
                continue
            neighbor_position = self.constellation.position(neighbor, time_s)
            link_length = float(math.dist(current_position, neighbor_position))
            neighbor_to_destination = float(math.dist(neighbor_position, destination_position))
            progress = current_to_destination - neighbor_to_destination
            lifetime_shortfall, normalized_pat_cost = relay_penalty_terms(
                is_dynamic=bool(is_dynamic),
                is_new_dynamic=is_new,
                remaining_s=float(remaining),
                required_s=float(required_s),
                pat_s=self.config.dynamic_setup_s,
                pat_reference_s=DEFAULT_CONFIG.dynamic_setup_s,
            )
            candidates.append(
                RelayCandidate(
                    neighbor=int(neighbor),
                    edge=canonical_edge(state.current, neighbor),
                    is_dynamic=bool(is_dynamic),
                    is_active=bool(is_active),
                    is_new_dynamic=is_new,
                    remaining_s=float(remaining),
                    link_length_km=link_length,
                    destination_progress_km=progress,
                    normalized_progress=progress / max(source_egress_km, 1.0e-12),
                    normalized_link_length=link_length / self.config.lisl_range_km,
                    lifetime_shortfall=lifetime_shortfall,
                    normalized_pat_cost=normalized_pat_cost,
                )
            )
        return tuple(candidates)

    def _cost_greedy(
        self,
        graph: Mapping[int, Sequence[Tuple[int, bool, bool, float]]],
        source: int,
        destination: int,
        source_station: str,
        destination_station: str,
        time_s: float,
        required_s: float,
        active_dynamic: Set[Edge],
    ) -> Optional[Route]:
        """Build one route by maximizing the shared one-step reward."""
        def selector(decision: RelayPolicyDecision) -> int:
            def ranking(item: RelayCandidate) -> Tuple[float, int, int]:
                cost = item.reward_breakdown(self.reward_config).total_cost
                return (cost, int(item.is_new_dynamic), item.neighbor)

            return min(decision.candidates, key=ranking).neighbor

        return self._sequential_route(
            graph,
            source,
            destination,
            source_station,
            destination_station,
            time_s,
            required_s,
            active_dynamic,
            selector,
        )

    def _sequential_route(
        self,
        graph: Mapping[int, Sequence[Tuple[int, bool, bool, float]]],
        source: int,
        destination: int,
        source_station: str,
        destination_station: str,
        time_s: float,
        required_s: float,
        active_dynamic: Set[Edge],
        selector: RelaySelector,
        current_route_edges: Sequence[Edge] = (),
    ) -> Optional[Route]:
        """Common zero-time route construction used by greedy and PPO."""
        source_egress = float(
            math.dist(
                self.constellation.position(source, time_s),
                self.constellation.position(destination, time_s),
            )
        )
        state = RelayDecisionState((int(source),), False)
        selected_dynamic: List[Edge] = []

        for _ in range(self.max_relay_actions):
            if state.current == destination:
                remaining = [
                    self.ground_trace.remaining_s(source_station, source, time_s),
                    self.ground_trace.remaining_s(destination_station, destination, time_s),
                ]
                remaining.extend(
                    self.trace.remaining_s(*edge, time_s) for edge in selected_dynamic
                )
                new_edges = tuple(edge for edge in selected_dynamic if edge not in active_dynamic)
                return Route(
                    state.satellites,
                    source,
                    destination,
                    new_edges,
                    min(remaining, default=math.inf),
                )

            candidates = self.relay_candidates(
                graph,
                state,
                destination_station,
                destination,
                time_s,
                required_s,
                source_egress,
            )
            if not candidates:
                return None
            decision = RelayPolicyDecision(
                graph=graph,
                state=state,
                candidates=candidates,
                source_station=source_station,
                destination_station=destination_station,
                destination_satellite=int(destination),
                time_s=float(time_s),
                required_s=float(required_s),
                source_egress_km=source_egress,
                current_route_edges=tuple(current_route_edges),
            )
            action = int(selector(decision))
            by_neighbor = {candidate.neighbor: candidate for candidate in candidates}
            if action not in by_neighbor:
                raise ValueError(f"policy selected masked relay {action}")
            chosen = by_neighbor[action]
            if chosen.is_dynamic:
                selected_dynamic.append(chosen.edge)
            state = RelayDecisionState(
                satellites=state.satellites + (chosen.neighbor,),
                incoming_dynamic=chosen.is_dynamic,
            )
        return None

    def _remaining_cost(
        self,
        graph: Mapping[int, Sequence[Tuple[int, bool, bool, float]]],
        sources: Sequence[int],
        destinations: Set[int],
        source_station: str,
        destination_station: str,
        time_s: float,
        active_dynamic: Set[Edge],
    ) -> Optional[Route]:
        """Adapted ISASR without a fitted coefficient.

        Every hop has unit base cost. A dynamic edge adds the paper's two
        concepts in PAT-normalized form: ``setup/remaining`` for stability and
        one setup unit when it is not already ACTIVE. Persistent links add
        neither term. This preserves ISASR's mechanism while avoiding a tuned
        gamma that could be selected after observing the test results.
        """
        distance: Dict[Tuple[int, bool], float] = {}
        parent: Dict[Tuple[int, bool], Tuple[Tuple[int, bool], Edge, bool]] = {}
        heap: List[Tuple[float, int, bool]] = []
        for satellite in sources:
            state = (satellite, False)
            distance[state] = 0.0
            heapq.heappush(heap, (0.0, satellite, False))

        goal: Optional[Tuple[int, bool]] = None
        while heap:
            cost, node, incoming_dynamic = heapq.heappop(heap)
            state = (node, incoming_dynamic)
            if cost > distance.get(state, math.inf) + 1.0e-12:
                continue
            if node in destinations:
                goal = state
                break
            for neighbor, is_dynamic, is_active, remaining in graph.get(node, ()):
                if incoming_dynamic and is_dynamic:
                    continue
                edge_cost = 1.0
                if is_dynamic:
                    edge_cost += self.config.dynamic_setup_s / max(
                        remaining, self.config.dynamic_setup_s, 1.0e-12
                    )
                    edge_cost += (
                        float(not is_active)
                        * self.config.dynamic_setup_s
                        / DEFAULT_CONFIG.dynamic_setup_s
                    )
                next_state = (neighbor, is_dynamic)
                candidate = cost + edge_cost
                if candidate + 1.0e-12 < distance.get(next_state, math.inf):
                    distance[next_state] = candidate
                    parent[next_state] = (state, canonical_edge(node, neighbor), is_dynamic)
                    heapq.heappush(heap, (candidate, neighbor, is_dynamic))

        if goal is None:
            return None
        return self._restore(
            goal, parent, source_station, destination_station, time_s, active_dynamic
        )

    def _restore(
        self,
        goal: Tuple[int, bool],
        parent: Mapping[Tuple[int, bool], Tuple[Tuple[int, bool], Edge, bool]],
        source_station: str,
        destination_station: str,
        time_s: float,
        active_dynamic: Set[Edge],
    ) -> Route:
        states = [goal]
        edges: List[Tuple[Edge, bool]] = []
        current = goal
        while current in parent:
            previous, edge, is_dynamic = parent[current]
            edges.append((edge, is_dynamic))
            states.append(previous)
            current = previous
        states.reverse()
        edges.reverse()
        satellites = tuple(state[0] for state in states)
        new_edges = tuple(edge for edge, dynamic in edges if dynamic and edge not in active_dynamic)
        remaining = [
            self.ground_trace.remaining_s(source_station, satellites[0], time_s),
            self.ground_trace.remaining_s(destination_station, satellites[-1], time_s),
        ]
        remaining.extend(self.trace.remaining_s(*edge, time_s) for edge, dynamic in edges if dynamic)
        return Route(satellites, satellites[0], satellites[-1], new_edges, min(remaining, default=math.inf))

    def _widest(
        self,
        graph: Mapping[int, Sequence[Tuple[int, bool, bool, float]]],
        sources: Sequence[int],
        destinations: Set[int],
        source_station: str,
        destination_station: str,
        time_s: float,
        topology: ActiveTopology,
        active_dynamic: Set[Edge],
    ) -> Optional[Route]:
        """Find the maximum-contact-bottleneck route under path constraints.

        This consumes exact future contact end times and is used only by the
        constrained feasibility witness.  The environment still performs all
        physical activation and continuity checks.  Completion therefore
        proves that a request is feasible; a failure alone is *not* an
        infeasibility certificate because this routine does not enumerate all
        future route sequences.
        """
        # Maximize bottleneck remaining time; tie-break by fewer setups then hops.
        best: Dict[Tuple[int, bool], Tuple[float, int, int]] = {}
        parent: Dict[Tuple[int, bool], Tuple[Tuple[int, bool], Edge, bool]] = {}
        heap: List[Tuple[float, int, int, int, bool]] = []
        for satellite in sources:
            bottleneck = self.ground_trace.remaining_s(source_station, satellite, time_s)
            state = (satellite, False)
            best[state] = (bottleneck, 0, 0)
            heapq.heappush(heap, (-bottleneck, 0, 0, satellite, False))

        selected: Optional[Tuple[int, bool]] = None
        selected_score: Optional[Tuple[float, int, int]] = None
        while heap:
            neg_bottle, setups, hops, node, incoming_dynamic = heapq.heappop(heap)
            score = (-neg_bottle, setups, hops)
            state = (node, incoming_dynamic)
            if best.get(state) != score:
                continue
            if node in destinations:
                end_bottle = min(score[0], self.ground_trace.remaining_s(destination_station, node, time_s))
                goal_score = (end_bottle, setups, hops)
                if selected_score is None or (-goal_score[0], goal_score[1], goal_score[2]) < (
                    -selected_score[0], selected_score[1], selected_score[2]
                ):
                    selected, selected_score = state, goal_score
                # The heap is ordered by bottleneck; once it drops below the best
                # completed route no later route can improve it.
                if heap and -heap[0][0] < selected_score[0] - 1.0e-9:
                    break
            for neighbor, is_dynamic, is_active, remaining in graph.get(node, ()):
                if incoming_dynamic and is_dynamic:
                    continue
                setup = self.config.dynamic_setup_s if is_dynamic and not is_active else 0.0
                margin = remaining - setup if is_dynamic else math.inf
                next_score = (min(score[0], margin), setups + int(is_dynamic and not is_active), hops + 1)
                next_state = (neighbor, is_dynamic)
                old = best.get(next_state)
                if old is None or (-next_score[0], next_score[1], next_score[2]) < (-old[0], old[1], old[2]):
                    best[next_state] = next_score
                    parent[next_state] = (state, canonical_edge(node, neighbor), is_dynamic)
                    heapq.heappush(heap, (-next_score[0], next_score[1], next_score[2], neighbor, is_dynamic))

        if selected is None:
            return None
        return self._restore(selected, parent, source_station, destination_station, time_s, active_dynamic)
