"""CPU-only session routing environment for the v3 paper model."""

from .config import DEFAULT_CONFIG, EnvironmentConfig
from .contacts import ContactInterval, ContactTrace, ContactTraceGenerator
from .environment import SessionEnvironment, SessionRequest, SessionResult
from .ground import GROUND_PAIRS, GROUND_STATIONS, GroundContactGenerator, GroundContactTrace
from .orbit import WalkerDeltaConstellation
from .oracle import PhysicalUpperBound
from .reward import (
    RelayRewardBreakdown,
    RewardConfig,
    relay_penalty_terms,
    relay_reward,
)
from .routing import (
    RelayCandidate,
    RelayDecisionState,
    RelayPolicyDecision,
    Route,
    RoutePlanner,
)
from .topology import ActiveTopology, build_initial_topology

__all__ = [
    "ActiveTopology",
    "ContactInterval",
    "ContactTrace",
    "ContactTraceGenerator",
    "DEFAULT_CONFIG",
    "EnvironmentConfig",
    "GROUND_PAIRS",
    "GROUND_STATIONS",
    "GroundContactGenerator",
    "GroundContactTrace",
    "PhysicalUpperBound",
    "RewardConfig",
    "RelayRewardBreakdown",
    "RelayCandidate",
    "RelayDecisionState",
    "RelayPolicyDecision",
    "Route",
    "RoutePlanner",
    "SessionEnvironment",
    "SessionRequest",
    "SessionResult",
    "WalkerDeltaConstellation",
    "build_initial_topology",
    "relay_penalty_terms",
    "relay_reward",
]
