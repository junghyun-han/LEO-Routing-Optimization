"""Sequential admission baselines for the multi-session problem.

Both baselines admit sessions one at a time in the given order and never
revisit an earlier decision.

* ``fixed_topology`` keeps the initial dynamic links (no reconfiguration) and
  gives each session a minimum-hop path on links with spare capacity that stay
  in contact for the whole session.
* ``greedy_reconfiguration`` first tries the same thing; if no such path
  exists it may point free dynamic terminals at new partners (20 s setup,
  break-before-make).  A terminal is free when its current dynamic link carries
  no admitted session.  Among paths it minimises (new links, hops).
"""
from __future__ import annotations

import time
from typing import Dict, List

from .kit import State, finish, route_session
from .model import Instance, Solution


def _run(instance: Instance, allow_reconfiguration: bool) -> Solution:
    start = time.perf_counter()
    state = State(instance)
    routes: Dict[int, List[int]] = {}
    for session in instance.sessions:
        found = route_session(state, session, allow_reconfiguration=allow_reconfiguration)
        if found is not None:
            state.admit(*found)
            routes[session.index] = found[0]
    return finish(state, routes, start)


def fixed_topology(instance: Instance) -> Solution:
    return _run(instance, allow_reconfiguration=False)


def greedy_reconfiguration(instance: Instance) -> Solution:
    return _run(instance, allow_reconfiguration=True)
