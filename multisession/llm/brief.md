# Task: design a heuristic for multi-session routing in a LEO satellite network

Write a Python function `solve(instance, kit)` that decides, at one instant, which
session requests to admit, which reconfigurable laser terminals to re-point, and
which route each admitted session takes. It must be fast (well under 2 s per
instance) because the exact CP-SAT model takes about a minute per instance.

## Problem
- 1,584 satellites. Each has two persistent laser links to its in-plane neighbours
  (always ACTIVE, never expire) and one reconfigurable terminal that links to a
  satellite in another plane (a dynamic link). Some dynamic links are ACTIVE at the
  decision time; any other pair in contact is a candidate.
- Re-pointing a terminal breaks its current dynamic link first; the new link is
  usable 20 s later. A session whose route uses any link switched on in this
  decision therefore starts 20 s late: every link and both ground links of its
  route must stay in contact for 20 s + X. A session with no new link needs X.
- Each laser link carries at most `instance.capacity` sessions (both directions).
- Every session goes from New York to London, Istanbul, or Hanoi and must hold
  its route for its whole duration X (no rerouting). Each ground station tracks
  the `instance.antennas` visible satellites with the longest remaining contact;
  a session may enter or leave through any of them.
- Objective, in strict priority: (1) admit as many sessions as possible,
  (2) switch on as few new links as possible, (3) use as few total ISL hops as possible.

## Data
- `instance.sessions`: tuple of `Session(index, source, destination, duration_s)`.
- `instance.capacity`, `instance.antennas`.
- `instance.usable_access(station, need_s)` -> `{satellite: remaining_s}` antenna
  satellites whose ground link lasts at least `need_s`.
- `instance.snapshot`: `adjacency_persistent[u]`, `adjacency_dynamic[u]` (candidate
  dynamic partners), `initial_dynamic` (ACTIVE dynamic links), `remaining(edge)`
  (seconds, `inf` for persistent), `setup_s` (20.0). Edges are `(min, max)` tuples;
  use `kit.canonical_edge(u, v)`.

## kit (use these to change state; they keep every rule above)
- `state = kit.State(instance)`: `state.load[edge]`, `state.matching` (ACTIVE dynamic
  links), `state.partner[sat]`, `state.has_room(edge)`, `state.terminal_free(sat)`.
- `state.search(sources, goals, need_s, allow_new, edge_cost)` -> `(path, new_edges)`
  or `None`. Cheapest simple path over links with spare capacity and remaining
  contact >= need_s. `allow_new=True` means a delayed session (need_s = 20 + X): it
  may switch free terminals on and use links switched on earlier.
  `edge_cost(info)` gets `kit.EdgeInfo(u, v, kind, remaining_s, load, capacity)` with
  `kind` in {"persistent", "dynamic", "new"} and returns a cost >= 0 (`math.inf`
  forbids the link). `kit.default_cost` = 1e6 for "new", else 1.
- `state.admit(path, new_edges)` commits a route. `state.remove(path)` releases its
  capacity (its links stay ACTIVE), so rip-up-and-reroute is possible.
- `kit.route_session(state, session, edge_cost=kit.default_cost, allow_reconfiguration=True)`
  tries an immediate route (need X), then a delayed one (need 20 + X).
- Return `kit.finish(state, routes)` where `routes = {session.index: path}`.

## Rules
- Standard library only (math, heapq, itertools, collections; `random` only with a
  fixed seed). Deterministic. Do not modify `instance`. Change state only through
  `kit.State` methods.
- Every returned solution is checked by an independent verifier; an infeasible
  solution scores zero admitted sessions for that instance.
- Reply with one ```python code block containing the complete module, starting
  with a comment of at most three lines that states the idea.
