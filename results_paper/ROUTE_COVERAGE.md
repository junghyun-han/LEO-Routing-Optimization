# Route and constellation coverage

This check distinguishes the single illustrative route figure from the full
paired evaluation. It replans the initial route for every one of the 3,600
frozen requests and all four achievable CPU methods.

## Initial topology over one orbit

- request times: 240
- distinct initial dynamic-topology signatures: 121
- ACTIVE dynamic ISLs per request time: mean 646.82, min 627, max 648
- union of initial ACTIVE dynamic ISLs over all request times: 805
- persistent intra-plane ISLs at every request time: 1,584
- maximum ACTIVE degree: 3

## Ground-access diversity

- New York: 125 distinct selected access satellites
- London: 161
- Istanbul: 131
- Hanoi: 82

## Initial route diversity

| Method | Distinct initial satellite sequences | Distinct satellites used | Requests without an initial route |
|---|---:|---:|---:|
| Raw Dijkstra | 558 | 1,544 | 0 |
| Lifetime-Dijkstra | 1,676 | 1,584 | 0 |
| PRP | 591 | 1,556 | 0 |
| Adapted ISASR | 571 | 1,325 | 0 |
| Union of all methods | 2,173 | 1,584 | 0 |

The counts above include only the first route selected for each request. Route
replacements after GSL/ISL expiry are not counted, so they are a conservative
lower bound on route diversity during the full session executions.

The figure `paper_artifacts/fig_initial_topology_route.png` intentionally shows
only one request in detail: New York-London, `t0=300 s`, `X=300 s`,
Lifetime-Dijkstra. It is a topology/route explanation, not the extent of the
experiment.
