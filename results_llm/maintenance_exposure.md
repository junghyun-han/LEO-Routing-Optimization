# Maintenance outage: does the first route use the announced satellite?

The outage satellite is the middle relay of the request-time Dijkstra route (scripts/make_scenarios.py), so
methods whose routes differ from Dijkstra's avoid it by chance. This table splits sessions by the first
established route. PPO-GCN counts are over 3 seeds x 180 requests; every other row is 180 sessions.

| Method | First route uses the announced satellite: sessions | completion (%) | Avoids it: sessions | completion (%) | No route set up | Share avoiding (%) |
|---|---:|---:|---:|---:|---:|---:|
| Dijkstra | 136 | 5.9 | 0 | – | 44 | 0.0 |
| Lifetime | 93 | 3.2 | 55 | 76.4 | 32 | 30.6 |
| PRP | 96 | 4.2 | 60 | 61.7 | 24 | 33.3 |
| Adapted ISASR | 86 | 2.3 | 80 | 77.5 | 14 | 44.4 |
| Cost-Greedy | 44 | 0.0 | 102 | 89.2 | 34 | 56.7 |
| PPO-GCN | 56 | 1.8 | 465 | 89.2 | 19 | 86.1 |
| GPT-4o | 118 | 8.5 | 22 | 72.7 | 40 | 12.2 |
| GPT-4o + tools | 9 | 0.0 | 166 | 78.9 | 5 | 92.2 |
| GPT-4o mini | 136 | 3.7 | 0 | – | 44 | 0.0 |
| GPT-4o mini + tools | 0 | – | 180 | 86.7 | 0 | 100.0 |

* Methods that cannot read the notice complete 0-6 % of the sessions whose first route uses the satellite.
  Their completion in the maintenance table comes from sessions that avoid it by chance (PPO-GCN 86 %).
* GPT-4o mini + tools and GPT-4o + tools avoid it in 100 % and 92 % of sessions because they read the notice.
  Without tools both models keep the Dijkstra route almost always (0 % and 12 % avoiding).
* Completion when the satellite is avoided is similar for PPO-GCN (89.2 %) and GPT-4o mini + tools (86.7 %).
