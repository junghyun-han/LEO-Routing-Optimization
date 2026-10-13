# 현재 v3 reward

현재 기준은 `TECHNICAL_OVERVIEW_KO.md` 7절과 실제 코드
`leo_session/reward.py`, `ppo_gcn/trainer.py::finish_trajectory()`다.

## Cost-Greedy와 PPO 공통 relay reward

`r = 1.0*progress - 0.05*normalized_link_length
     - 3.0*lifetime_shortfall - 0.15*new_PAT`

네 항은 각각 egress 방향 진행, 경로 길이, 남은 X에 대한 contact lifetime 부족,
신규 dynamic ISL PAT를 나타낸다. BFS/Dijkstra 정답은 없다.

Cost-Greedy는 이 네 항의 현재 action reward만 최대화한다. 아래 outcome 보상은
Cost-Greedy에 사용하지 않는다.

## PPO route outcome

각 ingress→egress route의 마지막 action에:

- 도달·activation 성공 `+5`, 실패 `-5`
- 실제 service `+5 * min(1, route_service_seconds/X)`
- 그 route의 전환으로 interruption 발생 시 `-5`

## PPO session outcome

session당 마지막 route의 마지막 action에 정확히 한 번:

- strict completion `+10`
- setup 후 incomplete `-5`
- initial setup 실패 `-10`

GAE는 각 route 경계에서 reset된다. route 성공 `+5`는 경로 수로 나누지 않는다.

현재 reward 구현은 위 relay, route, session 항만 포함한다. 사용하지 않는 과거
bounded-quality 함수와 설정은 최종 코드에서 제거했다.
