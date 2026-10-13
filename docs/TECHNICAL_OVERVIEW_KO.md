# LEO 세션 라우팅 환경과 GCN-PPO 기술 정리 (2026-09-05 기준)

이 문서가 현재 v3의 최우선 기준 문서다. 과거 문서와 내용이 충돌하면 이 문서,
실행된 checkpoint의 `training_metadata.json`, frozen-test 원본 `sessions.csv`, 현재
코드 순으로 확인한다.

이 문서의 `leo_session/...`, `results_ppo_gcn/...` 같은 경로는 모두 저장소 맨 위를
기준으로 한다. 새 환경에서 처음 재현할 때는 `REPRODUCE.md`를 그대로 따른다.

## 1. 현재 결론

- 현재 제안 모델은 **PPO-GCN, 3 GCN layers, hidden 128, 1M relay steps**이며
  seed 0·1·2의 학습과 frozen 평가가 끝났다.
- CPU baseline과 완전히 같은 frozen 3,600-request test에서 seed별 strict
  completion은 **3,400(94.44%), 3,308(91.89%), 3,288(91.33%)**다.
- 3-seed 평균은 **92.56% ± 1.66%p(sample SD)**이며 모든 seed가 Lifetime과
  ISASR/Remaining을 각각 넘는다.
- 이는 사전 성공 기준인 Lifetime **3,007/3,600 = 83.53%**와
  ISASR/Remaining **2,956/3,600 = 82.11%**를 모두 넘는다.
- PPO-GCN은 전체 strict completion에서 모든 기존 online baseline보다 높다.
- 모든 지표와 모든 X에서 우세한 것은 아니다. 전체 availability는 Lifetime
  98.97%가 PPO 97.98%보다 높다. PPO의 경로는 더 길고, dynamic setup은
  Lifetime보다 적지만 ISASR보다 많다.
- X=600에서도 PPO 80.83%가 Current-reward Cost-Greedy 65.42%보다 높다.
- `physical_upper`의 100%는 PAT와 terminal occupancy를 무시한 relaxed ceiling이다.
  이것만으로 실행 가능성을 주장하지 않는다.
- 동일 `SessionEnvironment` 제약을 실제로 적용한 constructive feasibility witness는
  frozen **3,600/3,600**을 완료했다. 따라서 이 frozen set의 물리적 불가능 요청은
  0개이고 conditional completion 분모도 3,600이다. 이 witness는 binary feasibility를
  증명하지만 secondary cost까지 완전탐색한 exact time-expanded optimizer는 아니다.
- 통제 ablation은 No-RCT **78.22%**, No-PAT **83.64%**다. main seed 0 대비 각각
  -16.22%p, -10.81%p로 remaining-contact 입력과 PAT reward 항의 기여가 확인됐다.
- 5-layer/128 seed 0 1M은 **3,237/3,600 = 89.92%**로 두 baseline은 넘지만
  3-layer seed 0보다 낮아 제안 기본모델로 선택하지 않는다.

전체 표: `results_ppo_gcn/long_run_2026-09-04/gcn3_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/eval_full_test/comparison_tables/OVERALL.md`  
X별 표: 같은 폴더의 `RESULTS_BY_X.md`

최신 ablation·multi-seed 요약:
`results_ppo_gcn/priority_evidence_2026-09-04/RESULTS.md`

## 2. 연구 문제와 목적

요청은 `(source ground station, destination ground station, arrival time t0,
required service duration X)`다. 중앙 제어기는 현재 움직이는 1,584개 위성과 ISL
연결관계를 보고 ingress satellite에서 egress satellite까지 경로를 만든다. 경로가
끊기기 전에는 새 전체 경로를 다시 계획하며, 살아 있는 기존 link는 재사용할 수
있다.

목표는 두 가지를 동시에 다루는 것이다.

1. 안정성: X초 서비스를 중단 없이 끝내는 strict completion을 높인다.
2. 효율성: 지나치게 긴 경로, 불필요한 신규 dynamic ISL/PAT, 재구성을 줄인다.

relay 하나를 packet처럼 1초씩 전송하는 문제가 아니다. 한 경로를 만드는 모든
relay 선택은 동일한 event-time snapshot에서 수행되어 물리시간이 흐르지 않는다.
물리시간은 PAT, contact 종료, GSL handover, session 종료 event에서만 흐른다.

## 3. v2와 v3의 핵심 차이

v2는 작은 synthetic satellite graph에서 한 경로 선택 episode를 다뤘다. v3는
1,584-node Walker constellation, ground endpoints, 움직이는 contact, 기존 ACTIVE
topology, PAT, break-before-make, X초 서비스와 여러 reroute를 포함한다.

v2에서 유지한 원리는 다음이다.

- 위성 graph를 GCN으로 encoding한다.
- 현재 위치에서 valid neighbor 하나를 action으로 선택해 경로를 순차 구성한다.
- ingress-to-egress 경로 하나를 PPO episode로 삼는다.
- 목적지에 도착한 마지막 action에 경로 성공 보상을 그대로 더하고 GAE를 역방향으로
  전파한다.
- actor는 current/candidate/destination embedding을 비교하고 critic은 상태가치를
  예측한다.

v3에서 추가한 것은 남은 X, contact lifetime, active/new dynamic link, PAT 비용,
현재 경로 재사용/파괴 여부와 실제 session outcome이다. BFS/Dijkstra 정답 경로,
남은 최단 hop 수, oracle action은 PPO 입력이나 reward에 넣지 않는다.

## 4. 고정 환경

| 항목 | 값 | 지위 |
|---|---:|---|
| Constellation | Walker Delta 1584/24/11 | 문헌 기반 |
| Altitude / inclination | 550 km / 53 deg | 문헌 기반 |
| Orbit period | 5738.992815 s | 위 설정에서 계산 |
| LISL / GSL range | 1500 / 1000 km | reference simulation 설정 |
| Persistent terminals | 위성당 2 | model assumption |
| Dynamic terminals | 위성당 1 | model assumption |
| New dynamic ISL PAT | 20 s | model assumption |
| Ground pairs | NYC–London, NYC–Istanbul, NYC–Hanoi | short/medium/long |
| X | 20, 60, 180, 300, 600 s | controlled factor |
| Frozen test | pair/X별 240개, 총 3,600개 | 모든 방법 동일 |
| Max relay actions/route | 128 | loop-free safety cap, 시간이 아님 |
| Max reroute events/session | 128 | event-loop safety cap |

상세 출처와 분류는 `leo_session/config.py`와 `REFERENCES.md`에 있다. 초기 topology는
빈 graph가 아니다. intra-plane persistent link는 항상 있고, inter-plane dynamic
link는 contact geometry와 terminal capacity를 만족하는 deterministic matching으로
요청 시각에 이미 ACTIVE인 상태에서 시작한다.

모든 실행 가능한 방법은 동일한 GSL 선택을 쓴다. 각 ground station에서 현재 보이는
위성 중 contact 종료가 가장 늦은 위성을 선택하고, 동률이면 작은 satellite ID를
선택한다.

## 5. 한 session의 실제 진행

1. frozen request에서 ground pair, `t0`, `X`를 읽는다.
2. 동일한 규칙으로 ingress와 egress를 선택하고 동일한 초기 ACTIVE topology를 clone한다.
3. 현재 `t0` snapshot에서 ingress→egress 전체 위성 경로를 계획한다.
4. 경로의 여러 relay action 동안 위성 위치와 graph time은 그대로다.
5. 새 dynamic ISL이 있으면 충돌 없는 link들을 병렬 PAT한다. 한 개 이상이면 elapsed
   setup delay는 20초이고 `dynamic_setups`는 실제 신규 link 개수다.
6. setup이 끝난 시점부터 X초의 service interval을 시작한다.
7. 현재 경로의 GSL 또는 dynamic ISL 중 가장 먼저 끊기는 시각 20초 전에 reroute를
   계획한다.
8. reroute도 당시 snapshot에서 새 ingress→egress 전체 경로를 만든다. 현재 ACTIVE
   link는 재사용할 수 있고, 새 연결이 점유 terminal을 쓰면 break-before-make로 기존
   연결을 해제한다.
9. 기존 경로가 새 PAT 완료 때까지 살아 있고 필요한 link를 해제하지 않으면 service
   continuity가 유지된다. 아니면 실제 중단시간에 20초가 더해진다.
10. setup, completion, interruption, reconfiguration, dynamic setup, hop, path km,
    bottleneck을 기록한다.

`completion=True`는 setup 성공만 뜻하지 않는다. setup 뒤 X초 전체에 interruption이
0인 strict 완료다. `availability`는 setup 실패면 0, 아니면
`max(0, 1 - interruption/X)`다.

## 6. 비교 방법의 정확한 동작

### Raw Dijkstra

현재 feasible graph에서 relay hop 수가 최소인 경로를 찾는다. 동률이면 신규 dynamic
link 수가 적은 경로를 택한다. contact가 X를 감당하는지는 경로 선택 cost에 없다.

### Lifetime

신규 link의 PAT까지 고려했을 때 남은 required time을 버티지 못하는 dynamic edge를
제거하고, 남은 graph에서 minimum-hop 경로를 찾는다. 전체 X를 못 버티는 link를
명시적으로 제외하는 강한 baseline이다.

### PRP

초기 경로에서 dynamic edge의 PAT-adjusted remaining contact가 180초 미만이면
제외한 뒤 minimum-hop 경로를 찾는다. 구현상 이후 replacement에는 `raw`를 쓴다.
180초는 원 논문의 `p=0.9` 예시를 반영한 고정값이다.

### ISASR/Remaining

adapted ISASR cost로 Dijkstra를 실행한다. 모든 edge의 기본 cost는 1이다. dynamic
edge에는 `20 / max(remaining, 20)`을 더하고, 현재 ACTIVE가 아니면 setup unit 1을
추가한다. test에 맞춘 fitted coefficient는 사용하지 않는다.

### Current-reward Cost-Greedy

PPO와 같은 valid candidate와 아래 네 relay 항을 사용하지만, 현재 action의 즉시
reward가 가장 큰 neighbor만 고른다. 경로 전체를 look-ahead하거나 학습하지 않는다.
**경로 성공 +5, service 보상, session completion +10 같은 PPO용 보상은 사용하지
않는다.** Greedy selector는 그 outcome을 action 전에 알 수 없기 때문이다.

### PPO-GCN

동일한 sequential candidate interface에서 GCN actor가 masked neighbor action을
sampling하며 학습한다. 평가에서는 sampling하지 않고 logit이 가장 큰 valid action을
고른다. 환경, 초기 topology, GSL, PAT, reroute와 metric 계산은 CPU baseline과 같은
`SessionEnvironment`를 사용한다.

### Constrained feasibility witness

`constrained_oracle` 실행 이름으로 exact contact 종료시각을 사용해 현재 가능한 전체
경로 중 bottleneck remaining-contact가 가장 큰 경로를 전역 탐색한다. 그러나 경로를
실행할 때는 다른 방법과 똑같이 초기 ACTIVE topology, 위성당 2+1 terminal, 신규
dynamic ISL의 20초 PAT, break-before-make, continuity를 `SessionEnvironment`가 모두
검사한다. frozen 3,600개를 실제로 전부 완료했으므로 모든 요청의 binary feasibility는
constructive하게 증명됐다.

이 이름을 secondary cost까지 최적화한 exact time-expanded oracle로 확대 해석하면
안 된다. 모든 미래 route sequence를 완전탐색하지 않으므로, 만약 실패 요청이 있었다면
그 실패만으로 physical infeasibility를 판정할 수 없다. 현재는 실패가 0이어서 binary
feasibility 판정에는 미확정 요청이 없다.

### Relaxed physical upper bound

각 5초 sample에서 모든 물리적으로 feasible inter-plane contact를 즉시 쓸 수 있다고
가정하고 graph connectivity만 확인한다. dynamic terminal occupancy, ACTIVE assignment,
PAT와 실제 route continuity를 무시한다. 따라서 `Oracle`이 아니라 relaxed physical
upper bound다. 현재 v3에는 동일 제약을 모두 지키는 exact time-expanded oracle이 없다.

## 7. 현재 reward — 무엇이 누구에게 들어가는가

### 7.1 Cost-Greedy와 PPO가 공유하는 네 relay 항

현재 위성 `i`에서 후보 `j`를 고를 때:

`relay reward = 목적지 진행 보상 - 경로 길이 비용 - lifetime 부족 비용 - 신규 PAT 비용`

정확한 식은 다음과 같다.

`r_relay = 1.0*rho - 0.05*l - 3.0*q_X - 0.15*q_PAT`

- `rho = (distance(i, egress) - distance(j, egress)) / distance(ingress, egress)`
- `l = link_length / 1500 km`
- persistent ISL의 usable lifetime은 무한대로 본다.
- dynamic ISL의 usable lifetime은 `remaining_contact - 20s`이며, 이미 ACTIVE면
  20초를 빼지 않는다.
- `q_X = clip((remaining_required_service - usable_lifetime) /
  max(remaining_required_service, 20s, 1s), 0, 1)`
- `q_PAT = 1`이면 새 dynamic link, 아니면 0이다.

즉 네 항의 계수는 **진행 1.0, 링크 길이 0.05, lifetime 부족 3.0, 신규 PAT 0.15**다.
이 중 경로 길이는 안정성 항보다 약하다. 충분히 오래 버티는 경로들 사이에서 짧은
경로를 선호시키는 효율성 tie-breaker 역할이다.

### 7.2 PPO에만 추가되는 route outcome

ingress→egress 전체 경로 하나의 마지막 relay action에 다음을 더한다.

- egress 도달 및 activation 성공: `+5`
- 경로 구성 또는 activation 실패: `-5`
- 해당 경로가 실제 제공한 service: `+5 * min(1, route_service_seconds / X)`
- 해당 경로 전환 때문에 interruption이 발생: `-5`

경로 수로 `+5`를 나누지 않는다. 각 ingress→egress 경로가 하나의 PPO episode이고,
GAE를 그 경로 안에서 역방향으로 계산하므로 마지막 성공 신호가 앞 relay 선택에
전파된다.

### 7.3 PPO에 한 session당 한 번만 추가되는 outcome

마지막 route의 마지막 action에 정확히 한 번 더한다.

- strict completion: `+10`
- setup은 됐지만 strict incomplete: `-5`
- initial setup 실패: `-10`

따라서 최종 성공은 명시적으로 보상한다. route 재구성이 많아져도 session `+10`은
반복되지 않는다.

### 7.4 reward 코드 기준

현재 PPO reward의 source of truth는 `ppo_gcn/trainer.py::finish_trajectory()`와
`leo_session/reward.py::relay_reward()`다. 호출되지 않던 과거 bounded-quality 함수와
관련 설정은 최종 코드에서 제거했다. 기존 checkpoint metadata에 남은 해당 필드는
학습 당시 직렬화됐지만 PPO update에는 사용되지 않은 기록이다.

## 8. PPO-GCN 입력, 출력, network

현재 성공 checkpoint는 `route_history=False`, `current_route_state=True`다.

### Node tensor: `1584 x 13`

각 위성마다:

1. orbital radius로 정규화한 ECI position `(x,y,z)` — 3
2. 그 위성에서 egress로 향하는 unit vector — 3
3. ingress–egress 거리로 정규화한 egress까지 거리 — 1
4. current node indicator — 1
5. 이번 route의 ingress indicator — 1
6. egress indicator — 1
7. 이번 route에서 이미 방문했는지 — 1
8. 현재 active dynamic terminal을 쓰는지 — 1
9. 남은 required service time / orbit period — 1

합계 13차원이다. route history를 켰을 때만 visit order, previous-node indicator,
hop fraction 3개가 추가되어 16차원이 되지만 현재 성공 모델에는 사용하지 않았다.

### Graph edge tensor: `E x 8`, edge index: `2 x E`

각 directed feasible edge마다 dynamic 여부, ACTIVE 여부, 현재 route에 포함됐는지,
선택 시 현재 route link를 끊는지, normalized link length, normalized remaining contact,
X-feasible indicator, PAT-adjusted lifetime margin을 넣는다.

한 route를 순차 구성하는 동안 time/topology가 바뀌지 않으므로 정적 node/edge 부분은
route 시작에 한 번 만들고 재사용한다. current와 visited만 action마다 갱신한다.

### Candidate/action tensor: `1584 x 8`, mask: `1584`

각 위성 ID가 action index다. 현재 위성의 실제 valid neighbor만 mask가 true다.
후보마다 dynamic, ACTIVE, current-route 포함, current-route 파괴, normalized link
length, remaining contact, lifetime margin, egress progress를 넣는다. loop, 존재하지
않는 edge, 같은 위성에서 연속 두 dynamic terminal을 요구하는 edge, PAT 자체를
완료할 수 없는 신규 edge는 mask한다.

### 3-layer/128 actor-critic

- 13→128 node projection
- sparse graph block 3개
- 각 block은 self projection + neighbor message + edge projection을 degree mean한 뒤
  LayerNorm/ReLU하고 원 입력에 residual로 더한다.
- actor input은 current embedding, candidate embedding, candidate-current difference,
  egress embedding과 action feature 8개다: `4*128+8 = 520`.
- actor MLP는 `520→128→128→1`; 1,584개 node logit 중 mask된 neighbor만 categorical
  action이 된다.
- critic input은 current, egress, 전체 node mean embedding: `3*128 = 384`.
- critic MLP는 `384→128→1`이다.
- trainable parameters: 3-layer 237,058개; 같은 폭의 5-layer 305,410개.

## 9. 한 학습 episode와 PPO update

한 session 안에 initial route와 여러 reroute가 있을 수 있다. 그러나 PPO episode
경계는 **각 ingress→egress route construction**이다. later reroute의 GAE가 앞 route로
넘어가지 않는다.

학습 흐름:

1. pair, X, t0를 무작위로 뽑는다.
2. route snapshot을 observation으로 만든다.
3. actor가 valid neighbor 중 하나를 sampling한다.
4. 공유 네 항의 relay reward를 즉시 기록한다.
5. egress 도달 또는 실패까지 2–4를 반복한다.
6. 실제 activation/service 결과를 route 마지막 action에 더한다.
7. session이 끝나면 마지막 route에 session outcome을 한 번 더한다.
8. route마다 뒤에서 앞으로 `gamma=0.99`, `lambda=0.95` GAE를 계산한다.
9. 여러 session에서 최소 1,024 relay transition을 모으면 PPO update cycle 하나를
   실행한다.
10. 최대 4 epoch, minibatch 64로 clipped policy/value loss를 최적화한다. 각 epoch의
    nominal minibatch 수는 16개지만 한 session을 끝까지 받은 뒤 update하므로 실제
    rollout은 1,024보다 약간 클 수 있다. target KL 초과 시 조기 종료한다.

간단한 학습 예: ingress A에서 B, B에서 egress C를 선택했다고 하자. A→B와 B→C에
각각 네 항 reward가 기록된다. C에 도착하고 activation되면 B→C action에 `+5`가
붙는다. 그 경로가 X의 절반을 서비스하면 `+2.5`가 붙는다. session이 최종적으로
strict complete면 마지막 route 마지막 action에 `+10`이 붙는다. GAE를 역방향으로
계산하므로 이 신호는 같은 경로의 A→B 선택에도 advantage로 전달된다. 숫자는 실제
link geometry에 따라 달라지며 정답 경로 자체는 제공하지 않는다.

## 10. 현재 1M 학습 설정

| 항목 | 값 |
|---|---:|
| Seed | 0, 1, 2 independently |
| GCN layers / hidden | 3 / 128 |
| Relay steps | seed 0: 1,000,013; seed 1: 1,000,110; seed 2: 1,000,102 |
| Sessions | seed 0: 17,097; seed 1: 18,053; seed 2: 16,610 |
| PPO update cycles | seed 0: 919; seed 1: 919; seed 2: 921 |
| Rollout / minibatch / epochs | 1024 / 64 / 4 |
| Initial learning rate | 0.003 |
| LR schedule | linear to 5% floor over configured run |
| Gamma / GAE lambda | 0.99 / 0.95 |
| PPO clip / value clip | 0.2 / 1.0 |
| Value / entropy coefficient | 0.5 / 0.01 |
| Max grad norm / target KL | 0.5 / 0.03 |
| Validation during run | 0 requests |
| Device | physical GPU 2 exposed as `cuda:0` |

minibatch optimizer step 수는 KL early stop과 1,024를 넘는 실제 rollout 크기 때문에
기존 CSV만으로 정확히 복원할 수 없다. 1,024개라고 가정하고 early stop이 없으면
cycle당 `16*4 = 64` optimizer steps이며 seed 0·1의 919 cycle 단순 상한식은
58,816회, seed 2의 921 cycle 상한식은 58,944회다.

## 11. Frozen 3,600 결과

| Method | Setup | Strict completion | Availability | Interruption | Reconfigs | New dynamic ISLs | Route km |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 86.11% | 51.14% | 74.19% | 34.93 s | 1.069 | 12.385 | 14,006 |
| Lifetime | 100.00% | 83.53% | 98.97% | 4.32 s | 0.924 | 9.416 | 14,885 |
| PRP | 100.00% | 72.39% | 93.30% | 31.33 s | 0.971 | 11.256 | 14,764 |
| ISASR/Remaining | 99.44% | 82.11% | 98.01% | 5.68 s | 1.085 | 4.594 | 14,586 |
| Current-reward Cost-Greedy | 89.00% | 80.64% | 86.23% | 13.47 s | 0.656 | 0.556 | 25,832 |
| PPO-GCN 3L/128 1M | 98.86% | **94.44%** | 97.98% | 4.13 s | 0.893 | 8.811 | 19,239 |
| Relaxed physical upper | 100.00% | 100.00% | 100.00% | 0.00 s | 0.000 | 0.000 | N/A |

위 PPO 행은 동일 조건 비교표를 위한 seed 0이다. 최종 3-seed 결과는 다음과 같다.

| Seed | Strict completion | Rate |
|---:|---:|---:|
| 0 | 3,400/3,600 | 94.44% |
| 1 | 3,308/3,600 | 91.89% |
| 2 | 3,288/3,600 | 91.33% |
| Mean ± sample SD | - | 92.56% ± 1.66%p |

모든 seed가 Lifetime 83.53%와 ISASR/Remaining 82.11%를 넘는다. 3-seed 평균은
각각 +9.03%p, +10.44%p 높다.

통제 ablation과 architecture 결과:

| Model | Setup | Strict completion | Availability | Interruption | New dynamic ISLs | Route km |
|---|---:|---:|---:|---:|---:|---:|
| Main 3L seed 0 | 98.86% | 94.44% | 97.98% | 4.13 s | 8.811 | 19,239 |
| No-RCT seed 0 | 99.56% | 78.22% | 97.08% | 6.87 s | 16.608 | 19,354 |
| No-PAT seed 0 | 99.36% | 83.64% | 96.13% | 14.69 s | 19.149 | 20,325 |
| 5-layer seed 0 | 99.33% | 89.92% | 98.17% | 6.06 s | 10.912 | 18,272 |

No-RCT는 observation에서 remaining-contact 정보만 제거했고 reward·환경·구조는
같다. main-only/No-RCT-only completion discordance는 696/112이며 exact two-sided
McNemar p=9.40e-104다. No-PAT는 물리 PAT와 observation은 유지하고 relay PAT reward
항만 0으로 만들었다. main-only/No-PAT-only는 493/104, p=1.71e-61이다.

Constrained feasibility witness는 동일 frozen 3,600개와 모든 X별 720개를 전부
완료했다. 그러므로 이 test set의 물리적 infeasibility는 0이며 conditional completion은
unconditional completion과 같은 분모를 쓴다.

모든 평균 route hop/path 값은 setup 성공 session의 route별 산술평균이다. 여러 새
dynamic link는 병렬 PAT하므로 dynamic setup 수와 elapsed PAT time은 같은 개념이
아니다. `2 * dynamic_setups`는 link endpoint terminal 참여 횟수이며 unique satellite
수는 아니다.

X별 720개 표는 현재 1M run의 `eval_full_test/comparison_tables/X_*s.md`에 각각 있고,
모든 원수치는 같은 폴더의 CSV에 있다. 요청 identity 검사 결과 일곱 방법 모두 정확히
3,600개가 일치한다.

현재 결과가 말하는 범위는 명확하다.

- PPO는 overall strict completion에서 Lifetime보다 +10.92%p, ISASR보다 +12.33%p,
  current Cost-Greedy보다 +13.81%p 높다.
- PPO는 X=180, X=300, X=600에서 실행 가능한 비교 방법 중 completion이 가장 높다.
- X=20/60에서는 Lifetime과 PRP가 100%이고 PPO는 99.58/99.86%다.
- PPO는 Lifetime보다 긴 경로와 더 많은 dynamic setup을 사용한다. 그러므로 현재
  결과는 안정성과 효율성의 모든 측면을 동시에 지배한다고 주장할 수 없다.

## 12. 평가 규칙

- 논문 성능 및 baseline 비교는 오직 `paper_artifacts/evaluation_requests.csv`의
  frozen 3,600개 전체 결과를 사용한다.
- 180-request validation은 checkpoint 내부 진단만 가능하다.
- 30개 또는 X 하나의 subset은 정성/디버깅 자료일 뿐 논문 수치가 아니다.
- validation이 좋았다는 말로 frozen test 실패를 정당화하지 않는다.
- PPO 성공 기준은 frozen strict completion이 Lifetime 83.53%와 ISASR 82.11%를
  모두 초과하는 것이다.
- reward/input/architecture 설정 검증은 최대 10k relay steps다. 검증되지 않은
  구성을 바로 50k 이상 돌리지 않는다.

## 13. 코드 지도

활성 환경:

- `leo_session/config.py`: 모든 수치, 출처 분류
- `leo_session/orbit.py`: Walker orbit/geometry
- `leo_session/contacts.py`: periodic ISL contact cache
- `leo_session/ground.py`: ground station/GSL contact/access
- `leo_session/topology.py`: 초기 ACTIVE topology와 terminal 상태
- `leo_session/routing.py`: baseline, Cost-Greedy, policy action interface
- `leo_session/environment.py`: PAT, reroute, interruption, metric event loop
- `leo_session/reward.py`: 네 relay 항과 reward config
- `leo_session/oracle.py`: relaxed physical upper bound

활성 PPO:

- `ppo_gcn/observation.py`: 13-D node, 8-D edge/action observation
- `ppo_gcn/model.py`: sparse residual GCN actor-critic와 PPO update
- `ppo_gcn/trainer.py`: route episode, reward 부착, GAE, rollout, checkpoint
- `scripts/train_gcn_ppo.py`: training CLI
- `scripts/evaluate_gcn_ppo.py`: frozen evaluation CLI
- `scripts/plot_ppo_training.py`: learning curve
- `scripts/audit_gcn_actions.py`: 10k 설정의 실제 action-level 정성 검사
- `scripts/run_priority_evidence.sh`: No-RCT→No-PAT→multi-seed 실행 체인

평가/인수인계:

- `scripts/run_cpu_baselines.py`: CPU baseline/Cost-Greedy 실행
- `scripts/build_comparison_tables.py`: 동일 request 검증과 전체/X별 표
- `scripts/analyze_constrained_feasibility.py`: feasible witness와 conditional completion
- `scripts/verify_reproduction_bundle.py`: 입력·checkpoint·결과 무결성 검사
- `scripts/build_contact_trace.py`: 3D geometry에서 periodic contact trace 재생성
- `results_paper/main_240_metrics/sessions.csv`: CPU baseline 원본
- `results_paper/cost_greedy_current_reward_frozen3600/sessions.csv`: 현재 reward Greedy 원본
- `results_paper/constrained_oracle_frozen3600_2026-09-04/`: 동일 제약 feasible witness
- `results_ppo_gcn/long_run_2026-09-04/gcn3_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/`: 현재 PPO 원본
- `results_ppo_gcn/priority_evidence_2026-09-04/`: seed 1·2, No-RCT, No-PAT와 통합 요약

실패한 PPO/Cost-Greedy 결과, obsolete BFS reward, 초기 draft 코드와 중간 checkpoint는
최종 정리에서 삭제했다. 10k 진단은 checkpoint 없이 action audit와 학습곡선만 남겼다.
최종 package에는 main seed 0·1·2, No-RCT, No-PAT, 5-layer secondary 모델의 최종
checkpoint만 포함한다.

## 14. 재현 명령

새 환경 생성, exact dependency, contact trace 재생성, main 3-seed, ablation, 평가와
표 생성까지의 전체 절차는 `REPRODUCE.md`에 있다. 아래는 핵심 명령이다.

CPU baseline 전체:

```bash
.venv/bin/python scripts/run_cpu_baselines.py \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --output-dir results_paper/reproduction_cpu \
  --methods raw lifetime prp remaining physical_upper
```

현재 reward Cost-Greedy:

```bash
.venv/bin/python scripts/run_cpu_baselines.py \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --output-dir results_paper/reproduction_cost_greedy \
  --methods cost_greedy \
  --reward-relay-progress-weight 1.0 \
  --reward-relay-link-weight 0.05 \
  --reward-relay-lifetime-shortfall-weight 3.0 \
  --reward-relay-pat-weight 0.15 \
  --max-relay-actions 128
```

동일 제약 feasibility witness:

```bash
.venv/bin/python scripts/run_cpu_baselines.py \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --output-dir results_paper/reproduction_constrained_witness \
  --methods constrained_oracle
```

PPO GPU 작업은 반드시 물리 GPU 2만 노출한다.

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/train_gcn_ppo.py \
  --seeds 0 1 2 --total-relay-steps 1000000 --device cuda:0 \
  --gcn-layers 3 --hidden-dim 128 --rollout-steps 1024 \
  --minibatch-size 64 --update-epochs 4 --learning-rate 0.003 \
  --gamma 0.99 --gae-lambda 0.95 --value-clip 1.0 \
  --validation-offsets 0 --output-dir OUTPUT_DIR
```

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/evaluate_gcn_ppo.py \
  --models OUTPUT_DIR/best_seed0.pt OUTPUT_DIR/best_seed1.pt OUTPUT_DIR/best_seed2.pt \
  --device cuda:0 \
  --request-file results_paper/paper_artifacts/evaluation_requests.csv \
  --output-dir OUTPUT_DIR/eval_full_test
```

검증:

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/verify_reproduction_bundle.py
```

## 15. 현재 실행 상태

요청된 장기 작업은 모두 2026-09-05 04:36:35 KST에 exit code 0으로 끝났다.
현재 실행 중인 v3 training/evaluation은 없다.

- main 3-layer seed 0·1·2: 학습과 seed별 frozen 3,600 평가 완료
- No-RCT/No-PAT: 각각 10k action audit, 1M 학습, frozen 3,600 평가 완료
- constrained feasibility witness: CPU frozen 3,600 완료
- 5-layer seed 0: 1M과 frozen 3,600 완료, secondary architecture 결과로만 보존

## 16. 포함 checkpoint와 용도

| 용도 | 파일 | SHA-256 |
|---|---|---|
| Main seed 0 | `long_run_2026-09-04/gcn3_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/best_seed0.pt` | `466875a5...e26e1ef8` |
| Main seed 1 | `priority_evidence_2026-09-04/main_3l_h128_seeds1_2_1m_mb64_gpu2/best_seed1.pt` | `146bdd4e...5ea2b53e` |
| Main seed 2 | 같은 폴더 `best_seed2.pt` | `f5502a3c...429e9905` |
| No-RCT seed 0 | `priority_evidence_2026-09-04/no_rct_3l_h128_seed0_1m_mb64_gpu2/best_seed0.pt` | `4166e151...bbeab62d` |
| No-PAT seed 0 | `priority_evidence_2026-09-04/no_pat_3l_h128_seed0_1m_mb64_gpu2/best_seed0.pt` | `a9fc5938...14f95702` |
| 5-layer secondary | `long_run_2026-09-04/gcn5_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/best_seed0.pt` | `1e5aca8f...f711ccc` |

위 표는 읽기 편하게 hash를 줄였다. 전체 hash는 package의 `FILE_MANIFEST.csv`,
`CHECKSUMS_SHA256.txt`, `CHECKPOINT_INDEX.csv`와
`scripts/verify_reproduction_bundle.py`에 있다. checkpoint는
network weights, PPO config, seed, relay steps를 포함하며 CPU로도 load 가능하다.
