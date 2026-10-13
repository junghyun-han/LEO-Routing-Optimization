# v3 구현 설명

## 한 문장 요약

지상 source에서 지상 destination으로 `X`초짜리 session 요청이 들어오면,
이미 운용 중인 LEO 위성망 위에서 경로를 만들고 필요한 동적 ISL의 PAT를
수행한 뒤, 연결이 `X`초 동안 끊기지 않는지를 측정한다.

이 구현은 정밀 임무설계 도구가 아니라 논문의 routing 아이디어를 검증하는
네트워크 수준 시뮬레이터다.

## 시간 0에 존재하는 것

session이 들어오기 전부터 다음 링크가 ACTIVE다.

- 각 위성의 같은 궤도면 앞/뒤 위성 링크 2개
- 각 위성당 최대 1개의 인접 궤도면 동적 링크

따라서 빈 네트워크에서 시작하지 않는다. 초기 동적 링크는 session의
source, destination, `X`를 보지 않는 deterministic matching으로 만든다.
모든 비교 방법은 같은 요청에서 같은 초기화 규칙을 사용한다.

## source와 destination

source와 destination은 위성이 아니라 지상 위치다. 현재 포함된 실험쌍은
다음과 같다.

- short: New York - London
- medium: New York - Istanbul
- long: New York - Hanoi

위성이 움직이므로 지상국에 접속하는 ingress/egress 위성도 시간에 따라
바뀐다. GSL은 Bhattacharjee et al. 2024 Table III의 reference simulation과
같은 1,000 km slant-range cutoff를 사용한다. 공식 hardware 사양이라는
뜻은 아니다.

현재 보이는 위성이 여러 대이면 contact 종료가 가장 늦은 위성을 모든 방법의
공통 access 위성으로 선택한다. routing 방법이 ingress/egress까지 다르게 골라
ISL 비교가 오염되지 않게 하기 위한 명시적 실험설계다.

## 한 session의 진행

1. 요청 `(ground_source, ground_destination, t0, X)`가 들어온다.
2. controller가 ACTIVE 링크와 현재 FEASIBLE 링크를 함께 보고 경로를 계산한다.
3. 경로에 아직 ACTIVE가 아닌 동적 ISL이 있으면 해당 terminal의 기존 링크를
   끊고 PAT를 시작한다.
4. 서로 다른 terminal의 PAT는 병렬로 수행되므로 경로 setup delay는 20초다.
5. setup이 끝났을 때 end-to-end path가 생기면 session이 establish된다.
6. 그 뒤 `X`초 동안 선택 경로를 추적한다. 링크/GSL 종료가 가까워지면 새
   경로를 준비하며, 어느 순간에도 active path가 없으면 interruption으로 센다.

relay를 여러 번 선택하는 계산 자체에는 물리시간이 흐르지 않는다. 시간은
PAT 완료, contact 종료, handover 및 session 종료 event에서만 진행된다.

## 현재 CPU 비교 방법

- `raw`: remaining contact time을 보지 않는 최소-hop Dijkstra
- `lifetime`: setup과 남은 session 시간을 버티지 못하는 동적 ISL을 제거한
  뒤 최소-hop Dijkstra
- `prp`: Uzunalioglu et al.의 `p=0.9`, `T_tr=180초` 문헌 사례를 사용한
  확률적 lifetime filtering. 이 비교법은 의도적으로 현재 요청의 정확한
  `X`를 사용하지 않는다.
- `remaining`: ISASR의 stability/activeness 구조를 PAT 단위로 정규화한
  remaining-contact-aware shortest-path heuristic
- `cost_greedy`: PPO-GCN과 동일한 네 항 local 보상
  `rho_d - 0.05*l - 3*q_X - 0.15*q_P`를 현재 relay마다 가장 크게 만드는 one-step
  baseline. BFS/Dijkstra 경로나 남은 최소 홉 수는 사용하지 않는다.
- `physical_upper`: 모든 physically feasible 동적 ISL이 즉시 ACTIVE라고
  완화한 물리적 상한
- `constrained_oracle`: exact contact 종료시각으로 max-bottleneck 전체 경로를
  고르되 실제 terminal/PAT/break-before-make를 그대로 실행하는 constructive
  feasibility witness

`physical_upper`는 terminal 점유와 PAT를 완화했기 때문에 실제 알고리즘이 아니다.
반면 constrained witness는 동일 제약에서 frozen 3,600개를 전부 실제 완료해 binary
feasibility 100%를 증명했다. 다만 모든 미래 route sequence와 secondary cost를
완전탐색하지 않으므로 exact time-expanded optimizer라고 부르면 안 된다.

## X의 의미와 최종 실험 값

`X`는 packet 전송시간이나 hop 수가 아니다. path가 establish된 뒤 유지해야
하는 session 길이다. 검증된 traffic-duration 분포가 없으므로 random 분포를
주장하지 않는다. 최종 CPU baseline은 다음 controlled factor를 쓴다.

- 20초: 이전 버전에서 사용한 시간 범위를 유지하는 짧은 조건
- 60초: PRP 문헌의 1분 조건과 대응하는 짧은 session
- 180초: PRP의 `p=0.9` 예시가 사용한 3분 threshold
- 300초, 600초: 생성된 단일 GSL contact 최대 239.2초보다 길어 access
  handover를 반드시 포함하는 조건

세 지상국 쌍과 각 X마다 한 궤도 주기 전체의 240개 요청 시점을 사용했다.
120-offset 예비 실행 후 표본을 240개로 확장했고, 방법·X·환경 규칙은 결과를
본 뒤 변경하지 않았다. 120-offset과 240-offset 집계의 최대 차이는 1.3%p다.

## 코드 위치

- `leo_session/config.py`: 모든 수치와 provenance 분류
- `leo_session/orbit.py`: 3D Walker-Delta 위성 위치
- `leo_session/contacts.py`: ISL contact interval 생성/조회
- `leo_session/ground.py`: 지상 위치와 움직이는 ingress/egress contact
- `leo_session/topology.py`: session 이전 initial ACTIVE topology
- `leo_session/routing.py`: CPU routing 방법 4개
- `leo_session/environment.py`: event-driven session 실행과 지표
- `leo_session/oracle.py`: 완화된 physical upper bound
- `scripts/run_cpu_baselines.py`: 동일 요청을 모든 방법에 paired 실행
- `ppo_gcn/observation.py`: 1,584-node sparse graph observation
- `ppo_gcn/model.py`: sparse GCN/MLP actor-critic와 PPO update
- `ppo_gcn/trainer.py`: session trajectory, terminal reward, GAE, checkpoint
- `scripts/train_gcn_ppo.py`: 5-seed 학습 entry point
- `scripts/evaluate_gcn_ppo.py`: frozen 3,600-request 평가
- `tests/test_session_environment.py`: 물리/terminal/session 기본 검증

실패하거나 superseded된 v3 초안 코드는 최종 정리에서 삭제했다.

## 실행

repository root에서:

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/run_cpu_baselines.py
```

두 번째 명령은 작은 smoke 결과를 `results_cpu/`에 만든다. 논문용 실행과
분석 명령은 `results_paper/README.md`에 고정해 두었다.

## 현재 상태

- 3-layer/128 seeds 0/1/2 1M은 94.44%, 91.89%, 91.33%를 기록했다.
- 3-seed mean ± sample SD는 92.56% ± 1.66%p다.
- No-RCT 78.22%, No-PAT 83.64%의 통제 ablation이 완료됐다.
- 5-layer/128 seed 0은 89.92%로 secondary architecture 결과다.
- constrained feasibility witness는 frozen 3,600/3,600을 완료했다.

GCN-PPO는 CPU 방법과 같은 `SessionEnvironment.run_policy()`를 쓴다. 자세한 현재
구조와 실행 방법은 `TECHNICAL_OVERVIEW_KO.md`와 `GCN_PPO.md`에 있다.

현재 `physical_upper`는 PAT와 terminal 점유를 완화한 ceiling으로만 사용한다.
constrained witness는 모든 frozen request의 실행 가능성을 증명하지만 path/PAT
cost 최적성을 증명하는 exact time-expanded optimizer는 아니다.
