# v3 평가 지표 정의

## 1. 논문의 1차 성능지표

- `setup_success_rate`: request 뒤 end-to-end session을 실제로 establish한 비율
- `completion_rate`: establish 이후 요구된 X초 동안 interruption이 0인 비율
- `availability`: 각 request에서 `1 - interruption_s/X`; setup 실패는 0
- `setup_delay_s`: 최초 session establish까지 소요된 PAT 시간

메인 주장은 strict completion으로 비교하고 setup success와 availability를 반드시
같이 보고한다. completion만 높고 setup 실패가 많거나 긴 interruption이 생기는
방법을 좋은 방법으로 해석하지 않는다.

## 2. 경로 효율

- `initial_hops`: 최초 경로의 GSL 2개 + ISL hop 수
- `mean_hops`: 한 session에서 실제 선택된 모든 경로의 hop 수 산술평균
- `max_hops`: 해당 session에서 선택된 가장 긴 경로의 hop 수
- `initial_path_km`: 최초 ground-to-ground path 거리
- `mean_path_km`: 선택된 모든 경로 거리의 산술평균
- `max_path_km`: 해당 session의 최대 경로 거리
- `mean_path_stretch_excess`: 3-D ground 직선거리 대비 path stretch excess 평균

`mean_*`은 경로별 평균이며 경로가 유지된 시간으로 가중하지 않는다. 논문 표의
`Route-mean`은 먼저 session 내부에서 평균한 뒤, setup에 성공한 session끼리 다시
평균한 값이다.

## 3. 연결 안정성

- `initial_bottleneck_s`: 최초 경로를 선택할 때 가장 짧은 GSL/dynamic ISL 잔여수명
- `mean_planned_bottleneck_s`: session에서 선택된 경로들의 bottleneck 평균
- `min_planned_bottleneck_s`: 해당 session이 선택한 경로 중 최소 bottleneck
- `interruption_s`: X 동안 end-to-end active path가 없었던 총시간

긴 bottleneck이 항상 좋은 경로를 뜻하지는 않으므로 hop/km와 함께 해석한다.

## 4. PAT와 제어 부담

- `reconfigurations`: 최초 경로 이후 경로 교체 횟수
- `route_decisions`: 최초 경로를 포함해 실제 활성화된 경로 수
- `dynamic_setups`: 새로 획득한 dynamic ISL 개수
- `total_pat_time_s`: 최초 및 재구성 단계에서 PAT가 진행된 event 시간의 합

여러 신규 ISL이 서로 다른 terminal에서 병렬 setup되면 한 event의 PAT 시간은
20초이고 `dynamic_setups`는 여러 개일 수 있다. 따라서 두 지표는 같은 의미가
아니다. old path가 PAT 동안 유지되면 `total_pat_time_s`는 증가하지만
`interruption_s`는 증가하지 않을 수 있다.

## 5. 통계 보고

- 각 pair/X/method에 setup과 completion의 Wilson 95% CI
- 같은 request ID를 사용한 completion 차이
- paired bootstrap 95% CI와 exact McNemar test
- ground-pair별 결과와 pooled 결과
- failure reason 전체 집계

## 6. 이 simulator가 주장하지 않는 지표

현재 환경에는 traffic load, queue, bandwidth sharing, congestion, energy model이
없다. 따라서 throughput, packet delay under load, packet loss, energy efficiency를
논문 결과처럼 보고하지 않는다. path km에서 진공 중 propagation delay를 단순
환산할 수는 있지만 routing mechanism의 주 지표로 사용하지 않는다.

