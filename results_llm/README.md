# LLM 실험 결과

LLM 라우팅 실험의 조건 시나리오, 실행 기록, 결과표. 실험 규약과 지표 정의는 [`docs/LLM_ROUTING.md`](../docs/LLM_ROUTING.md).

## 결과표

| 파일 | 내용 |
|---|---|
| [`table_none.md`](table_none.md) | 조건 없음 (180개): 기준 방법, PPO-GCN, GPT-4o, GPT-4o mini, GPT-4o mini + 도구 |
| [`table_maintenance.md`](table_maintenance.md) | 조건 1, 위성 정비 (180개) |
| [`table_hotspot.md`](table_hotspot.md) | 조건 2, 혼잡 지역 (180개) |
| [`table_avoid.md`](table_avoid.md) | 조건 3, 통과 금지 지역 (180개) |
| [`table_budget.md`](table_budget.md) | 조건 4, PAT 재지향 상한 (180개) |
| [`table_latency.md`](table_latency.md) | 조건 5, 지연 상한 (180개) |
| [`table_multi.md`](table_multi.md) | 여러 조건 동시 (20개): 기준 방법, PPO-GCN, LLM 경로 수정, LLM과 CP-SAT |
| [`maintenance_exposure.md`](maintenance_exposure.md) | 위성 정비 조건에서 첫 경로가 정비 위성을 지났는지에 따른 완주율 분석 |

각 표는 논문 표 II 지표(설정 성공, 완주, 가용도, 끊긴 시간, 재구성, 새 링크, 경로 길이), 완주율 95% 신뢰구간, 조건 준수
지표, LLM 결정 통계(유효 비율, Dijkstra 대비 McNemar 검정, 토큰, 비용, 도구 호출 수)로 구성.

## 시나리오 (`scenarios/`)

| 파일 | 내용 |
|---|---|
| `maintenance.json`, `hotspot.json`, `avoid.json`, `budget.json`, `latency.json` | 평가 요청 180개의 조건별 운영자 텍스트와 정답 조건 |
| `multi.json`, `multi20.json` | 여러 조건 시나리오 40개와 평가에 쓴 20개 |
| `*_dev.json` | 평가 요청과 겹치지 않는 개발용 요청의 시나리오 (프롬프트와 도구 개발에만 사용) |

모든 시나리오는 `scripts/make_scenarios.py`로 다시 생성 가능.

## 실행 기록 (방법별 폴더)

폴더 이름은 `모델_조건[_tools]` 형식. `_tools`는 Python 도구를 쓴 실행.

| 폴더 | 내용 |
|---|---|
| `gpt-4o_*`, `gpt-4o-mini_*` | LLM 경로 수정 실행. 조건은 `none`, `maintenance`, `hotspot`, `avoid`, `budget`, `latency`, `multi` |
| `baselines_*` | 같은 조건에서 기준 방법 5종과 PPO-GCN(시드 3개) 실행 |
| `reference_multi` | 여러 조건 실험의 CP-SAT 기준 모델 (정답 조건 사용) |
| `*_multi_extract` | LLM이 텍스트에서 조건 값을 추출해 CP-SAT 모델을 푼 실행 |
| `gpt-4o_multi_formulation` | LLM이 CP-SAT 코드를 직접 작성한 비교 실행 |
| `gpt-4o-mini_none_tools_v1`, `baselines_maintenance_v1` | 개발 기록 보존용 이전 실행 (도구 인자 수정 전, 경로 기록 추가 전). 결과표에는 미사용 |

폴더마다 들어 있는 파일:

| 파일 | 내용 |
|---|---|
| `sessions.csv` | 요청별 세션 결과 (설정 성공, 완주, 끊긴 시간, 재구성, 새 링크 수, 경로 길이 등) |
| `routes.jsonl` | 세션 동안 실제로 쓴 경로와 시각 (조건 준수 측정에 사용) |
| `decisions.jsonl` | LLM 결정 기록 (입력 크기, 제안 경로, 유효 여부, 오류, 도구 호출, 토큰) |
| `codes.jsonl` | 여러 조건 실험의 LLM 출력 (작성한 코드와 실행 오류, 또는 추출한 조건 값과 정답 일치 여부) |
