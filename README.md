# 시변 저궤도 위성망 경로 최적화

저궤도(LEO) 위성망에서는 위성 간 링크가 궤도 운동에 따라 생기고 끊어진다.
이 저장소는 이런 시변 위성망에서 통신 세션이 끊기지 않도록 경로를 정하는 문제를
강화학습(GCN-PPO), 수리최적화(CP-SAT), LLM 기반 최적화로 다룬다.

## 연구 흐름

| 단계 | 문제 설정 | 방법 | 상태 |
|---|---|---|---|
| 1 | 패킷을 한 홉씩 전달하는 모델. 링크의 남은 접촉 시간을 반영한 선제적 라우팅 | GCN-PPO | 석사 학위논문, 한국항공우주학회 학술대회 구두발표(2026.06) |
| 2 | 세션 단위 모델. 3차원 Walker-Delta 궤도(위성 1,584개), 위성당 고정 터미널 2개와 재구성 터미널 1개, 새 링크 설정 지연 20초 | GCN-PPO, 기준 방법 | WCNC 2027 투고(심사 중). 이 저장소의 환경 코드 |
| 3 | 여러 세션이 재구성 터미널과 링크 용량을 나눠 쓰는 터미널 배정과 경로의 결합 문제 | CP-SAT 정식화, LLM 최적화 에이전트, GCN-PPO 비교 | 진행 중 |

## 2단계 결과

고정된 테스트 요청 3,600개(유지 시간 X = 20, 60, 180, 300, 600초 각 720개)에서
세션을 끊김 없이 X초 동안 유지한 비율이다.

| 방법 | 세션 완주율 |
|---|---:|
| GCN-PPO (시드 0, 1, 2 평균) | 92.56% (표본 표준편차 1.66%p) |
| Lifetime (남은 접촉 시간으로 링크를 거른 최단 홉 경로) | 83.53% |
| ISASR/Remaining (남은 접촉 시간 기반 링크 비용) | 82.11% |
| GCN-PPO에서 남은 접촉 시간 관측 제거 (시드 0) | 78.22% |
| GCN-PPO에서 링크 설정 비용 보상 제거 (시드 0) | 83.64% |
| 제약 충족 해 구성기 (전체 경로를 한 번에 탐색) | 100% |

제약 충족 해 구성기는 같은 터미널, 설정 지연, 링크 전환 규칙을 지키면서 3,600개를
모두 완주한다. 세션이 하나뿐이면 완주할 수 있는 경로가 항상 존재한다는 뜻이다.
다만 이 구성기는 경로 길이나 링크 설정 횟수를 최적화하지 않는다.
3단계는 그래서 여러 세션이 한정된 터미널과 링크 용량을 두고 경쟁하는 문제로 넓힌다.

전체 표와 통계 검정은 [`results_ppo_gcn/priority_evidence_2026-09-04/RESULTS.md`](results_ppo_gcn/priority_evidence_2026-09-04/RESULTS.md)에 있다.

## 구성

```text
leo_session/       위성망 환경: 궤도, 접촉 구간, 초기 토폴로지, 기준 경로 방법, 세션 실행, 보상
ppo_gcn/           GCN-PPO: 희소 그래프 관측, actor-critic 모델, 학습
scripts/           접촉 구간 생성, 기준 방법 실행, 학습, 평가, 표와 그림 생성, 재현 검증
tests/             궤도, 토폴로지, 경로, 보상 검사
cache/             접촉 구간 캐시
results_paper/     기준 방법 결과(테스트 3,600개), 설정 지연 민감도, 제약 충족 해 검증
results_ppo_gcn/   GCN-PPO 체크포인트와 평가 결과(시드 0/1/2, 요소 제거 실험)
docs/              설계, 보상, 평가 지표, 실험 절차, 참고문헌, 재현 방법
```

## 실행

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m pytest -q tests                      # 테스트
.venv/bin/python scripts/verify_reproduction_bundle.py   # 입력, 체크포인트, 핵심 결과 검증
.venv/bin/python scripts/run_cpu_baselines.py            # 기준 방법 시험 실행(작은 규모)
```

테스트 요청 3,600개 전체 재현과 GCN-PPO 학습 명령은 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)에 있다.
설계 전반은 [`docs/TECHNICAL_OVERVIEW_KO.md`](docs/TECHNICAL_OVERVIEW_KO.md)와 [`docs/DESIGN.md`](docs/DESIGN.md)에 있다.

## 함께한 사람

2단계의 환경과 GCN-PPO 코드는 안태건(WCNC 2027 투고 논문 공동 제1저자)과 함께 개발했다.
지도교수는 주창희 교수다. 고려대학교 컴퓨터학과.
