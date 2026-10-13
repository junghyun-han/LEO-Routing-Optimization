# Frozen test 평가 원칙

1. 논문 성능 수치와 baseline 비교에는 CPU baseline과 동일한 frozen test 3,600개 요청만 사용한다.
2. 180개 validation은 학습/checkpoint 내부 진단 전용이며 논문 성능 근거로 사용하지 않는다.
3. 30개 또는 다른 표본 subset은 경로 정성분석·디버깅 전용이며 논문 수치로 인용하지 않는다.
4. frozen test 성능이 낮을 때 validation 성능을 근거로 변명하거나 우수성을 주장하지 않는다.
5. 성공 기준은 동일 환경·동일 request ID에서 PPO-GCN strict completion이 Lifetime 83.53% (3007/3600)와 ISASR/Remaining 82.11% (2956/3600)를 모두 넘는 것이다.
6. 이 기준을 넘지 못한 PPO-GCN 설정은 실패한 설정으로 기록하고 full-test 실패 전체를 분석한다.
