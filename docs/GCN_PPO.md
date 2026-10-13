# 현재 PPO-GCN 요약

전체 설명, 입력·출력, 학습 예, 설정, frozen 결과와 실행 명령은
`TECHNICAL_OVERVIEW_KO.md` 8–15절을 따른다.

- 현재 제안 모델: GCN 3 layers, hidden 128, seeds 0/1/2, seed당 1M relay steps
- 입력: 1,584x13 node, Ex8 graph edge, 1,584x8 action, action mask
- 출력: valid neighbor별 masked logit과 scalar state value
- PPO: rollout 1024, minibatch 64, epochs 4, LR 0.003, gamma 0.99,
  lambda 0.95, clip 0.2, value clip 1.0, target KL 0.03
- frozen 3,600 strict completion: seed 0/1/2 = 94.44%, 91.89%, 91.33%
- 3-seed mean ± sample SD: 92.56% ± 1.66 percentage points
- controlled ablations: No-RCT 78.22%, No-PAT 83.64% (seed 0)
- 평가 시 validation이 아니라 frozen request 3,600개 전체를 사용한다.
- GPU 작업은 `CUDA_VISIBLE_DEVICES=2`로 물리 GPU 2만 사용한다.

현재 checkpoint와 원본 결과:

`results_ppo_gcn/long_run_2026-09-04/gcn3_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/`

`results_ppo_gcn/priority_evidence_2026-09-04/`

새 환경에서의 설치·학습·평가·checkpoint 검증 명령은
`REPRODUCE.md`를 따른다.
