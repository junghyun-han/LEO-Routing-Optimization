# Current-reward Cost-Greedy

Cost-Greedy는 PPO와 같은 sequential route builder, valid candidate, 네 relay 항을
사용한다.

`r = progress - 0.05*link_length - 3*lifetime_shortfall - 0.15*new_PAT`

현재 후보 중 이 값이 가장 큰 neighbor를 결정론적으로 고른다. 동률이면 신규
dynamic 여부와 satellite ID로 결정한다. look-ahead, 학습, 경로 성공/실패 보상,
service 보상, session completion 보상은 없다.

현재 코드와 같은 reward로 frozen 3,600개를 재평가한 결과:

- setup: 3,204/3,600 = 89.00%
- strict completion: 2,903/3,600 = 80.64%
- availability: 86.23%
- mean interruption: 13.47 s
- mean reconfigurations: 0.656
- mean new dynamic ISLs: 0.556
- established-session route mean: 35.31 hops, 25,832.07 km

원본: `results_paper/cost_greedy_current_reward_frozen3600/`  
X별 비교: `results_ppo_gcn/long_run_2026-09-04/gcn3_h128_v2_exact_route_reward_seed0_1m_mb64_gpu2/eval_full_test/comparison_tables/RESULTS_BY_X.md`
