# 운영 조건 6가지와 근거

LLM 라우팅 에이전트가 텍스트로 받는 조건이다. 실제 위성통신 운영에서 자주 다루는
문제에서 골랐고, 각 조건은 WCNC 시뮬레이터 안에서 정확히 정의된다
(`multisession/conditions.py`). 반드시 지킬 조건은 문제에서 해당 요소를 빼거나 제약으로
넣고, 가능하면 지킬 조건은 목표의 한 단계로 넣는다.

| # | 조건 | 지시문 예 | 정식화 | 종류 |
|---|---|---|---|---|
| 1 | 위성·링크 고장 | "위성 812와 845가 고장 났어" | 그 위성의 모든 링크와 지상 접속 제거 | 반드시 |
| 2 | 트래픽 핫스팟 | "북대서양 상공은 혼잡하니 가능하면 피해" | 목표 단계 추가: 핫스팟 위성 통과 횟수 최소 | 가능하면 |
| 3 | 지역 정책 | "중동 상공은 지나가지 마" / "하노이 세션은 중동을 거쳐 가" | 지역 위성 제거 / 두 층 네트워크로 경유 강제 | 반드시 |
| 4 | 지상국 장애 | "런던 안테나 하나가 비 때문에 안 돼" | 그 지상국의 안테나 수 감소 | 반드시 |
| 5 | 재구성 제한 | "링크 재설정은 2번까지만" | 새 동적 링크 수 ≤ m | 반드시 |
| 6 | 우선 트래픽과 최소 보장 | "하노이를 먼저 최대한" / "도시마다 최소 3개" | 도착지별 수용을 먼저 최대화 / 도착지별 하한 | 목표·반드시 |

지역은 결정 시각의 위성 직하점 위도·경도로 판정하는 대략적인 상자다(국경이 아니다).
목록은 `REGIONS`에 고정되어 있다.

## 근거

1. **위성·링크 고장.** 링크 고장 시 우회 경로를 찾는 분산 방식(Distributed Detour Routing,
   [Sensors 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10708603/)), 간헐적 링크 고장에
   견디는 계층 라우팅(BlockFLEX, [arXiv 2512.09453](https://arxiv.org/abs/2512.09453)).
2. **트래픽 핫스팟.** 지상 수요가 고르지 않아 지리적 핫스팟이 생기고 최단 경로가 이를
   악화시킨다(DLR, [arXiv 2209.05984](https://arxiv.org/abs/2209.05984)). 핫스팟은 시간에 따라
   옮겨 다닌다([arXiv 2601.02537](https://arxiv.org/abs/2601.02537)).
3. **지역 정책.** 위성이 움직여 지리 기반 데이터 주권 규칙을 지키기 어렵다는 문제를 다룬
   특허(데이터 주권 정책에 따른 위성망 라우팅, USPTO 출원 12574108)와, 지상국 설치의 국가 간
   규제·지정학 위험을 위성 간 다중 홉으로 우회한다는 IETF 초안
   ([draft-miao-rtgwg-satellite-routing-reqs-00](https://ftp.kaist.ac.kr/ietf/draft-miao-rtgwg-satellite-routing-reqs-00.html)).
   특정 지역을 제외하는 공개 라우팅 알고리즘은 찾지 못했으므로 동기 수준의 근거로 쓴다.
4. **지상국 장애.** Q/V 대역 피더 링크는 가용도를 맞추려면 게이트웨이 다이버시티가 필요하다
   (Gharanjik et al., [arXiv 1305.1434](https://arxiv.org/abs/1305.1434)). 저앙각·강우 감쇠로
   비정지궤도 Ka 링크에 감쇠 대책이 필수다
   ([URSI GASS 2021](https://www.ursi.org/proceedings/procGA21/papers/URSIGASS2021-Fr-F07-PM3-1.pdf)).
5. **재구성 제한.** 광 위성 간 링크를 새로 잡으려면 PAT(pointing, acquisition, tracking) 지연이
   든다(Gerard, Fraire, Cespedes, *Modeling pointing, acquisition, and tracking delays in free-space
   optical satellite networks*, IEEE WiSEE 2025; Bhattacharjee et al., *Laser inter-satellite link setup
   delay: Quantification, impact, and tolerable value*, IEEE WCNC 2023). WCNC 2027 투고 논문의 20초
   설정 지연도 이 근거를 따른다.
6. **우선 트래픽.** 최소 홉·최소 지연 라우팅이 핫스팟을 혼잡하게 하고 고우선 호가 저우선
   트래픽 경로에 실린다는 지적과 우선순위 부하 분산(Jain et al.,
   [WUSTL](https://research.engineering.wustl.edu/~jain/papers/ftp/aero04.pdf)), 지연 기준과
   우선 분류(DSROQ, [arXiv 2508.21047](https://arxiv.org/abs/2508.21047)).

## 관련 LLM 연구

- STR-Agent: 자연어 요청을 구조화된 라우팅 의미로 바꾸고 혼잡과 과거 결과로 정책을 조정하는
  LEO QoS 라우팅 에이전트([arXiv 2609.20347](https://arxiv.org/abs/2609.20347)).
- Validated Intent Compilation for Constrained Routing in LEO Mega-Constellations: 의도를 제약
  라우팅으로 바꾸는 검증된 파이프라인([arXiv 2604.07264](https://arxiv.org/abs/2604.07264)).

이 프로젝트는 같은 흐름에서 (1) 재구성 터미널을 여러 세션이 나눠 쓰는 문제, (2) CP-SAT 정답을
기준으로 한 평가, (3) 도구 유무와 모델 종류에 따른 비교를 다룬다.
