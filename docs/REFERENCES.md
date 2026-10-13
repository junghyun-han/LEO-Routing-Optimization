# v3 parameter and algorithm references

## Reference-fixed physical values

| Item | Value | Source | Locator |
|---|---:|---|---|
| Satellites | 1,584 | Bhattacharjee et al., IEEE TAES 2024 | Table III, PDF p. 9 |
| Orbital planes | 24 | same | Table III, PDF p. 9 |
| Satellites per plane | 66 | same | Table III, PDF p. 9 |
| Altitude | 550 km | same | Table III and Sec. V-A, PDF p. 9 |
| Inclination | 53 degrees | same | Table III and Sec. V-A, PDF p. 9 |
| LISL range | 1,500 km | same | Table III and Sec. V-A, PDF p. 9 |
| Walker phasing | F=11 | Yang et al., Aerospace 2024 | Sec. 2, `(1584/24/11, 550 km, 53 deg)` |
| Earth radius | 6,378.137 km | WGS 84 | defining constants |
| Earth GM | 398,600.4418 km^3/s^2 | WGS 84 | defining constants |
| Light speed | 299,792.458 km/s | SI Brochure | exact defining constant |

Bhattacharjee's 1,500 km is used only as a published reference-simulation
parameter. v3 does not claim that it is an official Starlink hardware limit.

The low/medium/high endpoint cases are New York-London, New York-Istanbul and
New York-Hanoi, following Bhattacharjee et al., *Laser Inter-Satellite Link
Setup Delay: Quantification, Impact, and Tolerable Value* (2023), Secs. IV-V.
The city-centre coordinates merely locate endpoints; they are not operational
gateway locations.

## Reference-informed model value

Dynamic setup is modeled as deterministic 20 seconds. Smutny et al. report
routine full data communication in less than 20 seconds after optimization and
describe the interval as including spatial acquisition and laser phase locking
(Proc. SPIE 7199, PDF pp. 4 and 7, DOI 10.1117/12.812209). Therefore 20 seconds
is a conservative approximation, not a universal measured constant.

## Explicit model assumptions without a claimed hardware source

- two persistent plus one dynamic terminal per satellite;
- adjacent-plane dynamic candidates;
- break-before-make reconfiguration;
- deterministic plane-balanced initial matching;
- the controller knows `X` when the request arrives;
- uninterrupted connectivity for `X` defines completion.
- parallel setup of terminal-disjoint dynamic ISLs on one centrally planned
  route.

The 1,000 km GSL range is taken from Bhattacharjee et al. 2024 Table III as a
reference simulation value, not as an official commercial terminal range.

## Algorithm sources

PRP follows H. Uzunalioglu et al., *Wireless Networks* 6 (2000), Sec. 3,
Eqs. (1)-(2) and algorithm steps 1-4. The original evaluation uses minimum-hop
Dijkstra with unit ISL cost and discusses `p=0.9` as an empirical operating
point in Sec. 5. In its three-minute mean-call example, the route-usage CDF
reaches 0.9 at approximately three minutes; v3 therefore uses
`T_tr = 180 s` for the PRP baseline. This baseline intentionally does not use
the known deterministic `X`; that difference is part of the comparison.

The remaining-contact heuristic follows the stability-and-activeness structure
of ISASR in Bhattacharjee et al., Algorithm 3 and Eqs. (14)-(15). Any necessary
change for sessions or the 2+1 terminal model will be named `adapted ISASR` and
listed explicitly rather than presented as the original algorithm.

## Numerical settings

The 5-second scan interval and root tolerance are not physical claims. A
2.5-second regeneration produced the same 14,256 ISL contact intervals; the
maximum absolute contact-boundary difference from the 5-second scan was
`3.95e-7 s`. The relaxed upper-bound graph was also checked at 2.5-second
resolution over one orbit: all 6,888 `(time, ground-pair)` checks remained
connected. These checks support the selected numerical resolution but do not
turn the relaxed upper bound into an achievable oracle.

## Controlled experiment factors

The final duration set is `X = {20, 60, 180, 300, 600} s`. No literature
traffic distribution is claimed. There are 240 deterministic mid-bin request
times over one orbit for each ground pair and X. The 120-offset pilot was
compared with the final 240-offset run and changed any aggregate method/X
completion rate by at most 1.3 percentage points. Superseded pilot output was
removed from the live final workspace.
