"""All v3 numerical settings, with an explicit provenance category.

The four categories are deliberately kept in code so that a model assumption
cannot accidentally be described as a measured hardware fact in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


ParameterCategory = Literal[
    "REFERENCE_FIXED",
    "MODEL_ASSUMPTION",
    "EXPERIMENT_FACTOR",
    "NUMERICAL_PARAMETER",
]


@dataclass(frozen=True)
class Reference:
    key: str
    citation: str
    url: str
    locator: str


REFERENCES: Dict[str, Reference] = {
    "BHATTACHARJEE_2024": Reference(
        key="BHATTACHARJEE_2024",
        citation=(
            "D. Bhattacharjee et al., 'On-Demand Routing in LEO "
            "Mega-Constellations with Dynamic Laser Inter-Satellite Links,' "
            "IEEE Transactions on Aerospace and Electronic Systems, 2024."
        ),
        url="https://doi.org/10.1109/TAES.2024.3415571",
        locator="Table III and Sec. V-A (PDF p. 9)",
    ),
    "BHATTACHARJEE_2023": Reference(
        key="BHATTACHARJEE_2023",
        citation=(
            "D. Bhattacharjee et al., 'Laser Inter-Satellite Link Setup "
            "Delay: Quantification, Impact, and Tolerable Value,' 2023."
        ),
        url="https://arxiv.org/abs/2301.05285",
        locator=(
            "Sec. IV-V; New York-London, New York-Istanbul, and "
            "New York-Hanoi low/medium/high-distance cases"
        ),
    ),
    "YANG_2024": Reference(
        key="YANG_2024",
        citation=(
            "H. Yang et al., 'A Flexible Topology Control Strategy for "
            "Mega-Constellations via Inter-Satellite Links Based on Dynamic "
            "Link Optimization,' Aerospace 11(7), 510, 2024."
        ),
        url="https://doi.org/10.3390/aerospace11070510",
        locator="Sec. 2, Walker arrangement (1584/24/11, 550 km, 53 deg)",
    ),
    "SMUTNY_2009": Reference(
        key="SMUTNY_2009",
        citation=(
            "B. Smutny et al., '5.6 Gbps optical intersatellite communication "
            "link,' Proc. SPIE 7199, 719906, 2009."
        ),
        url="https://doi.org/10.1117/12.812209",
        locator="PDF pp. 4 and 7; routine full communication in <20 s",
    ),
    "UZUNALIOGLU_2000": Reference(
        key="UZUNALIOGLU_2000",
        citation=(
            "H. Uzunalioglu et al., 'A routing algorithm for connection-"
            "oriented Low Earth Orbit (LEO) satellite networks with dynamic "
            "connectivity,' Wireless Networks 6, 181-190, 2000."
        ),
        url="https://ianakyildiz.com/bwn/papers/2000/j3.pdf",
        locator="Sec. 3, Eqs. (1)-(2), algorithm steps 1-4; Sec. 5 p=0.9",
    ),
    "WGS84": Reference(
        key="WGS84",
        citation="NGA.STND.0036_1.0.0_WGS84, World Geodetic System 1984.",
        url="https://earth-info.nga.mil/",
        locator="WGS 84 defining constants",
    ),
    "SI_2019": Reference(
        key="SI_2019",
        citation="BIPM, The International System of Units (SI), 9th ed.",
        url="https://www.bipm.org/en/publications/si-brochure",
        locator="speed of light exact defining constant",
    ),
}


@dataclass(frozen=True)
class ParameterRecord:
    value: Any
    unit: str
    category: ParameterCategory
    reference: Optional[str]
    rationale: str

    def validate(self) -> None:
        if self.category == "REFERENCE_FIXED" and not self.reference:
            raise ValueError(f"REFERENCE_FIXED value {self.value!r} lacks a source")
        if self.reference and self.reference not in REFERENCES:
            raise ValueError(f"unknown reference key: {self.reference}")


PARAMETER_MANIFEST: Dict[str, ParameterRecord] = {
    "total_satellites": ParameterRecord(
        1584, "satellites", "REFERENCE_FIXED", "BHATTACHARJEE_2024",
        "Reference Starlink Phase-I version-2 simulation scale.",
    ),
    "orbital_planes": ParameterRecord(
        24, "planes", "REFERENCE_FIXED", "BHATTACHARJEE_2024",
        "Reference simulation constellation.",
    ),
    "satellites_per_plane": ParameterRecord(
        66, "satellites/plane", "REFERENCE_FIXED", "BHATTACHARJEE_2024",
        "Reference simulation constellation.",
    ),
    "walker_phasing": ParameterRecord(
        11, "dimensionless", "REFERENCE_FIXED", "YANG_2024",
        "Completes the Walker T/P/F definition missing from Table III.",
    ),
    "altitude_km": ParameterRecord(
        550.0, "km", "REFERENCE_FIXED", "BHATTACHARJEE_2024",
        "Circular reference orbit altitude.",
    ),
    "inclination_deg": ParameterRecord(
        53.0, "degree", "REFERENCE_FIXED", "BHATTACHARJEE_2024",
        "Reference orbit inclination.",
    ),
    "lisl_range_km": ParameterRecord(
        1500.0, "km", "REFERENCE_FIXED", "BHATTACHARJEE_2024",
        "Reference simulation range, not a claimed Starlink hardware limit.",
    ),
    "gsl_range_km": ParameterRecord(
        1000.0, "km", "REFERENCE_FIXED", "BHATTACHARJEE_2024",
        "Reference simulation GS range in Table III, not a hardware claim.",
    ),
    "earth_radius_km": ParameterRecord(
        6378.137, "km", "REFERENCE_FIXED", "WGS84",
        "Spherical Earth approximation uses the WGS-84 semi-major axis.",
    ),
    "earth_mu_km3_s2": ParameterRecord(
        398600.4418, "km^3/s^2", "REFERENCE_FIXED", "WGS84",
        "WGS-84 geocentric gravitational constant.",
    ),
    "speed_of_light_km_s": ParameterRecord(
        299792.458, "km/s", "REFERENCE_FIXED", "SI_2019",
        "Exact SI defining constant converted to km/s.",
    ),
    "dynamic_setup_s": ParameterRecord(
        20.0, "s", "MODEL_ASSUMPTION", "SMUTNY_2009",
        "Conservative deterministic approximation from a reported <20 s full establishment.",
    ),
    "persistent_terminals": ParameterRecord(
        2, "terminals/satellite", "MODEL_ASSUMPTION", None,
        "Professor-suggested model: forward/backward intra-plane links.",
    ),
    "dynamic_terminals": ParameterRecord(
        1, "terminal/satellite", "MODEL_ASSUMPTION", None,
        "Professor-suggested model for PAT-sensitive reconfiguration.",
    ),
    "prp_target_probability": ParameterRecord(
        0.9, "probability", "REFERENCE_FIXED", "UZUNALIOGLU_2000",
        "Empirical PRP setting discussed in the original evaluation.",
    ),
    "prp_route_hold_s": ParameterRecord(
        180.0, "s", "REFERENCE_FIXED", "UZUNALIOGLU_2000",
        "Published p=0.9 example: the route-usage CDF reaches 0.9 at 3 min.",
    ),
    "contact_scan_step_s": ParameterRecord(
        5.0, "s", "NUMERICAL_PARAMETER", None,
        "Only brackets contact transitions; roots are refined continuously and convergence is tested.",
    ),
    "root_tolerance_s": ParameterRecord(
        1.0e-6, "s", "NUMERICAL_PARAMETER", None,
        "Numerical root tolerance, verified by convergence tests.",
    ),
    "oracle_sample_step_s": ParameterRecord(
        5.0, "s", "NUMERICAL_PARAMETER", None,
        "Sampling interval for the deliberately relaxed physical upper bound.",
    ),
}


@dataclass(frozen=True)
class EnvironmentConfig:
    total_satellites: int = 1584
    orbital_planes: int = 24
    satellites_per_plane: int = 66
    walker_phasing: int = 11
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    lisl_range_km: float = 1500.0
    gsl_range_km: float = 1000.0
    earth_radius_km: float = 6378.137
    earth_mu_km3_s2: float = 398600.4418
    speed_of_light_km_s: float = 299792.458
    dynamic_setup_s: float = 20.0
    persistent_terminals: int = 2
    dynamic_terminals: int = 1
    prp_target_probability: float = 0.9
    prp_route_hold_s: float = 180.0
    contact_scan_step_s: float = 5.0
    root_tolerance_s: float = 1.0e-6
    oracle_sample_step_s: float = 5.0
    _manifest: Dict[str, ParameterRecord] = field(
        default_factory=lambda: PARAMETER_MANIFEST, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.total_satellites != self.orbital_planes * self.satellites_per_plane:
            raise ValueError("total_satellites must equal planes * satellites_per_plane")
        if self.persistent_terminals != 2 or self.dynamic_terminals != 1:
            raise ValueError("v3 paper model is fixed to 2 persistent + 1 dynamic terminal")
        for item in self._manifest.values():
            item.validate()

    @property
    def orbital_radius_km(self) -> float:
        return self.earth_radius_km + self.altitude_km

    def provenance(self, name: str) -> ParameterRecord:
        return self._manifest[name]


DEFAULT_CONFIG = EnvironmentConfig()
