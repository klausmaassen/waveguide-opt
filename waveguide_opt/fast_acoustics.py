from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .geometry import estimated_mouth_diameter, profile_points
from .models import Candidate
from .piston import SPEED_OF_SOUND, piston_db


FREQS_HZ = [
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0,
]
ANGLES_DEG = [float(angle) for angle in range(0, 181, 5)]
ON_AXIS_ANGLE = 0.0
ANNULUS_SAMPLES = 18
PTT8_DIAMETER_MM = 173.0
CROSSOVER_MATCH_FREQ_RANGE = (1000.0, 2500.0)
CROSSOVER_CENTER_HZ = 1600.0
CROSSOVER_MATCH_ANGLES = (30.0, 45.0, 60.0)
BEAMWIDTH_BAND = (1300.0, 6300.0)
DEFAULT_FAST_SCORE_WEIGHTS = {
    "constant_directivity": 0.30,
    "ptt8_crossover_match": 0.25,
    "off_axis_smoothness": 0.20,
    "response_ripple": 0.15,
    "resonance_penalty": 0.10,
}


@dataclass(frozen=True)
class Annulus:
    radius_m: float
    weight: float
    phase_m: float


@dataclass(frozen=True)
class FastAcousticResult:
    total: float
    constant_directivity: float
    ptt8_crossover_match: float
    off_axis_smoothness: float
    response_ripple: float
    resonance_penalty: float
    directivity_index_smoothness: float
    mouth_reflection: float
    throat_transition: float
    beamwidth_2000: float | None
    beamwidth_4000: float | None
    freqs_hz: list[float]
    angles_deg: list[float]
    polar_db: list[list[float]]
    on_axis_db: list[float]
    beamwidth_deg: list[float | None]
    directivity_index_db: list[float]
    ptt8_target_30_db: list[float]
    ptt8_target_60_db: list[float]

    def as_dict(self, include_curves: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "total": self.total,
            "constant_directivity": self.constant_directivity,
            "ptt8_crossover_match": self.ptt8_crossover_match,
            "off_axis_smoothness": self.off_axis_smoothness,
            "response_ripple": self.response_ripple,
            "resonance_penalty": self.resonance_penalty,
            "directivity_index_smoothness": self.directivity_index_smoothness,
            "mouth_reflection": self.mouth_reflection,
            "throat_transition": self.throat_transition,
            "beamwidth_2000": self.beamwidth_2000,
            "beamwidth_4000": self.beamwidth_4000,
        }
        if include_curves:
            data.update(
                {
                    "freqs_hz": self.freqs_hz,
                    "angles_deg": self.angles_deg,
                    "polar_db": self.polar_db,
                    "on_axis_db": self.on_axis_db,
                    "beamwidth_deg": self.beamwidth_deg,
                    "directivity_index_db": self.directivity_index_db,
                    "ptt8_target_30_db": self.ptt8_target_30_db,
                    "ptt8_target_60_db": self.ptt8_target_60_db,
                }
            )
        return data


def _j0(x: float) -> float:
    ax = abs(x)
    if ax < 1e-10:
        return 1.0
    if ax > 18.0:
        return math.sqrt(2.0 / (math.pi * ax)) * math.cos(ax - math.pi / 4.0)
    term = 1.0
    total = 1.0
    xx = (x * x) / 4.0
    for m in range(1, 36):
        term *= -xx / (m * m)
        total += term
        if abs(term) < 1e-13:
            break
    return total


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _profile_slopes(candidate: Candidate) -> tuple[float, float, float]:
    points = profile_points(candidate, 30)
    slopes: list[float] = []
    for a, b in zip(points, points[1:]):
        dz = max(b["z"] - a["z"], 1e-9)
        slopes.append((b["r"] - a["r"]) / dz)
    curvatures = [slopes[idx + 1] - slopes[idx] for idx in range(len(slopes) - 1)]
    throat_slope = slopes[0] if slopes else 0.0
    mouth_slope = slopes[-1] if slopes else 0.0
    curvature_rms = _rms(curvatures)
    return throat_slope, mouth_slope, curvature_rms


def _reflection_indicators(candidate: Candidate) -> tuple[float, float]:
    throat_slope, mouth_slope, curvature_rms = _profile_slopes(candidate)
    throat_angle = math.atan(max(throat_slope, 0.0))
    mouth_angle = math.atan(max(mouth_slope, 0.0))
    throat_transition = max(0.0, throat_angle - math.radians(18.0)) / math.radians(28.0)
    throat_transition += min(1.0, curvature_rms * 2.5) * 0.35
    mouth_reflection = max(0.0, mouth_angle - math.radians(32.0)) / math.radians(45.0)
    mouth_reflection += abs(candidate.term_s - 0.78) * 0.35
    mouth_reflection += max(0.0, candidate.term_q - 0.997) * 90.0
    return min(2.0, mouth_reflection), min(2.0, throat_transition)


def _aperture_annuli(mouth_radius_m: float, taper: float, edge_phase_mm: float) -> list[Annulus]:
    annuli: list[Annulus] = []
    for idx in range(ANNULUS_SAMPLES):
        rho0 = idx / ANNULUS_SAMPLES
        rho1 = (idx + 1) / ANNULUS_SAMPLES
        rho = (rho0 + rho1) / 2.0
        area_weight = rho1 * rho1 - rho0 * rho0
        amp = math.exp(-taper * rho * rho)
        annuli.append(
            Annulus(
                radius_m=rho * mouth_radius_m,
                weight=area_weight * amp,
                phase_m=(edge_phase_mm / 1000.0) * rho * rho,
            )
        )
    return annuli


def _axisym_boundary_polar(candidate: Candidate) -> tuple[dict[float, list[float]], list[float]]:
    """Fast axisymmetric boundary proxy using ring-source quadrature.

    This is the inexpensive optimizer model. The API is intentionally shaped like
    a BEM result so it can later be cross-checked against Bempp-cl.
    """
    mouth_radius_m = estimated_mouth_diameter(candidate) / 2000.0
    length_m = max(candidate.length, 1e-6) / 1000.0
    _, mouth_slope, curvature_rms = _profile_slopes(candidate)
    mouth_angle = math.atan(max(mouth_slope, 0.0))
    edge_phase_mm = (mouth_radius_m * 1000.0) * math.sin(mouth_angle) * 0.24
    edge_phase_mm += min(7.0, curvature_rms * 12.0)
    taper = max(0.0, min(0.9, (0.82 - candidate.os_k) * 0.45 + (0.78 - candidate.term_s) * 0.55))

    annuli = _aperture_annuli(mouth_radius_m, taper, edge_phase_mm)

    mouth_reflection, throat_transition = _reflection_indicators(candidate)
    reflection = min(0.34, 0.07 + 0.08 * mouth_reflection + 0.05 * throat_transition)
    polars: dict[float, list[float]] = {}
    on_axis_db: list[float] = []

    for freq in FREQS_HZ:
        k = 2.0 * math.pi * freq / SPEED_OF_SOUND
        ripple_phase = 2.0 * k * length_m + math.pi * candidate.term_s
        ripple_mag = abs(1.0 + reflection * complex(math.cos(ripple_phase), math.sin(ripple_phase)))
        on_axis_db.append(20.0 * math.log10(max(ripple_mag, 1e-9)))
        row: list[float] = []
        for angle in ANGLES_DEG:
            theta = math.radians(angle)
            sin_angle = math.sin(theta)
            real = 0.0
            imag = 0.0
            for annulus in annuli:
                bessel = _j0(k * annulus.radius_m * sin_angle)
                phase = k * annulus.phase_m
                real += annulus.weight * bessel * math.cos(phase)
                imag += annulus.weight * bessel * math.sin(phase)
            direction = _finite_baffle_weight(freq, angle)
            row.append(20.0 * math.log10(max(math.hypot(real, imag) * direction, 1e-9)))
        axis = row[ANGLES_DEG.index(ON_AXIS_ANGLE)]
        polars[freq] = [value - axis for value in row]
    return polars, on_axis_db


def _finite_baffle_weight(freq: float, angle: float) -> float:
    """Approximate front-to-rear wrap for the fast optimizer model.

    The ring-source term describes the circular aperture. This factor keeps the
    0..180 map from behaving like either a mirrored free-space piston or a hard
    infinite baffle with an immediate 90 degree null.
    """
    theta = math.radians(min(angle, 90.0))
    side_floor = 0.10 + 0.55 / (1.0 + (freq / 1700.0) ** 1.35)
    front = side_floor + (1.0 - side_floor) * (max(math.cos(theta), 0.0) ** 0.85)
    if angle <= 90.0:
        return front

    rear_angle = angle - 90.0
    rear_floor = 0.018 + 0.55 / (1.0 + (freq / 1300.0) ** 1.6)
    return rear_floor + (side_floor - rear_floor) * math.exp(-rear_angle / 35.0)


def _beamwidth(row: list[float]) -> float | None:
    target = -6.0
    for idx in range(1, len(ANGLES_DEG)):
        if ANGLES_DEG[idx] > 90.0:
            break
        if row[idx] <= target:
            a0, a1 = ANGLES_DEG[idx - 1], ANGLES_DEG[idx]
            y0, y1 = row[idx - 1], row[idx]
            if y1 == y0:
                return 2.0 * a1
            t = (target - y0) / (y1 - y0)
            return 2.0 * (a0 + t * (a1 - a0))
    return None


def _directivity_index(row: list[float]) -> float:
    weighted_power = 0.0
    weight_sum = 0.0
    for idx, angle in enumerate(ANGLES_DEG):
        theta = math.radians(angle)
        if idx == 0:
            delta = math.radians((ANGLES_DEG[1] - ANGLES_DEG[0]) / 2.0)
        elif idx == len(ANGLES_DEG) - 1:
            delta = math.radians((ANGLES_DEG[-1] - ANGLES_DEG[-2]) / 2.0)
        else:
            delta = math.radians((ANGLES_DEG[idx + 1] - ANGLES_DEG[idx - 1]) / 2.0)
        weight = max(0.0, math.sin(theta)) * delta
        weighted_power += (10.0 ** (row[idx] / 10.0)) * weight
        weight_sum += weight
    avg_power = weighted_power / max(weight_sum, 1e-9)
    return -10.0 * math.log10(max(avg_power, 1e-12))


def _local_roughness(rows: list[list[float]]) -> float:
    terms: list[float] = []
    for row in rows:
        for idx in range(1, len(row) - 1):
            terms.append(row[idx - 1] - 2.0 * row[idx] + row[idx + 1])
    for freq_idx in range(1, len(rows) - 1):
        for angle_idx in range(1, len(ANGLES_DEG)):
            terms.append(rows[freq_idx - 1][angle_idx] - 2.0 * rows[freq_idx][angle_idx] + rows[freq_idx + 1][angle_idx])
    return _rms(terms) / 2.0


def evaluate_fast(candidate: Candidate) -> FastAcousticResult:
    polars, on_axis_db = _axisym_boundary_polar(candidate)
    rows = [polars[freq] for freq in FREQS_HZ]
    beamwidths = [_beamwidth(row) for row in rows]
    bw_f1, bw_f2 = BEAMWIDTH_BAND
    valid_beamwidths = [bw for bw, freq in zip(beamwidths, FREQS_HZ) if bw is not None and bw_f1 <= freq <= bw_f2]
    directivity_indices = [_directivity_index(row) for row in rows]

    # Constant directivity: low variation of beamwidth and DI over the main waveguide band.
    normalized_bw = [math.log(max(bw, 1.0)) for bw in valid_beamwidths]
    constant_directivity = _std(normalized_bw) * 3.0
    directivity_index_smoothness = _std([directivity_indices[idx + 1] - directivity_indices[idx] for idx in range(len(directivity_indices) - 1)]) / 1.5
    constant_directivity += 0.45 * directivity_index_smoothness

    # PTT8 crossover match: compare relative polar response to 173 mm piston around the crossover band.
    crossover_terms: list[float] = []
    crossover_f1, crossover_f2 = CROSSOVER_MATCH_FREQ_RANGE
    for freq, row in polars.items():
        if not crossover_f1 <= freq <= crossover_f2:
            continue
        weight = math.exp(-(math.log2(freq / CROSSOVER_CENTER_HZ) ** 2) / 0.7)
        for angle in CROSSOVER_MATCH_ANGLES:
            idx = ANGLES_DEG.index(angle)
            target = piston_db(freq, angle, PTT8_DIAMETER_MM)
            crossover_terms.append(weight * (row[idx] - target) ** 2)
    ptt8_crossover_match = math.sqrt(sum(crossover_terms) / max(len(crossover_terms), 1))

    off_axis_smoothness = _local_roughness(rows)
    response_ripple = _std(on_axis_db) / 2.5
    resonance_terms: list[float] = []
    for idx in range(1, len(on_axis_db) - 1):
        local = abs(on_axis_db[idx] - (on_axis_db[idx - 1] + on_axis_db[idx + 1]) / 2.0)
        resonance_terms.append(max(0.0, local - 0.7))
    resonance_penalty = _rms(resonance_terms) / 1.5
    mouth_reflection, throat_transition = _reflection_indicators(candidate)

    total = sum(
        [
            DEFAULT_FAST_SCORE_WEIGHTS["constant_directivity"] * constant_directivity,
            DEFAULT_FAST_SCORE_WEIGHTS["ptt8_crossover_match"] * ptt8_crossover_match,
            DEFAULT_FAST_SCORE_WEIGHTS["off_axis_smoothness"] * off_axis_smoothness,
            DEFAULT_FAST_SCORE_WEIGHTS["response_ripple"] * response_ripple,
            DEFAULT_FAST_SCORE_WEIGHTS["resonance_penalty"] * resonance_penalty,
        ]
    )

    return FastAcousticResult(
        total=total,
        constant_directivity=constant_directivity,
        ptt8_crossover_match=ptt8_crossover_match,
        off_axis_smoothness=off_axis_smoothness,
        response_ripple=response_ripple,
        resonance_penalty=resonance_penalty,
        directivity_index_smoothness=directivity_index_smoothness,
        mouth_reflection=mouth_reflection,
        throat_transition=throat_transition,
        beamwidth_2000=_beamwidth(polars[2000.0]),
        beamwidth_4000=_beamwidth(polars[4000.0]),
        freqs_hz=FREQS_HZ,
        angles_deg=ANGLES_DEG,
        polar_db=rows,
        on_axis_db=on_axis_db,
        beamwidth_deg=beamwidths,
        directivity_index_db=directivity_indices,
        ptt8_target_30_db=[piston_db(freq, 30.0, PTT8_DIAMETER_MM) for freq in FREQS_HZ],
        ptt8_target_60_db=[piston_db(freq, 60.0, PTT8_DIAMETER_MM) for freq in FREQS_HZ],
    )
