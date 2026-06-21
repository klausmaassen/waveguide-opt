from __future__ import annotations

import math
from dataclasses import dataclass

from .io_abec import PolarData
from .models import Candidate
from .piston import piston_db


@dataclass(frozen=True)
class Score:
    total: float
    directivity_match: float
    beamwidth_smoothness: float
    response_ripple: float
    crossover_sum_error: float
    resonance_penalty: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def _freq_weights(freqs: list[float], center: float = 1500.0) -> list[float]:
    weights = []
    for f in freqs:
        octaves = math.log2(max(f, 1.0) / center)
        weights.append(math.exp(-(octaves * octaves) / 0.9))
    return weights


def _beamwidth(row: list[float], angles: list[float], drop_db: float = 6.0) -> float | None:
    on_axis = row[0]
    target = on_axis - drop_db
    for idx in range(1, len(angles)):
        if row[idx] <= target:
            a0, a1 = angles[idx - 1], angles[idx]
            y0, y1 = row[idx - 1], row[idx]
            if y1 == y0:
                return 2.0 * a1
            t = (target - y0) / (y1 - y0)
            return 2.0 * (a0 + t * (a1 - a0))
    return None


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def score_polar(candidate: Candidate, polar: PolarData) -> Score:
    angle_indices = [polar.angle_index(a) for a in [15, 30, 45, 60]]
    weights = _freq_weights(polar.freqs)

    directivity_terms = []
    ripple_values = []
    beamwidths = []
    resonance_terms = []
    for f, w, row in zip(polar.freqs, weights, polar.spl_db):
        if not 900 <= f <= 6000:
            continue
        for idx in angle_indices:
            angle = polar.angles[idx]
            wg_rel = row[idx] - row[0]
            target_rel = piston_db(f, angle, 173.0)
            directivity_terms.append(w * (wg_rel - target_rel) ** 2)
        bw = _beamwidth(row, polar.angles)
        if bw is not None and 1000 <= f <= 5000:
            beamwidths.append(bw)
        if 1000 <= f <= 20000:
            ripple_values.append(row[0])
    directivity = math.sqrt(sum(directivity_terms) / max(len(directivity_terms), 1))
    beam_smooth = _std([beamwidths[i + 1] - beamwidths[i] for i in range(len(beamwidths) - 1)]) / 10.0
    ripple = _std(ripple_values) / 3.0
    for values in [ripple_values[i : i + 5] for i in range(max(0, len(ripple_values) - 4))]:
        if len(values) == 5:
            center = values[2]
            side = (values[0] + values[1] + values[3] + values[4]) / 4
            resonance_terms.append(max(0.0, abs(center - side) - 1.5))
    resonance = sum(resonance_terms) / max(len(resonance_terms), 1)
    crossover = directivity * 0.6 + ripple * 0.4
    total = 0.35 * directivity + 0.22 * beam_smooth + 0.18 * ripple + 0.15 * crossover + 0.10 * resonance
    return Score(total, directivity, beam_smooth, ripple, crossover, resonance)
