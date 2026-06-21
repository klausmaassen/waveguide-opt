from __future__ import annotations

from dataclasses import fields
from typing import Any

from .fast_acoustics import evaluate_fast
from .geometry import candidate_preview, estimated_mouth_diameter
from .models import Candidate


DEFAULT_OBJECTIVE_WEIGHTS = {
    "constant_directivity": 0.30,
    "ptt8_crossover_match": 0.25,
    "off_axis_smoothness": 0.20,
    "response_ripple": 0.15,
    "resonance_penalty": 0.10,
}

ACOUSTIC_KEYS = [
    "pre_score",
    "acoustic_score",
    "constant_directivity",
    "ptt8_crossover_match",
    "off_axis_smoothness",
    "response_ripple",
    "resonance_penalty",
    "directivity_index_smoothness",
    "mouth_reflection",
    "throat_transition",
    "beamwidth_2000",
    "beamwidth_4000",
    "freqs_hz",
    "angles_deg",
    "polar_db",
    "on_axis_db",
    "beamwidth_deg",
    "directivity_index_db",
    "ptt8_target_30_db",
    "ptt8_target_60_db",
    "objective_weights",
]

_CANDIDATE_FIELDS = {field.name for field in fields(Candidate)}


def candidate_from_row(row: dict[str, Any]) -> Candidate:
    data = {key: row[key] for key in _CANDIDATE_FIELDS if key in row}
    return Candidate(**data)


def weighted_score(row: dict[str, Any], weights: dict[str, float]) -> float:
    total_weight = sum(max(0.0, float(value)) for value in weights.values())
    if total_weight <= 0:
        weights = DEFAULT_OBJECTIVE_WEIGHTS
        total_weight = sum(weights.values())
    score = 0.0
    for key, weight in weights.items():
        score += max(0.0, float(weight)) * float(row.get(key, 0.0))
    return score / total_weight


def candidate_row(candidate: Candidate, weights: dict[str, float], include_curves: bool = False) -> dict[str, Any]:
    acoustic = evaluate_fast(candidate)
    row = candidate.as_dict()
    row.update(acoustic.as_dict(include_curves=include_curves))
    score = weighted_score(row, weights)
    row["pre_score"] = score
    row["acoustic_score"] = score
    row["objective_weights"] = weights
    row["estimated_mouth_diameter"] = estimated_mouth_diameter(candidate)
    return row


def preview_from_row(row: dict[str, Any], default_weights: dict[str, float] | None = None) -> dict[str, Any]:
    weights = row.get("objective_weights") or default_weights or DEFAULT_OBJECTIVE_WEIGHTS
    candidate = candidate_from_row(row)
    preview = candidate_preview(candidate)
    acoustic_row = candidate_row(candidate, weights, include_curves=True)
    for key in ACOUSTIC_KEYS:
        preview[key] = acoustic_row.get(key)
    for key in ["cfg_path", "ath_output_root"]:
        if key in row:
            preview[key] = row.get(key)
    return preview
