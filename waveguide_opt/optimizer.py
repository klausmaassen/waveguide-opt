from __future__ import annotations

import csv
import random
from dataclasses import replace
from pathlib import Path

from .ath_config import write_cfg
from .geometry import estimated_mouth_diameter
from .models import Bounds, Candidate


SHAPE_PARAMS = [
    "length",
    "throat_diameter",
    "throat_angle",
    "coverage_angle",
    "os_k",
    "term_s",
    "term_q",
    "term_n",
]
CANDIDATE_CSV_FIELDS = ["name", *SHAPE_PARAMS, "target_mouth_diameter"]


def _candidate_values(rng: random.Random, bounds: Bounds) -> dict[str, float]:
    values = {}
    for name in SHAPE_PARAMS:
        lo, hi = getattr(bounds, name)
        values[name] = lo + rng.random() * (hi - lo)
    return values


def _within_actual_mouth_bounds(candidate: Candidate, bounds: Bounds) -> bool:
    lo, hi = bounds.target_mouth_diameter
    mouth = estimated_mouth_diameter(candidate)
    return lo <= mouth <= hi


def _with_actual_mouth(candidate: Candidate) -> Candidate:
    return replace(candidate, target_mouth_diameter=estimated_mouth_diameter(candidate))


def sample_candidates(count: int, seed: int = 1, bounds: Bounds | None = None) -> list[Candidate]:
    rng = random.Random(seed)
    b = bounds or Bounds()
    candidates: list[Candidate] = []
    attempts = 0
    max_attempts = max(count * 400, 2000)
    while len(candidates) < count and attempts < max_attempts:
        attempts += 1
        candidate = Candidate(name=f"cand_{len(candidates) + 1:04d}", **_candidate_values(rng, b))
        if not _within_actual_mouth_bounds(candidate, b):
            continue
        candidates.append(_with_actual_mouth(candidate))
    if len(candidates) < count:
        lo, hi = b.target_mouth_diameter
        raise ValueError(
            f"Only generated {len(candidates)} of {count} candidates with actual mouth diameter between {lo:.1f} and {hi:.1f} mm"
        )
    return candidates


def write_batch(count: int, seed: int, output_dir: str | Path, ath_output_root: str | None = None) -> Path:
    output = Path(output_dir)
    cfg_dir = output / "configs"
    candidates = sample_candidates(count, seed)
    csv_path = output / "candidates.csv"
    output.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = CANDIDATE_CSV_FIELDS
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            write_cfg(candidate, cfg_dir / f"{candidate.name}.cfg", output_root=ath_output_root)
            row = candidate.as_dict()
            writer.writerow({key: row[key] for key in fieldnames})
    return csv_path
