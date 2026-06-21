from __future__ import annotations

import csv
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

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


def _encode(candidate: Candidate, bounds: Bounds) -> list[float]:
    result = []
    for name in SHAPE_PARAMS:
        lo, hi = getattr(bounds, name)
        value = float(getattr(candidate, name))
        result.append((value - lo) / max(hi - lo, 1e-9))
    return result


def _decode(x: list[float], bounds: Bounds, name: str) -> Candidate:
    values: dict[str, float] = {}
    for i, param_name in enumerate(SHAPE_PARAMS):
        lo, hi = getattr(bounds, param_name)
        values[param_name] = lo + float(x[i]) * (hi - lo)
    return _with_actual_mouth(Candidate(name=name, **values))


def optimize_cma(
    bounds: Bounds | None = None,
    sigma0: float = 0.25,
    max_evals: int = 500,
    seed: int = 1,
    weights: dict[str, float] | None = None,
    progress_callback: Callable[[int, Candidate, float], None] | None = None,
) -> list[Candidate]:
    """CMA-ES optimisation; returns valid candidates sorted by acoustic score (ascending)."""
    try:
        import cma
    except ImportError as exc:
        raise ImportError("pip install cma") from exc

    from .evaluation import DEFAULT_OBJECTIVE_WEIGHTS, weighted_score
    from .fast_acoustics import evaluate_fast

    b = bounds or Bounds()
    w = weights or DEFAULT_OBJECTIVE_WEIGHTS.copy()
    mouth_lo, mouth_hi = b.target_mouth_diameter
    n = len(SHAPE_PARAMS)
    eval_count = [0]
    history: list[tuple[float, Candidate]] = []

    def objective(x: list[float]) -> float:
        candidate = _decode(x, b, f"cma_{eval_count[0]:04d}")
        eval_count[0] += 1
        mouth = estimated_mouth_diameter(candidate)
        violation = max(0.0, mouth_lo - mouth, mouth - mouth_hi)
        penalty = (violation / max(mouth_lo, 1.0)) ** 2 * 25.0
        acoustic_score = weighted_score(evaluate_fast(candidate).as_dict(), w)
        history.append((acoustic_score, candidate))
        if progress_callback is not None:
            progress_callback(eval_count[0], candidate, acoustic_score)
        return acoustic_score + penalty

    opts = {
        "bounds": [[0.0] * n, [1.0] * n],
        "maxfevals": max_evals,
        "seed": seed,
        "verbose": -9,
    }
    cma.CMAEvolutionStrategy([0.5] * n, sigma0, opts).optimize(objective)

    within = [
        (score, c)
        for score, c in history
        if mouth_lo <= estimated_mouth_diameter(c) <= mouth_hi
    ]
    within.sort(key=lambda t: t[0])
    seen: set[str] = set()
    result: list[Candidate] = []
    for _, candidate in within:
        key = "_".join(f"{getattr(candidate, p):.2f}" for p in SHAPE_PARAMS)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def select_diverse(
    rows: list[dict[str, Any]],
    count: int,
    top_fraction: float = 0.20,
    bounds: Bounds | None = None,
) -> list[dict[str, Any]]:
    """Greedy farthest-point selection from the top_fraction pool (by pre_score)."""
    b = bounds or Bounds()
    if not rows or count <= 0:
        return []
    sorted_rows = sorted(rows, key=lambda r: float(r.get("pre_score", float("inf"))))
    pool_size = max(count, int(len(sorted_rows) * top_fraction))
    pool = sorted_rows[:pool_size]
    if len(pool) <= count:
        return pool

    def vec(row: dict[str, Any]) -> list[float]:
        result = []
        for name in SHAPE_PARAMS:
            lo, hi = getattr(b, name)
            v = float(row.get(name, (lo + hi) / 2.0))
            result.append((v - lo) / max(hi - lo, 1e-9))
        return result

    vectors = [vec(row) for row in pool]

    def sq_dist(a: list[float], bv: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, bv))

    selected_set: set[int] = {0}
    order: list[int] = [0]
    while len(order) < count:
        best_idx, best_d = -1, -1.0
        for idx in range(len(pool)):
            if idx in selected_set:
                continue
            d = min(sq_dist(vectors[idx], vectors[s]) for s in order)
            if d > best_d:
                best_d, best_idx = d, idx
        if best_idx < 0:
            break
        selected_set.add(best_idx)
        order.append(best_idx)
    return [pool[i] for i in order]


def sensitivity_analysis(
    candidate: Candidate,
    bounds: Bounds | None = None,
    delta: float = 0.10,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Finite-difference sensitivity: score change when each parameter moves ±delta of its range."""
    from .evaluation import DEFAULT_OBJECTIVE_WEIGHTS, weighted_score
    from .fast_acoustics import evaluate_fast

    b = bounds or Bounds()
    w = weights or DEFAULT_OBJECTIVE_WEIGHTS.copy()

    def score(c: Candidate) -> float:
        return weighted_score(evaluate_fast(c).as_dict(), w)

    base = score(candidate)
    results: list[dict[str, Any]] = []
    for param in SHAPE_PARAMS:
        lo, hi = getattr(b, param)
        v0 = float(getattr(candidate, param))
        step = (hi - lo) * delta
        vlo = max(lo, v0 - step)
        vhi = min(hi, v0 + step)
        s_lo = score(_with_actual_mouth(replace(candidate, **{param: vlo})))
        s_hi = score(_with_actual_mouth(replace(candidate, **{param: vhi})))
        span = max(vhi - vlo, 1e-9)
        results.append({
            "parameter": param,
            "base_value": round(v0, 4),
            "value_minus": round(vlo, 4),
            "value_plus": round(vhi, 4),
            "score_minus": round(s_lo, 5),
            "score_base": round(base, 5),
            "score_plus": round(s_hi, 5),
            "sensitivity": round(abs(s_hi - s_lo), 5),
            "gradient_normalized": round((s_hi - s_lo) / span * (hi - lo), 5),
        })
    results.sort(key=lambda r: r["sensitivity"], reverse=True)
    return results


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
