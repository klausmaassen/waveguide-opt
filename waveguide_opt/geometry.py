from __future__ import annotations

import math

from .models import Candidate


def osse_radius(candidate: Candidate, z_mm: float) -> float:
    """Approximate ATH OS-SE profile radius for frontend visualization."""
    r0 = candidate.throat_diameter / 2.0
    length = max(candidate.length, 1e-6)
    z = max(0.0, min(z_mm, length))
    alpha = math.radians(candidate.coverage_angle)
    alpha0 = math.radians(candidate.throat_angle)
    k = candidate.os_k
    base = math.sqrt(
        (k * r0) ** 2
        + 2.0 * k * r0 * z * math.tan(alpha0)
        + (z * math.tan(alpha)) ** 2
    ) + r0 * (1.0 - k)
    qz_l = candidate.term_q * z / length
    qz_l = max(0.0, min(qz_l, 0.999999))
    term = candidate.term_s * length / candidate.term_q
    term *= 1.0 - (1.0 - qz_l**candidate.term_n) ** (1.0 / candidate.term_n)
    return base + term


def profile_points(candidate: Candidate, count: int = 80) -> list[dict[str, float]]:
    points = []
    for idx in range(count):
        z = candidate.length * idx / max(count - 1, 1)
        points.append({"z": z, "r": osse_radius(candidate, z)})
    return points


def estimated_mouth_diameter(candidate: Candidate) -> float:
    return 2.0 * osse_radius(candidate, candidate.length)


def candidate_preview(candidate: Candidate) -> dict[str, object]:
    return {
        "candidate": candidate.as_dict(),
        "estimated_mouth_diameter": estimated_mouth_diameter(candidate),
        "profile": profile_points(candidate),
    }
