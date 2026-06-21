from __future__ import annotations

import math


SPEED_OF_SOUND = 343.0


def j1(x: float) -> float:
    """Bessel J1 via power series, stable for the ka range used here."""
    if x == 0:
        return 0.0
    term = x / 2.0
    total = term
    for m in range(1, 100):
        term *= -((x * x) / 4.0) / (m * (m + 1))
        total += term
        if abs(term) < 1e-14:
            break
    return total


def piston_pressure_ratio(freq_hz: float, angle_deg: float, diameter_mm: float = 173.0) -> float:
    radius_m = diameter_mm / 2000.0
    x = 2.0 * math.pi * freq_hz / SPEED_OF_SOUND
    x *= radius_m * math.sin(math.radians(angle_deg))
    if abs(x) < 1e-12:
        return 1.0
    return abs(2.0 * j1(x) / x)


def piston_db(freq_hz: float, angle_deg: float, diameter_mm: float = 173.0) -> float:
    return 20.0 * math.log10(max(piston_pressure_ratio(freq_hz, angle_deg, diameter_mm), 1e-12))


def beamwidth_6db(freq_hz: float, diameter_mm: float = 173.0) -> float | None:
    target = 10.0 ** (-6.0 / 20.0)
    for tenths in range(0, 901):
        angle = tenths / 10.0
        if piston_pressure_ratio(freq_hz, angle, diameter_mm) <= target:
            return 2.0 * angle
    return None


def target_table(freqs: list[float] | None = None, diameter_mm: float = 173.0) -> list[dict[str, float | None]]:
    freqs = freqs or [1000, 1200, 1300, 1400, 1500, 1600, 1800, 2000, 2500, 3000]
    rows: list[dict[str, float | None]] = []
    for freq in freqs:
        rows.append(
            {
                "freq_hz": float(freq),
                "beamwidth_6db_deg": beamwidth_6db(freq, diameter_mm),
                "db_30": piston_db(freq, 30, diameter_mm),
                "db_45": piston_db(freq, 45, diameter_mm),
                "db_60": piston_db(freq, 60, diameter_mm),
            }
        )
    return rows
