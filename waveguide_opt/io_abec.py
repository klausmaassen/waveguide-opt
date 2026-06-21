from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolarData:
    freqs: list[float]
    angles: list[float]
    spl_db: list[list[float]]

    def angle_index(self, angle: float) -> int:
        return min(range(len(self.angles)), key=lambda i: abs(self.angles[i] - angle))


def _as_float(text: str) -> float:
    return float(text.strip().replace(",", "."))


def read_polar_csv(path: str | Path) -> PolarData:
    """Read either wide freq/angle columns or long freq,angle,spl exports."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        rows = list(csv.reader(handle, dialect))
    rows = [row for row in rows if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("empty polar file")
    header = [cell.strip().lower() for cell in rows[0]]
    data = rows[1:]
    if {"freq_hz", "angle_deg", "spl_db"}.issubset(set(header)):
        fi, ai, si = header.index("freq_hz"), header.index("angle_deg"), header.index("spl_db")
        freqs = sorted({_as_float(row[fi]) for row in data})
        angles = sorted({_as_float(row[ai]) for row in data})
        matrix = [[0.0 for _ in angles] for _ in freqs]
        f_index = {v: i for i, v in enumerate(freqs)}
        a_index = {v: i for i, v in enumerate(angles)}
        for row in data:
            matrix[f_index[_as_float(row[fi])]][a_index[_as_float(row[ai])]] = _as_float(row[si])
        return PolarData(freqs, angles, matrix)

    freqs = [_as_float(row[0]) for row in data]
    angles = []
    for cell in rows[0][1:]:
        cleaned = cell.lower().replace("deg", "").replace("angle", "").replace("@", "").strip()
        angles.append(_as_float(cleaned))
    matrix = [[_as_float(cell) for cell in row[1 : 1 + len(angles)]] for row in data]
    return PolarData(freqs, angles, matrix)
