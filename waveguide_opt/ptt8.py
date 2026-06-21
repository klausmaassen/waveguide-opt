from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path


SPL_NAME = "PTT8.0X04-NAB-01 - SPL.txt"
Z_NAME = "PTT8.0X04-NAB-01 - Z.txt"


@dataclass(frozen=True)
class ResponsePoint:
    freq_hz: float
    value: float
    phase_deg: float


def read_response_zip(path: str | Path, member: str) -> list[ResponsePoint]:
    with zipfile.ZipFile(path) as archive:
        text = archive.read(member).decode("utf-8", errors="replace").replace("\r", "")
    points: list[ResponsePoint] = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3:
            points.append(ResponsePoint(float(parts[0]), float(parts[1]), float(parts[2])))
    return points


def interpolate(points: list[ResponsePoint], freq_hz: float) -> ResponsePoint:
    if not points:
        raise ValueError("empty response")
    if freq_hz <= points[0].freq_hz:
        return points[0]
    if freq_hz >= points[-1].freq_hz:
        return points[-1]
    lo = 0
    hi = len(points) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if points[mid].freq_hz < freq_hz:
            lo = mid
        else:
            hi = mid
    a = points[lo]
    b = points[hi]
    t = (freq_hz - a.freq_hz) / (b.freq_hz - a.freq_hz)
    return ResponsePoint(freq_hz, a.value + t * (b.value - a.value), a.phase_deg + t * (b.phase_deg - a.phase_deg))


def export_csv(zip_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spl = read_response_zip(zip_path, SPL_NAME)
    imp = read_response_zip(zip_path, Z_NAME)
    spl_path = output / "ptt8_spl.csv"
    z_path = output / "ptt8_impedance.csv"
    for target, rows, value_name in [(spl_path, spl, "spl_db"), (z_path, imp, "ohm")]:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["freq_hz", value_name, "phase_deg"])
            for point in rows:
                writer.writerow([point.freq_hz, point.value, point.phase_deg])
    return spl_path, z_path
