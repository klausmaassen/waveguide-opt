from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    name: str
    length: float = 40.0
    throat_diameter: float = 40.0
    throat_angle: float = 6.0
    coverage_angle: float = 58.0
    os_k: float = 0.82
    term_s: float = 0.76
    term_q: float = 0.995
    term_n: float = 5.0
    target_mouth_diameter: float = 195.0
    mesh_length_segments: int = 90
    mesh_throat_resolution: float = 2.5
    mesh_mouth_resolution: float = 5.0
    abec_f1: float = 700.0
    abec_f2: float = 20000.0
    abec_num_frequencies: int = 160
    polar_max_angle: float = 180.0
    polar_points: int = 37

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Bounds:
    length: tuple[float, float] = (32.0, 52.0)
    throat_diameter: tuple[float, float] = (40.0, 41.5)
    throat_angle: tuple[float, float] = (0.0, 12.0)
    coverage_angle: tuple[float, float] = (45.0, 70.0)
    os_k: tuple[float, float] = (0.65, 1.0)
    term_s: tuple[float, float] = (0.60, 0.95)
    term_q: tuple[float, float] = (0.985, 0.999)
    term_n: tuple[float, float] = (3.5, 8.0)
    target_mouth_diameter: tuple[float, float] = (180.0, 205.0)


@dataclass(frozen=True)
class Settings:
    ath_exe: str | None = None
    ath_workdir: str | None = None
    abec_exe: str | None = None
    abec_workdir: str | None = None
    ptt8_zip: str | None = None
    ath_output_root: str = "runs/ath_output"

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
