from __future__ import annotations

from pathlib import Path

from .models import Candidate
from .t34b import source_contours


def render_cfg(candidate: Candidate, include_source_contours: bool = True, output_root: str | None = None) -> str:
    offset = max(candidate.length + 5.0, 10.0)
    lines = [
        f"; Generated candidate: {candidate.name}",
        "Throat.Profile = 1",
        f"Throat.Diameter = {candidate.throat_diameter:.3f}",
        f"Throat.Angle = {candidate.throat_angle:.3f}",
        f"Coverage.Angle = {candidate.coverage_angle:.3f}",
        f"Length = {candidate.length:.3f}",
        f"OS.k = {candidate.os_k:.5f}",
        "",
        f"Term.s = {candidate.term_s:.5f}",
        f"Term.q = {candidate.term_q:.5f}",
        f"Term.n = {candidate.term_n:.5f}",
        "",
        "Morph.TargetShape = 0",
        "",
        f"Mesh.LengthSegments = {candidate.mesh_length_segments}",
        f"Mesh.ThroatResolution = {candidate.mesh_throat_resolution:.3f}",
        f"Mesh.MouthResolution = {candidate.mesh_mouth_resolution:.3f}",
        "",
        "ABEC.SimType = 1",
        "ABEC.SimProfile = 0",
        f"ABEC.f1 = {candidate.abec_f1:.3f}",
        f"ABEC.f2 = {candidate.abec_f2:.3f}",
        f"ABEC.NumFrequencies = {candidate.abec_num_frequencies}",
        "ABEC.Abscissa = 1",
        "ABEC.MeshFrequency = 30000",
        "",
        "ABEC.Polars:SPL = {",
        f"  MapAngleRange = 0,{candidate.polar_max_angle:.0f},{candidate.polar_points}",
        "  NormAngle = 0",
        "  Distance = 3",
        f"  Offset = {offset:.3f}",
        "}",
        "",
        "Output.ABECProject = 1",
        "Output.STL = 0",
        "Output.MSH = 0",
    ]
    if output_root:
        lines.append(f'Output.DestDir = "{output_root}"')
    if include_source_contours:
        lines.extend(["", "Source.Contours = " + source_contours(candidate.throat_diameter)])
    return "\n".join(lines) + "\n"


def write_cfg(candidate: Candidate, path: str | Path, include_source_contours: bool = True, output_root: str | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if output_root:
        Path(output_root).mkdir(parents=True, exist_ok=True)
    target.write_text(render_cfg(candidate, include_source_contours, output_root), encoding="utf-8")
    return target
