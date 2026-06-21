from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class T34BGeometry:
    dome_diameter_mm: float = 34.0
    dome_height_mm: float = 7.0
    surround_profile_radius_mm: float = 1.5
    total_radiating_diameter_mm: float = 40.0

    @property
    def dome_radius_mm(self) -> float:
        return self.dome_diameter_mm / 2.0

    @property
    def spherical_cap_radius_mm(self) -> float:
        a = self.dome_radius_mm
        h = self.dome_height_mm
        return (a * a + h * h) / (2.0 * h)


def source_contours(throat_diameter_mm: float = 40.0, geometry: T34BGeometry | None = None) -> str:
    """Return an ATH Source.Contours script approximating the T34B dome and surround.

    The geometry is intentionally conservative: the moving dome receives full
    radiation weight, the surround receives tapered weights, and the final line
    to WG0 is non-moving if throat diameter exceeds the 40 mm radiating region.
    """
    g = geometry or T34BGeometry()
    throat_r = throat_diameter_mm / 2.0
    dome_r = g.dome_radius_mm
    nominal_total_r = g.total_radiating_diameter_mm / 2.0
    total_r = min(nominal_total_r, throat_r)
    cap_r = g.spherical_cap_radius_mm
    center_z = -(cap_r - g.dome_height_mm)

    lines = [
        "{",
        "  ; T34B approximate dome + surround source",
        f"  ; dome cap radius approx {cap_r:.3f} mm",
        f"  ; nominal radiating diameter {g.total_radiating_diameter_mm:.3f} mm",
        f"  point apex {g.dome_height_mm:.3f} 0.000 0.5",
        f"  point dome_edge 0.000 {dome_r:.3f} 0.5",
        f"  point surround_mid 0.900 {(dome_r + total_r) / 2.0:.3f} 0.5",
        f"  point surround_outer 0.000 {total_r:.3f} 0.5",
        f"  cpoint dome_center {center_z:.3f} 0.000",
        f"  cpoint surround_center 0.000 {(dome_r + total_r) / 2.0:.3f}",
        "  arc apex dome_center dome_edge 1.0",
        "  arc dome_edge surround_center surround_mid 0.65",
        "  arc surround_mid surround_center surround_outer 0.30",
    ]
    lines.append("  line surround_outer WG0 0.0")
    lines.append("}")
    return "\n".join(lines)
