import unittest
from pathlib import Path

from waveguide_opt.ath_config import render_cfg
from waveguide_opt.io_abec import read_polar_csv
from waveguide_opt.models import Candidate
from waveguide_opt.piston import beamwidth_6db
from waveguide_opt.scoring import score_polar
from waveguide_opt.t34b import T34BGeometry, source_contours


class PipelineTests(unittest.TestCase):
    def test_piston_beamwidth_reference(self):
        bw = beamwidth_6db(1500, 173)
        self.assertIsNotNone(bw)
        self.assertGreater(bw, 130)
        self.assertLess(bw, 145)

    def test_t34b_cap_radius(self):
        geom = T34BGeometry()
        self.assertGreater(geom.spherical_cap_radius_mm, 23.5)
        self.assertLess(geom.spherical_cap_radius_mm, 24.5)

    def test_ath_cfg_uses_40mm_start(self):
        cfg = render_cfg(Candidate(name="test"))
        self.assertIn("Throat.Diameter = 40.000", cfg)
        self.assertIn("Source.Contours", cfg)
        self.assertIn("line surround_outer WG0 0.0", cfg)

    def test_source_contours_for_larger_throat(self):
        text = source_contours(41)
        self.assertIn("point apex 7.000 0.000", text)
        self.assertIn("line surround_outer WG0 0.0", text)

    def test_score_smoke(self):
        tmp = Path(".test_tmp")
        tmp.mkdir(exist_ok=True)
        polar = tmp / "polar.csv"
        try:
            polar.write_text(
                "freq_hz,0,15,30,45,60,75,90\n"
                "1000,90,89.8,89.2,88.7,88.0,87.5,87.0\n"
                "1500,90,89.6,88.4,86.8,84.8,83.5,82.8\n"
                "2000,90,89.3,87.1,83.8,80.0,76.3,74.7\n",
                encoding="utf-8",
            )
            data = read_polar_csv(polar)
            score = score_polar(Candidate(name="x"), data)
            self.assertGreaterEqual(score.total, 0)
            self.assertNotIn("geometry_penalty", score.as_dict())
        finally:
            if polar.exists():
                polar.unlink()
            if tmp.exists():
                tmp.rmdir()


if __name__ == "__main__":
    unittest.main()
