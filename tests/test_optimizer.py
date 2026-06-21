import unittest

from waveguide_opt.geometry import estimated_mouth_diameter
from waveguide_opt.models import Bounds
from waveguide_opt.optimizer import sample_candidates


class OptimizerTests(unittest.TestCase):
    def test_sample_candidates_respects_actual_mouth_diameter_bounds(self):
        bounds = Bounds(target_mouth_diameter=(180.0, 205.0))
        candidates = sample_candidates(40, seed=11, bounds=bounds)
        self.assertEqual(len(candidates), 40)
        for candidate in candidates:
            mouth = estimated_mouth_diameter(candidate)
            self.assertAlmostEqual(candidate.target_mouth_diameter, mouth)
            self.assertGreaterEqual(mouth, 180.0)
            self.assertLessEqual(mouth, 205.0)

    def test_sample_candidates_reports_impossible_mouth_bounds(self):
        bounds = Bounds(target_mouth_diameter=(10.0, 11.0))
        with self.assertRaises(ValueError):
            sample_candidates(3, seed=11, bounds=bounds)


if __name__ == "__main__":
    unittest.main()
