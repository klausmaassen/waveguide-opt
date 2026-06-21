import unittest

from waveguide_opt.geometry import estimated_mouth_diameter, profile_points
from waveguide_opt.models import Candidate


class GeometryTests(unittest.TestCase):
    def test_profile_points_start_at_40mm_throat(self):
        candidate = Candidate(name="x")
        points = profile_points(candidate, 8)
        self.assertEqual(len(points), 8)
        self.assertAlmostEqual(points[0]["r"] * 2, 40.0, places=6)

    def test_estimated_mouth_is_plausible(self):
        mouth = estimated_mouth_diameter(Candidate(name="x"))
        self.assertGreater(mouth, 100)
        self.assertLess(mouth, 260)


if __name__ == "__main__":
    unittest.main()
