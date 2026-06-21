import unittest

from waveguide_opt.fast_acoustics import evaluate_fast
from waveguide_opt.models import Candidate


class FastAcousticsTests(unittest.TestCase):
    def test_fast_acoustic_score_is_finite(self):
        result = evaluate_fast(Candidate(name="x"))
        self.assertGreaterEqual(result.total, 0)
        self.assertGreaterEqual(result.constant_directivity, 0)
        self.assertGreaterEqual(result.ptt8_crossover_match, 0)
        self.assertGreaterEqual(result.response_ripple, 0)

    def test_fast_acoustic_beamwidth_is_plausible(self):
        result = evaluate_fast(Candidate(name="x"))
        self.assertIsNotNone(result.beamwidth_2000)
        self.assertGreater(result.beamwidth_2000, 20)
        self.assertLessEqual(result.beamwidth_2000, 180)

    def test_fast_acoustic_curves_are_exported(self):
        data = evaluate_fast(Candidate(name="x")).as_dict(include_curves=True)
        self.assertEqual(len(data["freqs_hz"]), len(data["polar_db"]))
        self.assertEqual(len(data["angles_deg"]), len(data["polar_db"][0]))
        self.assertEqual(len(data["freqs_hz"]), len(data["directivity_index_db"]))

    def test_fast_acoustic_chart_axis_spans_zero_to_180_degrees(self):
        data = evaluate_fast(Candidate(name="x")).as_dict(include_curves=True)
        on_axis_idx = data["angles_deg"].index(0.0)
        side_idx = data["angles_deg"].index(90.0)
        rear_idx = data["angles_deg"].index(180.0)
        self.assertEqual(len(data["angles_deg"]), 37)
        for row in data["polar_db"]:
            self.assertAlmostEqual(row[on_axis_idx], 0.0, places=7)
            self.assertLess(row[side_idx], row[on_axis_idx])
            self.assertLess(row[rear_idx], row[on_axis_idx])

    def test_fast_acoustic_keeps_low_mid_angles_visible(self):
        data = evaluate_fast(Candidate(name="x")).as_dict(include_curves=True)
        freq_idx = data["freqs_hz"].index(2000.0)
        angles = data["angles_deg"]
        row = data["polar_db"][freq_idx]
        self.assertGreater(row[angles.index(60.0)], -18.0)
        self.assertGreater(row[angles.index(90.0)], -30.0)


if __name__ == "__main__":
    unittest.main()
