import unittest

from waveguide_opt.models import Candidate
from waveguide_opt.evaluation import candidate_row


class ObjectiveWeightTests(unittest.TestCase):
    def test_single_weight_selects_single_metric(self):
        weights = {
            "constant_directivity": 0.0,
            "ptt8_crossover_match": 1.0,
            "off_axis_smoothness": 0.0,
            "response_ripple": 0.0,
            "resonance_penalty": 0.0,
        }
        row = candidate_row(Candidate(name="x"), weights)
        self.assertAlmostEqual(row["pre_score"], row["ptt8_crossover_match"])


if __name__ == "__main__":
    unittest.main()
