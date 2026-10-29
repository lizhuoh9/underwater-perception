"""Calibration recheck: per-image curves and the first-round threshold rule."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uwseg.recheck_calibration import first_round_threshold, image_curve, run
from uwseg.train_pixel_forest import threshold_curve


def counts_from_rates(fnr, fpr, positives=1000, negatives=1000):
    fn, fp = np.rint(positives * fnr).astype(int), np.rint(negatives * fpr).astype(int)
    return np.column_stack((positives - fn, fp, fn, negatives - fp))


class RecheckCalibrationTests(unittest.TestCase):
    def test_image_curves_match_the_locked_threshold_curve(self):
        rng = np.random.default_rng(5)
        score = rng.integers(0, 256, (6, 8), dtype=np.uint8)
        labels = np.zeros((12, 16), dtype=np.uint8)
        labels[3:9, 4:13] = 6
        counts, macro = image_curve(score, labels)
        objective, total = threshold_curve([(score, labels)])
        np.testing.assert_array_equal(counts, total)
        np.testing.assert_array_equal(macro, objective)

    def test_first_round_prefers_the_best_threshold_that_beats_the_baseline(self):
        q = np.arange(257)
        counts = counts_from_rates(q / 256 * 0.6, (256 - q) / 256 * 0.6)
        objective = np.exp(-(((q - 40) / 30.0) ** 2))
        chosen = first_round_threshold(objective, counts, baseline_fnr=0.3, baseline_fpr=0.3)
        eligible = (counts[:, 2] / 1000 <= 0.3) & (counts[:, 1] / 1000 <= 0.3)
        self.assertEqual(chosen["threshold_q"], int(np.flatnonzero(eligible).min()))
        self.assertTrue(chosen["calibration_dominates_baseline"])
        fallback = first_round_threshold(objective, counts, baseline_fnr=0.01, baseline_fpr=0.01)
        self.assertEqual((fallback["threshold_q"], fallback["calibration_dominates_baseline"]), (40, False))

    def test_recheck_refuses_a_populated_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            evidence = Path(folder) / "summary.json"
            evidence.write_text("previous evidence", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(Path(folder), Path(folder) / "unused_manifest.json")
            self.assertEqual(evidence.read_text(encoding="utf-8"), "previous evidence")


if __name__ == "__main__":
    unittest.main()
