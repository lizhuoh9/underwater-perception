"""Contract checks for recall-constrained calibration and train-only sampling."""

import json
import unittest

import numpy as np

from uwseg.optimize_pixel_forest import constrained_threshold, sample_training_rows
from uwseg.paths import RESULTS_DIR
from uwseg.train_pixel_forest import threshold_curve
from uwseg.verify_lowfp_control import run as verify_control

CONTROL = RESULTS_DIR / "forest_lowfp/threshold_control"


class LowFpTests(unittest.TestCase):
    def test_minimum_false_positive_threshold_respects_fnr_cap(self):
        scores = np.array([[10, 20, 30, 40, 80], [15, 50, 60, 70, 90]], dtype=np.uint8)
        labels = np.array([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=np.uint8)
        objective, counts = threshold_curve([(scores, labels)])
        choice = constrained_threshold(counts, objective, 0.2)
        eligible = []
        for q in range(257):
            predicted, truth = scores >= q, labels != 0
            fnr = np.count_nonzero(~predicted & truth) / truth.sum()
            fpr = np.count_nonzero(predicted & ~truth) / (~truth).sum()
            if fnr <= 0.2:
                eligible.append((fpr, -objective[q], fnr, q))
        self.assertEqual(choice["threshold_q"], min(eligible)[3])
        self.assertLessEqual(choice["calibration_fnr"], 0.2)

    def test_hard_mining_uses_unselected_training_water_only(self):
        features = np.zeros((40, 40, 16), dtype=np.uint8)
        features[..., 0] = np.arange(1600).reshape(40, 40) % 256
        features[..., 1] = np.arange(1600).reshape(40, 40) // 256
        labels = np.zeros((40, 40), dtype=np.uint8)
        labels[:10] = 1
        score = np.full((40, 40), 200, dtype=np.uint8)
        score[10:20] = 20
        first = sample_training_rows(features, labels, score, 42)
        second = sample_training_rows(features, labels, score, 42)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        ux, uy, mx, my = first
        self.assertEqual(len(ux), 1024)
        self.assertGreater(len(mx), len(ux))
        self.assertLessEqual(len(mx), 1280)
        np.testing.assert_array_equal(mx[:1024], ux)
        np.testing.assert_array_equal(my[:1024], uy)
        ids = mx[:, 0].astype(int) + mx[:, 1].astype(int) * 256
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(np.all(my[1024:] == 0))
        self.assertTrue(np.all(score.ravel()[ids[1024:]] >= 57))

    @unittest.skipUnless((CONTROL / "MODEL_EXPORT.json").exists(), "Export the frozen control first")
    def test_export_changes_only_the_pre_frozen_threshold(self):
        original = RESULTS_DIR / "forest_comparison/model"
        locked = json.loads((RESULTS_DIR / "forest_lowfp/LOCKED.json").read_text(encoding="utf-8"))
        old_locked = json.loads((RESULTS_DIR / "forest_comparison/LOCKED.json").read_text(encoding="utf-8"))
        q = locked["threshold_only_control"]["threshold_q"]
        old_q = old_locked["chosen"]["threshold_q"]
        self.assertEqual(
            (original / "pixel_forest_model.json").read_bytes(),
            (CONTROL / "model/pixel_forest_model.json").read_bytes(),
        )
        expected_header = (
            (original / "pixel_forest_model.h")
            .read_text(encoding="utf-8")
            .replace(f"#define PF_THRESHOLD_Q {old_q}\n", f"#define PF_THRESHOLD_Q {q}\n")
        )
        self.assertEqual((CONTROL / "model/pixel_forest_model.h").read_text(encoding="utf-8"), expected_header)

    @unittest.skipUnless((CONTROL / "evaluation/summary.json").exists(), "Verify the compiled control first")
    def test_control_verification_preserves_existing_evidence(self):
        path = CONTROL / "evaluation/summary.json"
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            verify_control(RESULTS_DIR / "forest_lowfp")
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
