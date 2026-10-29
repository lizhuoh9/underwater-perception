"""Regression checks for the portable C ABI and its Python integration."""

import ctypes as ct
import json
import unittest

import numpy as np

from uwseg.compare_pixel_forest import check_native, run_comparison
from uwseg.forest_native import NativeForest
from uwseg.paths import RESULTS_DIR

RUN = RESULTS_DIR / "forest_comparison"


@unittest.skipUnless((RUN / "pixel_forest.dll").exists(), "Build the host C library first (see embedded/README.md)")
class NativeForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = NativeForest(RUN / "pixel_forest.dll")

    def test_exact_integer_features_scores_and_masks(self):
        model = json.loads((RUN / "model/pixel_forest_model.json").read_text(encoding="utf-8"))
        locked = json.loads((RUN / "LOCKED.json").read_text(encoding="utf-8"))
        self.assertEqual(check_native(self.native, model, locked["chosen"]["threshold_q"]), 5)

    def test_c_rejects_bad_dimensions_workspace_and_overlap(self):
        lib = self.native.library
        self.assertEqual(lib.pf_workspace_bytes(161, 120), 0)
        self.assertEqual(lib.pf_workspace_bytes(0, 1), 0)
        size = lib.pf_workspace_bytes(1, 1)
        packed = np.array([0xF800], dtype=np.uint16)
        scores, masks = np.empty(1, dtype=np.uint8), np.empty(1, dtype=np.uint8)
        work = np.empty((size + 7) // 4, dtype=np.uint32)
        args = [packed.ctypes.data, 1, 1, scores.ctypes.data, masks.ctypes.data, work.ctypes.data, size]
        self.assertEqual(lib.pf_predict_rgb565(*args), 0)
        bad = args.copy()
        bad[0] = None
        self.assertEqual(lib.pf_predict_rgb565(*bad), -1)
        bad = args.copy()
        bad[-1] = size - 1
        self.assertEqual(lib.pf_predict_rgb565(*bad), -3)
        bad = args.copy()
        bad[-2] += 1
        self.assertEqual(lib.pf_predict_rgb565(*bad), -2)
        bad = args.copy()
        bad[4] = bad[3]
        self.assertEqual(lib.pf_predict_rgb565(*bad), -4)
        # Oversize feature-row multiplication is rejected before dereferencing.
        maximum = ct.c_size_t(-1).value
        self.assertEqual(lib.pf_predict_features(packed.ctypes.data, maximum // 16 + 1, scores.ctypes.data), -1)

    def test_existing_evaluation_is_preserved(self):
        path = RUN / "evaluation/summary.json"
        if not path.exists():
            self.skipTest("Run the comparison first")
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            run_comparison(RUN)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
