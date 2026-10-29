import unittest

import numpy as np

from uwseg.pixel_forest import FEATURE_NAMES, box_mean, extract_features, predict_features, validate_model


class PixelForestTests(unittest.TestCase):
    def test_box_mean_matches_replicated_windows(self):
        image = np.random.default_rng(3).integers(0, 256, (5, 7), dtype=np.uint8)
        padded = np.pad(image, 4, mode="edge")
        expected = np.array(
            [[padded[y : y + 9, x : x + 9].sum() // 81 for x in range(7)] for y in range(5)], dtype=np.uint8
        )
        np.testing.assert_array_equal(box_mean(image, 9), expected)

    def test_constant_black_features_have_no_wraparound(self):
        features = extract_features(np.zeros((2, 3, 3), dtype=np.uint8))
        self.assertEqual(features.shape, (2, 3, 16))
        np.testing.assert_array_equal(features[..., [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15]], 0)
        np.testing.assert_array_equal(features[..., 7:9], 127)

    def test_integer_soft_vote_and_split_boundary(self):
        model = {
            "feature_names": FEATURE_NAMES,
            "max_depth": 1,
            "roots": [0, 3],
            "nodes": [[1, 2, 0, 100, 0], [0, 0, 255, 0, 20], [0, 0, 255, 0, 220], [0, 0, 255, 0, 41]],
        }
        validate_model(model)
        features = np.zeros((2, 16), dtype=np.uint8)
        features[:, 0] = [100, 101]
        np.testing.assert_array_equal(predict_features(features, model), [31, 131])

    def test_model_rejects_cycle(self):
        model = {"feature_names": FEATURE_NAMES, "max_depth": 1, "roots": [0], "nodes": [[0, 0, 0, 100, 0]]}
        with self.assertRaises(ValueError):
            validate_model(model)


if __name__ == "__main__":
    unittest.main()
