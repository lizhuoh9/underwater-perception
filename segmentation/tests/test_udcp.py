import tempfile
import unittest
from pathlib import Path

import numpy as np

from uwseg.udcp import confusion, decode_labels, metrics, min_filter, rgb565_roundtrip, transmission
from uwseg.validate import choose_threshold, run


class UdcpTests(unittest.TestCase):
    def test_separable_min_matches_neighborhood_definition(self):
        values = np.random.default_rng(7).random((7, 11)).astype(np.float32)
        padded = np.pad(values, 2, mode="edge")
        expected = np.array([[padded[y : y + 5, x : x + 5].min() for x in range(11)] for y in range(7)])
        np.testing.assert_array_equal(min_filter(values, 5), expected)
        with self.assertRaises(ValueError):
            min_filter(values, 4)

    def test_known_optical_model_recovers_transmission(self):
        ambient = np.array([0.8, 0.7, 0.6], dtype=np.float32)
        original = np.zeros((9, 12, 3), dtype=np.float32)
        original[..., 0] = 0.4
        original[..., 1] = 0.3
        # The blue channel is exactly dark, satisfying the prior at every pixel.
        for known_t in (0.2, 0.5, 0.8):
            observed = original * known_t + ambient * (1 - known_t)
            actual = transmission(observed, 3, ambient)
            np.testing.assert_allclose(actual, known_t, atol=1e-6)

    def test_rgb_palette_and_small_encoding_errors(self):
        ids = np.arange(8, dtype=np.uint8)
        palette = np.stack(((ids >> 2) & 1, (ids >> 1) & 1, ids & 1), axis=-1) * 255
        np.testing.assert_array_equal(decode_labels(palette[None, ...]), ids[None, ...])
        noisy = np.where(palette == 0, 3, 252).astype(np.uint8)
        np.testing.assert_array_equal(decode_labels(noisy[None, ...]), ids[None, ...])

    def test_downloaded_mirror_grayscale_class_ids(self):
        ids = np.arange(8, dtype=np.uint8)[None, :]
        np.testing.assert_array_equal(decode_labels(ids), ids)
        np.testing.assert_array_equal(decode_labels(np.repeat(ids[..., None], 3, axis=2)), ids)

    def test_metrics_penalize_all_foreground(self):
        truth = np.array([[True, False], [True, False]])
        counts = confusion(np.ones_like(truth), truth)
        result = metrics(counts)
        self.assertEqual(result["foreground_iou"], 0.5)
        self.assertEqual(result["water_iou"], 0)
        self.assertEqual(result["miou"], 0.25)
        self.assertEqual(result["foreground_recall"], 1)
        self.assertEqual(result["water_false_positive_rate"], 1)
        self.assertEqual(metrics(confusion(truth, truth))["miou"], 1)

    def test_empty_foreground_is_undefined_not_perfect_iou(self):
        empty = np.zeros((3, 4), dtype=bool)
        result = metrics(confusion(empty, empty))
        self.assertIsNone(result["foreground_iou"])
        self.assertEqual(result["water_iou"], 1)

    def test_rgb565_has_expected_channel_precision(self):
        image = np.array([[[255, 255, 255], [0, 0, 0], [129, 130, 131]]], dtype=np.uint8)
        converted = rgb565_roundtrip(image)
        np.testing.assert_array_equal(converted[0, :2], image[0, :2])
        self.assertLessEqual(np.abs(converted.astype(int) - image.astype(int)).max(), 7)

    def test_histogram_threshold_search_matches_direct_masks(self):
        score = np.array([[10, 150], [200, 90]], dtype=np.uint8)
        labels = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        threshold, objective, curve = choose_threshold([(score, labels)])
        brute = np.array([metrics(confusion(score >= q, labels != 0))["miou"] for q in range(257)])
        np.testing.assert_allclose(curve, brute)
        self.assertEqual(threshold, int(brute.argmax()))
        self.assertEqual(objective, float(brute.max()))
        empty = np.zeros_like(score)
        self.assertEqual(choose_threshold([(empty, empty)])[0], 1)

    def test_existing_results_are_never_overwritten(self):
        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            output = Path(folder).resolve()
            self.assertTrue(output.is_relative_to(workspace))
            evidence = output / "summary.json"
            evidence.write_text("previous evidence", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(output / "missing_manifest.json", output)
            self.assertEqual(evidence.read_text(encoding="utf-8"), "previous evidence")


if __name__ == "__main__":
    unittest.main()
