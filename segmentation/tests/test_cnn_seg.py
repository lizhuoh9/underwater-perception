"""ESP-DL CNN pipeline: device-exact input emulation, network contract and export helpers.

Needs torch; run with the .venv-espdl interpreter. Skipped where torch is absent.
"""

import json
import re
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from uwseg.paths import RESULTS_DIR

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from uwseg import compare_cnn_espdl, export_cnn_threshold_variant, train_cnn_espdl
    from uwseg.cnn_seg import RGB565_KEEP, WaterSegNet, count_macs, int8_scores, model_frame
    from uwseg.optimize_pixel_forest import constrained_threshold

VARIANT = RESULTS_DIR / "cnn_espdl_run_01/threshold_cap5"


def esp_dl_expand(rgb):
    """esp-dl dl_image_color_common.hpp: RGB565 channels widened by shift, no bit replication."""
    packed = (
        ((rgb[..., 0].astype(np.uint16) >> 3) << 11) | ((rgb[..., 1].astype(np.uint16) >> 2) << 5) | (rgb[..., 2] >> 3)
    )
    return np.stack([(packed & 0xF800) >> 8, (packed & 0x7E0) >> 3, (packed & 0x1F) << 3], axis=-1).astype(np.uint8)


def codes_for(values, mean, std, exponent):
    """device_codes on one row of pixels; returns codes per channel (3 x pixels)."""
    frames = np.array(values, dtype=np.uint8).reshape(1, 1, -1, 3)
    return train_cnn_espdl.device_codes(frames, np.array(mean, np.float32), np.array(std, np.float32), exponent)[
        0, :, 0, :
    ]


@unittest.skipUnless(HAS_TORCH, "torch is only installed in .venv-espdl")
class CnnSegTests(unittest.TestCase):
    def test_model_frame_matches_esp_dl_rgb565_expansion(self):
        work = np.random.default_rng(5).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        frame = model_frame(work)
        np.testing.assert_array_equal(frame, esp_dl_expand(frame))
        pooled = np.rint(work.reshape(120, 2, 160, 2, 3).astype(np.float32).mean(axis=(1, 3))).astype(np.uint8)
        np.testing.assert_array_equal(frame, pooled & RGB565_KEEP)

    def test_device_codes_round_half_to_even_on_exact_ties(self):
        # Exponent +1 halves (v - mean), so odd differences land exactly on .5; the P4 rounds them to even.
        pixels = [[5, 5, 0], [3, 7, 1], [255, 255, 254], [0, 0, 255]]
        codes = codes_for(pixels, mean=[10, 0, 255], std=[1, 1, 1], exponent=1)
        np.testing.assert_array_equal(codes[0], [-2, -4, 122, -5])  # -2.5 -> -2, -3.5 -> -4, 122.5 -> 122
        np.testing.assert_array_equal(codes[1], [2, 4, 127, 0])  # 2.5 -> 2, 3.5 -> 4, 127.5 -> 128 clamps to 127
        np.testing.assert_array_equal(codes[2], [-128, -127, 0, 0])  # -127.5 -> -128, -0.5 -> 0

    def test_device_codes_clamp_both_ends_and_handle_inexact_std(self):
        np.testing.assert_array_equal(codes_for([[0, 255, 0]], [255, 0, 0], [1, 1, 1], -1)[:, 0], [-128, 127, 0])
        # 1/3 is inexact in float32: (v - 1) * 0.33333334 for v = 4, 5, 6, 1 -> 1.0000001, 1.33, 1.67, 0.
        np.testing.assert_array_equal(
            codes_for([[4, 0, 0], [5, 0, 0], [6, 0, 0], [1, 0, 0]], [1, 0, 0], [3, 1, 1], 0)[0], [1, 1, 2, 0]
        )

    def test_network_contract_shapes_and_macs(self):
        model = WaterSegNet().eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(2, 3, 120, 160)).shape), (2, 1, 60, 80))
        self.assertEqual(count_macs(model), 69_734_400)

    def test_int8_scores_shift_and_clamp_output_codes(self):
        logits = torch.tensor([-200.0, -16.0, -0.0625, 0.0, 0.0625, 15.9375, 99.0]).view(1, 1, 1, 7)
        np.testing.assert_array_equal(int8_scores(logits, -4)[0, 0], [0, 0, 127, 128, 129, 255, 255])

    def test_compile_req_lists_graph_ops_and_every_p4_int8_conv_kernel(self):
        info = "  %a = Conv[activation = 'Relu', kernel_shape = [3, 3]](%x)\n  %b = Resize[mode = 'nearest'](%a)\n  %c = Add[quant_type = 'S8'](%b, %a)\n  %w[INT8, 16x3x3x3]\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.compile_req.yml"
            ops = train_cnn_espdl.write_compile_req(path, info)
            text = path.read_text(encoding="utf-8")
        self.assertEqual(ops, ["Add", "Conv", "RequantizeLinear", "Resize"])
        kernels = [line[4:] for line in text.splitlines() if line.startswith("  - dl_")]
        self.assertEqual(len(kernels), 96)
        self.assertTrue(all(k.startswith("dl_esp32p4_s8_") for k in kernels))
        self.assertEqual(kernels, train_cnn_espdl.P4_INT8_CONV_KERNELS)
        self.assertIn("kernel_abi: 1", text)

    def test_header_floats_are_valid_cpp_literals_and_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "water_seg_model_config.h"
            train_cnn_espdl.write_header(
                path,
                np.array([56.0, 63.2324028, 1e-5], np.float32),
                np.array([1.0, 55.1069756, 2.5], np.float32),
                -5,
                -3,
                -128,
            )
            text = path.read_text(encoding="utf-8")
        mean = re.search(r"kMean = \{([^}]+)\}", text).group(1).split(", ")
        self.assertEqual(mean[0], "56.0f")
        self.assertTrue(all(re.fullmatch(r"-?\d+(\.\d*)?(e[-+]?\d+)?f|-?\d+e[-+]?\d+f", v) for v in mean), mean)
        self.assertEqual(
            [np.float32(v.rstrip("f")) for v in mean], [np.float32(56.0), np.float32(63.2324028), np.float32(1e-5)]
        )
        self.assertIn("constexpr int kThresholdCode = -128;", text)

    def test_test_frame_packs_rgb565_values_and_mask(self):
        frame = np.array([[[248, 0, 0], [0, 252, 0], [0, 0, 248]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frame.h"
            train_cnn_espdl.write_test_frame(path, frame, np.array([[1, 0]], dtype=np.uint8), "x")
            text = path.read_text(encoding="utf-8")
        self.assertIn("kTestFrameRgb565[3] = {63488,2016,31};", text)
        self.assertIn("kTestFrameExpectedMask[2] = {1,0};", text)

    def test_compare_reports_candidate_minus_reference(self):
        def row(sample, method, tp, fp, fn, tn, miou):
            return {"id": sample, "method": method, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "miou": miou}

        rows = [
            row("a", "new", 90, 5, 10, 95, 0.9),
            row("a", "old", 95, 30, 5, 70, 0.7),
            row("b", "new", 80, 10, 20, 90, 0.6),
            row("b", "old", 85, 20, 15, 80, 0.6),
        ]
        result = compare_cnn_espdl.compare(rows, ["a", "b"], "new", "old")
        self.assertAlmostEqual(result["fnr_percentage_points"], (30 / 200 - 20 / 200) * 100)
        self.assertAlmostEqual(result["fpr_percentage_points"], (15 / 200 - 50 / 200) * 100)
        self.assertLessEqual(result["fpr_ci95_pp"][0], result["fpr_ci95_pp"][1])
        self.assertEqual(result["paired_images"], {"improved": 1, "regressed": 0, "tied": 1})

    def test_augment_keeps_image_and_label_registered(self):
        generator = torch.Generator().manual_seed(1)
        images = torch.zeros(4, 3, 240, 320, dtype=torch.uint8)
        images[:, :, :, :120] = 220  # bright left part is the labelled object
        labels = torch.zeros(4, 1, 240, 320, dtype=torch.uint8)
        labels[:, :, :, :120] = 1
        mean, std = torch.tensor([60.0, 100.0, 120.0]), torch.tensor([50.0, 55.0, 60.0])
        inputs, target = train_cnn_espdl.augment(images, labels, mean, std, generator)
        self.assertEqual((tuple(inputs.shape), tuple(target.shape)), ((4, 3, 120, 160), (4, 1, 60, 80)))
        device_units = torch.round(inputs * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)).to(torch.int64)
        keep = torch.tensor(RGB565_KEEP, dtype=torch.int64).view(1, 3, 1, 1)
        self.assertTrue(bool(((device_units & keep) == device_units).all()))
        brightness = torch.nn.functional.avg_pool2d(device_units.float().mean(dim=1, keepdim=True), 2)
        for sample in range(4):  # flips/crops must move label and image together
            on, off = target[sample] == 1, target[sample] == 0
            if on.any() and off.any():
                self.assertGreater(float(brightness[sample][on].mean()), float(brightness[sample][off].mean()) + 50)

    def test_training_and_comparison_refuse_populated_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = Path(folder) / "LOCKED.json"
            evidence.write_text("previous", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                train_cnn_espdl.run(Path(folder), Path(folder) / "missing.json")
            with self.assertRaises(FileExistsError):
                compare_cnn_espdl.run(Path(folder), Path(folder), Path(folder))
            with self.assertRaises(FileExistsError):
                export_cnn_threshold_variant.run(Path(folder), Path(folder), 0.05, Path(folder))
            self.assertEqual(evidence.read_text(encoding="utf-8"), "previous")


@unittest.skipUnless(HAS_TORCH and (VARIANT / "MODEL_EXPORT.json").exists(), "Export the threshold variant first")
class ThresholdVariantTests(unittest.TestCase):
    """run_01's post-hoc variant. Its model files (an ESP32-S3 export) are not in this repository; its record is."""

    def test_variant_threshold_follows_the_calibration_rule(self):
        run = RESULTS_DIR / "cnn_espdl_run_01"
        export = json.loads((VARIANT / "MODEL_EXPORT.json").read_text(encoding="utf-8"))
        curves = json.loads((run / "calibration_curves.json").read_text(encoding="utf-8"))["int8"]
        choice = constrained_threshold(
            np.array(curves["counts"]), np.array(curves["macro_image_miou"]), export["fnr_cap"]
        )
        self.assertEqual(
            (choice["threshold_q"], choice["threshold_q"] - 128), (export["threshold_q"], export["threshold_code"])
        )
        self.assertTrue(json.loads((VARIANT / "evaluation/summary.json").read_text(encoding="utf-8"))["post_hoc"])


if __name__ == "__main__":
    unittest.main()
