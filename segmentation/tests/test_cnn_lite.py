"""Lite ESP-DL CNN (run_02): architecture contract, exploration record and the two locked thresholds.

Needs torch; run with the .venv-espdl interpreter. Skipped where torch is absent.
"""

import io
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from uwseg.paths import REPO_ROOT, RESULTS_DIR

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from uwseg import compare_cnn_lite, train_cnn_lite
    from uwseg.cnn_lite import ARCHITECTURES, build
    from uwseg.cnn_seg import count_macs
    from uwseg.optimize_pixel_forest import constrained_threshold
    from uwseg.train_cnn_espdl import write_header

DEPLOYED_RUN = RESULTS_DIR / "cnn_espdl_run_02"
REFERENCE_RUN = RESULTS_DIR / "cnn_espdl_run_01"
# The deployed run quantized for the ESP32-P4: the model files the firmware embeds.
P4_EXPORT = RESULTS_DIR / "esp32p4_export"
FOREST_CAP = 0.07879003161275763


def synthetic_curve() -> dict:
    """257 thresholds over 1000 non-water and 1000 water pixels: higher thresholds miss more, false-alarm less."""
    q = np.arange(257)
    fn = np.round(1000 * (q / 256) ** 2).astype(np.int64)
    fp = np.round(1000 * (1 - q / 256) ** 0.5).astype(np.int64)
    counts = np.column_stack((1000 - fn, fp, fn, 1000 - fp))
    return {"macro_image_miou": (1 - (fn + fp) / 2000).tolist(), "counts": counts.tolist()}


def locked_run(folder: Path) -> tuple[dict, dict]:
    """A minimal run directory whose lock follows its curves and header."""
    curves = {"int8": synthetic_curve(), "float": synthetic_curve()}
    (folder / "calibration_curves.json").write_text(json.dumps(curves), encoding="utf-8")

    def choose(kind, cap):
        return constrained_threshold(np.array(curves[kind]["counts"]), np.array(curves[kind]["macro_image_miou"]), cap)

    chosen, comparison = choose("int8", 0.05), choose("int8", FOREST_CAP)
    plan = {
        "architecture": "lite",
        "fnr_cap": 0.05,
        "operating_points": {"deployment": {"fnr_cap": 0.05}, "comparison": {"fnr_cap": FOREST_CAP}},
    }
    locked = {
        "architecture": "lite",
        "fnr_cap": 0.05,
        "input_exponent": -5,
        "output_exponent": -3,
        "mean": [60.5, 106.25, 119.0],
        "std": [55.0, 54.75, 61.5],
        "chosen": {**chosen, "threshold_code": chosen["threshold_q"] - 128},
        "float_reference": choose("float", 0.05),
        "comparison": {
            "fnr_cap": FOREST_CAP,
            "int8": {**comparison, "threshold_code": comparison["threshold_q"] - 128},
            "float": choose("float", FOREST_CAP),
        },
    }
    write_header(
        folder / "water_seg_model_config.h",
        np.array(locked["mean"], np.float32),
        np.array(locked["std"], np.float32),
        -5,
        -3,
        locked["chosen"]["threshold_code"],
    )
    return locked, plan


@unittest.skipUnless(HAS_TORCH, "torch is only installed in .venv-espdl")
class LiteArchitectureTests(unittest.TestCase):
    def test_architectures_keep_the_contract_and_their_recorded_cost(self):
        expected = {"lite": (22_310_400, 16), "lite_dw_down": (12_739_200, 18), "lite_wide": (21_595_200, 18)}
        self.assertEqual(set(expected), set(ARCHITECTURES))
        for name, cost in expected.items():
            model = build(name).eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.zeros(2, 3, 120, 160)).shape), (2, 1, 60, 80), name)
            self.assertEqual((count_macs(model), train_cnn_lite.conv_layers(model)), cost, name)
            convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
            for conv in convs:  # depthwise layers are per-channel 3x3; widths stay on 16-lane int8 vectors
                if conv.groups > 1:
                    self.assertEqual(
                        (conv.groups, conv.out_channels, conv.kernel_size), (conv.in_channels, conv.in_channels, (3, 3))
                    )
                self.assertTrue(conv.out_channels % 16 == 0 or conv is convs[-1], name)
            self.assertGreater(sum(conv.groups > 1 for conv in convs), 0)

    def test_onnx_export_needs_only_ops_esp_dl_supports(self):
        import onnx

        buffer = io.BytesIO()
        torch.onnx.export(
            build("lite").eval(),
            torch.zeros(1, 3, 120, 160),
            buffer,
            input_names=["image"],
            output_names=["logits"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
        ops = {node.op_type for node in onnx.load_from_string(buffer.getvalue()).graph.node}
        self.assertLessEqual(ops, {"Add", "Constant", "Conv", "Identity", "Relu", "Resize"})

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaises(ValueError):
            build("lite_huge")


@unittest.skipUnless(HAS_TORCH, "torch is only installed in .venv-espdl")
class LockProtocolTests(unittest.TestCase):
    def test_exploration_record_must_name_the_architecture_and_match_its_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            exploration = Path(folder)
            for name in train_cnn_lite.EXPLORATION_FILES[1:]:
                (exploration / name).write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                train_cnn_lite.load_selection(exploration, "lite", 22_310_400, 16)
            record = {"chosen": "lite", "candidates": {"lite": {"macs": 22_310_400, "conv_layers": 16}}}
            (exploration / "selection.json").write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(train_cnn_lite.load_selection(exploration, "lite", 22_310_400, 16)["chosen"], "lite")
            with self.assertRaises(ValueError):
                train_cnn_lite.load_selection(exploration, "lite_wide", 21_595_200, 18)
            with self.assertRaises(ValueError):
                train_cnn_lite.load_selection(exploration, "lite", 22_310_400, 18)

    def test_both_thresholds_must_follow_the_curves_and_the_header_the_deployment_one(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            locked, plan = locked_run(run)
            self.assertEqual((locked["chosen"]["threshold_q"], locked["comparison"]["int8"]["threshold_q"]), (57, 71))
            compare_cnn_lite.check_operating_points(run, locked, plan)
            moved = json.loads(json.dumps(locked))
            moved["comparison"]["int8"]["threshold_q"] += 1
            with self.assertRaises(ValueError):
                compare_cnn_lite.check_operating_points(run, moved, plan)
            swapped = json.loads(json.dumps(plan))
            swapped["fnr_cap"] = FOREST_CAP
            with self.assertRaises(ValueError):
                compare_cnn_lite.check_operating_points(run, locked, swapped)
            header = run / "water_seg_model_config.h"
            header.write_text(
                header.read_text(encoding="utf-8").replace("kThresholdCode = -71", "kThresholdCode = -57"),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                compare_cnn_lite.check_operating_points(run, locked, plan)

    def test_training_and_comparison_refuse_populated_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = Path(folder) / "LOCKED.json"
            evidence.write_text("previous", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                train_cnn_lite.run(Path(folder), Path(folder) / "missing.json", "lite", Path(folder))
            with self.assertRaises(FileExistsError):
                compare_cnn_lite.run(Path(folder), Path(folder), Path(folder))
            self.assertEqual(evidence.read_text(encoding="utf-8"), "previous")


@unittest.skipUnless(
    HAS_TORCH and (REFERENCE_RUN / "threshold_cap5/evaluation/summary.json").exists(), "Needs run_01 evaluations"
)
class RecordedReferenceRunTests(unittest.TestCase):
    def test_reference_rows_cover_test_at_both_thresholds_and_match_their_summaries(self):
        plan = json.loads((REFERENCE_RUN / "experiment_plan.json").read_text(encoding="utf-8"))
        audit = json.loads((REPO_ROOT / plan["dedup_manifest"]["path"]).read_text(encoding="utf-8"))
        rows, reference = compare_cnn_lite.reference_rows(audit, plan)
        for method in ("run01_locked", "run01_cap5"):
            self.assertEqual([row["id"] for row in rows if row["method"] == method], audit["test_subsets"]["all"])
        self.assertEqual(set(reference["rates"]), {"run01_cap5", "run01_locked", "forest_threshold_only"})


@unittest.skipUnless(HAS_TORCH and (DEPLOYED_RUN / "LOCKED.json").exists(), "Needs the deployed run's record")
class LockedDeployedRunTests(unittest.TestCase):
    def test_locked_run_follows_its_plan_and_its_p4_export_keeps_its_header(self):
        locked = json.loads((DEPLOYED_RUN / "LOCKED.json").read_text(encoding="utf-8"))
        plan = json.loads((DEPLOYED_RUN / "experiment_plan.json").read_text(encoding="utf-8"))
        compare_cnn_lite.check_operating_points(DEPLOYED_RUN, locked, plan)
        self.assertEqual(plan["architecture_selection"]["chosen"], locked["architecture"])
        self.assertEqual(plan["operating_points"]["deployment"]["fnr_cap"], 0.05)
        self.assertEqual(plan["operating_points"]["comparison"]["fnr_cap"], FOREST_CAP)
        header = (DEPLOYED_RUN / "water_seg_model_config.h").read_text(encoding="utf-8")
        self.assertIn(f"constexpr int kThresholdCode = {locked['chosen']['threshold_code']};", header)
        # The P4 export keeps the run's normalisation, exponents and threshold, and its operator list keeps the
        # P4's depthwise kernels.
        self.assertEqual((P4_EXPORT / "water_seg_model_config.h").read_text(encoding="utf-8"), header)
        compile_req = (P4_EXPORT / "water_seg.compile_req.yml").read_text(encoding="utf-8")
        self.assertIn("dl_esp32p4_s8_depthwise_conv2d_33c1_relu", compile_req)


if __name__ == "__main__":
    unittest.main()
