"""ESP32-P4 export: kernel and operator lists, threshold rescaling, the lock check, and the committed export.

Running the export itself (ESP-PPQ quantization) needs the prepared data; these checks do not. Needs torch; skipped
where it is absent.
"""

import json
import re
import shutil
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from uwseg.paths import RESULTS_DIR

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    from uwseg import export_p4
    from uwseg.train_cnn_espdl import P4_INT8_CONV_KERNELS

DEPLOYED_RUN = RESULTS_DIR / "cnn_espdl_run_02"
P4_EXPORT = RESULTS_DIR / "esp32p4_export"

SPEC = """kernel_abi: 1
symbols:
  - dl_c_s8_conv2d_11cn
  - dl_esp32p4_s8_conv2d_33cn_relu
  - dl_esp32p4_s8_add
  - dl_tie728_s8_conv2d_33cn
  - dl_esp32p4_s8_depthwise_conv2d_33c1
  - dl_esp32p4_s8_conv2d_33cn_relu
"""


@unittest.skipUnless(HAS_TORCH, "export_p4 needs torch")
class ExportHelperTests(unittest.TestCase):
    def test_kernels_are_the_p4_int8_convolutions_of_the_spec(self):
        self.assertEqual(
            export_p4.p4_conv_kernels(SPEC), ["dl_esp32p4_s8_conv2d_33cn_relu", "dl_esp32p4_s8_depthwise_conv2d_33c1"]
        )

    def test_graph_ops_are_the_info_operators_and_requantize(self):
        info = "  %a = Conv[activation = 'Relu'](%x)\n  %b = Resize[mode = 'nearest'](%a)\n  %c = Add[](%b, %a)\n"
        self.assertEqual(export_p4.graph_ops(info), ["Add", "Conv", "RequantizeLinear", "Resize"])

    def test_threshold_keeps_the_logit_across_output_exponents(self):
        self.assertEqual(export_p4.rescaled_threshold(3, -3, -3), 3)
        self.assertEqual(export_p4.rescaled_threshold(3, -3, -4), 6)
        self.assertEqual(export_p4.rescaled_threshold(4, -3, -2), 2)

    def test_export_refuses_a_populated_output_and_a_changed_model(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "out"
            out.mkdir()
            (out / "export.json").write_text("previous", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_p4.export(DEPLOYED_RUN, Path(folder) / "kernels.yml", Path(folder) / "dedup.json", out)
            self.assertEqual((out / "export.json").read_text(encoding="utf-8"), "previous")
            run = Path(folder) / "run"
            run.mkdir()
            for name in ("LOCKED.json", *export_p4.EXPORTED_ARTIFACTS):
                shutil.copyfile(DEPLOYED_RUN / name, run / name)
            export_p4.locked_run(run)  # an untouched copy verifies
            onnx = bytearray((run / "water_seg.onnx").read_bytes())
            onnx[-1] ^= 1
            (run / "water_seg.onnx").write_bytes(bytes(onnx))
            with self.assertRaises(ValueError):
                export_p4.locked_run(run)


@unittest.skipUnless(HAS_TORCH and (P4_EXPORT / "export.json").exists(), "Needs the committed ESP32-P4 export")
class CommittedExportTests(unittest.TestCase):
    def test_export_lists_the_p4_kernels_and_keeps_the_locked_threshold(self):
        summary = json.loads((P4_EXPORT / "export.json").read_text(encoding="utf-8"))
        locked = json.loads((DEPLOYED_RUN / "LOCKED.json").read_text(encoding="utf-8"))
        lines = (P4_EXPORT / "water_seg.compile_req.yml").read_text(encoding="utf-8").splitlines()
        self.assertEqual([line[4:] for line in lines if line.startswith("  - dl_")], P4_INT8_CONV_KERNELS)
        self.assertEqual(summary["ops"], ["Add", "Conv", "RequantizeLinear", "Resize"])
        self.assertEqual((summary["target"], summary["kernels"]), ("esp32p4", len(P4_INT8_CONV_KERNELS)))
        expected = export_p4.rescaled_threshold(
            locked["chosen"]["threshold_code"], locked["output_exponent"], summary["output_exponent"]
        )
        self.assertEqual(summary["threshold_code"], expected)
        header = (P4_EXPORT / "water_seg_model_config.h").read_text(encoding="utf-8")
        self.assertIn(f"constexpr int kThresholdCode = {expected};", header)
        self.assertIn(f"constexpr int kOutputExponent = {summary['output_exponent']};", header)

    def test_self_test_frame_is_the_locked_frame_with_the_recorded_mask(self):
        summary = json.loads((P4_EXPORT / "export.json").read_text(encoding="utf-8"))
        locked = json.loads((DEPLOYED_RUN / "LOCKED.json").read_text(encoding="utf-8"))
        text = (P4_EXPORT / "water_seg_test_frame.h").read_text(encoding="utf-8")
        self.assertEqual(summary["test_frame"], locked["test_frame_id"])
        self.assertIn(f'kTestFrameId[] = "{locked["test_frame_id"]}";', text)
        mask = re.search(r"kTestFrameExpectedMask\[(\d+)\] = \{([^}]*)\}", text)
        cells = [int(value) for value in mask.group(2).split(",")]
        self.assertEqual((int(mask.group(1)), len(cells), set(cells) <= {0, 1}), (80 * 60, 80 * 60, True))
        self.assertEqual(sum(cells), summary["test_mask_non_water"])


if __name__ == "__main__":
    unittest.main()
