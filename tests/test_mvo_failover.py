"""Recover a known weak-homography translation without inventing motion in negative cases."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from uwslam.core import CFLAGS
from uwslam.mvo import config_defines
from uwslam.paths import CORE_DIR


@unittest.skipUnless(shutil.which("gcc"), "gcc is needed")
class MvoFailoverTests(unittest.TestCase):
    def test_weak_homography_translation_and_noisy_turns(self):
        source = Path(__file__).resolve().parent / "c" / "mvo_failover_check.c"
        units = [CORE_DIR / name for name in ("mono_vo.c", "mono_vo_numerics.c", "mono_vo_homography.c",
                                               "mono_vo_essential.c")]
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "mvo_failover_check.exe"
            subprocess.run(["gcc", *CFLAGS, *config_defines(), *(["-static"] if sys.platform == "win32" else []),
                            f"-I{CORE_DIR}", str(source), *map(str, units), "-o", str(binary)], check=True)
            output = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
        values = {key: float(value) for key, value in (part.split("=") for part in output.split())}
        self.assertLess(values["homography"], 15, output)
        self.assertGreaterEqual(values["essential"], 15, output)
        self.assertEqual(values["kind"], 2, output)
        self.assertGreater(values["x"], 0.99, output)
        self.assertLess(abs(values["y"]), 0.01, output)
        self.assertLess(abs(values["z"]), 0.01, output)
        self.assertEqual(values["unrelated_kind"], 0, output)
        self.assertEqual([values["unrelated_x"], values["unrelated_y"], values["unrelated_z"]], [0.0, 0.0, 0.0], output)
        self.assertGreater(values["weak_cases"], 0, output)
        self.assertEqual(values["false_turns"], 0, output)
        self.assertEqual(values["weak_false"], 0, output)
        self.assertEqual(values["rng_bad"], 0, output)


if __name__ == "__main__":
    unittest.main()
