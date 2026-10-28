"""Known geometry checks for the conservative monocular obstacle map."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from uwslam.mvo import config_defines


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("gcc"), "gcc is needed")
class MonoMapSafetyTests(unittest.TestCase):
    def test_height_slice_and_crossing_rays(self):
        with tempfile.TemporaryDirectory() as folder:
            program = Path(folder) / "mono_map_safety.exe"
            subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", *config_defines(),
                            f"-I{ROOT / 'core'}", str(ROOT / "tests/c/mono_map_safety_check.c"),
                            str(ROOT / "core/mono_map.c"), "-o", str(program)], check=True)
            subprocess.run([str(program)], check=True)


if __name__ == "__main__":
    unittest.main()
