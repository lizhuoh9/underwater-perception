"""The build configurations of core/mono_vo.h (uwslam.mvo.CONFIGS): what each sets in C (the evidence penalty's 2 ln(4n)
among it), the Python build's check of them, an unknown one refused, and more than 255 features keeping their minimum
distance (the corner grid's indices widen with the count).

Needs gcc, and OpenCV for the textured frame; skipped without either.
"""

import ctypes
from importlib.util import find_spec
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from uwslam import mvo, mvo_stream, voxel_map
from uwslam.core import CFLAGS

ROOT = Path(__file__).resolve().parents[1]
STATIC = ["-static"] if sys.platform == "win32" else []  # MinGW: no runtime DLLs to find when the program runs


def float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def built(config: str, folder: Path) -> ctypes.CDLL:
    """The core built in `config`, as UWSLAM_CONFIG=<config> builds it."""
    saved = mvo.CONFIG, mvo.MAX_FEATURES
    mvo.CONFIG, mvo.MAX_FEATURES = config, mvo.CONFIGS[config]["max_features"]
    try:
        return mvo.build(folder)
    finally:
        mvo.CONFIG, mvo.MAX_FEATURES = saved


@unittest.skipUnless(shutil.which("gcc") is not None, "Needs gcc")
class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)  # Windows keeps loaded libraries locked

    def test_each_configuration_sets_its_count_and_map_admission(self):
        program = self.folder / "mvo_config_check.exe"
        for config, defines, spread in (("s3", [], 3.0e-4), ("p4", ["-DMVO_CONFIG_P4"], 3.0e-4)):  # one admission
            with self.subTest(config=config):
                subprocess.run(["gcc", *CFLAGS, *defines, *STATIC, f"-I{ROOT / 'core'}",
                                str(ROOT / "tests/c/mvo_config_check.c"), "-o", str(program)], check=True)
                output = subprocess.run([str(program)], check=True, capture_output=True, text=True).stdout
                fields = dict(word.split("=", 1) for word in output.split())
                self.assertEqual(fields["config"], config)
                self.assertEqual(int(fields["max_features"]), mvo.CONFIGS[config]["max_features"])
                for admission in ("mm", "vm", "mc"):  # the obstacle map, the voxel map and the cloud alike
                    self.assertEqual(int(fields[admission]), float_bits(spread))
                evidence = struct.unpack("<f", struct.pack("<I", int(fields["evidence_log"])))[0]
                count = mvo.CONFIGS[config]["max_features"]
                self.assertAlmostEqual(evidence, 2 * math.log(4 * count), delta=5e-4)  # the penalty's 2 ln(4n)
                # what the Python side repeats: the tracking window, the voxel map's layout, the plan's grouping
                self.assertEqual((int(fields["window"]), int(fields["lanes"])),
                                 (mvo.CONFIGS[config]["window"], mvo.CONFIGS[config]["lanes"]))
                capacity, slots, cells = voxel_map.LAYOUTS[config]
                layout = (int(fields["voxel_capacity"]), int(fields["voxel_slots"]))
                self.assertEqual(layout, (capacity, slots))
                self.assertEqual(layout, mvo_stream.VOXEL_LAYOUTS[config])
                self.assertEqual(int(fields["cells_per_depth"]), float_bits(float(cells)))
                self.assertEqual(int(fields["plan_group"]), 2 if config == "p4" else 1)

    def test_the_build_reports_its_configuration(self):
        for config in mvo.CONFIGS:
            with self.subTest(config=config):
                lib = built(config, self.folder / config)
                self.assertEqual(lib.mvo_config_name().decode(), config)
                self.assertEqual(lib.mvo_max_features(), mvo.CONFIGS[config]["max_features"])

    def test_an_unknown_configuration_is_refused(self):
        env = dict(os.environ, UWSLAM_CONFIG="s4", PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run([sys.executable, "-c", "import uwslam.mvo"], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("UWSLAM_CONFIG", result.stderr)

    @unittest.skipUnless(find_spec("cv2") is not None, "Needs OpenCV")
    def test_more_than_255_features_keep_their_distance(self):
        # The corner grid kept feature indices in bytes: past 255 they wrapped onto other features, and the minimum-
        # distance test looked at the wrong ones.
        import cv2
        lib = built("p4", self.folder / "p4_grid")
        rng = np.random.default_rng(7)
        frame = cv2.GaussianBlur(rng.integers(0, 256, (mvo.HEIGHT, mvo.WIDTH), dtype=np.uint8), (5, 5), 1.5)
        state = ctypes.create_string_buffer(lib.mvo_state_size())
        pyramids = [ctypes.create_string_buffer(lib.mvo_pyramid_bytes()) for _ in range(2)]
        camera = mvo.camera_for(110.0, 544 / 960)
        lib.mvo_init(state, ctypes.byref(camera), 1, pyramids[0], pyramids[1])
        ctypes.memmove(lib.mvo_frame(state), np.ascontiguousarray(frame).ctypes.data, mvo.WIDTH * mvo.HEIGHT)
        lib.mvo_start(state)
        capacity = mvo.CONFIGS["p4"]["max_features"]
        xy, ids = (ctypes.c_float * (2 * capacity))(), (ctypes.c_uint32 * capacity)()
        count = lib.mvo_tracks(state, xy, ids, capacity)
        self.assertGreater(count, 255)
        points = np.array(xy[:2 * count], dtype=np.float64).reshape(-1, 2)
        gaps = np.linalg.norm(points[:, None] - points[None], axis=2) + np.eye(count) * 1e9
        self.assertGreaterEqual(gaps.min(), 8.0)  # MVO_MIN_DISTANCE


if __name__ == "__main__":
    unittest.main()
