"""The build's constants in core/mono_vo.h and the headers after it: what the C sets (the evidence penalty's 2 ln(4n)
among it) and what the Python side repeats agree, the build reports itself as the board does, and more than 255
features keep their minimum distance (the corner grid's indices are 16-bit).

Needs gcc, and OpenCV for the textured frame; skipped without either.
"""

import ctypes
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from uwslam import mvo, mvo_stream, voxel_map
from uwslam.core import CFLAGS

ROOT = Path(__file__).resolve().parents[1]
STATIC = ["-static"] if sys.platform == "win32" else []  # MinGW: no runtime DLLs to find when the program runs


def float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


@unittest.skipUnless(shutil.which("gcc") is not None, "Needs gcc")
class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)  # Windows keeps loaded libraries locked

    def test_the_c_constants_are_the_ones_python_repeats(self):
        program = self.folder / "mvo_config_check.exe"
        subprocess.run(
            [
                "gcc",
                *CFLAGS,
                *STATIC,
                f"-I{ROOT / 'core'}",
                str(ROOT / "tests/c/mvo_config_check.c"),
                "-o",
                str(program),
            ],
            check=True,
        )
        output = subprocess.run([str(program)], check=True, capture_output=True, text=True).stdout
        fields = dict(word.split("=", 1) for word in output.split())
        self.assertEqual(fields["config"], mvo.CONFIG)
        self.assertEqual(int(fields["max_features"]), mvo.MAX_FEATURES)
        for admission in ("mm", "vm", "mc"):  # the obstacle map, the voxel map and the cloud alike: 1 degree
            self.assertEqual(int(fields[admission]), float_bits(3.0e-4))
        evidence = struct.unpack("<f", struct.pack("<I", int(fields["evidence_log"])))[0]
        self.assertAlmostEqual(evidence, 2 * math.log(4 * mvo.MAX_FEATURES), delta=5e-4)  # the penalty's 2 ln(4n)
        # what the Python side repeats: the tracking window, the voxel map's layout, the plan's grouping
        self.assertEqual((int(fields["window"]), int(fields["lanes"])), (mvo.WINDOW, mvo.LANES))
        layout = (int(fields["voxel_capacity"]), int(fields["voxel_slots"]))
        self.assertEqual(layout, (voxel_map.CAPACITY, voxel_map.SLOTS))
        self.assertEqual(layout, (mvo_stream.VOXEL_CAPACITY, mvo_stream.VOXEL_SLOTS))
        self.assertEqual(int(fields["cells_per_depth"]), float_bits(float(voxel_map.CELLS_PER_DEPTH)))
        self.assertEqual(int(fields["plan_group"]), 2)

    def test_the_build_reports_itself(self):
        lib = mvo.build(self.folder / "build")
        self.assertEqual(lib.mvo_config_name().decode(), mvo.CONFIG)
        self.assertEqual(lib.mvo_max_features(), mvo.MAX_FEATURES)
        self.assertEqual(mvo_stream.MAX_FEATURES, mvo.MAX_FEATURES)

    @unittest.skipUnless(find_spec("cv2") is not None, "Needs OpenCV")
    def test_more_than_255_features_keep_their_distance(self):
        # The corner grid kept feature indices in bytes: past 255 they wrapped onto other features, and the minimum-
        # distance test looked at the wrong ones.
        import cv2

        lib = mvo.build(self.folder / "grid")
        rng = np.random.default_rng(7)
        frame = cv2.GaussianBlur(rng.integers(0, 256, (mvo.HEIGHT, mvo.WIDTH), dtype=np.uint8), (5, 5), 1.5)
        state = ctypes.create_string_buffer(lib.mvo_state_size())
        pyramids = [ctypes.create_string_buffer(lib.mvo_pyramid_bytes()) for _ in range(2)]
        camera = mvo.camera_for(110.0, 544 / 960)
        lib.mvo_init(state, ctypes.byref(camera), 1, pyramids[0], pyramids[1])
        ctypes.memmove(lib.mvo_frame(state), np.ascontiguousarray(frame).ctypes.data, mvo.WIDTH * mvo.HEIGHT)
        lib.mvo_start(state)
        capacity = mvo.MAX_FEATURES
        xy, ids = (ctypes.c_float * (2 * capacity))(), (ctypes.c_uint32 * capacity)()
        count = lib.mvo_tracks(state, xy, ids, capacity)
        self.assertGreater(count, 255)
        points = np.array(xy[: 2 * count], dtype=np.float64).reshape(-1, 2)
        gaps = np.linalg.norm(points[:, None] - points[None], axis=2) + np.eye(count) * 1e9
        self.assertGreaterEqual(gaps.min(), 8.0)  # MVO_MIN_DISTANCE


if __name__ == "__main__":
    unittest.main()
