"""The continuing voxel map (core/voxel_map.c): its C contract, and on rendered views of a wall and a floor that it
files their surfaces, goes on through a tracking loss and files nothing on water."""

import ctypes
from importlib.util import find_spec
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from uwslam.mvo import config_defines

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    from tests.test_mvo import FLOOR_Y, WALL_Z, render, texture
    from uwslam import mvo, voxel_map
    from uwslam.experiments.mask_motion import euler_rotation

ROOT = Path(__file__).resolve().parents[1]
STATIC = ["-static"] if sys.platform == "win32" else []


@unittest.skipUnless(shutil.which("gcc") is not None, "Needs gcc")
class VoxelMapContractTests(unittest.TestCase):
    def test_c_contract(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            program = Path(folder) / "voxel_map_check.exe"
            subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-ffp-contract=off",
                            *config_defines(), *STATIC, f"-I{ROOT / 'core'}", str(ROOT / "tests/c/voxel_map_check.c"),
                            str(ROOT / "core/voxel_map.c"), "-o", str(program)], check=True)
            output = subprocess.run([str(program)], check=True, capture_output=True, text=True).stdout
        self.assertIn("voxel_map_check ok", output)


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs gcc and OpenCV")
class VoxelMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.vo = mvo.build(cls.folder)
        cls.camera = mvo.camera_for(110.0, 544 / 960)
        wall, floor = texture(1), texture(2)
        cls.poses = [(euler_rotation(np.radians(1.5 * k), 0.0, 0.0), np.array([0.30 * k, 0.0, 0.15 * k]))
                     for k in range(10)]
        cls.frames = [render(rotation, centre, cls.camera, wall, floor) for rotation, centre in cls.poses]
        cls.unit = float(np.linalg.norm(cls.poses[1][1] - cls.poses[0][1]))  # the odometry's first step is its unit

    @classmethod
    def tearDownClass(cls):
        del cls.vo
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_python_layout_matches_c(self):
        self.assertEqual(self.vo.vm_map_size(), ctypes.sizeof(voxel_map.VoxelMapC))

    def test_files_the_wall_and_the_floor(self):
        result = voxel_map.run_mapping(self.vo, self.frames, [None] * len(self.frames), self.camera)
        self.assertTrue(all(step.kind == 2 for step in result["steps"]))
        obstacles = voxel_map.centres_of(result["final"]) * self.unit
        self.assertGreater(len(obstacles), 100)
        side = result["final"]["side"] * self.unit
        on_surface = (np.abs(obstacles[:, 2] - WALL_Z) < 2 * side + 0.1) | (np.abs(obstacles[:, 1] - FLOOR_Y) < 2 * side + 0.1)
        self.assertGreater(on_surface.mean(), 0.9)
        self.assertEqual(result["stats"]["lost"], 0)
        self.assertEqual(result["stats"]["count"], len(result["final"]["keys"]))

    def test_goes_on_through_a_tracking_loss(self):
        frames = list(self.frames)
        frames[3] = np.full((mvo.HEIGHT, mvo.WIDTH), 128, dtype=np.uint8)
        result = voxel_map.run_mapping(self.vo, frames, [None] * len(frames), self.camera)
        kinds = [step.kind for step in result["steps"]]
        self.assertEqual(kinds[2:4], [0, 0])
        self.assertEqual(result["stats"]["lost"], 2)
        filed_after = [item["filed"] for item in result["progress"][4:]]
        self.assertGreater(sum(filed_after), 0)
        counts = [item["count"] for item in result["progress"]]
        self.assertGreater(counts[-1], counts[3])

    def test_water_everywhere_files_nothing(self):
        water = [np.zeros((60, 80), dtype=np.uint8)] * len(self.frames)
        result = voxel_map.run_mapping(self.vo, self.frames, water, self.camera)
        self.assertEqual(result["stats"]["observations"], 0)
        self.assertEqual(result["final"]["side"], 0.0)

    def test_same_run_same_hash(self):
        first = voxel_map.run_mapping(self.vo, self.frames, [None] * len(self.frames), self.camera)
        second = voxel_map.run_mapping(self.vo, self.frames, [None] * len(self.frames), self.camera)
        self.assertEqual(first["hash"], second["hash"])
        self.assertEqual(first["progress"], second["progress"])

    def test_alignment_refuses_a_straight_path(self):
        line = np.stack([np.arange(5.0), np.zeros(5), np.zeros(5)], axis=1)
        self.assertIsNone(voxel_map.alignment(line, line * 2.0))
        turned = np.array([[0.0, 0, 0], [1, 0, 0], [1, 0, 1], [2, 0, 1]])
        scale, rotation, offset = voxel_map.alignment(turned, 2.0 * turned + 1.0)
        self.assertAlmostEqual(scale, 2.0, places=6)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
