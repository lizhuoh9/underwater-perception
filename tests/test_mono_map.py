"""The obstacle map (core/mono_map.c through uwslam.mono_map) on the synthetic frames of tests/test_mvo.py: a camera
moving past a textured wall and floor, with a mask that marks only the wall as not water.

Needs gcc and OpenCV; skipped without either.
"""

import shutil
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    from tests.test_mvo import WALL_Z, render, texture
    from uwslam import mono_map, mvo
    from uwslam.experiments.mask_motion import euler_rotation

MASK_ROWS, MASK_COLUMNS, WALL_ROWS = 60, 80, 34  # the wall fills the image down to row ~39 of 60 on these frames


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs gcc and OpenCV")
class MonoMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.vo = mvo.build(cls.folder)
        cls.camera = mvo.camera_for(110.0, 544 / 960)
        wall, floor = texture(1), texture(2)
        cls.poses = [
            (euler_rotation(np.radians(1.5 * k), 0.0, 0.0), np.array([0.30 * k, 0.0, 0.15 * k])) for k in range(8)
        ]
        frames = [render(r, c, cls.camera, wall, floor) for r, c in cls.poses]
        mask = np.zeros((MASK_ROWS, MASK_COLUMNS), dtype=np.uint8)
        mask[:WALL_ROWS] = 1
        cls.result = mono_map.run_mapping(cls.vo, frames, [mask] * len(frames), cls.camera)
        cls.unit = float(np.linalg.norm(cls.poses[1][1] - cls.poses[0][1]))  # the odometry's unit: the first step

    @classmethod
    def tearDownClass(cls):
        del cls.vo
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_observations_lie_on_the_wall(self):
        points = np.vstack([p for p in self.result["observations"] if len(p)])
        self.assertGreater(len(points), 200)
        depth = points[:, 2] * self.unit  # world z of the first camera, in the scene's units
        self.assertLess(abs(np.median(depth) - WALL_Z), 0.1 * WALL_Z)
        self.assertGreater(np.mean(np.abs(depth - WALL_Z) < 0.2 * WALL_Z), 0.9)

    def test_the_grid_marks_the_wall_and_leaves_other_cells_unknown(self):
        odds, cell = self.result["odds"], self.result["cell"]
        origin_x, origin_z = self.result["origin"]
        self.assertGreater(cell, 0.0)
        rows = (np.arange(odds.shape[0]) + origin_z + 0.5) * cell * self.unit  # scene z of each grid row
        wall_rows = np.abs(rows - WALL_Z) < 0.15 * WALL_Z
        before = (rows > 1.5) & (rows < 0.7 * WALL_Z)  # past the camera's path, short of the wall
        self.assertGreater((odds[wall_rows] > 0).sum(), 10)
        self.assertEqual((odds[before] > 0).sum(), 0)
        self.assertEqual((odds < 0).sum(), 0)

    def test_the_map_is_deterministic(self):
        again = mono_map.run_mapping(self.vo, *self.inputs(), self.camera)
        self.assertEqual(again["map_hash"], self.result["map_hash"])

    def test_tracking_loss_freezes_the_old_map_after_motion_resumes(self):
        frames, masks = self.inputs()
        before = mono_map.run_mapping(self.vo, frames[:2], masks[:2], self.camera)
        frames[2] = np.full((mvo.HEIGHT, mvo.WIDTH), 128, dtype=np.uint8)
        masks[2] = None
        result = mono_map.run_mapping(self.vo, frames, masks, self.camera)
        self.assertEqual([mvo.KINDS[step.kind] for step in result["steps"]][1:4], ["lost", "lost", "essential"])
        self.assertGreater(before["hits"], 0)
        self.assertEqual(result["first_lost_frame"], 2)
        self.assertFalse(result["map_continuous"])
        self.assertEqual(result["map_continuity"], [1, 0, 0, 0, 0, 0, 0])
        self.assertTrue(all(len(points) == 0 for points in result["observations"][1:]))
        np.testing.assert_array_equal(result["odds"], before["odds"])
        for key in ("cell", "origin", "hits", "updates", "slice_y", "height_resets", "map_hash"):
            self.assertEqual(result[key], before[key], key)

    def inputs(self):
        wall, floor = texture(1), texture(2)
        frames = [render(r, c, self.camera, wall, floor) for r, c in self.poses]
        mask = np.zeros((MASK_ROWS, MASK_COLUMNS), dtype=np.uint8)
        mask[:WALL_ROWS] = 1
        return frames, [mask] * len(frames)


if __name__ == "__main__":
    unittest.main()
