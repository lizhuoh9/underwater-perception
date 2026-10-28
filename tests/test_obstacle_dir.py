"""The obstacle directions (core/obstacle_dir.c): the C contract, the Python binding against a direct computation, the
reference depth and the scores."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HAS_GCC = shutil.which("gcc") is not None
if HAS_GCC:
    from uwslam import mvo, obstacle_dir
    from uwslam.experiments.mask_motion import Camera

ROOT = Path(__file__).resolve().parents[1]
STATIC = ["-static"] if sys.platform == "win32" else []


@unittest.skipUnless(HAS_GCC, "Needs gcc")
class ObstacleDirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.lib = mvo.build(cls.folder)
        cls.camera = mvo.camera_for(110.0, 544 / 960)

    @classmethod
    def tearDownClass(cls):
        del cls.lib
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_c_contract(self):
        program = self.folder / "obstacle_dir_check.exe"
        subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-ffp-contract=off",
                        *mvo.config_defines(), *STATIC, f"-I{ROOT / 'core'}", str(ROOT / "tests/c/obstacle_dir_check.c"),
                        str(ROOT / "core/obstacle_dir.c"), str(ROOT / "core/mono_vo_numerics.c"), "-o", str(program)],
                       check=True)
        output = subprocess.run([str(program)], check=True, capture_output=True, text=True).stdout
        self.assertIn("obstacle_dir_check ok", output)

    def test_layout_follows_the_ray_angles(self):
        for hfov in (110.0, 70.0):
            camera = mvo.camera_for(hfov, 544 / 960)
            layout = obstacle_dir.layout_for(self.lib, camera)
            x = ((np.arange(80) + 0.5) * 4 - camera.cx) / camera.fx
            y = ((np.arange(60) + 0.5) * 4 - camera.cy) / camera.fy
            inner, outer = np.tan(np.radians(15)), np.tan(np.radians(40))
            sector = np.digitize(x, [-outer, -inner, inner + 1e-7, outer + 1e-7])
            band = np.digitize(y, [-inner, inner + 1e-7])
            np.testing.assert_array_equal(np.array(layout.sector[:80]), sector)
            np.testing.assert_array_equal(np.array(layout.band[:60]), band)
            cells = np.bincount((band[:, None] * 5 + sector[None, :]).ravel(), minlength=15)
            np.testing.assert_array_equal(np.array(layout.cells[:]), cells)

    def test_shares_and_choice_follow_the_mask(self):
        mask = np.ones((60, 80), dtype=np.uint8)
        mask[20:40, :32] = 0  # water on the left at eye level
        result = obstacle_dir.run_directions(self.lib, [mask, None, mask], self.camera)
        layout = result["layout"]
        directions = obstacle_dir.cell_directions(layout)
        share = np.bincount(directions.ravel(), weights=mask.ravel(), minlength=15) / np.array(layout.cells[:])
        np.testing.assert_array_equal(result["shares"][0], np.floor(share * 255 + 0.5).astype(int))
        self.assertEqual(result["choices"].tolist(), [6, 6, 6])  # the left at eye level: one step, then more open
        self.assertEqual(result["blocked_bits"][0], (1 << 15) - 1 - (1 << 5) - (1 << 6))
        self.assertEqual(result["frames"], 2)  # the frame without a mask leaves the state as it was

    def test_reference_depth_takes_the_near_surface(self):
        camera = Camera.from_fov(110.0, 544 / 960)
        rng = np.random.default_rng(0)
        plane = np.stack([rng.uniform(-3, 3, 400000), rng.uniform(-2, 2, 400000), np.full(400000, 1.5)], axis=1)
        behind = plane * np.array([1.0, 1.0, -1.0])  # behind the camera: ignored
        depth = obstacle_dir.reference_depth(np.vstack([plane, behind]).astype(np.float32), np.eye(3), np.zeros(3),
                                             camera)
        seen = np.isfinite(depth)
        self.assertGreater(seen.mean(), 0.9)
        np.testing.assert_allclose(depth[seen], 1.5, atol=1e-5)
        lone = obstacle_dir.reference_depth(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), np.eye(3), np.zeros(3),
                                            camera)
        self.assertTrue(np.isinf(lone).all())  # one point is too few to show a surface

    def test_scores_count_agreement(self):
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[40:, :] = 1  # the floor below
        result = obstacle_dir.run_directions(self.lib, [mask] * 4, self.camera)
        depth = np.full((4, 60, 80), np.inf, dtype=np.float32)
        depth[:, 40:, :] = 0.4  # the reference: the same floor, near
        scores = obstacle_dir.score(result, depth)
        near = scores["within_0.5_m"]
        self.assertEqual(near["bands"]["down"]["recall"], 1.0)
        self.assertEqual(near["bands"]["down"]["precision"], 1.0)
        self.assertIsNone(near["bands"]["up"]["recall"])  # nothing near up there
        self.assertEqual(near["choice_free"], 1.0)
        self.assertEqual(scores["no_free_direction"], 0.0)


if __name__ == "__main__":
    unittest.main()
