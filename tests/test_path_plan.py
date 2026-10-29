"""The planner (core/path_plan.c): its C contract, and on rendered views of a wall and a floor that its ranges in steps
follow the wall and that it steers by the mask; the reference corridors and the scoring."""

import ctypes
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from uwslam.mvo import config_defines

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    from tests.test_mvo import WALL_Z, render, texture
    from uwslam import mvo, obstacle_dir, path_plan
    from uwslam.experiments.mask_motion import Camera, euler_rotation

ROOT = Path(__file__).resolve().parents[1]
STATIC = ["-static"] if sys.platform == "win32" else []


@unittest.skipUnless(shutil.which("gcc") is not None, "Needs gcc")
class PathPlanContractTests(unittest.TestCase):
    def test_c_contract(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            program = Path(folder) / "path_plan_check.exe"
            units = (
                "mono_vo.c",
                "mono_vo_numerics.c",
                "mono_vo_tracking.c",
                "mono_vo_homography.c",
                "mono_vo_essential.c",
            )
            sources = [
                ROOT / "tests/c/path_plan_check.c",
                ROOT / "core/path_plan.c",
                ROOT / "core/obstacle_dir.c",
                ROOT / "core/voxel_map.c",
                *(ROOT / "core" / name for name in units),
            ]
            subprocess.run(
                [
                    "gcc",
                    "-std=c99",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-ffp-contract=off",
                    *config_defines(),
                    *STATIC,
                    f"-I{ROOT / 'core'}",
                    *map(str, sources),
                    "-o",
                    str(program),
                ],
                check=True,
            )
            output = subprocess.run([str(program)], check=True, capture_output=True, text=True).stdout
        self.assertIn("path_plan_check ok", output)


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs gcc and OpenCV")
class PathPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.lib = mvo.build(cls.folder)
        cls.camera = mvo.camera_for(110.0, 544 / 960)
        wall, floor = texture(1), texture(2)
        cls.poses = [(euler_rotation(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.2 * k])) for k in range(12)]
        cls.frames = [render(rotation, centre, cls.camera, wall, floor) for rotation, centre in cls.poses]

    @classmethod
    def tearDownClass(cls):
        del cls.lib
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_python_layout_matches_c(self):
        self.assertEqual(self.lib.pp_plan_size(), ctypes.sizeof(path_plan.PlanC))

    def test_ranges_in_steps_follow_the_wall(self):
        result = path_plan.run_planning(self.lib, self.frames, [None] * len(self.frames), self.camera)
        self.assertTrue(all(step.kind == 2 for step in result["steps"]))
        checked = 0
        for k, plan in enumerate(result["plans"]):
            if not (plan["known"] >> path_plan.CENTRE) & 1:
                continue
            steps = plan["range"][path_plan.CENTRE] / plan["speed"]
            wall = (WALL_Z - self.poses[k + 1][1][2]) / 0.2  # steps of 0.2 to the wall straight ahead
            self.assertLess(abs(np.log2(steps / wall)), 1.0, f"step {k + 1}: {steps:.1f} steps against {wall:.1f}")
            checked += 1
        self.assertGreater(checked, 3)

    def test_the_mask_steers_and_the_map_slows(self):
        mask = np.ones((60, 80), dtype=np.uint8)
        mask[20:40, :32] = 0  # open water to the left at eye level
        result = path_plan.run_planning(self.lib, self.frames, [mask] * len(self.frames), self.camera)
        directions = [p["direction"] for p in result["plans"]]
        actions = [p["action"] for p in result["plans"]]
        self.assertTrue(all(a in (path_plan.GO, path_plan.CREEP) for a in actions))
        self.assertIn(path_plan.CENTRE - 1, directions)  # left at eye level, the nearest free direction
        self.assertTrue(all((p["passable"] >> path_plan.CENTRE) & 1 == 0 for p in result["plans"]))
        self.assertEqual(result["plans"][-1]["water"], [path_plan.CENTRE - 2, path_plan.CENTRE - 1])
        bounded = [p for p in result["plans"] if p["clear"] >= 0]
        self.assertGreater(len(bounded), len(result["plans"]) // 2)  # the wall and floor in view bound the water
        self.assertTrue(all(0 < p["clear"] / p["speed"] <= path_plan.SAFE_STEPS * 5 for p in bounded))
        again = path_plan.run_planning(self.lib, self.frames, [mask] * len(self.frames), self.camera)
        self.assertEqual(result["plan_hashes"], again["plan_hashes"])

    def test_corridor_takes_the_central_cells(self):
        layout = obstacle_dir.layout_for(self.lib, self.camera)
        corridors = path_plan.corridor_cells(layout)
        self.assertEqual(corridors.shape, (15, 60, 80))
        self.assertEqual(int(corridors[path_plan.CENTRE].sum()), 8 * 10)  # the middle half of 16 columns by 20 rows
        grid = Camera.from_fov(110.0, 544 / 960)
        depth = np.full((60, 80), 1.5, dtype=np.float32)
        depth[:, :40] = np.inf  # open water on the left half
        ranges = path_plan.corridor_ranges(depth, corridors, grid)
        self.assertTrue(np.isinf(ranges[path_plan.CENTRE - 1]))
        self.assertGreaterEqual(ranges[path_plan.CENTRE + 1], 1.5)
        self.assertLess(ranges[path_plan.CENTRE + 1], 1.7)

    def test_scores_compare_like_steps(self):
        mask = np.ones((60, 80), dtype=np.uint8)
        mask[20:40, :32] = 0
        result = path_plan.run_planning(self.lib, self.frames, [mask] * len(self.frames), self.camera)
        truth = np.full((len(result["plans"]), 15), 2.0)
        truth[:, path_plan.CENTRE] = 0.2  # straight ahead blocked near
        scores = path_plan.score(result, truth, np.full(len(result["plans"]), 0.2))
        self.assertEqual(scores["straight_ahead"]["within_0.25_m_free"], 0.0)
        self.assertEqual(scores["plan"]["within_1_m_free"], 1.0)
        self.assertEqual(scores["plan"]["danger_go_below_0.25_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
