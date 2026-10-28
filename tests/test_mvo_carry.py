"""Known-geometry checks of the depth chain a LOST step carries on (tests/c/mvo_carry_check.c); not an image-tracking
test."""

import csv
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest

from uwslam.core import CFLAGS
from uwslam.mvo import config_defines
from uwslam.paths import CORE_DIR


SOURCE = Path(__file__).resolve().parent / "c" / "mvo_carry_check.c"
UNITS = [CORE_DIR / name for name in ("mono_vo.c", "mono_vo_numerics.c", "mono_vo_homography.c",
                                      "mono_vo_essential.c")]


def build(folder: Path) -> Path:
    executable = folder / "mvo_carry_check.exe"
    subprocess.run(["gcc", *CFLAGS, *config_defines(), *(["-static"] if sys.platform == "win32" else []), f"-I{CORE_DIR}",
                    str(SOURCE), *map(str, UNITS), "-lm", "-o", str(executable)], check=True)
    return executable


def trials(executable: Path, mode: str, sigma: float, outliers: float, yaw: float, forward: float, speed: float,
           first_seed: int, count: int) -> list[dict]:
    command = [str(executable), mode, *map(str, (sigma, outliers, yaw, forward, speed, first_seed, count))]
    text = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    return list(csv.DictReader(text.splitlines()))


@unittest.skipUnless(shutil.which("gcc"), "gcc is needed")
class CarryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.executable = build(Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_a_moving_pair_carries_the_true_scale(self):
        # 0.1 units of travel past points 1 to 6 units away, sub-pixel noise: the shared depth ratios agree, and the
        # carried points come out at their true depths, the scale the old points had.
        for sigma, yaw in ((0.3, 5.0), (0.5, 12.0)):
            with self.subTest(sigma=sigma, yaw=yaw):
                rows = [row for row in trials(self.executable, "direct", sigma, 0.0, yaw, 0.1, 0.1, 70000, 60)
                        if int(row["inliers"]) >= 15]
                carried = [float(row["depth_error"]) for row in rows if int(row["carried"])]
                self.assertGreaterEqual(len(carried), 0.8 * len(rows), (len(carried), len(rows)))
                self.assertLess(abs(statistics.median(carried)), 0.05)
                self.assertLess(max(abs(error) for error in carried), 0.1)

    def test_a_still_camera_carries_almost_nothing(self):
        # A pose fitted to a still camera's noise: its depth ratios scatter, so its points are not carried and the next
        # step that moves takes the recent speed (carried here in 19 of 992 poses).
        poses = carried = 0
        for sigma in (1.0, 2.0, 3.0):
            for outliers in (0.0, 0.2, 0.4):
                for yaw in (0.0, 5.0):
                    rows = [row for row in trials(self.executable, "direct", sigma, outliers, yaw, 0.0, 0.1, 80000, 60)
                            if int(row["inliers"]) >= 15]
                    poses += len(rows)
                    carried += sum(int(row["carried"]) for row in rows)
        self.assertGreater(poses, 800)
        self.assertLessEqual(carried, 0.05 * poses, (carried, poses))

    def test_hovering_does_not_upset_the_scale_of_the_next_move(self):
        # One to three still steps with noisy tracks (LOST), then a move of 0.1 with clean ones. Without a carry the
        # move's length is the recent speed, 0.1, almost exactly; a noise pose that slips through must stay rare.
        # Taking every lost step's pose gave a median error of 0.15-0.25 in log2, up to 1.2; carrying points without
        # shared ones, on their parallax alone, set the move off by more than 0.3 in 24-43% of trials after two hovers.
        for count in (1, 2, 3):
            with self.subTest(hovers=count):
                errors = []
                for sigma in (2.0, 3.0):
                    for outliers in (0.0, 0.2, 0.4):
                        for yaw in (0.0, 5.0):
                            rows = trials(self.executable, "hovers", sigma, outliers, yaw, count, 0.1, 50000, 40)
                            errors += [abs(float(row["length_error"])) for row in rows
                                       if int(row["kind_last"]) == 0 and int(row["kind2"]) == 2]
                self.assertGreater(len(errors), 400)
                self.assertLess(statistics.median(errors), 0.05)
                self.assertLessEqual(sum(error > 0.3 for error in errors), 0.05 * len(errors))

    def test_a_lost_step_that_moved_carries_its_scale(self):
        # The first step moves 0.2 while turning 10 degrees, with 1 px noise: in a sixth of the trials its translation
        # lacks evidence (LOST), but the depths of its pose agree with the old points. The recent speed is wrong (0.3,
        # three times the next move), so only the carried chain gives the next move its length (90-98% of those trials
        # over five seed blocks; none without the carry).
        rows = trials(self.executable, "steps", 1.0, 0.0, 10.0, 0.2, 0.3, 61000, 300)
        moves = [abs(float(row["length_error"])) for row in rows if int(row["kind1"]) == 0 and int(row["kind2"]) == 2]
        self.assertGreaterEqual(len(moves), 30)
        self.assertGreaterEqual(sum(error < 0.2 for error in moves), 0.8 * len(moves))


if __name__ == "__main__":
    unittest.main()
