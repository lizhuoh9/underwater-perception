"""Known-geometry checks of the rotation a lost step takes (tests/c/mvo_lost_turn_check.c); not an image-tracking
test."""

import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from uwslam.core import CFLAGS
from uwslam.paths import CORE_DIR

SOURCE = Path(__file__).resolve().parent / "c" / "mvo_lost_turn_check.c"
UNITS = [CORE_DIR / name for name in ("mono_vo.c", "mono_vo_numerics.c", "mono_vo_homography.c", "mono_vo_essential.c")]


def build(folder: Path) -> Path:
    executable = folder / "mvo_lost_turn_check.exe"
    subprocess.run(
        [
            "gcc",
            *CFLAGS,
            *(["-static"] if sys.platform == "win32" else []),
            f"-I{CORE_DIR}",
            str(SOURCE),
            *map(str, UNITS),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def trials(
    executable: Path,
    pairs: int,
    sigma: float,
    outliers: float,
    centre: tuple,
    yaw: float,
    pitch: float,
    first_seed: int,
    count: int,
) -> list[dict]:
    command = [str(executable), *map(str, (pairs, sigma, outliers, *centre, yaw, pitch, first_seed, count))]
    text = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    return list(csv.DictReader(text.splitlines()))


@unittest.skipUnless(shutil.which("gcc"), "gcc is needed")
class LostTurnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.executable = build(Path(cls.folder.name))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_a_lost_step_that_moved_keeps_the_essential_rotation(self):
        # Few pairs moving forward and down: the translation lacks evidence and the rotation fit lacks support, but the
        # homography has it, having read the parallax as a turn. Before the essential pose's rotation replaced it there,
        # these steps took the homography's and were as far off as it.
        for pairs, sigma, outliers, centre, yaw, pitch in (
            (20, 0.5, 0.2, (0, 0.1, 0.2), 10, 0),
            (20, 0.5, 0.1, (0, 0.1, 0.2), 0, 5),
            (30, 1.0, 0.0, (0, 0.1, 0.2), 0, 0),
        ):
            with self.subTest(pairs=pairs, sigma=sigma, outliers=outliers, yaw=yaw, pitch=pitch):
                rows = trials(self.executable, pairs, sigma, outliers, centre, yaw, pitch, 70000, 200)
                lost = [row for row in rows if int(row["kind"]) == 0 and int(row["fallback"])]
                self.assertGreaterEqual(len(lost), 30)
                step = sum(float(row["step_error"]) for row in lost) / len(lost)
                homography = sum(float(row["homography_error"]) for row in lost) / len(lost)
                self.assertLess(step, 0.7 * homography, (step, homography))

    def test_without_an_essential_pose_the_homography_stands(self):
        # 12 pairs: no essential pose can reach MVO_MIN_INLIERS (15), so a step that falls back on the homography takes
        # exactly its rotation.
        rows = trials(self.executable, 12, 0.5, 0.0, (0, 0.1, 0.2), 0, 5, 70000, 200)
        lost = [row for row in rows if int(row["kind"]) == 0 and int(row["fallback"])]
        self.assertGreaterEqual(len(lost), 30)
        for row in lost:
            self.assertAlmostEqual(float(row["step_error"]), float(row["homography_error"]), places=3)

    def test_a_still_camera_with_noisy_tracks_gains_no_turn(self):
        # Most of these steps are lost. A pose fitted to noise alone turns at random (taking the essential rotation on
        # every lost step gave a mean of 1.5 degrees here), so where no rotation has the support they keep none and the
        # camera stays put. The mean decides: 0.17 degrees at most over 204 seed blocks, against 1.2 or more for that
        # change. The worst single step is no test, since some blocks reach 18 degrees as it is.
        errors = []
        for pairs in (25, 49, 80):
            for sigma in (1.0, 2.0, 3.0):
                for outliers in (0.0, 0.2, 0.4):
                    rows = trials(self.executable, pairs, sigma, outliers, (0, 0, 0), 0, 0, 90000, 30)
                    errors += [float(row["step_error"]) for row in rows]
        self.assertLess(sum(errors) / len(errors), 0.2)


if __name__ == "__main__":
    unittest.main()
