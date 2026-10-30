"""The odometry's optional metric scale (core/mono_vo_scale.c through uwslam.mvo: mvo_depth_reading, mvo_range_setup,
mvo_range_reading, mvo_metric) on synthetic frames of a scene whose unit is METRES metres: depths and ranges leave the
odometry's steps as they were, bit for bit; a forward ranger's readings, and a depth sensor's changes once the IMU's
up has settled, put the path in metres; the first scale takes the path so far; the depth since the first frame is the
metric vertical; a beam without points, missing readings and readings the core refuses give no scale. Then the
unit alone on a hand-built state (tests/c/mvo_scale_check.c): the ring of depths wraps, a later observation is weighed
in by its variance, a weak first one is left out, and without the first frame's depth the vertical stays the
odometry's.

Needs gcc and OpenCV (the C check gcc alone); skipped without them.
"""

import ctypes
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    from tests.test_mvo import WALL_Z, render, texture
    from tests.test_mvo_gyro import readings
    from uwslam import mvo
    from uwslam.experiments.mask_motion import euler_rotation
    from uwslam.ranger import ate

METRES = 1.5  # metres per unit of the rendered scene
G_UP = np.array([0.0, -9.81, 0.0])  # the specific force at rest in the first camera's axes (y down)
SURFACE_DEPTH = 3.0  # the first frame's depth, m


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs gcc and OpenCV")
class ScaleTests(unittest.TestCase):
    vo = None
    folder = None

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.vo = mvo.build(cls.folder)
        cls.camera = mvo.camera_for(110.0, 544 / 960)
        cls.wall, cls.floor = texture(1), texture(2)
        # Turning slowly while moving right, forward and up (towards the surface): 18 frames.
        cls.poses = [
            (euler_rotation(np.radians(1.5 * k), 0.0, 0.0), np.array([0.30 * k, -0.12 * k, 0.15 * k]))
            for k in range(18)
        ]
        cls.frames = [render(r, c, cls.camera, cls.wall, cls.floor) for r, c in cls.poses]
        cls.truth = METRES * np.array([c for _, c in cls.poses])
        cls.depths = SURFACE_DEPTH + cls.truth[:, 1]  # y points down
        cls.ranges = np.array([METRES * (WALL_Z - c[2]) / r[2, 2] for r, c in cls.poses])  # along the optical axis
        steps = []
        for k in range(len(cls.poses) - 1):
            gyro = readings(cls.poses[k][0].T @ cls.poses[k + 1][0])
            force = cls.poses[k + 1][0].T @ G_UP
            steps.append(np.column_stack([gyro, np.tile(force, (len(gyro), 1))]).astype(np.float32))
        cls.imu = mvo.GyroReadings(np.eye(3), steps)
        cls.beam = mvo.Beam((0.0, 0.0, 1.0), 20.0)  # wide enough to hold five tracked points from the first step

    @classmethod
    def tearDownClass(cls):
        del cls.vo
        shutil.rmtree(cls.folder, ignore_errors=True)

    def run_with(self, imu=None, depths=None, ranges=None, beam=None):
        return mvo.run_odometry(self.vo, self.frames, self.camera, mvo.SEED, imu, depths, ranges, beam)

    def assert_in_metres(self, run, share: float):
        """The metric path after a rotation and a shift (no scale) lies within `share` of the path's length."""
        path = mvo.metric_path(run)
        self.assertIsNotNone(path)
        length = np.sum(np.linalg.norm(np.diff(self.truth, axis=0), axis=1))
        self.assertLess(ate(path, self.truth, with_scale=False), share * length)

    def test_readings_leave_the_odometrys_steps(self):
        camera_alone, imu_alone = self.run_with(), self.run_with(self.imu)
        ranged = self.run_with(ranges=self.ranges, beam=self.beam)
        self.assertEqual(ranged.hashes, camera_alone.hashes)
        both = self.run_with(self.imu, self.depths, self.ranges, self.beam)
        self.assertEqual(both.hashes, imu_alone.hashes)
        self.assertEqual(both.gyro_bias, imu_alone.gyro_bias)
        self.assertEqual(both.imu_up, imu_alone.imu_up)
        self.assertIsNone(camera_alone.metric)

    def test_a_forward_range_puts_the_path_in_metres(self):
        run = self.run_with(ranges=self.ranges, beam=self.beam)
        self.assertTrue(all(m is not None for m in run.metric))  # the first step's points already lie in the beam
        self.assert_in_metres(run, 0.05)

    def test_the_first_scale_takes_the_path_so_far(self):
        ranges = self.ranges.copy()
        ranges[:4] = np.nan  # the ranger is quiet for the first three steps
        run = self.run_with(ranges=ranges, beam=self.beam)
        first = [k for k, m in enumerate(run.metric) if m is not None][0]
        self.assertGreaterEqual(first, 3)  # the step of frame 4, or a later one with five points in the beam
        self.assertTrue(all(m is None for m in run.metric[:first]))
        centre = np.array(run.steps[first].world_centre, dtype=np.float32)
        self.assertEqual(run.metric[first], tuple(float(v) for v in np.float32(run.scales[first]) * centre))
        self.assert_in_metres(run, 0.05)

    def test_a_depth_change_puts_the_path_in_metres_once_the_up_settles(self):
        run = self.run_with(self.imu, self.depths)
        known = [k for k, m in enumerate(run.metric) if m is not None]
        self.assertGreaterEqual(known[0], 10)  # MVO_IMU_UP_AGREE steps settle the up first
        self.assertTrue(all(m is not None for m in run.metric[known[0] :]))
        self.assert_in_metres(run, 0.05)

    def test_the_depth_since_the_first_frame_is_the_metric_vertical(self):
        run = self.run_with(self.imu, self.depths, self.ranges, self.beam)
        down = -np.array(run.imu_up)
        self.assertAlmostEqual(float(np.linalg.norm(down)), 1.0, places=5)
        for k, metric in enumerate(run.metric):
            if k >= 11:  # once the up has settled
                self.assertAlmostEqual(float(np.dot(metric, down)), self.depths[k + 1] - self.depths[0], places=5)
        # Without the depth, the same ranges give another vertical.
        ranged = self.run_with(self.imu, None, self.ranges, self.beam)
        self.assertNotEqual(ranged.metric[-1], run.metric[-1])

    def test_no_scale_without_points_or_readings(self):
        behind = self.run_with(ranges=self.ranges, beam=mvo.Beam((0.0, 0.0, -1.0), 7.5))
        self.assertTrue(all(m is None for m in behind.metric))
        self.assertEqual(set(behind.scales), {0.0})
        quiet = self.run_with(ranges=np.full(len(self.frames), np.nan), beam=self.beam)
        self.assertTrue(all(m is None for m in quiet.metric))
        level = self.run_with(None, self.depths)  # no IMU: no down to compare the depth with
        self.assertTrue(all(m is None for m in level.metric))
        self.assertIsNone(mvo.metric_path(level))

    def test_the_core_refuses_readings_it_cannot_use(self):
        vo = self.vo
        state = ctypes.create_string_buffer(vo.mvo_state_size())
        pyramids = [ctypes.create_string_buffer(vo.mvo_pyramid_bytes()) for _ in range(2)]
        vo.mvo_init(state, ctypes.byref(self.camera), mvo.SEED, pyramids[0], pyramids[1])
        for depth in (np.nan, np.inf, -np.inf):
            self.assertEqual(vo.mvo_depth_reading(state, depth), 0)
        self.assertEqual(vo.mvo_depth_reading(state, -0.5), 1)  # above the surface's zero: a number like any other
        self.assertEqual(vo.mvo_range_reading(state, 2.0), 0)  # no beam yet

        def setup(axis, cosine):
            values = np.ascontiguousarray(axis, dtype=np.float32)
            return vo.mvo_range_setup(state, values.ctypes.data, cosine)

        for axis, cosine in (
            ((0, 0, 0), 0.9),
            ((np.nan, 0, 1), 0.9),
            ((0, 0, 1), 1.0),
            ((0, 0, 1), 0.0),
            ((0, 0, 1), -0.5),
        ):
            self.assertEqual(setup(axis, cosine), 0, (axis, cosine))
        self.assertEqual(setup((0, 0, 2), 0.99), 1)
        for value in (0.0, -1.0, np.nan, np.inf):
            self.assertEqual(vo.mvo_range_reading(state, value), 0, value)
        self.assertEqual(vo.mvo_range_reading(state, 2.0), 1)
        centre, scale, variance = np.ones(3, dtype=np.float32), ctypes.c_float(7.0), ctypes.c_float(7.0)
        self.assertEqual(vo.mvo_metric(state, centre.ctypes.data, ctypes.byref(scale), ctypes.byref(variance)), 0)
        self.assertEqual((centre.tolist(), scale.value, variance.value), ([0.0, 0.0, 0.0], 0.0, 0.0))
        self.assertEqual(vo.mvo_metric(state, centre.ctypes.data, None, None), 0)  # either may be NULL

    def test_run_odometry_checks_its_readings(self):
        with self.assertRaisesRegex(ValueError, "need a beam"):
            self.run_with(ranges=self.ranges)
        with self.assertRaisesRegex(ValueError, "need a beam"):
            self.run_with(beam=self.beam)
        with self.assertRaisesRegex(ValueError, "Expected depths"):
            self.run_with(self.imu, self.depths[:-1])
        with self.assertRaisesRegex(ValueError, "refused the ranger's beam"):
            self.run_with(ranges=self.ranges, beam=mvo.Beam((0.0, 0.0, 1.0), 0.0))
        with self.assertRaisesRegex(ValueError, "for 18 frames"):
            mvo.sense_bytes(self.depths[:-1], None, len(self.frames))


@unittest.skipUnless(shutil.which("gcc") is not None, "Needs gcc")
class ScaleContractTests(unittest.TestCase):
    def test_c_contract(self):
        from uwslam import mvo as binding
        from uwslam.core import CFLAGS

        source = Path(__file__).resolve().parent / "c" / "mvo_scale_check.c"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            program = Path(folder) / "mvo_scale_check.exe"
            subprocess.run(
                [
                    "gcc",
                    *CFLAGS,
                    *(["-static"] if sys.platform == "win32" else []),
                    f"-I{binding.CORE_DIR}",
                    str(source),
                    *map(str, binding.SOURCES),
                    "-lm",
                    "-o",
                    str(program),
                ],
                check=True,
            )
            result = subprocess.run([str(program)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("ok "), 11, result.stdout)


if __name__ == "__main__":
    unittest.main()
