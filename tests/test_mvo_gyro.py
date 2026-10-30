"""The odometry's optional gyro (core/mono_vo.c through uwslam.mvo) on synthetic frames: steps without readings stay
the camera's own, a pure turn takes the gyro's rotation, travel over a floor that the camera alone reads as a turn gets
its translation, and the bias is learnt from calm ESSENTIAL steps only.

Needs gcc and OpenCV; skipped without either.
"""

import ctypes
import shutil
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    from tests.test_mvo import render, texture
    from uwslam import mvo
    from uwslam.experiments.mask_motion import angle_deg, euler_rotation

STEP_S, READINGS = 0.2, 20  # a 5 Hz step, read at 100 Hz
NONE, TRANSLATION, ROTATION, LOST = 0, 1, 2, 3  # MVO_GYRO_*


def mvo_gyro_bias_limit() -> float:
    """MVO_GYRO_BIAS_LIMIT, read from the header."""
    import re

    text = (mvo.CORE_DIR / "mono_vo.h").read_text(encoding="utf-8")
    return float(re.search(r"#define MVO_GYRO_BIAS_LIMIT ([0-9.]+)f", text).group(1))


def axis_angle(r: np.ndarray) -> np.ndarray:
    angle = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return 0.5 * axis if angle < 1e-9 else angle / (2.0 * np.sin(angle)) * axis


def readings(turn: np.ndarray, bias=(0.0, 0.0, 0.0), mount: np.ndarray | None = None) -> np.ndarray:
    """A step's readings for the camera turn R_AB at a constant rate, as a gyro mounted with `mount` (camera axes from
    the gyro's) with this bias would give them."""
    rate = axis_angle(turn) / STEP_S
    if mount is not None:
        rate = mount.T @ rate
    row = [STEP_S / READINGS, *(rate + np.asarray(bias))]
    return np.array([row] * READINGS, dtype=np.float32)


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs gcc and OpenCV")
class GyroTests(unittest.TestCase):
    vo = None
    folder = None

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.vo = mvo.build(cls.folder)
        cls.camera = mvo.camera_for(110.0, 544 / 960)
        cls.wall, cls.floor = texture(1), texture(2)

    @classmethod
    def tearDownClass(cls):
        del cls.vo
        shutil.rmtree(cls.folder, ignore_errors=True)

    def frames(self, poses):
        return [render(r, c, self.camera, self.wall, self.floor) for r, c in poses]

    def moving(self, count: int, turn_deg: float = 1.5):
        return [
            (euler_rotation(np.radians(turn_deg * k), 0.0, 0.0), np.array([0.30 * k, 0.0, 0.15 * k]))
            for k in range(count)
        ]

    def gyro_for(self, poses, bias=(0.0, 0.0, 0.0), start_bias=None) -> "mvo.GyroReadings":
        steps = [readings(poses[k][0].T @ poses[k + 1][0], bias) for k in range(len(poses) - 1)]
        return mvo.GyroReadings(np.eye(3), steps, None if start_bias is None else np.array(start_bias))

    def test_without_readings_a_step_is_the_cameras(self):
        # A mounted gyro that gives no readings changes nothing, bit for bit: the lake clip stays as it was.
        poses = self.moving(5)
        frames = self.frames(poses)
        alone = mvo.run_odometry(self.vo, frames, self.camera)
        empty = [np.zeros((0, 4), dtype=np.float32)] * (len(frames) - 1)
        mounted = mvo.run_odometry(self.vo, frames, self.camera, gyro=mvo.GyroReadings(np.eye(3), empty))
        self.assertEqual(alone.hashes, mounted.hashes)
        self.assertEqual(mounted.gyro_used, [NONE] * len(alone.steps))

    def test_a_pure_turn_takes_the_gyros_rotation(self):
        turn = euler_rotation(np.radians(5.0), np.radians(-1.5), np.radians(0.5))
        frames = self.frames([(np.eye(3), np.zeros(3)), (turn, np.zeros(3))])
        for mount in (np.eye(3), euler_rotation(0.3, -1.1, 0.7)):  # the gyro's axes need not be the camera's
            run = mvo.run_odometry(
                self.vo, frames, self.camera, gyro=mvo.GyroReadings(mount, [readings(turn, mount=mount)])
            )
            self.assertEqual(run.gyro_used, [ROTATION])
            self.assertEqual(mvo.KINDS[run.steps[0].kind], "rotation")
            found = np.array(run.steps[0].rotation, dtype=np.float64).reshape(3, 3)
            self.assertLess(angle_deg(found.T @ turn), 0.05)

    def test_travel_over_a_floor_that_the_camera_reads_as_a_turn(self):
        # Looking down at a floor 1 m away, 2 cm of sideways travel a step: a homography explains the plane's flow as a
        # turn, and the camera alone files every step as ROTATION (AQUALOC's harbour floor). With the gyro's rotation
        # the translation alone is solved for and points the right way.
        down = euler_rotation(0.0, np.radians(-90.0), 0.0)
        self.assertGreater(down[1, 2], 0.99)  # the camera's forward axis points down the floor's y
        poses = [(down, np.array([0.02 * k, 0.0, 0.0])) for k in range(6)]
        frames = self.frames(poses)
        alone = mvo.run_odometry(self.vo, frames, self.camera)
        self.assertEqual([mvo.KINDS[s.kind] for s in alone.steps], ["rotation"] * 5)
        run = mvo.run_odometry(self.vo, frames, self.camera, gyro=self.gyro_for(poses))
        self.assertEqual(run.gyro_used, [TRANSLATION] * 5)
        for k, step in enumerate(run.steps):
            (r_a, c_a), (_, c_b) = poses[k], poses[k + 1]
            truth, estimate = r_a.T @ (c_b - c_a), np.array(step.centre, dtype=np.float64)
            cosine = truth @ estimate / (np.linalg.norm(truth) * np.linalg.norm(estimate))
            self.assertGreater(cosine, np.cos(np.radians(2.0)), (k, truth, estimate))

    def test_the_bias_is_learnt_from_calm_essential_steps(self):
        bias = np.array([0.02, -0.01, 0.015])  # rad/s, about 1 degree per second
        poses = self.moving(12)
        run = mvo.run_odometry(self.vo, self.frames(poses), self.camera, gyro=self.gyro_for(poses, bias))
        self.assertEqual(run.gyro_used, [NONE] * 11)  # every step is the camera's own, and teaches the bias
        np.testing.assert_allclose(run.gyro_bias[-1], bias, atol=np.radians(0.15))
        first_error = np.abs(np.array(run.gyro_bias[0]) - bias).max()
        self.assertLess(np.abs(np.array(run.gyro_bias[-1]) - bias).max(), first_error)

    def test_fast_turns_teach_no_bias(self):
        # 4 degrees a step is beyond MVO_GYRO_CALM (3): fast turns carry the camera's larger rotation errors.
        poses = self.moving(6, turn_deg=4.0)
        run = mvo.run_odometry(self.vo, self.frames(poses), self.camera, gyro=self.gyro_for(poses, (0.02, 0.0, 0.0)))
        self.assertEqual(run.gyro_used, [NONE] * 5)  # the camera's own steps, which would teach a calm one
        self.assertEqual(run.gyro_bias[-1], (0.0, 0.0, 0.0))

    def test_a_set_bias_is_taken_off_the_readings(self):
        turn = euler_rotation(np.radians(5.0), np.radians(-1.5), np.radians(0.5))
        frames = self.frames([(np.eye(3), np.zeros(3)), (turn, np.zeros(3))])
        bias = (0.05, -0.03, 0.04)  # 0.6-1 degree over the step if it stayed in
        gyro = mvo.GyroReadings(np.eye(3), [readings(turn, bias)], np.array(bias))
        run = mvo.run_odometry(self.vo, frames, self.camera, gyro=gyro)
        found = np.array(run.steps[0].rotation, dtype=np.float64).reshape(3, 3)
        self.assertLess(angle_deg(found.T @ turn), 0.05)
        np.testing.assert_allclose(run.gyro_bias[-1], bias, rtol=1e-6)  # a rotation step teaches nothing

    def test_readings_before_the_first_frame_or_without_duration_are_dropped(self):
        poses = self.moving(3)
        frames = self.frames(poses)
        alone = mvo.run_odometry(self.vo, frames, self.camera)
        vo = self.vo
        state = ctypes.create_string_buffer(vo.mvo_state_size())
        pyramids = [ctypes.create_string_buffer(vo.mvo_pyramid_bytes()) for _ in range(2)]
        vo.mvo_init(state, ctypes.byref(self.camera), mvo.SEED, pyramids[0], pyramids[1])
        rate = np.array([0.5, -0.2, 0.1], dtype=np.float32)
        hashes = []
        for k, frame in enumerate(frames):
            ctypes.memmove(vo.mvo_frame(state), np.ascontiguousarray(frame).ctypes.data, mvo.WIDTH * mvo.HEIGHT)
            if k == 0:
                vo.mvo_gyro_rate(state, rate.ctypes.data, 0.05)  # before the first frame: belongs to no step
                vo.mvo_start(state)
                continue
            vo.mvo_gyro_rate(state, rate.ctypes.data, 0.0)  # no duration
            vo.mvo_gyro_rate(state, rate.ctypes.data, float("nan"))
            vo.mvo_gyro_rate(state, rate.ctypes.data, -0.01)
            step = mvo.StepC()
            vo.mvo_step(state, ctypes.byref(step))
            hashes.append(mvo.fnv(mvo.FNV_OFFSET, bytes(step)))
            self.assertEqual(vo.mvo_gyro_used(state), NONE)
        self.assertEqual(hashes, alone.hashes)

    def run_raw(self, frames, steps, mount=None, parallel=None):
        """The core driven directly, readings as given (no checks on this side): hashes, rotations, final bias."""
        vo = self.vo
        state = ctypes.create_string_buffer(vo.mvo_state_size())
        pyramids = [ctypes.create_string_buffer(vo.mvo_pyramid_bytes()) for _ in range(2)]
        vo.mvo_init(state, ctypes.byref(self.camera), mvo.SEED, pyramids[0], pyramids[1])
        if parallel is not None:  # mvo_t ends with its two optional pointers: parallel, then kernels
            pointer = ctypes.sizeof(ctypes.c_void_p)
            ctypes.c_void_p.from_buffer(state, len(state) - 2 * pointer).value = ctypes.cast(
                parallel, ctypes.c_void_p
            ).value
        if mount is not None:
            matrix = np.ascontiguousarray(mount, dtype=np.float32).reshape(9)
            self.assertEqual(vo.mvo_gyro_mount(state, matrix.ctypes.data), 1)
        hashes, rotations = [], []
        for k, frame in enumerate(frames):
            ctypes.memmove(vo.mvo_frame(state), np.ascontiguousarray(frame).ctypes.data, mvo.WIDTH * mvo.HEIGHT)
            if k == 0:
                vo.mvo_start(state)
                continue
            for dt, *rate in steps[k - 1]:
                values = np.array(rate, dtype=np.float32)
                vo.mvo_gyro_rate(state, values.ctypes.data, float(dt))
            step = mvo.StepC()
            vo.mvo_step(state, ctypes.byref(step))
            hashes.append(mvo.fnv(mvo.FNV_OFFSET, bytes(step)))
            rotations.append(np.array(step.rotation, dtype=np.float64).reshape(3, 3))
        bias = np.zeros(3, dtype=np.float32)
        vo.mvo_gyro_bias(state, bias.ctypes.data)
        return hashes, rotations, bias

    def test_readings_that_are_not_finite_or_absurd_are_dropped(self):
        turn = euler_rotation(np.radians(5.0), np.radians(-1.5), np.radians(0.5))
        frames = self.frames([(np.eye(3), np.zeros(3)), (turn, np.zeros(3))])
        good = [tuple(row) for row in readings(turn)]
        bad = [(0.001, np.nan, 0.0, 0.0), (0.001, np.inf, 0.0, 0.0), (0.001, 1e30, 0.0, 0.0), (np.inf, 0.0, 0.0, 0.0)]
        _, rotations, bias = self.run_raw(frames, [good[:7] + bad + good[7:]])
        self.assertLess(angle_deg(rotations[0].T @ turn), 0.05)  # as if the bad readings never came
        self.assertTrue(np.all(np.isfinite(bias)))

    def test_a_gap_in_the_readings_teaches_no_bias(self):
        # One step's readings cover 5 ms of its 200: the gyro misses most of that step's turn, which over 5 ms would
        # read as a bias of hundreds of degrees per second. The bias is zero and must stay near it.
        poses = self.moving(12)
        steps = [readings(poses[k][0].T @ poses[k + 1][0]) for k in range(len(poses) - 1)]
        steps[4] = steps[4][:1] * np.array([[0.5, 1.0, 1.0, 1.0]], dtype=np.float32)  # one reading of 5 ms
        run = mvo.run_odometry(self.vo, self.frames(poses), self.camera, gyro=mvo.GyroReadings(np.eye(3), steps))
        self.assertEqual(run.gyro_used, [NONE] * 11)
        self.assertLess(np.degrees(np.abs(run.gyro_bias[-1])).max(), 0.2)

    def test_the_bias_estimate_has_a_limit(self):
        frames = self.frames(self.moving(2))
        too_much = np.full(3, mvo_gyro_bias_limit() * 1.01)
        with self.assertRaisesRegex(ValueError, "BIAS_LIMIT"):
            mvo.run_odometry(
                self.vo, frames, self.camera, gyro=mvo.GyroReadings(np.eye(3), [readings(np.eye(3))], too_much)
            )
        with self.assertRaisesRegex(ValueError, "three finite"):
            mvo.GyroReadings(np.eye(3), [readings(np.eye(3))], np.array([0.0, np.nan, 0.0]))
        bad_mount = np.eye(3)
        bad_mount[1, 1] = np.inf
        with self.assertRaisesRegex(ValueError, "finite 3x3"):
            mvo.GyroReadings(bad_mount, [readings(np.eye(3))])

    def test_the_gyro_steps_are_the_same_on_two_cores(self):
        # The translation solve for the gyro's rotation splits its samples and directions over two cores: running the
        # second half first must change nothing, as for the camera's own parts.
        down = euler_rotation(0.0, np.radians(-90.0), 0.0)
        poses = [(down, np.array([0.02 * k, 0.0, 0.0])) for k in range(5)]
        frames = self.frames(poses)
        steps = [[tuple(row) for row in readings(poses[k][0].T @ poses[k + 1][0])] for k in range(4)]
        work_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)
        calls = []

        @ctypes.CFUNCTYPE(None, work_type, ctypes.c_void_p)
        def backwards(work, context):
            calls.append(1)
            work(context, 1)
            work(context, 0)

        alone, _, bias_alone = self.run_raw(frames, steps)
        split, _, bias_split = self.run_raw(frames, steps, parallel=backwards)
        self.assertEqual(alone, split)
        self.assertEqual(bias_alone.tobytes(), bias_split.tobytes())
        self.assertGreater(len(calls), 4)

    def test_the_bias_is_learnt_in_the_gyros_own_axes(self):
        mount = euler_rotation(0.4, -0.9, 1.2)  # camera axes from the gyro's
        bias = np.array([0.02, -0.01, 0.015])
        poses = self.moving(12)
        steps = [readings(poses[k][0].T @ poses[k + 1][0], bias, mount) for k in range(len(poses) - 1)]
        run = mvo.run_odometry(self.vo, self.frames(poses), self.camera, gyro=mvo.GyroReadings(mount, steps))
        np.testing.assert_allclose(run.gyro_bias[-1], bias, atol=np.radians(0.15))

    def test_readings_are_checked(self):
        good = np.zeros((2, 4), dtype=np.float32)
        good[:, 0] = 0.01
        with self.assertRaisesRegex(ValueError, "3x3"):
            mvo.GyroReadings(np.eye(2), [good])
        with self.assertRaisesRegex(ValueError, "float32 rows"):
            mvo.GyroReadings(np.eye(3), [good.astype(np.float64)])
        bad = good.copy()
        bad[1, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "positive durations"):
            mvo.GyroReadings(np.eye(3), [bad])
        frames = self.frames(self.moving(3))
        with self.assertRaisesRegex(ValueError, "for 2 steps"):
            mvo.run_odometry(self.vo, frames, self.camera, gyro=mvo.GyroReadings(np.eye(3), [good]))


if __name__ == "__main__":
    unittest.main()
