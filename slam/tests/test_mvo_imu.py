"""The odometry's optional accelerometer (core/mono_vo.c through uwslam.mvo, mvo_imu_reading) on synthetic frames:
steps whose forces are missing, zero, not finite or beyond 5 g stay the gyro's alone, bit for bit; the world's up
settles on the first agreeing steps' measured up, not on thrust at the start; a tilt the camera did not see is
levelled towards the forces, about a level axis, at most MVO_IMU_LEVEL_MAX a step and never from a force against the
world's up; a fast turn with consistent readings stays level; the calibrated bias and the mounting are taken off the
forces; readings before the first frame belong to no step; two cores give the same steps.

Needs gcc and OpenCV; skipped without either.
"""

import ctypes
import re
import shutil
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    from tests.test_mvo import render, texture
    from tests.test_mvo_gyro import READINGS, STEP_S, readings
    from uwslam import mvo
    from uwslam.experiments.mask_motion import angle_deg, euler_rotation

G = 9.81
UP = np.array([0.0, -1.0, 0.0])  # the world's up in the first camera's axes (y down)


def header_value(name: str) -> float:
    text = (mvo.CORE_DIR / "mono_vo.h").read_text(encoding="utf-8")
    return float(re.search(rf"#define {name} ([0-9.]+)f?", text).group(1))


def still(force_camera: np.ndarray, mount: np.ndarray | None = None, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A still step's IMU readings: no turn, the force (camera axes) as an IMU mounted with `mount` reads it, plus a
    constant offset in the IMU's axes."""
    force = force_camera if mount is None else mount.T @ force_camera
    row = [STEP_S / READINGS, 0.0, 0.0, 0.0, *(force + np.asarray(offset))]
    return np.array([row] * READINGS, dtype=np.float32)


def tilted(degrees: float) -> np.ndarray:
    """The force a camera tilted by `degrees` about its x axis (from the first camera) measures."""
    return euler_rotation(0.0, 0.0, np.radians(degrees)).T @ (G * UP)


def aqualoc_exp(v: np.ndarray) -> np.ndarray:
    """The rotation of the rotation vector v."""
    from uwslam.aqualoc import exp_so3

    return exp_so3(v)


def unchecked(steps: list[np.ndarray]) -> "mvo.GyroReadings":
    """IMU readings past GyroReadings' checks, which refuse forces that are not finite: the core drops them itself."""
    readings_ = object.__new__(mvo.GyroReadings)
    for name, value in (("mount", np.eye(3)), ("steps", steps), ("bias", None), ("accel_bias", None)):
        object.__setattr__(readings_, name, value)
    return readings_


def world_rotations(run) -> list[np.ndarray]:
    return [np.array(s.world_rotation, dtype=np.float64).reshape(3, 3) for s in run.steps]


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs gcc and OpenCV")
class ImuTests(unittest.TestCase):
    vo = None
    folder = None

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.vo = mvo.build(cls.folder)
        cls.camera = mvo.camera_for(110.0, 544 / 960)
        cls.wall, cls.floor = texture(1), texture(2)
        cls.view = render(np.eye(3), np.zeros(3), cls.camera, cls.wall, cls.floor)

    @classmethod
    def tearDownClass(cls):
        del cls.vo
        shutil.rmtree(cls.folder, ignore_errors=True)

    def frames(self, poses):
        return [render(r, c, self.camera, self.wall, self.floor) for r, c in poses]

    def still_run(self, forces, mount=None, offset=(0.0, 0.0, 0.0), accel_bias=None):
        """A still camera (the same frame throughout) with one step per force (camera axes)."""
        steps = [still(f, mount, offset) for f in forces]
        imu = mvo.GyroReadings(np.eye(3) if mount is None else mount, steps, None, accel_bias)
        return mvo.run_odometry(self.vo, [self.view] * (len(forces) + 1), self.camera, gyro=imu)

    def test_forces_missing_zero_or_not_finite_leave_the_gyros_steps(self):
        poses = [(euler_rotation(np.radians(1.5 * k), 0.0, 0.0), np.array([0.3 * k, 0.0, 0.15 * k])) for k in range(6)]
        frames = self.frames(poses)
        gyro = [readings(poses[k][0].T @ poses[k + 1][0]) for k in range(5)]
        alone = mvo.run_odometry(self.vo, frames, self.camera, gyro=mvo.GyroReadings(np.eye(3), gyro))
        for force in (0.0, np.nan, np.inf, 1e30, 30.0):  # 30 on each axis: 52 m/s^2, beyond MVO_IMU_FORCE_MAX
            steps = [np.column_stack([g, np.full((len(g), 3), force, dtype=np.float32)]) for g in gyro]
            run = mvo.run_odometry(self.vo, frames, self.camera, gyro=unchecked(steps))
            self.assertEqual(run.hashes, alone.hashes, force)
            self.assertEqual(run.gyro_bias, alone.gyro_bias, force)
            self.assertEqual(run.imu_up, (0.0, 0.0, 0.0), force)

    def test_the_worlds_up_is_the_first_steps_measured_up(self):
        steps = int(header_value("MVO_IMU_UP_STEPS"))
        rng = np.random.default_rng(3)
        forces = [G * UP + rng.normal(0.0, 0.1, 3) for _ in range(steps + 4)]  # about 0.6 degrees apart: they agree
        run = self.still_run(forces[: steps - 1])
        self.assertEqual(run.imu_up, (0.0, 0.0, 0.0))  # not settled yet
        self.assertEqual(world_rotations(run)[-1].tolist(), np.eye(3).tolist())  # and nothing levelled
        run = self.still_run(forces)
        units = [f / np.linalg.norm(f) for f in forces[:steps]]  # the world rotation stays the first camera's till then
        expected = np.sum(units, axis=0)
        np.testing.assert_allclose(run.imu_up, expected / np.linalg.norm(expected), atol=1e-5)
        for rotation in world_rotations(run):
            self.assertLess(angle_deg(rotation), 1.0)

    def test_thrust_at_the_start_does_not_set_the_up(self):
        # Three steps of 2 m/s^2 of thrust read as an 11-degree tilt; the up settles on the still steps after them.
        steps = int(header_value("MVO_IMU_UP_STEPS"))
        thrust = G * UP + np.array([2.0, 0.0, 0.0])
        run = self.still_run([thrust] * 3 + [G * UP] * (steps + 3))
        np.testing.assert_allclose(run.imu_up, UP, atol=1e-6)
        for rotation in world_rotations(run):
            self.assertLess(angle_deg(rotation), 1e-3)

    def test_a_tilt_the_camera_did_not_see_is_levelled(self):
        # The frames stay still, so the camera sees no turn, while the forces say it tilted by 5 degrees about its x
        # axis after the world's up was set: the world rotation follows, about a level axis, a fifth of the way a step.
        steps = int(header_value("MVO_IMU_UP_STEPS"))
        run = self.still_run([G * UP] * steps + [tilted(5.0)] * 30)
        rotations = world_rotations(run)
        self.assertLess(angle_deg(rotations[steps - 1]), 1e-3)
        first = angle_deg(rotations[steps])
        self.assertAlmostEqual(first, 0.2 * 5.0, delta=0.02)  # gain 0.2 for 0.2 s of readings, sin 5 deg ~ 5 deg
        final = rotations[-1]
        self.assertAlmostEqual(angle_deg(final), 5.0, delta=0.05)
        axis = np.array([final[2, 1] - final[1, 2], final[0, 2] - final[2, 0], final[1, 0] - final[0, 1]])
        self.assertLess(abs(axis @ UP) / np.linalg.norm(axis), 1e-3)  # a tilt: the heading stays
        measured = tilted(5.0) / G
        self.assertLess(np.degrees(np.arccos(np.clip(final.T @ UP @ measured, -1, 1))), 0.05)

    def test_one_steps_levelling_is_capped(self):
        steps = int(header_value("MVO_IMU_UP_STEPS"))
        cap = np.degrees(header_value("MVO_IMU_LEVEL_MAX"))
        run = self.still_run([G * UP] * steps + [tilted(30.0)])
        self.assertAlmostEqual(angle_deg(world_rotations(run)[-1]), cap, delta=0.01)

    def test_a_force_against_the_worlds_up_turns_nothing(self):
        steps = int(header_value("MVO_IMU_UP_STEPS"))
        run = self.still_run([G * UP] * steps + [-G * UP, tilted(100.0)])
        for rotation in world_rotations(run)[steps:]:
            self.assertEqual(rotation.tolist(), np.eye(3).tolist())

    def test_the_accelerometers_bias_and_mounting_are_taken_off(self):
        steps = int(header_value("MVO_IMU_UP_STEPS"))
        forces = [G * UP] * steps + [tilted(5.0)] * 8
        plain = world_rotations(self.still_run(forces))
        offset = np.array([0.4, -0.3, 0.7])
        biased = world_rotations(self.still_run(forces, offset=offset, accel_bias=offset))
        mount = euler_rotation(0.4, -0.9, 1.2)  # camera axes from the IMU's
        mounted = world_rotations(self.still_run(forces, mount=mount))
        for a, b, c in zip(plain, biased, mounted):
            np.testing.assert_allclose(b, a, atol=1e-5)
            np.testing.assert_allclose(c, a, atol=1e-5)
        uncalibrated = world_rotations(self.still_run(forces, offset=offset))  # the offset tilts the levelling
        self.assertGreater(max(angle_deg(a.T @ b) for a, b in zip(plain, uncalibrated)), 0.1)

    def test_the_core_says_which_forces_it_took(self):
        steps = int(header_value("MVO_IMU_UP_STEPS"))
        rows = [still(G * UP) for _ in range(steps + 1)]
        rows[2][5, 4:] = 60.0  # a knock beyond 5 g
        rows[4][7, 4:] = np.nan
        run = mvo.run_odometry(self.vo, [self.view] * (steps + 2), self.camera, gyro=unchecked(rows))
        self.assertEqual(run.forces_dropped, 2)
        np.testing.assert_allclose(run.imu_up, UP, atol=1e-6)

    def test_readings_before_the_first_frame_belong_to_no_step(self):
        vo = self.vo
        state = ctypes.create_string_buffer(vo.mvo_state_size())
        pyramids = [ctypes.create_string_buffer(vo.mvo_pyramid_bytes()) for _ in range(2)]
        vo.mvo_init(state, ctypes.byref(self.camera), mvo.SEED, pyramids[0], pyramids[1])
        rate = np.zeros(3, dtype=np.float32)
        force = (G * UP).astype(np.float32)
        ctypes.memmove(vo.mvo_frame(state), np.ascontiguousarray(self.view).ctypes.data, mvo.WIDTH * mvo.HEIGHT)
        self.assertEqual(vo.mvo_imu_reading(state, rate.ctypes.data, force.ctypes.data, 0.05), 1)
        self.assertEqual(vo.mvo_imu_reading(state, rate.ctypes.data, force.ctypes.data, 0.0), 0)  # no duration
        vo.mvo_start(state)
        ctypes.memmove(vo.mvo_frame(state), np.ascontiguousarray(self.view).ctypes.data, mvo.WIDTH * mvo.HEIGHT)
        step = mvo.StepC()
        vo.mvo_step(state, ctypes.byref(step))
        up = np.ones(3, dtype=np.float32)
        vo.mvo_imu_up(state, up.ctypes.data)
        self.assertEqual(up.tolist(), [0.0, 0.0, 0.0])
        bias = np.array([0.0, np.nan, 0.0], dtype=np.float32)
        self.assertEqual(vo.mvo_imu_set_accel_bias(state, bias.ctypes.data), 0)

    def test_a_fast_turn_with_consistent_readings_stays_level(self):
        # Still until the up settles, then a roll of 40 degrees a second: each reading's force is gravity in the camera
        # at that moment. Turned into the step's start with the wrong hand, a step's forces would miss its up by about
        # the step's 8 degrees and the levelling would tilt the world.
        steps, rate = int(header_value("MVO_IMU_UP_STEPS")), np.radians(40.0)
        times = [0.0] * (steps + 1) + [STEP_S * k for k in range(1, 9)]
        truth = [euler_rotation(0.0, 0.0, 0.0) @ aqualoc_exp(np.array([0.0, 0.0, rate * t])) for t in times]
        frames = [render(r, np.zeros(3), self.camera, self.wall, self.floor) for r in truth]
        readings_ = []
        for k in range(len(truth) - 1):
            moving = times[k + 1] > times[k]
            rows = []
            for j in range(READINGS):
                middle = times[k] + (j + 0.5) * STEP_S / READINGS if moving else times[k]
                force = aqualoc_exp(np.array([0.0, 0.0, rate * middle])).T @ (G * UP)
                rows.append([STEP_S / READINGS, 0.0, 0.0, rate if moving else 0.0, *force])
            readings_.append(np.array(rows, dtype=np.float32))
        run = mvo.run_odometry(self.vo, frames, self.camera, gyro=mvo.GyroReadings(np.eye(3), readings_))
        np.testing.assert_allclose(run.imu_up, UP, atol=1e-6)
        for estimate, true in zip(world_rotations(run), truth[1:]):
            tilt = np.degrees(np.arccos(np.clip((estimate.T @ UP) @ (true.T @ UP), -1.0, 1.0)))
            self.assertLess(tilt, 0.1)

    def test_the_imu_steps_are_the_same_on_two_cores(self):
        down = euler_rotation(0.0, np.radians(-90.0), 0.0)
        poses = [(down, np.array([0.02 * k, 0.0, 0.0])) for k in range(14)]
        frames = self.frames(poses)
        steps = []
        for k in range(13):
            gyro = readings(poses[k][0].T @ poses[k + 1][0])
            force = poses[k + 1][0].T @ (G * UP) + np.array([0.0, 0.0, 0.4 * max(0, k - 9)])  # a tilt to level
            steps.append(np.column_stack([gyro, np.tile(force, (len(gyro), 1))]).astype(np.float32))
        work_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)
        calls = []

        @ctypes.CFUNCTYPE(None, work_type, ctypes.c_void_p)
        def backwards(work, context):
            calls.append(1)
            work(context, 1)
            work(context, 0)

        def run(parallel):
            vo = self.vo
            state = ctypes.create_string_buffer(vo.mvo_state_size())
            pyramids = [ctypes.create_string_buffer(vo.mvo_pyramid_bytes()) for _ in range(2)]
            vo.mvo_init(state, ctypes.byref(self.camera), mvo.SEED, pyramids[0], pyramids[1])
            if parallel is not None:  # mvo_t ends with its two optional pointers: parallel, then kernels
                pointer = ctypes.sizeof(ctypes.c_void_p)
                ctypes.c_void_p.from_buffer(state, len(state) - 2 * pointer).value = ctypes.cast(
                    parallel, ctypes.c_void_p
                ).value
            hashes = []
            for k, frame in enumerate(frames):
                ctypes.memmove(vo.mvo_frame(state), np.ascontiguousarray(frame).ctypes.data, mvo.WIDTH * mvo.HEIGHT)
                if k == 0:
                    vo.mvo_start(state)
                    continue
                for row in steps[k - 1]:
                    rate, force = np.ascontiguousarray(row[1:4]), np.ascontiguousarray(row[4:])
                    vo.mvo_imu_reading(state, rate.ctypes.data, force.ctypes.data, float(row[0]))
                step = mvo.StepC()
                vo.mvo_step(state, ctypes.byref(step))
                hashes.append(mvo.fnv(mvo.FNV_OFFSET, bytes(step)))
            return hashes

        self.assertEqual(run(None), run(backwards))
        self.assertGreater(len(calls), 4)

    def test_imu_readings_are_checked(self):
        gyro = np.zeros((2, 4), dtype=np.float32)
        gyro[:, 0] = 0.01
        imu = np.zeros((2, 7), dtype=np.float32)
        imu[:, 0] = 0.01
        imu[:, 5] = -G
        mvo.GyroReadings(np.eye(3), [imu, imu], None, np.zeros(3))
        in_g = imu.copy()
        in_g[:, 5] = -1.0
        with self.assertRaisesRegex(ValueError, "m/s"):
            mvo.GyroReadings(np.eye(3), [in_g])
        with self.assertRaisesRegex(ValueError, "every step alike"):
            mvo.GyroReadings(np.eye(3), [gyro, imu])
        with self.assertRaisesRegex(ValueError, "needs readings with forces"):
            mvo.GyroReadings(np.eye(3), [gyro], None, np.zeros(3))
        with self.assertRaisesRegex(ValueError, "three finite"):
            mvo.GyroReadings(np.eye(3), [imu], None, np.array([0.0, np.inf, 0.0]))
        bad = imu.copy()
        bad[0, 5] = np.nan
        with self.assertRaisesRegex(ValueError, "finite values"):
            mvo.GyroReadings(np.eye(3), [bad])


if __name__ == "__main__":
    unittest.main()
