"""uwslam.aqualoc and uwslam.aqualoc_score on a small synthetic sequence in AQUALOC's harbour layout: Kalibr's camera
and camera-IMU files, image, IMU and depth csv files, a Colmap trajectory, all from one known motion and gyro bias.

Needs gcc and OpenCV; skipped without either.
"""

import json
import shutil
import struct
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    import cv2

    from uwslam import aqualoc, aqualoc_score, mvo, mvo_stream

T0_NS = 1_550_000_000_000_000_000
IMAGES, FPS, IMU_HZ = 41, 20.0, 200.0  # 2 s of images: 11 frames at 5 Hz
RATE_IMU = np.array([0.05, -0.12, 0.08])  # rad/s, the IMU's axes
BIAS_IMU = np.array([0.004, 0.02, -0.006])  # rad/s
SHIFT = -0.02  # t_imu = t_cam + shift
MOUNT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # the IMU's axes into the camera's
CAMERA_YAML = """cam0:
  camera_model: pinhole
  distortion_coeffs: [-0.0612556, -0.0037967, 0.0273266, -0.0302964]
  distortion_model: equidistant
  intrinsics: [413.32595, 413.70199, 305.95075, 259.44399]
  resolution: [640, 512]
"""


def imu_yaml() -> str:
    rows = [[*MOUNT[k], 0.01 * k] for k in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
    lines = ["cam0:", "  T_cam_imu:"] + [f"  - [{', '.join(repr(float(v)) for v in row)}]" for row in rows]
    return "\n".join(lines + [f"  timeshift_cam_imu: {SHIFT}", ""])


def camera_to_world(t: float) -> np.ndarray:
    """The camera's orientation at camera time t: turning at the IMU's true rate, from the identity."""
    return aqualoc.exp_so3(MOUNT @ RATE_IMU * t)


def matrix_quaternion(r: np.ndarray) -> np.ndarray:
    """(qx, qy, qz, qw) of a rotation matrix (w well away from 0 here)."""
    w = np.sqrt(1.0 + np.trace(r)) / 2
    return np.array([(r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w])


def write_sequence(folder: Path) -> dict:
    raw = folder / "raw_data"
    images = raw / "harbor_images_sequence_01"
    images.mkdir(parents=True)
    stamps = T0_NS + (np.arange(IMAGES) * 1e9 / FPS).astype(np.int64)
    rng = np.random.default_rng(3)
    with (raw / "harbor_img_sequence_01.csv").open("w", encoding="utf-8") as out:
        out.write("#timestamp [ns],filename\n")
        for k, stamp in enumerate(stamps):
            name = f"frame{k:04d}.png"
            texture = cv2.resize(
                rng.integers(0, 256, (32, 40), dtype=np.uint8), (640, 512), interpolation=cv2.INTER_CUBIC
            )
            cv2.imwrite(str(images / name), texture)
            out.write(f"{stamp},{name}\n")
    imu_times = np.arange(-0.5, IMAGES / FPS + 0.5, 1.0 / IMU_HZ)
    with (raw / "harbor_imu_sequence_01.csv").open("w", encoding="utf-8") as out:
        out.write(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],"
            "a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        )
        for t in imu_times:
            rate = RATE_IMU + BIAS_IMU
            out.write(f"{T0_NS + int(round(t * 1e9))},{rate[0]},{rate[1]},{rate[2]},0.0,0.0,9.81\n")
    centre = lambda t: np.array([0.3 * np.sin(t), 0.1 * t * t, 3.0 + 0.05 * t])  # noqa: E731  z is the depth
    with (raw / "harbor_depth_sequence_01.csv").open("w", encoding="utf-8") as out:
        out.write("#timestamp [ns],depth [m]\n")
        for t in np.arange(0.0, IMAGES / FPS, 0.1):
            out.write(f"{T0_NS + int(round(t * 1e9))},{centre(t)[2]}\n")
    trajectory = folder / "trajectory.txt"
    with trajectory.open("w", encoding="utf-8") as out:
        out.write("# img_number tx ty tz qx qy qz qw\n")
        for k in range(0, IMAGES, 5):
            t = k / FPS
            out.write(" ".join(str(v) for v in [k, *centre(t), *matrix_quaternion(camera_to_world(t))]) + "\n")
    (folder / "camera.yaml").write_text(CAMERA_YAML, encoding="utf-8")
    (folder / "imu.yaml").write_text(imu_yaml(), encoding="utf-8")
    return {"raw": raw, "trajectory": trajectory, "camera": folder / "camera.yaml", "imu": folder / "imu.yaml"}


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs gcc and OpenCV")
class AqualocTests(unittest.TestCase):
    folder = None

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.files = write_sequence(cls.folder / "source")
        cls.prepared = cls.folder / "seq01"
        with patch("builtins.print"):
            cls.checks = aqualoc.prepare(
                cls.files["raw"], cls.files["trajectory"], cls.files["camera"], cls.files["imu"], cls.prepared
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_frames_are_every_fourth_image_inside_the_reference(self):
        frames = np.load(self.prepared / "frames_5hz.npy")
        data = np.load(self.prepared / "sequence.npz")
        self.assertEqual(frames.shape, (11, mvo.HEIGHT, mvo.WIDTH))
        np.testing.assert_array_equal(data["picks"], np.arange(0, IMAGES, 4))
        self.assertAlmostEqual(self.checks["step_ms"]["min"], 200.0, places=3)

    def test_readings_add_up_to_the_steps_and_the_reference_turns(self):
        sequence = aqualoc_score.load_prepared(self.prepared)
        for k, readings in enumerate(sequence.readings):
            self.assertAlmostEqual(float(readings[:, 0].astype(np.float64).sum()), 0.2, places=6)
            np.testing.assert_allclose(readings[:, 1:], np.tile(RATE_IMU + BIAS_IMU, (len(readings), 1)), atol=1e-6)
            turn = sequence.reference_rotations[k].T @ sequence.reference_rotations[k + 1]
            gyro = aqualoc.integrate(readings.astype(np.float64), sequence.mount, BIAS_IMU)
            self.assertLess(aqualoc.angle_deg(gyro.T @ turn), 1e-3)
        np.testing.assert_allclose(self.checks["fitted_bias_dps"], np.degrees(BIAS_IMU), atol=1e-3)
        self.assertGreater(self.checks["gyro_error_deg"]["median"], 0.2)  # the bias left in: 0.24 degrees a step
        self.assertLess(self.checks["gyro_error_without_fitted_bias_deg"]["median"], 1e-3)

    def test_the_core_models_the_resampled_camera_exactly(self):
        camera = self.checks["camera"]
        self.assertLess(camera["core_ray_error_deg"]["max"], 1e-3)
        self.assertEqual(camera["core_undistort_failures"], 0)
        self.assertEqual(camera["pixels_inside_source"], 1.0)
        self.assertGreater(camera["hfov_deg"], 80.0)

    def test_the_vertical_comes_from_the_depth(self):
        np.testing.assert_allclose(self.checks["vertical_world"], [0.0, 0.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(self.checks["depth_slope_norm"], 1.0, places=6)

    def test_step_readings_cut_the_samples_at_the_stamps(self):
        times = np.array([0.0, 0.01, 0.02, 0.03])
        rates = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
        readings = aqualoc.step_readings(times, rates, 0.005, 0.025)
        np.testing.assert_allclose(readings[:, 0], [0.005, 0.01, 0.005])
        np.testing.assert_allclose(readings[:, 1], [0.75, 1.5, 2.25])  # midpoints of the linear rate
        with self.assertRaisesRegex(ValueError, "outside the IMU"):
            aqualoc.step_readings(times, rates, -0.01, 0.02)

    def test_board_files_hold_the_readings_and_both_expectations(self):
        output = self.folder / "board"
        aqualoc.board_files(self.prepared, output)
        expected = json.loads((output / "board_expected_gyro.json").read_text(encoding="utf-8"))
        camera_only = json.loads((output / "board_expected_camera.json").read_text(encoding="utf-8"))
        self.assertNotIn("gyro", camera_only)
        payloads = mvo_stream.read_gyro_file(output / "gyro.bin", expected["frames"] - 1)
        sequence = aqualoc_score.load_prepared(self.prepared)
        for payload, readings in zip(payloads, sequence.readings):
            self.assertEqual(payload, readings.astype("<f4").tobytes())
        mount = [struct.unpack("<f", struct.pack("<I", bits))[0] for bits in expected["gyro"]["mount_bits"]]
        np.testing.assert_allclose(np.reshape(mount, (3, 3)), MOUNT)
        self.assertEqual(len(expected["gyro"]["used"]), expected["steps"])
        with self.assertRaises(FileExistsError):
            aqualoc.board_files(self.prepared, output)

    def test_scoring_pairs_every_gyro_run_with_the_camera_alone(self):
        sequence = aqualoc_score.load_prepared(self.prepared)
        with (
            patch.object(aqualoc_score, "WINDOW_FRAMES", 6),
            patch.object(aqualoc_score, "WINDOW_EVERY", 5),
            patch.object(mvo, "SEEDS", (mvo.SEED, 2)),
            patch("builtins.print"),
        ):
            import tempfile as scratch_files

            with scratch_files.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
                scores = aqualoc_score.score_sequence(mvo.build(Path(scratch)), sequence)
            tables = aqualoc_score.tables({"seq01": scores})
        self.assertEqual(len(scores["full"]["gyro"]), 2)
        self.assertEqual([r["first"] for r in scores["windows"]["warm"]], [0, 5, 0, 5])
        row = tables["seq01"]["windows"]["gyro"]
        self.assertEqual(row["runs"], 4)
        self.assertIn("lower_share", row["ate_rmse"])
        self.assertNotIn("lower_share", tables["seq01"]["windows"]["off"]["ate_rmse"])

    def test_seeds_spread_over_processes_score_the_same(self):
        with patch.object(mvo, "SEEDS", (mvo.SEED, 2)), patch("builtins.print"):
            serial = aqualoc_score.score_all([self.prepared], 1)
            spread = aqualoc_score.score_all([self.prepared], 2)
        self.assertEqual(serial, spread)
        self.assertEqual([r["seed"] for r in spread["seq01"]["full"]["gyro"]], [mvo.SEED, 2])

    def test_prepare_refuses_to_overwrite(self):
        with self.assertRaises(FileExistsError):
            aqualoc.prepare(
                self.files["raw"], self.files["trajectory"], self.files["camera"], self.files["imu"], self.prepared
            )


if __name__ == "__main__":
    unittest.main()
