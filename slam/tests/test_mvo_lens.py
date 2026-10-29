"""The lens model (mvo_camera_t's k1, k2, p1, p2, k3: OpenCV's): mvo_undistort against OpenCV, a pinhole's exactness,
the odometry on frames rendered through a distorted lens, and the obstacle directions' layout.

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
    import cv2

    from tests.test_mvo import render, texture
    from uwslam import mvo, obstacle_dir
    from uwslam.experiments.mask_motion import angle_deg, euler_rotation

# An OV2640-like wide lens (strong barrel, slight decentring) and a flat port's pincushion.
BARREL = (-0.25, 0.06, 0.001, -0.0008, -0.005)
PINCUSHION = (0.04, 0.01, 0.0, 0.0, 0.0)
CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12) if HAS_DEPENDENCIES else None


def lens_camera(fx: float, fy: float, lens: tuple) -> "mvo.Camera":
    return mvo.Camera(fx, fy, mvo.WIDTH / 2, mvo.HEIGHT / 2, *lens)


def opencv_undistorted(camera, points: np.ndarray) -> np.ndarray:
    """Pixels as a pinhole with the same intrinsics would see them, by OpenCV to convergence."""
    k = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([camera.k1, camera.k2, camera.p1, camera.p2, camera.k3], dtype=np.float64)
    return cv2.undistortPoints(points.reshape(-1, 1, 2).astype(np.float64), k, dist, R=None, P=k, criteria=CRITERIA)[
        :, 0
    ]


def distorted(camera, points: np.ndarray) -> np.ndarray:
    """Pinhole pixels put back through the lens by OpenCV's projection."""
    k = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([camera.k1, camera.k2, camera.p1, camera.p2, camera.k3], dtype=np.float64)
    rays = np.stack(
        [(points[:, 0] - camera.cx) / camera.fx, (points[:, 1] - camera.cy) / camera.fy, np.ones(len(points))], axis=1
    )
    return cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, dist)[0][:, 0]


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs gcc and OpenCV")
class LensTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.lib = mvo.build(cls.folder)
        cls.lib.mvo_undistort.argtypes = [
            ctypes.POINTER(mvo.Camera),
            ctypes.POINTER(ctypes.c_float * 2),
            ctypes.POINTER(ctypes.c_float * 2),
        ]
        cls.lib.mvo_undistort.restype = ctypes.c_int
        cls.wall, cls.floor = texture(1), texture(2)

    @classmethod
    def tearDownClass(cls):
        del cls.lib
        shutil.rmtree(cls.folder, ignore_errors=True)

    def undistort(self, camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out, ok = np.zeros((len(points), 2), dtype=np.float32), np.zeros(len(points), dtype=int)
        for i, (u, v) in enumerate(points):
            source, target = (ctypes.c_float * 2)(u, v), (ctypes.c_float * 2)()
            ok[i] = self.lib.mvo_undistort(ctypes.byref(camera), ctypes.byref(source), ctypes.byref(target))
            out[i] = target[:]
        return out, ok

    def test_undistortion_matches_opencv(self):
        v, u = np.mgrid[0 : mvo.HEIGHT + 1 : 8, 0 : mvo.WIDTH + 1 : 8]
        points = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float64)
        for name, camera in (
            ("barrel", lens_camera(160.0, 160.0, BARREL)),
            ("pincushion", lens_camera(112.0, 148.0, PINCUSHION)),
            ("decentred", lens_camera(200.0, 200.0, (-0.1, 0.0, 0.003, 0.002, 0.0))),
        ):
            with self.subTest(name):
                found, ok = self.undistort(camera, points)
                self.assertTrue(ok.all())
                back = np.abs(distorted(camera, found.astype(np.float64)) - points).max()
                self.assertLess(back, 1e-3, f"{name}: back through the lens {back:.2e} px off")
                error = np.abs(found - opencv_undistorted(camera, points)).max()
                self.assertLess(error, 1e-3, f"{name}: {error:.2e} px off OpenCV's")

    def test_a_pinhole_leaves_pixels_exact(self):
        rng = np.random.default_rng(3)
        points = rng.uniform([-20, -20], [340, 260], (500, 2)).astype(np.float32)
        for lens in ((0.0,) * 5, (-0.0,) * 5):
            found, ok = self.undistort(lens_camera(112.0302963, 148.2792358, lens), points)
            self.assertTrue(ok.all())
            np.testing.assert_array_equal(found.view(np.uint32), points.view(np.uint32))

    def test_a_folding_model_is_refused(self):
        # k1 = -1: the radius shrinks back to zero at r = 1, so beyond it no undistorted point maps there.
        camera = lens_camera(100.0, 100.0, (-1.0, 0.0, 0.0, 0.0, 0.0))
        _, ok = self.undistort(camera, np.array([[319.0, 239.0], [160.0, 120.0]]))
        self.assertEqual(ok.tolist(), [0, 1])
        with self.assertRaises(ValueError):
            obstacle_dir.layout_for(self.lib, camera)
        with self.assertRaises(ValueError):
            obstacle_dir.layout_for(self.lib, lens_camera(100.0, 100.0, (float("nan"), 0.0, 0.0, 0.0, 0.0)))

    def test_odometry_through_a_distorted_lens(self):
        # The same motion as test_mvo's moving camera, rendered through the barrel lens. Told the lens, the odometry
        # keeps the pinhole case's accuracy; treating the same frames as a pinhole bends every ray and costs it.
        camera = lens_camera(160.0, 160.0, BARREL)
        pinhole = lens_camera(160.0, 160.0, (0.0,) * 5)
        v, u = np.mgrid[0 : mvo.HEIGHT, 0 : mvo.WIDTH].astype(np.float64)
        undistorted = opencv_undistorted(camera, np.stack([u.ravel(), v.ravel()], axis=1))
        normalised = np.stack(
            [(undistorted[:, 0] - camera.cx) / camera.fx, (undistorted[:, 1] - camera.cy) / camera.fy], axis=1
        ).reshape(mvo.HEIGHT, mvo.WIDTH, 2)
        poses = [(euler_rotation(np.radians(1.5 * k), 0.0, 0.0), np.array([0.30 * k, 0.0, 0.15 * k])) for k in range(8)]
        frames = [render(r, c, camera, self.wall, self.floor, normalised) for r, c in poses]
        errors = {}
        for name, model in (("lens", camera), ("pinhole", pinhole)):
            steps, _, _ = mvo.run_frames(self.lib, frames, model)
            turn, travel = [], []
            for k, step in enumerate(steps):
                (r_a, c_a), (r_b, c_b) = poses[k], poses[k + 1]
                truth = r_a.T @ (c_b - c_a)
                estimate = np.array(step.centre, dtype=np.float64)
                cosine = truth @ estimate / (np.linalg.norm(truth) * max(np.linalg.norm(estimate), 1e-12))
                travel.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                turn.append(angle_deg(np.array(step.rotation, dtype=np.float64).reshape(3, 3).T @ (r_a.T @ r_b)))
            errors[name] = (max(turn), max(travel), [mvo.KINDS[s.kind] for s in steps])
        turn, travel, kinds = errors["lens"]
        self.assertEqual(kinds.count("essential"), len(kinds), kinds)
        self.assertLess(turn, 0.5)
        self.assertLess(travel, 5.0)
        self.assertGreater(errors["pinhole"][0], 2 * turn, errors)  # here 2.8 against 0.06 degrees
        self.assertGreater(errors["pinhole"][1], 2 * travel, errors)  # 11 against 0.7

    def test_layout_takes_the_undistorted_rays(self):
        camera = lens_camera(160.0, 160.0, BARREL)
        layout = obstacle_dir.layout_for(self.lib, camera)
        columns = opencv_undistorted(camera, np.stack([(np.arange(80) + 0.5) * 4, np.full(80, camera.cy)], axis=1))
        rows = opencv_undistorted(camera, np.stack([np.full(60, camera.cx), (np.arange(60) + 0.5) * 4], axis=1))
        x, y = (columns[:, 0] - camera.cx) / camera.fx, (rows[:, 1] - camera.cy) / camera.fy
        inner, outer = np.tan(np.radians(15)), np.tan(np.radians(40))
        np.testing.assert_array_equal(
            np.array(layout.sector[:80]), np.digitize(x, [-outer, -inner, inner + 1e-7, outer + 1e-7])
        )
        np.testing.assert_array_equal(np.array(layout.band[:60]), np.digitize(y, [-inner, inner + 1e-7]))
        pinhole = obstacle_dir.layout_for(self.lib, lens_camera(160.0, 160.0, (0.0,) * 5))
        self.assertGreater(sum(s in (0, 4) for s in layout.sector[:80]), sum(s in (0, 4) for s in pinhole.sector[:80]))


if __name__ == "__main__":
    unittest.main()
