"""Monocular visual odometry (uwslam.experiments.mono_vo): frame path, tracking, pure rotations, scale carried by
triangulated points, and the similarity alignment used to score it.

Needs OpenCV; skipped without it.
"""

from importlib.util import find_spec
import unittest

import numpy as np

HAS_DEPENDENCIES = find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    import cv2

    from uwslam.experiments import mono_vo as mv
    from uwslam.experiments.mask_motion import angle_deg, euler_rotation


def texture(seed: int, shape: tuple[int, int] = (240, 320)) -> np.ndarray:
    """Smooth random grey texture with plenty of corners."""
    coarse = np.random.default_rng(seed).integers(0, 256, (shape[0] // 6, shape[1] // 6), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(coarse, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC), (3, 3), 0)


def project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pixels = points @ matrix.T
    return np.float32(pixels[:, :2] / pixels[:, 2:3])


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs OpenCV")
class FrameAndTrackingTests(unittest.TestCase):
    def test_robot_grey_has_both_resolutions(self):
        rgb = np.random.default_rng(1).integers(0, 256, (544, 960, 3), dtype=np.uint8)
        self.assertEqual(mv.robot_grey(rgb, (320, 240)).shape, (240, 320))
        self.assertEqual(mv.robot_grey(rgb, (160, 120)).shape, (120, 160))
        with self.assertRaises(ValueError):
            mv.robot_grey(rgb, (80, 60))

    def test_tracks_follow_a_shift(self):
        first = texture(2)
        second = np.roll(np.roll(first, 2, axis=0), 3, axis=1)
        points = mv.detect(first, np.zeros((0, 2), np.float32), 1.0, 200)
        points = points[(points[:, 0] > 30) & (points[:, 0] < 290) & (points[:, 1] > 30) & (points[:, 1] < 210)]
        moved, keep = mv.track(first, second, points, 1.0)
        self.assertGreater(keep.mean(), 0.9)
        np.testing.assert_allclose(np.median(moved - points[keep], axis=0), [3.0, 2.0], atol=0.1)


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs OpenCV")
class MotionTests(unittest.TestCase):
    camera = None

    @classmethod
    def setUpClass(cls):
        cls.camera = mv.Camera.from_fov(110.0, 544 / 960, 320, 240)

    def test_a_pure_turn_is_found_from_the_homography(self):
        first = texture(3)
        rotation_ab = euler_rotation(np.radians(4.0), np.radians(-1.0), 0.0)
        matrix = self.camera.matrix()
        warp = matrix @ rotation_ab.T @ np.linalg.inv(matrix)  # x_B = K R K^-1 x_A with R = R_AB^T
        second = cv2.warpPerspective(first, warp, (320, 240), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        vo = mv.MonoVO(self.camera, 1.0)
        vo.start(first)
        step = vo.step(second)
        self.assertEqual(step.kind, "rotation")
        self.assertLess(angle_deg(step.rotation.T @ rotation_ab), 0.3)
        np.testing.assert_array_equal(step.centre, np.zeros(3))

    def test_scale_is_carried_by_points_seen_in_both_steps(self):
        rng = np.random.default_rng(4)
        world = np.column_stack([rng.uniform(-3, 3, 300), rng.uniform(-2, 2, 300), rng.uniform(4, 10, 300)])
        matrix = self.camera.matrix()
        poses = [(np.eye(3), np.zeros(3)),  # camera-to-world rotation and centre, world = first camera
                 (euler_rotation(np.radians(3.0), 0.0, 0.0), np.array([0.4, 0.0, 0.9])),
                 (euler_rotation(np.radians(5.0), np.radians(1.0), 0.0), np.array([0.6, 0.05, 1.35]))]
        views = [project((world - centre) @ rotation, matrix) for rotation, centre in poses]
        vo = mv.MonoVO(self.camera, 1.0)
        ids = np.arange(len(world))
        lengths = []
        for k in (0, 1):
            (r_a, c_a), (r_b, c_b) = poses[k], poses[k + 1]
            rotation_cv = r_b.T @ r_a  # X_B = R X_A + t
            translation = -rotation_cv @ (r_a.T @ (c_b - c_a))
            step = vo.translate(views[k], views[k + 1], ids, rotation_cv, translation / np.linalg.norm(translation))
            lengths.append(np.linalg.norm(step.centre))
            np.testing.assert_allclose(step.centre / lengths[-1], r_a.T @ (c_b - c_a) / np.linalg.norm(c_b - c_a), atol=1e-6)
        true_ratio = np.linalg.norm(poses[2][1] - poses[1][1]) / np.linalg.norm(poses[1][1] - poses[0][1])
        self.assertAlmostEqual(lengths[0], 1.0)
        self.assertAlmostEqual(lengths[1] / lengths[0], true_ratio, places=3)

    def test_triangulation_keeps_points_in_front_of_both_cameras(self):
        rng = np.random.default_rng(7)
        world = np.column_stack([rng.uniform(-2, 2, 100), rng.uniform(-1.5, 1.5, 100), rng.uniform(3, 8, 100)])
        matrix = self.camera.matrix()
        pose_a = (np.eye(3), np.zeros(3))
        pose_b = (euler_rotation(np.radians(4.0), 0.0, 0.0), np.array([0.5, 0.0, 0.2]))
        pixels = [project((world - centre) @ rotation, matrix) for rotation, centre in (pose_a, pose_b)]
        found, good = mv.triangulate_world(pixels[0], pixels[1], pose_a, pose_b, matrix, 1.0)
        self.assertTrue(good.all())
        np.testing.assert_allclose(found, world, atol=1e-3)
        found, good = mv.triangulate_world(pixels[0], pixels[0], pose_a, pose_a, matrix, 1.0)
        self.assertFalse(good.any())  # no baseline: no parallax angle

    def test_turned_pixels_follow_a_pure_rotation(self):
        rng = np.random.default_rng(8)
        world = np.column_stack([rng.uniform(-3, 3, 50), rng.uniform(-2, 2, 50), rng.uniform(3, 8, 50)])
        matrix = self.camera.matrix()
        rotation_ab = euler_rotation(np.radians(-7.0), np.radians(2.0), np.radians(1.0))
        np.testing.assert_allclose(mv.turned_pixels(project(world, matrix), rotation_ab, matrix),
                                   project(world @ rotation_ab, matrix), atol=1e-3)

    def test_parallax_ignores_rotation_and_sees_translation(self):
        rng = np.random.default_rng(5)
        world = np.column_stack([rng.uniform(-3, 3, 200), rng.uniform(-2, 2, 200), rng.uniform(3, 8, 200)])
        matrix = self.camera.matrix()
        rotation_ab = euler_rotation(np.radians(6.0), 0.0, 0.0)
        a = project(world, matrix)
        turned = project(world @ rotation_ab, matrix)  # same centre, turned camera
        moved = project((world - [0.5, 0.0, 0.0]) @ rotation_ab, matrix)
        self.assertLess(mv.parallax(a, turned, rotation_ab, matrix), 1e-3)
        self.assertGreater(mv.parallax(a, moved, rotation_ab, matrix), 3.0)


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs OpenCV")
class AlignmentTests(unittest.TestCase):
    def test_umeyama_recovers_a_similarity(self):
        rng = np.random.default_rng(6)
        source = rng.normal(size=(50, 3))
        rotation = euler_rotation(0.4, -0.2, 0.1)
        target = 2.5 * source @ rotation.T + [1.0, -2.0, 0.5]
        scale, found, offset = mv.umeyama(source, target)
        self.assertAlmostEqual(scale, 2.5, places=9)
        np.testing.assert_allclose(found, rotation, atol=1e-9)
        np.testing.assert_allclose(offset, [1.0, -2.0, 0.5], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
