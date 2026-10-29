"""Mask rotation probe (uwslam.experiments.mask_motion): boundary cells, the 3-4 chamfer transform, the rotation search,
the essential-matrix reference, reference interpolation and the turned-mask comparison.

Needs OpenCV; skipped without it.
"""

import unittest
from importlib.util import find_spec

import numpy as np

HAS_DEPENDENCIES = find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    from uwslam.experiments import mask_motion as mm

# Far scene for a pure rotation: obstacle caps on the view sphere as (yaw deg, pitch deg, radius deg) in world axes.
CAPS = ((-20.0, 5.0, 18.0), (25.0, -10.0, 12.0), (5.0, 22.0, 9.0))
ASPECT = 544 / 960  # the lake clip's source frame


def cap_mask(rotation: np.ndarray, camera: "mm.Camera") -> np.ndarray:
    """60x80 non-water mask of the far caps seen by a camera whose camera-to-world rotation is `rotation`."""
    rows, columns = np.mgrid[0 : mm.MASK_HEIGHT, 0 : mm.MASK_WIDTH]
    rays = np.stack(
        [(columns + 0.5 - camera.cx) / camera.fx, (rows + 0.5 - camera.cy) / camera.fy, np.ones(rows.shape)], axis=-1
    )
    world = rays @ rotation.T
    world /= np.linalg.norm(world, axis=-1, keepdims=True)
    solid = np.zeros(rows.shape, dtype=bool)
    for yaw, pitch, radius in CAPS:
        centre = mm.euler_rotation(np.radians(yaw), np.radians(pitch), 0.0) @ np.array([0.0, 0.0, 1.0])
        solid |= world @ centre >= np.cos(np.radians(radius))
    return solid


def chamfer_brute_force(cells: np.ndarray, truncate: int) -> np.ndarray:
    """3-4 chamfer distance by definition: 3 * max(|dx|, |dy|) + min(|dx|, |dy|) to the nearest set cell."""
    points = np.argwhere(cells)
    rows, columns = np.mgrid[0 : cells.shape[0], 0 : cells.shape[1]]
    dy = np.abs(rows[..., None] - points[:, 0])
    dx = np.abs(columns[..., None] - points[:, 1])
    return np.minimum((3 * np.maximum(dx, dy) + np.minimum(dx, dy)).min(axis=-1), truncate)


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs OpenCV")
class GeometryTests(unittest.TestCase):
    def test_boundary_is_the_inner_ring_and_ignores_the_image_border(self):
        mask = np.zeros((10, 12), dtype=bool)
        mask[2:6, 3:8] = True  # free-standing block
        mask[7:, 0:4] = True  # block touching the bottom and left border
        cells = mm.boundary_cells(mask)
        ring = np.zeros_like(mask)
        ring[2:6, 3:8] = True
        ring[3:5, 4:7] = False
        np.testing.assert_array_equal(cells[:7], ring[:7])
        self.assertTrue(cells[7, 0:4].all())  # top side faces water
        self.assertTrue(cells[7:, 3].all())  # right side faces water
        self.assertFalse(cells[8:, 0:3].any())  # the border sides and the inside are not edges

    def test_chamfer_matches_its_definition(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            cells = rng.random((17, 23)) < 0.04
            cells[rng.integers(17), rng.integers(23)] = True
            np.testing.assert_array_equal(mm.chamfer_distance(cells, truncate=40), chamfer_brute_force(cells, 40))
            np.testing.assert_array_equal(mm.chamfer_distance(cells), chamfer_brute_force(cells, mm.TRUNCATE))

    def test_euler_rotation_is_a_rotation_and_matches_its_vector(self):
        rotation = mm.euler_rotation(np.radians(14.0), np.radians(-6.0), np.radians(3.0))
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)
        self.assertAlmostEqual(
            np.degrees(np.linalg.norm(mm.rotation_vector(rotation))), mm.angle_deg(rotation), places=9
        )
        yaw_only = mm.rotation_vector(mm.euler_rotation(np.radians(10.0), 0.0, 0.0))
        np.testing.assert_allclose(np.degrees(yaw_only), [0.0, 10.0, 0.0], atol=1e-9)

    def test_intrinsics_follow_the_stretched_frame(self):
        camera = mm.Camera.from_fov(100.0, 544 / 960)
        self.assertAlmostEqual(camera.fx, 40 / np.tan(np.radians(50.0)))
        self.assertAlmostEqual(camera.fy, 30 / (544 / 960 * np.tan(np.radians(50.0))))
        self.assertEqual((camera.cx, camera.cy), (40, 30))


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs OpenCV")
class SearchTests(unittest.TestCase):
    camera = None

    @classmethod
    def setUpClass(cls):
        cls.camera = mm.Camera.from_fov(100.0, ASPECT)

    def test_recovers_a_pure_rotation_of_a_far_scene(self):
        for yaw, pitch, roll in ((7.0, -3.0, 2.0), (-31.0, 4.0, -5.0), (0.0, 0.0, 0.0)):
            rotation_b = mm.euler_rotation(np.radians(yaw), np.radians(pitch), np.radians(roll))
            first = mm.boundary_cells(cap_mask(np.eye(3), self.camera))
            second = mm.boundary_cells(cap_mask(rotation_b, self.camera))
            found = mm.search_rotation(first, second, self.camera)
            self.assertIsNotNone(found)
            self.assertLess(mm.angle_deg(found["rotation"].T @ rotation_b), 1.0, (yaw, pitch, roll, found))

    def test_identical_masks_give_no_rotation(self):
        cells = mm.boundary_cells(cap_mask(np.eye(3), self.camera))
        found = mm.search_rotation(cells, cells, self.camera)
        self.assertEqual((found["yaw_deg"], found["pitch_deg"], found["roll_deg"], found["cost"]), (0.0, 0.0, 0.0, 0))

    def test_declines_frames_without_enough_boundary(self):
        empty = np.zeros((mm.MASK_HEIGHT, mm.MASK_WIDTH), dtype=bool)
        cells = mm.boundary_cells(cap_mask(np.eye(3), self.camera))
        self.assertIsNone(mm.search_rotation(empty, cells, self.camera))
        self.assertIsNone(mm.search_rotation(cells, empty, self.camera))

    def test_turned_mask_reproduces_the_rotated_view(self):
        rotation_b = mm.euler_rotation(np.radians(9.0), np.radians(2.0), np.radians(-1.0))
        first, second = cap_mask(np.eye(3), self.camera), cap_mask(rotation_b, self.camera)
        seen, known = mm.turned_mask(first, rotation_b, self.camera)
        self.assertGreater(known.mean(), 0.7)
        self.assertLess(np.mean(seen[known] != second[known]), 0.02)  # only boundary cells may differ by discretisation
        same, everywhere = mm.turned_mask(first, np.eye(3), self.camera)
        self.assertTrue(everywhere.all())
        np.testing.assert_array_equal(same, first)

    def test_edge_cells_pool_canny_onto_the_mask_grid(self):
        gray = np.zeros((120, 160), dtype=np.uint8)
        gray[:, 81:] = 200
        cells = mm.edge_cells(gray)
        self.assertEqual(cells.shape, (mm.MASK_HEIGHT, mm.MASK_WIDTH))
        self.assertTrue(cells[:, 40].all())
        self.assertEqual(int(cells[:, :39].sum() + cells[:, 42:].sum()), 0)


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs OpenCV")
class ReferenceTests(unittest.TestCase):
    def test_interpolates_rotation_by_slerp_and_centres_linearly_in_video_time(self):
        half = np.sin(np.radians(45.0))
        poses = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 0.0, -4.0, 0.0, half, 0.0, half]]
        )  # 90 degrees about y, 1 s of images.txt time
        rotations, centres = mm.reference_poses(poses, np.array([0.0, 0.75, 3.0 + 5e-7]), time_scale=3.0)
        self.assertAlmostEqual(mm.angle_deg(rotations[1]), 22.5, places=9)
        np.testing.assert_allclose(centres[1], [0.5, 0.0, -1.0])
        self.assertAlmostEqual(mm.angle_deg(rotations[2]), 90.0, places=6)
        with self.assertRaises(ValueError):
            mm.reference_poses(poses, np.array([1.1]), time_scale=1.0)

    def test_essential_rotation_recovers_a_moving_camera(self):
        rng = np.random.default_rng(11)
        camera = mm.Camera.from_fov(90.0, 544 / 960, 960, 544)
        points = np.column_stack([rng.uniform(-3, 3, 300), rng.uniform(-2, 2, 300), rng.uniform(4, 12, 300)])
        rotation_ab = mm.euler_rotation(*np.radians([6.0, -2.0, 1.0]))
        centre_b = np.array([0.3, 0.05, 0.4])  # camera B's centre in A's frame
        in_b = (points - centre_b) @ rotation_ab  # X_B = R_AB^T (X_A - c_B)

        def pixels(xyz):
            return np.float32(
                np.column_stack(
                    [camera.fx * xyz[:, 0] / xyz[:, 2] + camera.cx, camera.fy * xyz[:, 1] / xyz[:, 2] + camera.cy]
                )
            )

        found = mm.essential_rotation(pixels(points), pixels(in_b), camera.matrix())
        self.assertIsNotNone(found)
        self.assertLess(mm.angle_deg(found.T @ rotation_ab), 0.05)
        self.assertIsNone(mm.essential_rotation(pixels(points)[:10], pixels(in_b)[:10], camera.matrix()))

    def test_pairs_start_every_third_frame(self):
        pairs = mm.frame_pairs(40)
        self.assertEqual(pairs[10.0][:3], [(0, 3), (3, 6), (6, 9)])
        self.assertEqual(pairs[3.0][-1], (27, 37))
        self.assertTrue(all(b < 40 for rate_pairs in pairs.values() for _, b in rate_pairs))


if __name__ == "__main__":
    unittest.main()
