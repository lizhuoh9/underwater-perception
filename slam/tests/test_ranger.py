"""The lake's simulated rangers (uwslam.ranger) without the clip: echoes from a synthetic wall, each run's own noise and
dropouts, and the alignments the scores use."""

import unittest

import numpy as np

from uwslam import ranger
from uwslam.aqualoc import exp_so3


def wall(z: float, half: float = 3.0, step: float = 0.02) -> np.ndarray:
    """A wall's points at z, facing a camera that looks along z."""
    u, v = np.meshgrid(np.arange(-half, half, step), np.arange(-half, half, step))
    return np.column_stack([u.ravel(), v.ravel(), np.full(u.size, z)]).astype(np.float32)


class RangerTests(unittest.TestCase):
    def test_the_forward_ranger_hears_the_wall_ahead_and_the_sides_nothing(self):
        centres = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 5.0]])  # the last one past the wall
        true = ranger.echoes(wall(4.0), np.stack([np.eye(3)] * 3), centres, 7.5)
        self.assertAlmostEqual(true["forward"][0], 4.0, delta=0.01)
        self.assertAlmostEqual(true["forward"][1], 3.0, delta=0.01)
        self.assertTrue(np.isnan(true["forward"][2]))  # the wall behind it
        self.assertTrue(np.all(np.isnan(true["left"])) and np.all(np.isnan(true["right"])))
        turned = exp_so3(np.array([0.0, np.pi / 2, 0.0]))  # the camera turned to face the wall with its left side
        sideways = ranger.echoes(wall(4.0), turned[None], np.zeros((1, 3)), 7.5)
        self.assertAlmostEqual(sideways["left"][0], 4.0, delta=0.01)
        self.assertTrue(np.isnan(sideways["forward"][0]))

    def test_each_run_reads_with_its_own_noise_and_dropouts(self):
        true = np.full(2000, 2.0)
        true[::10] = np.nan
        first, again, other = ranger.readings(true, 1), ranger.readings(true, 1), ranger.readings(true, 2)
        np.testing.assert_array_equal(first, again)  # a seed reads the same each time
        self.assertFalse(np.array_equal(first, other, equal_nan=True))
        self.assertTrue(np.all(np.isnan(first[::10])))  # no echo, no reading
        self.assertAlmostEqual(np.isnan(first).mean() - 0.1, 0.9 * ranger.DROPOUT, delta=0.02)
        heard = first[np.isfinite(first)]
        self.assertAlmostEqual(float(np.std(heard - 2.0)), ranger.NOISE_M + 2.0 * ranger.NOISE_SHARE, delta=0.003)
        np.testing.assert_array_equal(heard, np.round(heard, 3))  # millimetres

    def test_the_alignments(self):
        rng = np.random.default_rng(4)
        reference = rng.normal(size=(60, 3))
        moved = reference @ exp_so3(np.array([0.3, -0.5, 0.8])).T + np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(ranger.ate(moved, reference, False), 0.0, places=9)
        self.assertAlmostEqual(ranger.ate(0.5 * moved, reference, True), 0.0, places=9)
        self.assertGreater(ranger.ate(0.5 * moved, reference, False), 0.1)  # no scale fitted
        self.assertEqual(len(ranger.windows(moved[:51], reference[:51], False)), 1)
        self.assertEqual(len(ranger.windows(np.vstack([moved] * 2), np.vstack([reference] * 2), False)), 3)
        self.assertIsNone(ranger.spread([]))


if __name__ == "__main__":
    unittest.main()
