"""A camera's image onto the odometry's 320x240 frame (core/frame_prep.c through uwslam.mvo): each output pixel is the
mean of the source rectangle it covers, rounded half up, checked against an exact reference for whole and fractional
scales and the largest size; the grey value is every step-th byte along a row (a YUV444 image's Y, a YUYV image's
every other byte); complementary row ranges make the same frame however the rows are split, rows outside the range
are left alone, and refused sizes, steps, strides and rows leave the output untouched.

Needs gcc; skipped without it.
"""

import shutil
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    from uwslam import mvo

WIDTH, HEIGHT = 320, 240


def overlaps(size: int, out: int) -> np.ndarray:
    """weights[o, i]: the overlap of output cell o, [o size, (o + 1) size), with source cell i, [i out, (i + 1) out)."""
    o = np.arange(out)[:, None]
    i = np.arange(size)[None, :]
    return np.clip(np.minimum((o + 1) * size, (i + 1) * out) - np.maximum(o * size, i * out), 0, None)


def reference(grey: np.ndarray) -> np.ndarray:
    """The area mean in exact integers: weights add up to w h per output pixel; rounded half up."""
    h, w = grey.shape
    acc = overlaps(h, HEIGHT).astype(np.int64) @ grey.astype(np.int64) @ overlaps(w, WIDTH).T.astype(np.int64)
    return ((acc + (w * h) // 2) // (w * h)).astype(np.uint8)


@unittest.skipUnless(HAS_DEPENDENCIES, "needs gcc and OpenCV")
class FramePrepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.vo = mvo.build(cls.folder)
        cls.rng = np.random.default_rng(7)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def resize(self, data: np.ndarray, w: int, h: int, stride: int, step: int, rows=(0, HEIGHT), fill=0):
        out = np.full((HEIGHT, WIDTH), fill, dtype=np.uint8)
        ok = self.vo.fp_area_resize(data.ctypes.data, w, h, stride, step, out.ctypes.data, *rows)
        return ok, out

    def test_the_area_mean_for_whole_and_fractional_scales(self):
        for w, h in ((320, 240), (960, 544), (640, 480), (333, 241), (1001, 777), (2048, 2048)):
            with self.subTest(size=(w, h)):
                grey = self.rng.integers(0, 256, (h, w), dtype=np.uint8)
                self.assertTrue(np.array_equal(mvo.area_resize(self.vo, grey), reference(grey)))

    def test_extremes_and_rounding(self):
        white = np.full((544, 960), 255, dtype=np.uint8)
        self.assertTrue((mvo.area_resize(self.vo, white) == 255).all())  # no overflow at the largest sums
        grey = np.zeros((241, 321), dtype=np.uint8)
        grey[::2] = 1  # means of exactly one half and near it
        self.assertTrue(np.array_equal(mvo.area_resize(self.vo, grey), reference(grey)))

    def test_the_grey_value_every_step_bytes(self):
        for step, w, h in ((2, 800, 600), (3, 960, 544), (4, 333, 250)):
            with self.subTest(step=step):
                grey = self.rng.integers(0, 256, (h, w), dtype=np.uint8)
                pixels = self.rng.integers(0, 256, (h, w * step + 16), dtype=np.uint8)  # rows padded, as the engine's
                y = step - 1  # where the grey byte sits, as Y in the JPEG engine's V, U, Y
                pixels[:, y : y + w * step : step] = grey
                view = pixels[:, y:]
                ok, out = self.resize(view, w, h, pixels.strides[0], step)
                self.assertEqual(ok, 1)
                self.assertTrue(np.array_equal(out, reference(grey)))

    def test_any_split_of_the_rows_makes_the_same_frame(self):
        grey = self.rng.integers(0, 256, (544, 960), dtype=np.uint8)
        whole = mvo.area_resize(self.vo, grey)
        for cuts in ((0, 120, 240), (0, 1, 239, 240), (0, 77, 78, 200, 240)):
            out = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            for begin, end in zip(cuts, cuts[1:]):
                self.assertEqual(
                    self.vo.fp_area_resize(grey.ctypes.data, 960, 544, 960, 1, out.ctypes.data, begin, end), 1
                )
            self.assertTrue(np.array_equal(out, whole), cuts)
        ok, out = self.resize(grey, 960, 544, 960, 1, rows=(10, 20), fill=7)
        self.assertTrue(np.array_equal(out[10:20], whole[10:20]))
        self.assertTrue((out[:10] == 7).all() and (out[20:] == 7).all())

    def test_refusals_leave_the_output_untouched(self):
        grey = np.zeros((2049, 2049), dtype=np.uint8)
        cases = [
            (319, 240, 319, 1, (0, 240)),  # narrower than the frame
            (320, 239, 320, 1, (0, 240)),  # lower than the frame
            (2049, 240, 2049, 1, (0, 240)),  # wider than FP_MAX_SIDE
            (320, 2049, 320, 1, (0, 240)),  # taller than FP_MAX_SIDE
            (320, 240, 319, 1, (0, 240)),  # stride shorter than a row
            (320, 240, 3 * 319, 3, (0, 240)),  # stride shorter than the last pixel's byte
            (320, 240, 320, 0, (0, 240)),  # no step
            (320, 240, 320 * 5, 5, (0, 240)),  # step beyond 4
            (320, 240, 320, 1, (-1, 240)),  # rows before the frame
            (320, 240, 320, 1, (0, 241)),  # rows after it
            (320, 240, 320, 1, (100, 99)),  # rows reversed
        ]
        for w, h, stride, step, rows in cases:
            with self.subTest(w=w, h=h, stride=stride, step=step, rows=rows):
                ok, out = self.resize(grey, w, h, stride, step, rows, fill=9)
                self.assertEqual(ok, 0)
                self.assertTrue((out == 9).all())
        out = np.full((HEIGHT, WIDTH), 9, dtype=np.uint8)
        self.assertEqual(self.vo.fp_area_resize(None, 320, 240, 320, 1, out.ctypes.data, 0, 240), 0)  # no image
        self.assertEqual(self.vo.fp_area_resize(grey.ctypes.data, 320, 240, 320, 1, None, 0, 240), 0)  # no frame
        self.assertTrue((out == 9).all())
        self.assertEqual(self.resize(grey, 320, 240, 3 * 319 + 1, 3)[0], 1)  # just long enough
        self.assertEqual(self.resize(grey, 320, 240, 320, 1, (5, 5), fill=9)[0], 1)  # no rows is no work
        with self.assertRaises(ValueError):
            mvo.area_resize(self.vo, np.zeros((240, 320, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
