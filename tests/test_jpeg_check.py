"""The live build's JPEG check (firmware/mono_vo_stream/main/jpeg_check.c), built on the PC: OpenCV's baseline JPEGs
at 4:2:0, 4:2:2 and 4:4:4 pass with their size and MCU; grey, progressive, 4:1:1 and 4:4:0 JPEGs, sizes outside the
limits or with a pixel count the driver refuses, truncated or garbled data and data that is no JPEG fail, leaving the
shape untouched.

Needs gcc and OpenCV; skipped without either.
"""

import ctypes
import shutil
import subprocess
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from uwslam.paths import FIRMWARE_DIR

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
SOURCE = FIRMWARE_DIR / "mono_vo_stream" / "main" / "jpeg_check.c"
MIN_WIDTH, MIN_HEIGHT, MAX_SIDE = 320, 240, 2048


class Shape(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ("width", "height", "mcu_width", "mcu_height")]


def encode(image: np.ndarray, *params: int) -> bytes:
    import cv2

    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80, *params])
    if not ok:
        raise RuntimeError("OpenCV did not encode")
    return data.tobytes()


@unittest.skipUnless(HAS_DEPENDENCIES, "needs gcc and OpenCV")
class JpegCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        library = cls.folder / "jpeg_check.dll"
        subprocess.run(["gcc", "-std=c99", "-O2", "-shared", str(SOURCE), "-o", str(library)], check=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.jpeg_check.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(Shape),
        ]
        cls.lib.jpeg_check.restype = ctypes.c_int
        rng = np.random.default_rng(5)
        cls.colour = rng.integers(0, 256, (544, 960, 3), dtype=np.uint8)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def check(self, data: bytes) -> tuple[int, tuple]:
        shape = Shape(7, 7, 7, 7)
        ok = self.lib.jpeg_check(data, len(data), MIN_WIDTH, MIN_HEIGHT, MAX_SIDE, ctypes.byref(shape))
        return ok, (shape.width, shape.height, shape.mcu_width, shape.mcu_height)

    def test_baseline_colour_jpegs_pass_with_their_size_and_mcu(self):
        import cv2

        samplings = {"420": (16, 16), "422": (16, 8), "444": (8, 8)}
        for name, (mcu_width, mcu_height) in samplings.items():
            factor = getattr(cv2, f"IMWRITE_JPEG_SAMPLING_FACTOR_{name}")
            for width, height in ((960, 544), (320, 240), (1001, 776), (2048, 2048)):
                image = cv2.resize(self.colour, (width, height), interpolation=cv2.INTER_AREA)
                with self.subTest(sampling=name, size=(width, height)):
                    data = encode(image, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, factor)
                    self.assertEqual(self.check(data), (1, (width, height, mcu_width, mcu_height)))

    def test_jpegs_the_live_build_cannot_decode_fail(self):
        import cv2

        grey = cv2.cvtColor(self.colour, cv2.COLOR_BGR2GRAY)
        cases = {
            "grey": encode(grey),
            "progressive": encode(self.colour, cv2.IMWRITE_JPEG_PROGRESSIVE, 1),
            "4:1:1": encode(self.colour, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_411),
            "4:4:0": encode(self.colour, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_440),
            "too narrow": encode(self.colour[:240, :319]),
            "too low": encode(self.colour[:239, :320]),
            "too wide": encode(np.zeros((240, 2049, 3), dtype=np.uint8)),
            "pixel count not a multiple of 8": encode(self.colour[:241, :321]),
        }
        for name, data in cases.items():
            with self.subTest(name):
                self.assertEqual(self.check(data), (0, (7, 7, 7, 7)))

    def test_cut_garbled_and_foreign_data_fail(self):
        data = encode(self.colour)
        scan = data.index(b"\xff\xda")
        cases = {
            "empty": b"",
            "SOI only": b"\xff\xd8",
            "no EOI": data[:-2],
            "cut in the headers, EOI added": data[: scan // 2] + b"\xff\xd9",
            "cut before the scan, EOI added": data[:scan] + b"\xff\xd9",
            "a segment longer than the data": data[:4] + b"\xff\xff" + data[6:],
            "no SOI": b"\x00\x00" + data[2:],
            "PNG": b"\x89PNG\r\n\x1a\n" + bytes(100),
        }
        for name, case in cases.items():
            with self.subTest(name):
                self.assertEqual(self.check(case)[0], 0)
        self.assertEqual(self.lib.jpeg_check(None, 0, MIN_WIDTH, MIN_HEIGHT, MAX_SIDE, None), 0)


if __name__ == "__main__":
    unittest.main()
