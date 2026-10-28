"""The monocular odometry's units (core/mono_vo*.c): the PC build, the fingerprint the board is checked against and the
firmware's build must name the same files, in the same order where they are hashed. No compiler needed."""

import re
import unittest

from uwslam import mvo
from uwslam.paths import CORE_DIR, FIRMWARE_DIR

FIRMWARE_CMAKE = FIRMWARE_DIR / "esp32s3_mono_vo" / "main" / "CMakeLists.txt"


class MonoVoSourcesTests(unittest.TestCase):
    def test_every_unit_is_built_and_fingerprinted(self):
        units = sorted(path.name for path in CORE_DIR.iterdir()
                       if path.name.startswith("mono_vo") and path.suffix in (".c", ".h") and path.name != "mono_vo.h")
        self.assertEqual(sorted(mvo.CORE_FILES), units)
        self.assertEqual([path.name for path in mvo.SOURCES], [name for name in mvo.CORE_FILES if name.endswith(".c")])

    def test_the_firmware_builds_and_hashes_the_same_units(self):
        text = FIRMWARE_CMAKE.read_text(encoding="utf-8")
        hashed = re.search(r"set\(MVO_CORE_FILES([^)]*)\)", text).group(1).split()
        self.assertEqual(tuple(hashed), mvo.CORE_FILES)  # the order makes the fingerprint
        built = re.findall(r'"\.\./\.\./\.\./core/(mono_vo[^"]*\.c)"', text)
        self.assertEqual(sorted(built), sorted(path.name for path in mvo.SOURCES))


if __name__ == "__main__":
    unittest.main()
