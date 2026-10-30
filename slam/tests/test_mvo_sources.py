"""The monocular odometry's units (core/mono_vo*.c): the PC build, the fingerprint the board is checked against and the
firmware's build must name the same files, in the same order where they are hashed; the live build's frame preparation
(core/frame_prep.c and its header) is built and reported as uwslam.live_feed checks it. No compiler needed."""

import re
import unittest

from uwslam import live_feed, mvo
from uwslam.paths import CORE_DIR, FIRMWARE_DIR

FIRMWARE_CMAKE = FIRMWARE_DIR / "mono_vo_stream" / "main" / "CMakeLists.txt"


class MonoVoSourcesTests(unittest.TestCase):
    def test_every_unit_is_built_and_fingerprinted(self):
        units = sorted(
            path.name
            for path in CORE_DIR.iterdir()
            if path.name.startswith("mono_vo") and path.suffix in (".c", ".h") and path.name != "mono_vo.h"
        )
        self.assertEqual(sorted(mvo.CORE_FILES), units)
        self.assertEqual([path.name for path in mvo.SOURCES], [name for name in mvo.CORE_FILES if name.endswith(".c")])

    def test_the_firmware_builds_and_hashes_the_same_units(self):
        text = FIRMWARE_CMAKE.read_text(encoding="utf-8")
        hashed = re.search(r"set\(MVO_CORE_FILES([^)]*)\)", text).group(1).split()
        self.assertEqual(tuple(hashed), mvo.CORE_FILES)  # the order makes the fingerprint
        built = re.findall(r'"\.\./\.\./\.\./core/(mono_vo[^"]*\.c)"', text)
        self.assertEqual(sorted(built), sorted(path.name for path in mvo.SOURCES))

    def test_the_live_build_reports_the_frame_preparation_the_host_checks(self):
        text = FIRMWARE_CMAKE.read_text(encoding="utf-8")
        self.assertIn(f'"../../../core/{mvo.PREP_SOURCE.name}"', text)
        self.assertIn(f'file(SHA256 "${{CORE}}/{mvo.PREP_SOURCE.name}" FP_CORE_SHA256)', text)
        self.assertIn(f'file(SHA256 "${{CORE}}/{mvo.PREP_HEADER.name}" FP_HEADER_SHA256)', text)
        main = (FIRMWARE_CMAKE.parent / "main.c").read_text(encoding="utf-8")
        self.assertIn('" live=1 prep_sha256=" FP_CORE_SHA256 " prep_header_sha256=" FP_HEADER_SHA256', main)
        self.assertEqual(sorted(mvo.prep_sha256()), ["prep_header_sha256", "prep_sha256"])
        self.assertTrue(set(mvo.prep_sha256()) <= set(live_feed.SOURCES))


if __name__ == "__main__":
    unittest.main()
