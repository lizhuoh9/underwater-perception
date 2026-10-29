"""Recorded names resolve to this layout, and every source or model a committed record names is in the repository."""

import json
import unittest

from uwseg.paths import DATA_DIR, EMBEDDED_DIR, PACKAGE_DIR, RESULTS_DIR, resolve


def recorded_names(key: str) -> set[str]:
    """Names under `key` (a name -> SHA-256 mapping) in every committed JSON record."""
    names = set()
    for path in RESULTS_DIR.rglob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict) and isinstance(record.get(key), dict):
            names |= set(record[key])
    return names


class ResolveTests(unittest.TestCase):
    def test_each_kind_of_name_maps_to_its_folder(self):
        self.assertEqual(resolve("data/manifest.json"), DATA_DIR / "manifest.json")
        self.assertEqual(resolve("embedded/pixel_forest.c"), EMBEDDED_DIR / "pixel_forest.c")
        self.assertEqual(resolve("udcp.py"), PACKAGE_DIR / "udcp.py")
        self.assertEqual(resolve("forest_lowfp/LOCKED.json"), RESULTS_DIR / "forest_lowfp" / "LOCKED.json")

    def test_every_recorded_source_is_part_of_the_repository(self):
        names = recorded_names("source_sha256") | recorded_names("source_code_sha256")
        self.assertIn("cnn_lite.py", names)
        self.assertEqual(sorted(name for name in names if not resolve(name).is_file()), [])

    def test_every_model_the_calibration_recheck_scored_is_committed(self):
        summary = json.loads((RESULTS_DIR / "calibration_recheck" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(path for path in summary["model_sha256"] if not resolve(path).is_file()), [])


if __name__ == "__main__":
    unittest.main()
