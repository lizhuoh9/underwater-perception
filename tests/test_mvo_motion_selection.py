"""Reproducible known-geometry motion-selection diagnostics; not an image-tracking test."""

import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from uwslam.core import CFLAGS
from uwslam.paths import CORE_DIR


SOURCE = Path(__file__).resolve().parent / "c" / "mvo_motion_selection_check.c"
UNITS = [CORE_DIR / name for name in ("mono_vo.c", "mono_vo_numerics.c", "mono_vo_homography.c",
                                      "mono_vo_essential.c")]


def build(folder: Path) -> Path:
    executable = folder / "mvo_motion_selection_check.exe"
    subprocess.run(["gcc", *CFLAGS, *(["-static"] if sys.platform == "win32" else []), f"-I{CORE_DIR}",
                    str(SOURCE), *map(str, UNITS), "-lm", "-o", str(executable)], check=True)
    return executable


def samples(executable: Path, tag: str, pairs: int, sigma_x: float, sigma_y: float, outliers: float,
            centre_x: float, centre_z: float, yaw: float, first_seed: int, trials: int) -> list[dict]:
    command = [str(executable), *map(str, (pairs, sigma_x, sigma_y, outliers, centre_x, centre_z,
                                          yaw, first_seed, trials, tag))]
    text = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    return list(csv.DictReader(text.splitlines()))


PROFILES = (
    ("still_clean_25", 25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ("still_noise_25", 25, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ("still_anisotropic_25", 25, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ("still_noise_outliers_25", 25, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0),
    ("turn_outliers_25", 25, 1.0, 1.0, 0.4, 0.0, 0.0, 5.0),
    ("translate_clean_25", 25, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0),
    ("translate_noise_25", 25, 0.5, 0.5, 0.2, 0.3, 0.0, 0.0),
    ("translate_noise_80", 80, 0.5, 0.5, 0.2, 0.3, 0.0, 0.0),
    ("translate_small_80", 80, 0.5, 0.5, 0.0, 0.03, 0.0, 0.0),
    ("forward_noise_80", 80, 0.5, 0.5, 0.1, 0.0, 0.3, 0.0),
    ("turn_translate_80", 80, 0.5, 0.5, 0.1, 0.3, 0.0, 5.0),
    ("translate_noise_150", 150, 0.5, 0.5, 0.2, 0.3, 0.0, 0.0),
)


def summarize(rows: list[dict]) -> dict:
    translation = float(rows[0]["centre_x"]) != 0.0 or float(rows[0]["centre_z"]) != 0.0
    weak = [row for row in rows if int(row["homography"]) < 15]
    essential = [row for row in rows if int(row["kind"]) == 2]
    correct = [row for row in essential if float(row["direction_cos"]) >= 0.9] if translation else []
    weak_correct = [row for row in weak if int(row["kind"]) == 2 and float(row["direction_cos"]) >= 0.9] if translation else []
    return {"tag": rows[0]["tag"], "trials": len(rows), "weak": len(weak), "essential": len(essential),
            "correct_direction": len(correct), "weak_correct_direction": len(weak_correct),
            "false_translation": 0 if translation else len(essential),
            "mean_cpu_ms": sum(float(row["cpu_ms"]) for row in rows) / len(rows)}


@unittest.skipUnless(shutil.which("gcc"), "gcc is needed")
class MvoMotionSelectionTests(unittest.TestCase):
    def test_injected_geometry_is_deterministic_and_stationary_has_no_translation(self):
        with tempfile.TemporaryDirectory() as folder:
            executable = build(Path(folder))
            args = ("stationary", 25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 3)
            first = samples(executable, *args)
            second = samples(executable, *args)
        for a, b in zip(first, second):
            self.assertEqual({k: v for k, v in a.items() if k != "cpu_ms"},
                             {k: v for k, v in b.items() if k != "cpu_ms"})
            self.assertNotEqual(int(a["kind"]), 2)
            self.assertEqual([float(a[k]) for k in ("estimate_x", "estimate_y", "estimate_z")], [0.0, 0.0, 0.0])
            self.assertEqual(int(a["rotation_support_2px"]), 25)
            self.assertLess(float(a["rotation_median_px"]), 0.001)

    def test_weak_translation_and_noisy_rotation_are_distinguished(self):
        with tempfile.TemporaryDirectory() as folder:
            executable = build(Path(folder))
            clean = samples(executable, "weak", 25, 0, 0, 0, 0.3, 0, 0, 7001, 20)
            self.assertTrue(any(int(row["homography"]) < 15 for row in clean))
            self.assertTrue(all(int(row["kind"]) == 2 and float(row["direction_cos"]) > 0.99 for row in clean))
            for count, noise, outliers, yaw in ((25, 1.0, 0.0, 0.0), (25, 2.0, 0.2, 5.0),
                                                  (50, 2.0, 0.0, 5.0), (80, 2.0, 0.2, 0.0),
                                                  (150, 3.0, 0.4, 15.0)):
                with self.subTest(count=count, noise=noise, outliers=outliers, yaw=yaw):
                    rows = samples(executable, "rotation", count, noise, noise / 2, outliers,
                                   0, 0, yaw, 50001, 20)
                    self.assertTrue(all(int(row["kind"]) != 2 for row in rows), rows)

    def test_large_translation_does_not_require_rotation_support(self):
        with tempfile.TemporaryDirectory() as folder:
            executable = build(Path(folder))
            rows = samples(executable, "large", 80, 0, 0, 0, 2.0, 0, 0, 7001, 20)
        self.assertTrue(all(int(row["homography"]) < 15 for row in rows))
        self.assertGreaterEqual(sum(int(row["kind"]) == 2 and float(row["direction_cos"]) > 0.99 for row in rows), 18)
        self.assertTrue(any(0 <= int(row["rotation_model_support"]) < 3 and int(row["kind"]) == 2 for row in rows), rows)

    def test_small_rotation_subset_is_not_a_frame_wide_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            executable = build(Path(folder))
            rows = samples(executable, "minority", 150, 1.0, 0.5, 0.85, 0, 0, 5, 55001, 20)
        self.assertTrue(all(int(row["kind"]) == 0 for row in rows), rows)

    def test_high_noise_counterexamples_do_not_invent_translation(self):
        with tempfile.TemporaryDirectory() as folder:
            executable = build(Path(folder))
            for noise, outliers, yaw, seed in ((2.0, 0.2, 0.0, 250021), (3.0, 0.0, 0.0, 250021),
                                                (3.0, 0.0, 5.0, 250008), (3.0, 0.0, 0.0, 500446),
                                                (3.0, 0.0, 5.0, 500443), (3.0, 0.0, 0.0, 720117),
                                                (5.0, 0.0, 0.0, 900026), (3.0, 0.0, 0.0, 11100008)):
                row = samples(executable, "counterexample", 25, noise, noise / 2, outliers,
                              0, 0, yaw, seed, 1)[0]
                self.assertNotEqual(int(row["kind"]), 2, row)
            for count in (150, 250):
                rows = samples(executable, "extreme", count, 8.0, 4.0, 0.2, 0, 0, 5, 840001, 30)
                self.assertTrue(all(int(row["kind"]) != 2 for row in rows), rows)

    def test_noisy_weak_homography_can_recover_translation(self):
        with tempfile.TemporaryDirectory() as folder:
            executable = build(Path(folder))
            for seed in (720033, 720036, 720038):
                row = samples(executable, "noisy_weak", 25, 0.5, 0.5, 0.2, 0.3, 0, 0, seed, 1)[0]
                self.assertLess(int(row["homography"]), 15, row)
                self.assertEqual(int(row["kind"]), 2, row)
                self.assertGreater(float(row["direction_cos"]), 0.98, row)

    def test_narrow_field_can_fit_a_rotation_for_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "narrow_field_check.exe"
            source = SOURCE.with_name("mvo_narrow_field_check.c")
            subprocess.run(["gcc", *CFLAGS, *(["-static"] if sys.platform == "win32" else []), f"-I{CORE_DIR}",
                            str(source), *map(str, UNITS), "-o", str(binary)], check=True)
            output = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
        values = {key: float(value) for key, value in (part.split("=") for part in output.split())}
        self.assertLess(values["h"], 15, output)
        self.assertGreaterEqual(values["rotation_support"], 0, output)
        self.assertEqual(values["kind"], 2, output)
        self.assertGreater(values["x"], 0.99, output)
        self.assertLess(abs(values["y"]), 0.01, output)
        self.assertLess(abs(values["z"]), 0.01, output)

    def test_noisy_narrow_field_does_not_invent_translation(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "narrow_noise_check.exe"
            source = SOURCE.with_name("mvo_narrow_noise_check.c")
            subprocess.run(["gcc", *CFLAGS, *(["-static"] if sys.platform == "win32" else []), f"-I{CORE_DIR}",
                            str(source), *map(str, UNITS), "-o", str(binary)], check=True)
            for noise, outliers in ((0.5, 0.0), (1.0, 0.2)):
                output = subprocess.run([str(binary), str(noise), str(outliers), "1", "200"],
                                        check=True, capture_output=True, text=True).stdout
                result = {key: int(value) for key, value in
                          (part.split("=") for part in output.splitlines()[-1].split())}
                self.assertEqual(result["false"], 0, output)

    def test_rotation_discards_outlier_depths(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "rotation_points_check.exe"
            source = SOURCE.with_name("mvo_rotation_points_check.c")
            subprocess.run(["gcc", *CFLAGS, *(["-static"] if sys.platform == "win32" else []), f"-I{CORE_DIR}",
                            str(source), *map(str, UNITS), "-o", str(binary)], check=True)
            output = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
        lines = [{key: float(value) for key, value in (part.split("=") for part in line.split())}
                 for line in output.splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertLess(lines[0]["parallax"], 1)
        self.assertGreater(lines[1]["parallax"], 1)
        for row in lines:
            self.assertEqual(row["kind"], 1, output)
            self.assertEqual(row["outlier_points"], 0, output)
            self.assertEqual(row["kept"], 30, output)
            self.assertLess(row["max_error"], 0.01, output)


def sweep(first_seed: int = 1, trials: int = 30) -> None:
    with tempfile.TemporaryDirectory() as folder:
        executable = build(Path(folder))
        for profile in PROFILES:
            print(summarize(samples(executable, *profile, first_seed, trials)), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        sweep(int(sys.argv[2]) if len(sys.argv) > 2 else 1, int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    else:
        unittest.main()
