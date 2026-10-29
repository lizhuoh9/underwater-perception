"""Near-duplicate screening: hash copies, screen boundaries, review gating, and the frozen-split audit."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from uwseg import compare_dedup
from uwseg.near_duplicates import (
    AUTO_NEAR_DUPLICATE_BITS,
    CALIBRATION_COLOUR_GATE,
    HASH_BITS,
    SCREEN_BITS,
    TEST_COLOUR_GATE,
    Signature,
    decide,
    image_signature,
    leaked_ids,
    load_review,
    pair_key,
    screen_pairs,
)
from uwseg.paths import DATA_DIR, REPO_ROOT, resolve
from uwseg.prepare_dedup_split import (
    CODE,
    GROUPS,
    INPUT_MANIFESTS,
    audit,
    build_manifest,
    encode,
    write_manifest,
    write_review_sheets,
)

DEDUP = DATA_DIR / "dedup_manifest.json"
REVIEW = DATA_DIR / "near_duplicate_review.json"
# The frozen audit is committed; checking it against the images needs the prepared data.
DATA_PREPARED = all((DATA_DIR / name).exists() for name in INPUT_MANIFESTS)
REPORTED = ("n_l_26_", "f_r_403_", "n_l_102_", "n_l_79_", "f_r_1892_", "n_l_100_", "d_r_384_")


def scene(seed, tint, size=(640, 480)):
    """Smooth random structure over a vertical light falloff, like an object in open water."""
    coarse = np.random.default_rng(seed).integers(0, 256, (12, 16), dtype=np.uint8)
    field = np.asarray(Image.fromarray(coarse).resize(size, Image.Resampling.BICUBIC), dtype=np.float32)
    falloff = np.linspace(1.0, 0.55, size[1], dtype=np.float32)[:, None]
    rgb = np.stack([(0.35 * field + 0.65 * channel) * falloff for channel in tint], axis=-1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def signs(flipped=0, value=1.0):
    values = np.full(HASH_BITS, value, dtype=np.float32)
    values[:flipped] = -value
    return values


def colour(overlap):
    """Histogram whose intersection with all-mass-in-bin-0 is exactly `overlap`."""
    values = np.zeros(512, dtype=np.float32)
    values[0], values[1] = overlap, 1 - overlap
    return values


def signature(hash_signs, mirror_signs=None, overlap=1.0):
    return Signature(hash_signs, signs(value=-1.0) if mirror_signs is None else mirror_signs, colour(overlap))


def row(sample_id, method, tp, fp, fn, tn, miou):
    return {"id": sample_id, "method": method, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "miou": miou}


class NearDuplicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        folder = Path(cls.folder.name)
        base = scene(1, (40, 110, 170))
        base.save(folder / "base.jpg", quality=95)
        base.resize((512, 384), Image.Resampling.BILINEAR).save(folder / "resized.jpg", quality=70)
        base.resize((640, 360), Image.Resampling.BILINEAR).save(folder / "stretched.jpg", quality=80)
        base.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(folder / "mirrored.jpg", quality=90)
        scene(2, (170, 120, 50)).save(folder / "unrelated.jpg", quality=90)
        cls.signatures = {path.stem: image_signature(path) for path in folder.glob("*.jpg")}

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_reencoded_resized_and_mirrored_copies_are_hash_near_duplicates(self):
        others = {name: value for name, value in self.signatures.items() if name != "base"}
        found = {pair["b"]: pair for pair in screen_pairs({"base": self.signatures["base"]}, others, TEST_COLOUR_GATE)}
        for name in ("resized", "stretched", "mirrored"):
            self.assertLessEqual(found[name]["hash_bits"], AUTO_NEAR_DUPLICATE_BITS)
        self.assertTrue(found["mirrored"]["mirrored"])
        self.assertFalse(found["resized"]["mirrored"])
        self.assertNotIn("unrelated", found)
        decided, missing = decide(list(found.values()), {})
        self.assertEqual(missing, [])
        self.assertEqual({(pair["decision"], pair["decided_by"]) for pair in decided}, {("near_duplicate", "hash")})

    def test_signature_arrays_are_read_only(self):
        signature_ = self.signatures["base"]
        for values in (signature_.hash_signs, signature_.mirror_signs, signature_.colour):
            self.assertFalse(values.flags.writeable)
        self.assertAlmostEqual(float(signature_.colour.sum()), 1.0, places=5)

    def test_hash_distance_counts_flipped_signs_and_mirror_only_matches(self):
        base = {"a": signature(signs(), overlap=1.0)}
        found = screen_pairs(base, {"k37": signature(signs(37), overlap=0.0)}, TEST_COLOUR_GATE)
        self.assertEqual([(pair["b"], pair["hash_bits"], pair["mirrored"]) for pair in found], [("k37", 37, False)])
        mirror_only = {"a": signature(signs(value=-1.0), mirror_signs=signs(5), overlap=1.0)}
        found = screen_pairs(mirror_only, {"b": signature(signs(), overlap=0.0)}, TEST_COLOUR_GATE)
        self.assertEqual([(pair["b"], pair["hash_bits"], pair["mirrored"]) for pair in found], [("b", 5, True)])

    def test_screen_boundaries_for_hash_and_colour(self):
        base = {"a": signature(signs(), overlap=1.0)}
        candidates = {
            "bits260": signature(signs(SCREEN_BITS), overlap=0.0),
            "bits261": signature(signs(SCREEN_BITS + 1), overlap=0.0),
            "far_colour_in": signature(signs(500), overlap=TEST_COLOUR_GATE + 0.01),
            "far_colour_out": signature(signs(500), overlap=TEST_COLOUR_GATE - 0.01),
            "far_calibration_out": signature(signs(500), overlap=CALIBRATION_COLOUR_GATE - 0.01),
        }
        test_screen = {pair["b"] for pair in screen_pairs(base, candidates, TEST_COLOUR_GATE)}
        self.assertEqual(test_screen, {"bits260", "far_colour_in", "far_calibration_out"})
        calibration_screen = {pair["b"] for pair in screen_pairs(base, candidates, CALIBRATION_COLOUR_GATE)}
        self.assertEqual(calibration_screen, {"bits260"})

    def test_band_pairs_need_review_and_reviews_override_the_hash(self):
        band = {
            "a": "t1",
            "b": "r1",
            "hash_bits": AUTO_NEAR_DUPLICATE_BITS + 1,
            "mirrored": False,
            "colour_intersection": 0.9,
        }
        copy = {
            "a": "t2",
            "b": "r2",
            "hash_bits": AUTO_NEAR_DUPLICATE_BITS,
            "mirrored": False,
            "colour_intersection": 0.99,
        }
        decided, missing = decide([band, copy], {})
        self.assertEqual(missing, [band])
        self.assertEqual([pair["a"] for pair in decided], ["t2"])
        reviews = {
            pair_key("t1", "r1"): {"decision": "same_scene", "note": "same clip"},
            pair_key("r2", "t2"): {"decision": "different", "note": "flat frames"},
        }
        decided, missing = decide([band, copy], reviews)
        self.assertEqual(missing, [])
        self.assertEqual(
            [
                (pair["decision"], pair["decided_by"], pair["note"], pair.get("overrides_hash", False))
                for pair in decided
            ],
            [("same_scene", "review", "same clip", False), ("different", "review", "flat frames", True)],
        )
        with self.assertRaisesRegex(ValueError, "needs a note"):
            decide([copy], {pair_key("t2", "r2"): {"decision": "different"}})

    def test_review_file_rejects_unknown_and_duplicate_decisions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "review.json"
            path.write_text(json.dumps({"decisions": [{"a": "x", "b": "y", "decision": "maybe"}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown review decision"):
                load_review(path)
            duplicate = [{"a": "x", "b": "y", "decision": "different"}, {"a": "y", "b": "x", "decision": "same_scene"}]
            path.write_text(json.dumps({"decisions": duplicate}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate review decision"):
                load_review(path)
            path.write_text(json.dumps({"decisions": duplicate[:1]}), encoding="utf-8")
            self.assertEqual(load_review(path)[pair_key("y", "x")]["decision"], "different")

    def test_leak_tiers_keep_the_strongest_decision_per_image(self):
        def pair(a, b, decision):
            return {
                "a": a,
                "b": b,
                "hash_bits": 200,
                "mirrored": False,
                "colour_intersection": 0.8,
                "decision": decision,
                "decided_by": "review",
                "note": "",
            }

        decided = [
            pair("t1", "r1", "same_scene"),
            pair("t1", "r2", "near_duplicate"),
            pair("t2", "r3", "different"),
            pair("c1", "t3", "same_scene"),
        ]
        leaks = leaked_ids(decided, {"t1", "t2", "t3"})
        self.assertEqual(set(leaks), {"t1", "t3"})
        self.assertEqual(leaks["t1"]["tier"], "near_duplicate")
        self.assertEqual([n["id"] for n in leaks["t1"]["neighbours"]], ["r2", "r1"])
        self.assertEqual(
            leaks["t3"],
            {
                "tier": "same_scene",
                "neighbours": [
                    {
                        "id": "c1",
                        "decision": "same_scene",
                        "decided_by": "review",
                        "hash_bits": 200,
                        "mirrored": False,
                        "colour_intersection": 0.8,
                        "note": "",
                    }
                ],
            },
        )

    def test_manifest_attributes_leaks_by_split_not_pair_orientation(self):
        def pair(a, b, decision, bits=300, overlap=0.7):
            return {
                "a": a,
                "b": b,
                "hash_bits": bits,
                "mirrored": False,
                "colour_intersection": overlap,
                "decision": decision,
                "decided_by": "review",
                "note": "",
            }

        splits = {
            "test": [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}],
            "calibration": [{"id": "c1"}],
            "train": [{"id": "r1"}],
        }
        decided = [
            pair("c1", "t1", "same_scene"),
            pair("t2", "r1", "near_duplicate", bits=40),
            pair("c1", "r1", "same_scene", overlap=0.9),
        ]
        manifest = build_manifest(splits, {}, {"r1"}, {name: [] for name in GROUPS}, decided)
        self.assertEqual(
            {i: leak["tier"] for i, leak in manifest["test_leaks"].items()},
            {"t1": "same_scene", "t2": "near_duplicate"},
        )
        self.assertEqual(manifest["test_leaks"]["t1"]["neighbours"][0]["split"], "calibration")
        self.assertTrue(manifest["test_leaks"]["t2"]["neighbours"][0]["in_first_512_training"])
        self.assertEqual([n["id"] for n in manifest["calibration_leaks"]["c1"]["neighbours"]], ["r1"])
        self.assertEqual(
            manifest["test_subsets"],
            {"all": ["t1", "t2", "t3"], "without_near_duplicates": ["t1", "t3"], "without_same_scene": ["t3"]},
        )
        self.assertEqual(len(manifest["screened_pairs"]["test-calibration"]), 1)
        self.assertEqual(manifest["counts"]["test_leak_bands"]["hash"], 1)
        self.assertEqual(manifest["counts"]["test_leak_bands"]["0.70-0.75"], 1)

    def test_goal_allows_exactly_two_points_and_needs_lower_fpr(self):
        before = row("x", "before", 95, 10, 5, 90, 0.5)
        equal_fpr = paired_goal([before, row("x", "after", 93, 10, 7, 90, 0.6)])
        lower_fpr = paired_goal([before, row("x", "after", 93, 9, 7, 91, 0.6)])
        self.assertAlmostEqual(lower_fpr["fnr_percentage_points"], 2.0)
        self.assertEqual((equal_fpr["goal_passed"], lower_fpr["goal_passed"]), (False, True))
        self.assertEqual(lower_fpr["bootstrap"]["goal_probability"], 1.0)
        self.assertEqual(lower_fpr["paired_images"], {"improved": 1, "regressed": 0, "tied": 0})

    def test_bootstrap_interval_brackets_the_pooled_change(self):
        rows = [
            row("x", "before", 90, 10, 10, 90, 0.5),
            row("x", "after", 85, 5, 15, 95, 0.5),
            row("y", "before", 50, 40, 5, 60, 0.4),
            row("y", "after", 49, 30, 6, 70, 0.6),
        ]
        result = compare_dedup.paired(rows, ["x", "y"], "before", "after")
        low, high = result["bootstrap"]["fnr_percentage_points_95ci"]
        self.assertLessEqual(low, result["fnr_percentage_points"])
        self.assertLessEqual(result["fnr_percentage_points"], high)
        self.assertEqual(result, compare_dedup.paired(rows, ["x", "y"], "before", "after"))

    def test_manifest_writer_never_replaces_different_content(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "dedup.json"
            write_manifest(path, b"first")
            write_manifest(path, b"first")
            with self.assertRaisesRegex(ValueError, "nonmatching"):
                write_manifest(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_review_sheets_are_paged_and_never_overwritten(self):
        folder = Path(self.folder.name)
        rows = {name: {"image": (folder / f"{name}.jpg").relative_to(REPO_ROOT).as_posix()} for name in self.signatures}
        pairs = [
            {"a": "base", "b": name, "hash_bits": 300, "colour_intersection": 0.8}
            for name in ("resized", "stretched", "unrelated")
        ]
        written = write_review_sheets(pairs, rows, folder / "pending", per_sheet=2)
        self.assertEqual([path.name for path in written], ["pending_00.jpg", "pending_01.jpg"])
        with Image.open(written[0]) as first, Image.open(written[1]) as second:
            self.assertEqual((first.height, second.height), (400, 200))
        with self.assertRaises(FileExistsError):
            write_review_sheets(pairs, rows, folder / "pending.png", per_sheet=2)

    def test_dedup_evaluation_refuses_a_populated_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            evidence = Path(folder) / "summary.json"
            evidence.write_text("previous evidence", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                compare_dedup.run(Path(folder), DEDUP)
            self.assertEqual(evidence.read_text(encoding="utf-8"), "previous evidence")


def paired_goal(rows):
    return compare_dedup.paired(rows, ["x"], "before", "after")


@unittest.skipUnless(DEDUP.exists() and DATA_PREPARED, "Prepare the data first (python -m uwseg.prepare_data ...)")
class FrozenSplitAuditTests(unittest.TestCase):
    """The committed audit's code_sha256 records the screening scripts it was made with, before they became this
    package; everything else in it must recompute from the images and the reviews."""

    def test_audit_matches_inputs_and_contains_the_reported_leaks(self):
        manifest = json.loads(DEDUP.read_text(encoding="utf-8"))
        for name, digest in manifest["inputs"].items():
            self.assertEqual(hashlib.sha256(resolve(name).read_bytes()).hexdigest(), digest)
        self.assertEqual(set(manifest["code_sha256"]), set(CODE))
        for sample_id in REPORTED:
            self.assertEqual(manifest["test_leaks"][sample_id]["tier"], "near_duplicate")
        subsets = {name: set(ids) for name, ids in manifest["test_subsets"].items()}
        self.assertEqual(len(subsets["all"]), 110)
        self.assertTrue(subsets["without_same_scene"] < subsets["without_near_duplicates"] < subsets["all"])
        self.assertEqual(subsets["all"] - subsets["without_same_scene"], set(manifest["test_leaks"]))

    def test_manifest_recomputes_from_images_and_reviews(self):
        recomputed = audit(REVIEW)
        recomputed["code_sha256"] = json.loads(DEDUP.read_text(encoding="utf-8"))["code_sha256"]
        self.assertEqual(encode(recomputed), DEDUP.read_bytes())

    def test_evaluation_rejects_edited_subsets_or_leaks(self):
        # The audit as this package's screening code records it, which compare_dedup accepts.
        original = json.loads(DEDUP.read_text(encoding="utf-8"))
        original["code_sha256"] = {name: hashlib.sha256(resolve(name).read_bytes()).hexdigest() for name in CODE}
        edits = {
            "subsets disagree": lambda m: m["test_subsets"]["without_same_scene"].append(next(iter(m["test_leaks"]))),
            "disagree with its screened pairs": lambda m: m["test_leaks"].pop(next(iter(m["test_leaks"]))),
            "screening code hashes": lambda m: m.pop("code_sha256"),
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            current = Path(folder) / "current_manifest.json"
            current.write_text(json.dumps(original), encoding="utf-8")
            _, subsets = compare_dedup.load_audit(current)
            self.assertEqual(len(subsets["without_reported_7"]), 103)
            for message, edit in edits.items():
                manifest = json.loads(json.dumps(original))
                edit(manifest)
                tampered = Path(folder) / "dedup_manifest.json"
                tampered.write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    compare_dedup.load_audit(tampered)


if __name__ == "__main__":
    unittest.main()
