"""Audit decoded-pixel near duplicates between the frozen SUIM splits.

Run after prepare_data.py, prepare_forest_training.py and prepare_lowfp_training.py.
Existing manifests and extracted images are only read and hash-verified. This
stage writes one new manifest listing TEST images with a near-duplicate or
same-scene neighbour in calibration or training, and the TEST subsets without
them. Screened pairs that the hash cannot decide must have a recorded decision
in the review file; ``--review-sheet`` renders the pending pairs for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from uwseg.near_duplicates import (
    AUTO_NEAR_DUPLICATE_BITS,
    CALIBRATION_COLOUR_GATE,
    COLOUR_LEVELS,
    COLOUR_SIZE,
    DECISIONS,
    HASH_BITS,
    SCREEN_BITS,
    TEST_COLOUR_GATE,
    decide,
    image_signature,
    leaked_ids,
    load_review,
    pair_key,
    screen_pairs,
)
from uwseg.paths import resolve
from uwseg.prepare_data import ARCHIVE_SHA256, WORKSPACE, file_sha256

DATA_ROOT = WORKSPACE / "data"
INPUT_MANIFESTS = ("manifest.json", "forest_training_manifest.json", "lowfp_training_manifest.json")
CODE = ("near_duplicates.py", "prepare_dedup_split.py")
# Pair group -> (first split, second split, colour gate).
GROUPS = {
    "test-calibration": ("test", "calibration", TEST_COLOUR_GATE),
    "test-train": ("test", "train", TEST_COLOUR_GATE),
    "calibration-train": ("calibration", "train", CALIBRATION_COLOUR_GATE),
}
COLOUR_BANDS = ((0.75, "0.75+"), (0.70, "0.70-0.75"), (0.65, "0.65-0.70"), (0.60, "0.60-0.65"))
DECISION_MEANINGS = {
    "near_duplicate": "Same photo (re-encoded, resized, recropped or mirrored) or an adjacent video frame.",
    "same_scene": "Same clip or encounter at a different moment: same animal/vehicle/diver, site and light.",
    "different": "Different photo; similar layout or water colour only.",
}
LIMITS = (
    "Every screened pair is decided, but the screen is a cut-off, not a complete search: counts.test_leak_bands "
    "shows how many leaked TEST images each lower colour band added, and same-clip frames below the gate very "
    "likely remain. SUIM has no video or session IDs, so scene-level independence is not established."
)


def load_inputs() -> tuple[dict, dict, set]:
    raw = {name: (DATA_ROOT / name).read_bytes() for name in INPUT_MANIFESTS}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    original, first_train, train = (json.loads(raw[name]) for name in INPUT_MANIFESTS)
    if any(manifest["source"]["sha256"] != ARCHIVE_SHA256 for manifest in (original, first_train, train)):
        raise ValueError("Input manifests do not describe the pinned SUIM archive")
    audit = train["audit"]
    if (audit["original_manifest_sha256"], audit["previous_training_manifest_sha256"]) != (
        hashes["manifest.json"],
        hashes["forest_training_manifest.json"],
    ):
        raise ValueError("Training manifests were prepared against different inputs")
    splits = {
        "test": [row for row in original["samples"] if row["split"] == "test"],
        "calibration": [row for row in original["samples"] if row["split"] == "calibration"],
        "train": train["samples"],
    }
    if {name: len(rows) for name, rows in splits.items()} != {"test": 110, "calibration": 80, "train": 1395}:
        raise ValueError("Expected the frozen 110/80/1395 test/calibration/train splits")
    ids = [row["id"] for rows in splits.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Sample IDs must be unique across splits")
    first_512 = {row["id"] for row in first_train["samples"]}
    return splits, {f"data/{name}": digest for name, digest in hashes.items()}, first_512


def signatures(rows: list[dict]) -> dict:
    result = {}
    for row in rows:
        path = WORKSPACE / row["image"]
        if file_sha256(path) != row["image_sha256"]:
            raise ValueError(f"Local image differs from its manifest hash: {path}")
        result[row["id"]] = image_signature(path)
    return result


def write_review_sheets(pairs: list[dict], rows: dict, path: Path, per_sheet: int = 40) -> list[Path]:
    """JPEG pages of side-by-side pairs; never replaces an existing page."""
    written = []
    for page, start in enumerate(range(0, len(pairs), per_sheet)):
        chunk = pairs[start : start + per_sheet]
        sheet = Image.new("RGB", (500, 200 * len(chunk)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, pair in enumerate(chunk):
            for column, key in enumerate(("a", "b")):
                with Image.open(WORKSPACE / rows[pair[key]]["image"]) as image:
                    sheet.paste(ImageOps.contain(image.convert("RGB"), (240, 180)), (column * 250, index * 200 + 18))
            draw.text(
                (4, index * 200 + 2),
                f"{pair['a']} ~ {pair['b']}  bits={pair['hash_bits']} colour={pair['colour_intersection']:.2f}",
                fill="black",
            )
        written.append(path.with_name(f"{path.stem}_{page:02d}.jpg"))
        with written[-1].open("xb") as target:
            sheet.save(target, format="JPEG", quality=85)
    return written


def write_manifest(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite nonmatching manifest: {path}")
        return
    with path.open("xb") as target:
        target.write(payload)


def describe(leaks: dict, split_of: dict, first_512: set) -> dict:
    return {
        own: {
            "tier": leak["tier"],
            "neighbours": [
                {**row, "split": split_of[row["id"]], "in_first_512_training": row["id"] in first_512}
                for row in leak["neighbours"]
            ],
        }
        for own, leak in sorted(leaks.items())
    }


def leak_bands(leaks: dict) -> dict:
    """Leaked images by the screen that first finds them: hash alone, else their best colour band."""
    bands = {"hash": 0, **{label: 0 for _, label in COLOUR_BANDS}, "below_gate": 0}
    for leak in leaks.values():
        if any(row["hash_bits"] <= SCREEN_BITS for row in leak["neighbours"]):
            bands["hash"] += 1
            continue
        best = max(row["colour_intersection"] for row in leak["neighbours"])
        bands[next((label for low, label in COLOUR_BANDS if best >= low), "below_gate")] += 1
    return bands


def build_manifest(splits: dict, inputs: dict, first_512: set, screened: dict, decided: list[dict]) -> dict:
    split_of = {row["id"]: name for name, rows in splits.items() for row in rows}
    test_ids = {row["id"] for row in splits["test"]}
    calibration_ids = {row["id"] for row in splits["calibration"]}
    # Split membership, not pair orientation, selects which pairs count for each audit.
    test_leaks = leaked_ids(decided, test_ids)
    calibration_leaks = leaked_ids(
        [pair for pair in decided if "test" not in (split_of[pair["a"]], split_of[pair["b"]])], calibration_ids
    )
    ordered = [row["id"] for row in splits["test"]]
    subsets = {
        "all": ordered,
        "without_near_duplicates": [i for i in ordered if test_leaks.get(i, {}).get("tier") != "near_duplicate"],
        "without_same_scene": [i for i in ordered if i not in test_leaks],
    }

    def tiers(leaks):
        return {tier: sum(leak["tier"] == tier for leak in leaks.values()) for tier in ("near_duplicate", "same_scene")}

    return {
        "schema_version": 2,
        "source": {"archive_sha256": ARCHIVE_SHA256},
        "inputs": inputs,
        "code_sha256": {name: file_sha256(resolve(name)) for name in CODE},
        "method": {
            "signature": f"{HASH_BITS}-bit grayscale difference hash (bilinear 33x32, horizontal gradient signs), "
            f"also compared mirrored; {COLOUR_LEVELS}-level-per-channel RGB histogram of a bilinear "
            f"{COLOUR_SIZE[0]}x{COLOUR_SIZE[1]} thumbnail",
            "screen": {
                name: f"hash_bits <= {SCREEN_BITS} or colour_intersection >= {gate}"
                for name, (_, _, gate) in GROUPS.items()
            },
            "automatic_rule": f"hash_bits <= {AUTO_NEAR_DUPLICATE_BITS} is near_duplicate unless a noted review overrides it",
            "pairs_screened": "TEST x calibration, TEST x all 1395 training images, calibration x training",
            "decision_meanings": DECISION_MEANINGS,
            "limits": LIMITS,
        },
        "counts": {
            "test": len(ordered),
            "calibration": len(calibration_ids),
            "train": len(splits["train"]),
            "screened_pairs": {name: len(pairs) for name, pairs in screened.items()},
            "decided_by": {
                source: sum(pair["decided_by"] == source for pair in decided) for source in ("hash", "review")
            },
            "hash_overrides": sum(pair.get("overrides_hash", False) for pair in decided),
            "decisions": {name: sum(pair["decision"] == name for pair in decided) for name in DECISIONS},
            "test_leak_tiers": tiers(test_leaks),
            "calibration_leak_tiers": tiers(calibration_leaks),
            "test_leak_bands": leak_bands(test_leaks),
            "test_subset_sizes": {name: len(ids) for name, ids in subsets.items()},
        },
        "test_leaks": describe(test_leaks, split_of, first_512),
        "calibration_leaks": describe(calibration_leaks, split_of, first_512),
        "test_subsets": subsets,
        "screened_pairs": {
            name: sorted(
                (pair for pair in decided if {split_of[pair["a"]], split_of[pair["b"]]} == {first, second}),
                key=lambda pair: (pair["a"], pair["hash_bits"], pair["b"]),
            )
            for name, (first, second, _) in GROUPS.items()
        },
    }


def audit(review_path: Path, review_sheet: Path | None = None) -> dict:
    splits, inputs, first_512 = load_inputs()
    signed = {name: signatures(rows) for name, rows in splits.items()}
    screened = {
        name: screen_pairs(signed[first], signed[second], gate) for name, (first, second, gate) in GROUPS.items()
    }
    reviews = load_review(review_path) if review_path.exists() else {}
    decided, missing = decide([pair for pairs in screened.values() for pair in pairs], reviews)
    if missing:
        sheets = []
        if review_sheet is not None:
            rows = {row["id"]: row for split in splits.values() for row in split}
            sheets = write_review_sheets(missing, rows, review_sheet)
        listed = ", ".join(f"{pair['a']}~{pair['b']}" for pair in missing[:12])
        pages = f"; review sheets: {sheets[0].name} .. {sheets[-1].name}" if sheets else ""
        raise ValueError(f"{len(missing)} screened pairs need a review decision in {review_path}: {listed}{pages}")
    unknown = set(reviews) - {pair_key(pair["a"], pair["b"]) for pair in decided}
    if unknown:
        raise ValueError(f"Review decisions outside the screened pairs: {sorted(unknown)[:12]}")
    inputs[review_path.relative_to(WORKSPACE).as_posix()] = file_sha256(review_path)
    return build_manifest(splits, inputs, first_512, screened, decided)


def encode(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def prepare(review_path: Path, output_path: Path, review_sheet: Path | None) -> dict:
    if not output_path.resolve().is_relative_to(DATA_ROOT.resolve()):
        raise ValueError("The audit manifest must be written inside the data directory")
    manifest = audit(review_path, review_sheet)
    write_manifest(output_path, encode(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DATA_ROOT / "near_duplicate_review.json")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "dedup_manifest.json")
    parser.add_argument(
        "--review-sheet", type=Path, help="write pending pairs as numbered JPEG pages named after this path"
    )
    args = parser.parse_args()
    try:
        manifest = prepare(args.review.resolve(), args.output.resolve(), args.review_sheet)
    except (OSError, ValueError, KeyError) as error:
        print(f"Near-duplicate audit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Manifest: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
