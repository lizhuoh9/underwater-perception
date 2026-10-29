"""Prepare all 1395 unique non-held-out SUIM training images for low-FP trials.

Requires data/SUIM.zip, data/manifest.json, and data/forest_training_manifest.json.
Existing forest_train files are reused; only additional pairs are extracted.
TEST image/mask bytes are hashed for identity checks, never decoded or tuned on.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile

from uwseg.prepare_data import (
    ARCHIVE_BYTES,
    ARCHIVE_SHA256,
    WORKSPACE,
    checked_members,
    destination_path,
    file_sha256,
    mask_class_ids,
    mask_encoding_audit,
    member_sha256,
    paired_members,
    verify_archive,
)

TRAIN_COUNT = 1395


def prepare() -> dict:
    data_root = WORKSPACE / "data"
    if not data_root.resolve().is_relative_to(WORKSPACE):
        raise ValueError("Data directory escapes the workspace")
    archive_path = data_root / "SUIM.zip"
    verify_archive(archive_path)
    input_paths = [data_root / "manifest.json", data_root / "forest_training_manifest.json"]
    input_bytes = [path.read_bytes() for path in input_paths]
    input_hashes = [hashlib.sha256(raw).hexdigest() for raw in input_bytes]
    original, previous = [json.loads(raw) for raw in input_bytes]
    for manifest in (original, previous):
        if manifest["source"]["sha256"] != ARCHIVE_SHA256:
            raise ValueError("Input manifest does not describe the pinned SUIM archive")
    if previous["audit"]["original_manifest_sha256"] != input_hashes[0]:
        raise ValueError("Previous training manifest refers to a different baseline manifest")

    with zipfile.ZipFile(archive_path) as archive:
        members = checked_members(archive)
        train_pairs = paired_members(members, "train_val")
        test_pairs = paired_members(members, "TEST")
        if len(train_pairs) != 1488 or len(test_pairs) != 110:
            raise ValueError("Pinned archive pair counts differ from the baseline")
        source_pairs = {image: (mask, stem) for stem, image, mask in train_pairs + test_pairs}
        split_hashes = {"calibration": set(), "test": set(), "previous_train": set()}
        split_counts = {name: 0 for name in split_hashes}
        previous_by_source = {}
        for manifest, is_previous in ((original, False), (previous, True)):
            for sample in manifest["samples"]:
                split = "previous_train" if is_previous else sample["split"]
                if split not in split_hashes or (is_previous and sample["split"] != "train"):
                    raise ValueError(f"Unexpected input split: {sample['split']}")
                prefix = "SUIM/TEST/images/" if split == "test" else "SUIM/train_val/images/"
                image = sample["source_image"]
                if not image.startswith(prefix) or source_pairs.get(image) != (sample["source_mask"], sample["id"]):
                    raise ValueError(f"Input sample is not a matching archive split/pair: {sample['id']}")
                for kind in ("image", "mask"):
                    local = WORKSPACE / sample[kind]
                    expected = sample[f"{kind}_sha256"]
                    if not local.resolve().is_relative_to(data_root.resolve()):
                        raise ValueError(f"Input sample path escapes data directory: {local}")
                    if file_sha256(local) != expected or member_sha256(archive, sample[f"source_{kind}"]) != expected:
                        raise ValueError(f"Input local/archive sample hash mismatch: {local}")
                split_hashes[split].add(sample["image_sha256"])
                split_counts[split] += 1
                if is_previous:
                    previous_by_source[image] = sample
        expected_counts = {"calibration": 80, "test": 110, "previous_train": 512}
        if (
            split_counts != expected_counts
            or {key: len(value) for key, value in split_hashes.items()} != expected_counts
        ):
            raise ValueError(f"Input split sample or unique-image counts differ: {split_counts}")
        for first, second in (("calibration", "test"), ("previous_train", "calibration"), ("previous_train", "test")):
            if split_hashes[first] & split_hashes[second]:
                raise ValueError(f"Input image hashes overlap between {first} and {second}")

        ordered = sorted(
            train_pairs,
            key=lambda pair: (
                hashlib.sha256(pair[1].encode("utf-8")).hexdigest(),
                pair[1],
            ),
        )
        blocked = split_hashes["calibration"] | split_hashes["test"]
        eligible, excluded_original, excluded_duplicate = [], [], []
        seen = set()
        for pair in ordered:
            stem, image, _ = pair
            digest = member_sha256(archive, image)
            entry = {"id": stem, "source_image": image, "image_sha256": digest}
            if digest in blocked:
                excluded_original.append(entry)
            elif digest in seen:
                excluded_duplicate.append(entry)
            else:
                seen.add(digest)
                eligible.append((pair, digest))
        if len(eligible) != TRAIN_COUNT:
            raise ValueError(f"Expected {TRAIN_COUNT} unique training images, got {len(eligible)}")
        if [pair[1] for pair, _ in eligible[:512]] != [sample["source_image"] for sample in previous["samples"]]:
            raise ValueError("Previous 512 samples differ from the deterministic training prefix")

        samples, extraction, classes = [], [], set()
        for (stem, image, mask), image_hash in eligible:
            raw_mask = archive.read(mask)
            mask_hash = hashlib.sha256(raw_mask).hexdigest()
            classes.update(mask_class_ids(raw_mask))
            if image in previous_by_source:
                old_sample = previous_by_source[image]
                if (old_sample["image_sha256"], old_sample["mask_sha256"]) != (image_hash, mask_hash):
                    raise ValueError(f"Reused sample hash mismatch: {image}")
                image_out = WORKSPACE / old_sample["image"]
                mask_out = WORKSPACE / old_sample["mask"]
            else:
                image_out = destination_path(data_root, "lowfp_train", "images", stem + ".jpg")
                mask_out = destination_path(data_root, "lowfp_train", "masks", stem + ".bmp")
            samples.append(
                {
                    "split": "train",
                    "id": stem,
                    "image": image_out.relative_to(WORKSPACE).as_posix(),
                    "mask": mask_out.relative_to(WORKSPACE).as_posix(),
                    "source_image": image,
                    "source_mask": mask,
                    "image_sha256": image_hash,
                    "mask_sha256": mask_hash,
                }
            )
            extraction.extend(((image, image_out, image_hash), (mask, mask_out, mask_hash)))
        overlaps = {
            "train_calibration": len(seen & split_hashes["calibration"]),
            "train_test": len(seen & split_hashes["test"]),
            "calibration_test": len(split_hashes["calibration"] & split_hashes["test"]),
        }
        if any(overlaps.values()) or not split_hashes["previous_train"] <= seen:
            raise ValueError("New training split overlap/subset verification failed")
        manifest = {
            "schema_version": 1,
            "source": original["source"],
            "counts": {
                "mirror_train_val": len(train_pairs),
                "mirror_test": len(test_pairs),
                "calibration": 80,
                "test": 110,
                "train": TRAIN_COUNT,
                "reused_previous_training": 512,
                "newly_extracted_training": TRAIN_COUNT - 512,
                "excluded_train_val_original_split_images": len(excluded_original),
                "excluded_train_val_within_train_duplicates": len(excluded_duplicate),
            },
            "audit": {
                "original_manifest": "data/manifest.json",
                "original_manifest_sha256": input_hashes[0],
                "previous_training_manifest": "data/forest_training_manifest.json",
                "previous_training_manifest_sha256": input_hashes[1],
                "verified_archive_bytes": ARCHIVE_BYTES,
                "verified_archive_sha256": ARCHIVE_SHA256,
                "input_local_image_and_mask_files_hash_verified": sum(split_counts.values()) * 2,
                "training_selection": "Exclude every calibration/TEST image SHA256; order all train_val pairs by SHA256 of ZIP image member UTF-8 name, then name; retain first occurrence of each image SHA256; use all 1395 unique eligible pairs.",
                "duplicate_definition": "Exact SHA256 of compressed-image file bytes only; decoded-pixel duplicates, near duplicates and related scenes are not excluded.",
                "previous_512_samples_verified_as_ordered_prefix": True,
                "training_unique_image_sha256_count": len(seen),
                "split_pairwise_image_hash_overlap_counts": overlaps,
                "calibration_labels_used_for_training": False,
                "test_labels_used_for_training": False,
                "test_labels_used_for_parameter_selection": False,
                "test_pixels_decoded": False,
                "excluded_original_split_images": excluded_original,
                "excluded_within_train_duplicate_images": excluded_duplicate,
                "selected_mask_encoding": mask_encoding_audit(classes, TRAIN_COUNT),
            },
            "samples": samples,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        manifest_path = data_root / "lowfp_training_manifest.json"
        if not manifest_path.resolve().is_relative_to(data_root.resolve()):
            raise ValueError("Output manifest path escapes data directory")
        for _, output, digest in extraction:
            if output.exists() and (not output.is_file() or file_sha256(output) != digest):
                raise ValueError(f"Refusing to overwrite nonmatching extracted file: {output}")
        if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
            raise ValueError(f"Refusing to overwrite nonmatching manifest: {manifest_path}")
        for source, output, _ in extraction:
            if not output.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(source) as stream, output.open("xb") as target:
                    shutil.copyfileobj(stream, target)
        for _, output, digest in extraction:
            if file_sha256(output) != digest:
                raise ValueError(f"Extracted/reused file hash verification failed: {output}")
        for path, expected in zip(input_paths, input_hashes):
            if file_sha256(path) != expected:
                raise ValueError(f"Input manifest changed during preparation: {path}")
        if not manifest_path.exists():
            with manifest_path.open("xb") as target:
                target.write(manifest_bytes)
        return manifest


def main() -> int:
    try:
        manifest = prepare()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Low-FP training preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["counts"], indent=2))
    print(json.dumps(manifest["audit"]["split_pairwise_image_hash_overlap_counts"], indent=2))
    print(f"Manifest: {WORKSPACE / 'data' / 'lowfp_training_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
