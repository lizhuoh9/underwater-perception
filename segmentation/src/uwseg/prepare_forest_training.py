"""Extract 512 deterministic SUIM training pairs, disjoint from the baseline splits.

Run ``python -m uwseg.prepare_forest_training`` after preparing data/manifest.json.
The pinned data/SUIM.zip must already exist; this script never downloads data.
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

TRAIN_COUNT = 512


def prepare() -> dict:
    data_root = WORKSPACE / "data"
    archive_path = data_root / "SUIM.zip"
    original_path = data_root / "manifest.json"
    verify_archive(archive_path)
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()
    if original["source"]["sha256"] != ARCHIVE_SHA256:
        raise ValueError("Original manifest does not describe the pinned SUIM archive")

    with zipfile.ZipFile(archive_path) as archive:
        members = checked_members(archive)
        train_pairs = paired_members(members, "train_val")
        test_pairs = paired_members(members, "TEST")
        if len(train_pairs) != 1488 or len(test_pairs) != 110:
            raise ValueError("Pinned archive pair counts differ from the baseline")
        source_pairs = {image: (mask, stem) for stem, image, mask in train_pairs + test_pairs}
        split_hashes = {"calibration": set(), "test": set()}
        split_counts = {"calibration": 0, "test": 0}
        for sample in original["samples"]:
            split = sample["split"]
            if split not in split_hashes:
                raise ValueError(f"Unexpected original split: {split}")
            if source_pairs.get(sample["source_image"]) != (sample["source_mask"], sample["id"]):
                raise ValueError(f"Original manifest sample is not an archive pair: {sample['id']}")
            for kind in ("image", "mask"):
                source = sample[f"source_{kind}"]
                expected = sample[f"{kind}_sha256"]
                local = WORKSPACE / sample[kind]
                if not local.resolve().is_relative_to(data_root.resolve()):
                    raise ValueError(f"Original sample path escapes data directory: {local}")
                if member_sha256(archive, source) != expected or file_sha256(local) != expected:
                    raise ValueError(f"Original manifest hash mismatch: {source}")
            split_hashes[split].add(sample["image_sha256"])
            split_counts[split] += 1
        if split_counts != {"calibration": 80, "test": 110}:
            raise ValueError(f"Unexpected original split counts: {split_counts}")
        if split_hashes["calibration"] & split_hashes["test"]:
            raise ValueError("Original calibration and TEST image hashes overlap")

        blocked = split_hashes["calibration"] | split_hashes["test"]
        ordered = sorted(
            train_pairs,
            key=lambda pair: (
                hashlib.sha256(pair[1].encode("utf-8")).hexdigest(),
                pair[1],
            ),
        )
        eligible = []
        seen = set()
        excluded_original = []
        excluded_duplicate = []
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
        if len(eligible) < TRAIN_COUNT:
            raise ValueError("Insufficient distinct training images after split deduplication")

        samples, extraction, classes = [], [], set()
        for (stem, image, mask), image_hash in eligible[:TRAIN_COUNT]:
            image_out = destination_path(data_root, "forest_train", "images", stem + ".jpg")
            mask_out = destination_path(data_root, "forest_train", "masks", stem + ".bmp")
            raw_mask = archive.read(mask)
            mask_hash = hashlib.sha256(raw_mask).hexdigest()
            classes.update(mask_class_ids(raw_mask))
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
        training_hashes = {sample["image_sha256"] for sample in samples}
        overlaps = {
            "train_calibration": len(training_hashes & split_hashes["calibration"]),
            "train_test": len(training_hashes & split_hashes["test"]),
            "calibration_test": len(split_hashes["calibration"] & split_hashes["test"]),
        }
        if len(training_hashes) != TRAIN_COUNT or any(overlaps.values()):
            raise ValueError("Selected split hash disjointness check failed")
        manifest = {
            "schema_version": 1,
            "source": original["source"],
            "counts": {
                "mirror_train_val": len(train_pairs),
                "mirror_test": len(test_pairs),
                **split_counts,
                "train": TRAIN_COUNT,
                "excluded_train_val_original_split_images": len(excluded_original),
                "excluded_train_val_within_train_duplicates": len(excluded_duplicate),
                "eligible_unique_training_images": len(eligible),
            },
            "audit": {
                "original_manifest": original_path.relative_to(WORKSPACE).as_posix(),
                "original_manifest_sha256": original_sha256,
                "verified_archive_bytes": ARCHIVE_BYTES,
                "verified_archive_sha256": ARCHIVE_SHA256,
                "original_split_image_and_mask_files_hash_verified": sum(split_counts.values()) * 2,
                "training_selection": "Exclude all calibration/TEST image SHA256 values; order train_val pairs by SHA256 of ZIP image member UTF-8 name, then lexicographic name; keep first occurrence of each image SHA256; take first 512.",
                "test_labels_used_for_parameter_selection": False,
                "test_labels_used_for_training": False,
                "duplicate_definition": "Exact SHA256 of compressed-image file bytes; near duplicates and related scenes are not excluded.",
                "split_pairwise_image_hash_overlap_counts": overlaps,
                "training_unique_image_sha256_count": len(training_hashes),
                "excluded_original_split_images": excluded_original,
                "excluded_within_train_duplicate_images": excluded_duplicate,
                "selected_mask_encoding": mask_encoding_audit(classes, TRAIN_COUNT),
            },
            "samples": samples,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        manifest_path = data_root / "forest_training_manifest.json"
        if not manifest_path.resolve().is_relative_to(data_root.resolve()):
            raise ValueError("Training manifest path escapes data directory")
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
                raise ValueError(f"Extracted file hash verification failed: {output}")
        if file_sha256(original_path) != original_sha256:
            raise ValueError("Original manifest changed during training preparation")
        if not manifest_path.exists():
            with manifest_path.open("xb") as target:
                target.write(manifest_bytes)
        return manifest


def main() -> int:
    try:
        manifest = prepare()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Training preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["counts"], indent=2))
    print(json.dumps(manifest["audit"]["split_pairwise_image_hash_overlap_counts"], indent=2))
    print(f"Manifest: {WORKSPACE / 'data' / 'forest_training_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
