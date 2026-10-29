"""Prepare a checksum-pinned SUIM subset without importing remote code.

Run ``python -m uwseg.prepare_data`` with data/SUIM.zip already present, or use
``python -m uwseg.prepare_data --download`` to explicitly permit the archive download.
Only standard-library modules are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from uwseg.paths import DATA_DIR, REPO_ROOT

# Manifests record sample paths relative to the repository root.
WORKSPACE = REPO_ROOT
REVISION = "0f07fb8db0e067f66061cdf18a04a87c7cf247d8"
MIRROR_URL = "https://huggingface.co/datasets/SatwikKambham/suim"
ARCHIVE_URL = f"{MIRROR_URL}/resolve/{REVISION}/SUIM.zip"
ARCHIVE_SHA256 = "5ab7bd0763f7e674e5949cd2fa9b77023c40cdeb0ac2fca9924380dd664d7fba"
ARCHIVE_BYTES = 173794160
OFFICIAL_URL = "https://irvlab.cs.umn.edu/resources/suim-dataset"
PAPER_URL = "https://arxiv.org/abs/2004.01241"
CALIBRATION_COUNT = 80


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def verify_archive(path: Path) -> None:
    if path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError(f"Archive size mismatch: {path}")
    actual = file_sha256(path)
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"Archive SHA256 mismatch: expected {ARCHIVE_SHA256}, got {actual}")


def obtain_archive(path: Path, allow_download: bool) -> None:
    if path.exists():
        verify_archive(path)
        return
    if not allow_download:
        raise ValueError(f"Missing archive {path}; provide it or explicitly use --download")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    # Exclusive creation preserves any partial download left by a previous run.
    with partial.open("xb") as target:
        request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "SUIM-validation/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            received = 0
            while chunk := response.read(min(1024 * 1024, ARCHIVE_BYTES - received + 1)):
                received += len(chunk)
                if received > ARCHIVE_BYTES:
                    raise ValueError("Download exceeds pinned archive size")
                target.write(chunk)
    verify_archive(partial)
    partial.rename(path)


def checked_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = {}
    for member in archive.infolist():
        name = member.filename
        path = PurePosixPath(name)
        if (
            "\\" in name
            or path.is_absolute()
            or any(part in {".", ".."} or ":" in part for part in path.parts)
            or stat.S_ISLNK(member.external_attr >> 16)
        ):
            raise ValueError(f"Unsafe ZIP member: {name!r}")
        if name in members:
            raise ValueError(f"Duplicate ZIP member name: {name!r}")
        members[name] = member
    return members


def paired_members(members: dict, archive_split: str) -> list[tuple[str, str, str]]:
    groups = []
    for folder, suffix in (("images", ".jpg"), ("masks", ".bmp")):
        prefix = f"SUIM/{archive_split}/{folder}/"
        items = {}
        for name, member in members.items():
            if not name.startswith(prefix) or member.is_dir():
                continue
            basename = name[len(prefix) :]
            if "/" in basename or PurePosixPath(basename).suffix != suffix:
                raise ValueError(f"Unexpected image/mask member: {name}")
            stem = PurePosixPath(basename).stem
            if stem in items:
                raise ValueError(f"Duplicate sample ID: {name}")
            items[stem] = name
        groups.append(items)
    images, masks = groups
    if images.keys() != masks.keys():
        raise ValueError(
            f"Unpaired {archive_split} samples: "
            f"missing masks={sorted(images.keys() - masks.keys())}; "
            f"missing images={sorted(masks.keys() - images.keys())}"
        )
    if not images:
        raise ValueError(f"No paired samples in {archive_split}")
    return [(stem, images[stem], masks[stem]) for stem in sorted(images)]


def member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    with archive.open(name) as stream:
        return sha256_stream(stream)


def mask_class_ids(raw: bytes) -> set[int]:
    """Validate the pinned mirror's actual 8-bit ID-mask encoding, not its docs."""
    if len(raw) < 54 or raw[:2] != b"BM":
        raise ValueError("Mask is not a supported BMP file")
    offset = struct.unpack_from("<I", raw, 10)[0]
    header_size = struct.unpack_from("<I", raw, 14)[0]
    width, height = struct.unpack_from("<ii", raw, 18)
    bit_depth = struct.unpack_from("<H", raw, 28)[0]
    compression = struct.unpack_from("<I", raw, 30)[0]
    if header_size < 40 or width <= 0 or height == 0 or bit_depth != 8 or compression != 0:
        raise ValueError("Pinned mirror masks must be uncompressed 8-bit BMP class IDs")
    palette_start = 14 + header_size
    if offset < palette_start + 8 * 4 or len(raw) < palette_start + 8 * 4:
        raise ValueError("Missing mask palette entries")
    for class_id in range(8):
        blue, green, red, _ = struct.unpack_from("4B", raw, palette_start + class_id * 4)
        if (red, green, blue) != (class_id, class_id, class_id):
            raise ValueError("Mask palette does not map class IDs to identity grayscale")
    stride = (width + 3) // 4 * 4
    if offset + abs(height) * stride > len(raw):
        raise ValueError("Truncated mask pixel data")
    classes = set()
    for row in range(abs(height)):
        start = offset + row * stride
        classes.update(raw[start : start + width])
    if not classes <= set(range(8)):
        raise ValueError(f"Mask pixels contain invalid class IDs: {sorted(classes)}")
    return classes


def mask_encoding_audit(classes: set[int], sample_count: int) -> dict:
    return {
        "archive_encoding": "8-bit uncompressed BMP scalar class IDs; identity grayscale palette",
        "decoded_mode": "L",
        "bit_depth": 8,
        "compression": 0,
        "verified_samples": sample_count,
        "observed_class_ids": sorted(classes),
        "official_documented_encoding": "RGB bit palette; class_id = 4*R_bit + 2*G_bit + B_bit",
        "decoding_rule": "Read scalar IDs directly; do not threshold these grayscale bytes as RGB bits.",
    }


def destination_path(data_root: Path, split: str, folder: str, filename: str) -> Path:
    destination = data_root / split / folder / filename
    # resolve() also rejects an existing symlink redirecting an output directory.
    if not destination.resolve().is_relative_to(data_root.resolve()):
        raise ValueError(f"Output path escapes data directory: {destination}")
    return destination


def prepare(archive_path: Path, data_root: Path) -> dict:
    if not data_root.resolve().is_relative_to(WORKSPACE):
        raise ValueError("--data-dir must be inside this workspace")
    with zipfile.ZipFile(archive_path) as archive:
        members = checked_members(archive)
        test_pairs = paired_members(members, "TEST")
        train_pairs = paired_members(members, "train_val")
        if len(test_pairs) != 110 or len(train_pairs) != 1488:
            raise ValueError("Pinned mirror should contain 110 TEST and 1488 train_val pairs")

        test_hashes = {}
        image_hashes = {}
        for stem, image, _ in test_pairs:
            digest = member_sha256(archive, image)
            image_hashes[image] = digest
            test_hashes.setdefault(digest, []).append(stem)
        eligible, excluded = [], []
        for pair in train_pairs:
            stem, image, _ = pair
            digest = member_sha256(archive, image)
            image_hashes[image] = digest
            if digest in test_hashes:
                excluded.append(
                    {
                        "id": stem,
                        "source_image": image,
                        "image_sha256": digest,
                        "matching_test_ids": test_hashes[digest],
                    }
                )
            else:
                eligible.append(pair)
        eligible.sort(key=lambda pair: (hashlib.sha256(pair[1].encode("utf-8")).hexdigest(), pair[1]))
        if len(eligible) < CALIBRATION_COUNT:
            raise ValueError("Not enough train_val samples after exact-image TEST deduplication")

        samples = []
        extraction = []
        observed_classes = set()
        for split, pairs in (("calibration", eligible[:CALIBRATION_COUNT]), ("test", test_pairs)):
            for stem, image, mask in pairs:
                image_out = destination_path(data_root, split, "images", stem + ".jpg")
                mask_out = destination_path(data_root, split, "masks", stem + ".bmp")
                image_hash = image_hashes[image]
                mask_hash = member_sha256(archive, mask)
                observed_classes.update(mask_class_ids(archive.read(mask)))
                samples.append(
                    {
                        "split": split,
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

        manifest = {
            "schema_version": 1,
            "source": {
                "url": ARCHIVE_URL,
                "revision": REVISION,
                "sha256": ARCHIVE_SHA256,
                "archive_bytes": ARCHIVE_BYTES,
                "official_url": OFFICIAL_URL,
                "mirror_url": MIRROR_URL,
                "paper_url": PAPER_URL,
                "mirror_discrepancy": "Official train_val count is 1525; this pinned mirror contains 1488 pairs.",
            },
            "counts": {
                "official_train_val": 1525,
                "mirror_train_val": len(train_pairs),
                "mirror_test": len(test_pairs),
                "calibration": CALIBRATION_COUNT,
                "test": len(test_pairs),
                "excluded_train_val_exact_test_duplicates": len(excluded),
                "eligible_train_val_after_test_dedup": len(eligible),
            },
            "audit": {
                "selected_mask_encoding": mask_encoding_audit(observed_classes, len(samples)),
                "calibration_selection": "First 80 eligible train_val pairs ordered by SHA256 of ZIP image member UTF-8 name; lexicographic name breaks ties.",
                "test_labels_used_for_parameter_selection": False,
                "duplicate_definition": "Exact SHA256 match of compressed-image file bytes, not decoded pixels or perceptual similarity.",
                "excluded_train_val_test_duplicates": excluded,
                "within_test_exact_image_duplicate_groups": [
                    {"image_sha256": digest, "ids": ids} for digest, ids in sorted(test_hashes.items()) if len(ids) > 1
                ],
            },
            "samples": samples,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        manifest_path = data_root / "manifest.json"
        # Preflight all existing outputs before writing any selected image/mask.
        for _, output, digest in extraction:
            if output.exists() and (not output.is_file() or file_sha256(output) != digest):
                raise ValueError(f"Refusing to overwrite nonmatching extracted file: {output}")
        if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
            raise ValueError(f"Refusing to overwrite nonmatching manifest: {manifest_path}")
        for source, output, _ in extraction:
            if output.exists():
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(source) as stream, output.open("xb") as target:
                shutil.copyfileobj(stream, target)
        if not manifest_path.exists():
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("xb") as target:
                target.write(manifest_bytes)
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DATA_DIR / "SUIM.zip")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--download", action="store_true", help="explicitly permit downloading the missing pinned archive"
    )
    args = parser.parse_args()
    try:
        obtain_archive(args.archive, args.download)
        manifest = prepare(args.archive, args.data_dir.resolve())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Manifest: {args.data_dir.resolve() / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
