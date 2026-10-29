"""Decoded-pixel near-duplicate screening between dataset splits.

A 1024-bit difference hash of the grayscale image, also compared mirrored,
finds re-encoded, resized, aspect-changed and flipped copies. A coarse RGB
histogram adds a colour screen for frames of one clip that moved too far for
the hash; it has no hash cap, because reviewed same-clip frames reached about
520 bits. On SUIM, adjacent frames and unrelated open-water photos overlap at about
140-260 hash bits, so only distances up to AUTO_NEAR_DUPLICATE_BITS are decided
automatically; every other screened pair needs a recorded review.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HASH_WIDTH, HASH_HEIGHT = 32, 32
HASH_BITS = HASH_WIDTH * HASH_HEIGHT
COLOUR_SIZE = (64, 48)
COLOUR_LEVELS = 8
# Same-photo copies measured <= 81 bits; the closest unrelated pair was 164.
AUTO_NEAR_DUPLICATE_BITS = 100
# Reviewed adjacent frames and recrops reached 210 bits on hash alone; widening
# past 260 bits adds hundreds of unrelated open-water pairs per 40 bits.
SCREEN_BITS = 260
# Colour gates, lowered until each band added few new leaked images. TEST pairs
# go deeper because they bias reported metrics; calibration pairs are reported.
TEST_COLOUR_GATE = 0.60
CALIBRATION_COLOUR_GATE = 0.70
DECISIONS = {"near_duplicate": 2, "same_scene": 1, "different": 0}


@dataclass(frozen=True)
class Signature:
    hash_signs: np.ndarray
    mirror_signs: np.ndarray
    colour: np.ndarray


def _frozen(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def _gradient_signs(gray: Image.Image) -> np.ndarray:
    small = np.asarray(gray.resize((HASH_WIDTH + 1, HASH_HEIGHT), Image.Resampling.BILINEAR), dtype=np.float32)
    return _frozen(np.where(small[:, 1:] > small[:, :-1], 1.0, -1.0).astype(np.float32).ravel())


def image_signature(path: Path) -> Signature:
    """Hash signs as +/-1 so Hamming distances become one matrix product."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    gray = rgb.convert("L")
    levels = np.asarray(rgb.resize(COLOUR_SIZE, Image.Resampling.BILINEAR), dtype=np.int32) * COLOUR_LEVELS // 256
    bins = (levels[..., 0] * COLOUR_LEVELS + levels[..., 1]) * COLOUR_LEVELS + levels[..., 2]
    colour = np.bincount(bins.ravel(), minlength=COLOUR_LEVELS**3).astype(np.float32) / bins.size
    return Signature(_gradient_signs(gray), _gradient_signs(ImageOps.mirror(gray)), _frozen(colour))


def screen_pairs(first: dict[str, Signature], second: dict[str, Signature], colour_gate: float) -> list[dict]:
    """Every first x second pair close enough in hash, or in colour, to need a decision."""
    first_ids, second_ids = list(first), list(second)
    plain_a = np.stack([first[key].hash_signs for key in first_ids])
    mirror_a = np.stack([first[key].mirror_signs for key in first_ids])
    plain_b = np.stack([second[key].hash_signs for key in second_ids])
    colours_b = np.stack([second[key].colour for key in second_ids])
    plain = np.rint((HASH_BITS - plain_a @ plain_b.T) / 2).astype(np.int32)
    mirrored = np.rint((HASH_BITS - mirror_a @ plain_b.T) / 2).astype(np.int32)
    bits = np.minimum(plain, mirrored)
    pairs = []
    for i, a in enumerate(first_ids):
        colour = np.minimum(first[a].colour, colours_b).sum(axis=1)
        candidates = (bits[i] <= SCREEN_BITS) | (colour >= colour_gate)
        for j in np.flatnonzero(candidates):
            pairs.append(
                {
                    "a": a,
                    "b": second_ids[j],
                    "hash_bits": int(bits[i, j]),
                    "mirrored": bool(mirrored[i, j] < plain[i, j]),
                    "colour_intersection": round(float(colour[j]), 4),
                }
            )
    return pairs


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def load_review(path: Path) -> dict[tuple[str, str], dict]:
    reviews = {}
    for row in json.loads(path.read_text(encoding="utf-8"))["decisions"]:
        key = pair_key(row["a"], row["b"])
        if row["decision"] not in DECISIONS:
            raise ValueError(f"Unknown review decision {row['decision']!r} for {key}")
        if key in reviews:
            raise ValueError(f"Duplicate review decision for {key}")
        reviews[key] = row
    return reviews


def decide(pairs: list[dict], reviews: dict[tuple[str, str], dict]) -> tuple[list[dict], list[dict]]:
    """Recorded reviews win; otherwise only same-photo hash distances decide automatically."""
    decided, missing = [], []
    for pair in pairs:
        review = reviews.get(pair_key(pair["a"], pair["b"]))
        if review is not None:
            row = {**pair, "decision": review["decision"], "decided_by": "review", "note": review.get("note", "")}
            if pair["hash_bits"] <= AUTO_NEAR_DUPLICATE_BITS and row["decision"] != "near_duplicate":
                if not row["note"]:
                    raise ValueError(f"Overriding a same-photo hash match needs a note: {pair['a']} / {pair['b']}")
                row["overrides_hash"] = True
            decided.append(row)
        elif pair["hash_bits"] <= AUTO_NEAR_DUPLICATE_BITS:
            decided.append({**pair, "decision": "near_duplicate", "decided_by": "hash", "note": ""})
        else:
            missing.append(pair)
    return decided, missing


def leaked_ids(decided: list[dict], ids: set[str]) -> dict[str, dict]:
    """Strongest non-different decision per id, with its neighbours strongest first."""
    neighbours = {}
    for pair in decided:
        if pair["decision"] == "different":
            continue
        for own, other in ((pair["a"], pair["b"]), (pair["b"], pair["a"])):
            if own in ids:
                neighbours.setdefault(own, []).append(
                    {
                        "id": other,
                        "decision": pair["decision"],
                        "decided_by": pair["decided_by"],
                        "hash_bits": pair["hash_bits"],
                        "mirrored": pair["mirrored"],
                        "colour_intersection": pair["colour_intersection"],
                        "note": pair["note"],
                    }
                )
    leaks = {}
    for own, rows in neighbours.items():
        rows = sorted(rows, key=lambda row: (-DECISIONS[row["decision"]], row["hash_bits"], row["id"]))
        leaks[own] = {"tier": rows[0]["decision"], "neighbours": rows}
    return leaks
