"""A top-view obstacle map from the C monocular odometry and the segmentation's water masks (core/mono_vo*.c and
core/mono_map.c, camera only), built on the lake clip and scored against the reconstruction's point cloud.

Before an explicit tracking loss, the tracked features with a well-triangulated point (rays at least 1 degree apart) whose
pixel the frame's 60x80 mask marks as not water are obstacle observations; core/mono_map.c files them in a 128x128
obstacle-evidence grid seen from above. Unobserved cells remain unknown; the cell size is one eighth of the scene's
median depth since one camera fixes no scale. The old grid freezes after the first LOST step: later tracking does
not relocalize it. A new map requires starting both VO and map again.

Scoring: the camera centres before the first LOST step are aligned to the reference by one similarity (Umeyama), the same transform
carries every observation into the reference, and its distance to the nearest point of the reconstruction's cloud
compares those observations with the reference. Underdetermined alignments are not scored. Everything is drawn in the first reference camera's axes (x right,
z forward, seen from above). The masks are results/lake_clip/mask_motion/frames.npz (the segmenter on the robot's
160x120 frames); the reference poses follow uwslam.experiments.mask_motion (3x timing, 110 degree field of view).
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from uwslam import mvo
from uwslam.paths import CLIP_RESULTS, CORE_DIR
from uwslam.utils import dump_json

MASKS = CLIP_RESULTS / "mask_motion" / "frames.npz"
MAP_SIZE = 128  # MM_SIZE
MASK_BYTES = 60 * 80 // 8
MAP_POLICY = 1  # freeze the old map after the first explicit tracking loss
RATES = {10.0: 3, 5.0: 6}  # Hz: stride in 30 FPS frames
BOARD_RATES = mvo.BOARD_RATES  # the rates the board runs, every how many of the 10 Hz board frames
CLOUD_VOXEL = 0.03  # metres: the reference cloud is thinned to one point per voxel before the nearest-point search
NEAR = (0.1, 0.2, 0.3)  # metres
OCCUPIED = 0  # log-odds above this count as obstacle cells
MATCH_CHUNK = 262143  # OpenCV BFMatcher's per-train-image index limit


def load_matched_masks(video: Path, fps: float) -> np.ndarray:
    """Load masks only for their source video and frame clock."""
    metadata = json.loads((MASKS.parent / "summary.json").read_text(encoding="utf-8"))
    source = metadata["source"]
    if source["sha256"] != mvo.file_sha256(video) or abs(source["fps"] - fps) > 0.001:
        raise ValueError("The mask source video or frame rate does not match the input video")
    with np.load(MASKS) as data:
        masks, times = data["mask_nonwater"], data["times"]
    if (
        masks.ndim != 3
        or masks.shape[1:] != (60, 80)
        or len(masks) != len(times)
        or not np.allclose(times, np.arange(len(times)) / fps, rtol=0.0, atol=1e-6)
    ):
        raise ValueError("Mask dimensions or frame times do not match the input video")
    return masks


def distance_spread(values: list[float | None]) -> dict | None:
    usable = [value for value in values if value is not None]
    return mvo.spread(usable) if usable else None


def number(value: float | None, fmt: str, unit: str = "") -> str:
    return "n/a" if value is None else format(value, fmt) + unit


def run_mapping(
    vo: ctypes.CDLL, frames: list[np.ndarray], masks: list[np.ndarray | None], camera: mvo.Camera, seed: int = mvo.SEED
) -> dict:
    """One odometry run over the frames with the map updated after every step from that frame's mask (None: no
    update). A LOST step interrupts the old map until a new VO and map are started; later tracking does not relocalize
    it. Continuity only records explicit LOST steps, not whether the trajectory or map is accurate or initialized."""
    if not frames or len(frames) != len(masks):
        raise ValueError("Expected one mask (or None) for every frame")
    state = ctypes.create_string_buffer(vo.mvo_state_size())
    pyramids = [ctypes.create_string_buffer(vo.mvo_pyramid_bytes()) for _ in range(2)]
    grid = ctypes.create_string_buffer(vo.mm_map_size())
    out = (ctypes.c_float * (3 * mvo.MAX_FEATURES))()
    vo.mvo_init(state, ctypes.byref(camera), seed, pyramids[0], pyramids[1])
    vo.mm_init(grid)

    def load(frame: np.ndarray) -> None:
        if frame.shape != (mvo.HEIGHT, mvo.WIDTH) or frame.dtype != np.uint8:
            raise ValueError(f"Expected a {mvo.HEIGHT}x{mvo.WIDTH} uint8 frame, got {frame.shape} {frame.dtype}")
        pixels = np.ascontiguousarray(frame)
        ctypes.memmove(vo.mvo_frame(state), pixels.ctypes.data, mvo.WIDTH * mvo.HEIGHT)

    load(frames[0])
    vo.mvo_start(state)
    steps, observations, continuity = [], [], []
    first_lost_frame = None
    for frame_index, (frame, mask) in enumerate(zip(frames[1:], masks[1:]), start=1):
        load(frame)
        step = mvo.StepC()
        vo.mvo_step(state, ctypes.byref(step))
        steps.append(step)
        if step.kind == 0 and first_lost_frame is None:
            first_lost_frame = frame_index
        continuous = first_lost_frame is None
        continuity.append(int(continuous))
        points = np.zeros((0, 3))
        count = 0
        if mask is not None:
            cells = np.ascontiguousarray(mask, dtype=np.uint8)
            if cells.ndim != 2 or not all(cells.shape):
                raise ValueError("Mask must be a nonempty two-dimensional array")
            if continuous:
                count = vo.mm_observe(state, cells.ctypes.data, cells.shape[1], cells.shape[0], out, mvo.MAX_FEATURES)
                points = np.array(out[: 3 * count], dtype=np.float64).reshape(-1, 3)
        if continuous:
            vo.mm_add(grid, state, out, count)
        observations.append(points)
    raw = bytes(grid)
    odds = np.frombuffer(raw[: MAP_SIZE * MAP_SIZE], dtype=np.int8).reshape(MAP_SIZE, MAP_SIZE).copy()
    cell, origin_x, origin_z, hits, updates, slice_y, height_resets = np.frombuffer(
        raw[MAP_SIZE * MAP_SIZE : MAP_SIZE * MAP_SIZE + 28], dtype="<f4,<i4,<i4,<u4,<u4,<f4,<u4"
    )[0]
    return {
        "steps": steps,
        "observations": observations,
        "odds": odds,
        "cell": float(cell),
        "origin": (int(origin_x), int(origin_z)),
        "hits": int(hits),
        "updates": int(updates),
        "slice_y": float(slice_y),
        "height_resets": int(height_resets),
        "map_continuity": continuity,
        "map_continuous": first_lost_frame is None,
        "first_lost_frame": first_lost_frame,
        "map_hash": mvo.fnv(mvo.FNV_OFFSET, raw),
    }


def load_cloud_world(reconstruction: Path) -> np.ndarray:
    """The reconstruction's cloud in its own world axes (x right, y down, z forward), thinned to CLOUD_VOXEL."""
    raw = (reconstruction / "images.ply").read_bytes()
    start = raw.index(b"end_header\n") + len(b"end_header\n")
    points = np.frombuffer(
        raw[start:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    world = np.stack([points["x"], points["y"], points["z"]], axis=1).astype(np.float64)
    _, keep = np.unique(np.floor(world / CLOUD_VOXEL).astype(np.int64), axis=0, return_index=True)
    return world[np.sort(keep)]


def nearest_distances(query: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """Distance from every query point to the nearest cloud point: OpenCV's brute-force matcher, exact and the same on
    every run (its approximate FLANN search missed the nearest point for about a fifth of the queries, which moved a
    run's median by up to a few centimetres)."""
    import cv2

    if not len(cloud):
        raise ValueError("Reference cloud is empty; distances cannot be scored")
    if not len(query):
        return np.zeros(0)
    query = np.ascontiguousarray(query, dtype=np.float32)
    cloud = np.ascontiguousarray(cloud, dtype=np.float32)
    if not np.isfinite(query).all() or not np.isfinite(cloud).all():
        raise ValueError("Observation and cloud coordinates must be finite")
    distances = np.full(len(query), np.inf)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    for start in range(0, len(cloud), MATCH_CHUNK):
        for match in matcher.match(query, cloud[start : start + MATCH_CHUNK]):
            distances[match.queryIdx] = min(distances[match.queryIdx], match.distance)
    if not np.isfinite(distances).all():
        raise ValueError("Some observations could not be matched to the reference cloud")
    return distances


def evaluate_map(
    result: dict, indices: list[int], ref_rotations: np.ndarray, ref_centres: np.ndarray, cloud_cam0: np.ndarray
) -> dict:
    """Score only the uninterrupted prefix, or return no score when its 3D alignment is underdetermined."""
    from uwslam.experiments.mono_vo import umeyama

    centres = np.vstack([np.zeros(3)] + [np.array(s.world_centre, dtype=np.float64) for s in result["steps"]])
    rotation0, centre0 = ref_rotations[indices[0]], ref_centres[indices[0]]
    reference = (ref_centres[indices] - centre0) @ rotation0
    first_lost = result.get("first_lost_frame")
    cutoff = len(centres) if first_lost is None else first_lost
    if cutoff < 1 or cutoff > len(centres) or len(reference) != len(centres):
        raise ValueError("Tracking prefix and reference frames do not agree")
    centres, reference = centres[:cutoff], reference[:cutoff]
    prefix_observations = result["observations"][: cutoff - 1]
    observations = np.vstack([points for points in prefix_observations if len(points)] or [np.zeros((0, 3))])
    centred = centres - centres.mean(axis=0)
    ref_centred = reference - reference.mean(axis=0)
    covariance = ref_centred.T @ centred
    can_align = len(centres) >= 3 and np.isfinite(covariance).all() and np.linalg.matrix_rank(covariance) >= 2
    scale, rotation, offset = umeyama(centres, reference) if can_align else (None, None, None)
    if can_align and (
        not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(rotation).all() or not np.isfinite(offset).all()
    ):
        can_align = False
        scale, rotation, offset = None, None, None
    aligned_track = scale * centres @ rotation.T + offset if can_align else np.empty((0, 3))
    aligned = scale * observations @ rotation.T + offset if can_align else np.empty((0, 3))
    distances = nearest_distances(aligned, cloud_cam0) if len(aligned) else np.empty(0)
    occupied = np.argwhere(result["odds"] > OCCUPIED)  # (row z, column x)
    origin_x, origin_z = result["origin"]
    cells_world = np.stack(
        [
            (occupied[:, 1] + origin_x + 0.5) * result["cell"],
            np.full(len(occupied), result["slice_y"]),
            (occupied[:, 0] + origin_z + 0.5) * result["cell"],
        ],
        axis=1,
    )
    spread = np.linalg.svd(covariance, compute_uv=False) if can_align else None
    return {
        "alignment_scale_m_per_unit": float(scale) if can_align else None,
        "alignment_spread_ratio": float(spread[1] / spread[0]) if can_align else None,
        "ate_rmse_m": float(np.sqrt(np.mean(np.sum((aligned_track - reference) ** 2, axis=1)))) if can_align else None,
        "first_lost_frame": first_lost,
        "map_continuous": first_lost is None,
        "continuous_frames": cutoff,
        "scored_frames": cutoff if can_align else 0,
        "total_frames": len(result["steps"]) + 1,
        "alignment_reason": None if can_align else "insufficient noncollinear continuous poses",
        "observations": int(len(observations)),
        "frames_with_observations": int(sum(len(p) > 0 for p in prefix_observations)),
        "distance_to_cloud_m": {
            "median": float(np.median(distances)) if len(distances) else None,
            **{f"within_{near:g}": float(np.mean(distances < near)) if len(distances) else None for near in NEAR},
        },
        "grid": {
            "cell_units": result["cell"],
            "origin_cells": list(result["origin"]),
            "occupied_cells": int(len(occupied)),
            "unknown_cells": int(np.sum(result["odds"] == 0)),
            "slice_y_units": result["slice_y"],
            "height_resets": result["height_resets"],
            "hits": result["hits"],
            "updates": result["updates"],
            "hash": result["map_hash"],
        },
        "_aligned": aligned,
        "_track": aligned_track,
        "_reference": reference,
        "_cells": scale * cells_world @ rotation.T + offset if can_align else np.empty((0, 3)),
        "_distances": distances,
    }


PANEL = 600  # pixels per top view
GREY, BLUE, ORANGE, GREEN, BLACK = (190, 190, 190), (31, 111, 178), (217, 72, 15), (43, 138, 62), (0, 0, 0)


def draw(output: Path, rate: float, evaluation: dict, cloud_cam0: np.ndarray, result: dict) -> None:
    """Two top views side by side: the aligned observations over the reference cloud (metres), and the robot's own
    grid (its units): red obstacle evidence, white unknown."""
    from PIL import Image, ImageDraw

    from uwslam.utils import font

    track, reference, aligned = evaluation["_track"], evaluation["_reference"], evaluation["_aligned"]
    span = np.vstack([reference, aligned]) if len(aligned) else reference
    low = np.percentile(span[:, [0, 2]], 1, axis=0) - 0.5
    high = np.percentile(span[:, [0, 2]], 99, axis=0) + 0.5
    size = float(max(high - low))
    image = Image.new("RGB", (2 * PANEL + 30, PANEL + 70), "white")
    pen = ImageDraw.Draw(image)

    def to_pixel(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return 10 + (x - low[0]) / size * PANEL, 50 + PANEL - (z - low[1]) / size * PANEL

    def dots(points: np.ndarray, colour: tuple[int, int, int], radius: float) -> None:
        for u, v in zip(*to_pixel(points[:, 0], points[:, 2])):
            if 10 <= u <= 10 + PANEL and 50 <= v <= 50 + PANEL:
                pen.ellipse([u - radius, v - radius, u + radius, v + radius], fill=colour)

    def line(points: np.ndarray, colour: tuple[int, int, int]) -> None:
        if not len(points):
            return
        u, v = to_pixel(points[:, 0], points[:, 2])
        if len(points) == 1:
            pen.ellipse((u[0] - 2, v[0] - 2, u[0] + 2, v[0] + 2), fill=colour)
        else:
            pen.line(list(zip(u.tolist(), v.tolist())), fill=colour, width=2)

    stride = max(1, len(cloud_cam0) // 60000)
    dots(cloud_cam0[::stride], GREY, 0.6)
    if len(aligned):
        near = evaluation["_distances"] < 0.2
        dots(aligned[~near], ORANGE, 1.1)
        dots(aligned[near], BLUE, 1.1)
    line(reference, BLACK)
    line(track, GREEN)
    pen.rectangle([10, 50, 10 + PANEL, 50 + PANEL], outline=BLACK)
    alignment = (
        "aligned to the reference" if evaluation["alignment_scale_m_per_unit"] is not None else "alignment unavailable"
    )
    pen.text(
        (10, 8), f"{rate:g} Hz, {alignment} (top view, {size:.1f} m across; z forward is up)", fill=BLACK, font=font(15)
    )
    pen.text(
        (10, 28),
        "grey cloud | blue map point within 0.2 m of it, orange farther | black reference path, green odometry",
        fill=BLACK,
        font=font(12),
    )
    odds = result["odds"].astype(np.int32)
    tile = np.full((MAP_SIZE, MAP_SIZE, 3), 255, dtype=np.uint8)
    strength = np.clip(np.abs(odds) * 8, 0, 200).astype(np.uint8)
    tile[odds > 0] = np.stack(
        [np.full(int((odds > 0).sum()), 220), 220 - strength[odds > 0], 220 - strength[odds > 0]], axis=1
    )
    grid = Image.fromarray(tile[::-1]).resize((PANEL, PANEL), Image.Resampling.NEAREST)  # row 0 (z low) at the bottom
    image.paste(grid, (PANEL + 20, 50))
    path = np.vstack([np.zeros(3)] + [np.array(s.world_centre, dtype=np.float64) for s in result["steps"]])
    if result.get("first_lost_frame") is not None:
        path = path[: result["first_lost_frame"]]
    if result["cell"] > 0:
        origin_x, origin_z = result["origin"]
        u = PANEL + 20 + (path[:, 0] / result["cell"] - origin_x) / MAP_SIZE * PANEL
        v = 50 + PANEL - (path[:, 2] / result["cell"] - origin_z) / MAP_SIZE * PANEL
        inside = (u >= PANEL + 20) & (u <= 2 * PANEL + 20) & (v >= 50) & (v <= 50 + PANEL)
        if inside.sum() > 1:
            pen.line(list(zip(u[inside].tolist(), v[inside].tolist())), fill=GREEN, width=2)
    pen.rectangle([PANEL + 20, 50, 2 * PANEL + 20, 50 + PANEL], outline=BLACK)
    pen.text(
        (PANEL + 20, 8),
        f"the robot's grid at the end: {MAP_SIZE}x{MAP_SIZE} cells of {result['cell']:.2f} "
        "odometry units, rolling with it",
        fill=BLACK,
        font=font(15),
    )
    pen.text((PANEL + 20, 28), "slice: red obstacle evidence, white unknown, green path", fill=BLACK, font=font(12))
    image.save(output / f"map_{rate:g}hz.png")


def write_report(output: Path, summary: dict) -> None:
    lines = [
        "# Obstacle map from the monocular odometry and the water masks (lake clip)",
        "",
        "Camera only: `core/mono_vo*.c` gives the motion (up to scale), `core/mono_map.c` files the tracked points "
        "that the segmentation marks as not water in a top-view grid. Scored after one similarity alignment of "
        "the camera path to the reference: the distance from every obstacle observation to the reconstruction's "
        "nearest cloud point (thinned to 3 cm). Each row of the first table is one run with one RANSAC seed; on this "
        "clip the seed alone can change these distances by a factor of two or more, so the second table gives the "
        "median and range over 16 seeds. A LOST step freezes the map; scores use only the preceding continuous "
        "prefix and are unavailable when its 3D alignment is underdetermined. These scores do not establish "
        "continuity or accuracy after loss. Different seeds can have different continuous prefix lengths. "
        "The alignment spread ratio is the second divided by the largest singular value of the centred "
        "cross-covariance; values near zero indicate weak alignment constraints, often from a nearly straight path. "
        "A larger ratio does not guarantee accuracy. The ratio in the first table is for that run's seed only.",
        "",
        "| Rate | Observations (frames) | Distance to cloud, median | Within 0.1 / 0.2 / 0.3 m | Path RMSE after "
        "alignment | Continuous frames / total | Alignment spread ratio | Grid cell (odometry units) | Occupied / unknown cells | Slice height / resets |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["runs"]:
        d, g = row["distance_to_cloud_m"], row["grid"]
        lines.append(
            f"| {row['rate_hz']:g} Hz | {row['observations']} ({row['frames_with_observations']}) | "
            f"{number(d['median'], '.2f', ' m')} | {number(d['within_0.1'], '.0%')} / "
            f"{number(d['within_0.2'], '.0%')} / {number(d['within_0.3'], '.0%')} | "
            f"{number(row['ate_rmse_m'], '.2f', ' m')} | {row['continuous_frames']} / {row['total_frames']} | "
            f"{number(row['alignment_spread_ratio'], '.2g')} | "
            f"{g['cell_units']:.2f} | {g['occupied_cells']} / {g['unknown_cells']} | "
            f"{g['slice_y_units']:.2f} / {g['height_resets']} |"
        )
    lines += [
        "",
        f"Over {len(mvo.SEEDS)} RANSAC seeds, median [smallest–largest]:",
        "",
        "| Rate | Distance to cloud, median of the run | Path RMSE after alignment |",
        "|---:|---:|---:|",
    ]
    for row in summary["runs"]:
        d, a = row["over_seeds"]["distance_median_m"], row["over_seeds"]["ate_rmse_m"]
        distance = "n/a" if d is None else f"{d['median']:.2f} m [{d['min']:.2f}–{d['max']:.2f}]"
        valid = row["over_seeds"]["distance_valid_seeds"]
        ate = "n/a" if a is None else f"{a['median']:.2f} m [{a['min']:.2f}–{a['max']:.2f}]"
        lines.append(
            f"| {row['rate_hz']:g} Hz | {distance} ({valid}/{len(mvo.SEEDS)} seeds) | "
            f"{ate} ({row['over_seeds']['ate_valid_seeds']}/{len(mvo.SEEDS)} seeds) |"
        )
    lines += [
        "",
        "The grid's cell is an eighth of the scene's median depth when the map starts; it stays in odometry units, "
        "whose length in metres drifts with the odometry's scale, so the grid has no fixed metric size. The grid "
        "takes a slice within half a scene depth of its fixed height; the grid resets when the estimated camera "
        "height moves one cell from it. Unobserved cells remain unknown: a ray at one height cannot prove the "
        "whole slice is free. On this clip most observations are the lake bed below the robot, so the slice is sparse.",
        "",
        "Top views: `map_10hz.png`, `map_5hz.png` (left: observations over the cloud; right: the grid the robot "
        "keeps, in its own units).",
        "",
        "Re-run (the output folder must not exist or be empty):",
        "",
        "```bash",
        "python -m uwslam.mono_map <video> --reconstruction <folder with images.txt and images.ply> "
        "--output results/lake_clip/mono_map_2",
        "```",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def sources_sha256() -> dict:
    """SHA-256 of the C files a board build must share with this run, keyed as the firmware reports them (the
    odometry's as uwslam.mvo.sources_sha256 gives them)."""
    names = {"map_sha256": "mono_map.c", "map_header_sha256": "mono_map.h"}
    return {
        **mvo.sources_sha256(),
        **{key: hashlib.sha256((CORE_DIR / name).read_bytes()).hexdigest() for key, name in names.items()},
    }


def write_board_files(
    output: Path, board_masks: Path, result: dict, masks: list[np.ndarray | None], rate: float
) -> None:
    """What the board must report at `rate` and, at 10 Hz, the masks it gets with each frame (a flag byte, then the
    60x80 cells as 600 bytes of bits, least significant first, row-major; flag 0 and zeros where the frame has none).
    At 5 Hz the board gets every second of those frames and masks."""
    if BOARD_RATES[rate] == 1:
        records = []
        for mask in masks:
            packed = (
                np.zeros(MASK_BYTES, dtype=np.uint8)
                if mask is None
                else np.packbits(np.asarray(mask, bool).ravel(), bitorder="little")
            )
            records.append(bytes([0 if mask is None else 1]) + packed.tobytes())
        board_masks.parent.mkdir(parents=True, exist_ok=True)
        board_masks.write_bytes(b"".join(records))
    expected = {
        "rate_hz": rate,
        "every": BOARD_RATES[rate],
        "frames": len(masks),
        "masks_file": board_masks.name,
        "step_hashes": [mvo.fnv(mvo.FNV_OFFSET, bytes(step)) for step in result["steps"]],
        "map_policy": MAP_POLICY,
        "map_continuity": result["map_continuity"],
        "map_continuous": int(result["map_continuous"]),
        "map_hash": result["map_hash"],
        "map_observations": result["hits"],
        "map_frames": result["updates"],
        "map_cell": result["cell"],
        "map_origin": list(result["origin"]),
        "map_slice_y": result["slice_y"],
        "map_height_resets": result["height_resets"],
        **sources_sha256(),
        **mvo.build_config(),
    }
    dump_json(output / f"board_expected_map_{rate:g}hz.json", expected)


def run(video: Path, reconstruction: Path, output: Path, time_scale: float, board_masks: Path | None = None) -> dict:
    from uwslam.experiments.mask_motion import reference_poses
    from uwslam.experiments.mono_vo import HFOV_DEG, LENS, read_frames

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite results; choose a fresh --output")
    if board_masks is not None and board_masks.exists():
        raise FileExistsError("Refusing to overwrite board masks; choose a fresh --board-masks")
    poses = np.loadtxt(reconstruction / "images.txt")
    frames, fps, _ = read_frames(video, poses[-1, 0] * time_scale)
    mvo.require_30fps(fps)
    grey = frames["320x240"]
    masks = load_matched_masks(video, fps)
    if len(grey) > len(masks) + 1:
        raise ValueError("The mask file does not cover the input video frames")
    ref_rotations, ref_centres = reference_poses(poses, np.arange(len(grey)) / fps, time_scale)
    cloud = load_cloud_world(reconstruction)
    cloud_cam0 = (cloud - ref_centres[0]) @ ref_rotations[0]
    camera = mvo.clip_camera()
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    import tempfile

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        vo = mvo.build(Path(scratch))
        for rate, stride in RATES.items():
            indices = list(range(0, len(grey), stride))
            frame_masks = [masks[i] if i < len(masks) else None for i in indices]
            result = run_mapping(vo, [grey[i] for i in indices], frame_masks, camera)
            if board_masks is not None and rate in BOARD_RATES:
                write_board_files(output, board_masks, result, frame_masks, rate)
            evaluation = evaluate_map(result, indices, ref_rotations, ref_centres, cloud_cam0)
            draw(output, rate, evaluation, cloud_cam0, result)
            seeds = [
                evaluate_map(
                    run_mapping(vo, [grey[i] for i in indices], frame_masks, camera, seed),
                    indices,
                    ref_rotations,
                    ref_centres,
                    cloud_cam0,
                )
                for seed in mvo.SEEDS
            ]
            distances = [e["distance_to_cloud_m"]["median"] for e in seeds]
            ates = [e["ate_rmse_m"] for e in seeds]
            over_seeds = {
                "seeds": list(mvo.SEEDS),
                "distance_median_m": distance_spread(distances),
                "distance_valid_seeds": sum(value is not None for value in distances),
                "ate_rmse_m": distance_spread(ates),
                "ate_valid_seeds": sum(value is not None for value in ates),
            }
            runs.append(
                {
                    "rate_hz": rate,
                    **{k: v for k, v in evaluation.items() if not k.startswith("_")},
                    "over_seeds": over_seeds,
                }
            )
            d = evaluation["distance_to_cloud_m"]
            median = over_seeds["distance_median_m"]
            print(
                f"{rate:g} Hz: {evaluation['observations']} observations, median distance {number(d['median'], '.2f', ' m')}, "
                f"within 0.2 m {number(d['within_0.2'], '.0%')}, path RMSE {number(evaluation['ate_rmse_m'], '.2f', ' m')}; over "
                f"{over_seeds['distance_valid_seeds']} seeds median distance "
                f"{number(median['median'] if median else None, '.2f', ' m')}",
                flush=True,
            )
    summary = {
        "task": "Top-view obstacle map from the C monocular odometry and the segmentation's water masks on the "
        "lake clip, scored against the reconstruction's cloud after aligning the camera path",
        "sources_sha256": sources_sha256(),
        **mvo.build_config(),
        "masks": MASKS.name,
        "time_scale": time_scale,
        "hfov_deg": HFOV_DEG,
        "lens": list(LENS),
        "cloud_voxel_m": CLOUD_VOXEL,
        "runs": runs,
    }
    dump_json(output / "summary.json", summary)
    write_report(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=CLIP_RESULTS / "mono_map")
    parser.add_argument("--time-scale", type=float, default=3.0)
    parser.add_argument(
        "--board-masks",
        type=Path,
        help="Where to write the 10 Hz masks the board test streams (with "
        "the frames from `uwslam.mvo compare --board-frames`)",
    )
    args = parser.parse_args()
    run(args.video, args.reconstruction, args.output, args.time_scale, args.board_masks)


if __name__ == "__main__":
    main()
