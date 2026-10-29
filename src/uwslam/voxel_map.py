"""The continuing 3D map (core/voxel_map.c) built from the C monocular odometry and the segmentation's water masks on the
lake clip, scored against the reference reconstruction's cloud.

After every odometry step the voxel map files the well-triangulated tracked points on cells that are not water, LOST
steps included (the map goes on through them in the odometry's frame). Two scores, both after similarity alignments of
the camera path to the reference (the map's unit is the odometry's): globally, the obstacles held at the end, with one
alignment of the whole path; locally, every 5 s, the obstacles the robot holds at that moment, with an alignment of
the last 5 s of its path, which is what avoidance uses. Distances are to the nearest point of the reconstruction's
cloud (thinned to 3 cm), over 16 RANSAC seeds since the seed alone can change them twofold.
"""

import argparse
import ctypes
import hashlib
import struct
from pathlib import Path

import numpy as np

from uwslam import mvo
from uwslam.paths import CLIP_RESULTS, CORE_DIR
from uwslam.utils import dump_json

# VM_CAPACITY, VM_SLOTS and VM_CELLS_PER_DEPTH of each build configuration (core/voxel_map.h)
LAYOUTS = {"s3": (1024, 2048, 8), "p4": (4096, 8192, 16)}
CAPACITY, SLOTS, CELLS_PER_DEPTH = LAYOUTS[mvo.CONFIG]
CONFIRM = 2  # VM_CONFIRM
WINDOW_S = 5.0
RATES = {10.0: 3, 5.0: 6}  # Hz: stride in 30 FPS frames
BOARD_RATES = mvo.BOARD_RATES
NEAR = (0.1, 0.2, 0.3)  # metres
FPS = 30.0


class VoxelC(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_int16 * 3),
        ("hits", ctypes.c_uint8),
        ("frames", ctypes.c_uint8),
        ("seen", ctypes.c_uint16),
    ]


class VoxelMapC(ctypes.Structure):
    _fields_ = [
        ("voxel", VoxelC * CAPACITY),
        ("index", ctypes.c_uint16 * SLOTS),
        ("side", ctypes.c_float),
        ("count", ctypes.c_uint32),
        ("frames", ctypes.c_uint32),
        ("observations", ctypes.c_uint32),
        ("evicted", ctypes.c_uint32),
        ("merged", ctypes.c_uint32),
        ("rescales", ctypes.c_uint32),
        ("lost", ctypes.c_uint32),
    ]


def sources_sha256() -> dict:
    return {
        "voxel_sha256": hashlib.sha256((CORE_DIR / "voxel_map.c").read_bytes()).hexdigest(),
        "voxel_header_sha256": hashlib.sha256((CORE_DIR / "voxel_map.h").read_bytes()).hexdigest(),
    }


def held(vmap: VoxelMapC) -> dict:
    """The voxels the map holds: keys, observations, frames, the frame each was last seen (as ages) and the side."""
    count = int(vmap.count)
    raw = np.frombuffer(
        bytes(vmap.voxel), dtype=[("key", "<i2", 3), ("hits", "u1"), ("frames", "u1"), ("seen", "<u2")]
    )[:count]
    ages = (int(vmap.frames) - raw["seen"].astype(np.int64)) & 0xFFFF
    return {
        "keys": raw["key"].astype(np.float64),
        "hits": raw["hits"].astype(np.int64),
        "frames": raw["frames"].astype(np.int64),
        "ages": ages,
        "side": float(vmap.side),
    }


def centres_of(voxels: dict, confirmed: bool = True) -> np.ndarray:
    keep = voxels["frames"] >= CONFIRM if confirmed else np.ones(len(voxels["keys"]), bool)
    return (voxels["keys"][keep] + 0.5) * voxels["side"]


def run_mapping(
    lib: ctypes.CDLL,
    frames: list[np.ndarray],
    masks: list[np.ndarray | None],
    camera: mvo.Camera,
    seed: int = mvo.SEED,
    snapshot_every: int | None = None,
) -> dict:
    """One odometry run with the voxel map updated after every step (a None mask: no mask for that frame). Keeps the
    steps, every step's voxel count and filed observations, the map every `snapshot_every` steps and at the end, and the
    map's FNV-1a hash."""
    if lib.vm_map_size() != ctypes.sizeof(VoxelMapC):
        raise RuntimeError("Python and C voxel-map layouts do not match")
    state = ctypes.create_string_buffer(lib.mvo_state_size())
    pyramids = [ctypes.create_string_buffer(lib.mvo_pyramid_bytes()) for _ in range(2)]
    vmap = VoxelMapC()
    lib.mvo_init(state, ctypes.byref(camera), seed, pyramids[0], pyramids[1])
    lib.vm_init(ctypes.byref(vmap))

    def load(frame: np.ndarray) -> None:
        if frame.shape != (mvo.HEIGHT, mvo.WIDTH) or frame.dtype != np.uint8:
            raise ValueError(f"Expected a {mvo.HEIGHT}x{mvo.WIDTH} uint8 frame, got {frame.shape} {frame.dtype}")
        ctypes.memmove(lib.mvo_frame(state), np.ascontiguousarray(frame).ctypes.data, mvo.WIDTH * mvo.HEIGHT)

    if not frames or len(frames) != len(masks):
        raise ValueError("Expected one mask (or None) for every frame")
    load(frames[0])
    lib.mvo_start(state)
    steps, progress, snapshots = [], [], []
    for number, (frame, mask) in enumerate(zip(frames[1:], masks[1:]), start=1):
        load(frame)
        step = mvo.StepC()
        lib.mvo_step(state, ctypes.byref(step))
        if mask is None:
            filed = lib.vm_update(ctypes.byref(vmap), state, ctypes.byref(step), None, 0, 0)
        else:
            cells = np.ascontiguousarray(mask, dtype=np.uint8)
            filed = lib.vm_update(
                ctypes.byref(vmap), state, ctypes.byref(step), cells.ctypes.data, cells.shape[1], cells.shape[0]
            )
        steps.append(step)
        progress.append({"count": int(vmap.count), "filed": int(filed)})
        if snapshot_every and (number % snapshot_every == 0 or number == len(frames) - 1):
            snapshots.append((number, held(vmap)))
    kinds = [s.kind for s in steps]
    return {
        "steps": steps,
        "progress": progress,
        "snapshots": snapshots,
        "final": held(vmap),
        "stats": {
            name: int(getattr(vmap, name))
            for name in ("count", "frames", "observations", "evicted", "merged", "rescales", "lost")
        },
        "side_bits": struct.unpack("<I", struct.pack("<f", vmap.side))[0],
        "first_lost_step": kinds.index(0) + 1 if 0 in kinds else None,
        "hash": mvo.fnv(mvo.FNV_OFFSET, bytes(vmap)),
    }


def alignment(centres: np.ndarray, reference: np.ndarray):
    """Similarity carrying odometry centres onto the reference, or None when the path is too short or straight."""
    from uwslam.experiments.mono_vo import umeyama

    centred = centres - centres.mean(axis=0)
    covariance = (reference - reference.mean(axis=0)).T @ centred
    if len(centres) < 3 or not np.isfinite(covariance).all() or np.linalg.matrix_rank(covariance) < 2:
        return None
    scale, rotation, offset = umeyama(centres, reference)
    return (scale, rotation, offset) if np.isfinite(scale) and scale > 0 else None


def distances(points: np.ndarray, transform, cloud: np.ndarray) -> np.ndarray:
    from uwslam.mono_map import nearest_distances

    if not len(points) or transform is None:
        return np.zeros(0)
    scale, rotation, offset = transform
    return nearest_distances(scale * points @ rotation.T + offset, cloud)


def evaluate(
    result: dict, indices: list[int], ref_rotations: np.ndarray, ref_centres: np.ndarray, cloud: np.ndarray, window: int
) -> dict:
    """The global and local scores (see the module's description), the time the held obstacles span and how much of
    the clip a map frozen at the first LOST step would have covered."""
    centres = np.vstack([np.zeros(3)] + [np.array(s.world_centre, dtype=np.float64) for s in result["steps"]])
    reference = (ref_centres[indices] - ref_centres[indices[0]]) @ ref_rotations[indices[0]]
    whole = alignment(centres, reference)
    final = result["final"]
    global_d = distances(centres_of(final), whole, cloud)
    local, placed = [], []
    for number, voxels in result["snapshots"]:
        first = max(0, number - window)
        part = alignment(centres[first : number + 1], reference[first : number + 1])
        points = centres_of(voxels)
        local.append(distances(points, part, cloud))
        if part is not None and len(points):
            placed.append(part[0] * points @ part[1].T + part[2])
    local_d = np.concatenate(local) if local else np.zeros(0)
    step_s = (indices[1] - indices[0]) / FPS
    ages = final["ages"][final["frames"] >= CONFIRM] * step_s

    def stats(values: np.ndarray) -> dict:
        return {
            "median_m": float(np.median(values)) if len(values) else None,
            **{f"within_{near:g}": float(np.mean(values < near)) if len(values) else None for near in NEAR},
        }

    frozen = result["first_lost_step"]
    return {
        "ate_rmse_m": float(
            np.sqrt(np.mean(np.sum((whole[0] * centres @ whole[1].T + whole[2] - reference) ** 2, axis=1)))
        )
        if whole
        else None,
        "global": stats(global_d),
        "local": stats(local_d),
        "local_snapshots": len(local),
        "obstacles_held": int((final["frames"] >= CONFIRM).sum()),
        "voxels_held": int(len(final["keys"])),
        "held_age_s": {
            "median": float(np.median(ages)) if len(ages) else None,
            "p90": float(np.percentile(ages, 90)) if len(ages) else None,
        },
        "side_m": float(whole[0] * final["side"]) if whole else None,
        "frozen_map_would_cover_s": None if frozen is None else frozen * step_s,
        "clip_s": len(result["steps"]) * step_s,
        **result["stats"],
        "_placed": np.vstack(placed) if placed else np.zeros((0, 3)),
        "_track": whole[0] * centres @ whole[1].T + whole[2] if whole else np.zeros((0, 3)),
        "_reference": reference,
    }


def spread(values: list) -> dict | None:
    usable = [v for v in values if v is not None]
    return mvo.spread(usable) if usable else None


WIDTH_PX = 1100  # the forward axis across the figure
GREY, BLUE, ORANGE, GREEN, BLACK = (195, 195, 195), (31, 111, 178), (217, 72, 15), (43, 138, 62), (0, 0, 0)


def draw(output: Path, rate: float, evaluation: dict, cloud: np.ndarray) -> None:
    """map_<rate>hz.png: the obstacles the robot held over the clip, each 5 s map placed by its own alignment, over
    the reference cloud: from above (forward to the right, the camera's right downwards) and, below it at the same
    scale, from the side (forward to the right, down downwards)."""
    from PIL import Image, ImageDraw

    from uwslam.mono_map import nearest_distances
    from uwslam.utils import font

    placed, track, reference = evaluation["_placed"], evaluation["_track"], evaluation["_reference"]
    near = nearest_distances(placed, cloud) < 0.2 if len(placed) else np.zeros(0, bool)
    span = np.vstack([reference, placed]) if len(placed) else reference
    low = np.percentile(span, 1, axis=0) - 0.3
    high = np.percentile(span, 99, axis=0) + 0.3
    scale = WIDTH_PX / float(high[2] - low[2])  # pixels per metre
    panels = []
    for axis, label in (
        (0, "from above (x, the camera's right, downwards)"),
        (1, "from the side (y, down, downwards)"),
    ):
        height = max(120, int(round((high[axis] - low[axis]) * scale)))
        panel = Image.new("RGB", (WIDTH_PX, height), "white")
        pen = ImageDraw.Draw(panel)  # drawing stops at the panel's edges

        def pixel(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (points[:, 2] - low[2]) * scale, (points[:, axis] - low[axis]) * scale

        def dots(points: np.ndarray, colour: tuple, radius: float) -> None:
            for u, v in zip(*pixel(points)):
                pen.ellipse([u - radius, v - radius, u + radius, v + radius], fill=colour)

        dots(cloud[:: max(1, len(cloud) // 60000)], GREY, 0.6)
        if len(placed):
            dots(placed[~near], ORANGE, 1.3)
            dots(placed[near], BLUE, 1.3)
        for path, colour in ((reference, BLACK), (track, GREEN)):
            if len(path) > 1:
                u, v = pixel(path)
                pen.line(list(zip(u.tolist(), v.tolist())), fill=colour, width=2)
        pen.rectangle([0, 0, WIDTH_PX - 1, height - 1], outline=BLACK)
        panels.append((label, panel))
    image = Image.new("RGB", (WIDTH_PX + 20, 50 + sum(p.height + 30 for _, p in panels)), "white")
    pen = ImageDraw.Draw(image)
    pen.text(
        (10, 8),
        f"{rate:g} Hz: obstacles the robot held, each 5 s map at its own alignment: blue within 0.2 m of "
        "the reference cloud (grey), orange farther; black reference path, green odometry",
        fill=BLACK,
        font=font(13),
    )
    top = 34
    for label, panel in panels:
        pen.text(
            (10, top), f"{label}; forward to the right, {high[2] - low[2]:.1f} m across", fill=BLACK, font=font(13)
        )
        image.paste(panel, (10, top + 18))
        top += panel.height + 30
    image.save(output / f"map_{rate:g}hz.png")


def number(value: float | None, fmt: str, unit: str = "") -> str:
    return "n/a" if value is None else format(value, fmt) + unit


def write_report(output: Path, summary: dict) -> None:
    lines = [
        "# The continuing voxel map from the monocular odometry (lake clip)",
        "",
        "`core/voxel_map.c` files the odometry's well-triangulated tracked points on cells the segmentation marks "
        "as not water, voxel by voxel, after every step and through LOST steps (the map goes on in the odometry's "
        "frame; it is not frozen at the first loss). The voxel side follows the scene: the median depth over "
        f"{CELLS_PER_DEPTH}, reset when the scene asks for twice or half of it (the monocular unit drifts tenfold "
        "over the clip). A voxel seen in two frames is an obstacle. It holds at most "
        f"{CAPACITY} voxels ({ctypes.sizeof(VoxelMapC) // 1024} KB): far voxels, then single sightings, then the "
        "longest unseen obstacles make room.",
        "",
        "Scores, after similarity alignments of the camera path (the map's unit is the odometry's): **global**, the "
        "obstacles held at the end with one alignment of the whole path (drift included); **local**, every 5 s "
        "the obstacles held at that moment with an alignment of the last 5 s of the path, what avoidance uses. "
        "Distances are to the reconstruction's nearest cloud point (thinned to 3 cm). Medians and ranges over "
        "16 RANSAC seeds.",
        "",
        "| Rate | Local: median distance | Local: within 0.1 / 0.2 m | Global: median distance | Global: within 0.2 m | "
        "Obstacles held at the end | Their age, median / 90% | Voxel side at the end | Side resets | "
        "LOST steps gone through | A map frozen at the first LOST would cover |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["runs"]:
        s = row["over_seeds"]

        def cell(key: str, fmt: str, unit: str = "") -> str:
            value = s.get(key)
            return (
                "n/a"
                if value is None
                else f"{format(value['median'], fmt)}{unit} [{format(value['min'], fmt)}–{format(value['max'], fmt)}]"
            )

        lines.append(
            f"| {row['rate_hz']:g} Hz | {cell('local_median_m', '.3f', ' m')} | "
            f"{cell('local_within_0.1', '.0%')} / {cell('local_within_0.2', '.0%')} | "
            f"{cell('global_median_m', '.2f', ' m')} | {cell('global_within_0.2', '.0%')} | "
            f"{cell('obstacles_held', '.0f')} | {cell('held_age_median_s', '.1f', ' s')} / "
            f"{cell('held_age_p90_s', '.1f', ' s')} | {cell('side_m', '.3f', ' m')} | {cell('rescales', '.0f')} | "
            f"{cell('lost', '.0f')} | {cell('frozen_s', '.1f', ' s')} of {row['clip_s']:.1f} s |"
        )
    lines += [
        "",
        "The default seed's run is drawn in `map_<rate>hz.png`. Its board expectation "
        "(`board_expected_voxels_<rate>hz.json`) holds every step's voxel count and filed observations and the "
        "final map's hash for `uwslam.mvo_stream --expected-voxels`.",
        "",
        "Limits: one clip; the reference's own errors; positions drift with the odometry (no loop closure); "
        "unobserved space is unknown, not free.",
        "",
        "Re-run (the output folder must not exist or be empty):",
        "",
        "```bash",
        "python -m uwslam.voxel_map <video> --reconstruction <folder with images.txt and images.ply> "
        "--output results/lake_clip/voxel_map_2",
        "```",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_board_files(output: Path, result: dict, rate: float) -> None:
    """What the board must report at `rate`: every step's voxel count and filed observations, and the final map."""
    expected = {
        "rate_hz": rate,
        "every": BOARD_RATES[rate],
        "frames": len(result["steps"]) + 1,
        "step_hashes": [mvo.fnv(mvo.FNV_OFFSET, bytes(step)) for step in result["steps"]],
        "voxel_progress": result["progress"],
        "voxel_hash": result["hash"],
        "voxel_side_bits": result["side_bits"],
        **{f"voxel_{name}": value for name, value in result["stats"].items()},
        **sources_sha256(),
        **mvo.build_config(),
    }
    dump_json(output / f"board_expected_voxels_{rate:g}hz.json", expected)


def run(video: Path, reconstruction: Path, output: Path, time_scale: float) -> dict:
    import tempfile

    from uwslam.experiments.mask_motion import reference_poses
    from uwslam.experiments.mono_vo import HFOV_DEG, LENS, read_frames
    from uwslam.mono_map import load_cloud_world, load_matched_masks

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite results; choose a fresh --output")
    poses = np.loadtxt(reconstruction / "images.txt")
    frames, fps, _ = read_frames(video, poses[-1, 0] * time_scale)
    mvo.require_30fps(fps)
    grey = frames["320x240"]
    masks = load_matched_masks(video, fps)
    ref_rotations, ref_centres = reference_poses(poses, np.arange(len(grey)) / fps, time_scale)
    cloud = (load_cloud_world(reconstruction) - ref_centres[0]) @ ref_rotations[0]
    camera = mvo.clip_camera()
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        lib = mvo.build(Path(scratch))
        for rate, stride in RATES.items():
            indices = list(range(0, len(grey), stride))
            frame_masks = [masks[i] if i < len(masks) else None for i in indices]
            window = int(round(WINDOW_S * rate))
            seeds = []
            for seed in mvo.SEEDS:
                result = run_mapping(lib, [grey[i] for i in indices], frame_masks, camera, seed, window)
                evaluation = evaluate(result, indices, ref_rotations, ref_centres, cloud, window)
                if seed == mvo.SEED:
                    draw(output, rate, evaluation, cloud)
                    if rate in BOARD_RATES:
                        write_board_files(output, result, rate)
                    default = {k: v for k, v in evaluation.items() if not k.startswith("_")}
                seeds.append(evaluation)
            over = {
                "local_median_m": spread([e["local"]["median_m"] for e in seeds]),
                "global_median_m": spread([e["global"]["median_m"] for e in seeds]),
                "obstacles_held": spread([e["obstacles_held"] for e in seeds]),
                "held_age_median_s": spread([e["held_age_s"]["median"] for e in seeds]),
                "held_age_p90_s": spread([e["held_age_s"]["p90"] for e in seeds]),
                "side_m": spread([e["side_m"] for e in seeds]),
                "rescales": spread([e["rescales"] for e in seeds]),
                "lost": spread([e["lost"] for e in seeds]),
                "frozen_s": spread(
                    [
                        e["frozen_map_would_cover_s"] if e["frozen_map_would_cover_s"] is not None else e["clip_s"]
                        for e in seeds
                    ]
                ),
                **{f"local_within_{near:g}": spread([e["local"][f"within_{near:g}"] for e in seeds]) for near in NEAR},
                **{
                    f"global_within_{near:g}": spread([e["global"][f"within_{near:g}"] for e in seeds]) for near in NEAR
                },
            }
            runs.append({"rate_hz": rate, "clip_s": seeds[0]["clip_s"], "default_seed": default, "over_seeds": over})
            print(
                f"{rate:g} Hz: local median {over['local_median_m']['median']:.3f} m, global median "
                f"{over['global_median_m']['median']:.2f} m, {over['obstacles_held']['median']:.0f} obstacles held, "
                f"{over['lost']['median']:.0f} LOST steps gone through",
                flush=True,
            )
    summary = {
        "task": "Continuing voxel map from the C monocular odometry and the water masks on the lake clip, "
        "scored against the reference cloud",
        "sources_sha256": {**mvo.sources_sha256(), **sources_sha256()},
        **mvo.build_config(),
        "masks": "frames.npz",
        "time_scale": time_scale,
        "hfov_deg": HFOV_DEG,
        "lens": list(LENS),
        "seeds": list(mvo.SEEDS),
        "window_s": WINDOW_S,
        "runs": runs,
    }
    dump_json(output / "summary.json", summary)
    write_report(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=CLIP_RESULTS / "voxel_map")
    parser.add_argument("--time-scale", type=float, default=3.0)
    args = parser.parse_args()
    run(args.video, args.reconstruction, args.output, args.time_scale)


if __name__ == "__main__":
    main()
