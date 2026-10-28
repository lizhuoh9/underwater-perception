"""Obstacle directions from the segmentation's water masks (core/obstacle_dir.c), on the lake clip, scored against the
depth of the reference reconstruction.

Each frame's 60x80 non-water mask (results/lake_clip/mask_motion/frames.npz: the segmenter on the robot's 160x120
frames) splits into 3 x 5 directions of the camera's view (bands up / level / down, sectors from hard left to hard
right; see core/obstacle_dir.h for the angles, the blocking and the choice). The reference: the reconstruction's cloud,
seen from the reference camera of each frame through the same 60x80 grid (the 5th percentile depth of the points in a
cell, where at least two land). A direction is blocked at range R when half of its cells hold a surface nearer than R.
The reconstruction has no water surface, so above the robot it shows open water where the mask, rightly, does not.
"""

import argparse
import ctypes
import hashlib
from pathlib import Path

import numpy as np

from uwslam import mvo
from uwslam.paths import CORE_DIR, LAKE_CLIP_RESULTS
from uwslam.utils import dump_json

ROWS, COLUMNS = 3, 5  # OD_ROWS, OD_COLUMNS
DIRECTIONS = ROWS * COLUMNS
CENTRE = DIRECTIONS // 2
MAX_SIDE = 128  # OD_MAX_SIDE
MASK_HEIGHT, MASK_WIDTH = 60, 80
BANDS = ("up", "level", "down")
SECTORS = ("hard left", "left", "ahead", "right", "hard right")
RANGES = (0.5, 1.0, 2.0)  # metres
MIN_POINTS = 2  # reference points a cell needs to show a surface
DEPTH_PERCENTILE = 0.05  # of the points in a cell: near the nearest, past a stray floater
RATES = {10.0: 3, 5.0: 6}  # Hz: stride in 30 FPS frames
BOARD_RATES = mvo.BOARD_RATES
EXAMPLES = 4  # frames drawn in directions.png


class LayoutC(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int32), ("height", ctypes.c_int32), ("sector", ctypes.c_uint8 * MAX_SIDE),
                ("band", ctypes.c_uint8 * MAX_SIDE), ("cells", ctypes.c_uint16 * DIRECTIONS)]


class StateC(ctypes.Structure):
    _fields_ = [("share", ctypes.c_uint8 * DIRECTIONS), ("blocked", ctypes.c_uint8 * DIRECTIONS),
                ("clearing", ctypes.c_uint8 * DIRECTIONS), ("pad", ctypes.c_uint8 * 3), ("choice", ctypes.c_int32),
                ("frames", ctypes.c_uint32)]


def sources_sha256() -> dict:
    return {"dir_sha256": hashlib.sha256((CORE_DIR / "obstacle_dir.c").read_bytes()).hexdigest(),
            "dir_header_sha256": hashlib.sha256((CORE_DIR / "obstacle_dir.h").read_bytes()).hexdigest()}


def layout_for(lib: ctypes.CDLL, camera: mvo.Camera, width: int = MASK_WIDTH, height: int = MASK_HEIGHT) -> LayoutC:
    if lib.od_layout_size() != ctypes.sizeof(LayoutC) or lib.od_state_size() != ctypes.sizeof(StateC):
        raise RuntimeError("Python and C obstacle-direction layouts do not match")
    layout = LayoutC()
    if lib.od_layout(ctypes.byref(layout), ctypes.byref(camera), width, height) != 1:
        raise ValueError("The camera or the mask size was refused")
    return layout


def blocked_bits(state: StateC) -> int:
    """The blocked directions as the board reports them: bit d for direction d."""
    return sum(1 << d for d in range(DIRECTIONS) if state.blocked[d])


def run_directions(lib: ctypes.CDLL, masks: list[np.ndarray | None], camera: mvo.Camera) -> dict:
    """The directions after every frame's mask (None: no mask, the state stays): shares, blocked flags, choices and
    the FNV-1a hash of the state."""
    layout = layout_for(lib, camera)
    state = StateC()
    lib.od_init(ctypes.byref(state))
    shares, blocked, choices, bits = [], [], [], []
    for mask in masks:
        if mask is not None:
            cells = np.ascontiguousarray(mask, dtype=np.uint8)
            if cells.shape != (MASK_HEIGHT, MASK_WIDTH):
                raise ValueError(f"Expected a {MASK_HEIGHT}x{MASK_WIDTH} mask, got {cells.shape}")
            lib.od_update(ctypes.byref(state), ctypes.byref(layout), cells.ctypes.data)
        shares.append(np.array(state.share[:], dtype=np.int32))
        blocked.append(np.array(state.blocked[:], dtype=bool))
        choices.append(int(state.choice))
        bits.append(blocked_bits(state))
    return {"layout": layout, "shares": np.array(shares), "blocked": np.array(blocked), "choices": np.array(choices),
            "blocked_bits": bits, "frames": int(state.frames), "hash": mvo.fnv(mvo.FNV_OFFSET, bytes(state))}


def load_cloud(reconstruction: Path) -> np.ndarray:
    """Every point of the reconstruction's cloud (its own world axes), unthinned: near surfaces need them all."""
    raw = (reconstruction / "images.ply").read_bytes()
    start = raw.index(b"end_header\n") + len(b"end_header\n")
    points = np.frombuffer(raw[start:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"),
                                                ("b", "u1")])
    return np.stack([points["x"], points["y"], points["z"]], axis=1).astype(np.float32)


def reference_depth(cloud: np.ndarray, rotation: np.ndarray, centre: np.ndarray, camera) -> np.ndarray:
    """The reconstruction's depth (along the optical axis) in every 60x80 mask cell seen from one reference camera:
    the DEPTH_PERCENTILE point of those landing in the cell, infinity where fewer than MIN_POINTS land."""
    local = (cloud - centre.astype(np.float32)) @ rotation.astype(np.float32)
    ahead = local[:, 2] > 0.02
    local = local[ahead]
    depth = local[:, 2]
    u = camera.fx * local[:, 0] / depth + camera.cx
    v = camera.fy * local[:, 1] / depth + camera.cy
    inside = (u >= 0) & (u < MASK_WIDTH) & (v >= 0) & (v < MASK_HEIGHT)
    cell = v[inside].astype(np.int64) * MASK_WIDTH + u[inside].astype(np.int64)
    depth = depth[inside]
    order = np.lexsort((depth, cell))
    cell, depth = cell[order], depth[order]
    counts = np.bincount(cell, minlength=MASK_HEIGHT * MASK_WIDTH)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    enough = counts >= MIN_POINTS
    picks = starts + np.floor(DEPTH_PERCENTILE * np.maximum(counts - 1, 0)).astype(np.int64)
    out = np.full(MASK_HEIGHT * MASK_WIDTH, np.inf, dtype=np.float32)
    out[enough] = depth[picks[enough]]
    return out.reshape(MASK_HEIGHT, MASK_WIDTH)


def cell_directions(layout: LayoutC) -> np.ndarray:
    """The direction of every mask cell (60 x 80)."""
    sector = np.array(layout.sector[:MASK_WIDTH], dtype=np.int64)
    band = np.array(layout.band[:MASK_HEIGHT], dtype=np.int64)
    return band[:, None] * COLUMNS + sector[None, :]


def reference_blocked(depth: np.ndarray, layout: LayoutC, near: float) -> tuple[np.ndarray, np.ndarray]:
    """Per frame and direction: the share of cells with a reference surface nearer than `near`, and whether it is half."""
    directions = cell_directions(layout).ravel()
    cells = np.array(layout.cells[:], dtype=np.float64)
    seen = cells > 0
    share = np.zeros((len(depth), DIRECTIONS))
    for k, frame in enumerate(depth):
        counts = np.bincount(directions, weights=(frame.ravel() < near).astype(np.float64), minlength=DIRECTIONS)
        share[k, seen] = counts[seen] / cells[seen]
    return share, share >= 0.5


def score(result: dict, depth: np.ndarray, seed: int = mvo.SEED) -> dict:
    """Blocked directions against the reference at each range, per band; how often the chosen direction, straight ahead
    and a random seen direction are free at that range."""
    layout = result["layout"]
    blocked, choices = result["blocked"], result["choices"]
    seen = np.flatnonzero(np.array(layout.cells[:]) > 0)
    rng = np.random.default_rng(seed)
    random_pick = seen[rng.integers(len(seen), size=len(choices))]
    out = {"frames": int(len(choices)), "no_free_direction": float(np.mean(choices < 0)),
           "blocked_rate": {f"{BANDS[d // COLUMNS]} {SECTORS[d % COLUMNS]}": float(blocked[:, d].mean())
                            for d in range(DIRECTIONS)}}
    for near in RANGES:
        share, truth = reference_blocked(depth, layout, near)
        bands = {}
        for band, name in enumerate(BANDS + ("all",)):
            part = slice(band * COLUMNS, (band + 1) * COLUMNS) if name != "all" else slice(0, DIRECTIONS)
            t, b = truth[:, part], blocked[:, part]
            bands[name] = {"reference_blocked": float(t.mean()), "mask_blocked": float(b.mean()),
                           "recall": float((t & b).sum() / t.sum()) if t.any() else None,
                           "precision": float((t & b).sum() / b.sum()) if b.any() else None}
        chosen = choices >= 0
        out[f"within_{near:g}_m"] = {
            "bands": bands,
            "choice_free": float(np.mean(~truth[np.flatnonzero(chosen), choices[chosen]])) if chosen.any() else None,
            "ahead_free": float(np.mean(~truth[:, CENTRE])),
            "random_free": float(np.mean(~truth[np.arange(len(choices)), random_pick])),
            "share_correlation": float(np.corrcoef(result["shares"].ravel() / 255.0, share.ravel())[0, 1])}
    return out


def draw(output: Path, rate: float, frames: list[np.ndarray], masks: list[np.ndarray | None], depth: np.ndarray,
         result: dict) -> None:
    """directions.png: example frames with the directions over them (red blocked, green outline the choice), the mask
    and the reference depth; level_band.png: the level band over the clip."""
    from PIL import Image, ImageDraw
    from uwslam.utils import font
    layout = result["layout"]
    sector = np.array(layout.sector[:MASK_WIDTH])
    band = np.array(layout.band[:MASK_HEIGHT])
    columns = [int(np.flatnonzero(sector == s)[0]) * 4 if (sector == s).any() else None for s in range(COLUMNS)]
    rows = [int(np.flatnonzero(band == b)[0]) * 4 if (band == b).any() else None for b in range(ROWS)]
    edges_x = [c for c in columns if c is not None] + [320]
    edges_y = [r for r in rows if r is not None] + [240]
    picks = np.linspace(len(frames) * 0.1, len(frames) * 0.9, EXAMPLES).astype(int)
    sheet = Image.new("RGB", (3 * 330 + 10, EXAMPLES * 260 + 40), "white")
    pen = ImageDraw.Draw(sheet)
    pen.text((10, 8), f"{rate:g} Hz: red blocked, green outline the chosen direction | mask (red not water) | "
             "reference depth (white near, black none)", fill="black", font=font(14))
    for row, k in enumerate(picks):
        top = 40 + row * 260
        grey = Image.fromarray(frames[k]).convert("RGB")
        tint = Image.new("RGBA", grey.size, (0, 0, 0, 0))
        tinter = ImageDraw.Draw(tint)
        for d in range(DIRECTIONS):
            b, s = divmod(d, COLUMNS)
            if columns[s] is None or rows[b] is None:
                continue
            box = [columns[s], rows[b], edges_x[edges_x.index(columns[s]) + 1] - 1,
                   edges_y[edges_y.index(rows[b]) + 1] - 1]
            if result["blocked"][k, d]:
                tinter.rectangle(box, fill=(220, 40, 40, 90))
            tinter.rectangle(box, outline=(255, 255, 255, 120))
            if d == result["choices"][k]:
                tinter.rectangle(box, outline=(40, 200, 60, 255), width=4)
        grey = Image.alpha_composite(grey.convert("RGBA"), tint).convert("RGB")
        sheet.paste(grey, (10, top))
        mask = masks[k]
        tile = np.zeros((MASK_HEIGHT, MASK_WIDTH, 3), dtype=np.uint8)
        if mask is not None:
            tile[np.asarray(mask, bool)] = (220, 40, 40)
        sheet.paste(Image.fromarray(tile).resize((320, 240), Image.Resampling.NEAREST), (340, top))
        near = np.where(np.isfinite(depth[k]), 255 * (1 - np.clip(depth[k] / 2.0, 0, 1)), 0).astype(np.uint8)
        sheet.paste(Image.fromarray(near).convert("RGB").resize((320, 240), Image.Resampling.NEAREST), (670, top))
        pen.text((14, top + 4), f"frame {k}", fill="white", font=font(13))
    sheet.save(output / f"directions_{rate:g}hz.png")
    # the level band over the clip: one row per sector, a column per frame
    width = len(frames)
    strip = np.full((COLUMNS * 2, width, 3), 255, dtype=np.uint8)
    _, truth = reference_blocked(depth, layout, 1.0)
    for s in range(COLUMNS):
        d = COLUMNS + s
        strip[2 * s, result["blocked"][:, d]] = (220, 40, 40)
        strip[2 * s + 1, truth[:, d]] = (60, 60, 60)
        strip[2 * s, result["choices"] == d] = (40, 200, 60)
    image = Image.fromarray(strip).resize((max(700, 2 * width), COLUMNS * 2 * 14), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (image.width + 110, image.height + 64), "white")
    canvas.paste(image, (100, 54))
    pen = ImageDraw.Draw(canvas)
    pen.text((10, 6), f"Level band at {rate:g} Hz over the clip, one column per frame. Per sector, upper row: the mask's",
             fill="black", font=font(13))
    pen.text((10, 24), "blocked (red; green where chosen); lower row: the reference has a surface within 1 m (grey)",
             fill="black", font=font(13))
    for s in range(COLUMNS):
        pen.text((8, 54 + s * 28 + 6), SECTORS[s], fill="black", font=font(12))
    canvas.save(output / f"level_band_{rate:g}hz.png")


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def write_report(output: Path, summary: dict) -> None:
    lines = ["# Obstacle directions from the water masks (lake clip)", "",
             "The segmentation's 60×80 mask of each frame (not water: red in the figures) is split into 3 × 5 "
             "directions of the camera's view by `core/obstacle_dir.c`: bands up / level / down at ±15° and sectors at "
             "±15° and ±40° from the optical axis. A direction is blocked as soon as half of it is not water and freed "
             "only after two frames below 30%; the choice is the free direction nearest straight ahead (fewest steps, "
             "turning before rising or diving, then the more open). It is scored against the reference "
             "reconstruction, seen through the same grid from the reference camera of each frame: a direction counts "
             "as blocked at range R when half of its cells hold a reconstructed surface nearer than R.", "",
             "The reconstruction has no water surface, so in the upper band the mask blocks directions the reference "
             "calls open: the surface bounds the robot too. The mask says nothing about range; the voxel map does.", ""]
    for row in summary["runs"]:
        lines += [f"## {row['rate_hz']:g} Hz ({row['frames']} frames)", "",
                  f"No free direction in {percent(row['no_free_direction'])} of the frames.", "",
                  "| Range | Band | Reference blocked | Mask blocked | Reference-blocked directions the mask blocks | "
                  "Mask-blocked directions the reference blocks |", "|---:|---|---:|---:|---:|---:|"]
        for near in RANGES:
            entry = row[f"within_{near:g}_m"]
            for name in BANDS + ("all",):
                b = entry["bands"][name]
                lines.append(f"| {near:g} m | {name} | {percent(b['reference_blocked'])} | {percent(b['mask_blocked'])} | "
                             f"{percent(b['recall'])} | {percent(b['precision'])} |")
        lines += ["", "| Range | The choice is free | Straight ahead is free | A random direction is free | "
                  "Share correlation |", "|---:|---:|---:|---:|---:|"]
        for near in RANGES:
            entry = row[f"within_{near:g}_m"]
            lines.append(f"| {near:g} m | {percent(entry['choice_free'])} | {percent(entry['ahead_free'])} | "
                         f"{percent(entry['random_free'])} | {entry['share_correlation']:.2f} |")
        lines.append("")
    lines += ["Figures: `directions_<rate>hz.png` (example frames with their directions, mask and reference depth) and "
              "`level_band_<rate>hz.png` (the level band over the clip).", "",
              "Re-run (the output folder must not exist or be empty):", "", "```bash",
              "python -m uwslam.obstacle_dir <video> --reconstruction <folder with images.txt and images.ply> "
              "--output results/lake_clip/obstacle_dir_2", "```", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_board_files(output: Path, result: dict, masks: list[np.ndarray | None], rate: float) -> None:
    """What the board must report at `rate` for the directions: after every frame the choice and the blocked bits, at
    the end the state's hash (it takes the masks `uwslam.mono_map --board-masks` wrote)."""
    expected = {"rate_hz": rate, "every": BOARD_RATES[rate], "frames": len(masks),
                "with_mask": [mask is not None for mask in masks], "choices": result["choices"].tolist(),
                "blocked_bits": result["blocked_bits"], "dir_frames": result["frames"], "dir_hash": result["hash"],
                **sources_sha256(), **mvo.build_config()}
    dump_json(output / f"board_expected_dirs_{rate:g}hz.json", expected)


def run(video: Path, reconstruction: Path, output: Path, time_scale: float) -> dict:
    from uwslam.experiments.mask_motion import Camera, reference_poses
    from uwslam.experiments.mono_vo import ASPECT, HFOV_DEG, read_frames
    from uwslam.mono_map import load_matched_masks
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite results; choose a fresh --output")
    poses = np.loadtxt(reconstruction / "images.txt")
    frames, fps, _ = read_frames(video, poses[-1, 0] * time_scale)
    mvo.require_30fps(fps)
    grey = frames["320x240"]
    masks = load_matched_masks(video, fps)
    rotations, centres = reference_poses(poses, np.arange(len(grey)) / fps, time_scale)
    cloud = load_cloud(reconstruction)
    grid_camera = Camera.from_fov(HFOV_DEG, ASPECT)
    camera = mvo.camera_for(HFOV_DEG, ASPECT)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    import tempfile
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        lib = mvo.build(Path(scratch))
        depth_cache = {}
        for rate, stride in RATES.items():
            indices = list(range(0, len(grey), stride))
            frame_masks = [masks[i] if i < len(masks) else None for i in indices]
            result = run_directions(lib, frame_masks, camera)
            for i in indices:
                if i not in depth_cache:
                    depth_cache[i] = reference_depth(cloud, rotations[i], centres[i], grid_camera)
            depth = np.stack([depth_cache[i] for i in indices])
            scores = score(result, depth)
            runs.append({"rate_hz": rate, **scores})
            draw(output, rate, [grey[i] for i in indices], frame_masks, depth, result)
            if rate in BOARD_RATES:
                write_board_files(output, result, frame_masks, rate)
            level = scores["within_1_m"]
            print(f"{rate:g} Hz: the choice is free within 1 m in {percent(level['choice_free'])} of the frames "
                  f"(straight ahead {percent(level['ahead_free'])}); reference-blocked directions the mask blocks: "
                  f"{percent(level['bands']['all']['recall'])}", flush=True)
    summary = {"task": "Obstacle directions from the segmentation's water masks on the lake clip, scored against the "
                       "reference reconstruction's depth", "sources_sha256": sources_sha256(), **mvo.build_config(),
               "masks": "frames.npz",
               "time_scale": time_scale, "hfov_deg": HFOV_DEG, "ranges_m": list(RANGES),
               "reference_cell": {"min_points": MIN_POINTS, "depth_percentile": DEPTH_PERCENTILE}, "runs": runs}
    dump_json(output / "summary.json", summary)
    write_report(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=LAKE_CLIP_RESULTS / "obstacle_dir")
    parser.add_argument("--time-scale", type=float, default=3.0)
    args = parser.parse_args()
    run(args.video, args.reconstruction, args.output, args.time_scale)


if __name__ == "__main__":
    main()
