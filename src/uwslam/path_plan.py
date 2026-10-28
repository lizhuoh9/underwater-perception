"""Local avoidance and 3D path choice (core/path_plan.c) on the lake clip: where the robot would head at each step from
the voxel map's ranges and the water mask's blocked directions, scored against the reference reconstruction.

Each step runs as on the board: the frame's mask updates the obstacle directions, the odometry steps, the voxel map
files the step, and the planner chooses among the camera's 3 x 5 directions, towards straight ahead. The reference for
a direction is its free corridor: the 10th percentile range (metres, from the reference camera) over the central half
of the direction's 60x80 mask cells, where a cell holds the reconstruction's nearest surface and open water where it
holds none. The scoring is open-loop: the camera follows the diver's path, not the plan. A plan's range in steps is
compared with the reference's, its corridor range over the reference camera's recent step length, so that neither
needs the odometry's scale. So is the bound the plan puts on the water directions (`clear`, see core/path_plan.h), and
so are the map's vetoes in directions the mask sees as water: right when the reference has a surface within the safe
range there.
"""

import argparse
import ctypes
import hashlib
from pathlib import Path

import numpy as np

from uwslam import mvo, obstacle_dir, voxel_map
from uwslam.paths import CORE_DIR, LAKE_CLIP_RESULTS
from uwslam.utils import dump_json

DIRECTIONS, COLUMNS, CENTRE = obstacle_dir.DIRECTIONS, obstacle_dir.COLUMNS, obstacle_dir.CENTRE
ACTIONS = ("stop", "go", "creep", "turn")  # PP_STOP, PP_GO, PP_CREEP, PP_TURN
GO, CREEP = 1, 2
SAFE_STEPS = 8.0  # PP_SAFE_STEPS
RATES = {10.0: 3, 5.0: 6}  # Hz: stride in 30 FPS frames
BOARD_RATES = mvo.BOARD_RATES
NEAR = (0.25, 0.5, 1.0)  # metres
CORRIDOR_PERCENTILE = 0.1
SPEED_STEPS = 5  # reference steps averaged into the reference speed
EXAMPLES = 4


class PlanC(ctypes.Structure):
    _fields_ = [("action", ctypes.c_int32), ("direction", ctypes.c_int32), ("turn", ctypes.c_int32),
                ("goal", ctypes.c_int32), ("speed", ctypes.c_float), ("range", ctypes.c_float * DIRECTIONS),
                ("passable", ctypes.c_uint16), ("known", ctypes.c_uint16), ("nearest", ctypes.c_float),
                ("clear", ctypes.c_float), ("frames", ctypes.c_uint32)]


def sources_sha256() -> dict:
    return {"plan_sha256": hashlib.sha256((CORE_DIR / "path_plan.c").read_bytes()).hexdigest(),
            "plan_header_sha256": hashlib.sha256((CORE_DIR / "path_plan.h").read_bytes()).hexdigest()}


def run_planning(lib: ctypes.CDLL, frames: list[np.ndarray], masks: list[np.ndarray | None], camera: mvo.Camera,
                 seed: int = mvo.SEED) -> dict:
    """One run as the board does it: each frame's mask (None: none) updates the directions, then the odometry steps,
    the voxel map files the step and the planner plans. Keeps every step and plan, the mask-only choice and the hashes."""
    if lib.pp_plan_size() != ctypes.sizeof(PlanC) or lib.vm_map_size() != ctypes.sizeof(voxel_map.VoxelMapC):
        raise RuntimeError("Python and C planner or voxel-map layouts do not match")
    if not frames or len(frames) != len(masks):
        raise ValueError("Expected one mask (or None) for every frame")
    state = ctypes.create_string_buffer(lib.mvo_state_size())
    pyramids = [ctypes.create_string_buffer(lib.mvo_pyramid_bytes()) for _ in range(2)]
    vmap, plan = voxel_map.VoxelMapC(), PlanC()
    layout, dirs = obstacle_dir.layout_for(lib, camera), obstacle_dir.StateC()
    lib.mvo_init(state, ctypes.byref(camera), seed, pyramids[0], pyramids[1])
    lib.vm_init(ctypes.byref(vmap))
    lib.od_init(ctypes.byref(dirs))
    lib.pp_init(ctypes.byref(plan))
    steps, plans, plan_hashes, mask_choices = [], [], [], []
    for number, (frame, mask) in enumerate(zip(frames, masks)):
        if frame.shape != (mvo.HEIGHT, mvo.WIDTH) or frame.dtype != np.uint8:
            raise ValueError(f"Expected a {mvo.HEIGHT}x{mvo.WIDTH} uint8 frame, got {frame.shape} {frame.dtype}")
        ctypes.memmove(lib.mvo_frame(state), np.ascontiguousarray(frame).ctypes.data, mvo.WIDTH * mvo.HEIGHT)
        cells = None if mask is None else np.ascontiguousarray(mask, dtype=np.uint8)
        if cells is not None:
            lib.od_update(ctypes.byref(dirs), ctypes.byref(layout), cells.ctypes.data)
        if number == 0:
            lib.mvo_start(state)
            continue
        step = mvo.StepC()
        lib.mvo_step(state, ctypes.byref(step))
        if cells is None:
            lib.vm_update(ctypes.byref(vmap), state, ctypes.byref(step), None, 0, 0)
        else:
            lib.vm_update(ctypes.byref(vmap), state, ctypes.byref(step), cells.ctypes.data, cells.shape[1],
                          cells.shape[0])
        lib.pp_update(ctypes.byref(plan), ctypes.byref(vmap), state, ctypes.byref(layout), ctypes.byref(dirs), None)
        steps.append(step)
        plans.append({"action": int(plan.action), "direction": int(plan.direction), "turn": int(plan.turn),
                      "speed": float(plan.speed), "range": list(plan.range), "passable": int(plan.passable),
                      "known": int(plan.known), "nearest": float(plan.nearest), "clear": float(plan.clear),
                      "water": [d for d in range(DIRECTIONS) if dirs.frames > 0 and not dirs.blocked[d]
                                and layout.cells[d] > 0]})
        plan_hashes.append(mvo.fnv(mvo.FNV_OFFSET, bytes(plan)))
        mask_choices.append(int(dirs.choice))
    return {"layout": layout, "steps": steps, "plans": plans, "plan_hashes": plan_hashes, "mask_choices": mask_choices,
            "hash": mvo.fnv(mvo.FNV_OFFSET, bytes(plan))}


def corridor_cells(layout: obstacle_dir.LayoutC) -> np.ndarray:
    """For each direction (15 x 60 x 80): the central half of its mask rows and of its columns, its corridor."""
    sector = np.array(layout.sector[:obstacle_dir.MASK_WIDTH])
    band = np.array(layout.band[:obstacle_dir.MASK_HEIGHT])

    def middle(indices: np.ndarray) -> np.ndarray:
        quarter = len(indices) // 4
        return indices[quarter:len(indices) - quarter] if len(indices) >= 4 else indices

    out = np.zeros((DIRECTIONS, obstacle_dir.MASK_HEIGHT, obstacle_dir.MASK_WIDTH), bool)
    for d in range(DIRECTIONS):
        rows, columns = middle(np.flatnonzero(band == d // COLUMNS)), middle(np.flatnonzero(sector == d % COLUMNS))
        out[d][np.ix_(rows, columns)] = True
    return out


def corridor_ranges(depth: np.ndarray, corridors: np.ndarray, grid_camera) -> np.ndarray:
    """Each direction's reference corridor range (metres; infinity for open water): the CORRIDOR_PERCENTILE range over
    its corridor cells, a cell's range being its depth along the optical axis times its ray's length."""
    rows, columns = np.mgrid[0:obstacle_dir.MASK_HEIGHT, 0:obstacle_dir.MASK_WIDTH]
    length = np.sqrt(1 + ((columns + 0.5 - grid_camera.cx) / grid_camera.fx) ** 2
                     + ((rows + 0.5 - grid_camera.cy) / grid_camera.fy) ** 2)
    ranges = depth * length
    out = np.full(DIRECTIONS, np.inf)
    for d in range(DIRECTIONS):
        values = np.sort(ranges[corridors[d]])
        if len(values):
            out[d] = values[int(CORRIDOR_PERCENTILE * (len(values) - 1))]
    return out


def score(result: dict, truth: np.ndarray, reference_step_m: np.ndarray) -> dict:
    """The plan against the reference corridors (truth: steps x 15 metres; reference_step_m: the reference camera's
    recent step length at each step)."""
    plans, choices = result["plans"], np.array(result["mask_choices"])
    actions = np.array([p["action"] for p in plans])
    directions = np.array([p["direction"] for p in plans])
    seen = np.flatnonzero(np.array(result["layout"].cells[:]) > 0)
    going, creeping = np.flatnonzero(actions == GO), np.flatnonzero(actions == CREEP)
    gone, crept = truth[going, directions[going]], truth[creeping, directions[creeping]]
    ahead = truth[:, CENTRE]
    masked = np.flatnonzero(choices >= 0)
    mask_chosen = truth[masked, choices[masked]]
    best = truth[:, seen].max(axis=1)

    def free(values: np.ndarray) -> dict:
        return {f"within_{near:g}_m_free": float(np.mean(values >= near)) if len(values) else None for near in NEAR}

    ratios, chosen_ratios = [], []
    for k, p in enumerate(plans):
        if not (p["speed"] > 0) or not (reference_step_m[k] > 0):
            continue
        for d in range(DIRECTIONS):
            if (p["known"] >> d) & 1 and np.isfinite(truth[k, d]) and truth[k, d] > 0:
                ratio = np.log2((p["range"][d] / p["speed"]) / (truth[k, d] / reference_step_m[k]))
                ratios.append(ratio)
                if d == p["direction"]:
                    chosen_ratios.append(ratio)

    def agreement(values: list) -> dict:
        v = np.abs(np.array(values))
        return {"directions": len(values), "median_log2_ratio": float(np.median(values)) if values else None,
                "within_2x": float(np.mean(v <= 1.0)) if values else None}

    bound, vetoes = water_scores(plans, choices, truth, reference_step_m)
    return {"steps": len(plans), "actions": {name: float(np.mean(actions == code)) for code, name in enumerate(ACTIONS)},
            "water_bound": bound, "water_vetoes": vetoes,
            "plan": {**free(gone), "danger_go_below_0.25_m": float(np.mean(gone < 0.25)) if len(gone) else None},
            "plan_creeping": {**free(crept), "steps": len(creeping)},
            "straight_ahead": free(ahead), "mask_choice": {**free(mask_chosen), "steps_with_a_choice": len(masked)},
            "best_direction": free(best), "steps_in_steps": {"all_known": agreement(ratios),
                                                             "chosen": agreement(chosen_ratios)}}


def water_scores(plans: list[dict], choices: np.ndarray, truth: np.ndarray, reference_step_m: np.ndarray) -> tuple:
    """How the plan's bound on the water directions and the map's vetoes there hold against the reference, all in
    steps: the bound (`clear`) against each water direction's corridor, and each veto (the map putting an obstacle
    within the safe range in a direction the mask sees as water) against a reference surface within that range."""
    below, beyond_safe, values, near, water_total = 0, 0, [], 0, 0
    veto_steps, veto_on_choice, veto_right, veto_wrong = 0, 0, 0, 0
    for k, p in enumerate(plans):
        if not (p["speed"] > 0) or not (reference_step_m[k] > 0):
            continue
        reference = truth[k] / reference_step_m[k]
        water_total += len(p["water"])
        near += int(sum(reference[d] < SAFE_STEPS for d in p["water"]))
        vetoed = [d for d in p["water"] if (p["known"] >> d) & 1 and p["range"][d] / p["speed"] < SAFE_STEPS]
        veto_steps += bool(vetoed)
        veto_on_choice += int(choices[k] in vetoed)
        veto_right += int(sum(reference[d] < SAFE_STEPS for d in vetoed))
        veto_wrong += int(sum(reference[d] >= SAFE_STEPS for d in vetoed))
        if p["clear"] >= 0 and p["water"]:
            clear = p["clear"] / p["speed"]
            values.append(clear)
            below += int(sum(reference[d] < clear for d in p["water"]))
            beyond_safe += len(p["water"]) if clear >= SAFE_STEPS else 0
    bounded = sum(len(p["water"]) for k, p in enumerate(plans)
                  if p["clear"] >= 0 and p["speed"] > 0 and reference_step_m[k] > 0)
    bound = {"water_directions": water_total, "with_a_reference_surface_within_safe": near,
             "bounded_share": bounded / water_total if water_total else None,
             "median_steps": float(np.median(values)) if values else None,
             "nearer_than_the_bound": below / bounded if bounded else None,
             "clear_beyond_safe": beyond_safe / bounded if bounded else None}
    vetoes = {"steps": veto_steps, "on_the_mask_choice": veto_on_choice, "right": veto_right, "wrong": veto_wrong}
    return bound, vetoes


def reference_steps(ref_centres: np.ndarray, indices: list[int]) -> np.ndarray:
    """The reference camera's step length (metres) at each step, averaged over the last SPEED_STEPS steps."""
    lengths = np.linalg.norm(np.diff(ref_centres[indices], axis=0), axis=1)
    return np.array([lengths[max(0, k - SPEED_STEPS + 1):k + 1].mean() for k in range(len(lengths))])


def spread(values: list) -> dict | None:
    usable = [v for v in values if v is not None]
    return mvo.spread(usable) if usable else None


GREEN, RED, ORANGE, GREY, BLUE, BLACK = (40, 160, 70), (210, 50, 40), (230, 140, 30), (150, 150, 150), (40, 110, 190), (0, 0, 0)
ACTION_COLOURS = {0: (120, 120, 120), 1: GREEN, 2: (230, 190, 40), 3: BLUE}


def draw(output: Path, rate: float, frames: list[np.ndarray], result: dict, truth: np.ndarray) -> None:
    """plan_<rate>hz.png: over the clip, the reference corridor of the planned direction, straight ahead and the mask's
    own choice, under a strip of the actions; plan_frames_<rate>hz.png: example frames with the plan over them."""
    from PIL import Image, ImageDraw
    from uwslam.utils import font
    plans, choices = result["plans"], result["mask_choices"]
    width, height, top, left = max(900, 3 * len(plans)), 300, 70, 60
    image = Image.new("RGB", (width + left + 20, height + top + 60), "white")
    pen = ImageDraw.Draw(image)
    pen.text((10, 6), f"{rate:g} Hz: free corridor (reference, metres) of the planned direction (green), straight ahead "
             "(grey) and the mask's own choice (orange)", fill=BLACK, font=font(13))
    pen.text((10, 24), "strip: the plan's action (green go, yellow creep, blue turn, grey stop)", fill=BLACK, font=font(12))
    cap = 1.5
    for k, p in enumerate(plans):
        x0 = left + k * width / len(plans)
        pen.rectangle([x0, 44, x0 + width / len(plans), 58], fill=ACTION_COLOURS[p["action"]])

    def y_of(value: float) -> float:
        return top + height - min(value, cap) / cap * height

    for near in (0.25, 0.5, 1.0):
        pen.line([(left, y_of(near)), (left + width, y_of(near))], fill=(225, 225, 225))
        pen.text((8, y_of(near) - 7), f"{near:g} m", fill=BLACK, font=font(11))
    series = [([truth[k, CENTRE] for k in range(len(plans))], GREY),
              ([truth[k, c] if c >= 0 else np.nan for k, c in enumerate(choices)], ORANGE),
              ([truth[k, p["direction"]] if p["direction"] >= 0 else np.nan for k, p in enumerate(plans)], GREEN)]
    for values, colour in series:
        for k, value in enumerate(values):
            if np.isfinite(value) or value == np.inf:
                x = left + (k + 0.5) * width / len(plans)
                y = y_of(value if np.isfinite(value) else cap)
                pen.ellipse([x - 2, y - 2, x + 2, y + 2], fill=colour)
    pen.rectangle([left, top, left + width, top + height], outline=BLACK)
    pen.text((left, top + height + 8), "steps over the clip; above 1.5 m (or open water) drawn at the top", fill=BLACK,
             font=font(12))
    image.save(output / f"plan_{rate:g}hz.png")
    layout = result["layout"]
    sector = np.array(layout.sector[:obstacle_dir.MASK_WIDTH])
    band = np.array(layout.band[:obstacle_dir.MASK_HEIGHT])
    picks = np.linspace(len(plans) * 0.1, len(plans) * 0.9, EXAMPLES).astype(int)
    sheet = Image.new("RGB", (2 * 330 + 10, EXAMPLES * 260 + 40), "white")
    pen = ImageDraw.Draw(sheet)
    pen.text((10, 8), f"{rate:g} Hz: green passable, red not; numbers: the map's range in steps; thick: the plan",
             fill=BLACK, font=font(13))
    for row, k in enumerate(picks):
        p, frame_top = plans[k], 40 + row * 260
        grey = Image.fromarray(frames[k + 1]).convert("RGBA")
        tint = Image.new("RGBA", grey.size, (0, 0, 0, 0))
        tinter = ImageDraw.Draw(tint)
        for d in range(DIRECTIONS):
            columns, rows = np.flatnonzero(sector == d % COLUMNS), np.flatnonzero(band == d // COLUMNS)
            if not len(columns) or not len(rows):
                continue
            box = [4 * columns[0], 4 * rows[0], 4 * columns[-1] + 3, 4 * rows[-1] + 3]
            passable = (p["passable"] >> d) & 1
            tinter.rectangle(box, fill=(40, 200, 80, 60) if passable else (220, 40, 40, 80), outline=(255, 255, 255, 140))
            if (p["known"] >> d) & 1 and p["speed"] > 0:
                tinter.text((box[0] + 4, box[1] + 4), f"{p['range'][d] / p['speed']:.0f}", fill=(255, 255, 255, 255),
                            font=font(13))
            if d == p["direction"]:
                tinter.rectangle(box, outline=(255, 255, 0, 255), width=4)
        sheet.paste(Image.alpha_composite(grey, tint).convert("RGB"), (10, frame_top))
        truth_tile = Image.new("RGB", (320, 240), "white")
        tpen = ImageDraw.Draw(truth_tile)
        for d in range(DIRECTIONS):
            columns, rows = np.flatnonzero(sector == d % COLUMNS), np.flatnonzero(band == d // COLUMNS)
            if not len(columns) or not len(rows):
                continue
            box = [4 * columns[0], 4 * rows[0], 4 * columns[-1] + 3, 4 * rows[-1] + 3]
            value = truth[k, d]
            shade = 255 if not np.isfinite(value) else int(255 * min(value, 1.5) / 1.5)
            tpen.rectangle(box, fill=(shade, shade, shade), outline=(200, 200, 200))
            tpen.text((box[0] + 4, box[1] + 4), "water" if not np.isfinite(value) else f"{value:.2f} m",
                      fill=RED if np.isfinite(value) and value < 0.5 else BLACK, font=font(12))
        sheet.paste(truth_tile, (340, frame_top))
        pen.text((14, frame_top + 222), f"step {k + 1}: {ACTIONS[p['action']]}", fill=(255, 255, 255), font=font(13))
    sheet.save(output / f"plan_frames_{rate:g}hz.png")


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def number(value: float | None, fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def write_report(output: Path, summary: dict) -> None:
    lines = ["# Local avoidance and 3D path choice (lake clip)", "",
             "`core/path_plan.c` chooses, after every odometry step, which of the camera's 3 × 5 directions to head for "
             "(towards straight ahead): passable means water in the mask and no obstacle in the voxel map within 8 "
             "steps (1.6 s at 5 Hz); the map adds vetoes but lifts none. With none passable it creeps into the longest "
             "free water the map knows, else turns in place. Ranges count in steps of the odometry's recent speed, so "
             "they need no metric scale. Scored open-loop against the reconstruction: a direction's free corridor "
             "is the 10th percentile range over the central half of its cells; the camera follows the diver, not the "
             "plan. Medians and ranges over 16 RANSAC seeds.", ""]
    for row in summary["runs"]:
        s = row["over_seeds"]
        lines += [f"## {row['rate_hz']:g} Hz ({row['steps']} steps)", "",
                  "| Heading for | Free beyond 0.25 m | Free beyond 0.5 m | Free beyond 1 m |", "|---|---:|---:|---:|"]
        for key, label in (("plan", "the plan, going"), ("plan_creeping", "the plan, creeping"),
                           ("mask_choice", "the mask's own choice (when it has one)"),
                           ("straight_ahead", "straight ahead"), ("best_direction", "the best direction (hindsight)")):
            cells = []
            for near in NEAR:
                value = s[f"{key}_within_{near:g}_m_free"]
                cells.append("n/a" if value is None else f"{percent(value['median'])} [{percent(value['min'])}–"
                                                         f"{percent(value['max'])}]")
            lines.append(f"| {label} | {' | '.join(cells)} |")
        actions = ", ".join(f"{name} {percent(s[f'action_{name}']['median'])}" for name in ACTIONS
                            if s[f"action_{name}"] is not None)
        bound, vetoes = row["default_seed"]["water_bound"], row["default_seed"]["water_vetoes"]
        known = s["known_within_2x"]
        counts = row["default_seed"]["steps_in_steps"]
        lines += ["", f"Actions: {actions}. The plan went into a corridor blocked within 0.25 m in "
                  f"{percent(s['danger']['median'])} of its go steps; it creeps where the mask has no free direction.",
                  f"Its ranges in steps agree with the reference's within a factor of 2 for "
                  f"{percent(known['median'] if known else None)} of the directions with a known range (the default "
                  f"seed: {counts['all_known']['directions']} direction-steps, median log2 ratio "
                  f"{number(counts['all_known']['median_log2_ratio'], '+.2f')}; below zero is short, the safe side). "
                  f"The direction it heads for seldom has one ({counts['chosen']['directions']} steps): it mostly "
                  "heads into open water, where the map holds nothing.", "",
                  f"Water directions (the default seed: {bound['water_directions']} of them over the steps): the map "
                  f"vetoed one in {vetoes['steps']} steps, never the mask's own choice "
                  f"({vetoes['on_the_mask_choice']} times); the reference had a surface within 8 steps for "
                  f"{vetoes['right']} of those vetoes and none for {vetoes['wrong']}. The reference had a surface within "
                  f"8 steps in {bound['with_a_reference_surface_within_safe']} water directions in all: nothing to "
                  "veto. The plan bounds them instead (`clear`, 0.75 of the median range of the surfaces in view): "
                  f"known in {percent(s['bounded_share']['median'])} of them, median "
                  f"{number(s['clear_steps']['median'], '.1f')} steps, beyond 8 steps in "
                  f"{percent(s['clear_beyond_safe']['median'])}; the reference was nearer than it in "
                  f"{percent(s['nearer_than_clear']['median'])} [{percent(s['nearer_than_clear']['min'])}–"
                  f"{percent(s['nearer_than_clear']['max'])}].", ""]
    lines += ["Figures: `plan_<rate>hz.png` (the planned direction's corridor over the clip) and "
              "`plan_frames_<rate>hz.png` (example frames with the plan). `board_expected_plan_<rate>hz.json` holds "
              "every step's plan hash and the final one for `uwslam.mvo_stream --expected-plan`.", "",
              "Limits: one clip, open-loop; the goal is straight ahead; the map holds seconds, not the whole dive.", "",
              "Re-run (the output folder must not exist or be empty):", "", "```bash",
              "python -m uwslam.path_plan <video> --reconstruction <folder with images.txt and images.ply> "
              "--output results/lake_clip/path_plan_2", "```", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_board_files(output: Path, result: dict, rate: float) -> None:
    """What the board must report at `rate`: every step's plan (action, direction, turn and the hash of the whole
    plan) and the final plan's hash."""
    expected = {"rate_hz": rate, "every": BOARD_RATES[rate], "frames": len(result["steps"]) + 1,
                "step_hashes": [mvo.fnv(mvo.FNV_OFFSET, bytes(step)) for step in result["steps"]],
                "actions": [p["action"] for p in result["plans"]], "directions": [p["direction"] for p in result["plans"]],
                "turns": [p["turn"] for p in result["plans"]], "plan_hashes": result["plan_hashes"],
                "plan_hash": result["hash"], **sources_sha256()}
    dump_json(output / f"board_expected_plan_{rate:g}hz.json", expected)


def run(video: Path, reconstruction: Path, output: Path, time_scale: float) -> dict:
    import tempfile
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
    cloud = obstacle_dir.load_cloud(reconstruction)
    grid_camera = Camera.from_fov(HFOV_DEG, ASPECT)
    camera = mvo.camera_for(HFOV_DEG, ASPECT)
    output.mkdir(parents=True, exist_ok=True)
    runs, depth_cache = [], {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        lib = mvo.build(Path(scratch))
        corridors = corridor_cells(obstacle_dir.layout_for(lib, camera))
        for rate, stride in RATES.items():
            indices = list(range(0, len(grey), stride))
            frame_masks = [masks[i] if i < len(masks) else None for i in indices]
            for i in indices[1:]:
                if i not in depth_cache:
                    depth_cache[i] = corridor_ranges(obstacle_dir.reference_depth(cloud, rotations[i], centres[i],
                                                                                  grid_camera), corridors, grid_camera)
            truth = np.stack([depth_cache[i] for i in indices[1:]])
            ref_steps = reference_steps(centres, indices)
            scores = []
            for seed in mvo.SEEDS:
                result = run_planning(lib, [grey[i] for i in indices], frame_masks, camera, seed)
                scores.append(score(result, truth, ref_steps))
                if seed == mvo.SEED:
                    draw(output, rate, [grey[i] for i in indices], result, truth)
                    if rate in BOARD_RATES:
                        write_board_files(output, result, rate)
                    default = scores[-1]
            over = {"danger": spread([s["plan"]["danger_go_below_0.25_m"] for s in scores]),
                    "bounded_share": spread([s["water_bound"]["bounded_share"] for s in scores]),
                    "clear_steps": spread([s["water_bound"]["median_steps"] for s in scores]),
                    "clear_beyond_safe": spread([s["water_bound"]["clear_beyond_safe"] for s in scores]),
                    "nearer_than_clear": spread([s["water_bound"]["nearer_than_the_bound"] for s in scores]),
                    "known_within_2x": spread([s["steps_in_steps"]["all_known"]["within_2x"] for s in scores]),
                    "chosen_within_2x": spread([s["steps_in_steps"]["chosen"]["within_2x"] for s in scores]),
                    **{f"action_{name}": spread([s["actions"][name] for s in scores]) for name in ACTIONS}}
            for key in ("plan", "plan_creeping", "mask_choice", "straight_ahead", "best_direction"):
                for near in NEAR:
                    name = f"within_{near:g}_m_free"
                    over[f"{key}_{name}"] = spread([s[key][name] for s in scores])
            runs.append({"rate_hz": rate, "steps": len(indices) - 1, "default_seed": default, "over_seeds": over})
            free = over["plan_within_0.5_m_free"]
            print(f"{rate:g} Hz: planned direction free beyond 0.5 m in {percent(free['median'])} of its go steps "
                  f"(straight ahead {percent(over['straight_ahead_within_0.5_m_free']['median'])}, the mask's choice "
                  f"{percent(over['mask_choice_within_0.5_m_free']['median'])})", flush=True)
    summary = {"task": "Local avoidance and 3D path choice from the voxel map and the water masks on the lake clip, "
                       "scored open-loop against the reference reconstruction",
               "sources_sha256": {**mvo.sources_sha256(), **voxel_map.sources_sha256(), **obstacle_dir.sources_sha256(),
                                  **sources_sha256()},
               "masks": "frames.npz", "time_scale": time_scale, "hfov_deg": HFOV_DEG, "seeds": list(mvo.SEEDS),
               "corridor": {"part": "central half of each direction's rows and columns",
                            "percentile": CORRIDOR_PERCENTILE}, "runs": runs}
    dump_json(output / "summary.json", summary)
    write_report(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=LAKE_CLIP_RESULTS / "path_plan")
    parser.add_argument("--time-scale", type=float, default=3.0)
    args = parser.parse_args()
    run(args.video, args.reconstruction, args.output, args.time_scale)


if __name__ == "__main__":
    main()
