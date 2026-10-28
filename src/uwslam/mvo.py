"""Python binding of the C monocular visual odometry (core/mono_vo*.c), its check on the lake clip, and the files the
board stream test needs.

`compare` runs the C odometry on the lake clip's 320x240 robot frames at 10, 5 and 3 Hz and scores it against the
reference exactly as uwslam.experiments.mono_vo scores the OpenCV prototype (whose summary it reads for the side by
side), once with the default RANSAC seed and then over 16 seeds. It writes pc_summary.json, RESULTS.md and board_expected_10hz.json (each 10 Hz step's FNV-1a hash and counts)
to the output folder and, with --board-frames, the 10 Hz chain's frames as raw 76800-byte records for
`uwslam.mvo_stream`, which sends them to firmware/esp32s3_mono_vo and checks the board's replies.
"""

import argparse
import ctypes
import hashlib
import json
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from uwslam.core import CFLAGS, LIBRARY_SUFFIX, LINK_FLAGS
from uwslam.paths import CORE_DIR, LAKE_CLIP_RESULTS
from uwslam.utils import dump_json

# The odometry's implementation, in the order its fingerprint hashes it (core/mono_vo_internal.h lists the units);
# firmware/esp32s3_mono_vo/main/CMakeLists.txt hashes the same list.
CORE_FILES = ("mono_vo.c", "mono_vo_numerics.c", "mono_vo_tracking.c", "mono_vo_homography.c", "mono_vo_essential.c",
              "mono_vo_internal.h")
SOURCES = [CORE_DIR / name for name in CORE_FILES if name.endswith(".c")]
HEADER = CORE_DIR / "mono_vo.h"
MAP_SOURCE = CORE_DIR / "mono_map.c"
CLOUD_SOURCE = CORE_DIR / "mono_cloud.c"
CLOUD_HEADER = CORE_DIR / "mono_cloud.h"
VOXEL_SOURCE = CORE_DIR / "voxel_map.c"
DIRECTION_SOURCE = CORE_DIR / "obstacle_dir.c"
PLAN_SOURCE = CORE_DIR / "path_plan.c"
WIDTH, HEIGHT = 320, 240
# The build configuration (core/mono_vo.h): "s3", the default, or "p4" (MVO_CONFIG_P4: 300 features), chosen with the
# UWSLAM_CONFIG environment variable for every tool that builds the core.
CONFIGS = {"s3": {"defines": [], "max_features": 250}, "p4": {"defines": ["-DMVO_CONFIG_P4"], "max_features": 300}}
CONFIG = os.environ.get("UWSLAM_CONFIG", "s3")
if CONFIG not in CONFIGS:
    raise ValueError(f"UWSLAM_CONFIG must be one of {', '.join(CONFIGS)}, got {CONFIG!r}")
MAX_FEATURES = CONFIGS[CONFIG]["max_features"]  # MVO_MAX_FEATURES of that configuration
KINDS = {0: "lost", 1: "rotation", 2: "essential"}
SEED = 12345
SEEDS = (SEED, *range(1, 16))  # the seed alone can change the scores twofold: their median and range over 16 runs
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619
RATES = {10.0: 3, 5.0: 6, 3.0: 10}
BOARD_RATES = {10.0: 1, 5.0: 2}  # Hz: every how many of the 10 Hz board frames the board gets


class Camera(ctypes.Structure):
    """mvo_camera_t: pinhole intrinsics of the 320x240 frame and its lens distortion in OpenCV's model and order (all
    zero for a pinhole)."""
    _fields_ = [("fx", ctypes.c_float), ("fy", ctypes.c_float), ("cx", ctypes.c_float), ("cy", ctypes.c_float),
                ("k1", ctypes.c_float), ("k2", ctypes.c_float), ("p1", ctypes.c_float), ("p2", ctypes.c_float),
                ("k3", ctypes.c_float)]


CAMERA_FIELDS = tuple(name for name, _ in Camera._fields_)


def camera_bits(camera: Camera) -> list[int]:
    """The camera's floats as their IEEE-754 bits, in mvo_camera_t's order: what the board's start line carries."""
    return [struct.unpack("<I", struct.pack("<f", getattr(camera, name)))[0] for name in CAMERA_FIELDS]


class StepC(ctypes.Structure):
    _fields_ = [("rotation", ctypes.c_float * 9), ("centre", ctypes.c_float * 3), ("kind", ctypes.c_int32),
                ("tracks", ctypes.c_int32), ("inliers", ctypes.c_int32), ("scale_points", ctypes.c_int32),
                ("world_rotation", ctypes.c_float * 9), ("world_centre", ctypes.c_float * 3)]


def sources_sha256() -> dict:
    """The odometry's fingerprints as the firmware reports them: core_sha256 is the SHA-256 of CORE_FILES back to back
    (`cat` them in that order and hash), header_sha256 that of the public header."""
    return {"core_sha256": hashlib.sha256(b"".join((CORE_DIR / name).read_bytes() for name in CORE_FILES)).hexdigest(),
            "header_sha256": hashlib.sha256(HEADER.read_bytes()).hexdigest()}


def build_config() -> dict:
    """The build configuration, as the board reports it: expectations carry it, so the host can refuse a board built
    another way."""
    return {"config": CONFIG, "max_features": MAX_FEATURES}


def config_defines() -> list[str]:
    """The compiler definitions of the build configuration (CONFIG): every gcc command that compiles the core passes
    them."""
    return list(CONFIGS[CONFIG]["defines"])


def build(output: Path) -> ctypes.CDLL:
    """Compile the VO, obstacle grid, sparse cloud, voxel map, obstacle directions and planner in the build
    configuration (CONFIG) with the board's floating-point flags and bind them."""
    output.mkdir(parents=True, exist_ok=True)
    library = output / f"mono_vo{LIBRARY_SUFFIX}"
    subprocess.run(["gcc", *CFLAGS, *config_defines(), "-shared", *LINK_FLAGS, f"-I{CORE_DIR}",
                    *map(str, SOURCES), str(MAP_SOURCE), str(CLOUD_SOURCE), str(VOXEL_SOURCE), str(DIRECTION_SOURCE),
                    str(PLAN_SOURCE), "-o", str(library)], check=True)
    vo = ctypes.CDLL(str(library))
    vo.mvo_config_name.restype = ctypes.c_char_p
    vo.mvo_max_features.restype = ctypes.c_uint32
    if vo.mvo_config_name().decode() != CONFIG or vo.mvo_max_features() != MAX_FEATURES:
        raise RuntimeError(f"The core built as {vo.mvo_config_name().decode()} with {vo.mvo_max_features()} features, "
                           f"not as {CONFIG} with {MAX_FEATURES}")
    vo.vm_map_size.restype = ctypes.c_uint32
    vo.vm_init.argtypes = [ctypes.c_void_p]
    vo.vm_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(StepC), ctypes.c_void_p, ctypes.c_int,
                             ctypes.c_int]
    vo.vm_update.restype = ctypes.c_int
    vo.vm_key.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    vo.vm_key.restype = ctypes.c_int
    vo.vm_find.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    vo.vm_find.restype = ctypes.c_void_p
    vo.vm_occupied.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    vo.vm_occupied.restype = ctypes.c_int
    vo.od_layout_size.restype = ctypes.c_uint32
    vo.od_state_size.restype = ctypes.c_uint32
    vo.od_layout.argtypes = [ctypes.c_void_p, ctypes.POINTER(Camera), ctypes.c_int, ctypes.c_int]
    vo.od_layout.restype = ctypes.c_int
    vo.od_init.argtypes = [ctypes.c_void_p]
    vo.od_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    vo.od_update.restype = ctypes.c_int
    vo.od_direction_of.argtypes = [ctypes.c_float, ctypes.c_float]
    vo.od_direction_of.restype = ctypes.c_int
    vo.pp_plan_size.restype = ctypes.c_uint32
    vo.pp_init.argtypes = [ctypes.c_void_p]
    vo.pp_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p]
    vo.pp_update.restype = ctypes.c_int
    vo.mm_map_size.restype = ctypes.c_uint32
    vo.mm_init.argtypes = [ctypes.c_void_p]
    vo.mm_observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    vo.mm_observe.restype = ctypes.c_int
    vo.mm_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    vo.mm_add.restype = ctypes.c_int
    vo.mc_map_size.restype = ctypes.c_uint32
    vo.mc_init.argtypes = [ctypes.c_void_p]
    vo.mc_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(StepC)]
    vo.mc_update.restype = ctypes.c_int
    vo.mvo_state_size.restype = ctypes.c_uint32
    vo.mvo_pyramid_bytes.restype = ctypes.c_uint32
    vo.mvo_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(Camera), ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
    vo.mvo_frame.argtypes = [ctypes.c_void_p]
    vo.mvo_frame.restype = ctypes.POINTER(ctypes.c_uint8)
    vo.mvo_start.argtypes = [ctypes.c_void_p]
    vo.mvo_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(StepC)]
    vo.mvo_tracks.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    vo.mvo_tracks.restype = ctypes.c_int
    vo.mvo_step_length.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    vo.mvo_step_length.restype = ctypes.c_float
    vo.mvo_sqrt.argtypes = [ctypes.c_float]
    vo.mvo_sqrt.restype = ctypes.c_float
    vo.mvo_lk_template_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    vo.mvo_lk_sums_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    vo.mvo_block_c.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return vo


def fnv(hash_value: int, data: bytes) -> int:
    for byte in data:
        hash_value = ((hash_value ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return hash_value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_30fps(fps: float) -> None:
    if not np.isfinite(fps) or abs(fps - 30.0) > 0.001:
        raise ValueError(f"This experiment uses strides for 30 FPS, got {fps:g} FPS")


def camera_for(hfov_deg: float, aspect: float) -> Camera:
    """Intrinsics of the 320x240 frame, as uwslam.experiments.mask_motion.Camera.from_fov gives them."""
    half = float(np.tan(np.radians(hfov_deg) / 2))
    return Camera(WIDTH / 2 / half, HEIGHT / 2 / (aspect * half), WIDTH / 2, HEIGHT / 2)


def run_frames(vo: ctypes.CDLL, frames: list[np.ndarray], camera: Camera, seed: int = SEED) -> tuple[list[StepC], list[int], float]:
    """One run over the frames (the first starts it): the steps, each step's FNV-1a hash and the PC's ms per step."""
    state = ctypes.create_string_buffer(vo.mvo_state_size())
    pyramids = [ctypes.create_string_buffer(vo.mvo_pyramid_bytes()) for _ in range(2)]  # alive as long as state
    vo.mvo_init(state, ctypes.byref(camera), seed, pyramids[0], pyramids[1])

    def load(frame: np.ndarray) -> None:
        if frame.shape != (HEIGHT, WIDTH) or frame.dtype != np.uint8:
            raise ValueError(f"Expected a {HEIGHT}x{WIDTH} uint8 frame, got {frame.shape} {frame.dtype}")
        pixels = np.ascontiguousarray(frame)
        ctypes.memmove(vo.mvo_frame(state), pixels.ctypes.data, WIDTH * HEIGHT)

    load(frames[0])
    vo.mvo_start(state)
    steps, hashes = [], []
    started = time.perf_counter()
    for frame in frames[1:]:
        load(frame)
        step = StepC()
        vo.mvo_step(state, ctypes.byref(step))
        steps.append(step)
        hashes.append(fnv(FNV_OFFSET, bytes(step)))
    return steps, hashes, (time.perf_counter() - started) * 1000 / max(1, len(steps))


def as_steps(steps: list[StepC]):
    """The C steps as uwslam.experiments.mono_vo.Step records, and the camera-to-world rotations and centres."""
    from uwslam.experiments.mono_vo import Step
    records = [Step(np.array(s.rotation, dtype=np.float64).reshape(3, 3), np.array(s.centre, dtype=np.float64),
                    KINDS[s.kind], s.tracks, s.inliers, s.scale_points) for s in steps]
    rotations = np.stack([np.eye(3)] + [np.array(s.world_rotation, dtype=np.float64).reshape(3, 3) for s in steps])
    centres = np.stack([np.zeros(3)] + [np.array(s.world_centre, dtype=np.float64) for s in steps])
    return records, rotations, centres


def spread(values: list[float]) -> dict:
    return {"median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def over_seeds(vo: ctypes.CDLL, frames: list[np.ndarray], indices: list[int], camera: Camera, times: np.ndarray,
               reference_rotations: np.ndarray, reference_centres: np.ndarray) -> dict:
    """Median and range over SEEDS of the scores, scored as uwslam.experiments.mono_vo.evaluate scores one run; the
    scale spread is the largest 5 s window ratio over the smallest."""
    from uwslam.experiments.mono_vo import evaluate
    scores = {"ate_rmse": [], "heading_rmse_deg": [], "scale_spread": [], "lost": []}
    for seed in SEEDS:
        records, rotations, centres = as_steps(run_frames(vo, frames, camera, seed)[0])
        result = evaluate(indices, records, rotations, centres, times, reference_rotations, reference_centres)
        scores["ate_rmse"].append(result["ate_rmse"])
        scores["heading_rmse_deg"].append(result["heading_rmse_deg"])
        scores["scale_spread"].append(result["scale_drift_window_ratio_max"] / result["scale_drift_window_ratio_min"])
        scores["lost"].append(result["lost"])
    return {"seeds": list(SEEDS), **{key: spread(values) for key, values in scores.items()}}


def write_report(output: Path, summary: dict) -> None:
    lines = ["# Monocular visual odometry in C (core/mono_vo*.c) on the lake clip", "",
             "The C port of the *pairwise* method runs on the robot camera's 320×240 grey frames and is scored exactly as "
             "the OpenCV prototype in `results/lake_clip/mono_vo` (similarity alignment for positions; rotation, direction of "
             "travel and heading directly). It follows the prototype's design with its own numerics: its own pyramid "
             "and fixed-point Lucas-Kanade, and instead of OpenCV's five-point solver an eight-point MSAC with "
             "least-squares refits, a search over travel directions and a robust Levenberg-Marquardt, pairs behind a "
             "camera counting as outliers (see core/mono_vo.h). Five changes in design. The prototype stops tracking the "
             "pairs that recoverPose drops as farther than 50 baselines, while the C port keeps every epipolar inlier "
             "tracked and gives only the nearer ones a point: on short steps the far pairs were most of the features, and "
             "dropping them broke the chain of shared depths that carries the scale. The prototype takes each step's "
             "median depth ratio as it comes, while the C port keeps it within 3 times the median of the last 10 step "
             "lengths (and takes that median when a step shares fewer than 8 points): on a fast turn a median from a few "
             "points could be far off, and the chain of shared depths then kept the wrong scale. The prototype takes "
             "every essential matrix's translation, while the C port takes none where the pairs give it no evidence "
             "over a separately fitted rotation: such a step counts as rotation only when that rotation has 15 tracks "
             "and half of them, else as lost. A step lost on a fast turn keeps the last pose in the prototype, while in "
             "the C port it keeps a turn when at least 8 tracks and half of them agree on one (at 5 Hz the lost steps "
             "turn by 11 degrees in the median, all of which the heading lost): the fitted rotation's, and where only "
             "the homography has that support, the essential pose's when it had 15 inliers, else the homography's. A "
             "homography the fitted rotation does not share has mostly read the travel's parallax as a turn, over the "
             "lake's floor as pitch: on such lost steps at 5 Hz it errs +1.5 degrees in pitch a step against -0.2 for "
             "the essential pose, though by as much in all (6.0 and 6.2 degrees), and that bias made the path climb. "
             "Any lost step whose essential pose had 15 inliers carries the chain of shared depths on with that pose, "
             "still without its translation, when at least 8 depth ratios it shares with the earlier points agree on "
             "one scale (their interquartile range at most half their median): dropping every point at each of the 11 "
             "lost steps a run restarted the scale from the recent speed each time. "
             "Last, where the homography and the essential matrix both turn a step by more than 5 "
             "degrees, the C port pulls the essential rotation's angle about the homography's turn axis towards the "
             "homography's: on a fast turn two views trade the turn against sideways travel for a fraction of a pixel, "
             "and the essential matrix took up to a quarter of each fast step as travel, 14 degrees over the clip's "
             "two-second swing at 5 Hz, which one long-baseline estimate across the swing does not lose. Only along "
             "that axis: pulling the whole rotation also dragged in the homography's tilt, which the floor's parallax "
             "biases, 10 degrees more pitch over the clip. At 10 Hz the steps are half as large and the pull mends the "
             "swing's first half more than its second, so the heading over the whole clip comes out worse there than "
             "without it.", "",
             "Each row of the first table is one run with one RANSAC seed. On this clip the seed alone can change the "
             "position and heading errors by a factor of two or more, so a difference between two rows is not "
             "significant on its own; the second table gives the C port's median and range over 16 seeds.", "",
             "| Rate | Version | Steps (lost / rotation only) | Position RMSE after alignment | Heading RMSE / max | "
             "Direction of travel, median | Scale ratio over 5 s windows | ms per step (PC) |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["runs"]:
        for label, entry in (("C", row["c"]), ("OpenCV prototype", row.get("prototype"))):
            if entry is None:
                continue
            ms = entry.get("ms_per_frame")
            lines.append(f"| {row['rate_hz']:g} Hz | {label} | {entry['steps']} ({entry['lost']} / {entry['rotation_only']}) | "
                         f"{entry['ate_rmse']:.2f} m | {entry['heading_rmse_deg']:.1f}° / {entry['heading_max_deg']:.0f}° | "
                         f"{entry['direction_median_deg']:.1f}° | {entry['scale_drift_window_ratio_min']:.2f}–"
                         f"{entry['scale_drift_window_ratio_max']:.2f} | {'—' if ms is None else f'{ms:.1f}'} |")
    lines += ["", f"C port over {len(SEEDS)} RANSAC seeds, median [smallest–largest]:", "",
              "| Rate | Position RMSE after alignment | Heading RMSE | Scale ratio spread over 5 s windows (largest / "
              "smallest) | Lost steps |", "|---:|---:|---:|---:|---:|"]
    for row in summary["runs"]:
        s = row["c_over_seeds"]
        lines.append(f"| {row['rate_hz']:g} Hz | {s['ate_rmse']['median']:.2f} m [{s['ate_rmse']['min']:.2f}–"
                     f"{s['ate_rmse']['max']:.2f}] | {s['heading_rmse_deg']['median']:.1f}° [{s['heading_rmse_deg']['min']:.0f}–"
                     f"{s['heading_rmse_deg']['max']:.0f}] | {s['scale_spread']['median']:.1f}× [{s['scale_spread']['min']:.1f}–"
                     f"{s['scale_spread']['max']:.1f}] | {s['lost']['median']:g} [{s['lost']['min']:g}–{s['lost']['max']:g}] |")
    lines += ["", f"`board_expected_10hz.json` holds the FNV-1a hash of every 10 Hz step ({summary['board']['steps']} steps) for "
              "the board stream test (uwslam.mvo_stream).", "",
              "Re-run (the output folder must not exist or be empty):", "", "```bash",
              "python -m uwslam.mvo compare <video> --reconstruction <folder with images.txt> --output results/lake_clip/mono_vo_c_2 "
              "--board-frames <file for the board's frames>", "```", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def compare(video: Path, reconstruction: Path, output: Path, board_frames: Path | None, time_scale: float) -> dict:
    from uwslam.experiments.mask_motion import reference_poses
    from uwslam.experiments.mono_vo import ASPECT, HFOV_DEG, evaluate, read_frames
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite results; choose a fresh --output")
    if board_frames is not None and board_frames.exists():
        raise FileExistsError("Refusing to overwrite board frames; choose a fresh --board-frames")
    poses = np.loadtxt(reconstruction / "images.txt")
    frames, fps, _ = read_frames(video, poses[-1, 0] * time_scale)
    require_30fps(fps)
    grey = frames["320x240"]
    times = np.arange(len(grey)) / fps
    reference_rotations, reference_centres = reference_poses(poses, times, time_scale)
    camera = camera_for(HFOV_DEG, ASPECT)
    prototype_path = LAKE_CLIP_RESULTS / "mono_vo/summary.json"
    prototype = json.loads(prototype_path.read_text(encoding="utf-8")) if prototype_path.exists() else {"runs": []}
    source = prototype.get("source", {})
    if (source.get("sha256") != file_sha256(video)
            or source.get("poses_sha256") != file_sha256(reconstruction / "images.txt")
            or prototype.get("time_scale") != time_scale):
        prototype = {"runs": []}
    runs, boards, state_bytes, pyramid_bytes = [], {}, 0, 0
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:  # Windows keeps the loaded library locked
        vo = build(Path(scratch))
        state_bytes, pyramid_bytes = int(vo.mvo_state_size()), int(vo.mvo_pyramid_bytes())
        for rate, stride in RATES.items():
            indices = list(range(0, len(grey), stride))
            steps, hashes, ms = run_frames(vo, [grey[i] for i in indices], camera)
            records, rotations, centres = as_steps(steps)
            result = evaluate(indices, records, rotations, centres, times, reference_rotations, reference_centres)
            result.pop("aligned")
            match = [r for r in prototype["runs"] if r.get("method", "pairwise") == "pairwise"
                     and r["resolution"] == "320x240" and r["rate_hz"] == rate]
            seeds = over_seeds(vo, [grey[i] for i in indices], indices, camera, times, reference_rotations,
                               reference_centres)
            runs.append({"rate_hz": rate, "c": {**result, "ms_per_frame": ms}, "c_over_seeds": seeds,
                         "prototype": match[0] if match else None})
            print(f"C {rate:g} Hz: ATE {result['ate_rmse']:.2f}, heading RMSE {result['heading_rmse_deg']:.1f} deg, "
                  f"lost {result['lost']}/{result['steps']}, {ms:.1f} ms per step; over {len(SEEDS)} seeds median ATE "
                  f"{seeds['ate_rmse']['median']:.2f} [{seeds['ate_rmse']['min']:.2f}-{seeds['ate_rmse']['max']:.2f}]",
                  flush=True)
            if rate in BOARD_RATES:
                boards[rate] = {"rate_hz": rate, "stride": stride, "every": BOARD_RATES[rate], "frames": len(indices),
                                "steps": len(steps), "camera": [getattr(camera, name) for name in CAMERA_FIELDS],
                                "seed": SEED, "camera_bits": camera_bits(camera),
                                "step_hashes": hashes, "kinds": [KINDS[s.kind] for s in steps],
                                "final_world_centre": list(steps[-1].world_centre), **sources_sha256(),
                                **build_config()}
                if board_frames is not None and BOARD_RATES[rate] == 1:
                    board_frames.parent.mkdir(parents=True, exist_ok=True)
                    board_frames.write_bytes(b"".join(np.ascontiguousarray(grey[i]).tobytes() for i in indices))
    output.mkdir(parents=True, exist_ok=True)
    for rate, board in boards.items():
        dump_json(output / f"board_expected_{rate:g}hz.json", board)
    board = boards[10.0]
    summary = {"task": "C monocular visual odometry (pairwise method) on the lake clip's 320x240 robot frames, scored "
                       "against the reference poses and set beside the OpenCV prototype",
               "core_sha256": board["core_sha256"], **build_config(), "state_bytes": state_bytes,
               "pyramid_bytes_each": pyramid_bytes,
               "time_scale": time_scale, "hfov_deg": HFOV_DEG, "runs": runs,
               "board": {key: board[key] for key in ("rate_hz", "frames", "steps", "seed", "core_sha256", "header_sha256")}}
    dump_json(output / "pc_summary.json", summary)
    write_report(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("compare", help="Score the C odometry on the lake clip and write the board's files")
    run.add_argument("video", type=Path)
    run.add_argument("--reconstruction", type=Path, required=True)
    run.add_argument("--output", type=Path, default=LAKE_CLIP_RESULTS / "mono_vo_c")
    run.add_argument("--board-frames", type=Path, help="Where to write the 10 Hz frames the board test streams")
    run.add_argument("--time-scale", type=float, default=3.0)
    args = parser.parse_args()
    compare(args.video, args.reconstruction, args.output, args.board_frames, args.time_scale)


if __name__ == "__main__":
    main()
