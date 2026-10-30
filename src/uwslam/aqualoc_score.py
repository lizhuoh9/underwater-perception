"""The odometry on an AQUALOC sequence prepared by uwslam.aqualoc, with and without the dataset's IMU: 10 s windows
every 5 s and the whole sequence, every seed of uwslam.mvo.SEEDS, each run paired with a base mode on the same window
and seed.

Modes: `off` (the camera alone); `gyro` (the gyro's readings, the bias learnt online from zero at the run's start);
`warm` (windows only: the bias starts from the estimate the whole-sequence gyro run of the same seed held after the step
before the window, what a robot running all along would have); `imu` (the gyro and the accelerometer, its bias
calibrated: the mean of the biases fitted to the other sequences given, standing in for a calibration on another
day; not run with one sequence); `imu_raw` (the accelerometer uncalibrated); `depth` (imu and the depth sensor at
each frame: the metric scale, mvo_depth_reading). gyro and warm pair with off, the imu modes with gyro, depth with
imu. Each run is scored as uwslam.experiments.mono_vo scores the lake clip (ATE after a similarity
alignment, its heading, lost steps), plus scores from the run's first frame, where estimate and reference coincide:
the yaw about the reference's vertical, the tilt (the angle between the estimated and the reference's up in the
camera), the whole orientation error (the angle of R_est^T R_ref), and the ATE after an alignment that turns only about
the reference's vertical (scale, that turn and an offset: a path that leans is not turned back), with its vertical
part, the height error. With depth, the metric ATE: the world centre in metres (mvo_metric; before the scale is
known, the path the first scale gives) after an alignment that turns only about the reference's vertical and
shifts, no scale (the reference is metric through the same depth sensor); none when the scale is never known.

  python -m uwslam.aqualoc_score <prepared folder> [<prepared folder> ...] --output <fresh folder> [--jobs <processes>]
"""

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uwslam import mvo
from uwslam.utils import dump_json

WINDOW_FRAMES, WINDOW_EVERY = 50, 25  # 10 s windows every 5 s at 5 Hz
METRICS = (
    "ate_rmse",
    "levelled_ate_rmse",
    "metric_ate_rmse",
    "height_rmse",
    "yaw_rmse_deg",
    "tilt_rmse_deg",
    "orientation_rmse_deg",
    "heading_rmse_deg",
    "lost",
)
LABELS = (
    "ATE (m)",
    "levelled ATE (m)",
    "metric ATE (m)",
    "height (m)",
    "yaw RMSE (°)",
    "tilt RMSE (°)",
    "orientation RMSE (°)",
    "heading RMSE (°)",
    "lost steps",
)
DIGITS = (3, 3, 3, 3, 2, 2, 2, 2, 0)
BASES = {
    "gyro": "off",
    "warm": "off",
    "imu": "gyro",
    "imu_raw": "gyro",
    "depth": "imu",
}  # what each mode is paired with
USES = ("camera", "gyro_translation", "gyro_rotation", "gyro_lost")  # MVO_GYRO_*, the first as the camera's own step


@dataclass(frozen=True)
class Prepared:
    """A prepared sequence (uwslam.aqualoc prepare)."""

    name: str
    frames: np.ndarray
    camera: "mvo.Camera"
    stamps: np.ndarray
    reference_rotations: np.ndarray
    reference_centres: np.ndarray
    readings: list[np.ndarray]
    forces: list[np.ndarray]
    mount: np.ndarray
    vertical: np.ndarray
    accel_bias_fit: np.ndarray
    depths: np.ndarray | None = None  # the depth sensor at each frame (m), when prepared with it

    def imu_readings(self, first: int = 0, end: int | None = None) -> list[np.ndarray]:
        """The steps' IMU readings, float32 rows of (dt, rate x, y, z, force x, y, z) (uwslam.mvo.GyroReadings)."""
        return [
            np.column_stack([r, f]).astype(np.float32) for r, f in zip(self.readings[first:end], self.forces[first:end])
        ]


def load_prepared(folder: Path) -> Prepared:
    with np.load(folder / "sequence.npz") as archive:  # closed at once: Windows keeps an open file locked
        data = {name: archive[name] for name in archive.files}
    frames = np.load(folder / "frames_5hz.npy")
    if frames.ndim != 3 or frames.shape[1:] != (mvo.HEIGHT, mvo.WIDTH) or frames.dtype != np.uint8:
        raise ValueError(f"{folder.name}: frames_5hz.npy must hold 320x240 grey frames")
    if len(frames) != len(data["stamps"]) or len(data["reading_offsets"]) != len(frames):
        raise ValueError(f"{folder.name}: frames, stamps and readings disagree")
    if "forces" not in data or data["forces"].shape != (len(data["readings"]), 3):
        raise ValueError(f"{folder.name}: no accelerometer readings; prepare it again with uwslam.aqualoc")
    offsets, rows, forces = data["reading_offsets"], data["readings"], data["forces"]
    values = json.loads((folder / "camera.json").read_text(encoding="utf-8"))["camera"]
    return Prepared(
        folder.name,
        frames,
        mvo.Camera(*(float(values[name]) for name in mvo.CAMERA_FIELDS)),
        data["stamps"],
        data["reference_rotations"],
        data["reference_centres"],
        [rows[offsets[k] : offsets[k + 1]] for k in range(len(offsets) - 1)],
        [forces[offsets[k] : offsets[k + 1]] for k in range(len(offsets) - 1)],
        data["mount"],
        data["vertical"],
        data["accel_bias_fit"],
        data.get("depths"),
    )


def angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def yaw_about(rotations: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Unwrapped turn (radians) of camera-to-first rotations about `axis` (the first camera's axes): each camera's x
    axis projected on the plane normal to it, against the first camera's."""
    axis = axis / np.linalg.norm(axis)
    x = rotations[:, :, 0] - np.outer(rotations[:, :, 0] @ axis, axis)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.unwrap(np.arctan2(np.cross(x[0], x) @ axis, x @ x[0]))


def turn_about(x: np.ndarray, y: np.ndarray, up: np.ndarray) -> np.ndarray:
    """The turn about `up` (unit) that best brings the centred rows x onto y: the vertical parts take no part in it,
    which maximises the horizontal parts' agreement."""
    xh, yh = x - np.outer(x @ up, up), y - np.outer(y @ up, up)
    angle = np.arctan2(np.sum(np.cross(xh, yh) @ up), np.sum(xh * yh))
    k = np.array([[0.0, -up[2], up[1]], [up[2], 0.0, -up[0]], [-up[1], up[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def levelled(centres: np.ndarray, reference: np.ndarray, up: np.ndarray) -> np.ndarray:
    """The centres after the similarity alignment onto the reference that turns only about `up` (unit)."""
    x, y = centres - centres.mean(axis=0), reference - reference.mean(axis=0)
    xr = x @ turn_about(x, y, up).T
    size = np.sum(xr * xr)
    scale = np.sum(xr * y) / size if size > 0 else 0.0
    return scale * xr + reference.mean(axis=0)


def metric_ate(path: np.ndarray, reference: np.ndarray, up: np.ndarray) -> float:
    """RMSE after the best turn about `up` (unit) and shift, no scale."""
    x, y = path - path.mean(axis=0), reference - reference.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((x @ turn_about(x, y, up).T - y) ** 2, axis=1))))


def score_run(sequence: Prepared, first: int, run: "mvo.Run") -> dict:
    """evaluate's scores plus yaw and orientation for a run over the frames from `first`, the reference re-expressed
    from that frame."""
    from uwslam.experiments.mono_vo import evaluate

    count = len(run.steps) + 1
    ref_r = sequence.reference_rotations[first : first + count]
    ref_c = sequence.reference_centres[first : first + count]
    window_r = ref_r[0].T @ ref_r
    window_c = (ref_c - ref_c[0]) @ ref_r[0]
    records, rotations, centres = mvo.as_steps(run.steps)
    indices = list(range(count))
    stamps = sequence.stamps[first : first + count]
    result = evaluate(indices, records, rotations, centres, stamps, window_r, window_c)
    vertical = ref_r[0].T @ sequence.vertical
    yaw = yaw_about(rotations, vertical) - yaw_about(window_r, vertical)
    yaw = np.degrees(np.abs((yaw + np.pi) % (2 * np.pi) - np.pi))
    orientation = np.array([angle_deg(a.T @ b) for a, b in zip(rotations, window_r)])
    est_up, ref_up = rotations.transpose(0, 2, 1) @ vertical, window_r.transpose(0, 2, 1) @ vertical
    tilt = np.degrees(np.arccos(np.clip(np.sum(est_up * ref_up, axis=1), -1.0, 1.0)))
    error = levelled(centres, window_c, vertical) - window_c
    metric = mvo.metric_path(run)
    used = run.gyro_used or [0] * len(run.steps)
    return {
        "first": first,
        **{key: result[key] for key in ("ate_rmse", "heading_rmse_deg", "lost", "kinds")},
        "levelled_ate_rmse": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "metric_ate_rmse": None if metric is None else metric_ate(metric, window_c, -vertical),
        "scale_known": None if run.metric is None else run.metric[-1] is not None,
        "height_rmse": float(np.sqrt(np.mean((error @ vertical) ** 2))),
        "yaw_rmse_deg": float(np.sqrt(np.mean(yaw**2))),
        "tilt_rmse_deg": float(np.sqrt(np.mean(tilt**2))),
        "orientation_rmse_deg": float(np.sqrt(np.mean(orientation**2))),
        "uses": {name: used.count(code) for code, name in enumerate(USES)},
    }


def readings_for(sequence: Prepared, mode: str, accel_bias, first: int = 0, end: int | None = None, bias=None):
    """The mode's readings for the steps first..end (None for off)."""
    if mode == "off":
        return None
    if mode in ("gyro", "warm"):
        return mvo.GyroReadings(sequence.mount, sequence.readings[first:end], bias)
    return mvo.GyroReadings(
        sequence.mount, sequence.imu_readings(first, end), None, accel_bias if mode in ("imu", "depth") else None
    )


def run_mode(
    vo, sequence: Prepared, frames: list, seed: int, mode: str, accel_bias, first: int = 0, end=None, bias=None
):
    """One run of the mode over the frames first..end (end exclusive, None: to the last)."""
    readings = readings_for(sequence, mode, accel_bias, first, None if end is None else end - 1, bias)
    depths = sequence.depths[first:end] if mode == "depth" else None
    return mvo.run_odometry(vo, frames, sequence.camera, seed, readings, depths)


def score_seed(vo, sequence: Prepared, seed: int, accel_bias: np.ndarray | None = None) -> dict:
    """One seed's runs on the sequence, keyed by command and mode: the whole sequence, then every window. The imu mode
    runs only with an accelerometer bias to calibrate it."""
    n = len(sequence.frames)
    spans = [(s, s + WINDOW_FRAMES) for s in range(0, n - WINDOW_FRAMES + 1, WINDOW_EVERY)]
    imu_modes = ("imu", "imu_raw") if accel_bias is not None else ("imu_raw",)
    if accel_bias is not None and sequence.depths is not None:
        imu_modes = (*imu_modes, "depth")
    full_modes, window_modes = ("off", "gyro", *imu_modes), ("off", "gyro", "warm", *imu_modes)
    out = {"full": {m: [] for m in full_modes}, "windows": {m: [] for m in window_modes}}
    frames = list(sequence.frames)
    for mode in full_modes:
        run = run_mode(vo, sequence, frames, seed, mode, accel_bias)
        out["full"][mode].append({"seed": seed, **score_run(sequence, 0, run)})
        if mode != "off":
            out["full"][mode][-1]["final_bias_dps"] = np.degrees(run.gyro_bias[-1]).tolist()
        if mode == "gyro":
            carried = run.gyro_bias
    for first, end in spans:
        window = list(sequence.frames[first:end])
        for mode in window_modes:
            bias = np.array(carried[first - 1], dtype=np.float32) if mode == "warm" and first > 0 else None
            run = run_mode(vo, sequence, window, seed, mode, accel_bias, first, end, bias)
            out["windows"][mode].append({"seed": seed, **score_run(sequence, first, run)})
    return out


def merge(per_seed: list[dict]) -> dict:
    """The seeds' runs in seed order, under each command and mode."""
    return {
        command: {mode: [run for seed in per_seed for run in seed[command][mode]] for mode in modes}
        for command, modes in per_seed[0].items()
    }


def score_sequence(vo, sequence: Prepared, accel_bias: np.ndarray | None) -> dict:
    """Every seed's runs on the sequence, one after the other."""
    per_seed = []
    for seed in mvo.SEEDS:
        per_seed.append(score_seed(vo, sequence, seed, accel_bias))
        print(f"{sequence.name} seed {seed} scored", flush=True)
    return merge(per_seed)


def fitted_accel_bias(folder: Path) -> np.ndarray:
    """The accelerometer bias uwslam.aqualoc fitted to the prepared sequence's reference, without loading its frames."""
    with np.load(folder / "sequence.npz") as archive:
        return archive["accel_bias_fit"]


def calibrations(folders: list[Path]) -> dict:
    """Each sequence's accelerometer calibration: the mean of the biases fitted to the other sequences, or None."""
    fitted = {f.name: fitted_accel_bias(f) for f in folders}
    return {
        name: np.mean([b for other, b in fitted.items() if other != name], axis=0) if len(fitted) > 1 else None
        for name in fitted
    }


WORKER = {}  # a worker process's core and the sequences it has loaded


def start_worker(scratch: str) -> None:
    WORKER["vo"] = mvo.build(Path(scratch) / f"worker{os.getpid()}")


def worker_score(folder: str, seed: int, accel_bias: np.ndarray | None) -> dict:
    if folder not in WORKER:
        WORKER[folder] = load_prepared(Path(folder))
    return score_seed(WORKER["vo"], WORKER[folder], seed, accel_bias)


def score_all(folders: list[Path], jobs: int) -> dict:
    """Every sequence's runs, the seeds spread over `jobs` processes (each with its own build of the core)."""
    accel = calibrations(folders)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:  # Windows keeps the loaded library locked
        if jobs <= 1:
            vo = mvo.build(Path(scratch))
            return {f.name: score_sequence(vo, load_prepared(f), accel[f.name]) for f in folders}
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(jobs, initializer=start_worker, initargs=(scratch,)) as pool:
            futures = {
                (f.name, seed): pool.submit(worker_score, str(f), seed, accel[f.name])
                for f in folders
                for seed in mvo.SEEDS
            }
            done = {key: future.result() for key, future in futures.items()}
        return {f.name: merge([done[(f.name, seed)] for seed in mvo.SEEDS]) for f in folders}


def paired(runs: list[dict], base: list[dict] | None, metric: str) -> dict | None:
    """The metric's median over the runs that have it (None when none has: the metric ATE without depth; a NaN
    shows as one) and, with a base that has it too, the shares of the pairs in which the run is lower and in which
    it is equal."""
    has = np.array([r.get(metric) is not None for r in runs], dtype=bool)
    if not has.any():
        return None
    values = np.array([r[metric] if h else np.nan for r, h in zip(runs, has)], dtype=float)
    out = {"median": float(np.median(values[has])), "runs": int(has.sum())}
    if base is not None:
        if [(r["seed"], r["first"]) for r in runs] != [(r["seed"], r["first"]) for r in base]:
            raise ValueError("Paired runs must be the same windows and seeds")
        both = has & np.array([r.get(metric) is not None for r in base], dtype=bool)
        others = np.array([r[metric] if b else np.nan for r, b in zip(base, both)], dtype=float)
        if both.any():
            out["lower_share"] = float(np.mean(values[both] < others[both]))
            out["equal_share"] = float(np.mean(values[both] == others[both]))
    return out


def mode_row(mode: str, runs: list[dict], base: list[dict] | None) -> dict:
    """A mode's row: its runs, its base's name and each metric paired with the base's runs."""
    return {
        "runs": len(runs),
        **({"base": BASES[mode]} if mode in BASES else {}),
        **{m: paired(runs, base, m) for m in METRICS},
    }


def tables(scores: dict) -> dict:
    """Per sequence (and all together), command and mode: each metric's median and the share of the pairs in which the
    mode is lower than its base (BASES). All together pools a mode's runs over the sequences that have it (the depth
    mode only those prepared with their depths), and its base's over the same sequences."""
    groups = list(scores.items())
    if len(scores) > 1:
        groups.append(("all", None))
    out = {}
    for name, runs in groups:
        out[name] = {}
        for command in next(iter(scores.values())):
            having = list(scores.values()) if runs is None else [runs]
            modes = dict.fromkeys(mode for one in having for mode in one[command])  # in order, each once
            rows = {}
            for mode in modes:
                with_mode = [one[command] for one in having if one[command].get(mode)]
                pooled = [r for by_mode in with_mode for r in by_mode[mode]]
                base = [r for by_mode in with_mode for r in by_mode[BASES[mode]]] if mode in BASES else None
                rows[mode] = mode_row(mode, pooled, base)
            if rows.get("off", {}).get("runs"):  # a sequence shorter than a window has no windows
                out[name][command] = rows
    return out


def write_report(output: Path, summary: dict) -> None:
    lines = [
        "# The odometry with and without the IMU on AQUALOC harbour sequences",
        "",
        f"5 Hz, {len(mvo.SEEDS)} seeds; 10 s windows every 5 s and whole sequences. Each cell is the median over the "
        "runs and, for a mode with a base, the share of the runs paired with the base (same window, same seed) in "
        "which it is lower: `gyro` and `warm` against `off`, `imu` and `imu_raw` against `gyro`. `gyro` learns the "
        "bias from zero at the run's start; `warm` starts each window from the whole-sequence run's estimate at that "
        "point; `imu` adds the accelerometer, its bias calibrated with the other sequences' fitted one; `imu_raw` "
        "adds it uncalibrated; `depth` adds the depth sensor to `imu`, and its metric ATE scores the path in metres (no "
        "scale fitted; [n runs] where the scale was known in fewer than all; its other scores are the odometry's, "
        "(same) as `imu`'s in every pair). The camera looks down at the seabed: yaw is about the vertical, tilt the "
        "angle between the estimated and the reference's up, orientation the whole rotation error, heading the turn "
        "about the first "
        "camera's y axis (a tilt here); the levelled ATE turns the path only about the vertical, and height is its "
        "vertical part.",
        "",
    ]
    for command, title in (("windows", "10 s windows"), ("full", "Whole sequences")):
        lines += [f"## {title}", "", "| Sequence | Mode | Runs | " + " | ".join(LABELS) + " |"]
        lines.append("|---|---|---:|" + "---:|" * len(LABELS))
        for name, by_command in summary["tables"].items():
            for mode, row in by_command.get(command, {}).items():
                cells = []
                for metric, digits in zip(METRICS, DIGITS):
                    if row[metric] is None:
                        cells.append("–")
                        continue
                    cell = f"{row[metric]['median']:.{digits}f}"
                    if row[metric].get("equal_share") == 1.0:
                        cell += " (same)"
                    elif "lower_share" in row[metric]:
                        cell += f" ({row[metric]['lower_share'] * 100:.0f}%)"
                    if row[metric]["runs"] < row["runs"]:
                        cell += f" [{row[metric]['runs']} runs]"
                    cells.append(cell)
                lines.append(f"| {name} | {mode} | {row['runs']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## The accelerometer's calibration (m/s², the IMU's axes)", ""]
    for name, bias in summary["accel_bias_calibration"].items():
        fitted = summary["accel_bias_fit"][name]
        lines.append(
            f"- {name}: fitted to its reference "
            + ", ".join(f"{axis} {v:+.3f}" for axis, v in zip("xyz", fitted))
            + "; calibrated with "
            + ("none" if bias is None else ", ".join(f"{axis} {v:+.3f}" for axis, v in zip("xyz", bias)))
        )
    lines += ["", "## The bias the whole-sequence gyro runs learnt (deg/s, the IMU's axes)", ""]
    for name, biases in summary["final_bias_dps"].items():
        b = np.array(biases)
        lines.append(
            f"- {name}: median "
            + ", ".join(f"{axis} {np.median(b[:, k]):+.3f}" for k, axis in enumerate("xyz"))
            + f" over {len(b)} seeds; fitted to the reference: "
            + ", ".join(f"{axis} {v:+.3f}" for axis, v in zip("xyz", summary["fitted_bias_dps"][name]))
        )
    lines += [
        "",
        "Re-run (the output folder must not exist or be empty):",
        "",
        "```bash",
        "python -m uwslam.aqualoc_score <prepared seq01> <prepared seq07> --output <fresh folder>",
        "```",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prepared", type=Path, nargs="+", help="Folders from python -m uwslam.aqualoc prepare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="Processes to spread the seeds over")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite results; choose a fresh --output")
    for folder in args.prepared:
        load_prepared(folder)  # refuse a broken folder before any run
    scores = score_all(args.prepared, args.jobs)
    summary = {
        **mvo.sources_sha256(),
        **mvo.build_config(),
        "seeds": list(mvo.SEEDS),
        "window_frames": WINDOW_FRAMES,
        "window_every": WINDOW_EVERY,
        "tables": tables(scores),
        "final_bias_dps": {name: [r["final_bias_dps"] for r in s["full"]["gyro"]] for name, s in scores.items()},
        "accel_bias_fit": {f.name: fitted_accel_bias(f).tolist() for f in args.prepared},
        "accel_bias_calibration": {
            name: None if bias is None else bias.tolist() for name, bias in calibrations(args.prepared).items()
        },
        "fitted_bias_dps": {
            folder.name: json.loads((folder / "checks.json").read_text(encoding="utf-8"))["fitted_bias_dps"]
            for folder in args.prepared
        },
        "runs": scores,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    dump_json(args.output / "summary.json", summary)
    write_report(args.output, summary)


if __name__ == "__main__":
    main()
