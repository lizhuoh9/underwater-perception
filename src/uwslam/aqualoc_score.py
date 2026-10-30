"""The odometry on an AQUALOC sequence prepared by uwslam.aqualoc, with and without the dataset's gyro: 10 s windows
every 5 s and the whole sequence, every seed of uwslam.mvo.SEEDS, each gyro run paired with the camera alone on the same
window and seed.

Modes: `off` (the camera alone); `gyro` (the readings, the bias learnt online from zero at the run's start); `warm`
(windows only: the bias starts from the estimate the whole-sequence gyro run of the same seed held after the step
before the window, what a robot running all along would have). Each run is scored as uwslam.experiments.mono_vo
scores the lake clip (ATE after a similarity alignment, its heading, lost steps), plus two attitude scores, since the
harbour camera looks down and that heading is then a tilt: the yaw about the reference's vertical and the whole
orientation error (the angle of R_est^T R_ref), both from the run's first frame, where estimate and reference coincide.

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
METRICS = ("ate_rmse", "yaw_rmse_deg", "orientation_rmse_deg", "heading_rmse_deg", "lost")
LABELS = ("ATE (m)", "yaw RMSE (°)", "orientation RMSE (°)", "heading RMSE (°)", "lost steps")
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
    mount: np.ndarray
    vertical: np.ndarray


def load_prepared(folder: Path) -> Prepared:
    with np.load(folder / "sequence.npz") as archive:  # closed at once: Windows keeps an open file locked
        data = {name: archive[name] for name in archive.files}
    frames = np.load(folder / "frames_5hz.npy")
    if frames.ndim != 3 or frames.shape[1:] != (mvo.HEIGHT, mvo.WIDTH) or frames.dtype != np.uint8:
        raise ValueError(f"{folder.name}: frames_5hz.npy must hold 320x240 grey frames")
    if len(frames) != len(data["stamps"]) or len(data["reading_offsets"]) != len(frames):
        raise ValueError(f"{folder.name}: frames, stamps and readings disagree")
    offsets, rows = data["reading_offsets"], data["readings"]
    values = json.loads((folder / "camera.json").read_text(encoding="utf-8"))["camera"]
    return Prepared(
        folder.name,
        frames,
        mvo.Camera(*(float(values[name]) for name in mvo.CAMERA_FIELDS)),
        data["stamps"],
        data["reference_rotations"],
        data["reference_centres"],
        [rows[offsets[k] : offsets[k + 1]] for k in range(len(offsets) - 1)],
        data["mount"],
        data["vertical"],
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
    used = run.gyro_used or [0] * len(run.steps)
    return {
        "first": first,
        **{key: result[key] for key in ("ate_rmse", "heading_rmse_deg", "lost", "kinds")},
        "yaw_rmse_deg": float(np.sqrt(np.mean(yaw**2))),
        "orientation_rmse_deg": float(np.sqrt(np.mean(orientation**2))),
        "uses": {name: used.count(code) for code, name in enumerate(USES)},
    }


def score_seed(vo, sequence: Prepared, seed: int) -> dict:
    """One seed's runs on the sequence, keyed by command and mode: the whole sequence, then every window."""
    n = len(sequence.frames)
    spans = [(s, s + WINDOW_FRAMES) for s in range(0, n - WINDOW_FRAMES + 1, WINDOW_EVERY)]
    out = {"full": {"off": [], "gyro": []}, "windows": {"off": [], "gyro": [], "warm": []}}
    frames = list(sequence.frames)
    for mode, gyro in (("off", None), ("gyro", mvo.GyroReadings(sequence.mount, sequence.readings))):
        run = mvo.run_odometry(vo, frames, sequence.camera, seed, gyro)
        out["full"][mode].append({"seed": seed, **score_run(sequence, 0, run)})
        if gyro is not None:
            out["full"][mode][-1]["final_bias_dps"] = np.degrees(run.gyro_bias[-1]).tolist()
            carried = run.gyro_bias
    for first, end in spans:
        window = list(sequence.frames[first:end])
        readings = sequence.readings[first : end - 1]
        for mode in ("off", "gyro", "warm"):
            gyro = None
            if mode != "off":
                bias = np.array(carried[first - 1], dtype=np.float32) if mode == "warm" and first > 0 else None
                gyro = mvo.GyroReadings(sequence.mount, readings, bias)
            run = mvo.run_odometry(vo, window, sequence.camera, seed, gyro)
            out["windows"][mode].append({"seed": seed, **score_run(sequence, first, run)})
    return out


def merge(per_seed: list[dict]) -> dict:
    """The seeds' runs in seed order, under each command and mode."""
    return {
        command: {mode: [run for seed in per_seed for run in seed[command][mode]] for mode in modes}
        for command, modes in per_seed[0].items()
    }


def score_sequence(vo, sequence: Prepared) -> dict:
    """Every seed's runs on the sequence, one after the other."""
    per_seed = []
    for seed in mvo.SEEDS:
        per_seed.append(score_seed(vo, sequence, seed))
        print(f"{sequence.name} seed {seed} scored", flush=True)
    return merge(per_seed)


WORKER = {}  # a worker process's core and the sequences it has loaded


def start_worker(scratch: str) -> None:
    WORKER["vo"] = mvo.build(Path(scratch) / f"worker{os.getpid()}")


def worker_score(folder: str, seed: int) -> dict:
    if folder not in WORKER:
        WORKER[folder] = load_prepared(Path(folder))
    return score_seed(WORKER["vo"], WORKER[folder], seed)


def score_all(folders: list[Path], jobs: int) -> dict:
    """Every sequence's runs, the seeds spread over `jobs` processes (each with its own build of the core)."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:  # Windows keeps the loaded library locked
        if jobs <= 1:
            vo = mvo.build(Path(scratch))
            return {folder.name: score_sequence(vo, load_prepared(folder)) for folder in folders}
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(jobs, initializer=start_worker, initargs=(scratch,)) as pool:
            futures = {(f.name, seed): pool.submit(worker_score, str(f), seed) for f in folders for seed in mvo.SEEDS}
            done = {key: future.result() for key, future in futures.items()}
        return {f.name: merge([done[(f.name, seed)] for seed in mvo.SEEDS]) for f in folders}


def paired(runs: list[dict], base: list[dict] | None, metric: str) -> dict:
    values = np.array([r[metric] for r in runs], dtype=float)
    out = {"median": float(np.median(values))}
    if base is not None:
        if [(r["seed"], r["first"]) for r in runs] != [(r["seed"], r["first"]) for r in base]:
            raise ValueError("Paired runs must be the same windows and seeds")
        out["lower_share"] = float(np.mean(values < np.array([r[metric] for r in base], dtype=float)))
    return out


def tables(scores: dict) -> dict:
    """Per sequence (and all together), command and mode: each metric's median and the share of the pairs in which the
    mode is lower than off."""
    out = {}
    groups = {name: runs for name, runs in scores.items()}
    if len(scores) > 1:
        groups["all"] = {
            command: {mode: [r for runs in scores.values() for r in runs[command][mode]] for mode in modes}
            for command, modes in next(iter(scores.values())).items()
        }
    for name, runs in groups.items():
        out[name] = {
            command: {
                mode: {
                    "runs": len(mode_runs),
                    **{m: paired(mode_runs, None if mode == "off" else by_mode["off"], m) for m in METRICS},
                }
                for mode, mode_runs in by_mode.items()
            }
            for command, by_mode in runs.items()
            if by_mode["off"]  # a sequence shorter than a window has no windows
        }
    return out


def write_report(output: Path, summary: dict) -> None:
    lines = [
        "# The odometry with and without the gyro on AQUALOC harbour sequences",
        "",
        f"5 Hz, {len(mvo.SEEDS)} seeds; 10 s windows every 5 s and whole sequences. Each cell is the median over the "
        "runs and, for a gyro mode, the share of the runs paired with `off` (same window, same seed) in which it is "
        "lower. `gyro` learns the bias from zero at the run's start; `warm` starts each window from the whole-sequence "
        "run's estimate at that point. The camera looks down at the seabed: yaw is about the vertical, orientation the "
        "whole rotation error, heading the turn about the first camera's y axis (a tilt here).",
        "",
    ]
    for command, title in (("windows", "10 s windows"), ("full", "Whole sequences")):
        lines += [f"## {title}", "", "| Sequence | Mode | Runs | " + " | ".join(LABELS) + " |"]
        lines.append("|---|---|---:|" + "---:|" * len(LABELS))
        for name, by_command in summary["tables"].items():
            for mode, row in by_command.get(command, {}).items():
                cells = []
                for metric, digits in zip(METRICS, (3, 2, 2, 2, 0)):
                    cell = f"{row[metric]['median']:.{digits}f}"
                    if "lower_share" in row[metric]:
                        cell += f" ({row[metric]['lower_share'] * 100:.0f}%)"
                    cells.append(cell)
                lines.append(f"| {name} | {mode} | {row['runs']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## The bias the whole-sequence runs learnt (deg/s, the IMU's axes)", ""]
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
