"""Score the C odometry on windows of the lake clip, and set two scorings side by side.

`windows` runs the odometry of the uwslam tree on the Python path (its core is built from that tree's core/) on
windows of the lake clip: 10 s windows starting every 2 s, each run on its own from its first frame, at each rate and
over uwslam.mvo.SEEDS. Each run is scored by uwslam.experiments.mono_vo.evaluate against the reference re-expressed
from the window's first pose. `compare` pairs two such scorings run for run (same rate, window and seed); `full` sets
the 16-seed tables of two `uwslam.mvo compare` summaries side by side.

usage:
  PYTHONPATH=<tree>/src python lake_score.py windows <video> --reconstruction <folder> --output <file.json> [--rates 5,10]
  python lake_score.py compare <control windows.json> <candidate windows.json>
  python lake_score.py full <control pc_summary.json> <candidate pc_summary.json>
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

METRICS = ("ate_rmse", "heading_rmse_deg", "heading_max_deg", "lost")
SETTINGS = ("length_s", "every_s", "time_scale", "seeds")
FULL_METRICS = ("ate_rmse", "heading_rmse_deg", "scale_spread", "lost")


def window_indices(first: int, length_s: float, fps: float, rate: float, count: int) -> list[int]:
    """One window's frames: every fps / rate frames from `first` for length_s seconds, within the clip."""
    picks = np.round(np.arange(first, first + length_s * fps, fps / rate)).astype(int)
    return [int(i) for i in sorted(set(picks)) if i < count]


def print_rate(rate: float, runs: list[dict]) -> None:
    median = {key: float(np.median([run[key] for run in runs])) for key in METRICS}
    print(f"{rate:g} Hz, {len(runs)} runs: median ATE {median['ate_rmse']:.3f} m, heading RMSE "
          f"{median['heading_rmse_deg']:.1f} deg, heading max {median['heading_max_deg']:.1f} deg, "
          f"lost {median['lost']:g}", flush=True)


def windows(video: Path, reconstruction: Path, output: Path, rates: list[float], length_s: float, every_s: float,
            time_scale: float) -> dict:
    from uwslam import mvo
    from uwslam.experiments.mask_motion import reference_poses
    from uwslam.experiments.mono_vo import ASPECT, HFOV_DEG, evaluate, read_frames
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output.name}; choose a fresh --output")
    poses = np.loadtxt(reconstruction / "images.txt")
    frames, fps, _ = read_frames(video, poses[-1, 0] * time_scale)
    mvo.require_30fps(fps)
    grey = frames["320x240"]
    times = np.arange(len(grey)) / fps
    reference_rotations, reference_centres = reference_poses(poses, times, time_scale)
    camera = mvo.camera_for(HFOV_DEG, ASPECT)
    starts = np.arange(0.0, times[-1] - length_s + 1e-6, every_s)
    runs = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:  # Windows keeps the loaded library locked
        vo = mvo.build(Path(scratch))
        for rate in rates:
            for start in starts:
                indices = window_indices(int(round(start * fps)), length_s, fps, rate, len(grey))
                first_rotation, first_centre = reference_rotations[indices[0]], reference_centres[indices[0]]
                window_rotations = first_rotation.T @ reference_rotations[indices]
                window_centres = (reference_centres[indices] - first_centre) @ first_rotation
                local = list(range(len(indices)))
                for seed in mvo.SEEDS:
                    records, rotations, centres = mvo.as_steps(mvo.run_frames(vo, [grey[i] for i in indices], camera, seed)[0])
                    result = evaluate(local, records, rotations, centres, times[indices], window_rotations, window_centres)
                    runs.append({"rate_hz": rate, "start_s": round(float(start), 3), "seed": seed,
                                 **{key: result[key] for key in METRICS}})
            print_rate(rate, [run for run in runs if run["rate_hz"] == rate])
    summary = {**mvo.sources_sha256(), "length_s": length_s, "every_s": every_s, "time_scale": time_scale,
               "seeds": list(mvo.SEEDS), "windows": len(starts), "runs": runs}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    return summary


def run_key(run: dict) -> tuple:
    return run["rate_hz"], run["start_s"], run["seed"]


def compare(control_path: Path, candidate_path: Path) -> None:
    control, candidate = (json.loads(path.read_text(encoding="utf-8")) for path in (control_path, candidate_path))
    differing = [key for key in SETTINGS if control[key] != candidate[key]]
    if differing:
        raise ValueError(f"The scorings differ in {', '.join(differing)}; score both with the same settings")
    base = {run_key(run): run for run in control["runs"]}
    print(f"control core {control['core_sha256'][:8]}, candidate core {candidate['core_sha256'][:8]}")
    for rate in sorted({run["rate_hz"] for run in candidate["runs"]}):
        pairs = [(base[run_key(run)], run) for run in candidate["runs"]
                 if run["rate_hz"] == rate and run_key(run) in base]
        print(f"{rate:g} Hz, {len(pairs)} paired runs:")
        for metric in METRICS:
            a = np.array([pair[0][metric] for pair in pairs], dtype=float)
            b = np.array([pair[1][metric] for pair in pairs], dtype=float)
            print(f"  {metric:17s} control {np.median(a):8.3f}  candidate {np.median(b):8.3f}  "
                  f"candidate lower in {np.mean(b < a):4.0%}, equal in {np.mean(b == a):4.0%}")


def full(control_path: Path, candidate_path: Path) -> None:
    control, candidate = (json.loads(path.read_text(encoding="utf-8")) for path in (control_path, candidate_path))
    print(f"control core {control['core_sha256'][:8]}, candidate core {candidate['core_sha256'][:8]}; "
          f"median [smallest-largest] over {len(control['runs'][0]['c_over_seeds']['seeds'])} seeds")
    rows = {row["rate_hz"]: row["c_over_seeds"] for row in control["runs"]}
    for row in candidate["runs"]:
        if row["rate_hz"] not in rows:
            continue
        print(f"{row['rate_hz']:g} Hz:")
        for metric in FULL_METRICS:
            a, b = rows[row["rate_hz"]][metric], row["c_over_seeds"][metric]
            print(f"  {metric:17s} control {a['median']:7.3f} [{a['min']:.3f}-{a['max']:.3f}]  "
                  f"candidate {b['median']:7.3f} [{b['min']:.3f}-{b['max']:.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("windows", help="Score the tree on the Python path on windows of the clip")
    score.add_argument("video", type=Path)
    score.add_argument("--reconstruction", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--rates", default="5,10", help="Comma-separated input rates in Hz")
    score.add_argument("--length", type=float, default=10.0, help="Window length in seconds")
    score.add_argument("--every", type=float, default=2.0, help="Seconds between window starts")
    score.add_argument("--time-scale", type=float, default=3.0)
    for name, text in (("compare", "Pair two window scorings run for run"),
                       ("full", "Set two `uwslam.mvo compare` pc_summary.json tables side by side")):
        command = commands.add_parser(name, help=text)
        command.add_argument("control", type=Path)
        command.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "windows":
        windows(args.video, args.reconstruction, args.output, [float(v) for v in args.rates.split(",") if v],
                args.length, args.every, args.time_scale)
    elif args.command == "compare":
        compare(args.control, args.candidate)
    else:
        full(args.control, args.candidate)


if __name__ == "__main__":
    main()
