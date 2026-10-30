"""Underwater ultrasonic rangers on the lake clip: three of them (forward along the camera's axis, left and right at a
right angle) simulated from the reconstruction along the reference poses at the board's 5 Hz frames, and the odometry's
metric scale from the forward one (mvo_range_setup, mvo_range_reading), every seed of uwslam.mvo.SEEDS, scored in
metres against the reference.

A ranger's echo is the nearest surface in its beam: the distance within which ECHO_SHARE of the reconstruction's
points inside the cone (at least ECHO_POINTS of them, within MAX_RANGE) lie. Each run reads it with its own noise
(NOISE_M and NOISE_SHARE of the range, one standard deviation, rounded to the millimetre) and loses DROPOUT of the
readings, drawn from NOISE_SEED and the run's seed. The reconstruction holds only what the camera filmed, so a side
ranger facing what the camera never saw hears nothing here. Its unit is taken as the metre, as in every lake score; the
rangers measure in that same unit, so this checks the fusion, not a sensor.

Scores: the metric ATE, the world centre in metres (uwslam.mvo.metric_path) after the best rotation and shift, no scale
(the lake has no gravity to level by), over the whole clip and over 10 s windows cut from the whole runs (a robot keeps
the scale it has); beside them the same runs' odometry after a similarity alignment, whose scale comes from the
reference (the whole clip, and each window with its own scale), and the scale that alignment would give the metric path
(1: in metres).

For the board (uwslam.mvo_stream --sense): sense_5hz.bin (the forward range at each frame, as the run with seed
uwslam.mvo.SEED read it) and board_expected_ranger_5hz.json (that run's steps and every step's metric centre and
scale), and with --board-frames the 5 Hz frames.

  python -m uwslam.ranger <video> --reconstruction <folder> --output <fresh folder> [--beam 7.5] [--board-frames <file>]
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from uwslam import mvo
from uwslam.experiments.mono_vo import umeyama
from uwslam.utils import dump_json

STRIDE = 6  # the 30 FPS clip's 5 Hz frames
MAX_RANGE, ECHO_POINTS, ECHO_SHARE = 8.0, 20, 0.02
NOISE_M, NOISE_SHARE, DROPOUT = 0.01, 0.005, 0.05
NOISE_SEED = 11
BEAM_DEG = 7.5  # the half angle: a 15 degree beam
AXES = {"forward": (0.0, 0.0, 1.0), "left": (-1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0)}  # the camera's axes
WINDOW_FRAMES, WINDOW_EVERY = 50, 25  # 10 s windows every 5 s
SCORES = ("metric", "metric_windows", "similarity", "similarity_windows", "scale_ratio", "first_scale_step")


def echoes(cloud: np.ndarray, rotations: np.ndarray, centres: np.ndarray, half_angle_deg: float) -> dict:
    """Each ranger's true range at each frame (NaN: no echo), from the reference's camera-to-world poses."""
    cos_limit = np.cos(np.radians(half_angle_deg))
    out = {name: np.full(len(rotations), np.nan) for name in AXES}
    for k, (rotation, centre) in enumerate(zip(rotations, centres)):
        local = (cloud - centre.astype(np.float32)) @ rotation.astype(np.float32)
        distance = np.linalg.norm(local, axis=1)
        near = distance < MAX_RANGE
        local, distance = local[near], distance[near]
        for name, axis in AXES.items():
            cone = (local @ np.array(axis, dtype=np.float32)) > cos_limit * distance
            if cone.sum() >= ECHO_POINTS:
                out[name][k] = float(np.quantile(distance[cone], ECHO_SHARE))
    return out


def readings(true: np.ndarray, seed: int) -> np.ndarray:
    """One run's readings of the true ranges: noise and dropouts drawn from NOISE_SEED and the run's seed."""
    rng = np.random.default_rng((NOISE_SEED, seed))
    noisy = np.round(true + rng.normal(0.0, 1.0, len(true)) * (NOISE_M + NOISE_SHARE * np.nan_to_num(true)), 3)
    return np.where(rng.random(len(true)) < DROPOUT, np.nan, noisy)


def ate(path: np.ndarray, reference: np.ndarray, with_scale: bool) -> float:
    """RMSE after the best rotation and shift, and with_scale the best scale too."""
    scale, rotation, _ = umeyama(path, reference)
    moved = (path - path.mean(axis=0)) @ rotation.T * (scale if with_scale else 1.0)
    return float(np.sqrt(np.mean(np.sum((moved - (reference - reference.mean(axis=0))) ** 2, axis=1))))


def windows(path: np.ndarray, reference: np.ndarray, with_scale: bool) -> list[float]:
    return [
        ate(path[f : f + WINDOW_FRAMES], reference[f : f + WINDOW_FRAMES], with_scale)
        for f in range(0, len(path) - WINDOW_FRAMES + 1, WINDOW_EVERY)
    ]


def spread(values: list[float]) -> dict | None:
    """Median, smallest and largest (uwslam.mvo.spread), None without values."""
    return mvo.spread(values) if values else None


def score(run: mvo.Run, reference: np.ndarray, scores: dict) -> None:
    """One run's scores appended to `scores` (SCORES); the metric ones only when its scale was known."""
    centres = mvo.as_steps(run.steps)[2]
    scores["similarity"].append(ate(centres, reference, True))
    scores["similarity_windows"] += windows(centres, reference, True)
    known = [k for k, m in enumerate(run.metric) if m is not None]
    scores["known_share"].append(len(known) / len(run.metric))
    path = mvo.metric_path(run)
    if path is None:
        return
    scores["first_scale_step"].append(known[0] + 1)
    scores["metric"].append(ate(path, reference, False))
    scores["metric_windows"] += windows(path, reference, False)
    scores["scale_ratio"].append(umeyama(path, reference)[0])


def run(video: Path, reconstruction: Path, output: Path, beam_deg: float, board_frames: Path | None = None) -> dict:
    from uwslam.experiments.mask_motion import REFERENCE_TIME_SCALE, reference_poses
    from uwslam.experiments.mono_vo import read_frames
    from uwslam.obstacle_dir import load_cloud

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite; choose a fresh --output")
    if board_frames is not None and (board_frames.exists() or board_frames.resolve().parent == output.resolve()):
        raise FileExistsError("Refusing to overwrite board frames; choose a fresh --board-frames outside --output")
    poses = np.loadtxt(reconstruction / "images.txt")
    frames, fps, _ = read_frames(video, poses[-1, 0] * REFERENCE_TIME_SCALE)
    mvo.require_30fps(fps)
    grey = frames["320x240"]
    ref_r, ref_c = reference_poses(poses, np.arange(len(grey)) / fps, REFERENCE_TIME_SCALE)
    indices = list(range(0, len(grey), STRIDE))
    true = echoes(load_cloud(reconstruction), ref_r[indices], ref_c[indices], beam_deg)
    camera, clip, reference = mvo.clip_camera(), [grey[i] for i in indices], ref_c[indices]
    beam = mvo.Beam(AXES["forward"], beam_deg)
    scores = {key: [] for key in (*SCORES, "known_share")}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        vo = mvo.build(Path(scratch))
        for seed in mvo.SEEDS:
            ranges = readings(true["forward"], seed)
            result = mvo.run_odometry(vo, clip, camera, seed, ranges=ranges, beam=beam)
            if seed == mvo.SEED:
                board_run, board_ranges = result, ranges
            score(result, reference, scores)
    summary = {
        "video": video.name,
        "frames": len(indices),
        "beam_half_angle_deg": beam_deg,
        "simulation": {
            "max_range_m": MAX_RANGE,
            "echo_points": ECHO_POINTS,
            "echo_share": ECHO_SHARE,
            "noise_m": NOISE_M,
            "noise_share": NOISE_SHARE,
            "dropout": DROPOUT,
            "seed": NOISE_SEED,
        },
        "echoes": {name: float(np.isfinite(values).mean()) for name, values in true.items()},
        "seeds": list(mvo.SEEDS),
        "runs_with_scale": len(scores["metric"]),
        **{key: spread(values) for key, values in scores.items()},
        **mvo.sources_sha256(),
        **mvo.build_config(),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "ranges_5hz.npz", indices=np.array(indices), **{f"true_{k}": v for k, v in true.items()})
    sense = mvo.sense_bytes(None, board_ranges, len(indices))
    (output / "sense_5hz.bin").write_bytes(sense)
    dump_json(output / "board_expected_ranger_5hz.json", board_expectation(video, camera, board_run, sense, beam))
    if board_frames is not None:
        board_frames.parent.mkdir(parents=True, exist_ok=True)
        with board_frames.open("xb") as out:
            out.write(b"".join(np.ascontiguousarray(frame).tobytes() for frame in clip))
    dump_json(output / "summary.json", summary)
    write_report(output, summary)
    return summary


def board_expectation(video: Path, camera: mvo.Camera, run: mvo.Run, sense: bytes, beam: mvo.Beam) -> dict:
    """The board's expectation for the run with the forward ranger (uwslam.mvo_stream --sense)."""
    return {
        "video": video.name,
        "rate_hz": 5.0,
        "every": 1,
        "frames": len(run.steps) + 1,
        "steps": len(run.steps),
        "camera": [getattr(camera, field) for field in mvo.CAMERA_FIELDS],
        "seed": mvo.SEED,
        "camera_bits": mvo.camera_bits(camera),
        "step_hashes": run.hashes,
        "kinds": [mvo.KINDS[s.kind] for s in run.steps],
        "final_world_centre": list(run.steps[-1].world_centre),
        **mvo.sources_sha256(),
        **mvo.build_config(),
        "metric": {"sense_file": "sense_5hz.bin", **mvo.metric_expectation(run, sense, beam)},
    }


def write_report(output: Path, summary: dict) -> None:
    def cell(values: dict | None, digits: int = 3) -> str:
        if values is None:
            return "–"
        return f"{values['median']:.{digits}f} [{values['min']:.{digits}f}–{values['max']:.{digits}f}]"

    simulation = summary["simulation"]
    echoed = ", ".join(f"{name} {share:.0%}" for name, share in summary["echoes"].items())
    lines = [
        "# Underwater rangers on the lake clip, simulated",
        "",
        f"Three rangers simulated from the reconstruction along the reference poses at the {summary['frames']} frames "
        f"of 5 Hz: a {2 * summary['beam_half_angle_deg']:g} degree beam, the nearest surface within "
        f"{simulation['max_range_m']:g} m, each run with its own noise of {simulation['noise_m'] * 100:g} cm plus "
        f"{simulation['noise_share'] * 100:g}% of the range and {simulation['dropout']:.0%} of its readings lost (see "
        f"uwslam.ranger). Echoes: {echoed} of the frames; the reconstruction holds only what the camera filmed. Its "
        "unit is taken as the metre, as in every lake score. The forward ranger's readings give the odometry its "
        "metric scale (mvo_range_reading); the core's own steps stay as they were.",
        "",
        f"Over {len(summary['seeds'])} seeds, median [smallest–largest]; windows are 10 s cut every 5 s from the whole "
        "runs:",
        "",
        "| Score | Whole clip | Windows |",
        "|---|---:|---:|",
        f"| Metric ATE (m): the path in metres, rotation and shift only | {cell(summary['metric'])} | "
        f"{cell(summary['metric_windows'])} |",
        f"| ATE after a similarity alignment (m): the scale from the reference | {cell(summary['similarity'])} | "
        f"{cell(summary['similarity_windows'])} |",
        "",
        f"The scale a similarity alignment would give the metric path (1: in metres): {cell(summary['scale_ratio'])}. "
        f"The scale was first known at step {cell(summary['first_scale_step'], 0)} and then in a share "
        f"{cell(summary['known_share'], 2)} of the steps; runs with a scale: {summary['runs_with_scale']} of "
        f"{len(summary['seeds'])}.",
        "",
        "Re-run (the output folder must not exist or be empty), and check the board against the run with seed "
        "uwslam.mvo.SEED (its steps, and every step's metric centre and scale):",
        "",
        "```bash",
        "python -m uwslam.ranger <video> --reconstruction <folder> --output <fresh folder> --board-frames <frames file>",
        "python -m uwslam.mvo_stream --port <port> --frames <frames file> --sense <folder>/sense_5hz.bin "
        "--expected <folder>/board_expected_ranger_5hz.json --label <name>",
        "```",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam", type=float, default=BEAM_DEG, help="the ranger's half angle, degrees")
    parser.add_argument("--board-frames", type=Path, help="Where to write the 5 Hz frames the board test streams")
    args = parser.parse_args()
    summary = run(args.video, args.reconstruction, args.output, args.beam, args.board_frames)
    print({key: summary[key] for key in ("echoes", "metric", "metric_windows", "similarity", "similarity_windows")})


if __name__ == "__main__":
    main()
