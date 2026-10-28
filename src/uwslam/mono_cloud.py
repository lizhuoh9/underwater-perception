"""Build a bounded rolling sparse 3D point map from monocular video, without a segmentation mask.

The camera intrinsics must be for the 320x240 image after resizing. This tool does not
undistort frames: calibrate and undistort upstream when needed. Coordinates are in the
first camera's axes (x right, y down, z forward) and arbitrary monocular odometry units,
not metres or gravity-aligned height. The map freezes on an explicit LOST step; it does
not relocalize, close loops, reconstruct every surface, or certify free space.

Example: python -m uwslam.mono_cloud video.mp4 --fx 165 --fy 165 --output results/cloud_run
Optionally add --board-files to save the same streamed frames and PC expectations for
an ESP32 replay. Every frame is processed online; only the trajectory and bounded map
are kept in memory.
"""

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
from typing import BinaryIO, Iterable

import numpy as np

from uwslam import mvo
from uwslam.paths import CORE_DIR

MAX_POINTS = 512  # MC_MAX_POINTS in core/mono_cloud.h
POLICY = 1


class PointC(ctypes.Structure):
    _fields_ = [("xyz", ctypes.c_float * 3), ("id", ctypes.c_uint32),
                ("spread", ctypes.c_float), ("observations", ctypes.c_uint32)]


class CloudC(ctypes.Structure):
    _fields_ = [("points", PointC * MAX_POINTS), ("count", ctypes.c_uint32),
                ("dropped", ctypes.c_uint32), ("continuous", ctypes.c_int32)]


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    source_index: int
    source_timestamp_s: float | None = None


def check_camera(camera: mvo.Camera) -> None:
    values = np.array([camera.fx, camera.fy, camera.cx, camera.cy], dtype=np.float64)
    if not np.isfinite(values).all() or camera.fx <= 0 or camera.fy <= 0:
        raise ValueError("Camera intrinsics must be finite, with positive fx and fy")


def run_frames(vo, frames: Iterable[np.ndarray | Frame], camera: mvo.Camera, seed: int = mvo.SEED,
               frame_sink: BinaryIO | None = None) -> dict:
    """Process gray images once; Frame metadata retains source indices and decoder timestamps.

    Plain arrays use their sequential index and an unavailable timestamp. Map points are
    kept by the C core; segmentation and masks are neither required nor consulted.
    """
    check_camera(camera)
    if vo.mc_map_size() != ctypes.sizeof(CloudC):
        raise RuntimeError("Python and C point-cloud layouts do not match")
    state = ctypes.create_string_buffer(vo.mvo_state_size())
    pyramids = [ctypes.create_string_buffer(vo.mvo_pyramid_bytes()) for _ in range(2)]
    cloud = CloudC()
    vo.mvo_init(state, ctypes.byref(camera), seed, pyramids[0], pyramids[1])
    vo.mc_init(ctypes.byref(cloud))
    trajectory, step_hashes, progress, frame_hashes = [], [], [], []
    last_source_index = -1
    last_source_time = None
    first_lost_frame = None
    for ordinal, value in enumerate(frames):
        sample = value if isinstance(value, Frame) else Frame(value, ordinal)
        frame = sample.image
        if not isinstance(frame, np.ndarray) or frame.shape != (mvo.HEIGHT, mvo.WIDTH) or frame.dtype != np.uint8:
            raise ValueError(f"Expected a {mvo.HEIGHT}x{mvo.WIDTH} uint8 gray frame")
        if (not isinstance(sample.source_index, int) or sample.source_index <= last_source_index
                or (sample.source_timestamp_s is not None
                    and (not np.isfinite(sample.source_timestamp_s) or sample.source_timestamp_s < 0))):
            raise ValueError("Source frame index and timestamp must be valid")
        if sample.source_timestamp_s is not None:
            if last_source_time is not None and sample.source_timestamp_s < last_source_time:
                raise ValueError("Source timestamps must be nondecreasing")
            last_source_time = sample.source_timestamp_s
        last_source_index = sample.source_index
        pixels = np.ascontiguousarray(frame)
        ctypes.memmove(vo.mvo_frame(state), pixels.ctypes.data, mvo.WIDTH * mvo.HEIGHT)
        if frame_sink is not None:
            raw_frame = pixels.tobytes()
            if frame_sink.write(raw_frame) != len(raw_frame):
                raise OSError("Could not write a complete board frame")
            frame_hashes.append(hashlib.sha256(raw_frame).hexdigest())
        item = {"frame": ordinal, "source_frame": sample.source_index,
                "source_timestamp_s": sample.source_timestamp_s}
        if ordinal == 0:
            vo.mvo_start(state)
            item.update(kind="start", world_centre=[0.0, 0.0, 0.0],
                        world_rotation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        else:
            step = mvo.StepC()
            vo.mvo_step(state, ctypes.byref(step))
            if step.kind not in mvo.KINDS:
                raise RuntimeError(f"Unknown odometry step kind {step.kind}")
            accepted = int(vo.mc_update(ctypes.byref(cloud), state, ctypes.byref(step)))
            centre = [float(v) for v in step.world_centre]
            rotation = [float(v) for v in step.world_rotation]
            if not np.isfinite(centre + rotation).all():
                raise RuntimeError("Odometry returned a non-finite pose")
            if step.kind == 0 and first_lost_frame is None:
                first_lost_frame = ordinal
            state_info = {"count": int(cloud.count), "accepted": accepted, "dropped": int(cloud.dropped),
                          "continuous": int(cloud.continuous)}
            progress.append(state_info)
            step_hashes.append(mvo.fnv(mvo.FNV_OFFSET, bytes(step)))
            item.update(kind=mvo.KINDS[step.kind], tracks=int(step.tracks), inliers=int(step.inliers),
                        scale_points=int(step.scale_points), world_centre=centre, world_rotation=rotation,
                        cloud=state_info)
        trajectory.append(item)
    if not trajectory:
        raise ValueError("Video or frame sequence contains no frames")
    return {"cloud": cloud, "trajectory": trajectory, "step_hashes": step_hashes,
            "cloud_progress": progress, "frame_sha256": frame_hashes, "first_lost_frame": first_lost_frame,
            "cloud_hash": mvo.fnv(mvo.FNV_OFFSET, bytes(cloud))}


def video_frames(capture, every: int, read_stats: dict | None = None) -> Iterable[Frame]:
    """Yield resized grayscale frames and the source decoder's optional timestamps."""
    import cv2

    source_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            if read_stats is not None:
                read_stats["decoded_frames"] = source_index
                read_stats["stop_reason"] = "read_false_eof_or_decode_error"
            break
        if source_index % every == 0:
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_s = timestamp_ms / 1000.0 if np.isfinite(timestamp_ms) and timestamp_ms >= 0 else None
            if source_index > 0 and timestamp_s == 0:
                timestamp_s = None  # some decoders report zero for every frame
            if image is None or image.size == 0:
                raise ValueError("Video decoder returned an empty frame")
            resized = cv2.resize(image, (mvo.WIDTH, mvo.HEIGHT), interpolation=cv2.INTER_AREA)
            grey = resized if resized.ndim == 2 else cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            yield Frame(grey, source_index, timestamp_s)
        source_index += 1


def fingerprints() -> dict:
    return {**mvo.sources_sha256(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "binding_sha256": hashlib.sha256(Path(mvo.__file__).read_bytes()).hexdigest(),
            **{key: hashlib.sha256((CORE_DIR / name).read_bytes()).hexdigest()
               for key, name in (("cloud_sha256", "mono_cloud.c"), ("cloud_header_sha256", "mono_cloud.h"))}}


def _check_output(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Refusing to overwrite results; choose a new or empty --output directory")


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def _write_ply(path: Path, cloud: CloudC, summary: dict) -> None:
    with path.open("x", encoding="ascii", newline="\n") as target:
        target.write("ply\nformat ascii 1.0\n")
        target.write("comment first camera axes: x right, y down, z forward; not gravity aligned\n")
        target.write("comment arbitrary monocular odometry units; not metres\n")
        target.write("comment rolling sparse tracked points; unknown is not free space\n")
        target.write(f"comment continuous {summary['continuous']} (no explicit LOST; not an accuracy guarantee)\n")
        target.write(f"comment evicted_points {summary['dropped']}\n")
        target.write(f"comment cloud_sha256 {summary['sources_sha256']['cloud_sha256']}\n")
        target.write(f"comment cloud_header_sha256 {summary['sources_sha256']['cloud_header_sha256']}\n")
        target.write(f"comment source_video_sha256 {summary['source_sha256']}\n")
        target.write("comment spread is squared sine of the triangulating rays' separation\n")
        target.write(f"element vertex {cloud.count}\n")
        target.write("property float x\nproperty float y\nproperty float z\nproperty uint id\n"
                     "property uint observations\nproperty float spread\nend_header\n")
        for point in cloud.points[:cloud.count]:
            target.write(f"{point.xyz[0]:.9g} {point.xyz[1]:.9g} {point.xyz[2]:.9g} "
                         f"{point.id} {point.observations} {point.spread:.9g}\n")


def run_video(video: Path, output: Path, camera: mvo.Camera, every: int = 1, board_files: bool = False) -> dict:
    """Run one video and write a sparse point cloud, trajectory and explicit quality limits."""
    import cv2

    check_camera(camera)
    if every < 1:
        raise ValueError("--every must be at least 1")
    _check_output(output)
    source_stat = video.stat()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {video}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if np.isfinite(fps) and fps > 0 else None
        reported_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_count = int(reported_count) if (np.isfinite(reported_count) and reported_count > 0
                                                 and reported_count.is_integer()) else None
        read_stats = {}
        before_sources = fingerprints()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            scratch = Path(folder)
            vo = mvo.build(scratch)
            if board_files:
                with (scratch / "frames.bin").open("wb") as sink:
                    result = run_frames(vo, video_frames(capture, every, read_stats), camera, frame_sink=sink)
            else:
                result = run_frames(vo, video_frames(capture, every, read_stats), camera)
            if fingerprints() != before_sources:
                raise RuntimeError("VO or cloud source changed during the run; results were not saved")
            current_stat = video.stat()
            if (current_stat.st_size, current_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
                raise RuntimeError("Source video changed during the run; results were not saved")
            source_hash = mvo.file_sha256(video)
            current_stat = video.stat()
            if (current_stat.st_size, current_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
                raise RuntimeError("Source video changed during hashing; results were not saved")
            cloud, trajectory = result["cloud"], result["trajectory"]
            summary = {"frames": len(trajectory), "source_video": str(video), "source_sha256": source_hash,
                       "source_fps": fps, "every": every, "timestamp_source": "video decoder, null when unavailable",
                       "source_reported_frame_count": reported_count,
                       "source_decoded_frame_count": read_stats["decoded_frames"],
                       "source_read_termination": read_stats["stop_reason"],
                       "source_frame_count_consistent": (read_stats["decoded_frames"] == reported_count
                                                         if reported_count is not None else None),
                       "camera_320x240": {name: float(getattr(camera, name)) for name in ("fx", "fy", "cx", "cy")},
                       "frame_coordinates": "first camera: x right, y down, z forward; not gravity aligned",
                       "point_units": "arbitrary monocular odometry units, not metres", "distortion_correction": False,
                       "count": int(cloud.count), "capacity": MAX_POINTS, "dropped": int(cloud.dropped),
                       "dropped_definition": "historical points evicted as the rolling map fills",
                       "continuous": int(cloud.continuous), "first_lost_frame": result["first_lost_frame"],
                       "first_lost_source_frame": (trajectory[result["first_lost_frame"]]["source_frame"]
                                                   if result["first_lost_frame"] is not None else None),
                       "cloud_hash": result["cloud_hash"], "sources_sha256": before_sources,
                       "limitations": ["Bounded rolling sparse map: older points are evicted as capacity is reached; "
                                       "the result is not a complete global map, dense surface or free-space map.",
                                       "Tracking loss freezes the old cloud; recovery does not relocalize it.",
                                       "Decoder timestamps and nominal FPS are metadata, not verified exposure times.",
                                       "A decoder read failure cannot distinguish EOF from a decode error; the "
                                       "container's reported frame count can itself be inaccurate.",
                                       "The current VO assumes frame-to-frame step lengths without using timestamps; "
                                       "dropped or variable-rate frames can invalidate that assumption.",
                                       "No loop closure or global drift correction; motion and geometry can be wrong "
                                       "without an explicit LOST step."]}
            _check_output(output)
            output.mkdir(parents=True, exist_ok=True)
            _write_ply(output / "cloud.ply", cloud, summary)
            _write_json(output / "trajectory.json", trajectory)
            _write_json(output / "summary.json", summary)
            if board_files:
                frames_hash = mvo.file_sha256(scratch / "frames.bin")
                with (scratch / "frames.bin").open("rb") as source, (output / "frames.bin").open("xb") as target:
                    shutil.copyfileobj(source, target)
                if mvo.file_sha256(output / "frames.bin") != frames_hash:
                    raise RuntimeError("Copied board frames differ from the processed frames")
                camera_bits = [struct.unpack("<I", struct.pack("<f", getattr(camera, name)))[0]
                               for name in ("fx", "fy", "cx", "cy")]
                expected = {"frames": len(trajectory), "every": 1, "rate_hz": fps / every if fps else None,
                            "step_hashes": result["step_hashes"],
                            "camera_bits": camera_bits, "seed": mvo.SEED,
                            **{key: before_sources[key] for key in ("core_sha256", "header_sha256")}}
                expected_cloud = {"frames": len(trajectory), "step_hashes": result["step_hashes"],
                                  "cloud_policy": POLICY, "cloud_sha256": before_sources["cloud_sha256"],
                                  "cloud_header_sha256": before_sources["cloud_header_sha256"],
                                  "frames_sha256": frames_hash, "frame_sha256": result["frame_sha256"],
                                  "source_sha256": source_hash, "frame_coordinates": summary["frame_coordinates"],
                                  "point_units": summary["point_units"],
                                  "cloud_progress": result["cloud_progress"], "cloud_hash": result["cloud_hash"],
                                  "cloud_count": int(cloud.count), "cloud_dropped": int(cloud.dropped),
                                  "cloud_continuous": int(cloud.continuous)}
                _write_json(output / "board_expected.json", expected)
                _write_json(output / "cloud_expected.json", expected_cloud)
            return summary
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--fx", type=float, required=True, help="Focal length in x for the resized 320x240 image, pixels")
    parser.add_argument("--fy", type=float, required=True, help="Focal length in y for the resized 320x240 image, pixels")
    parser.add_argument("--cx", type=float, default=160.0, help="Principal point x for 320x240, pixels")
    parser.add_argument("--cy", type=float, default=120.0, help="Principal point y for 320x240, pixels")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth decoded frame")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--board-files", action="store_true", help="Also prepare frames and expectations for ESP32 replay")
    args = parser.parse_args()
    summary = run_video(args.video, args.output, mvo.Camera(args.fx, args.fy, args.cx, args.cy), args.every, args.board_files)
    print(f"{summary['frames']} frames, {summary['count']} sparse points, "
          f"continuous={summary['continuous']}, dropped={summary['dropped']}; coordinates are not metres")


if __name__ == "__main__":
    main()
