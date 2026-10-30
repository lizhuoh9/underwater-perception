"""A camera for the ESP32-P4, played from a video: the frames reach the board the way a camera would deliver them,
compressed (JPEG, at the camera's own size and in colour), one every 1/rate seconds whatever the board is doing. The
board does the rest itself (firmware/mono_vo_stream's live build, `jpeg`): its JPEG engine decodes each frame to
YUV444, core/frame_prep.c averages the luminance onto 320x240, and a worker task runs the odometry, the maps and the
plan on the newest frame while the next one comes in. A frame still waiting when a newer one arrives is
dropped there, as a camera's would be. The host plays the camera and records what the board did with each frame; it
prepares nothing.

- prepare (numpy and OpenCV): the lake clip's frames at the rate (every EVERY-th 30 FPS video frame for 5 Hz, as
  uwslam.mvo's board frames take them, up to the reference's last keyframe), each the video frame as it is, JPEG at
  --quality with --sampling's chroma subsampling, into frames.jpg.bin (per frame a little-endian uint32 length, then
  its bytes) and feed.json (the clip's camera, the seed, every frame's FNV-1a hash). The frames do not depend on the C
  sources.
- stream (pyserial only, so it runs in ESP-IDF's Python): plays the frames to the board in real time and writes
  board/<label>.json (with the sources and build the board reported) and <label>_serial.log beside feed.json. With
  --check it waits for each frame's step instead and reads back the 320x240 frame the odometry saw (`grey`), into
  board/<label>_grey.bin.
- check (numpy and OpenCV): for a --check run of a board built from this checkout's sources, the PC core on the board's
  own frames against the board's steps (they must be equal, bit for bit), the board's frames against the PC's decoding
  of the same JPEGs (their luminance, averaged by the same core/frame_prep.c; in all and frame by frame), and the paths
  of both over every seed against the reconstruction, beside the PC tools' own scores at the rate. Writes summary.json
  into a fresh --output.

  python -m uwslam.live_feed prepare <video> --reconstruction <folder> --output <fresh folder> [--rate 5] [--quality 80]
      [--sampling 420]
  python -m uwslam.live_feed stream <prepared folder> --port <port> --label <name> [--check]
  python -m uwslam.live_feed check <prepared folder> --label <check run's name> --reconstruction <folder>
      --output <fresh folder>
"""

import argparse
import hashlib
import json
import struct
import threading
import time
from pathlib import Path

from uwslam.mvo_stream import (
    LINK_BAUD,
    REPLY_TIMEOUT_S,
    SELFTEST_CASES,
    SELFTEST_SEED,
    SELFTEST_TIMEOUT_S,
    Board,
    fields,
    fnv,
    summarise,
)

FRAME_BYTES = 320 * 240
RATE_HZ, QUALITY = 5.0, 80
SAMPLINGS = ("420", "422", "444")  # the JPEGs' chroma subsampling; 4:2:0 is OpenCV's default
VIDEO_FPS = 30.0  # the lake clip's; uwslam.mvo refuses another
READY_TIMEOUT_S = 10.0
SETTLE_S = 0.5  # between the board's selftest and the first frame
RESULT_WAIT_S = 30.0  # after the last frame, for the board to answer every frame
WRITE_TIMEOUT_S = 10.0  # a board that stops reading fails the run instead of hanging it
SOURCES = ("core_sha256", "header_sha256", "prep_sha256", "prep_header_sha256")
TIMINGS = ("wait_us", "decode_us", "prep_us", "total_us", "voxel_us", "dir_us", "plan_us", "cloud_us", "map_us")
WORK = ("decode_us", "prep_us", "total_us", "cloud_us", "voxel_us", "dir_us", "plan_us", "map_us")  # a frame's work


# prepare


def prepare(video: Path, reconstruction: Path, output: Path, rate: float, quality: int, sampling: str = "420") -> dict:
    """The camera's frames: every frame uwslam.mvo's board stream at this rate would take, as JPEG."""
    import cv2
    import numpy as np

    from uwslam import mvo
    from uwslam.experiments.mask_motion import REFERENCE_TIME_SCALE as TIME_SCALE
    from uwslam.experiments.mask_motion import TIME_SLACK

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite; choose a fresh --output")
    if rate not in mvo.RATES:
        raise ValueError(f"The rate must be one of {sorted(mvo.RATES)} Hz")
    if sampling not in SAMPLINGS:
        raise ValueError(f"The sampling must be one of {SAMPLINGS}")
    every = mvo.RATES[rate]
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if sampling != "420":  # left at OpenCV's default otherwise, which is 4:2:0
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, getattr(cv2, f"IMWRITE_JPEG_SAMPLING_FACTOR_{sampling}")]
    end_s = float(np.loadtxt(reconstruction / "images.txt")[-1, 0]) * TIME_SCALE
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video.name}")
    jpegs, indices, size = [], [], None
    try:
        mvo.require_30fps(capture.get(cv2.CAP_PROP_FPS))
        index = 0
        while index / VIDEO_FPS <= end_s + TIME_SLACK:
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"{video.name} ends before the last reference keyframe")
            if index % every == 0:
                ok, encoded = cv2.imencode(".jpg", bgr, params)
                if not ok:
                    raise RuntimeError(f"Frame {index} would not encode")
                jpegs.append(encoded.tobytes())
                indices.append(index)
                size = (bgr.shape[1], bgr.shape[0])
            index += 1
    finally:
        capture.release()
    camera = mvo.clip_camera()
    output.mkdir(parents=True, exist_ok=True)
    frames_file = output / "frames.jpg.bin"
    frames_file.write_bytes(b"".join(struct.pack("<I", len(j)) + j for j in jpegs))
    feed = {
        "video": video.name,
        "video_sha256": mvo.file_sha256(video),
        "rate_hz": rate,
        "every": every,
        "video_indices": indices,
        "source_size": list(size),
        "jpeg_quality": quality,
        "jpeg_sampling": sampling,
        "frames": len(jpegs),
        "frames_file": frames_file.name,
        "frames_sha256": hashlib.sha256(frames_file.read_bytes()).hexdigest(),
        "jpeg_fnv": [fnv(j) for j in jpegs],
        "jpeg_bytes": summarise([float(len(j)) for j in jpegs]),
        "camera": [getattr(camera, name) for name in mvo.CAMERA_FIELDS],
        "camera_bits": mvo.camera_bits(camera),
        "seed": mvo.SEED,
    }
    (output / "feed.json").write_text(json.dumps(feed, indent=1) + "\n", encoding="utf-8", newline="\n")
    return {key: feed[key] for key in ("frames", "source_size", "jpeg_quality", "jpeg_bytes")}


# stream


def read_jpegs(path: Path, count: int) -> list[bytes]:
    data = path.read_bytes()
    out, at = [], 0
    while at < len(data):
        if at + 4 > len(data):
            raise ValueError(f"{path.name} ends inside a frame's length")
        (size,) = struct.unpack_from("<I", data, at)
        if at + 4 + size > len(data):
            raise ValueError(f"{path.name} ends inside frame {len(out)}")
        out.append(data[at + 4 : at + 4 + size])
        at += 4 + size
    if len(out) != count:
        raise ValueError(f"{path.name} holds {len(out)} frames; feed.json names {count}")
    return out


def open_session(board: Board, feed: dict, cores: int) -> dict:
    """The ready line (the live build; check compares its sources with the PC's), then start with the feed's camera
    and the vector kernels' selftest; returns the ready line's fields."""
    ready = fields(board.reply(("ready",), READY_TIMEOUT_S))
    if ready.get("live") != 1:
        raise RuntimeError("The board runs the replay build: flash the live build (sdkconfig.live)")
    camera = feed["camera_bits"]
    board.send(
        (
            "start "
            + " ".join(str(bits) for bits in camera[:4])
            + f" {feed['seed']} {cores} 1"
            + "".join(f" {bits}" for bits in camera[4:])
            + "\n"
        ).encode()
    )
    board.reply(("started",), REPLY_TIMEOUT_S)
    board.send(f"selftest {SELFTEST_CASES} {SELFTEST_SEED}\n".encode())
    selftest = fields(board.reply(("selftest",), SELFTEST_TIMEOUT_S))
    if selftest["template_mismatches"] or selftest["sums_mismatches"] or selftest["block_mismatches"]:
        raise RuntimeError(f"The board's vector kernels differ from the portable ones: {selftest}")
    return ready


def frame_message(index: int, jpeg: bytes, digest: int) -> bytes:
    """The header on a fresh line (the live build skips empty lines), so that a frame whose bytes went missing
    cannot hide the next header; then the frame."""
    return f"\njpeg {index} {len(jpeg)} {digest}\n".encode() + jpeg


def play(board: Board, jpegs: list[bytes], digests: list[int], rate: float) -> tuple[list[dict], list[tuple]]:
    """Every frame at its time, whatever the board answers; a reader thread keeps every line with when it came,
    and in the board's log."""
    lines: list[tuple[float, str]] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            raw = board.port.readline()
            if raw:
                line = raw.decode("ascii", errors="replace").strip()
                lines.append((time.monotonic(), line))
                board.lines.append(line)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    start = time.monotonic() + SETTLE_S
    sent = []
    try:
        for k, jpeg in enumerate(jpegs):
            due = start + k / rate
            while time.monotonic() < due:
                time.sleep(min(0.002, max(0.0, due - time.monotonic())))
            began = time.monotonic()
            board.port.write(frame_message(k, jpeg, digests[k]))
            board.port.flush()
            written = time.monotonic() - start
            sent.append({"index": k, "late_s": began - due, "written_s": written, "bytes": len(jpeg)})
        deadline = time.monotonic() + RESULT_WAIT_S
        while time.monotonic() < deadline and len(answered(lines)) < len(jpegs):
            time.sleep(0.05)
    finally:
        stop.set()
        thread.join(2.0)
    return sent, [(t - start, line) for t, line in lines]


def answered(lines: list[tuple]) -> dict[int, tuple]:
    """Each frame's answer: its first, step or dropped line, or an error naming it."""
    out = {}
    for t, line in lines:
        words = line.split()
        if not words:
            continue
        if words[0] in ("first", "step", "dropped"):
            index = fields(line).get("index")
            if type(index) is int:
                out[index] = (t, line)
        elif words[0] == "error" and len(words) > 2 and words[1] == "jpeg" and words[2].isdigit():
            out[int(words[2])] = (t, line)
    return out


def read_exact(board: Board, size: int, timeout: float) -> bytes:
    data, deadline = b"", time.monotonic() + timeout
    while len(data) < size and time.monotonic() < deadline:
        data += board.port.read(size - len(data))
    if len(data) != size:
        raise TimeoutError(f"{len(data)} of {size} bytes came back")
    return data


def check_frames(board: Board, jpegs: list[bytes], digests: list[int], grey_path: Path) -> list[tuple]:
    """Each frame in turn, waiting for its step, then the 320x240 frame the odometry saw read back."""
    lines = []
    with grey_path.open("xb") as out:
        for k, jpeg in enumerate(jpegs):
            board.send(frame_message(k, jpeg, digests[k]))
            line = board.reply(("first", "step"), REPLY_TIMEOUT_S)
            lines.append((0.0, line))
            board.send(b"grey\n")
            head = fields(board.reply(("grey",), REPLY_TIMEOUT_S))
            pixels = read_exact(board, head["bytes"], REPLY_TIMEOUT_S)
            if head["bytes"] != FRAME_BYTES or fnv(pixels) != head["hash"] or head["hash"] != fields(line)["grey_hash"]:
                raise RuntimeError(f"Frame {k}'s grey frame came back damaged or differs from the one it stepped on")
            out.write(pixels)
            if k % 50 == 0:
                print(f"frame {k}/{len(jpegs)}", flush=True)
    return lines


def finish(board: Board) -> dict:
    board.send(b"end\n")
    out = {}
    for prefix in ("map", "voxels", "dirs", "plan", "gyro", "imu", "live", "end"):
        out[prefix] = fields(board.reply((prefix + " ",), REPLY_TIMEOUT_S))
    return out


def stream(prepared: Path, port: str, label: str, check: bool, cores: int = 2, baud: int = LINK_BAUD) -> dict:
    feed = json.loads((prepared / "feed.json").read_text(encoding="utf-8"))
    frames_file = prepared / feed["frames_file"]
    if hashlib.sha256(frames_file.read_bytes()).hexdigest() != feed["frames_sha256"]:
        raise ValueError(f"{frames_file.name} differs from the one feed.json describes")
    jpegs = read_jpegs(frames_file, feed["frames"])
    folder = prepared / "board"
    record_path, log_path = folder / f"{label}.json", folder / f"{label}_serial.log"
    grey_path = folder / f"{label}_grey.bin"
    if record_path.exists() or log_path.exists() or (check and grey_path.exists()):
        raise FileExistsError(f"Refusing to overwrite board/{label}*; choose another --label")
    folder.mkdir(parents=True, exist_ok=True)
    board = Board(port, baud)
    board.port.write_timeout = WRITE_TIMEOUT_S
    try:
        ready = open_session(board, feed, cores)
        began = time.monotonic()
        if check:
            sent, lines = [], check_frames(board, jpegs, feed["jpeg_fnv"], grey_path)
        else:
            sent, lines = play(board, jpegs, feed["jpeg_fnv"], feed["rate_hz"])
        wall = time.monotonic() - began
        end = finish(board)
    finally:
        board.close()
        with log_path.open("x", encoding="utf-8", newline="\n") as out:  # kept when the run fails too
            out.write("\n".join(board.lines) + "\n")
    record = summarise_run(feed, ready, sent, lines, end, check, wall)
    with record_path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(record, out, indent=1, allow_nan=False)
    return record


def summarise_run(feed: dict, ready: dict, sent: list, lines: list, end: dict, check: bool, wall: float) -> dict:
    answers = answered(lines)
    frames = []
    for k in range(feed["frames"]):
        t, line = answers.get(k, (None, ""))
        kind = line.split()[0] if line else "none"
        entry = {"index": k, "video_index": feed["video_indices"][k], "result": kind}
        if kind in ("first", "step"):
            entry.update(fields(line))
        if sent:
            entry.update(sent[k])
            if t is not None:
                entry["answer_s"] = t - sent[k]["written_s"]  # from the frame's last byte leaving to its answer
        frames.append(entry)
    stepped = [f for f in frames if f["result"] == "step"]
    timing = {key: summarise([f[key] / 1000.0 for f in stepped]) for key in TIMINGS if stepped and key in stepped[0]}
    if stepped:
        timing["work_ms"] = summarise([sum(f[key] for key in WORK) / 1000.0 for f in stepped])
    record = {
        "mode": "check" if check else "live",
        "rate_hz": feed["rate_hz"],
        "frames": feed["frames"],
        "source_size": feed["source_size"],
        "jpeg_quality": feed["jpeg_quality"],
        "jpeg_bytes": feed["jpeg_bytes"],
        "cpu_mhz": ready.get("cpu_mhz"),
        "config": ready.get("config"),
        "max_features": ready.get("max_features"),
        **{key: ready.get(key) for key in SOURCES},
        "results": {
            kind: sum(f["result"] == kind for f in frames) for kind in ("first", "step", "dropped", "error", "none")
        },
        "board_received": end["live"].get("received"),
        "board_dropped": end["live"].get("dropped"),
        "wall_seconds": wall,
        "timing": {key.replace("_us", "_ms"): value for key, value in timing.items()},
        "memory": {
            "heap_internal_free": end["end"].get("heap_internal_free"),
            "psram_free": end["live"].get("psram_free"),
            "worker_stack_free_bytes": end["live"].get("worker_stack_free_bytes"),
        },
        "voxels": end["voxels"],
        "plan": end["plan"],
        "per_frame": frames,
    }
    if sent:
        record["late_ms"] = summarise([s["late_s"] * 1000.0 for s in sent])
        answered_s = [f["answer_s"] for f in frames if "answer_s" in f and f["result"] in ("first", "step")]
        if answered_s:
            record["answer_ms"] = summarise([a * 1000.0 for a in answered_s])
    return record


# check


def check(prepared: Path, label: str, reconstruction: Path, output: Path) -> dict:
    """The --check run against the PC: steps, frames and paths (see the module's docstring)."""
    import tempfile

    import cv2
    import numpy as np

    from uwslam import mvo
    from uwslam.experiments.mask_motion import REFERENCE_TIME_SCALE as TIME_SCALE
    from uwslam.experiments.mask_motion import reference_poses
    from uwslam.paths import CLIP_RESULTS

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite; choose a fresh --output")
    feed = json.loads((prepared / "feed.json").read_text(encoding="utf-8"))
    record = json.loads((prepared / "board" / f"{label}.json").read_text(encoding="utf-8"))
    if record["mode"] != "check":
        raise ValueError(f"{label} is a live run; check needs a run streamed with --check")
    expected = {**mvo.sources_sha256(), **mvo.prep_sha256()}
    for key in SOURCES:
        if record.get(key) != expected[key]:  # a record from an older firmware lacks some
            raise ValueError(
                f"The board ran other sources than this checkout's ({key}): flash this build, stream again"
            )
    if any(record[key] != value for key, value in mvo.build_config().items()):
        raise ValueError(f"The board ran the {record['config']!r} build, this checkout {mvo.CONFIG!r}")
    raw = np.fromfile(prepared / "board" / f"{label}_grey.bin", dtype=np.uint8)
    board_frames = list(raw.reshape(-1, mvo.HEIGHT, mvo.WIDTH))
    if len(board_frames) != feed["frames"]:
        raise ValueError(f"{label}_grey.bin holds {len(board_frames)} frames; the feed has {feed['frames']}")
    jpegs = read_jpegs(prepared / feed["frames_file"], feed["frames"])
    board_hashes = [f["hash"] for f in record["per_frame"][1:]]
    poses = np.loadtxt(reconstruction / "images.txt")
    indices = feed["video_indices"]
    times = np.arange(indices[-1] + 1) / VIDEO_FPS
    ref_r, ref_c = reference_poses(poses, times, TIME_SCALE)
    camera = mvo.clip_camera()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        vo = mvo.build(Path(scratch))
        _, pc_hashes, _ = mvo.run_frames(vo, board_frames, camera, feed["seed"])
        decoded = [mvo.area_resize(vo, cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_GRAYSCALE)) for j in jpegs]
        paths = {}
        for name, frames in (("board", board_frames), ("pc_decoded", decoded)):
            paths[name] = mvo.over_seeds(vo, frames, indices, camera, times, ref_r, ref_c)
    summary = {
        "label": label,
        "frames": feed["frames"],
        "source_size": feed["source_size"],
        "jpeg_quality": feed["jpeg_quality"],
        "steps_equal": sum(a == b for a, b in zip(pc_hashes, board_hashes)),
        "steps": len(board_hashes),
        "decode_difference": difference(board_frames, decoded),
        "decode_difference_by_frame": [difference([b], [p]) for b, p in zip(board_frames, decoded)],
        "paths": paths,
        "pc_pipeline": pc_pipeline(CLIP_RESULTS / "mono_vo_c" / "pc_summary.json", feed["rate_hz"]),
        **{key: record[key] for key in SOURCES},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    return summary


def difference(board: list, pc: list) -> dict:
    """The board's frames less the PC's, in grey levels."""
    import numpy as np

    d = np.stack(board).astype(int) - np.stack(pc).astype(int)
    return {
        "mean_abs": float(np.abs(d).mean()),
        "mean": float(d.mean()),
        "max_abs": int(np.abs(d).max()),
        "share_equal": float((d == 0).mean()),
    }


def pc_pipeline(path: Path, rate: float) -> dict | None:
    """The same clip's scores at the rate from the PC tools' own 320x240 frames (uwslam.mvo compare), if present."""
    if not path.exists():
        return None
    runs = [r for r in json.loads(path.read_text(encoding="utf-8"))["runs"] if r["rate_hz"] == rate]
    return runs[0]["c_over_seeds"] if runs else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare", help="The camera's JPEG frames from the lake clip")
    make.add_argument("video", type=Path)
    make.add_argument("--reconstruction", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--rate", type=float, default=RATE_HZ)
    make.add_argument("--quality", type=int, default=QUALITY)
    make.add_argument("--sampling", choices=SAMPLINGS, default="420", help="the JPEGs' chroma subsampling")
    play_parser = commands.add_parser("stream", help="Play the frames to the board and record what it did")
    play_parser.add_argument("prepared", type=Path)
    play_parser.add_argument("--port", required=True)
    play_parser.add_argument("--label", required=True)
    play_parser.add_argument("--check", action="store_true", help="No pacing: each frame's step, then its grey frame")
    play_parser.add_argument("--cores", type=int, choices=(1, 2), default=2)
    verify = commands.add_parser("check", help="A --check run against the PC")
    verify.add_argument("prepared", type=Path)
    verify.add_argument("--label", required=True)
    verify.add_argument("--reconstruction", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            json.dumps(
                prepare(args.video, args.reconstruction, args.output, args.rate, args.quality, args.sampling), indent=1
            )
        )
    elif args.command == "stream":
        record = stream(args.prepared, args.port, args.label, args.check, args.cores)
        print(json.dumps({key: record[key] for key in ("mode", "results", "board_dropped", "timing")}, indent=1))
    else:
        summary = check(args.prepared, args.label, args.reconstruction, args.output)
        print(json.dumps({key: summary[key] for key in ("steps_equal", "steps", "decode_difference")}, indent=1))


if __name__ == "__main__":
    main()
