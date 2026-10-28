"""Streams 320x240 grayscale frames (and optional water masks) to firmware/esp32s3_mono_vo over its
USB-Serial-JTAG port and checks the board against the PC build of the same C code (core/mono_vo*.c, core/mono_map.c).
Needs pyserial and no numpy, so it also runs in the ESP-IDF Python environment. With --expected-cloud it also checks
the mask-independent 3D point cloud, verifies every exported float's bits and can save that cloud to a PLY file. With
--expected-voxels (from `python -m uwslam.voxel_map`) every step's voxel count and filed observations and the final voxel
map's hash must equal the PC's; with --expected-dirs (from `python -m uwslam.obstacle_dir`, masks required) every
frame's obstacle directions (the choice and the blocked set) and the final state's hash; with --expected-plan (from
`python -m uwslam.path_plan`) every step's plan and the final plan's hash.

The frames file contains raw 320x240 grey frames back to back, from `python -m uwslam.mvo compare --board-frames` or
`python -m uwslam.mono_cloud --board-files`. Its VO expectation gives the camera, the seed, the sources' SHA-256
(core_sha256 covers all of the odometry's units, see uwslam.mvo.sources_sha256) and every step's FNV-1a hash. With
--masks (what `python -m uwslam.mono_map --board-masks` wrote) every frame also carries its mask, the board builds the
obstacle map, and the map's hash must equal the one in board_expected_map_10hz.json. A map-policy check also compares
the explicit continuity state at every step and at the end: it means no LOST was reported since start, not pose accuracy.
With board_expected_5hz.json (and
board_expected_map_5hz.json) the board gets every second frame and mask: the clip at 5 Hz. A board that reports other
sources is refused, and so is one whose vector tracker kernels and safe edge dispatch give other integers than the
portable ones on random windows (its `selftest`, run before every stream). Every step's hash must
match. The record
(per-step kinds, counts and stage times, their summary, the memory report and the matches) goes to board/<label>.json
beside the expected file, and the board's raw lines to board/<label>_serial.log.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import re
import statistics
import struct
import time
from pathlib import Path

from uwslam.paths import LAKE_CLIP_RESULTS

FRAME_BYTES = 320 * 240
MASK_RECORD = 1 + 60 * 80 // 8  # a flag byte, then the packed cells
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619
RESULTS = LAKE_CLIP_RESULTS / "mono_vo_c"
READY_TIMEOUT_S = 20.0
REPLY_TIMEOUT_S = 30.0
SELFTEST_CASES, SELFTEST_SEED = 2000, 12345
SELFTEST_TIMEOUT_S = 120.0
SOURCES = ("core_sha256", "header_sha256")
MAP_SOURCES = ("map_sha256", "map_header_sha256")
MAP_POLICY = 1
CLOUD_POLICY, CLOUD_CAPACITY, CLOUD_POINT_BYTES = 1, 512, 24
CLOUD_BYTES = CLOUD_CAPACITY * CLOUD_POINT_BYTES + 12
CLOUD_SOURCES = ("cloud_sha256", "cloud_header_sha256")
VOXEL_SOURCES = ("voxel_sha256", "voxel_header_sha256")
DIR_SOURCES = ("dir_sha256", "dir_header_sha256")
PLAN_SOURCES = ("plan_sha256", "plan_header_sha256")
VOXEL_CAPACITY, VOXEL_SLOTS, DIRECTIONS = 1024, 2048, 15  # VM_CAPACITY, VM_SLOTS, OD_DIRECTIONS
VOXEL_BYTES = VOXEL_CAPACITY * 10 + VOXEL_SLOTS * 2 + 4 + 7 * 4  # sizeof(vm_map_t)


def fnv(data: bytes) -> int:
    value = FNV_OFFSET
    for byte in data:
        value = ((value ^ byte) * FNV_PRIME) & 0xFFFFFFFF
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def fields(line: str) -> dict:
    """The key=value words of a board line, numbers as int when they are."""
    out = {}
    for word in line.split()[1:]:
        if "=" in word:
            key, value = word.split("=", 1)
            out[key] = int(value) if value.lstrip("-").isdigit() else value
    return out


def validate_expected_map(expected_map: dict, expected: dict) -> None:
    """Refuse old or inconsistent map expectations before opening the board."""
    if not isinstance(expected_map, dict):
        raise ValueError("The map expectation must be a JSON object with map_policy=1")
    if type(expected_map.get("map_policy")) is not int or expected_map["map_policy"] != MAP_POLICY:
        raise ValueError("The map expectation needs map_policy=1: regenerate it with the current PC code")
    if expected_map.get("step_hashes") != expected.get("step_hashes"):
        raise ValueError("The map run and the odometry run disagree on the steps: re-run both on the same sources")
    steps = expected.get("step_hashes")
    continuity = expected_map.get("map_continuity")
    if (not isinstance(steps, list) or len(steps) != expected.get("frames", 0) - 1
            or not isinstance(continuity, list) or len(continuity) != len(steps)
            or any(type(value) is not int or value not in (0, 1) for value in continuity)
            or any(a < b for a, b in zip(continuity, continuity[1:]))):
        raise ValueError("The map expectation needs one non-recovering 0/1 map_continuity value per step")
    final = expected_map.get("map_continuous")
    if type(final) is not int or final not in (0, 1) or final != (continuity[-1] if continuity else 1):
        raise ValueError("The map expectation's map_continuous must match its final step")


def check_continuity(reply: dict, expected: int, where: str) -> None:
    actual = reply.get("map_continuous")
    if type(actual) is not int or actual not in (0, 1) or actual != expected:
        raise RuntimeError(f"Board {where} map_continuous={actual!r}; expected {expected}")


def validate_expected_cloud(cloud: dict, expected: dict) -> None:
    """Check the complete PC cloud run before opening a serial port."""
    if not isinstance(cloud, dict) or type(cloud.get("cloud_policy")) is not int or cloud["cloud_policy"] != CLOUD_POLICY:
        raise ValueError("The cloud expectation needs cloud_policy=1")
    for key in (*CLOUD_SOURCES, "source_sha256", "frames_sha256"):
        value = cloud.get(key)
        if key == "source_sha256" and value is None:
            continue
        if not is_sha256(value):
            raise ValueError(f"Invalid {key} in cloud expectation")
    steps, progress = expected.get("step_hashes"), cloud.get("cloud_progress")
    if (type(expected.get("frames")) is not int or expected["frames"] < 1
            or type(cloud.get("frames")) is not int or cloud["frames"] != expected["frames"]
            or cloud.get("step_hashes") != steps
            or not isinstance(steps, list) or len(steps) != expected["frames"] - 1
            or not isinstance(progress, list) or len(progress) != len(steps)):
        raise ValueError("The cloud expectation and VO run must cover the same frames and steps")
    frame_hashes = cloud.get("frame_sha256")
    if not isinstance(frame_hashes, list) or len(frame_hashes) != expected["frames"] or not all(map(is_sha256, frame_hashes)):
        raise ValueError("The cloud expectation needs one SHA-256 per streamed frame")
    previous = {"count": 0, "dropped": 0, "continuous": 1}
    for item in progress:
        if not isinstance(item, dict) or any(type(item.get(key)) is not int for key in ("count", "accepted", "dropped", "continuous")):
            raise ValueError("Invalid cloud progress")
        if (not previous["count"] <= item["count"] <= CLOUD_CAPACITY or not 0 <= item["accepted"] <= 250
                or not previous["dropped"] <= item["dropped"] <= 0xffffffff
                or item["continuous"] not in (0, 1) or item["continuous"] > previous["continuous"]
                or item["count"] - previous["count"] > item["accepted"]
                or (not item["continuous"] and (item["accepted"] != 0 or item["count"] != previous["count"]
                                                or item["dropped"] != previous["dropped"]))):
            raise ValueError("Inconsistent cloud progress")
        previous = item
    if any(type(cloud.get("cloud_" + key)) is not int or cloud["cloud_" + key] != previous[key]
           for key in ("count", "dropped", "continuous")):
        raise ValueError("Final cloud state does not match its progress")
    if type(cloud.get("cloud_hash")) is not int or not 0 <= cloud["cloud_hash"] <= 0xffffffff:
        raise ValueError("Invalid cloud hash")


def validate_expected_voxels(voxels: dict, expected: dict) -> None:
    """Refuse a voxel-map expectation that does not come from the same odometry run, before opening the board."""
    if not isinstance(voxels, dict) or voxels.get("step_hashes") != expected.get("step_hashes"):
        raise ValueError("The voxel run and the odometry run disagree on the steps: re-run both on the same sources")
    progress = voxels.get("voxel_progress")
    if (voxels.get("frames") != expected.get("frames") or not isinstance(progress, list)
            or len(progress) != len(expected["step_hashes"])
            or any(not isinstance(item, dict) or type(item.get("count")) is not int or type(item.get("filed")) is not int
                   or not 0 <= item["count"] <= VOXEL_CAPACITY or not 0 <= item["filed"] <= 250 for item in progress)):
        raise ValueError("The voxel expectation needs one voxel count and filed number per step")
    if type(voxels.get("voxel_hash")) is not int or not all(is_sha256(voxels.get(key)) for key in VOXEL_SOURCES):
        raise ValueError("The voxel expectation needs the map's hash and its sources' SHA-256")


def validate_expected_dirs(dirs: dict, expected: dict) -> None:
    """Refuse a direction expectation for another number of frames, before opening the board."""
    frames = expected.get("frames")
    if not isinstance(dirs, dict) or dirs.get("frames") != frames:
        raise ValueError("The direction expectation must cover the odometry run's frames")
    lists = [dirs.get(key) for key in ("choices", "blocked_bits", "with_mask")]
    if (any(not isinstance(values, list) or len(values) != frames for values in lists)
            or any(type(c) is not int or not -1 <= c < DIRECTIONS for c in dirs["choices"])
            or any(type(b) is not int or not 0 <= b < 1 << DIRECTIONS for b in dirs["blocked_bits"])
            or any(type(m) is not bool for m in dirs["with_mask"])
            or type(dirs.get("dir_hash")) is not int or not all(is_sha256(dirs.get(key)) for key in DIR_SOURCES)):
        raise ValueError("The direction expectation needs one choice, blocked set and mask flag per frame, the state's "
                         "hash and its sources' SHA-256")


def validate_expected_plan(plan: dict, expected: dict) -> None:
    """Refuse a plan expectation that does not come from the same odometry run, before opening the board."""
    if not isinstance(plan, dict) or plan.get("step_hashes") != expected.get("step_hashes"):
        raise ValueError("The plan run and the odometry run disagree on the steps: re-run both on the same sources")
    steps = len(expected["step_hashes"])
    lists = [plan.get(key) for key in ("actions", "directions", "turns", "plan_hashes")]
    if (any(not isinstance(values, list) or len(values) != steps
            or any(type(v) is not int for v in values) for values in lists)
            or type(plan.get("plan_hash")) is not int or not all(is_sha256(plan.get(key)) for key in PLAN_SOURCES)):
        raise ValueError("The plan expectation needs one action, direction, turn and hash per step, the final hash and "
                         "its sources' SHA-256")


def check_dirs(reply: dict, dirs: dict, index: int, mismatches: list[int]) -> None:
    """Note frame `index` when the board's directions differ from the PC's."""
    if reply.get("dir_choice") != dirs["choices"][index] or reply.get("dir_blocked") != dirs["blocked_bits"][index]:
        mismatches.append(index)


def read_cloud(board, end: dict) -> dict:
    """Read every exported float bit pattern and verify the complete fixed-size C map hash."""
    board.send(b"cloud\n")
    header = fields(board.reply(("cloud ",), REPLY_TIMEOUT_S))
    for key in ("count", "dropped", "continuous", "hash"):
        if type(header.get(key)) is not int or header[key] != end.get("cloud_" + key):
            raise RuntimeError(f"Cloud export disagrees with end: {key}")
    if (not 0 <= header["count"] <= CLOUD_CAPACITY or header["continuous"] not in (0, 1)
            or not 0 <= header["dropped"] <= 0xffffffff or not 0 <= header["hash"] <= 0xffffffff):
        raise RuntimeError("Invalid cloud export size or continuity")
    raw, points, ids = bytearray(CLOUD_BYTES), [], set()
    for index in range(header["count"]):
        item = fields(board.reply(("point ",), REPLY_TIMEOUT_S))
        names = ("x_bits", "y_bits", "z_bits", "id", "spread_bits", "observations")
        if (item.get("index") != index or any(type(item.get(key)) is not int or not 0 <= item[key] <= 0xffffffff for key in names)
                or item["id"] in ids or item["observations"] == 0):
            raise RuntimeError(f"Invalid cloud point {index}")
        bits = [item[key] for key in names]
        struct.pack_into("<6I", raw, index * CLOUD_POINT_BYTES, *bits)
        xyz = [struct.unpack("<f", struct.pack("<I", value))[0] for value in bits[:3]]
        spread = struct.unpack("<f", struct.pack("<I", bits[4]))[0]
        if not all(math.isfinite(value) for value in (*xyz, spread)):
            raise RuntimeError(f"Non-finite cloud point {index}")
        ids.add(item["id"])
        points.append({"id": item["id"], "xyz": xyz, "spread": spread, "observations": item["observations"]})
    footer = fields(board.reply(("cloud_end ",), REPLY_TIMEOUT_S))
    if footer.get("count") != header["count"]:
        raise RuntimeError("Cloud export ended with a different count")
    struct.pack_into("<IIi", raw, CLOUD_CAPACITY * CLOUD_POINT_BYTES,
                     header["count"], header["dropped"], header["continuous"])
    if fnv(raw) != header["hash"]:
        raise RuntimeError("Cloud export does not match the board's map hash")
    return {**header, "points": points}


def write_cloud_ply(path: Path, cloud: dict) -> None:
    """Write the self-checked board cloud without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as out:
        out.write("ply\nformat ascii 1.0\n")
        out.write("comment Sparse rolling monocular cloud; arbitrary odometry units, not free space\n")
        out.write("comment First camera axes: x right, y down, z forward\n")
        out.write(f"comment continuous {cloud['continuous']} evicted {cloud['dropped']} cloud_hash {cloud['hash']}\n")
        verdict = "matched" if cloud["match"] is True else "not_compared_partial" if cloud["match"] is None else "different"
        out.write(f"comment pc_match {verdict}\n")
        for name in ("cloud_sha256", "cloud_header_sha256", "source_sha256"):
            if cloud.get(name) is not None:
                out.write(f"comment {name} {cloud[name]}\n")
        if cloud.get("frames") is not None:
            out.write(f"comment processed_frames {cloud['frames']}\n")
        out.write(f"element vertex {len(cloud['points'])}\n")
        out.write("property float x\nproperty float y\nproperty float z\nproperty uint id\n")
        out.write("property float spread\nproperty uint observations\nend_header\n")
        for point in cloud["points"]:
            x, y, z = point["xyz"]
            out.write(f"{x:.9g} {y:.9g} {z:.9g} {point['id']} {point['spread']:.9g} {point['observations']}\n")


class Board:
    """Line-level access to the firmware, keeping every line it sends for the log."""

    def __init__(self, port_name: str):
        import serial
        self.port = serial.Serial()
        self.port.port, self.port.baudrate, self.port.timeout = port_name, 115200, 0.5
        self.port.dtr = False  # opening must not reset the chip
        self.port.rts = False
        self.port.open()
        self.lines: list[str] = []

    def close(self) -> None:
        self.port.close()

    def send(self, data: bytes) -> None:
        self.port.write(data)
        self.port.flush()

    def reply(self, prefixes: tuple[str, ...], timeout: float) -> str:
        """The next line starting with one of `prefixes`; `error` lines raise, `ready` lines are skipped."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.port.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            self.lines.append(line)
            if line.startswith("error"):
                raise RuntimeError(f"Board reported: {line}")
            if line.startswith(prefixes):
                return line
        raise TimeoutError(f"No {' or '.join(prefixes)} line from the board within {timeout:.0f} s")


def summarise(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"mean": statistics.fmean(ordered), "median": statistics.median(ordered),
            "p90": ordered[int(0.9 * (len(ordered) - 1))], "max": ordered[-1]}


def stream(port_name: str, frames_path: Path, expected_path: Path, limit: int | None = None, cores: int = 2,
           masks_path: Path | None = None, expected_map_path: Path | None = None,
           simd: bool = True, expected_cloud_path: Path | None = None, expected_voxels_path: Path | None = None,
           expected_dirs_path: Path | None = None, expected_plan_path: Path | None = None) -> tuple[dict, list[str]]:
    """Runs the frames (the first `limit` of them, when given) through the board on `cores` cores, tracking with the
    vector kernels or (simd False) the portable ones and, with masks, building the map; the record and the board's
    lines."""
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if masks_path is not None and expected_map_path is None:
        raise ValueError("A map expectation is required when streaming masks")
    expected_map = json.loads(expected_map_path.read_text(encoding="utf-8")) if masks_path is not None else None
    if expected_map is not None:
        validate_expected_map(expected_map, expected)
    expected_cloud = json.loads(expected_cloud_path.read_text(encoding="utf-8")) if expected_cloud_path is not None else None
    if expected_cloud is not None:
        validate_expected_cloud(expected_cloud, expected)
    expected_voxels = json.loads(expected_voxels_path.read_text(encoding="utf-8")) if expected_voxels_path else None
    if expected_voxels is not None:
        validate_expected_voxels(expected_voxels, expected)
    expected_dirs = json.loads(expected_dirs_path.read_text(encoding="utf-8")) if expected_dirs_path else None
    if expected_dirs is not None:
        if masks_path is None:
            raise ValueError("The obstacle directions need the masks")
        validate_expected_dirs(expected_dirs, expected)
    expected_plan = json.loads(expected_plan_path.read_text(encoding="utf-8")) if expected_plan_path else None
    if expected_plan is not None:
        validate_expected_plan(expected_plan, expected)
    if limit is not None and limit < 1:
        raise ValueError("The frame limit must be at least 1")
    every = expected.get("every", 1)  # 2: every second frame of the 10 Hz file, the clip at 5 Hz
    if type(every) is not int or every < 1:
        raise ValueError("Expected frame stride must be a positive integer")
    frame_bytes = frames_path.stat().st_size
    total = frame_bytes // FRAME_BYTES
    if total * FRAME_BYTES != frame_bytes:
        raise ValueError(f"{frames_path.name} does not hold whole 320x240 frames")
    if masks_path is not None and masks_path.stat().st_size != total * MASK_RECORD:
        raise ValueError(f"{masks_path.name} does not hold one mask record per frame")
    if expected_cloud is not None and sha256_file(frames_path) != expected_cloud["frames_sha256"]:
        raise ValueError("The frame file SHA-256 differs from the PC cloud run")
    count = (total + every - 1) // every
    if count != expected["frames"]:
        raise ValueError(f"{frames_path.name} gives {count} frames at this rate; the expected run has {expected['frames']}")
    if count < 1:
        raise ValueError("The run must contain at least one frame")
    full_run = limit is None or limit >= count
    count = min(count, limit) if limit is not None else count
    wanted = expected["step_hashes"][:count - 1]
    wanted_continuity = expected_map["map_continuity"][:count - 1] if expected_map is not None else []
    wanted_final = wanted_continuity[-1] if wanted_continuity else 1
    resources = ExitStack()
    try:
        frame_stream = resources.enter_context(frames_path.open("rb"))
        mask_stream = resources.enter_context(masks_path.open("rb")) if masks_path is not None else None
        board = Board(port_name)
        resources.callback(board.close)
        ready = fields(board.reply(("ready",), READY_TIMEOUT_S))
        if expected_map is not None and (type(ready.get("map_policy")) is not int or ready["map_policy"] != MAP_POLICY):
            raise RuntimeError("The board needs map_policy=1: rebuild and flash the current firmware")
        if expected_cloud is not None:
            if (type(ready.get("cloud_policy")) is not int or ready["cloud_policy"] != CLOUD_POLICY
                    or ready.get("cloud_capacity") != CLOUD_CAPACITY or ready.get("cloud_bytes") != CLOUD_BYTES):
                raise RuntimeError("The board needs the current cloud policy and memory layout")
            for key in CLOUD_SOURCES:
                if key not in expected_cloud or ready.get(key) != expected_cloud[key]:
                    raise RuntimeError(f"The board's {key} differs from the PC run's")
        for key, source in ([(key, expected_voxels) for key in VOXEL_SOURCES] if expected_voxels is not None else []) + \
                ([(key, expected_dirs) for key in DIR_SOURCES] if expected_dirs is not None else []) + \
                ([(key, expected_plan) for key in PLAN_SOURCES] if expected_plan is not None else []):
            if ready.get(key) != source[key]:
                raise RuntimeError(f"The board's {key} differs from the PC run's: rebuild and flash the firmware")
        if expected_voxels is not None and ready.get("voxel_bytes") != VOXEL_BYTES:
            raise RuntimeError("The board's voxel map has another memory layout")
        for key in SOURCES + (MAP_SOURCES if expected_map is not None else ()):
            source = expected_map if expected_map is not None else expected
            if key not in ready or key not in source or ready[key] != source[key]:
                raise RuntimeError(f"The board's {key} differs from the PC run's: rebuild and flash the firmware, or "
                                   "re-run the PC side on the current sources")
        board.send(("start " + " ".join(str(bits) for bits in expected["camera_bits"])
                    + f" {expected['seed']} {cores} {int(simd)}\n").encode())
        started = fields(board.reply(("started",), REPLY_TIMEOUT_S))
        board.send(f"selftest {SELFTEST_CASES} {SELFTEST_SEED}\n".encode())
        selftest = fields(board.reply(("selftest",), SELFTEST_TIMEOUT_S))
        if selftest["template_mismatches"] or selftest["sums_mismatches"] or selftest["block_mismatches"]:
            raise RuntimeError(f"The board's vector kernels differ from the portable ones: {selftest}")
        steps, first, continuous_from_kind = [], None, 1
        voxel_mismatches, dir_mismatches, plan_mismatches = [], [], []
        checked = expected_map is not None or expected_cloud is not None or expected_voxels is not None \
            or expected_dirs is not None or expected_plan is not None
        began = time.monotonic()
        for index in range(count):
            frame_stream.seek(index * every * FRAME_BYTES)
            data = frame_stream.read(FRAME_BYTES)
            if len(data) != FRAME_BYTES:
                raise RuntimeError(f"Frame {index} became incomplete during replay")
            if expected_cloud is not None and hashlib.sha256(data).hexdigest() != expected_cloud["frame_sha256"][index]:
                raise RuntimeError(f"Frame {index} SHA-256 differs from the PC cloud run")
            record = b""
            if mask_stream is not None:
                mask_stream.seek(index * every * MASK_RECORD)
                record = mask_stream.read(MASK_RECORD)
                if len(record) != MASK_RECORD:
                    raise RuntimeError(f"Mask {index} became incomplete during replay")
            if expected_dirs is not None and bool(record and record[0]) != expected_dirs["with_mask"][index]:
                raise RuntimeError(f"Mask {index} is {'present' if record and record[0] else 'absent'}, unlike in the "
                                   "direction expectation")
            if record and record[0]:
                board.send(f"frame {index} {fnv(data)} {fnv(record[1:])}\n".encode() + data + record[1:])
            else:
                board.send(f"frame {index} {fnv(data)}\n".encode() + data)
            line = board.reply(("first", "step"), REPLY_TIMEOUT_S)
            if line.startswith("first"):
                if checked and index != 0:
                    raise RuntimeError(f"Board sent first instead of step for frame {index}")
                first = fields(line)
                if checked and first.get("index") != index:
                    raise RuntimeError(f"Board sent an invalid first frame: {line}")
                if expected_dirs is not None:
                    check_dirs(first, expected_dirs, index, dir_mismatches)
            else:
                step = fields(line)
                if checked:
                    if index == 0 or step.get("index") != index or step.get("kind") not in (0, 1, 2):
                        raise RuntimeError(f"Board sent an invalid step for frame {index}: {line}")
                    if step["kind"] == 0:
                        continuous_from_kind = 0
                if expected_map is not None:
                    check_continuity(step, wanted_continuity[index - 1], f"step {index}")
                    check_continuity(step, continuous_from_kind, f"step {index} kind history")
                if expected_cloud is not None:
                    wanted_cloud = expected_cloud["cloud_progress"][index - 1]
                    for key in ("count", "accepted", "dropped", "continuous"):
                        actual = step.get("cloud_" + key)
                        if type(actual) is not int or actual != wanted_cloud[key]:
                            raise RuntimeError(f"Board step {index} cloud_{key}={actual!r}; expected {wanted_cloud[key]}")
                    if step["cloud_continuous"] != continuous_from_kind:
                        raise RuntimeError(f"Board step {index} cloud continuity disagrees with LOST history")
                    if any(type(step.get(key)) is not int or step[key] < 0 for key in ("total_us", "map_us", "cloud_us")):
                        raise RuntimeError(f"Board step {index} has invalid timing")
                if expected_voxels is not None:
                    wanted_voxels = expected_voxels["voxel_progress"][index - 1]
                    if (step.get("voxels") != wanted_voxels["count"] or step.get("voxel_filed") != wanted_voxels["filed"]):
                        voxel_mismatches.append(index)
                    if type(step.get("voxel_us")) is not int or step["voxel_us"] < 0:
                        raise RuntimeError(f"Board step {index} has invalid voxel timing")
                if expected_dirs is not None:
                    check_dirs(step, expected_dirs, index, dir_mismatches)
                    if type(step.get("dir_us")) is not int or step["dir_us"] < 0:
                        raise RuntimeError(f"Board step {index} has invalid direction timing")
                if expected_plan is not None:
                    if (step.get("plan") != expected_plan["actions"][index - 1]
                            or step.get("plan_dir") != expected_plan["directions"][index - 1]
                            or step.get("plan_turn") != expected_plan["turns"][index - 1]
                            or step.get("plan_hash") != expected_plan["plan_hashes"][index - 1]):
                        plan_mismatches.append(index)
                    if type(step.get("plan_us")) is not int or step["plan_us"] < 0:
                        raise RuntimeError(f"Board step {index} has invalid plan timing")
                steps.append(step)
            if index % 50 == 0:
                print(f"frame {index}/{count}", flush=True)
        board.send(b"end\n")
        board_map = fields(board.reply(("map",), REPLY_TIMEOUT_S))
        board_voxels = fields(board.reply(("voxels",), REPLY_TIMEOUT_S)) if expected_voxels is not None else None
        board_dirs = fields(board.reply(("dirs",), REPLY_TIMEOUT_S)) if expected_dirs is not None else None
        board_plan = fields(board.reply(("plan ",), REPLY_TIMEOUT_S)) if expected_plan is not None else None
        memory = fields(board.reply(("end",), REPLY_TIMEOUT_S))
        if expected_map is not None:
            check_continuity(board_map, wanted_final, "map")
            check_continuity(memory, wanted_final, "end")
        cloud = None
        if expected_cloud is not None:
            wanted_state = (expected_cloud["cloud_progress"][count - 2] if count > 1
                            else {"count": 0, "dropped": 0, "continuous": 1})
            for key in ("count", "dropped", "continuous"):
                actual = memory.get("cloud_" + key)
                if type(actual) is not int or actual != wanted_state[key]:
                    raise RuntimeError(f"Board end cloud_{key}={actual!r}; expected {wanted_state[key]}")
            if [step.get("hash") for step in steps] != wanted:
                raise RuntimeError("Board VO step hashes differ from the cloud's PC run")
            cloud = read_cloud(board, memory)
            if full_run and cloud["hash"] != expected_cloud["cloud_hash"]:
                raise RuntimeError("Board cloud hash differs from the PC's")
            cloud["match"] = cloud["hash"] == expected_cloud["cloud_hash"] if full_run else None
        wall = time.monotonic() - began
    finally:
        resources.close()
    hashes = [step["hash"] for step in steps]
    mismatches = [k for k, (got, want) in enumerate(zip(hashes, wanted)) if got != want]
    kinds = {name: sum(step["kind"] == code for step in steps) for code, name in enumerate(("lost", "rotation", "essential"))}
    stages = (("total", "track", "motion", "detect") + (("map",) if expected_map is not None or expected_cloud is not None else ())
              + (("cloud",) if expected_cloud is not None else ()) + (("voxel",) if expected_voxels is not None else ())
              + (("dir",) if expected_dirs is not None else ()) + (("plan",) if expected_plan is not None else ()))
    timing = {f"{stage}_ms": summarise([step[f"{stage}_us"] / 1000 for step in steps]) for stage in stages} if steps else {}
    if expected_cloud is not None and steps:
        timing["vo_cloud_map_ms"] = summarise([(step["total_us"] + step["cloud_us"] + step["map_us"]) / 1000
                                               for step in steps])
    record = {"task": "Monocular VO" + (" and obstacle grid" if expected_map is not None else "")
                      + (" and sparse cloud" if expected_cloud is not None else "")
                      + " on the ESP32-S3, replayed over USB against the PC build",
              "frames_file": frames_path.name, "expected_file": expected_path.name, "rate_hz": expected.get("rate_hz", 10.0),
              "cpu_mhz": ready["cpu_mhz"],
              **{key: ready[key] for key in SOURCES + (MAP_SOURCES if expected_map is not None else ())},
              "state_bytes": started["state_bytes"], "cores": started["cores"], "simd": bool(started["simd"]),
              "kernel_selftest": selftest, "frames": count,
              "steps": len(steps), "steps_expected": len(wanted),
              "all_match": not mismatches and len(steps) == len(wanted),
              "mismatched_steps": mismatches, "kinds": kinds, "first_frame": first, "timing": timing, "memory": memory,
              "wall_seconds": wall}
    if expected_map is not None:
        record["map"] = {"masks_file": masks_path.name, "expected_file": expected_map_path.name, **board_map,
                         "policy": MAP_POLICY, "continuous_expected": wanted_final,
                         "observations_expected": expected_map["map_observations"] if full_run else None,
                         "hash_expected": expected_map["map_hash"] if full_run else None,
                         "match": board_map["hash"] == expected_map["map_hash"] if full_run else None}
    if cloud is not None:
        record["cloud"] = {"policy": CLOUD_POLICY, "expected_file": expected_cloud_path.name, "frames": count,
                           **{key: expected_cloud.get(key) for key in (*CLOUD_SOURCES, "source_sha256")}, **cloud}
    if expected_voxels is not None:
        record["voxels"] = {"expected_file": expected_voxels_path.name, **board_voxels,
                            "hash_expected": expected_voxels["voxel_hash"] if full_run else None,
                            "match": False if voxel_mismatches
                            else board_voxels.get("hash") == expected_voxels["voxel_hash"] if full_run else None,
                            "mismatched_steps": voxel_mismatches,
                            **{key: expected_voxels[key] for key in VOXEL_SOURCES}}
    if expected_dirs is not None:
        record["dirs"] = {"expected_file": expected_dirs_path.name, **board_dirs,
                          "hash_expected": expected_dirs["dir_hash"] if full_run else None,
                          "match": False if dir_mismatches else board_dirs.get("hash") == expected_dirs["dir_hash"]
                          if full_run else None, "mismatched_frames": dir_mismatches,
                          **{key: expected_dirs[key] for key in DIR_SOURCES}}
    if expected_plan is not None:
        record["plan"] = {"expected_file": expected_plan_path.name, **board_plan,
                          "hash_expected": expected_plan["plan_hash"] if full_run else None,
                          "match": False if plan_mismatches else board_plan.get("hash") == expected_plan["plan_hash"]
                          if full_run else None, "mismatched_steps": plan_mismatches,
                          **{key: expected_plan[key] for key in PLAN_SOURCES}}
    record["per_step"] = steps
    return record, board.lines


BOARD_HEADING = "## ESP32-S3"


def write_board_section(results: Path) -> None:
    """Replace only the ESP32-S3 section of RESULTS.md from records in board/."""
    rows = []
    for path in sorted((results / "board").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        t = record["timing"]
        board_map = record.get("map")
        if board_map is None:
            map_cell = "—"
        else:
            verdict = {True: "equal to the PC's", False: "differs from the PC's", None: "not compared (part of the run)"}
            map_cell = f"{board_map['observations']} observations, hash {verdict[board_map['match']]}"
        map_time = f"{t['map_ms']['mean']:.1f} ms" if "map_ms" in t else "—"
        verdict = {True: "equal to the PC's", False: "differs from the PC's", None: "not compared (part of the run)"}
        extra = []
        if record.get("voxels"):
            extra.append(f"{record['voxels']['count']} voxels, hash {verdict[record['voxels']['match']]}"
                         + (f" ({t['voxel_ms']['mean']:.1f} ms)" if "voxel_ms" in t else ""))
        if record.get("dirs"):
            extra.append(f"directions {verdict[record['dirs']['match']]}"
                         + (f" ({t['dir_ms']['mean']:.2f} ms)" if "dir_ms" in t else ""))
        if record.get("plan"):
            extra.append(f"plans {verdict[record['plan']['match']]}"
                         + (f" ({t['plan_ms']['mean']:.2f} ms)" if "plan_ms" in t else ""))
        extra_cell = "; ".join(extra) if extra else "—"
        step_time = (f"{t['total_ms']['mean']:.0f} / {t['total_ms']['median']:.0f} / {t['total_ms']['max']:.0f} ms"
                     if "total_ms" in t else "—")
        stage_times = [f"{t[f'{stage}_ms']['mean']:.0f} ms" if f"{stage}_ms" in t else "—"
                       for stage in ("track", "motion", "detect")]
        kernels = "vector" if record.get("simd") else "C"
        rate = record.get("rate_hz", 10.0)
        rate_text = f"{rate:g} Hz" if isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0 else "unknown"
        rows.append(f"| `{path.stem}` | {rate_text} | {record['cores']} | {kernels} | {record['frames']} | "
                    f"{record['steps'] - len(record['mismatched_steps'])} of {record['steps']} | {map_cell} | "
                    f"{step_time} | {' | '.join(stage_times)} | {map_time} | {extra_cell} |")
    section = [BOARD_HEADING, "",
               "The host streams grayscale frames and optional masks over USB (`python -m uwslam.mvo_stream`); "
               "the board reports each step's hash and stage times. These timings exclude capture and USB transfer. "
               "Each record provides its own rate, core count and kernel choice. Before streaming, the board checks "
               "its vector dispatch against C on random windows. Records and raw serial lines are in `board/`.", "",
               "| Record | Rate | Cores | Kernels | Frames | Steps equal to the PC | Map | Per step, mean / median / max | "
               "Tracking | Motion | Detection | Map update | Voxel map / directions / plan (update) |",
               "|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
               *rows, "",
               "Re-run (after `uwslam.mvo compare --board-frames` and `uwslam.mono_map --board-masks`):", "", "```bash",
               "idf.py -C firmware/esp32s3_mono_vo -B firmware/esp32s3_mono_vo/build_240 -p <port> flash",
               "python -m uwslam.mvo_stream --port <port> --frames <frames file> --masks <masks file> --label <name>",
               "```", ""]
    report = results / "RESULTS.md"
    text = report.read_text(encoding="utf-8") if report.exists() else ""
    heading = re.search(r"(?m)^## ESP32-S3[ \t]*$", text)
    prefix, suffix = text, ""
    if heading:
        prefix = text[:heading.start()]
        later = re.search(r"(?m)^## [^\n]+", text[heading.end():])
        if later:
            suffix = text[heading.end() + later.start():]
    prefix = prefix.rstrip("\n") + "\n\n" if prefix.strip() else ""
    report.write_text(prefix + "\n".join(section) + ("\n" + suffix if suffix else ""), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", required=True, help="Serial port of the board")
    parser.add_argument("--frames", type=Path, required=True, help="Raw grayscale frames from uwslam.mvo or uwslam.mono_cloud")
    parser.add_argument("--expected", type=Path, default=RESULTS / "board_expected_10hz.json")
    parser.add_argument("--masks", type=Path, help="The masks uwslam.mono_map wrote with --board-masks: build the map too")
    parser.add_argument("--expected-map", type=Path, help="Default: board_expected_map_<rate>hz.json in the lake map results")
    parser.add_argument("--expected-cloud", type=Path, help="Cloud expectation from uwslam.mono_cloud --board-files")
    parser.add_argument("--cloud-output", type=Path, help="Save the verified board cloud as a new PLY file")
    parser.add_argument("--expected-voxels", type=Path, help="Voxel-map expectation from uwslam.voxel_map "
                        "(board_expected_voxels_<rate>hz.json)")
    parser.add_argument("--expected-dirs", type=Path, help="Obstacle-direction expectation from uwslam.obstacle_dir "
                        "(board_expected_dirs_<rate>hz.json); needs --masks")
    parser.add_argument("--expected-plan", type=Path, help="Plan expectation from uwslam.path_plan "
                        "(board_expected_plan_<rate>hz.json); give the masks and directions it was made with")
    parser.add_argument("--label", required=True, help="Name of the saved record, e.g. 240mhz")
    parser.add_argument("--frames-limit", type=int, help="Stream only the first this many frames")
    parser.add_argument("--cores", type=int, choices=(1, 2), default=2, help="Run on one core or on both")
    parser.add_argument("--no-simd", action="store_true", help="Track with the portable C kernels, not the vector ones")
    args = parser.parse_args()
    if args.cloud_output is not None and args.expected_cloud is None:
        parser.error("--cloud-output requires --expected-cloud")
    if args.cloud_output is not None and args.cloud_output.exists():
        raise FileExistsError("Refusing to overwrite the cloud output")
    folder = args.expected.parent / "board"
    target, log = folder / f"{args.label}.json", folder / f"{args.label}_serial.log"
    if args.cloud_output is not None:
        reserved = (target, log, args.expected, args.frames, args.expected_cloud, args.masks,
                    args.expected_map, args.expected.parent / "RESULTS.md")
        if any(path is not None and args.cloud_output.resolve() == path.resolve() for path in reserved):
            raise ValueError("The cloud output must be distinct from replay inputs and records")
    if target.exists() or log.exists():
        raise FileExistsError(f"Refusing to overwrite board/{target.name}; choose another --label")
    rate = json.loads(args.expected.read_text(encoding="utf-8")).get("rate_hz", 10.0)
    expected_map = args.expected_map
    if args.masks is not None and expected_map is None:
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            parser.error("--expected-map is required with masks when the input frame rate is unknown")
        expected_map = LAKE_CLIP_RESULTS / "mono_map" / f"board_expected_map_{rate:g}hz.json"
    record, lines = stream(args.port, args.frames, args.expected, args.frames_limit, args.cores, args.masks,
                           expected_map, not args.no_simd, args.expected_cloud, args.expected_voxels, args.expected_dirs,
                           args.expected_plan)
    if args.cloud_output is not None:
        write_cloud_ply(args.cloud_output, record["cloud"])
    folder.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8", newline="\n") as out:
        out.write("\n".join(lines) + "\n")
    with target.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(record, out, indent=1, allow_nan=False)
    if args.expected_cloud is None:
        write_board_section(args.expected.parent)
    if not record["steps"]:
        print("One frame streamed; no odometry steps or timings yet")
        if "map" in record:
            print(f"map: continuous = {record['map']['map_continuous']}")
        if "cloud" in record:
            print(f"cloud: {record['cloud']['count']} points, continuous = {record['cloud']['continuous']}")
        return
    total = record["timing"]["total_ms"]
    print(f"{record['steps']} steps at {record['cpu_mhz']} MHz on {record['cores']} core(s), "
          f"{'vector' if record['simd'] else 'C'} kernels: all hashes match = "
          f"{record['all_match']} ({len(record['mismatched_steps'])} mismatched); {total['mean']:.0f} ms per step (median "
          f"{total['median']:.0f}, max {total['max']:.0f}); track {record['timing']['track_ms']['mean']:.0f} ms, motion "
          f"{record['timing']['motion_ms']['mean']:.0f} ms, detect {record['timing']['detect_ms']['mean']:.0f} ms")
    if "map" in record:
        expected_observations = record["map"]["observations_expected"]
        comparison = "not compared for a partial run" if expected_observations is None else f"PC {expected_observations}"
        print(f"map: {record['map']['observations']} observations filed ({comparison}), "
              f"hash match = {record['map']['match']}, {record['timing']['map_ms']['mean']:.1f} ms per update")
    if "voxels" in record:
        print(f"voxels: {record['voxels']['count']} held, hash match = {record['voxels']['match']}, "
              f"{len(record['voxels']['mismatched_steps'])} steps differ, {record['timing']['voxel_ms']['mean']:.2f} ms "
              "per update")
    if "dirs" in record:
        print(f"directions: hash match = {record['dirs']['match']}, {len(record['dirs']['mismatched_frames'])} frames "
              f"differ, {record['timing']['dir_ms']['mean']:.2f} ms per update")
    if "plan" in record:
        print(f"plans: hash match = {record['plan']['match']}, {len(record['plan']['mismatched_steps'])} steps differ, "
              f"{record['timing']['plan_ms']['mean']:.2f} ms per plan")
    if "cloud" in record:
        print(f"cloud: {record['cloud']['count']} points, {record['cloud']['dropped']} historical points evicted, "
              f"continuous = {record['cloud']['continuous']}, hash match = {record['cloud']['match']}; "
              f"VO + cloud + obstacle map {record['timing']['vo_cloud_map_ms']['mean']:.1f} ms/step "
              "(excludes capture, segmentation and USB)")


if __name__ == "__main__":
    main()
