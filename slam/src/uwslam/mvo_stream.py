"""Streams 320x240 grayscale frames (and optional water masks) to firmware/mono_vo_stream on an ESP32-P4 over its
console (a USB-UART bridge, or the chip's own USB-Serial-JTAG port) and checks the board against the PC build of the
same C code (core/mono_vo*.c, core/mono_map.c).
Needs pyserial and no numpy, so it also runs in the ESP-IDF Python environment. With --expected-cloud it also checks
the mask-independent 3D point cloud, verifies every exported float's bits and can save that cloud to a PLY file. With
--expected-voxels (from `python -m uwslam.voxel_map`) every step's voxel count and filed observations and the final voxel
map's hash must equal the PC's; with --expected-dirs (from `python -m uwslam.obstacle_dir`, masks required) every
frame's obstacle directions (the choice and the blocked set) and the final state's hash; with --expected-plan (from
`python -m uwslam.path_plan`) every step's plan and the final plan's hash. With --gyro (the readings file
`python -m uwslam.aqualoc board` writes beside an expectation with a gyro entry) the board gets the gyro's mounting
once and each step's readings before its frame, and how every step used the gyro and the final bias estimate must
equal the PC's. With --imu (imu.bin, beside an expectation with an imu entry) the same with each step's IMU readings
(the gyro's and the accelerometer's), after the accelerometer's calibrated bias, and the world's up the board measured
must equal the PC's too. With --sense (each frame's depth and range: the sense.bin `python -m uwslam.aqualoc board`
writes, or the ranger's from `python -m uwslam.ranger`, beside an expectation with a metric entry) the board gets the
ranger's beam once and each frame's readings before it, and every step's metric centre and scale must equal the PC's.

The frames file contains raw 320x240 grey frames back to back, from `python -m uwslam.mvo compare --board-frames` or
`python -m uwslam.mono_cloud --board-files`. Its VO expectation gives the camera, the seed, the sources' SHA-256
(core_sha256 covers all of the odometry's units, see uwslam.mvo.sources_sha256) and every step's FNV-1a hash. With
--masks (what `python -m uwslam.mono_map --board-masks` wrote) every frame also carries its mask, the board builds the
obstacle map, and the map's hash must equal the one in board_expected_map_10hz.json. A map-policy check also compares
the explicit continuity state at every step and at the end: it means no LOST was reported since start, not pose accuracy.
With board_expected_5hz.json (and
board_expected_map_5hz.json) the board gets every second frame and mask: the clip at 5 Hz. A board that reports other
sources is refused, and so is one whose vector window kernels give other integers than the portable ones on random
windows (its `selftest`, run before every stream). Every step's hash must
match. The record
(per-step kinds, counts and stage times, their summary, the memory report and the matches) goes to board/<label>.json
beside the expected file, and the board's raw lines to board/<label>_serial.log.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import struct
import time
from contextlib import ExitStack
from pathlib import Path

from uwslam.paths import CLIP_RESULTS

FRAME_BYTES = 320 * 240
MASK_RECORD = 1 + 60 * 80 // 8  # a flag byte, then the packed cells
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619
RESULTS = CLIP_RESULTS / "mono_vo_c"
READY_TIMEOUT_S = 20.0
REPLY_TIMEOUT_S = 30.0
LINK_BAUD = 921_600  # a UART link's rate (the firmware's LINK_BAUD); a USB-Serial-JTAG port ignores it. A CH340
# bridge's link at 2 Mbaud lost single bytes of the board's replies in two of three lake runs.
BUILD, MAX_FEATURES = "p4", 300  # the core's build as the board names it (uwslam.mvo.CONFIG), MVO_MAX_FEATURES
STEP_FIELDS = (
    "index",
    "kind",
    "tracks",
    "inliers",
    "scale_points",
    "hash",
    "detect_us",
    "track_us",
    "motion_us",
    "total_us",
)  # every step line carries these
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
VOXEL_CAPACITY, VOXEL_SLOTS = 4096, 8192  # VM_CAPACITY, VM_SLOTS
DIRECTIONS = 15  # OD_DIRECTIONS
GYRO_READINGS, GYRO_READING_BYTES = 256, 16  # the firmware's: the most readings a step takes, and one's size
IMU_READING_BYTES = 28  # an IMU reading: dt, the rate and the specific force
GYRO_USES = ("none", "translation", "rotation", "lost")  # MVO_GYRO_*
SENSE_BYTES = 8  # a frame's depth and range, little-endian float32, NaN where none (uwslam.mvo.sense_bytes)
METRIC_FIELDS = ("metric", "metric_x_bits", "metric_y_bits", "metric_z_bits", "scale_bits", "scale_var_bits")


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


def read_gyro_file(path: Path, steps: int, size: int = GYRO_READING_BYTES) -> list[bytes]:
    """The readings file's steps as the payloads of the firmware's gyro (or, size IMU_READING_BYTES, imu) command: per
    step a little-endian uint32 count, then that many readings of `size` bytes (dt, rate x, y, z and for an IMU the
    force x, y, z, as little-endian floats)."""
    data = path.read_bytes()
    payloads, at = [], 0
    while at < len(data):
        if at + 4 > len(data):
            raise ValueError(f"{path.name} ends inside a step's count")
        (count,) = struct.unpack_from("<I", data, at)
        if count > GYRO_READINGS:
            raise ValueError(
                f"{path.name}: step {len(payloads)} has {count} readings; the firmware takes {GYRO_READINGS}"
            )
        end = at + 4 + count * size
        if end > len(data):
            raise ValueError(f"{path.name} ends inside step {len(payloads)}'s readings")
        payloads.append(data[at + 4 : end])
        at = end
    if len(payloads) != steps:
        raise ValueError(f"{path.name} holds readings for {len(payloads)} steps; the expected run has {steps}")
    return payloads


def validate_expected_gyro(gyro: dict, expected: dict, imu: bool = False) -> None:
    """Refuse a gyro (or, imu True, an IMU) expectation that does not fit the run before opening the board."""
    bits = gyro.get("mount_bits")
    if not isinstance(bits, list) or len(bits) != 9 or not all(type(b) is int and 0 <= b <= 0xFFFFFFFF for b in bits):
        raise ValueError("The gyro expectation needs nine mount_bits")
    if not is_sha256(gyro.get("readings_sha256")):
        raise ValueError("The gyro expectation needs the readings file's SHA-256")
    used = gyro.get("used")
    if not isinstance(used, list) or len(used) != expected["frames"] - 1 or any(u not in (0, 1, 2, 3) for u in used):
        raise ValueError("The gyro expectation needs how each step used the gyro")
    final = gyro.get("final_bias_bits")
    if not isinstance(final, list) or len(final) != 3 or not all(type(b) is int for b in final):
        raise ValueError("The gyro expectation needs the final bias estimate's bits")
    dropped = gyro.get("forces_dropped")
    if imu and (type(dropped) is not int or dropped < 0):
        raise ValueError("The IMU expectation needs the count of forces the core left out")
    for key, what in (("accel_bias_bits", "the accelerometer's bias"), ("final_up_bits", "the final world up")):
        bits = gyro.get(key)
        if imu and (
            not isinstance(bits, list)
            or len(bits) != 3
            or not all(type(b) is int and 0 <= b <= 0xFFFFFFFF for b in bits)
        ):
            raise ValueError(f"The IMU expectation needs {what}'s bits")


def is_bits(values, count: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) == count
        and all(type(b) is int and 0 <= b <= 0xFFFFFFFF for b in values)
    )


def read_sense_file(path: Path, frames: int) -> list[tuple[int | None, int | None]]:
    """Each frame's depth and range from the sense file, as the bits the firmware's depth and range commands take;
    None where the core takes no reading (uwslam.mvo.as_readings): not finite, or a range not positive."""
    data = path.read_bytes()
    if len(data) != frames * SENSE_BYTES:
        raise ValueError(
            f"{path.name} holds {len(data)} bytes; the expected run's {frames} frames need {frames * SENSE_BYTES}"
        )
    readings = []
    for at in range(0, len(data), SENSE_BYTES):
        (depth, distance), bits = struct.unpack_from("<2f", data, at), struct.unpack_from("<2I", data, at)
        taken = (math.isfinite(depth), math.isfinite(distance) and distance > 0)
        readings.append(tuple(b if keep else None for keep, b in zip(taken, bits)))
    return readings


def validate_expected_metric(metric: dict, expected: dict) -> None:
    """Refuse a metric expectation that does not fit the run before opening the board."""
    steps = expected["frames"] - 1
    if not is_sha256(metric.get("sense_sha256")):
        raise ValueError("The metric expectation needs the sense file's SHA-256")
    if "beam_bits" not in metric or (metric["beam_bits"] is not None and not is_bits(metric["beam_bits"], 4)):
        raise ValueError("The metric expectation's beam needs four float bits (the axis and the cosine) or null")
    known = metric.get("known")
    if not isinstance(known, list) or len(known) != steps or any(k not in (0, 1) for k in known):
        raise ValueError("The metric expectation needs whether each step knew the scale")
    centres = metric.get("centre_bits")
    if not isinstance(centres, list) or len(centres) != steps or not all(is_bits(c, 3) for c in centres):
        raise ValueError("The metric expectation needs each step's metric centre bits")
    if not is_bits(metric.get("scale_bits"), steps):
        raise ValueError("The metric expectation needs each step's scale bits")
    if not is_bits(metric.get("variance_bits"), steps):
        raise ValueError("The metric expectation needs each step's scale variance bits")


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
    if (
        not isinstance(steps, list)
        or len(steps) != expected.get("frames", 0) - 1
        or not isinstance(continuity, list)
        or len(continuity) != len(steps)
        or any(type(value) is not int or value not in (0, 1) for value in continuity)
        or any(a < b for a, b in zip(continuity, continuity[1:]))
    ):
        raise ValueError("The map expectation needs one non-recovering 0/1 map_continuity value per step")
    final = expected_map.get("map_continuous")
    if type(final) is not int or final not in (0, 1) or final != (continuity[-1] if continuity else 1):
        raise ValueError("The map expectation's map_continuous must match its final step")


def check_continuity(reply: dict, expected: int, where: str) -> None:
    actual = reply.get("map_continuous")
    if type(actual) is not int or actual not in (0, 1) or actual != expected:
        raise RuntimeError(f"Board {where} map_continuous={actual!r}; expected {expected}")


def expectation_config(expected: dict) -> str:
    """The build an expectation was made with, which must be this one (BUILD with MAX_FEATURES)."""
    config = expected.get("config")
    if config != BUILD or expected.get("max_features", MAX_FEATURES) != MAX_FEATURES:
        raise ValueError(
            f"The expectation was made with the {config!r} build and {expected.get('max_features')} features, not "
            f"{BUILD!r} with {MAX_FEATURES}: make it again with this tree"
        )
    return config


def max_features(expected: dict) -> int:
    """MVO_MAX_FEATURES of the expectation's build: no step tracks, maps or files more points."""
    expectation_config(expected)
    return MAX_FEATURES


def voxel_capacity(expected: dict) -> int:
    """VM_CAPACITY of the expectation's build."""
    expectation_config(expected)
    return VOXEL_CAPACITY


def voxel_bytes(expected: dict) -> int:
    """sizeof(vm_map_t) in the expectation's build."""
    expectation_config(expected)
    return VOXEL_CAPACITY * 10 + VOXEL_SLOTS * 2 + 4 + 7 * 4


def check_same_config(name: str, auxiliary: dict, expected: dict) -> None:
    """Refuse a map, cloud, voxel or plan expectation made with another build than the odometry's (the obstacle
    directions come from the masks alone)."""
    try:
        expectation_config(auxiliary)
    except ValueError as error:
        raise ValueError(f"The {name} expectation: {error}") from None
    expectation_config(expected)


def validate_expected_cloud(cloud: dict, expected: dict) -> None:
    """Check the complete PC cloud run before opening a serial port."""
    if (
        not isinstance(cloud, dict)
        or type(cloud.get("cloud_policy")) is not int
        or cloud["cloud_policy"] != CLOUD_POLICY
    ):
        raise ValueError("The cloud expectation needs cloud_policy=1")
    for key in (*CLOUD_SOURCES, "source_sha256", "frames_sha256"):
        value = cloud.get(key)
        if key == "source_sha256" and value is None:
            continue
        if not is_sha256(value):
            raise ValueError(f"Invalid {key} in cloud expectation")
    steps, progress = expected.get("step_hashes"), cloud.get("cloud_progress")
    if (
        type(expected.get("frames")) is not int
        or expected["frames"] < 1
        or type(cloud.get("frames")) is not int
        or cloud["frames"] != expected["frames"]
        or cloud.get("step_hashes") != steps
        or not isinstance(steps, list)
        or len(steps) != expected["frames"] - 1
        or not isinstance(progress, list)
        or len(progress) != len(steps)
    ):
        raise ValueError("The cloud expectation and VO run must cover the same frames and steps")
    frame_hashes = cloud.get("frame_sha256")
    if (
        not isinstance(frame_hashes, list)
        or len(frame_hashes) != expected["frames"]
        or not all(map(is_sha256, frame_hashes))
    ):
        raise ValueError("The cloud expectation needs one SHA-256 per streamed frame")
    previous = {"count": 0, "dropped": 0, "continuous": 1}
    for item in progress:
        if not isinstance(item, dict) or any(
            type(item.get(key)) is not int for key in ("count", "accepted", "dropped", "continuous")
        ):
            raise ValueError("Invalid cloud progress")
        if (
            not previous["count"] <= item["count"] <= CLOUD_CAPACITY
            or not 0 <= item["accepted"] <= max_features(expected)
            or not previous["dropped"] <= item["dropped"] <= 0xFFFFFFFF
            or item["continuous"] not in (0, 1)
            or item["continuous"] > previous["continuous"]
            or item["count"] - previous["count"] > item["accepted"]
            or (
                not item["continuous"]
                and (
                    item["accepted"] != 0
                    or item["count"] != previous["count"]
                    or item["dropped"] != previous["dropped"]
                )
            )
        ):
            raise ValueError("Inconsistent cloud progress")
        previous = item
    if any(
        type(cloud.get("cloud_" + key)) is not int or cloud["cloud_" + key] != previous[key]
        for key in ("count", "dropped", "continuous")
    ):
        raise ValueError("Final cloud state does not match its progress")
    if type(cloud.get("cloud_hash")) is not int or not 0 <= cloud["cloud_hash"] <= 0xFFFFFFFF:
        raise ValueError("Invalid cloud hash")


def validate_expected_voxels(voxels: dict, expected: dict) -> None:
    """Refuse a voxel-map expectation that does not come from the same odometry run, before opening the board."""
    if not isinstance(voxels, dict) or voxels.get("step_hashes") != expected.get("step_hashes"):
        raise ValueError("The voxel run and the odometry run disagree on the steps: re-run both on the same sources")
    progress = voxels.get("voxel_progress")
    if (
        voxels.get("frames") != expected.get("frames")
        or not isinstance(progress, list)
        or len(progress) != len(expected["step_hashes"])
        or any(
            not isinstance(item, dict)
            or type(item.get("count")) is not int
            or type(item.get("filed")) is not int
            or not 0 <= item["count"] <= voxel_capacity(expected)
            or not 0 <= item["filed"] <= max_features(expected)
            for item in progress
        )
    ):
        raise ValueError("The voxel expectation needs one voxel count and filed number per step")
    if type(voxels.get("voxel_hash")) is not int or not all(is_sha256(voxels.get(key)) for key in VOXEL_SOURCES):
        raise ValueError("The voxel expectation needs the map's hash and its sources' SHA-256")


def validate_expected_dirs(dirs: dict, expected: dict) -> None:
    """Refuse a direction expectation for another number of frames, before opening the board."""
    frames = expected.get("frames")
    if not isinstance(dirs, dict) or dirs.get("frames") != frames:
        raise ValueError("The direction expectation must cover the odometry run's frames")
    lists = [dirs.get(key) for key in ("choices", "blocked_bits", "with_mask")]
    if (
        any(not isinstance(values, list) or len(values) != frames for values in lists)
        or any(type(c) is not int or not -1 <= c < DIRECTIONS for c in dirs["choices"])
        or any(type(b) is not int or not 0 <= b < 1 << DIRECTIONS for b in dirs["blocked_bits"])
        or any(type(m) is not bool for m in dirs["with_mask"])
        or type(dirs.get("dir_hash")) is not int
        or not all(is_sha256(dirs.get(key)) for key in DIR_SOURCES)
    ):
        raise ValueError(
            "The direction expectation needs one choice, blocked set and mask flag per frame, the state's "
            "hash and its sources' SHA-256"
        )


def validate_expected_plan(plan: dict, expected: dict) -> None:
    """Refuse a plan expectation that does not come from the same odometry run, before opening the board."""
    if not isinstance(plan, dict) or plan.get("step_hashes") != expected.get("step_hashes"):
        raise ValueError("The plan run and the odometry run disagree on the steps: re-run both on the same sources")
    steps = len(expected["step_hashes"])
    lists = [plan.get(key) for key in ("actions", "directions", "turns", "plan_hashes")]
    if (
        any(
            not isinstance(values, list) or len(values) != steps or any(type(v) is not int for v in values)
            for values in lists
        )
        or type(plan.get("plan_hash")) is not int
        or not all(is_sha256(plan.get(key)) for key in PLAN_SOURCES)
    ):
        raise ValueError(
            "The plan expectation needs one action, direction, turn and hash per step, the final hash and "
            "its sources' SHA-256"
        )


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
    if (
        not 0 <= header["count"] <= CLOUD_CAPACITY
        or header["continuous"] not in (0, 1)
        or not 0 <= header["dropped"] <= 0xFFFFFFFF
        or not 0 <= header["hash"] <= 0xFFFFFFFF
    ):
        raise RuntimeError("Invalid cloud export size or continuity")
    raw, points, ids = bytearray(CLOUD_BYTES), [], set()
    for index in range(header["count"]):
        item = fields(board.reply(("point ",), REPLY_TIMEOUT_S))
        names = ("x_bits", "y_bits", "z_bits", "id", "spread_bits", "observations")
        if (
            item.get("index") != index
            or any(type(item.get(key)) is not int or not 0 <= item[key] <= 0xFFFFFFFF for key in names)
            or item["id"] in ids
            or item["observations"] == 0
        ):
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
    struct.pack_into(
        "<IIi", raw, CLOUD_CAPACITY * CLOUD_POINT_BYTES, header["count"], header["dropped"], header["continuous"]
    )
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
        verdict = (
            "matched" if cloud["match"] is True else "not_compared_partial" if cloud["match"] is None else "different"
        )
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

    def __init__(self, port_name: str, baud: int = LINK_BAUD):
        import serial

        self.port = serial.Serial()
        self.port.port, self.port.baudrate, self.port.timeout = port_name, baud, 0.5
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
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def stream(
    port_name: str,
    frames_path: Path,
    expected_path: Path,
    limit: int | None = None,
    cores: int = 2,
    masks_path: Path | None = None,
    expected_map_path: Path | None = None,
    simd: bool = True,
    expected_cloud_path: Path | None = None,
    expected_voxels_path: Path | None = None,
    expected_dirs_path: Path | None = None,
    expected_plan_path: Path | None = None,
    baud: int = LINK_BAUD,
    gyro_path: Path | None = None,
    imu_path: Path | None = None,
    sense_path: Path | None = None,
) -> tuple[dict, list[str]]:
    """Runs the frames (the first `limit` of them, when given) through the board on `cores` cores, tracking with the
    vector kernels or (simd False) the portable ones and, with masks, building the map, and with a gyro or IMU
    expectation sending each step's readings before its frame, and with a metric one each frame's depth and range;
    the record and the board's lines."""
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    expectation_config(expected)  # refuse expectations of another build before opening the board
    if (expected.get("gyro") is None) != (gyro_path is None):
        raise ValueError("A gyro expectation and a gyro readings file go together")
    if (expected.get("imu") is None) != (imu_path is None):
        raise ValueError("An IMU expectation and an IMU readings file go together")
    if gyro_path is not None and imu_path is not None:
        raise ValueError("A run takes a gyro's readings or an IMU's, not both")
    sensor = "imu" if imu_path is not None else "gyro"
    expected_gyro = expected.get(sensor)  # the gyro's or the IMU's entry: both report the gyro's use and bias
    readings_path = imu_path if imu_path is not None else gyro_path
    reading_bytes = IMU_READING_BYTES if sensor == "imu" else GYRO_READING_BYTES
    gyro_payloads = []
    if expected_gyro is not None:
        validate_expected_gyro(expected_gyro, expected, sensor == "imu")
        if sha256_file(readings_path) != expected_gyro["readings_sha256"]:
            raise ValueError(f"The {sensor} readings file's SHA-256 differs from the PC run's")
        gyro_payloads = read_gyro_file(readings_path, expected["frames"] - 1, reading_bytes)
    expected_metric = expected.get("metric")
    if (expected_metric is None) != (sense_path is None):
        raise ValueError("A metric expectation and a sense file go together")
    sense = []
    if expected_metric is not None:
        validate_expected_metric(expected_metric, expected)
        if sha256_file(sense_path) != expected_metric["sense_sha256"]:
            raise ValueError("The sense file's SHA-256 differs from the PC run's")
        sense = read_sense_file(sense_path, expected["frames"])
        if expected_metric["beam_bits"] is None and any(reading[1] is not None for reading in sense):
            raise ValueError("The sense file holds ranges, and the metric expectation no ranger's beam")
    if masks_path is not None and expected_map_path is None:
        raise ValueError("A map expectation is required when streaming masks")
    expected_map = json.loads(expected_map_path.read_text(encoding="utf-8")) if masks_path is not None else None
    if expected_map is not None:
        check_same_config("map", expected_map, expected)
        validate_expected_map(expected_map, expected)
    expected_cloud = (
        json.loads(expected_cloud_path.read_text(encoding="utf-8")) if expected_cloud_path is not None else None
    )
    if expected_cloud is not None:
        check_same_config("cloud", expected_cloud, expected)
        validate_expected_cloud(expected_cloud, expected)
    expected_voxels = json.loads(expected_voxels_path.read_text(encoding="utf-8")) if expected_voxels_path else None
    if expected_voxels is not None:
        check_same_config("voxel", expected_voxels, expected)
        validate_expected_voxels(expected_voxels, expected)
    expected_dirs = json.loads(expected_dirs_path.read_text(encoding="utf-8")) if expected_dirs_path else None
    if expected_dirs is not None:
        if masks_path is None:
            raise ValueError("The obstacle directions need the masks")
        validate_expected_dirs(expected_dirs, expected)
    expected_plan = json.loads(expected_plan_path.read_text(encoding="utf-8")) if expected_plan_path else None
    if expected_plan is not None:
        check_same_config("plan", expected_plan, expected)
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
        raise ValueError(
            f"{frames_path.name} gives {count} frames at this rate; the expected run has {expected['frames']}"
        )
    if count < 1:
        raise ValueError("The run must contain at least one frame")
    full_run = limit is None or limit >= count
    count = min(count, limit) if limit is not None else count
    wanted = expected["step_hashes"][: count - 1]
    wanted_continuity = expected_map["map_continuity"][: count - 1] if expected_map is not None else []
    wanted_final = wanted_continuity[-1] if wanted_continuity else 1
    resources = ExitStack()
    try:
        frame_stream = resources.enter_context(frames_path.open("rb"))
        mask_stream = resources.enter_context(masks_path.open("rb")) if masks_path is not None else None
        board = Board(port_name, baud)
        resources.callback(board.close)
        ready = fields(board.reply(("ready",), READY_TIMEOUT_S))
        if expected_map is not None and (type(ready.get("map_policy")) is not int or ready["map_policy"] != MAP_POLICY):
            raise RuntimeError("The board needs map_policy=1: rebuild and flash the current firmware")
        if expected_cloud is not None:
            if (
                type(ready.get("cloud_policy")) is not int
                or ready["cloud_policy"] != CLOUD_POLICY
                or ready.get("cloud_capacity") != CLOUD_CAPACITY
                or ready.get("cloud_bytes") != CLOUD_BYTES
            ):
                raise RuntimeError("The board needs the current cloud policy and memory layout")
            for key in CLOUD_SOURCES:
                if key not in expected_cloud or ready.get(key) != expected_cloud[key]:
                    raise RuntimeError(f"The board's {key} differs from the PC run's")
        for key, source in (
            ([(key, expected_voxels) for key in VOXEL_SOURCES] if expected_voxels is not None else [])
            + ([(key, expected_dirs) for key in DIR_SOURCES] if expected_dirs is not None else [])
            + ([(key, expected_plan) for key in PLAN_SOURCES] if expected_plan is not None else [])
        ):
            if ready.get(key) != source[key]:
                raise RuntimeError(f"The board's {key} differs from the PC run's: rebuild and flash the firmware")
        if expected_voxels is not None and ready.get("voxel_bytes") != voxel_bytes(expected):
            raise RuntimeError("The board's voxel map has another memory layout")
        for key in SOURCES + (MAP_SOURCES if expected_map is not None else ()):
            source = expected_map if expected_map is not None else expected
            if key not in ready or key not in source or ready[key] != source[key]:
                raise RuntimeError(
                    f"The board's {key} differs from the PC run's: rebuild and flash the firmware, or "
                    "re-run the PC side on the current sources"
                )
        if ready.get("config") != expectation_config(expected) or ready.get("max_features") != max_features(expected):
            raise RuntimeError(
                f"The board runs the {ready.get('config')!r} build with {ready.get('max_features')} features, the "
                f"PC run {expectation_config(expected)!r}: flash the firmware of this tree"
            )
        camera = expected["camera_bits"]  # fx fy cx cy, then the lens's k1 k2 p1 p2 k3 (none: a pinhole)
        if len(camera) not in (4, 9) or not all(type(bits) is int and 0 <= bits <= 0xFFFFFFFF for bits in camera):
            raise RuntimeError("The expectation's camera_bits must be 4 or 9 float bit patterns")
        board.send(
            (
                "start "
                + " ".join(str(bits) for bits in camera[:4])
                + f" {expected['seed']} {cores} {int(simd)}"
                + "".join(f" {bits}" for bits in camera[4:])
                + "\n"
            ).encode()
        )
        started = fields(board.reply(("started",), REPLY_TIMEOUT_S))
        if expected_gyro is not None:
            board.send(("mount " + " ".join(str(bits) for bits in expected_gyro["mount_bits"]) + "\n").encode())
            board.reply(("mounted",), REPLY_TIMEOUT_S)
        if sensor == "imu" and expected_gyro is not None:
            board.send(("accel_bias " + " ".join(str(b) for b in expected_gyro["accel_bias_bits"]) + "\n").encode())
            board.reply(("accel_bias set",), REPLY_TIMEOUT_S)
        if expected_metric is not None and expected_metric["beam_bits"] is not None:
            board.send(("beam " + " ".join(str(b) for b in expected_metric["beam_bits"]) + "\n").encode())
            board.reply(("beam set",), REPLY_TIMEOUT_S)
        board.send(f"selftest {SELFTEST_CASES} {SELFTEST_SEED}\n".encode())
        selftest = fields(board.reply(("selftest",), SELFTEST_TIMEOUT_S))
        if selftest["template_mismatches"] or selftest["sums_mismatches"] or selftest["block_mismatches"]:
            raise RuntimeError(f"The board's vector kernels differ from the portable ones: {selftest}")
        steps, first, continuous_from_kind = [], None, 1
        voxel_mismatches, dir_mismatches, plan_mismatches, gyro_mismatches, gyro_us = [], [], [], [], []
        metric_mismatches = []
        forces_dropped = 0
        checked = (
            expected_map is not None
            or expected_cloud is not None
            or expected_voxels is not None
            or expected_dirs is not None
            or expected_plan is not None
        )
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
                raise RuntimeError(
                    f"Mask {index} is {'present' if record and record[0] else 'absent'}, unlike in the "
                    "direction expectation"
                )
            for name, bits in zip(("depth", "range"), sense[index] if sense else ()):  # the frame's readings
                if bits is not None:
                    board.send(f"{name} {bits}\n".encode())
                    board.reply((f"{name} set",), REPLY_TIMEOUT_S)
            if expected_gyro is not None and index > 0:  # the readings of the step this frame ends
                payload = gyro_payloads[index - 1]
                board.send(f"{sensor} {len(payload) // reading_bytes} {fnv(payload)}\n".encode() + payload)
                reply = fields(board.reply((f"{sensor} readings",), REPLY_TIMEOUT_S))
                gyro_us.append(reply["us"])
                forces_dropped += reply.get("dropped", 0)
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
                missing = [key for key in STEP_FIELDS if type(step.get(key)) is not int]
                if missing:  # a UART link that drops a byte loses a field's "=" or digits
                    raise RuntimeError(
                        f"Board step line for frame {index} arrived damaged (no valid {missing[0]}); "
                        f"on a UART link try a lower --baud: {line[:160]}"
                    )
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
                            raise RuntimeError(
                                f"Board step {index} cloud_{key}={actual!r}; expected {wanted_cloud[key]}"
                            )
                    if step["cloud_continuous"] != continuous_from_kind:
                        raise RuntimeError(f"Board step {index} cloud continuity disagrees with LOST history")
                    if any(
                        type(step.get(key)) is not int or step[key] < 0 for key in ("total_us", "map_us", "cloud_us")
                    ):
                        raise RuntimeError(f"Board step {index} has invalid timing")
                if expected_voxels is not None:
                    wanted_voxels = expected_voxels["voxel_progress"][index - 1]
                    if (
                        step.get("voxels") != wanted_voxels["count"]
                        or step.get("voxel_filed") != wanted_voxels["filed"]
                    ):
                        voxel_mismatches.append(index)
                    if type(step.get("voxel_us")) is not int or step["voxel_us"] < 0:
                        raise RuntimeError(f"Board step {index} has invalid voxel timing")
                if expected_dirs is not None:
                    check_dirs(step, expected_dirs, index, dir_mismatches)
                    if type(step.get("dir_us")) is not int or step["dir_us"] < 0:
                        raise RuntimeError(f"Board step {index} has invalid direction timing")
                if expected_gyro is not None and step.get("gyro") != expected_gyro["used"][index - 1]:
                    gyro_mismatches.append(index)
                if expected_metric is not None and [step.get(key) for key in METRIC_FIELDS] != [
                    expected_metric["known"][index - 1],
                    *expected_metric["centre_bits"][index - 1],
                    expected_metric["scale_bits"][index - 1],
                    expected_metric["variance_bits"][index - 1],
                ]:
                    metric_mismatches.append(index)
                if expected_plan is not None:
                    if (
                        step.get("plan") != expected_plan["actions"][index - 1]
                        or step.get("plan_dir") != expected_plan["directions"][index - 1]
                        or step.get("plan_turn") != expected_plan["turns"][index - 1]
                        or step.get("plan_hash") != expected_plan["plan_hashes"][index - 1]
                    ):
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
        board_gyro = fields(board.reply(("gyro bias",), REPLY_TIMEOUT_S)) if expected_gyro is not None else None
        board_up = fields(board.reply(("imu up",), REPLY_TIMEOUT_S)) if sensor == "imu" else None
        memory = fields(board.reply(("end",), REPLY_TIMEOUT_S))
        if expected_map is not None:
            check_continuity(board_map, wanted_final, "map")
            check_continuity(memory, wanted_final, "end")
        cloud = None
        if expected_cloud is not None:
            wanted_state = (
                expected_cloud["cloud_progress"][count - 2]
                if count > 1
                else {"count": 0, "dropped": 0, "continuous": 1}
            )
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
    kinds = {
        name: sum(step["kind"] == code for step in steps) for code, name in enumerate(("lost", "rotation", "essential"))
    }
    stages = (
        ("total", "track", "motion", "detect")
        + (("map",) if expected_map is not None or expected_cloud is not None else ())
        + (("cloud",) if expected_cloud is not None else ())
        + (("voxel",) if expected_voxels is not None else ())
        + (("dir",) if expected_dirs is not None else ())
        + (("plan",) if expected_plan is not None else ())
    )
    timing = (
        {f"{stage}_ms": summarise([step[f"{stage}_us"] / 1000 for step in steps]) for stage in stages} if steps else {}
    )
    if expected_cloud is not None and steps:
        timing["vo_cloud_map_ms"] = summarise(
            [(step["total_us"] + step["cloud_us"] + step["map_us"]) / 1000 for step in steps]
        )
    record = {
        "task": "Monocular VO"
        + (" and obstacle grid" if expected_map is not None else "")
        + (" and sparse cloud" if expected_cloud is not None else "")
        + " on an ESP32 board, replayed over its serial link against the PC build",
        "frames_file": frames_path.name,
        "expected_file": expected_path.name,
        "rate_hz": expected.get("rate_hz", 10.0),
        "cpu_mhz": ready["cpu_mhz"],
        "config": ready["config"],
        **{key: ready[key] for key in SOURCES + (MAP_SOURCES if expected_map is not None else ())},
        "state_bytes": started["state_bytes"],
        "cores": started["cores"],
        "simd": bool(started["simd"]),
        "kernel_selftest": selftest,
        "frames": count,
        "steps": len(steps),
        "steps_expected": len(wanted),
        "all_match": not mismatches and len(steps) == len(wanted),
        "mismatched_steps": mismatches,
        "kinds": kinds,
        "first_frame": first,
        "timing": timing,
        "memory": memory,
        "wall_seconds": wall,
    }
    if expected_map is not None:
        record["map"] = {
            "masks_file": masks_path.name,
            "expected_file": expected_map_path.name,
            **board_map,
            "policy": MAP_POLICY,
            "continuous_expected": wanted_final,
            "observations_expected": expected_map["map_observations"] if full_run else None,
            "hash_expected": expected_map["map_hash"] if full_run else None,
            "match": board_map["hash"] == expected_map["map_hash"] if full_run else None,
        }
    if cloud is not None:
        record["cloud"] = {
            "policy": CLOUD_POLICY,
            "expected_file": expected_cloud_path.name,
            "frames": count,
            **{key: expected_cloud.get(key) for key in (*CLOUD_SOURCES, "source_sha256")},
            **cloud,
        }
    if expected_voxels is not None:
        record["voxels"] = {
            "expected_file": expected_voxels_path.name,
            **board_voxels,
            "hash_expected": expected_voxels["voxel_hash"] if full_run else None,
            "match": False
            if voxel_mismatches
            else board_voxels.get("hash") == expected_voxels["voxel_hash"]
            if full_run
            else None,
            "mismatched_steps": voxel_mismatches,
            **{key: expected_voxels[key] for key in VOXEL_SOURCES},
        }
    if expected_dirs is not None:
        record["dirs"] = {
            "expected_file": expected_dirs_path.name,
            **board_dirs,
            "hash_expected": expected_dirs["dir_hash"] if full_run else None,
            "match": False
            if dir_mismatches
            else board_dirs.get("hash") == expected_dirs["dir_hash"]
            if full_run
            else None,
            "mismatched_frames": dir_mismatches,
            **{key: expected_dirs[key] for key in DIR_SOURCES},
        }
    if expected_plan is not None:
        record["plan"] = {
            "expected_file": expected_plan_path.name,
            **board_plan,
            "hash_expected": expected_plan["plan_hash"] if full_run else None,
            "match": False
            if plan_mismatches
            else board_plan.get("hash") == expected_plan["plan_hash"]
            if full_run
            else None,
            "mismatched_steps": plan_mismatches,
            **{key: expected_plan[key] for key in PLAN_SOURCES},
        }
    if expected_gyro is not None:
        bias_bits = [board_gyro.get(f"bias_{axis}_bits") for axis in "xyz"]
        bias_match = bias_bits == expected_gyro["final_bias_bits"] if full_run else None
        record[sensor] = {
            "readings_file": readings_path.name,
            "readings_sha256": expected_gyro["readings_sha256"],
            "used": {name: sum(step.get("gyro") == code for step in steps) for code, name in enumerate(GYRO_USES)},
            "mismatched_steps": gyro_mismatches,
            "final_bias_bits": bias_bits,
            "bias_steps": board_gyro.get("bias_steps"),
            "bias_match": bias_match,
            "match": not gyro_mismatches and bias_match is not False,
            "integration_us": summarise([float(us) for us in gyro_us]) if gyro_us else None,
        }
        if sensor == "imu":
            up_bits = [board_up.get(f"up_{axis}_bits") for axis in "xyz"]
            up_match = (
                up_bits == expected_gyro["final_up_bits"] and forces_dropped == expected_gyro["forces_dropped"]
                if full_run
                else None
            )
            record["imu"].update(
                {
                    "accel_bias_bits": expected_gyro["accel_bias_bits"],
                    "final_up_bits": up_bits,
                    "up_steps": board_up.get("up_steps"),
                    "forces_dropped": forces_dropped,
                    "up_match": up_match,
                    "match": record["imu"]["match"] and up_match is not False,
                }
            )
    if expected_metric is not None:
        record["metric"] = {
            "sense_file": sense_path.name,
            "sense_sha256": expected_metric["sense_sha256"],
            "beam_bits": expected_metric["beam_bits"],
            "depths": sum(reading[0] is not None for reading in sense[:count]),
            "ranges": sum(reading[1] is not None for reading in sense[:count]),
            "known_steps": sum(step.get("metric") == 1 for step in steps),
            "mismatched_steps": metric_mismatches,
            "match": not metric_mismatches,
        }
    record["per_step"] = steps
    return record, board.lines


BOARD_HEADING = "## ESP32 boards"


def write_board_section(results: Path) -> None:
    """Replace only the boards section of RESULTS.md (headed BOARD_HEADING) from records in board/."""
    rows = []
    for path in sorted((results / "board").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        t = record["timing"]
        board_map = record.get("map")
        if board_map is None:
            map_cell = "—"
        else:
            verdict = {
                True: "equal to the PC's",
                False: "differs from the PC's",
                None: "not compared (part of the run)",
            }
            map_cell = f"{board_map['observations']} observations, hash {verdict[board_map['match']]}"
        map_time = f"{t['map_ms']['mean']:.1f} ms" if "map_ms" in t else "—"
        verdict = {True: "equal to the PC's", False: "differs from the PC's", None: "not compared (part of the run)"}
        extra = []
        if record.get("voxels"):
            extra.append(
                f"{record['voxels']['count']} voxels, hash {verdict[record['voxels']['match']]}"
                + (f" ({t['voxel_ms']['mean']:.1f} ms)" if "voxel_ms" in t else "")
            )
        if record.get("dirs"):
            extra.append(
                f"directions {verdict[record['dirs']['match']]}"
                + (f" ({t['dir_ms']['mean']:.2f} ms)" if "dir_ms" in t else "")
            )
        if record.get("plan"):
            extra.append(
                f"plans {verdict[record['plan']['match']]}"
                + (f" ({t['plan_ms']['mean']:.2f} ms)" if "plan_ms" in t else "")
            )
        if record.get("gyro"):
            integration = record["gyro"].get("integration_us")
            extra.append(
                f"gyro's use and final bias {verdict[record['gyro']['match']]}"
                + (f" ({integration['mean']:.0f} µs to integrate a step's readings)" if integration else "")
            )
        if record.get("imu"):
            integration = record["imu"].get("integration_us")
            extra.append(
                f"IMU: the gyro's use, its final bias and the world's up {verdict[record['imu']['match']]}"
                + (f" ({integration['mean']:.0f} µs to integrate a step's readings)" if integration else "")
            )
        if record.get("metric"):
            extra.append(f"metric path and scale {verdict[record['metric']['match']]}")
        extra_cell = "; ".join(extra) if extra else "—"
        step_time = (
            f"{t['total_ms']['mean']:.0f} / {t['total_ms']['median']:.0f} / {t['total_ms']['max']:.0f} ms"
            if "total_ms" in t
            else "—"
        )
        stage_times = [
            f"{t[f'{stage}_ms']['mean']:.0f} ms" if f"{stage}_ms" in t else "—"
            for stage in ("track", "motion", "detect")
        ]
        kernels = "vector" if record.get("simd") else "C"
        rate = record.get("rate_hz", 10.0)
        rate_text = f"{rate:g} Hz" if isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0 else "unknown"
        clock = f"{record['cpu_mhz']} MHz" if "cpu_mhz" in record else "—"
        rows.append(
            f"| `{path.stem}` | {clock} | {record.get('config', '')} | {rate_text} | {record['cores']} | {kernels} | "
            f"{record['frames']} | "
            f"{record['steps'] - len(record['mismatched_steps'])} of {record['steps']} | {map_cell} | "
            f"{step_time} | {' | '.join(stage_times)} | {map_time} | {extra_cell} |"
        )
    section = [
        BOARD_HEADING,
        "",
        "The host streams grayscale frames and optional masks over the board's serial link (`python -m "
        "uwslam.mvo_stream`) of an ESP32-P4: its USB-UART bridge, or the chip's own USB-Serial-JTAG port. The "
        "board reports each step's hash and stage times. These timings exclude capture and transfer. Each record "
        "provides its own clock, build (core/mono_vo.h), rate, core count and kernel choice (the vector kernels are "
        "mvo_vector.S). Before streaming, the board checks its vector dispatch against C on random windows. Records "
        "and raw serial lines are in `board/`.",
        "",
        "| Record | Clock | Config | Rate | Cores | Kernels | Frames | Steps equal to the PC | Map | Per step, "
        "mean / median / max | Tracking | Motion | Detection | Map update | Voxel map / directions / plan / gyro "
        "(update) |",
        "|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
    ]
    if any("_usj" in path.stem for path in sorted((results / "board").glob("*.json"))):
        section += [
            "Records named `*_usj*` were streamed over the ESP32-P4's own USB-Serial-JTAG port, not its USB-UART "
            "bridge: the firmware was built as in the second line below, which changes only the console.",
            "",
        ]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((results / "board").glob("*.json"))]
    with_gyro = any(record.get("gyro") for record in records)
    with_imu = any(record.get("imu") for record in records)
    with_metric = any(record.get("metric") for record in records)
    section += [
        "Re-run (after `uwslam.mvo compare --board-frames` and `uwslam.mono_map --board-masks`"
        + (", or for a gyro or IMU sequence `uwslam.aqualoc board`" if with_gyro or with_imu else "")
        + "):",
        "",
        "```bash",
        "idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build -D SDKCONFIG=build/sdkconfig "
        "-p <port> flash   # the console on a USB-UART bridge",
        "idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build_usj -D SDKCONFIG=build_usj/sdkconfig "
        '-D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.usb_serial_jtag" -p <port> flash   # on USB-Serial-JTAG',
        "python -m uwslam.mvo_stream --port <port> --frames <frames file> --masks <masks file> --label <name>",
        *(
            [
                "python -m uwslam.mvo_stream --port <port> --frames <frames file> --gyro <gyro readings> "
                "--expected <board_expected_gyro.json> --label <name>   # with the gyro"
            ]
            if with_gyro
            else []
        ),
        *(
            [
                "python -m uwslam.mvo_stream --port <port> --frames <frames file> --imu <IMU readings> "
                "--expected <board_expected_imu.json> --label <name>   # with the IMU"
            ]
            if with_imu
            else []
        ),
        *(
            [
                "python -m uwslam.mvo_stream --port <port> --frames <frames file> [--imu <IMU readings>] --sense "
                "<sense file> --expected <expectation with a metric entry> --label <name>   # depths or ranges"
            ]
            if with_metric
            else []
        ),
        "```",
        "",
    ]
    report = results / "RESULTS.md"
    text = report.read_text(encoding="utf-8") if report.exists() else ""
    heading = re.search(r"(?m)^## ESP32 boards[ \t]*$", text)
    prefix, suffix = text, ""
    if heading:
        prefix = text[: heading.start()]
        later = re.search(r"(?m)^## [^\n]+", text[heading.end() :])
        if later:
            suffix = text[heading.end() + later.start() :]
    prefix = prefix.rstrip("\n") + "\n\n" if prefix.strip() else ""
    report.write_text(prefix + "\n".join(section) + ("\n" + suffix if suffix else ""), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", required=True, help="Serial port of the board")
    parser.add_argument(
        "--frames", type=Path, required=True, help="Raw grayscale frames from uwslam.mvo or uwslam.mono_cloud"
    )
    parser.add_argument("--expected", type=Path, default=RESULTS / "board_expected_10hz.json")
    parser.add_argument(
        "--masks", type=Path, help="The masks uwslam.mono_map wrote with --board-masks: build the map too"
    )
    parser.add_argument(
        "--expected-map", type=Path, help="Default: board_expected_map_<rate>hz.json in the lake map results"
    )
    parser.add_argument("--expected-cloud", type=Path, help="Cloud expectation from uwslam.mono_cloud --board-files")
    parser.add_argument("--cloud-output", type=Path, help="Save the verified board cloud as a new PLY file")
    parser.add_argument(
        "--expected-voxels",
        type=Path,
        help="Voxel-map expectation from uwslam.voxel_map (board_expected_voxels_<rate>hz.json)",
    )
    parser.add_argument(
        "--expected-dirs",
        type=Path,
        help="Obstacle-direction expectation from uwslam.obstacle_dir "
        "(board_expected_dirs_<rate>hz.json); needs --masks",
    )
    parser.add_argument(
        "--expected-plan",
        type=Path,
        help="Plan expectation from uwslam.path_plan "
        "(board_expected_plan_<rate>hz.json); give the masks and directions it was made with",
    )
    parser.add_argument("--gyro", type=Path, help="Each step's gyro readings, for an expectation with a gyro entry")
    parser.add_argument("--imu", type=Path, help="Each step's IMU readings, for an expectation with an imu entry")
    parser.add_argument(
        "--sense", type=Path, help="Each frame's depth and range, for an expectation with a metric entry"
    )
    parser.add_argument("--label", required=True, help="Name of the saved record, e.g. p4_360mhz_usj_5hz")
    parser.add_argument("--frames-limit", type=int, help="Stream only the first this many frames")
    parser.add_argument("--cores", type=int, choices=(1, 2), default=2, help="Run on one core or on both")
    parser.add_argument("--no-simd", action="store_true", help="Track with the portable C kernels, not the vector ones")
    parser.add_argument(
        "--baud",
        type=int,
        default=LINK_BAUD,
        help="A UART link's rate (a USB-Serial-JTAG port ignores "
        "it); the firmware switches its console UART to this once it runs",
    )
    args = parser.parse_args()
    if args.cloud_output is not None and args.expected_cloud is None:
        parser.error("--cloud-output requires --expected-cloud")
    if args.cloud_output is not None and args.cloud_output.exists():
        raise FileExistsError("Refusing to overwrite the cloud output")
    folder = args.expected.parent / "board"
    target, log = folder / f"{args.label}.json", folder / f"{args.label}_serial.log"
    if args.cloud_output is not None:
        reserved = (
            target,
            log,
            args.expected,
            args.frames,
            args.expected_cloud,
            args.masks,
            args.expected_map,
            args.expected.parent / "RESULTS.md",
        )
        if any(path is not None and args.cloud_output.resolve() == path.resolve() for path in reserved):
            raise ValueError("The cloud output must be distinct from replay inputs and records")
    if target.exists() or log.exists():
        raise FileExistsError(f"Refusing to overwrite board/{target.name}; choose another --label")
    rate = json.loads(args.expected.read_text(encoding="utf-8")).get("rate_hz", 10.0)
    expected_map = args.expected_map
    if args.masks is not None and expected_map is None:
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            parser.error("--expected-map is required with masks when the input frame rate is unknown")
        expected_map = CLIP_RESULTS / "mono_map" / f"board_expected_map_{rate:g}hz.json"
    record, lines = stream(
        args.port,
        args.frames,
        args.expected,
        args.frames_limit,
        args.cores,
        args.masks,
        expected_map,
        not args.no_simd,
        args.expected_cloud,
        args.expected_voxels,
        args.expected_dirs,
        args.expected_plan,
        args.baud,
        args.gyro,
        args.imu,
        args.sense,
    )
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
    print(
        f"{record['steps']} steps at {record['cpu_mhz']} MHz on {record['cores']} core(s), "
        f"{'vector' if record['simd'] else 'C'} kernels: all hashes match = "
        f"{record['all_match']} ({len(record['mismatched_steps'])} mismatched); {total['mean']:.0f} ms per step (median "
        f"{total['median']:.0f}, max {total['max']:.0f}); track {record['timing']['track_ms']['mean']:.0f} ms, motion "
        f"{record['timing']['motion_ms']['mean']:.0f} ms, detect {record['timing']['detect_ms']['mean']:.0f} ms"
    )
    if "map" in record:
        expected_observations = record["map"]["observations_expected"]
        comparison = (
            "not compared for a partial run" if expected_observations is None else f"PC {expected_observations}"
        )
        print(
            f"map: {record['map']['observations']} observations filed ({comparison}), "
            f"hash match = {record['map']['match']}, {record['timing']['map_ms']['mean']:.1f} ms per update"
        )
    if "voxels" in record:
        print(
            f"voxels: {record['voxels']['count']} held, hash match = {record['voxels']['match']}, "
            f"{len(record['voxels']['mismatched_steps'])} steps differ, {record['timing']['voxel_ms']['mean']:.2f} ms "
            "per update"
        )
    if "dirs" in record:
        print(
            f"directions: hash match = {record['dirs']['match']}, {len(record['dirs']['mismatched_frames'])} frames "
            f"differ, {record['timing']['dir_ms']['mean']:.2f} ms per update"
        )
    for sensor in ("gyro", "imu"):
        if sensor in record:
            print(
                f"{sensor}: steps by use {record[sensor]['used']}, {len(record[sensor]['mismatched_steps'])} differ, "
                f"final bias match = {record[sensor]['bias_match']}"
                + (f", final up match = {record['imu']['up_match']}" if sensor == "imu" else "")
            )
    if "metric" in record:
        print(
            f"metric: {record['metric']['known_steps']} steps with the scale known, "
            f"{len(record['metric']['mismatched_steps'])} differ from the PC's"
        )
    if "plan" in record:
        print(
            f"plans: hash match = {record['plan']['match']}, {len(record['plan']['mismatched_steps'])} steps differ, "
            f"{record['timing']['plan_ms']['mean']:.2f} ms per plan"
        )
    if "cloud" in record:
        print(
            f"cloud: {record['cloud']['count']} points, {record['cloud']['dropped']} historical points evicted, "
            f"continuous = {record['cloud']['continuous']}, hash match = {record['cloud']['match']}; "
            f"VO + cloud + obstacle map {record['timing']['vo_cloud_map_ms']['mean']:.1f} ms/step "
            "(excludes capture, segmentation and USB)"
        )


if __name__ == "__main__":
    main()
