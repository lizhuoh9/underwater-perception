"""AQUALOC harbour sequences (Ferrera et al., IJRR 2019) as the odometry's input with the dataset's own IMU: the
camera's frames at 5 Hz in a 320x240 camera the core models exactly, each step's raw gyro and accelerometer readings
for the core to integrate (mvo_gyro_rate, mvo_imu_reading), the reference poses, and the board's files. Scoring is
uwslam.aqualoc_score.

The data is not in the repository: from https://www.lirmm.fr/aqualoc/ take a harbour sequence's raw-data archive
(images, imu, depth csv files) and the meta files harbor_camera_calib.yaml, harbor_imu_camera_calib.yaml and
new_harbor_colmap_traj_sequence_NN.txt (the README calls the new_ references the better scaled ones).

- Camera: Kalibr's pinhole + equidistant lens (640x512) has no exact counterpart in OpenCV's radial model, so each
  320x240 pixel takes the ray of a chosen OpenCV model (square pixels, the lens's principal point in the middle 240 rows
  of the 2x2-averaged 320x256 frame, radial terms fitted to the lens's shape with the slope held at CORNER_SLOPE at the
  farthest corner, which lies CORNER_DEG off the axis) and samples the averaged frame where the equidistant model puts
  that ray (bilinear). The core's camera is then exact up to its float32 undistortion, which prepare checks on every
  pixel through the core itself.
- Frames: every fourth of the 20 FPS images (5 Hz), from the reference's first frame to its last, inside the IMU's
  record.
- Gyro: between each pair of frame stamps moved onto the IMU's clock (Kalibr's timeshift_cam_imu), the IMU's angular
  rates cut at the stamps (linear interpolation there) into readings of (dt, the mean of the reading's two end rates):
  the midpoint rule. The mount is T_cam_imu's rotation (IMU axes into the camera's).
- Accelerometer: its specific forces cut the same way, into the same pieces. The checks compare the up they measure
  over each step with the reference's and fit the accelerometer's bias to it (force = bias + s up, least squares).
  That fit serves the checks and calibrates the other sequences in uwslam.aqualoc_score; the core never sees the
  reference.
- Reference: the Colmap trajectory (every 5th image, metric through the pressure sensor), camera-to-world, slerped and
  linearly interpolated to the frames' stamps. Its vertical comes from the depth sensor regressed on its centres.

  python -m uwslam.aqualoc prepare <raw_data folder> --trajectory <txt> --camera-calib <yaml> --imu-calib <yaml>
      --output <fresh folder>
  python -m uwslam.aqualoc board <prepared folder> --output <fresh folder> [--calibration <other prepared folder>]
"""

import argparse
import csv
import hashlib
import json
import struct
import tempfile
from pathlib import Path

import numpy as np

from uwslam import mvo
from uwslam.utils import dump_json

WIDTH, HEIGHT = mvo.WIDTH, mvo.HEIGHT
SOURCE_SIZE = (640, 512)  # the harbour camera's frames
HALF_HEIGHT = SOURCE_SIZE[1] // 2  # after the 2x2 average
STRIDE = 4  # 20 FPS -> 5 Hz
RATE_HZ = 5.0
CORNER_DEG, CORNER_SLOPE = 60.0, 0.3  # every 320x240 pixel then falls inside the source frame, the model monotonic
DEPTH_OUTLIER_M = 10.0  # depth samples this far from the sequence's median are dropped (sequence 01 has one at -266 m)


def one(folder: Path, pattern: str) -> Path:
    """The single entry of the raw data folder that matches the pattern (harbor_<kind>_sequence_NN...)."""
    found = sorted(folder.glob(pattern))
    if len(found) != 1:
        raise ValueError(f"Expected one {pattern} in {folder.name}, found {len(found)}")
    return found[0]


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """A dataset csv: its header (the first row, starting with #, names lower-cased) and its rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows or not rows[0][0].startswith("#"):
        raise ValueError(f"{path.name}: expected a header starting with #")
    return [name.strip().lstrip("#").strip().lower() for name in rows[0]], rows[1:]


def column(header: list[str], prefix: str) -> int:
    matches = [k for k, name in enumerate(header) if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one column starting with {prefix!r}, got {header}")
    return matches[0]


def read_camera_calibration(path: Path) -> dict:
    """The fields of Kalibr's camera YAML this needs, without a YAML library."""
    text = path.read_text(encoding="utf-8").replace("\n    ", " ")
    fields = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        value = value.strip()
        fields[name.strip()] = [float(v) for v in value.strip("[]").split(",") if v.strip()] if "[" in value else value
    if fields.get("distortion_model") != "equidistant" or fields.get("camera_model") != "pinhole":
        raise ValueError("Expected Kalibr's pinhole camera with the equidistant distortion model")
    if fields.get("resolution") != [float(SOURCE_SIZE[0]), float(SOURCE_SIZE[1])]:
        raise ValueError(f"Expected a {SOURCE_SIZE[0]}x{SOURCE_SIZE[1]} calibration, got {fields.get('resolution')}")
    if len(fields.get("intrinsics", [])) != 4 or len(fields.get("distortion_coeffs", [])) != 4:
        raise ValueError("Expected four intrinsics and four distortion coefficients")
    return fields


def read_imu_calibration(path: Path) -> tuple[np.ndarray, float]:
    """T_cam_imu's rotation (a vector in the IMU's axes into the camera's) and timeshift_cam_imu (t_imu = t_cam +
    shift) from Kalibr's camchain-imucam YAML."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((k for k, line in enumerate(lines) if line.strip().startswith("T_cam_imu")), None)
    shift_line = next((line for line in lines if "timeshift_cam_imu" in line), None)
    if start is None or shift_line is None or start + 4 >= len(lines):
        raise ValueError(f"{path.name}: expected Kalibr's T_cam_imu and timeshift_cam_imu")
    matrix = np.array(
        [
            [float(v) for v in lines[start + 1 + row].strip().lstrip("-").strip().strip("[]").split(",")]
            for row in range(4)
        ]
    )
    shift = float(shift_line.split(":")[1])
    rotation = matrix[:3, :3]
    if abs(np.linalg.det(rotation) - 1) > 1e-3 or np.abs(rotation @ rotation.T - np.eye(3)).max() > 1e-3:
        raise ValueError("T_cam_imu's rotation is not a rotation")
    return rotation, shift


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image numbers, camera centres and camera-to-world rotations of a Colmap trajectory
    (`img_number tx ty tz qx qy qz qw`)."""
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape[1] != 8:
        raise ValueError(f"{path.name}: expected rows of img_number tx ty tz qx qy qz qw")
    frames = rows[:, 0].astype(int)
    if np.any(np.diff(frames) <= 0):
        raise ValueError(f"{path.name}: image numbers must increase")
    return frames, rows[:, 1:4], np.array([quaternion_matrix(q) for q in rows[:, 4:8]])


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """(qx, qy, qz, qw) -> rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def exp_so3(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3) + skew(v)
    k = skew(v / angle)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def log_so3(r: np.ndarray) -> np.ndarray:
    angle = float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return 0.5 * axis if angle < 1e-8 else angle / (2.0 * np.sin(angle)) * axis


def angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(log_so3(r))))


# The camera


def equidistant(theta: np.ndarray, k: np.ndarray) -> np.ndarray:
    """The equidistant lens's distorted angle for the ray angle theta."""
    t2 = theta**2
    return theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))


def radial(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """OpenCV's radial model (k1, k2, k3) on the undistorted radius r."""
    r2 = r * r
    return r * (1 + r2 * (c[0] + r2 * (c[1] + r2 * c[2])))


def radial_slope(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    r2 = r * r
    return 1 + r2 * (3 * c[0] + r2 * (5 * c[1] + r2 * 7 * c[2]))


def radial_inverse(rd: np.ndarray, c: np.ndarray) -> np.ndarray:
    """r with radial(r) = rd, by bisection on the monotonic branch from 0 to the first fold (float64)."""
    grid = np.linspace(0.0, 10.0, 100001)
    falling = np.nonzero(radial_slope(grid, c) <= 0)[0]
    top = grid[falling[0]] if len(falling) else grid[-1]
    if np.any(rd > radial(np.array([top]), c)[0]):
        raise ValueError("A radius lies beyond the model's fold")
    low, high = np.zeros_like(rd), np.full_like(rd, top)
    for _ in range(200):
        middle = (low + high) / 2
        below = radial(middle, c) < rd
        low, high = np.where(below, middle, low), np.where(below, high, middle)
    return (low + high) / 2


def half_frame(calibration: dict) -> dict:
    """The source frame's intrinsics after the 2x2 average (pixel centres at integers)."""
    fx, fy, cx, cy = calibration["intrinsics"]
    return {"fx": fx / 2, "fy": fy / 2, "cx": (cx - 0.5) / 2, "cy": (cy - 0.5) / 2}


def fit_camera(calibration: dict) -> dict:
    """The 320x240 camera (mvo_camera_t's fields): the radial terms that follow the lens's distorted angle over the
    radii up to the farthest corner's, by least squares with the slope held at CORNER_SLOPE there; the focal length puts
    that corner at CORNER_DEG off the axis."""
    k = np.array(calibration["distortion_coeffs"])
    half = half_frame(calibration)
    cx, cy = half["cx"], half["cy"] - (HALF_HEIGHT - HEIGHT) / 2
    corner_r = np.tan(np.radians(CORNER_DEG))
    r = np.linspace(1e-3, corner_r, 400)
    design = np.column_stack([r**3, r**5, r**7])
    constraint = np.array([3 * corner_r**2, 5 * corner_r**4, 7 * corner_r**6])
    system = np.zeros((4, 4))  # least squares with one equality (constraint . c = CORNER_SLOPE - 1), by its KKT system
    system[:3, :3] = 2 * design.T @ design
    system[:3, 3] = constraint
    system[3, :3] = constraint
    rhs = np.concatenate([2 * design.T @ (equidistant(np.arctan(r), k) - r), [CORNER_SLOPE - 1]])
    c = np.linalg.solve(system, rhs)[:3]
    corner_pixels = np.hypot(max(cx, WIDTH - 1 - cx), max(cy, HEIGHT - 1 - cy))
    focal = float(corner_pixels / radial(np.array([corner_r]), c)[0])
    return {"fx": focal, "fy": focal, "cx": cx, "cy": cy, "k1": c[0], "k2": c[1], "p1": 0.0, "p2": 0.0, "k3": c[2]}


def pixel_grid() -> np.ndarray:
    columns, rows = np.meshgrid(np.arange(WIDTH, dtype=float), np.arange(HEIGHT, dtype=float))
    return np.column_stack([columns.ravel(), rows.ravel()])


def exact_rays(camera: dict) -> np.ndarray:
    """Every 320x240 pixel's undistorted normalised point (x, y) under the camera's radial model, in float64."""
    c = np.array([camera["k1"], camera["k2"], camera["k3"]])
    distorted = (pixel_grid() - [camera["cx"], camera["cy"]]) / [camera["fx"], camera["fy"]]
    rd = np.linalg.norm(distorted, axis=1)
    return distorted * np.where(rd > 0, radial_inverse(rd, c) / np.maximum(rd, 1e-12), 1.0)[:, None]


def remap_tables(calibration: dict, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    """For cv2.remap on the 2x2-averaged frame: where each 320x240 pixel's ray lands in it."""
    k = np.array(calibration["distortion_coeffs"])
    half = half_frame(calibration)
    rays = exact_rays(camera)
    r = np.linalg.norm(rays, axis=1)
    scale = np.where(r > 0, equidistant(np.arctan(r), k) / np.maximum(r, 1e-12), 1.0)
    source = rays * scale[:, None]
    map_x = (half["fx"] * source[:, 0] + half["cx"]).reshape(HEIGHT, WIDTH).astype(np.float32)
    map_y = (half["fy"] * source[:, 1] + half["cy"]).reshape(HEIGHT, WIDTH).astype(np.float32)
    return map_x, map_y


def camera_check(vo, calibration: dict, camera: dict) -> dict:
    """How closely the core's own undistortion (mvo_undistort, float32) recovers every pixel's ray, and whether every
    pixel samples inside the source frame."""
    import ctypes

    vo.mvo_undistort.argtypes = [ctypes.POINTER(mvo.Camera), ctypes.c_void_p, ctypes.c_void_p]
    vo.mvo_undistort.restype = ctypes.c_int
    model = mvo.Camera(*(float(camera[name]) for name in mvo.CAMERA_FIELDS))
    exact = exact_rays(camera)
    pixel, out = np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
    found, failures = np.zeros_like(exact), 0
    for k, (u, v) in enumerate(pixel_grid()):
        pixel[:] = (u, v)
        failures += not vo.mvo_undistort(ctypes.byref(model), pixel.ctypes.data, out.ctypes.data)
        found[k] = (out - [model.cx, model.cy]) / [model.fx, model.fy]
    a = np.column_stack([found, np.ones(len(found))])
    b = np.column_stack([exact, np.ones(len(exact))])
    cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    error = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    map_x, map_y = remap_tables(calibration, camera)
    inside = (map_x >= 0) & (map_x <= WIDTH - 1) & (map_y >= 0) & (map_y <= HALF_HEIGHT - 1)
    c = np.array([camera["k1"], camera["k2"], camera["k3"]])
    return {
        "core_ray_error_deg": {"median": float(np.median(error)), "max": float(error.max())},
        "core_undistort_failures": int(failures),
        "pixels_inside_source": float(inside.mean()),
        "hfov_deg": float(2 * np.degrees(np.arctan(radial_inverse(np.array([camera["cx"] / camera["fx"]]), c)[0]))),
        "vfov_deg": float(2 * np.degrees(np.arctan(radial_inverse(np.array([camera["cy"] / camera["fy"]]), c)[0]))),
    }


def resample(path: Path, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[:2] != (SOURCE_SIZE[1], SOURCE_SIZE[0]):
        raise ValueError(f"{path.name}: expected a {SOURCE_SIZE[0]}x{SOURCE_SIZE[1]} image")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        raise ValueError(f"{path.name}: expected 8-bit pixels, got {image.dtype}")
    half = cv2.resize(image, (SOURCE_SIZE[0] // 2, HALF_HEIGHT), interpolation=cv2.INTER_AREA)
    return cv2.remap(half, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# The gyro and the reference


def step_readings(times: np.ndarray, rates: np.ndarray, start: float, end: float) -> np.ndarray:
    """The readings of one step on the IMU's clock: (dt, the mean of each piece's two end rates) for the pieces between
    start, the samples inside and end, the rates linearly interpolated at start and end."""
    if start < times[0] or end > times[-1] or end <= start:
        raise ValueError("A step lies outside the IMU's record")
    inside = (times > start) & (times < end)
    knots = np.concatenate([[start], times[inside], [end]])
    w = np.stack([np.interp(knots, times, rates[:, axis]) for axis in range(3)], axis=1)
    return np.column_stack([np.diff(knots), 0.5 * (w[:-1] + w[1:])])


def integrate(readings: np.ndarray, mount: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """The camera's R_AB that the readings (less the bias) add up to, in float64: what mvo_gyro_rate builds."""
    rotation = np.eye(3)
    for dt, *rate in readings:
        rotation = rotation @ exp_so3((np.array(rate) - (0 if bias is None else bias)) * dt)
    return mount @ rotation @ mount.T


def reference_at(
    frames: np.ndarray, centres: np.ndarray, rotations: np.ndarray, stamps: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotations (camera-to-world) and centres at the query stamps: slerp and linear between the reference's images."""
    keys = stamps[frames]
    if query.min() < keys[0] or query.max() > keys[-1]:
        raise ValueError("Query stamps outside the reference")
    out_r, out_c = [], []
    for t in query:
        k = int(np.clip(np.searchsorted(keys, t, side="right") - 1, 0, len(keys) - 2))
        weight = float((t - keys[k]) / (keys[k + 1] - keys[k]))
        out_r.append(rotations[k] @ exp_so3(weight * log_so3(rotations[k].T @ rotations[k + 1])))
        out_c.append(centres[k] + weight * (centres[k + 1] - centres[k]))
    return np.array(out_r), np.array(out_c)


def mean_force(readings: np.ndarray, forces: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """A step's time-weighted mean specific force in the IMU's axes at its end frame B (float64): each reading's force
    turned into A with the gyro's rotation (less the bias) at the reading's middle, the sum turned into B."""
    rotation = np.eye(3)
    total = np.zeros(3)
    for (dt, *w), f in zip(readings, forces):
        half = exp_so3((np.array(w) - bias) * dt / 2)
        total += rotation @ half @ f * dt
        rotation = rotation @ half @ half
    return rotation.T @ total / readings[:, 0].sum()


def fit_accel_bias(forces: np.ndarray, ups: np.ndarray) -> tuple[np.ndarray, float]:
    """The accelerometer's bias b and the scale s with force = b + s up (both in the IMU's axes), least squares."""
    design = np.zeros((3 * len(forces), 4))
    for k, up in enumerate(ups):
        design[3 * k : 3 * k + 3, :3] = np.eye(3)
        design[3 * k : 3 * k + 3, 3] = up
    solution, *_ = np.linalg.lstsq(design, forces.reshape(-1), rcond=None)
    return solution[:3], float(solution[3])


def up_angles_deg(forces: np.ndarray, ups: np.ndarray) -> np.ndarray:
    cosine = np.sum(forces * ups, axis=1) / np.linalg.norm(forces, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def accel_checks(steps: list[np.ndarray], forces: list[np.ndarray], gyro_bias: np.ndarray, ups: np.ndarray) -> dict:
    """The up the accelerometer measures over each step against the reference's (ups: unit, the IMU's axes at each
    step's end), before and after the fitted bias."""
    mean = np.array([mean_force(r, f, gyro_bias) for r, f in zip(steps, forces)])
    bias, scale = fit_accel_bias(mean, ups)
    before, after = up_angles_deg(mean, ups), up_angles_deg(mean - bias, ups)
    return {
        "accel_force_median": float(np.median(np.linalg.norm(mean, axis=1))),
        "accel_up_error_deg": {"median": float(np.median(before)), "p90": float(np.percentile(before, 90))},
        "accel_bias_fit": bias.tolist(),
        "accel_scale_fit": scale,
        "accel_up_error_without_fitted_bias_deg": {
            "median": float(np.median(after)),
            "p90": float(np.percentile(after, 90)),
        },
    }


def fit_bias(steps: list[np.ndarray], mount: np.ndarray, reference: list[np.ndarray]) -> np.ndarray:
    """The constant bias (rad/s, the IMU's axes) that best fits the steps' reference turns: to first order the
    gyro's miss log(gyro^T reference), in the IMU's axes, is minus the bias times the step's time; least squares, two
    Gauss-Newton passes. For the checks only: the core learns its own."""
    bias = np.zeros(3)
    for _ in range(2):
        rows, values = [], []
        for readings, turn in zip(steps, reference):
            gyro = mount.T @ integrate(readings, mount, bias) @ mount
            rows.append(-readings[:, 0].sum() * np.eye(3))
            values.append(log_so3(gyro.T @ (mount.T @ turn @ mount)))
        bias = bias + np.linalg.lstsq(np.concatenate(rows), np.concatenate(values), rcond=None)[0]
    return bias


def read_depth(raw: Path, t0: int) -> tuple[np.ndarray, np.ndarray, int]:
    """The depth sensor's plausible readings: times (seconds from t0), depths (m) and how many were dropped."""
    _, rows = read_csv(one(raw, "harbor_depth_sequence_*.csv"))
    depth = np.array([[float(v) for v in row[:2]] for row in rows])
    plausible = np.abs(depth[:, 1] - np.median(depth[:, 1])) < DEPTH_OUTLIER_M
    return (depth[plausible, 0].astype(np.int64) - t0) / 1e9, depth[plausible, 1], int((~plausible).sum())


def vertical_from_depth(raw: Path, t0: int, stamps: np.ndarray, frames: np.ndarray, centres: np.ndarray) -> dict:
    """The reference's downward vertical: the depth sensor regressed on the reference's centres (depth = a + g . c, so
    g points down, and |g| is 1 for a metric reference)."""
    times, depths, dropped = read_depth(raw, t0)
    measured = np.interp(stamps[frames], times, depths)
    design = np.column_stack([np.ones(len(frames)), centres])
    solution, *_ = np.linalg.lstsq(design, measured, rcond=None)
    g = solution[1:]
    return {
        "vertical_world": (g / np.linalg.norm(g)).tolist(),
        "depth_slope_norm": float(np.linalg.norm(g)),
        "depth_residual_rms_m": float(np.sqrt(np.mean((measured - design @ solution) ** 2))),
        "depth_samples_dropped": dropped,
    }


def prepare(raw: Path, trajectory: Path, camera_calib: Path, imu_calib: Path, output: Path) -> dict:
    """Writes frames_5hz.npy, sequence.npz, camera.json and checks.json into a fresh output folder."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite; choose a fresh --output")
    header, rows = read_csv(one(raw, "harbor_img_sequence_*.csv"))
    stamps_ns = np.array([int(row[0]) for row in rows], dtype=np.int64)
    names = [row[1].strip() for row in rows]
    if np.any(np.diff(stamps_ns) <= 0):
        raise ValueError("Image stamps must increase")
    t0 = int(stamps_ns[0])
    stamps = (stamps_ns - t0) / 1e9
    imu_header, imu_rows = read_csv(one(raw, "harbor_imu_sequence_*.csv"))
    imu = np.array([[float(v) for v in row] for row in imu_rows])
    imu_times = (imu[:, 0].astype(np.int64) - t0) / 1e9
    rate_columns = [column(imu_header, f"w_rs_s_{axis}") for axis in "xyz"]
    force_columns = [column(imu_header, f"a_rs_s_{axis}") for axis in "xyz"]
    if not all("rad s^-1" in imu_header[c] for c in rate_columns):
        raise ValueError(f"Expected angular rates in rad/s: {imu_header}")
    if not all("m s^-2" in imu_header[c] for c in force_columns):
        raise ValueError(f"Expected specific forces in m/s^2: {imu_header}")
    if np.any(np.diff(imu_times) <= 0):
        raise ValueError("IMU stamps must increase")
    rates = imu[:, rate_columns]
    mount, shift = read_imu_calibration(imu_calib)
    frames, centres, rotations = read_trajectory(trajectory)
    if frames[-1] >= len(stamps):
        raise ValueError("The trajectory names images beyond the image list")
    picks = np.arange(frames[0], frames[-1] + 1, STRIDE)
    picks = picks[(stamps[picks] + shift >= imu_times[0]) & (stamps[picks] + shift <= imu_times[-1])]
    if len(picks) < 2:
        raise ValueError("Fewer than two frames inside both the reference and the IMU's record")
    ref_r, ref_c = reference_at(frames, centres, rotations, stamps, stamps[picks])
    steps = [step_readings(imu_times, rates, stamps[a] + shift, stamps[b] + shift) for a, b in zip(picks, picks[1:])]
    forces = [
        step_readings(imu_times, imu[:, force_columns], stamps[a] + shift, stamps[b] + shift)[:, 1:]
        for a, b in zip(picks, picks[1:])
    ]
    turns = [ref_r[k].T @ ref_r[k + 1] for k in range(len(steps))]
    errors = np.array([angle_deg(integrate(s, mount).T @ turn) for s, turn in zip(steps, turns)])
    bias = fit_bias(steps, mount, turns)
    debiased = np.array([angle_deg(integrate(s, mount, bias).T @ turn) for s, turn in zip(steps, turns)])
    calibration = read_camera_calibration(camera_calib)
    camera = fit_camera(calibration)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        camera_checks = camera_check(mvo.build(Path(scratch)), calibration, camera)
    durations = np.diff(stamps[picks]) * 1000
    counts = np.array([len(s) for s in steps])
    checks = {
        "images": len(stamps),
        "frames_5hz": len(picks),
        "step_ms": {"min": float(durations.min()), "max": float(durations.max())},
        "imu_rate_hz": float(1 / np.median(np.diff(imu_times))),
        "readings_per_step": {"min": int(counts.min()), "max": int(counts.max())},
        "timeshift_cam_imu_s": shift,
        "reference_turn_median_deg": float(np.median([angle_deg(t) for t in turns])),
        "gyro_error_deg": {"median": float(np.median(errors)), "p90": float(np.percentile(errors, 90))},
        "fitted_bias_dps": np.degrees(bias).tolist(),
        "gyro_error_without_fitted_bias_deg": {
            "median": float(np.median(debiased)),
            "p90": float(np.percentile(debiased, 90)),
        },
        **vertical_from_depth(raw, t0, stamps, frames, centres),
        "camera": camera_checks,
    }
    ups = np.array([mount.T @ -(r.T @ np.array(checks["vertical_world"])) for r in ref_r[1:]])
    checks.update(accel_checks(steps, forces, bias, ups))
    map_x, map_y = remap_tables(calibration, camera)
    images = one(raw, "harbor_images_sequence_*")
    grey = np.stack([resample(images / names[k], map_x, map_y) for k in picks])
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "frames_5hz.npy", grey)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    np.savez(
        output / "sequence.npz",
        picks=picks,
        stamps=stamps[picks],
        reference_rotations=ref_r,
        reference_centres=ref_c,
        reading_offsets=offsets,
        readings=np.concatenate(steps).astype(np.float32),
        forces=np.concatenate(forces).astype(np.float32),
        mount=mount,
        vertical=np.array(checks["vertical_world"]),
        accel_bias_fit=np.array(checks["accel_bias_fit"]),
        depths=np.interp(stamps[picks], *read_depth(raw, t0)[:2]),  # the depth sensor at each frame, m
    )
    dump_json(output / "camera.json", {"source": camera_calib.name, "camera": camera})
    dump_json(output / "checks.json", checks)
    return checks


def readings_file(path: Path, steps: list[np.ndarray]) -> None:
    """Per step a little-endian uint32 count, then that many float32 rows (uwslam.mvo_stream --gyro or --imu)."""
    path.write_bytes(b"".join(struct.pack("<I", len(rows)) + rows.astype("<f4").tobytes() for rows in steps))


def board_files(prepared: Path, output: Path, calibration: Path | None = None) -> dict:
    """The board's files for a prepared sequence at 5 Hz: frames.bin (raw 320x240 frames), gyro.bin (per step a
    little-endian uint32 count, then that many (dt, x, y, z) float32 readings: uwslam.mvo_stream --gyro), imu.bin (the
    same with (dt, x, y, z, ax, ay, az): --imu) and three expectations from the PC build of the same core, seed
    uwslam.mvo.SEED: board_expected_camera.json (the camera alone), board_expected_gyro.json (the gyro, its bias
    learnt from zero) and board_expected_imu.json (the IMU, the accelerometer's bias the one fitted to the
    `calibration` sequence, or none); for a sequence prepared with its depths also sense.bin (each frame's depth:
    --sense) and board_expected_depth.json (the IMU and the depth sensor: the metric scale)."""
    from uwslam.aqualoc_score import load_prepared

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite; choose a fresh --output")
    sequence = load_prepared(prepared)
    accel_bias = None if calibration is None else load_prepared(calibration).accel_bias_fit
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames.bin").write_bytes(b"".join(np.ascontiguousarray(f).tobytes() for f in sequence.frames))
    gyro_file, imu_file = output / "gyro.bin", output / "imu.bin"
    readings_file(gyro_file, sequence.readings)
    readings_file(imu_file, sequence.imu_readings())
    imu = mvo.GyroReadings(sequence.mount, sequence.imu_readings(), None, accel_bias)
    runs = [
        ("camera", None, None, None),
        ("gyro", mvo.GyroReadings(sequence.mount, sequence.readings), gyro_file, None),
        ("imu", imu, imu_file, None),
    ]
    if sequence.depths is not None:
        sense = mvo.sense_bytes(sequence.depths, None, len(sequence.frames))
        (output / "sense.bin").write_bytes(sense)
        runs.append(("depth", imu, imu_file, sequence.depths))
    summary = {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        vo = mvo.build(Path(scratch))
        for name, gyro, source, depths in runs:
            run = mvo.run_odometry(vo, list(sequence.frames), sequence.camera, mvo.SEED, gyro, depths)
            sensor = "imu" if name == "depth" else name  # the depth run reads the IMU too
            expected = {
                "sequence": prepared.name,
                "rate_hz": RATE_HZ,
                "every": 1,
                "frames": len(sequence.frames),
                "steps": len(run.steps),
                "camera": [getattr(sequence.camera, field) for field in mvo.CAMERA_FIELDS],
                "seed": mvo.SEED,
                "camera_bits": mvo.camera_bits(sequence.camera),
                "step_hashes": run.hashes,
                "kinds": [mvo.KINDS[s.kind] for s in run.steps],
                "final_world_centre": list(run.steps[-1].world_centre),
                **mvo.sources_sha256(),
                **mvo.build_config(),
            }
            if gyro is not None:
                expected[sensor] = {
                    "mount_bits": [mvo.float_bits(v) for v in np.asarray(sequence.mount, dtype=np.float32).reshape(9)],
                    "readings_file": source.name,
                    "readings_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "used": run.gyro_used,
                    "final_bias_bits": [mvo.float_bits(v) for v in run.gyro_bias[-1]],
                }
            if sensor == "imu":
                bias = np.zeros(3) if accel_bias is None else accel_bias
                expected["imu"].update(
                    {
                        "accel_bias_from": None if calibration is None else calibration.name,
                        "accel_bias_bits": [mvo.float_bits(v) for v in np.asarray(bias, dtype=np.float32)],
                        "final_up_bits": [mvo.float_bits(v) for v in run.imu_up],
                        "forces_dropped": run.forces_dropped,
                    }
                )
            if depths is not None:
                expected["metric"] = {"sense_file": "sense.bin", **mvo.metric_expectation(run, sense, None)}
            dump_json(output / f"board_expected_{name}.json", expected)
            summary[name] = {"kinds": {k: expected["kinds"].count(k) for k in mvo.KINDS.values()}}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare", help="Frames, IMU readings and reference of one harbour sequence")
    make.add_argument("raw", type=Path, help="The sequence's raw_data folder (images, imu and depth csv files)")
    make.add_argument("--trajectory", type=Path, required=True, help="new_harbor_colmap_traj_sequence_NN.txt")
    make.add_argument("--camera-calib", type=Path, required=True, help="harbor_camera_calib.yaml")
    make.add_argument("--imu-calib", type=Path, required=True, help="harbor_imu_camera_calib.yaml")
    make.add_argument("--output", type=Path, required=True)
    board = commands.add_parser("board", help="The board's frames, gyro and IMU readings and expectations")
    board.add_argument("prepared", type=Path)
    board.add_argument("--output", type=Path, required=True)
    board.add_argument(
        "--calibration", type=Path, help="A prepared sequence whose fitted accelerometer bias calibrates the IMU's"
    )
    args = parser.parse_args()
    if args.command == "prepare":
        checks = prepare(args.raw, args.trajectory, args.camera_calib, args.imu_calib, args.output)
        print(json.dumps(checks, indent=1))
    else:
        print(json.dumps(board_files(args.prepared, args.output, args.calibration), indent=1))


if __name__ == "__main__":
    main()
