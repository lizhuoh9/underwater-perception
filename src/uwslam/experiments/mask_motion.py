"""Probe: how much camera rotation 60x80 water masks carry, checked against the lake clip's reference poses.

Every frame of the video up to the last reference keyframe takes the robot's input path (segmenter.device_frame) and
an int8 water-segmentation model (segmenter.EspdlSegmenter, passed with --model) gives the 60x80 non-water mask. The
reconstruction's camera-to-world poses (images.txt, TUM, world = first camera with x right, y down, z forward) are
interpolated to every frame: rotations by slerp, centres linearly. Its time column is image index / 30, but the images
were sampled from the 30 FPS video at 10 FPS, so video time = REFERENCE_TIME_SCALE (3) x that column: the 88 keyframes
span 33.8 s of video. With that scale, rotations between full-resolution frames 0.1 s apart from SIFT and the
essential matrix follow the reference closely; read 1:1 they do not correlate at all (both are in the report).

Mask gyro: a pure camera rotation R moves every image point by the homography K R K^-1, whatever its depth, so the
rotation between two frames can be searched by turning one mask's boundary cells into the other frame and scoring how
far they land from its boundary. Boundary cells are non-water cells with a water 4-neighbour inside the grid. The score
is the symmetric truncated chamfer distance on 3-4 distance transforms (integer, in thirds of a cell), summed as
sum_A * n_B + sum_B * n_A so it stays an integer and ranks like the sum of the two means. A coarse yaw / pitch / roll
grid is refined twice around its best cell. Pairs where either mask has fewer than MIN_BOUNDARY boundary cells are
declined. References: the same search on Canny edges of the 160x120 grey frame, OR-pooled to the same 60x80 grid
(what the robot's own frame holds); SIFT with the essential matrix on the full-resolution frames (what a feature
method on a good camera gets, which also checks the reference); and "zero", no rotation, the floor a useful estimate
must beat. Translation moves boundaries too (parallax); the chamfer search absorbs part of it as rotation, which is
part of what is measured.

Mask change: per pair, the share of cells that differ, raw and after turning the earlier mask by the reference
rotation (cells that turn out of view are left out). What the reference rotation leaves is translation, the
visibility edge moving with the camera, and segmentation flicker.

Intrinsics: the horizontal field of view is not calibrated, so it is chosen among HFOV_CANDIDATES as the one whose
SIFT rotations at 10 Hz differ least from the reference (median angle) on pairs turning by more than 3 degrees, and
every source uses it. On smaller turns the feature noise dominates and favours any narrow view. The 80x60 grid's
vertical field of view follows from the source frame's aspect, because the device path stretches the frame to 4:3.
Rotations are reported as rotation-vector components in the earlier camera's frame: pitch about x, yaw about y
(down), roll about z.
"""

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uwslam.paths import CLIP_RESULTS
from uwslam.utils import dump_json

MASK_HEIGHT, MASK_WIDTH = 60, 80
TRUNCATE = 15  # 3-4 chamfer units: 5 cells
STRAIGHT, DIAGONAL = 3, 4
MIN_BOUNDARY = 20
MAX_POINTS = 400  # boundary cells kept per frame (evenly thinned beyond this)
MIN_DEPTH = 0.05  # a turned ray must keep this much forward component to project
REFERENCE_TIME_SCALE = 3.0  # video seconds per images.txt second (10 FPS images stamped at 30 FPS)
TIME_SLACK = 1e-6  # s: images.txt stores float32 times, so the last frame can sit a hair past the last keyframe
# (half-widths yaw, pitch, roll in degrees, step in degrees): a coarse grid, then two refinements around the best cell.
GRIDS = (((60.0, 15.0, 12.0), 3.0), ((2.0, 2.0, 2.0), 1.0), ((0.5, 0.5, 0.5), 0.25))
CANNY_LOW, CANNY_HIGH = 40, 100
STRIDES = {10.0: 3, 5.0: 6, 3.0: 10}  # pair spacing in 30 FPS frames per emulated rate (Hz)
PAIR_SPACING = 3  # frames between the first frames of consecutive pairs
SOURCES = ("mask", "edges", "features")
ERROR_BOUNDS = (2.0, 5.0)  # degrees
SIFT_FEATURES, SIFT_RATIO, RANSAC_PIXELS, MIN_POSE_INLIERS = 1000, 0.75, 1.0, 30
HFOV_CANDIDATES = tuple(range(60, 135, 5))  # degrees
CALIBRATION_RATE = 10.0
CALIBRATION_MIN_ANGLE_DEG = (
    3.0  # only turns this large calibrate: below it feature noise dominates and favours narrow views
)


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, hfov_deg: float, aspect: float, width: int = MASK_WIDTH, height: int = MASK_HEIGHT) -> "Camera":
        """Pinhole intrinsics of a width x height grid showing a source frame of height/width `aspect` (the 80x60 mask
        grid stretches a 960x544 frame, so its fx and fy differ; for the source itself pass its own aspect)."""
        half = np.tan(np.radians(hfov_deg) / 2)
        return cls(width / 2 / half, height / 2 / (aspect * half), width / 2, height / 2)

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


def boundary_cells(nonwater: np.ndarray) -> np.ndarray:
    """Non-water cells with a water 4-neighbour inside the grid; the image border itself is not an edge."""
    solid = nonwater.astype(bool)
    wet = np.zeros_like(solid)
    wet[1:] |= ~solid[:-1]
    wet[:-1] |= ~solid[1:]
    wet[:, 1:] |= ~solid[:, :-1]
    wet[:, :-1] |= ~solid[:, 1:]
    return solid & wet


def edge_cells(gray: np.ndarray) -> np.ndarray:
    """Canny edges of a 120x160 grey frame, OR-pooled over 2x2 blocks onto the 60x80 grid."""
    import cv2

    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), CANNY_LOW, CANNY_HIGH) > 0
    return edges.reshape(MASK_HEIGHT, 2, MASK_WIDTH, 2).any(axis=(1, 3))


def chamfer_distance(cells: np.ndarray, truncate: int = TRUNCATE) -> np.ndarray:
    """3-4 chamfer distance from every cell to the nearest set cell, capped at `truncate`.

    The classic two-pass sweep, with each row's horizontal recurrence written as a running minimum; integer, so a
    sequential C version gives the same values."""
    height, width = cells.shape
    far = truncate + STRAIGHT * (height + width)
    distance = np.where(cells, 0, far).astype(np.int64)
    columns = STRAIGHT * np.arange(width)
    for row in range(height):
        if row:
            distance[row] = np.minimum(distance[row], neighbour_row(distance[row - 1]))
        distance[row] = np.minimum.accumulate(distance[row] - columns) + columns
    for row in range(height - 1, -1, -1):
        if row < height - 1:
            distance[row] = np.minimum(distance[row], neighbour_row(distance[row + 1]))
        distance[row] = np.minimum.accumulate((distance[row] + columns)[::-1])[::-1] - columns
    return np.minimum(distance, truncate)


def neighbour_row(adjacent: np.ndarray) -> np.ndarray:
    """Best distance reachable from the row above or below: straight down/up, or diagonally."""
    best = adjacent + STRAIGHT
    best[1:] = np.minimum(best[1:], adjacent[:-1] + DIAGONAL)
    best[:-1] = np.minimum(best[:-1], adjacent[1:] + DIAGONAL)
    return best


def cell_rays(cells: np.ndarray, camera: Camera, max_points: int = MAX_POINTS) -> np.ndarray:
    """Viewing rays (x, y, 1) through the centres of the set cells, evenly thinned to `max_points`, row-major order."""
    rows, columns = np.nonzero(cells)
    if len(rows) > max_points:
        keep = np.linspace(0, len(rows) - 1, max_points).round().astype(int)
        rows, columns = rows[keep], columns[keep]
    x = (columns + 0.5 - camera.cx) / camera.fx
    y = (rows + 0.5 - camera.cy) / camera.fy
    return np.stack([x, y, np.ones_like(x)], axis=1)


def euler_rotation(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Ry(yaw) Rx(pitch) Rz(roll) in camera axes (x right, y down, z forward), radians; broadcasts to (..., 3, 3)."""
    yaw, pitch, roll = np.broadcast_arrays(*(np.asarray(angle, dtype=np.float64) for angle in (yaw, pitch, roll)))
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.stack(
        [
            np.stack([cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp], axis=-1),
            np.stack([cp * sr, cp * cr, -sp], axis=-1),
            np.stack([-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp], axis=-1),
        ],
        axis=-2,
    )


def landing_costs(rays: np.ndarray, rotations: np.ndarray, distance: np.ndarray, camera: Camera) -> np.ndarray:
    """Per rotation, the summed truncated distance at which the rays turned by R^T land on the other frame's map;
    rays leaving the view (or turning backwards) cost the truncation."""
    turned = np.einsum("nji,mj->nmi", rotations, rays)
    depth = turned[..., 2]
    ahead = depth > MIN_DEPTH
    safe = np.where(ahead, depth, 1.0)
    column = np.floor(camera.fx * turned[..., 0] / safe + camera.cx)
    row = np.floor(camera.fy * turned[..., 1] / safe + camera.cy)
    inside = ahead & (column >= 0) & (column < MASK_WIDTH) & (row >= 0) & (row < MASK_HEIGHT)
    cost = np.full(depth.shape, TRUNCATE, dtype=np.int64)
    cost[inside] = distance[row[inside].astype(int), column[inside].astype(int)]
    return cost.sum(axis=1)


def grid_angles(
    centre: tuple[float, float, float], spans: tuple[float, float, float], step: float
) -> tuple[np.ndarray, np.ndarray]:
    """(3, n) yaw, pitch, roll in degrees around `centre`, yaw slowest and roll fastest, and each candidate's squared
    distance from the centre in grid steps (the tie-break)."""
    offsets = [np.arange(-round(spans[k] / step), round(spans[k] / step) + 1) for k in range(3)]
    grids = np.meshgrid(*offsets, indexing="ij")
    steps = np.stack([grid.ravel() for grid in grids])
    return np.array(centre)[:, None] + step * steps, (steps**2).sum(axis=0)


def search_rotation(cells_a: np.ndarray, cells_b: np.ndarray, camera: Camera) -> dict | None:
    """Rotation from frame A to frame B (R_AB = R_A^T R_B, so a ray d_A appears in B as R_AB^T d_A) by the symmetric
    chamfer search; None when either frame has fewer than MIN_BOUNDARY cells. Equal scores go to the candidate nearest
    the grid centre, then to the first in grid order."""
    rays_a, rays_b = cell_rays(cells_a, camera), cell_rays(cells_b, camera)
    if min(len(rays_a), len(rays_b)) < MIN_BOUNDARY:
        return None
    distance_a, distance_b = chamfer_distance(cells_a), chamfer_distance(cells_b)
    centre, cost = (0.0, 0.0, 0.0), None
    for spans, step in GRIDS:
        angles, offset = grid_angles(centre, spans, step)
        rotations = euler_rotation(*np.radians(angles))
        costs = landing_costs(rays_a, rotations, distance_b, camera) * len(rays_b) + landing_costs(
            rays_b, rotations.transpose(0, 2, 1), distance_a, camera
        ) * len(rays_a)
        best = int(np.lexsort((np.arange(len(costs)), offset, costs))[0])
        centre, cost = tuple(float(value) for value in angles[:, best]), int(costs[best])
    yaw, pitch, roll = centre
    return {
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "cost": cost,
        "points": (len(rays_a), len(rays_b)),
        "rotation": euler_rotation(*np.radians(centre)),
    }


def essential_rotation(points_a: np.ndarray, points_b: np.ndarray, matrix: np.ndarray) -> np.ndarray | None:
    """R_AB from matched pixels via RANSAC on the essential matrix; None below MIN_POSE_INLIERS pose inliers.
    OpenCV's recoverPose gives R with x_B = R x_A + t, i.e. R_AB^T."""
    import cv2

    if len(points_a) < MIN_POSE_INLIERS:
        return None
    essential, inliers = cv2.findEssentialMat(points_a, points_b, matrix, cv2.RANSAC, 0.999, RANSAC_PIXELS)
    if essential is None or essential.shape != (3, 3):
        return None
    count, rotation, _, _ = cv2.recoverPose(essential, points_a, points_b, matrix, mask=inliers)
    return rotation.T if count >= MIN_POSE_INLIERS else None


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Axis times angle (radians) of a 3x3 rotation."""
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    if np.pi - angle < 1e-6:
        raise ValueError("Rotation of 180 degrees has no unique vector")
    return angle / (2 * np.sin(angle)) * axis


def rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    """Rodrigues: the 3x3 rotation of a rotation vector (radians); inverse of rotation_vector."""
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def angle_deg(rotation: np.ndarray) -> float:
    """Rotation angle of a 3x3 rotation, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))))


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> 3x3 rotation."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def slerp(q0: np.ndarray, q1: np.ndarray, weight: float) -> np.ndarray:
    """Shortest-arc spherical interpolation of unit quaternions (x, y, z, w)."""
    if np.dot(q0, q1) < 0:
        q1 = -q1
    angle = np.arccos(np.clip(np.dot(q0, q1), -1.0, 1.0))
    q = q0 if angle < 1e-9 else (np.sin((1 - weight) * angle) * q0 + np.sin(weight * angle) * q1) / np.sin(angle)
    return q / np.linalg.norm(q)


def reference_poses(poses: np.ndarray, times: np.ndarray, time_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera-to-world rotations (n, 3, 3) and centres (n, 3) at video `times` from TUM rows (t tx ty tz qx qy qz qw)
    whose t is video time / time_scale."""
    keys = poses[:, 0] * time_scale
    if times.min() < keys[0] - TIME_SLACK or times.max() > keys[-1] + TIME_SLACK:
        raise ValueError("Requested times fall outside the reference keyframes")
    times = np.clip(times, keys[0], keys[-1])
    rotations = []
    for t in times:
        k = int(np.clip(np.searchsorted(keys, t, side="right") - 1, 0, len(keys) - 2))
        weight = float((t - keys[k]) / (keys[k + 1] - keys[k]))
        rotations.append(quaternion_matrix(slerp(poses[k, 4:8], poses[k + 1, 4:8], weight)))
    centres = np.stack([np.interp(times, keys, poses[:, 1 + axis]) for axis in range(3)], axis=1)
    return np.stack(rotations), centres


def turned_mask(mask_a: np.ndarray, rotation_ab: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Mask A as seen from B under a pure rotation: each B cell's ray d_B looks along R_AB d_B in A. Returns the
    values and which B cells still see inside A."""
    rows, columns = np.mgrid[0:MASK_HEIGHT, 0:MASK_WIDTH]
    rays = np.stack(
        [(columns + 0.5 - camera.cx) / camera.fx, (rows + 0.5 - camera.cy) / camera.fy, np.ones(rows.shape)], axis=-1
    ).reshape(-1, 3)
    turned = rays @ rotation_ab.T
    depth = turned[:, 2]
    safe = np.where(depth > MIN_DEPTH, depth, 1.0)
    column = np.floor(camera.fx * turned[:, 0] / safe + camera.cx)
    row = np.floor(camera.fy * turned[:, 1] / safe + camera.cy)
    known = (depth > MIN_DEPTH) & (column >= 0) & (column < MASK_WIDTH) & (row >= 0) & (row < MASK_HEIGHT)
    values = np.zeros(len(rays), dtype=bool)
    values[known] = mask_a[row[known].astype(int), column[known].astype(int)]
    return values.reshape(MASK_HEIGHT, MASK_WIDTH), known.reshape(MASK_HEIGHT, MASK_WIDTH)


def frame_pairs(frames: int) -> dict[float, list[tuple[int, int]]]:
    """Per rate, the pairs (a, a + stride) with a every PAIR_SPACING frames."""
    return {
        rate: [(a, a + stride) for a in range(0, frames - stride, PAIR_SPACING)] for rate, stride in STRIDES.items()
    }


def pair_row(rate: float, a: int, b: int, data: dict, camera: Camera) -> dict:
    """Reference motion, the estimates and their errors, and the mask change for frames a < b."""
    r_ab = data["rotations"][a].T @ data["rotations"][b]
    reference = np.degrees(rotation_vector(r_ab))
    shift = data["rotations"][a].T @ (data["centres"][b] - data["centres"][a])
    row = {
        "rate_hz": rate,
        "frame_a": a,
        "frame_b": b,
        "t_a": round(float(data["times"][a]), 4),
        "ref_angle_deg": angle_deg(r_ab),
        "ref_pitch_deg": reference[0],
        "ref_yaw_deg": reference[1],
        "ref_roll_deg": reference[2],
        "ref_shift": float(np.linalg.norm(shift)),
        "zero_error_deg": angle_deg(r_ab),
    }
    estimates = {source: search_rotation(data[source][a], data[source][b], camera) for source in ("mask", "edges")}
    feature = data["feature_rotations"].get((a, b))
    estimates["features"] = None if feature is None else {"rotation": feature}
    for source in SOURCES:
        found = estimates[source]
        if found is None:
            row.update({f"{source}_{key}": None for key in ("pitch_deg", "yaw_deg", "roll_deg", "error_deg")})
            continue
        pitch, yaw, roll = np.degrees(rotation_vector(found["rotation"]))
        row.update(
            {
                f"{source}_pitch_deg": pitch,
                f"{source}_yaw_deg": yaw,
                f"{source}_roll_deg": roll,
                f"{source}_error_deg": angle_deg(found["rotation"].T @ r_ab),
            }
        )
    nonwater = data["mask_nonwater"]
    seen, known = turned_mask(nonwater[a], r_ab, camera)
    row["changed_raw"] = float(np.mean(nonwater[a] != nonwater[b]))
    row["changed_after_reference_rotation"] = float(np.mean(seen[known] != nonwater[b][known])) if known.any() else None
    return row


def summarise(rows: list[dict]) -> dict:
    """Error distributions per source over all pairs (declined pairs fall back to zero rotation) and over answered ones."""
    zero = np.array([row["zero_error_deg"] for row in rows])
    summary = {
        "pairs": len(rows),
        "ref_angle_median_deg": float(np.median(zero)),
        "ref_angle_p90_deg": float(np.percentile(zero, 90)),
        "ref_shift_median": float(np.median([row["ref_shift"] for row in rows])),
        "zero": error_stats(zero),
    }
    for source in SOURCES:
        answered = [row for row in rows if row[f"{source}_error_deg"] is not None]
        filled = np.array(
            [
                row[f"{source}_error_deg"] if row[f"{source}_error_deg"] is not None else row["zero_error_deg"]
                for row in rows
            ]
        )
        better = np.mean([row[f"{source}_error_deg"] < row["zero_error_deg"] for row in answered]) if answered else None
        summary[source] = {
            "answered_share": len(answered) / len(rows),
            **error_stats(filled),
            "beats_zero_share_of_answered": None if better is None else float(better),
            "yaw_slope": yaw_slope(answered, source),
            "yaw_correlation": yaw_correlation(answered, source),
        }
    changed = [
        row["changed_after_reference_rotation"] for row in rows if row["changed_after_reference_rotation"] is not None
    ]
    summary["changed_raw_median"] = float(np.median([row["changed_raw"] for row in rows]))
    summary["changed_after_reference_rotation_median"] = float(np.median(changed)) if changed else None
    return summary


def error_stats(errors: np.ndarray) -> dict:
    """Median, 90th percentile and the shares under each ERROR_BOUNDS of rotation errors in degrees."""
    return {
        "median_deg": float(np.median(errors)),
        "p90_deg": float(np.percentile(errors, 90)),
        **{f"under_{bound:g}_deg": float(np.mean(errors < bound)) for bound in ERROR_BOUNDS},
    }


def yaw_slope(rows: list[dict], source: str) -> float | None:
    """Least-squares slope through the origin of estimated against reference yaw; 1 when the field of view is right."""
    reference = np.array([row["ref_yaw_deg"] for row in rows])
    estimate = np.array([row[f"{source}_yaw_deg"] for row in rows])
    return float(reference @ estimate / (reference @ reference)) if len(rows) and reference @ reference > 0 else None


def yaw_correlation(rows: list[dict], source: str) -> float | None:
    if len(rows) < 3:
        return None
    reference = np.array([row["ref_yaw_deg"] for row in rows])
    estimate = np.array([row[f"{source}_yaw_deg"] for row in rows])
    return float(np.corrcoef(reference, estimate)[0, 1]) if reference.std() > 0 and estimate.std() > 0 else None


def match_features(first: tuple, second: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Ratio-test SIFT matches between two (points, uint8 descriptors) frames as two (n, 2) pixel arrays."""
    import cv2

    (points_a, descriptors_a), (points_b, descriptors_b) = first, second
    empty = np.zeros((0, 2), dtype=np.float32)
    if len(points_a) < 2 or len(points_b) < 2:
        return empty, empty
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        descriptors_a.astype(np.float32), descriptors_b.astype(np.float32), k=2
    )
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < SIFT_RATIO * pair[1].distance]
    return (
        np.float32([points_a[m.queryIdx] for m in good]).reshape(-1, 2),
        np.float32([points_b[m.trainIdx] for m in good]).reshape(-1, 2),
    )


def calibrate_hfov(matches: dict, data: dict, source_size: tuple[int, int]) -> tuple[float, list]:
    """The HFOV_CANDIDATES value whose essential-matrix rotations at CALIBRATION_RATE differ least (median angle) from
    the reference over the pairs that turn by more than CALIBRATION_MIN_ANGLE_DEG, and the whole table."""
    width, height = source_size
    pairs = [
        (a, b)
        for a, b in frame_pairs(len(data["times"]))[CALIBRATION_RATE]
        if angle_deg(data["rotations"][a].T @ data["rotations"][b]) > CALIBRATION_MIN_ANGLE_DEG
    ]
    table = []
    for hfov in HFOV_CANDIDATES:
        matrix = Camera.from_fov(hfov, height / width, width, height).matrix()
        errors = []
        for a, b in pairs:
            found = essential_rotation(*matches[(a, b)], matrix)
            if found is not None:
                errors.append(angle_deg(found.T @ (data["rotations"][a].T @ data["rotations"][b])))
        table.append((hfov, float(np.median(errors)) if errors else None, len(errors)))
    best = min((row for row in table if row[1] is not None), key=lambda row: row[1])
    return float(best[0]), table


def read_inputs(video: Path, reconstruction: Path, segmenter, time_scale: float) -> dict:
    """Device frames up to the last keyframe with their masks and edge cells, full-resolution SIFT features of every
    frame a pair uses, and the reference poses at each frame."""
    import cv2

    from uwslam.experiments.segmenter import device_frame

    poses = np.loadtxt(reconstruction / "images.txt")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(np.floor(poses[-1, 0] * time_scale * fps + TIME_SLACK)) + 1
    needed = {index for pairs in frame_pairs(count).values() for pair in pairs for index in pair}
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    nonwater, edges, features = [], [], {}
    try:
        while len(nonwater) < count:
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"{video.name} ends before the last reference keyframe")
            index = len(nonwater)
            frame = device_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            nonwater.append(segmenter(frame))
            edges.append(edge_cells(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)))
            if index in needed:
                keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None)
                features[index] = (
                    np.float32([point.pt for point in keypoints]).reshape(-1, 2),
                    np.zeros((0, 128), np.uint8)
                    if descriptors is None
                    else np.clip(descriptors, 0, 255).astype(np.uint8),
                )
        source_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    times = np.arange(count) / fps
    rotations, centres = reference_poses(poses, times, time_scale)
    nonwater = np.stack(nonwater)
    return {
        "fps": float(fps),
        "times": times,
        "rotations": rotations,
        "centres": centres,
        "mask_nonwater": nonwater,
        "mask": np.stack([boundary_cells(mask) for mask in nonwater]),
        "edges": np.stack(edges),
        "features": features,
        "source_size": source_size,
        "poses_sha256": hashlib.sha256((reconstruction / "images.txt").read_bytes()).hexdigest(),
    }


def alignment_check(matches: dict, data: dict, poses: np.ndarray, hfov: float) -> dict:
    """Yaw correlation and median angle between the 10 Hz feature rotations and the reference read with the chosen time
    scale, and read 1:1 (video time = images.txt time) over the frames both readings cover."""
    width, height = data["source_size"]
    matrix = Camera.from_fov(hfov, height / width, width, height).matrix()
    found = {
        pair: essential_rotation(*matches[pair], matrix) for pair in frame_pairs(len(data["times"]))[CALIBRATION_RATE]
    }
    result = {}
    for name, scale in (("chosen_scale", None), ("one_to_one", 1.0)):
        rows = []
        for (a, b), rotation in found.items():
            if rotation is None:
                continue
            if scale is None:
                reference = data["rotations"][a].T @ data["rotations"][b]
            else:
                if data["times"][b] > poses[-1, 0] * scale:
                    continue
                pair_rotations, _ = reference_poses(poses, data["times"][[a, b]], scale)
                reference = pair_rotations[0].T @ pair_rotations[1]
            rows.append(
                (
                    np.degrees(rotation_vector(reference))[1],
                    np.degrees(rotation_vector(rotation))[1],
                    angle_deg(rotation.T @ reference),
                )
            )
        values = np.array(rows)
        result[name] = {
            "pairs": len(rows),
            "yaw_correlation": float(np.corrcoef(values[:, 0], values[:, 1])[0, 1]),
            "median_angle_deg": float(np.median(values[:, 2])),
        }
    return result


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def degrees(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}°"


def ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def write_report(output: Path, summary: dict) -> None:
    """RESULTS.md in the style of the sibling experiments."""
    source, camera, check = summary["source"], summary["camera"], summary["alignment_check"]
    lines = [
        "# Camera rotation from water masks alone (lake clip, reference poses)",
        "",
        f"Source: `{source['name']}`, {source['frames']} frames ({source['duration_s']:.1f} s at {source['fps']:g} FPS); "
        f"reference: the reconstruction's {source['keyframes']} keyframes, interpolated to every frame.",
        "",
        f"- **Reference timing.** The reconstruction's time column is image index / 30 but its images were taken at 10 FPS, "
        f"so video time = {summary['time_scale']:g} × that column. With this scale, SIFT rotations between full-resolution "
        f"frames 0.1 s apart follow the reference: yaw correlation {check['chosen_scale']['yaw_correlation']:.2f}, "
        f"median difference {degrees(check['chosen_scale']['median_angle_deg'])} ({check['chosen_scale']['pairs']} pairs). "
        f"Read 1:1 they do not: {check['one_to_one']['yaw_correlation']:.2f} and "
        f"{degrees(check['one_to_one']['median_angle_deg'])} ({check['one_to_one']['pairs']} pairs).",
        "- **Masks.** Each frame takes the robot's input path to 160×120 and the int8 water-segmentation model at its "
        "deployment threshold gives the 60×80 non-water mask. Boundary cells are non-water cells with a water neighbour.",
        "- **Mask gyro.** A pure rotation moves image points by K R K⁻¹ whatever their depth. One frame's boundary cells "
        "are turned into the other frame and scored by the 3-4 chamfer distance to its boundary (capped at 5 cells), "
        "in both directions. A coarse grid (yaw ±60°, pitch ±15°, roll ±12° in 3° steps) is refined twice (1°, 0.25°). "
        f"Frames with fewer than {MIN_BOUNDARY} boundary cells get no estimate; those pairs count as zero rotation.",
        "- **References.** *Edges*: the same search on Canny edges of the 160×120 grey frame, pooled onto the same "
        "60×80 grid. *Features*: SIFT and the essential matrix on the full-resolution frames. *Zero*: no rotation.",
        f"- **Intrinsics.** Not calibrated. A {camera['hfov_deg']:g}° horizontal field of view was chosen because, among "
        f"{HFOV_CANDIDATES[0]}–{HFOV_CANDIDATES[-1]}°, it gives the smallest median difference between the 10 Hz feature "
        f"rotations and the reference on pairs turning by more than {CALIBRATION_MIN_ANGLE_DEG:g}° (table below; on "
        "smaller turns feature noise dominates and favours any narrow view). All sources use it.",
        "",
        "## Rotation error (angle between estimated and reference rotation)",
        "",
        "| Rate | Pairs | Reference median / P90 | Zero | Mask median / P90 | Mask < 2° | Mask beats zero | Mask yaw corr. | "
        "Edges median / P90 | Edges yaw corr. | Features median / P90 | Features answered |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rate, entry in summary["rates"].items():
        mask, edges, features = entry["mask"], entry["edges"], entry["features"]
        lines.append(
            f"| {float(rate):g} Hz | {entry['pairs']} | {degrees(entry['ref_angle_median_deg'])} / "
            f"{degrees(entry['ref_angle_p90_deg'])} | {degrees(entry['zero']['median_deg'])} | "
            f"{degrees(mask['median_deg'])} / {degrees(mask['p90_deg'])} | {percent(mask['under_2_deg'])} | "
            f"{percent(mask['beats_zero_share_of_answered'])} | {ratio(mask['yaw_correlation'])} | "
            f"{degrees(edges['median_deg'])} / {degrees(edges['p90_deg'])} | {ratio(edges['yaw_correlation'])} | "
            f"{degrees(features['median_deg'])} / {degrees(features['p90_deg'])} | {percent(features['answered_share'])} |"
        )
    lines += [
        "",
        "Pairs start every 3 frames. Medians and P90 count pairs without an estimate as zero rotation; "
        '"beats zero" counts answered pairs only. Yaw correlation is between estimated and reference yaw over '
        "answered pairs.",
        "",
        "## How much of the mask change is rotation",
        "",
        "| Rate | Reference translation (median, reconstruction units) | Cells changed (raw) | Still changed after the reference rotation |",
        "|---:|---:|---:|---:|",
    ]
    for rate, entry in summary["rates"].items():
        lines.append(
            f"| {float(rate):g} Hz | {entry['ref_shift_median']:.3f} | {percent(entry['changed_raw_median'])} | "
            f"{percent(entry['changed_after_reference_rotation_median'])} |"
        )
    lines += [
        "",
        "What the reference rotation leaves comes from translation (parallax), the visibility edge moving with the "
        "camera and segmentation flicker; they cannot be told apart here.",
        "",
        "## Field-of-view choice",
        "",
        f"| HFOV | Median feature-vs-reference angle at 10 Hz (turns > {CALIBRATION_MIN_ANGLE_DEG:g}°) | Pairs |",
        "|---:|---:|---:|",
    ]
    lines += [f"| {hfov}° | {degrees(median)} | {pairs} |" for hfov, median, pairs in summary["hfov_table"]]
    lines += [
        "",
        "## Limits",
        "",
        "- One 34 s lake clip filmed by a diver, not the robot in a pool; the reference is a reconstruction whose "
        "keyframes are interpolated, with units of its own.",
        "- The field of view is fitted to the same reference it is scored against.",
        "- The search estimates rotation only, so translation parallax leaks into it.",
        "",
        "Re-run (the output folder must not exist or be empty):",
        "",
        "```bash",
        "python -m uwslam.experiments.mask_motion <video> --reconstruction <folder with images.txt> "
        "--model <segmentation model folder> --output results/lake_clip/mask_motion_2",
        "```",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run(video: Path, reconstruction: Path, output: Path, segmenter, time_scale: float = REFERENCE_TIME_SCALE) -> dict:
    """`segmenter` maps a 160x120 device frame to the 60x80 non-water mask and describe()s itself for the summary."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite results; choose a fresh --output")
    data = read_inputs(video, reconstruction, segmenter, time_scale)
    print(f"Read and segmented {len(data['times'])} frames", flush=True)
    pairs = frame_pairs(len(data["times"]))
    matches = {
        pair: match_features(data["features"][pair[0]], data["features"][pair[1]])
        for rate_pairs in pairs.values()
        for pair in rate_pairs
    }
    hfov, hfov_table = calibrate_hfov(matches, data, data["source_size"])
    width, height = data["source_size"]
    camera = Camera.from_fov(hfov, height / width)
    full = Camera.from_fov(hfov, height / width, width, height).matrix()
    data["feature_rotations"] = {pair: essential_rotation(*matches[pair], full) for pair in matches}
    check = alignment_check(matches, data, np.loadtxt(reconstruction / "images.txt"), hfov)
    print(f"Field of view {hfov:g} deg; reference check {check}", flush=True)
    rows, rates = [], {}
    for rate, rate_pairs in pairs.items():
        chunk = [pair_row(rate, a, b, data, camera) for a, b in rate_pairs]
        rows.extend(chunk)
        rates[str(rate)] = summarise(chunk)
        print(
            f"{rate:g} Hz: mask {rates[str(rate)]['mask']['median_deg']:.1f} deg, edges {rates[str(rate)]['edges']['median_deg']:.1f}, "
            f"features {rates[str(rate)]['features']['median_deg']:.1f}, zero {rates[str(rate)]['zero']['median_deg']:.1f}",
            flush=True,
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output / "frames.npz",
        times=data["times"],
        mask_nonwater=data["mask_nonwater"],
        mask=data["mask"],
        edges=data["edges"],
        rotations=data["rotations"],
        centres=data["centres"],
    )
    summary = {
        "task": "Rotation recoverable from 60x80 water masks (symmetric chamfer search on boundary cells) against "
        "reconstructed reference poses, with Canny edges, full-resolution SIFT and zero rotation as references",
        "source": {
            "name": video.name,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "fps": data["fps"],
            "frames": len(data["times"]),
            "duration_s": float(data["times"][-1]),
            "width": width,
            "height": height,
            "keyframes": int(len(np.loadtxt(reconstruction / "images.txt"))),
            "poses_sha256": data["poses_sha256"],
        },
        "time_scale": time_scale,
        "alignment_check": check,
        "camera": {"hfov_deg": hfov, "fx": camera.fx, "fy": camera.fy, "cx": camera.cx, "cy": camera.cy},
        "hfov_table": hfov_table,
        "settings": {
            "truncate": TRUNCATE,
            "min_boundary": MIN_BOUNDARY,
            "max_points": MAX_POINTS,
            "grids": [[list(spans), step] for spans, step in GRIDS],
            "canny": [CANNY_LOW, CANNY_HIGH],
            "strides": {str(rate): stride for rate, stride in STRIDES.items()},
            "pair_spacing": PAIR_SPACING,
            "sift": {
                "features": SIFT_FEATURES,
                "ratio": SIFT_RATIO,
                "ransac_pixels": RANSAC_PIXELS,
                "min_pose_inliers": MIN_POSE_INLIERS,
            },
        },
        "segmenter": segmenter.describe(),
        "rates": rates,
        "frames_with_few_mask_cells": float(np.mean([cells.sum() < MIN_BOUNDARY for cells in data["mask"]])),
        "source_sha256": {"mask_motion.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    dump_json(output / "summary.json", summary)
    write_report(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path, help="The lake clip video")
    parser.add_argument("--reconstruction", type=Path, required=True, help="Reconstruction folder holding images.txt")
    parser.add_argument(
        "--model", type=Path, required=True, help="Folder of the exported int8 water-segmentation model"
    )
    parser.add_argument(
        "--output", type=Path, default=CLIP_RESULTS / "mask_motion", help="Fresh empty results directory"
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=REFERENCE_TIME_SCALE,
        help="Video seconds per second of the images.txt time column",
    )
    args = parser.parse_args()
    from uwslam.experiments.segmenter import EspdlSegmenter

    run(args.video, args.reconstruction, args.output, EspdlSegmenter(args.model), args.time_scale)


if __name__ == "__main__":
    main()
