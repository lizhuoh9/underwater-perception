"""Monocular visual odometry on the robot camera's own frames (no IMU, no rangers), checked against the lake clip.

Every frame of the video up to the last reference keyframe takes the robot camera's path: stretched to 320x240 and
truncated to RGB565, as the camera delivers it (segmenter.device_frame's first step), then grey; the 160x120 variant
2x2-averages it first, as the segmentation model's input does. Frames are processed at 10, 5 or 3 Hz (every 3rd, 6th
or 10th frame of the 30 FPS clip).

Tracking: Shi-Tomasi corners followed by pyramidal Lucas-Kanade from each processed frame to the next, kept when the
backward track returns within FB_MAX of the start; new corners fill the gaps whenever fewer than REFILL_SHARE of
MAX_CORNERS survive. Motion: a RANSAC homography gives the rotation a pure turn would need; when the tracks stray from
it by less than MIN_PARALLAX (median) the pair is that pure rotation, otherwise the essential matrix by RANSAC gives
the rotation and the direction of travel.
A pair with fewer than MIN_INLIERS keeps the last pose and counts as lost. Scale: a single camera sees translation only
up to scale, so the first step sets the unit and every later step takes its length from the points that both it and
the previous step triangulate (median depth ratio in their shared frame); with fewer than MIN_SCALE_POINTS such points,
or after a lost pair, the step repeats the previous length. All pixel thresholds are given at 320x240 and halve at
160x120.

The estimated camera centres are compared with the reference after one similarity alignment for the whole run
(Umeyama: scale, rotation, offset), which is all a monocular map can promise; rotation per pair, the direction of
travel and the chained heading need no alignment because both start from the same first camera. Reference poses,
their 3x time scale and the 110 degree horizontal field of view follow uwslam.experiments.mask_motion.
"""

import argparse
import csv
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uwslam.experiments.mask_motion import (
    REFERENCE_TIME_SCALE,
    TIME_SLACK,
    Camera,
    angle_deg,
    reference_poses,
    rotation_vector,
)
from uwslam.paths import CLIP_RESULTS
from uwslam.utils import dump_json

ASPECT = 544 / 960  # the lake clip's source frame
HFOV_DEG = 110.0  # fitted in mask_motion on SIFT rotations of turns above 3 degrees
# The lens distortion the C tools give the lake camera (k1, k2, p1, p2, k3 in OpenCV's model): p1 alone, a distortion
# centred off the frame's middle in height, as a frame cropped off the lens's axis has. Found without the reference: with
# it the chain of 30 FPS steps agrees in pitch with direct rotations across 1.6 s of the same tracks, and the floating
# dock's straight bottom edge keeps its vanishing point through the clip; without it every step pitched up a little.
LENS = (0.0, 0.0, -0.02, 0.0, 0.0)
RATES = {10.0: 3, 5.0: 6, 3.0: 10}  # Hz: stride in 30 FPS frames
RESOLUTIONS = {"320x240": (320, 240), "160x120": (160, 120)}
MAX_CORNERS, QUALITY, MIN_DISTANCE = 250, 0.01, 8.0
REFILL_SHARE = 0.6
LK_WINDOW, LK_LEVELS = 21, 3
FB_MAX, RANSAC_PIXELS, MIN_PARALLAX = 1.0, 1.0, 1.0  # pixels at 320x240
MIN_INLIERS, MIN_SCALE_POINTS = 15, 8
MIN_INIT_PARALLAX, MIN_INIT_INLIERS = 8.0, 40  # pixels at 320x240; matches against the first keyframe
KF_PARALLAX = 6.0  # pixels at 320x240, rotation-compensated median against the last keyframe
MIN_PNP_POINTS, PNP_PIXELS = 12, 2.0
MAX_REPROJECTION, MIN_TRIANGULATION_DEG = 2.0, 1.0
METHODS = ("pairwise", "keyframe")
MIN_REFERENCE_STEP = 0.02  # reconstruction units: shorter reference steps carry no usable direction of travel
DRIFT_WINDOW_S = 5.0


@dataclass
class Step:
    """One processed pair: R_AB (a ray in B is R_AB d_B in A), B's centre in A's frame, and how it was found."""

    rotation: np.ndarray
    centre: np.ndarray
    kind: str  # "essential", "rotation", "lost"
    tracks: int
    inliers: int
    scale_points: int


def robot_grey(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Any RGB frame -> the robot camera's grey frame at 320x240, or 2x2-averaged to 160x120."""
    import cv2
    from PIL import Image

    from uwslam.experiments.segmenter import RGB565_KEEP, WORK_HEIGHT, WORK_WIDTH, model_frame

    work = np.array(Image.fromarray(rgb).resize((WORK_WIDTH, WORK_HEIGHT), Image.Resampling.BILINEAR)) & RGB565_KEEP
    frame = work if size == (WORK_WIDTH, WORK_HEIGHT) else model_frame(work)
    if frame.shape[1::-1] != size:
        raise ValueError(f"Unsupported size {size}")
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def detect(grey: np.ndarray, existing: np.ndarray, scale: float, count: int) -> np.ndarray:
    """Up to `count` new Shi-Tomasi corners at least MIN_DISTANCE from each other and from the existing points."""
    import cv2

    mask = np.full(grey.shape, 255, dtype=np.uint8)
    for x, y in existing.reshape(-1, 2):
        cv2.circle(mask, (int(round(x)), int(round(y))), int(round(MIN_DISTANCE * scale)), 0, -1)
    corners = cv2.goodFeaturesToTrack(
        grey, maxCorners=count, qualityLevel=QUALITY, minDistance=MIN_DISTANCE * scale, mask=mask, blockSize=3
    )
    return np.zeros((0, 2), np.float32) if corners is None else corners.reshape(-1, 2).astype(np.float32)


def track(
    first: np.ndarray, second: np.ndarray, points: np.ndarray, scale: float, predicted: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Pyramidal Lucas-Kanade forward and back, starting from `predicted` positions when given; returns the kept
    points' positions in `second` and the keep mask."""
    import cv2

    if not len(points):
        return np.zeros((0, 2), np.float32), np.zeros(0, bool)
    window = (max(7, int(LK_WINDOW * scale) | 1),) * 2
    levels = LK_LEVELS if scale >= 1 else LK_LEVELS - 1
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    if predicted is None:
        forward, ok, _ = cv2.calcOpticalFlowPyrLK(
            first, second, points.reshape(-1, 1, 2), None, winSize=window, maxLevel=levels, criteria=criteria
        )
    else:
        forward, ok, _ = cv2.calcOpticalFlowPyrLK(
            first,
            second,
            points.reshape(-1, 1, 2),
            predicted.reshape(-1, 1, 2).astype(np.float32).copy(),
            winSize=window,
            maxLevel=levels,
            criteria=criteria,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(
        second, first, forward, None, winSize=window, maxLevel=levels, criteria=criteria
    )
    error = np.linalg.norm(back.reshape(-1, 2) - points.reshape(-1, 2), axis=1)
    height, width = first.shape
    moved = forward.reshape(-1, 2)
    inside = (moved[:, 0] >= 0) & (moved[:, 0] < width) & (moved[:, 1] >= 0) & (moved[:, 1] < height)
    keep = ok.ravel().astype(bool) & ok_back.ravel().astype(bool) & (error < FB_MAX * scale) & inside
    return moved[keep], keep


def triangulate(
    points_a: np.ndarray, points_b: np.ndarray, rotation_cv: np.ndarray, translation_cv: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """3D points in A's camera frame from two views with X_B = R X_A + t (OpenCV's convention)."""
    import cv2

    first = matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
    second = matrix @ np.hstack([rotation_cv, translation_cv.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(first, second, points_a.T.astype(np.float64), points_b.T.astype(np.float64))
    return (homogeneous[:3] / homogeneous[3]).T


def pure_rotation(points_a: np.ndarray, points_b: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray | None, int]:
    """R_AB of a pure rotation from a RANSAC homography H = K R K^-1 (X_B = R X_A), and its inlier count."""
    import cv2

    homography, inliers = cv2.findHomography(points_a, points_b, cv2.RANSAC, 2.0 * RANSAC_PIXELS)
    if homography is None:
        return None, 0
    u, _, vt = np.linalg.svd(np.linalg.inv(matrix) @ homography @ matrix)
    rotation_cv = u @ vt
    if np.linalg.det(rotation_cv) < 0:
        rotation_cv = -rotation_cv
    return rotation_cv.T, int(inliers.sum())


def parallax(points_a: np.ndarray, points_b: np.ndarray, rotation_ab: np.ndarray, matrix: np.ndarray) -> float:
    """Median distance (pixels) between where the tracks went and where the rotation alone would have taken them: the
    part of the motion that only translation can explain."""
    rays = np.hstack([points_a, np.ones((len(points_a), 1))]) @ np.linalg.inv(matrix).T
    turned = rays @ (matrix @ rotation_ab.T).T
    predicted = turned[:, :2] / turned[:, 2:3]
    return float(np.median(np.linalg.norm(points_b - predicted, axis=1)))


class MonoVO:
    """Frame-to-frame monocular odometry with scale carried by triangulated points (see the module docstring)."""

    def __init__(self, camera: Camera, scale: float):
        self.matrix, self.scale = camera.matrix(), scale
        self.grey, self.points, self.ids = None, np.zeros((0, 2), np.float32), np.zeros(0, int)
        self.next_id, self.depths, self.step_length = 0, {}, None

    def refill(self, grey: np.ndarray) -> None:
        if len(self.points) < REFILL_SHARE * MAX_CORNERS:
            new = detect(grey, self.points, self.scale, MAX_CORNERS - len(self.points))
            self.points = np.vstack([self.points, new])
            self.ids = np.concatenate([self.ids, np.arange(self.next_id, self.next_id + len(new))])
            self.next_id += len(new)

    def start(self, grey: np.ndarray) -> None:
        self.grey = grey
        self.refill(grey)

    def step(self, grey: np.ndarray) -> Step:
        import cv2

        moved, keep = track(self.grey, grey, self.points, self.scale)
        before, ids = self.points[keep], self.ids[keep]
        self.grey, self.points, self.ids = grey, moved, ids
        rotation, inliers = pure_rotation(before, moved, self.matrix) if len(moved) >= MIN_INLIERS else (None, 0)
        if rotation is None or inliers < MIN_INLIERS:
            result = self.lose(len(moved))
        elif parallax(before, moved, rotation, self.matrix) < MIN_PARALLAX * self.scale:
            kept = set(ids.tolist())
            self.depths = {key: rotation.T @ point for key, point in self.depths.items() if key in kept}
            result = Step(rotation, np.zeros(3), "rotation", len(moved), inliers, 0)
        else:
            essential, mask = cv2.findEssentialMat(
                before, moved, self.matrix, cv2.RANSAC, 0.999, RANSAC_PIXELS * self.scale
            )
            if essential is None or essential.shape != (3, 3):
                result = self.lose(len(moved))
            else:
                count, rotation_cv, translation_cv, pose_mask = cv2.recoverPose(
                    essential, before, moved, self.matrix, mask=mask
                )
                inlier = pose_mask.ravel() > 0
                if count < MIN_INLIERS:
                    result = self.lose(len(moved))
                else:
                    result = self.translate(
                        before[inlier], moved[inlier], ids[inlier], rotation_cv, translation_cv.ravel()
                    )
                    self.points, self.ids = (
                        moved[inlier],
                        ids[inlier],
                    )  # outliers of the epipolar model stop being tracked
        self.refill(grey)
        return result

    def lose(self, tracks: int) -> Step:
        """No pose for this pair; the points' depths are dropped but the last step length is kept, so the next step
        continues at the same scale until shared points re-anchor it."""
        self.depths = {}
        return Step(np.eye(3), np.zeros(3), "lost", tracks, 0, 0)

    def translate(self, before, moved, ids, rotation_cv, translation_cv) -> Step:
        """Scale the unit translation by the depth ratio of points the previous step also triangulated."""
        in_a = triangulate(before, moved, rotation_cv, translation_cv, self.matrix)
        in_b = in_a @ rotation_cv.T + translation_cv
        valid = (in_a[:, 2] > 0) & (in_b[:, 2] > 0)
        ratios = [
            self.depths[key][2] / point[2]
            for key, point, ok in zip(ids.tolist(), in_a, valid)
            if ok and key in self.depths and self.depths[key][2] > 0
        ]
        if len(ratios) >= MIN_SCALE_POINTS:
            length = float(np.median(ratios))
        else:
            length = self.step_length if self.step_length is not None else 1.0
        self.step_length = length
        self.depths = {key: length * point for key, point, ok in zip(ids.tolist(), in_b, valid) if ok}
        centre = -rotation_cv.T @ translation_cv * length
        return Step(rotation_cv.T, centre, "essential", len(before), int(len(before)), len(ratios))


def turned_pixels(points: np.ndarray, rotation_ab: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Where pixels of frame A land in frame B if the camera only turned by R_AB (x_B ~ K R_AB^T K^-1 x_A)."""
    rays = np.hstack([points, np.ones((len(points), 1))]) @ np.linalg.inv(matrix).T
    turned = rays @ (matrix @ rotation_ab.T).T
    return (turned[:, :2] / turned[:, 2:3]).astype(np.float32)


def triangulate_world(
    pixels_a: np.ndarray, pixels_b: np.ndarray, pose_a: tuple, pose_b: tuple, matrix: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """World points seen from two posed cameras (camera-to-world rotation, centre) and which of them to trust: in front
    of both cameras, reprojecting within MAX_REPROJECTION and seen under at least MIN_TRIANGULATION_DEG."""
    import cv2

    projections = []
    for rotation, centre in (pose_a, pose_b):
        projections.append(matrix @ np.hstack([rotation.T, (-rotation.T @ centre).reshape(3, 1)]))
    homogeneous = cv2.triangulatePoints(
        projections[0], projections[1], pixels_a.T.astype(np.float64), pixels_b.T.astype(np.float64)
    )
    world = (homogeneous[:3] / homogeneous[3]).T
    good = np.isfinite(world).all(axis=1)
    for (rotation, centre), pixels in zip((pose_a, pose_b), (pixels_a, pixels_b)):
        local = (world - centre) @ rotation
        good &= local[:, 2] > 0
        projected = local @ matrix.T
        error = np.linalg.norm(projected[:, :2] / np.where(local[:, 2:3] > 0, projected[:, 2:3], 1.0) - pixels, axis=1)
        good &= error < MAX_REPROJECTION * scale
    rays_a, rays_b = world - pose_a[1], world - pose_b[1]
    cosine = np.einsum("ij,ij->i", rays_a, rays_b) / (
        np.linalg.norm(rays_a, axis=1) * np.linalg.norm(rays_b, axis=1) + 1e-12
    )
    good &= cosine < np.cos(np.radians(MIN_TRIANGULATION_DEG))
    return world, good


class KeyframeVO:
    """Monocular odometry with keyframes and a local map. Until it is initialised the camera only turns (a homography
    against the first keyframe); once the tracks show MIN_INIT_PARALLAX against that keyframe, the essential matrix
    places the current frame at distance `baseline` (1 for the very first start) and its inliers become map points.
    Every later frame takes its pose from PnP (RANSAC, then refined) against the map points it still tracks, so the
    scale lives in the map. A new keyframe is made when the rotation-compensated parallax against the last one reaches
    KF_PARALLAX or too few tracked features have map points; features seen in both keyframes then join the map. When
    PnP fails, one step against the previous frame takes its place: a pure rotation keeps the map, otherwise the
    essential matrix at the last step length gives the pose and its inliers the new map; only when that fails too is
    the map dropped and initialised again, at the last keyframe spacing. Lucas-Kanade starts from where the previous
    frame-to-frame rotation would move each feature."""

    def __init__(self, camera: Camera, scale: float):
        self.matrix, self.scale = camera.matrix(), scale
        self.grey, self.points, self.ids, self.next_id = None, np.zeros((0, 2), np.float32), np.zeros(0, int), 0
        self.map, self.rotation, self.centre, self.turn = {}, np.eye(3), np.zeros(3), np.eye(3)
        self.keyframe, self.baseline, self.step_length, self.initialised = None, 1.0, 1.0, False

    refill = MonoVO.refill

    def set_keyframe(self) -> None:
        if self.keyframe is not None and self.initialised:
            spacing = float(np.linalg.norm(self.centre - self.keyframe["centre"]))
            self.baseline = spacing if spacing > 0 else self.baseline
        self.keyframe = {
            "rotation": self.rotation.copy(),
            "centre": self.centre.copy(),
            "pixels": {int(key): point.copy() for key, point in zip(self.ids, self.points)},
        }

    def start(self, grey: np.ndarray) -> None:
        self.grey = grey
        self.refill(grey)
        self.set_keyframe()

    def common(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ids tracked since the last keyframe, their keyframe pixels and their current pixels."""
        rows = [k for k, key in enumerate(self.ids.tolist()) if key in self.keyframe["pixels"]]
        ids = self.ids[rows]
        return ids, np.array([self.keyframe["pixels"][key] for key in ids.tolist()]).reshape(-1, 2), self.points[rows]

    def step(self, grey: np.ndarray) -> Step:
        rotation_before, centre_before = self.rotation.copy(), self.centre.copy()
        predicted = turned_pixels(self.points, self.turn, self.matrix) if len(self.points) else None
        moved, keep = track(self.grey, grey, self.points, self.scale, predicted)
        before = self.points[keep]
        self.grey, self.points, self.ids = grey, moved, self.ids[keep]
        if self.initialised:
            kind, inliers = self.track_map()
            if kind == "lost":
                kind, inliers = self.recover(before, rotation_before, centre_before)
        else:
            kind, inliers = self.initialise()
        if kind == "pnp" and self.needs_keyframe():
            self.extend_map()
            self.set_keyframe()
        travelled = float(np.linalg.norm(self.centre - centre_before))
        if kind in ("pnp", "init", "recovered") and travelled > 0:
            self.step_length = travelled
        self.refill(grey)
        self.turn = rotation_before.T @ self.rotation
        centre = rotation_before.T @ (self.centre - centre_before)
        return Step(
            self.turn.copy(), centre, kind, len(moved), inliers, sum(int(key) in self.map for key in self.ids.tolist())
        )

    def initialise(self) -> tuple[str, int]:
        import cv2

        ids, before, after = self.common()
        if len(ids) < MIN_INIT_INLIERS:
            self.set_keyframe()
            return "lost", 0
        rotation, inliers = pure_rotation(before, after, self.matrix)
        if rotation is None:
            return "lost", 0
        keyframe_pose = (self.keyframe["rotation"], self.keyframe["centre"])
        if parallax(before, after, rotation, self.matrix) < MIN_INIT_PARALLAX * self.scale:
            self.rotation = keyframe_pose[0] @ rotation
            return "rotation", inliers
        essential, mask = cv2.findEssentialMat(
            before, after, self.matrix, cv2.RANSAC, 0.999, RANSAC_PIXELS * self.scale
        )
        if essential is None or essential.shape != (3, 3):
            return "lost", 0
        count, rotation_cv, translation_cv, pose_mask = cv2.recoverPose(
            essential, before, after, self.matrix, mask=mask
        )
        if count < MIN_INIT_INLIERS:
            self.rotation = keyframe_pose[0] @ rotation
            return "rotation", inliers
        inlier = pose_mask.ravel() > 0
        self.rotation = keyframe_pose[0] @ rotation_cv.T
        self.centre = keyframe_pose[1] + keyframe_pose[0] @ (-rotation_cv.T @ translation_cv.ravel()) * self.baseline
        world, good = triangulate_world(
            before[inlier], after[inlier], keyframe_pose, (self.rotation, self.centre), self.matrix, self.scale
        )
        self.map = {int(key): point for key, point, ok in zip(ids[inlier].tolist(), world, good) if ok}
        if len(self.map) < MIN_PNP_POINTS:
            self.map, self.rotation, self.centre = {}, keyframe_pose[0] @ rotation, keyframe_pose[1].copy()
            return "rotation", inliers
        self.initialised = True
        self.set_keyframe()
        return "init", int(count)

    def drop_map(self) -> tuple[str, int]:
        self.map, self.initialised = {}, False
        self.set_keyframe()
        return "lost", 0

    def recover(self, before: np.ndarray, rotation_before: np.ndarray, centre_before: np.ndarray) -> tuple[str, int]:
        """PnP failed: one step against the previous frame instead, a pure rotation (map kept) or the essential matrix
        at the last step length, whose inliers become the new map."""
        import cv2

        if len(before) < MIN_INLIERS:
            return self.drop_map()
        rotation, inliers = pure_rotation(before, self.points, self.matrix)
        if rotation is None or inliers < MIN_INLIERS:
            return self.drop_map()
        if parallax(before, self.points, rotation, self.matrix) < MIN_PARALLAX * self.scale:
            self.rotation = rotation_before @ rotation
            self.set_keyframe()
            return "rotation", inliers
        essential, mask = cv2.findEssentialMat(
            before, self.points, self.matrix, cv2.RANSAC, 0.999, RANSAC_PIXELS * self.scale
        )
        if essential is None or essential.shape != (3, 3):
            return self.drop_map()
        count, rotation_cv, translation_cv, pose_mask = cv2.recoverPose(
            essential, before, self.points, self.matrix, mask=mask
        )
        if count < MIN_INLIERS:
            return self.drop_map()
        inlier = pose_mask.ravel() > 0
        self.rotation = rotation_before @ rotation_cv.T
        self.centre = centre_before + rotation_before @ (-rotation_cv.T @ translation_cv.ravel()) * self.step_length
        world, good = triangulate_world(
            before[inlier],
            self.points[inlier],
            (rotation_before, centre_before),
            (self.rotation, self.centre),
            self.matrix,
            self.scale,
        )
        self.map = {int(key): point for key, point, ok in zip(self.ids[inlier].tolist(), world, good) if ok}
        self.set_keyframe()
        return "recovered", int(count)

    def track_map(self) -> tuple[str, int]:
        import cv2

        rows = [k for k, key in enumerate(self.ids.tolist()) if key in self.map]
        if len(rows) < MIN_PNP_POINTS:
            return "lost", 0
        objects = np.array([self.map[key] for key in self.ids[rows].tolist()])
        pixels = self.points[rows].astype(np.float64)
        to_camera = self.rotation.T
        rvec, tvec = cv2.Rodrigues(to_camera)[0], (-to_camera @ self.centre).reshape(3, 1)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objects,
            pixels,
            self.matrix,
            None,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            iterationsCount=100,
            reprojectionError=PNP_PIXELS * self.scale,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < MIN_PNP_POINTS:
            return "lost", 0
        inliers = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(objects[inliers], pixels[inliers], self.matrix, None, rvec, tvec)
        to_camera = cv2.Rodrigues(rvec)[0]
        self.rotation, self.centre = to_camera.T, -to_camera.T @ tvec.ravel()
        outliers = set(range(len(rows))) - set(inliers.tolist())
        for k in outliers:  # points that disagree with the pose leave the map
            self.map.pop(int(self.ids[rows[k]]), None)
        return "pnp", int(len(inliers))

    def needs_keyframe(self) -> bool:
        ids, before, after = self.common()
        mapped = sum(int(key) in self.map for key in self.ids.tolist())
        if mapped < 2 * MIN_PNP_POINTS:
            return True
        if not len(ids):
            return True
        turn = self.keyframe["rotation"].T @ self.rotation
        return parallax(before, after, turn, self.matrix) >= KF_PARALLAX * self.scale

    def extend_map(self) -> None:
        ids, before, after = self.common()
        new = np.array([int(key) not in self.map for key in ids.tolist()], dtype=bool)
        if not new.any():
            return
        world, good = triangulate_world(
            before[new],
            after[new],
            (self.keyframe["rotation"], self.keyframe["centre"]),
            (self.rotation, self.centre),
            self.matrix,
            self.scale,
        )
        for key, point, ok in zip(ids[new].tolist(), world, good):
            if ok:
                self.map[int(key)] = point


def run_vo(
    frames: list[np.ndarray], stride: int, camera: Camera, scale: float, method: type = MonoVO
) -> tuple[list[int], list[Step], np.ndarray, np.ndarray, float]:
    """Processed frame indices, the steps, camera-to-world rotations and centres (world = first camera), ms per frame."""
    vo = method(camera, scale)
    indices = list(range(0, len(frames), stride))
    vo.start(frames[0])
    rotations, centres, steps = [np.eye(3)], [np.zeros(3)], []
    started = time.perf_counter()
    for index in indices[1:]:
        step = vo.step(frames[index])
        steps.append(step)
        rotations.append(rotations[-1] @ step.rotation)
        centres.append(centres[-1] + rotations[-2] @ step.centre)
    elapsed = (time.perf_counter() - started) * 1000 / max(1, len(steps))
    return indices, steps, np.stack(rotations), np.stack(centres), elapsed


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising |s R source + t - target|^2 over corresponding rows."""
    mean_s, mean_t = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mean_s, target - mean_t
    u, d, vt = np.linalg.svd(b.T @ a / len(source))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    variance = (a**2).sum() / len(source)
    scale = float(np.trace(np.diag(d) @ sign) / variance) if variance > 0 else 0.0
    return scale, rotation, mean_t - scale * rotation @ mean_s


def map_heading(rotations: np.ndarray) -> np.ndarray:
    """Unwrapped heading of camera-to-world rotations (forward axis, map X = world z, Y = -world x)."""
    forward = rotations[..., :, 2]
    return np.unwrap(np.arctan2(-forward[..., 0], forward[..., 2]))


def evaluate(
    indices: list[int],
    steps: list[Step],
    rotations: np.ndarray,
    centres: np.ndarray,
    times: np.ndarray,
    reference_rotations: np.ndarray,
    reference_centres: np.ndarray,
) -> dict:
    """Errors of one run against the reference (see the module docstring)."""
    ref_r, ref_c = reference_rotations[indices], reference_centres[indices]
    scale, rotation, offset = umeyama(centres, ref_c)
    aligned = scale * centres @ rotation.T + offset
    position_error = np.linalg.norm(aligned - ref_c, axis=1)
    pair_errors, zero_errors, direction_errors = [], [], []
    for k, step in enumerate(steps):
        reference = ref_r[k].T @ ref_r[k + 1]
        pair_errors.append(angle_deg(step.rotation.T @ reference))
        zero_errors.append(angle_deg(reference))
        reference_step = ref_r[k].T @ (ref_c[k + 1] - ref_c[k])
        if np.linalg.norm(step.centre) > 0 and np.linalg.norm(reference_step) > MIN_REFERENCE_STEP:
            cosine = step.centre @ reference_step / (np.linalg.norm(step.centre) * np.linalg.norm(reference_step))
            direction_errors.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    heading = np.degrees(np.abs((map_heading(rotations) - map_heading(ref_r) + np.pi) % (2 * np.pi) - np.pi))
    path_reference = np.linalg.norm(np.diff(ref_c, axis=0), axis=1)
    path_estimate = np.linalg.norm(np.diff(aligned, axis=0), axis=1)
    windows = []
    t = times[indices][1:]
    for start in np.arange(0.0, t[-1], DRIFT_WINDOW_S):
        inside = (t >= start) & (t < start + DRIFT_WINDOW_S)
        if path_reference[inside].sum() > 0.1:
            windows.append(float(path_estimate[inside].sum() / path_reference[inside].sum()))
    kinds = [step.kind for step in steps]
    return {
        "steps": len(steps),
        "lost": kinds.count("lost"),
        "rotation_only": kinds.count("rotation"),
        "kinds": {kind: kinds.count(kind) for kind in sorted(set(kinds))},
        "median_tracks": float(np.median([s.tracks for s in steps])),
        "ate_rmse": float(np.sqrt(np.mean(position_error**2))),
        "ate_max": float(position_error.max()),
        "final_error": float(position_error[-1]),
        "path_length_reference": float(path_reference.sum()),
        "alignment_scale": scale,
        "pair_rotation_median_deg": float(np.median(pair_errors)),
        "pair_rotation_p90_deg": float(np.percentile(pair_errors, 90)),
        "zero_rotation_median_deg": float(np.median(zero_errors)),
        "direction_median_deg": float(np.median(direction_errors)) if direction_errors else None,
        "direction_pairs": len(direction_errors),
        "heading_rmse_deg": float(np.sqrt(np.mean(heading**2))),
        "heading_max_deg": float(heading.max()),
        "scale_drift_window_ratio_min": float(min(windows)) if windows else None,
        "scale_drift_window_ratio_max": float(max(windows)) if windows else None,
        "aligned": aligned,
    }


def read_frames(video: Path, end_s: float) -> tuple[dict[str, list[np.ndarray]], float, tuple[int, int]]:
    """Grey robot frames at both resolutions for every video frame up to end_s, the frame rate and the source size."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = {name: [] for name in RESOLUTIONS}
    try:
        while len(frames["320x240"]) / fps <= end_s + TIME_SLACK:
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"{video.name} ends before the last reference keyframe")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for name, size in RESOLUTIONS.items():
                frames[name].append(robot_grey(rgb, size))
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    return frames, float(fps), size


def number(value, digits: int = 2, unit: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{unit}"


def write_report(output: Path, summary: dict) -> None:
    lines = [
        "# Monocular visual odometry on the robot camera's frames (lake clip, reference poses)",
        "",
        f"Source: `{summary['source']['name']}`, {summary['source']['frames']} frames ({summary['source']['duration_s']:.1f} s "
        f"at {summary['source']['fps']:g} FPS); reference: the reconstruction's keyframes at the {summary['time_scale']:g}× pose "
        f"timing, interpolated to every frame. No IMU, depth or ranger data are used.",
        "",
        "- Frames take the robot camera's path (320×240, RGB565) and turn grey; 160×120 is the 2×2 average the "
        "segmentation model sees. Intrinsics: a 110° horizontal field of view (fitted in the mask_motion probe).",
        "- Both methods track Shi-Tomasi corners with pyramidal Lucas-Kanade and a forward-backward check. *pairwise*: "
        "the essential matrix by RANSAC between consecutive processed frames (a homography when the tracks barely "
        "move), with scale carried from step to step by the points both steps triangulate. *keyframe*: the essential "
        "matrix only to initialise, then PnP against a map of points triangulated between keyframes (made when the "
        "rotation-compensated parallax reaches 6 px at 320×240), and tracks predicted by the last rotation.",
        "- Positions are compared after one similarity alignment of the whole run (scale, rotation, offset); "
        "rotation, the direction of travel and the heading are compared directly.",
        "",
        "## Results",
        "",
        "| Method | Frames | Rate | Steps (lost / rotation only) | Position RMSE after alignment | Final error | Heading RMSE / max | "
        "Rotation per pair, median (no rotation) | Direction of travel, median | Scale ratio over 5 s windows | ms per frame (PC) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["runs"]:
        lines.append(
            f"| {row['method']} | {row['resolution']} | {row['rate_hz']:g} Hz | {row['steps']} ({row['lost']} / {row['rotation_only']}) | "
            f"{number(row['ate_rmse'], 2, ' m')} | {number(row['final_error'], 2, ' m')} | "
            f"{number(row['heading_rmse_deg'], 1, '°')} / {number(row['heading_max_deg'], 0, '°')} | "
            f"{number(row['pair_rotation_median_deg'], 2, '°')} ({number(row['zero_rotation_median_deg'], 2, '°')}) | "
            f"{number(row['direction_median_deg'], 1, '°')} | "
            f"{number(row['scale_drift_window_ratio_min'], 2)}–{number(row['scale_drift_window_ratio_max'], 2)} | "
            f"{number(row['ms_per_frame'], 1)} |"
        )
    lines += [
        "",
        f"The reference camera travels {summary['runs'][0]['path_length_reference']:.1f} reconstruction units "
        '(taken as metres). "Scale ratio over 5 s windows" is the aligned estimated path length over the reference '
        "path length in each 5 s window: 1 everywhere means no scale drift.",
        "",
        "## Limits",
        "",
        "- One diver clip, not the robot; the reference is a reconstruction interpolated between keyframes, in its "
        "own units.",
        "- A monocular run has no metric scale: the alignment supplies it after the fact.",
        "- PC timings are for the OpenCV implementation and say nothing about the ESP32-P4.",
        "",
        "Re-run (the output folder must not exist or be empty):",
        "",
        "```bash",
        "python -m uwslam.experiments.mono_vo <video> --reconstruction <folder with images.txt> "
        "--output results/lake_clip/mono_vo_2",
        "```",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(video: Path, reconstruction: Path, output: Path, time_scale: float = REFERENCE_TIME_SCALE) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite results; choose a fresh --output")
    poses = np.loadtxt(reconstruction / "images.txt")
    frames, fps, size = read_frames(video, poses[-1, 0] * time_scale)
    times = np.arange(len(frames["320x240"])) / fps
    reference_rotations, reference_centres = reference_poses(poses, times, time_scale)
    runs, trajectories = [], []
    for method, vo_class in zip(METHODS, (MonoVO, KeyframeVO)):
        for name, (width, height) in RESOLUTIONS.items():
            camera = Camera.from_fov(HFOV_DEG, ASPECT, width, height)
            for rate, stride in RATES.items():
                indices, steps, rotations, centres, ms = run_vo(frames[name], stride, camera, width / 320, vo_class)
                result = evaluate(indices, steps, rotations, centres, times, reference_rotations, reference_centres)
                aligned = result.pop("aligned")
                runs.append({"method": method, "resolution": name, "rate_hz": rate, "ms_per_frame": ms, **result})
                for k, index in enumerate(indices):
                    trajectories.append(
                        {
                            "method": method,
                            "resolution": name,
                            "rate_hz": rate,
                            "frame": index,
                            "t": round(float(times[index]), 4),
                            **{f"est_{a}": round(float(v), 4) for a, v in zip("xyz", aligned[k])},
                            **{f"ref_{a}": round(float(v), 4) for a, v in zip("xyz", reference_centres[index])},
                            **{
                                f"est_rot_{a}": round(float(v), 4)
                                for a, v in zip("xyz", np.degrees(rotation_vector(rotations[k])))
                            },
                        }
                    )
                print(
                    f"{method} {name} {rate:g} Hz: ATE {result['ate_rmse']:.2f}, heading RMSE {result['heading_rmse_deg']:.1f} deg, "
                    f"lost {result['lost']}/{result['steps']}",
                    flush=True,
                )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectories[0]))
        writer.writeheader()
        writer.writerows(trajectories)
    summary = {
        "task": "Monocular visual odometry (KLT + essential matrix, scale from triangulated points) on the robot "
        "camera's grey frames, no IMU or rangers, against reconstructed reference poses",
        "source": {
            "name": video.name,
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "fps": fps,
            "frames": len(times),
            "duration_s": float(times[-1]),
            "width": size[0],
            "height": size[1],
            "poses_sha256": hashlib.sha256((reconstruction / "images.txt").read_bytes()).hexdigest(),
        },
        "time_scale": time_scale,
        "hfov_deg": HFOV_DEG,
        "methods": list(METHODS),
        "settings": {
            "max_corners": MAX_CORNERS,
            "quality": QUALITY,
            "min_distance": MIN_DISTANCE,
            "refill_share": REFILL_SHARE,
            "lk_window": LK_WINDOW,
            "lk_levels": LK_LEVELS,
            "fb_max": FB_MAX,
            "ransac_pixels": RANSAC_PIXELS,
            "min_parallax": MIN_PARALLAX,
            "min_inliers": MIN_INLIERS,
            "min_scale_points": MIN_SCALE_POINTS,
            "min_init_parallax": MIN_INIT_PARALLAX,
            "min_init_inliers": MIN_INIT_INLIERS,
            "kf_parallax": KF_PARALLAX,
            "min_pnp_points": MIN_PNP_POINTS,
            "pnp_pixels": PNP_PIXELS,
            "max_reprojection": MAX_REPROJECTION,
            "min_triangulation_deg": MIN_TRIANGULATION_DEG,
            "strides": {str(rate): stride for rate, stride in RATES.items()},
        },
        "runs": runs,
        "source_sha256": {"mono_vo.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    dump_json(output / "summary.json", summary)
    write_report(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path, help="The lake clip video")
    parser.add_argument("--reconstruction", type=Path, required=True, help="Reconstruction folder holding images.txt")
    parser.add_argument("--output", type=Path, default=CLIP_RESULTS / "mono_vo", help="Fresh empty results directory")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=REFERENCE_TIME_SCALE,
        help="Video seconds per second of the images.txt time column",
    )
    args = parser.parse_args()
    run(args.video, args.reconstruction, args.output, args.time_scale)


if __name__ == "__main__":
    main()
