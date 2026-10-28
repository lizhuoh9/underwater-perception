"""C monocular visual odometry (core/mono_vo.c through uwslam.mvo) on synthetic frames: a pure turn, and a camera moving
past two textured planes rendered by ray casting; and the refinement's Jacobian and a fast turn's pull, checked by C
programs (tests/c).

Needs gcc and OpenCV; skipped without either.
"""

import ctypes
from importlib.util import find_spec
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    import cv2

    from uwslam import mvo
    from uwslam.core import CFLAGS
    from uwslam.experiments.mask_motion import angle_deg, euler_rotation

WALL_Z, FLOOR_Y, TEXELS_PER_UNIT = 4.0, 1.0, 120.0
C_CHECKS = Path(__file__).resolve().parent / "c"
STATIC = ["-static"] if sys.platform == "win32" else []  # MinGW: no runtime DLLs to find when the program runs


def texture(seed: int, size: int = 1024) -> np.ndarray:
    coarse = np.random.default_rng(seed).integers(0, 256, (size // 16, size // 16), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC), (5, 5), 0)


def render(rotation: np.ndarray, centre: np.ndarray, camera, wall: np.ndarray, floor: np.ndarray,
           normalised: np.ndarray | None = None) -> np.ndarray:
    """320x240 grey view of a wall at z = WALL_Z and a floor at y = FLOOR_Y (world = first camera, y down). normalised:
    every pixel's ray as (x, y) at z = 1 (240 x 320 x 2), for a lens; the pinhole's by default."""
    v, u = np.mgrid[0:mvo.HEIGHT, 0:mvo.WIDTH].astype(np.float64)
    if normalised is None:
        normalised = np.stack([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy], axis=-1)
    rays = np.concatenate([normalised, np.ones_like(normalised[..., :1])], axis=-1) @ rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t_wall = np.where(rays[..., 2] > 1e-6, (WALL_Z - centre[2]) / rays[..., 2], np.inf)
        t_floor = np.where(rays[..., 1] > 1e-6, (FLOOR_Y - centre[1]) / rays[..., 1], np.inf)
    on_floor = t_floor < t_wall
    t = np.where(on_floor, t_floor, t_wall)
    hit = centre + rays * t[..., None]
    size = wall.shape[0]
    wall_view = cv2.remap(wall, (hit[..., 0] * TEXELS_PER_UNIT + size / 2).astype(np.float32),
                          (hit[..., 1] * TEXELS_PER_UNIT + size / 2).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    floor_view = cv2.remap(floor, (hit[..., 0] * TEXELS_PER_UNIT + size / 2).astype(np.float32),
                           (hit[..., 2] * TEXELS_PER_UNIT).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.where(on_floor, floor_view, wall_view).astype(np.uint8)


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs gcc and OpenCV")
class MonoVoCTests(unittest.TestCase):
    vo = None
    folder = None

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.vo = mvo.build(cls.folder)
        cls.camera = mvo.camera_for(110.0, 544 / 960)
        cls.wall, cls.floor = texture(1), texture(2)

    @classmethod
    def tearDownClass(cls):
        del cls.vo
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_state_fits_the_board(self):
        # The S3's largest free internal block is about 200 KB, so the two pyramids are separate blocks.
        self.assertLess(self.vo.mvo_state_size(), 16_000)
        self.assertLess(self.vo.mvo_pyramid_bytes(), 110_000)

    def test_a_pure_turn_is_a_rotation_step(self):
        rotation_b = euler_rotation(np.radians(5.0), np.radians(-1.5), np.radians(0.5))
        frames = [render(np.eye(3), np.zeros(3), self.camera, self.wall, self.floor),
                  render(rotation_b, np.zeros(3), self.camera, self.wall, self.floor)]
        steps, _, _ = mvo.run_frames(self.vo, frames, self.camera)
        self.assertEqual(mvo.KINDS[steps[0].kind], "rotation")
        found = np.array(steps[0].rotation, dtype=np.float64).reshape(3, 3)
        self.assertLess(angle_deg(found.T @ rotation_b), 0.5)

    def test_a_lost_step_keeps_its_turn(self):
        # Fast turns leave few tracks: with fewer than MVO_MIN_INLIERS (15) the step counts as lost, but with at least
        # MVO_MIN_TURN_INLIERS (8) agreeing on a homography it takes that rotation instead of none, which lost all of
        # an 8 degree turn here (on the lake clip at 5 Hz, 11 degrees per lost step in the median).
        wall, floor = np.full((1024, 1024), 128, np.uint8), np.full((1024, 1024), 128, np.uint8)
        wall[452:572, 452:572] = self.wall[452:572, 452:572]  # one textured patch: a dozen corners
        rotation_b = euler_rotation(np.radians(8.0), np.radians(-1.0), 0.0)
        frames = [render(np.eye(3), np.zeros(3), self.camera, wall, floor),
                  render(rotation_b, np.zeros(3), self.camera, wall, floor)]
        steps, _, _ = mvo.run_frames(self.vo, frames, self.camera)
        self.assertEqual(mvo.KINDS[steps[0].kind], "lost")
        self.assertLess(steps[0].tracks, 15)
        found = np.array(steps[0].rotation, dtype=np.float64).reshape(3, 3)
        self.assertLess(angle_deg(found.T @ rotation_b), 0.4 * angle_deg(rotation_b))

    def test_a_fast_turn_with_travel_keeps_its_rotation(self):
        # A 10 degree turn pulls the essential rotation towards the homography's, which slow travel's parallax biases:
        # with a clear sideways step past the near floor the pairs must still decide, so the pull may cost a little but
        # not the rotation (on the lake clip's fast swing it is what keeps a quarter of each turn from becoming travel).
        rotation_b, centre_b = euler_rotation(np.radians(10.0), np.radians(-1.0), 0.0), np.array([-0.3, 0.0, 0.15])
        frames = [render(np.eye(3), np.zeros(3), self.camera, self.wall, self.floor),
                  render(rotation_b, centre_b, self.camera, self.wall, self.floor)]
        steps, _, _ = mvo.run_frames(self.vo, frames, self.camera)
        self.assertEqual(mvo.KINDS[steps[0].kind], "essential")
        found = np.array(steps[0].rotation, dtype=np.float64).reshape(3, 3)
        self.assertLess(angle_deg(found.T @ rotation_b), 0.5)
        estimate = np.array(steps[0].centre, dtype=np.float64)
        cosine = centre_b @ estimate / (np.linalg.norm(centre_b) * np.linalg.norm(estimate))
        self.assertGreater(cosine, np.cos(np.radians(5.0)))

    def test_a_moving_camera_gives_direction_and_relative_scale(self):
        # Sideways and forward with a turn, so the direction of travel is well determined and the errors mostly measure
        # the solver; creeping straight towards a far wall leaves it ambiguous even for exact correspondences.
        poses = [(euler_rotation(np.radians(1.5 * k), 0.0, 0.0), np.array([0.30 * k, 0.0, 0.15 * k])) for k in range(8)]
        frames = [render(r, c, self.camera, self.wall, self.floor) for r, c in poses]
        steps, hashes, _ = mvo.run_frames(self.vo, frames, self.camera)
        kinds = [mvo.KINDS[s.kind] for s in steps]
        self.assertEqual(kinds.count("essential"), len(steps), kinds)
        lengths = []
        for k, step in enumerate(steps):
            (r_a, c_a), (r_b, c_b) = poses[k], poses[k + 1]
            truth = r_a.T @ (c_b - c_a)
            estimate = np.array(step.centre, dtype=np.float64)
            cosine = truth @ estimate / (np.linalg.norm(truth) * np.linalg.norm(estimate))
            self.assertGreater(cosine, np.cos(np.radians(5.0)), (k, truth, estimate))
            self.assertLess(angle_deg(np.array(step.rotation, dtype=np.float64).reshape(3, 3).T @ (r_a.T @ r_b)), 0.5)
            lengths.append(np.linalg.norm(estimate))
        ratios = np.array(lengths[1:]) / lengths[0]  # the true steps are all equal
        np.testing.assert_allclose(ratios, np.ones_like(ratios), atol=0.2)
        again, hashes_again, _ = mvo.run_frames(self.vo, frames, self.camera)
        self.assertEqual(hashes, hashes_again)  # deterministic: the board must reproduce these

    def test_far_features_stay_tracked(self):
        # Creeping towards the far wall: most points lie beyond MVO_MAX_DEPTH baselines, so they get no point, but they
        # must stay tracked. Dropping them (as recoverPose's mask would) left a few dozen of 250 features per step and
        # broke the chain of depths that carries the scale.
        poses = [(np.eye(3), np.array([0.0, 0.0, 0.06 * k])) for k in range(8)]
        frames = [render(r, c, self.camera, self.wall, self.floor) for r, c in poses]
        state = ctypes.create_string_buffer(self.vo.mvo_state_size())
        pyramids = [ctypes.create_string_buffer(self.vo.mvo_pyramid_bytes()) for _ in range(2)]
        self.vo.mvo_init(state, ctypes.byref(self.camera), mvo.SEED, pyramids[0], pyramids[1])
        xy, ids = (ctypes.c_float * (2 * mvo.MAX_FEATURES))(), (ctypes.c_uint32 * mvo.MAX_FEATURES)()
        tracked, kinds = [], []
        for k, frame in enumerate(frames):
            ctypes.memmove(self.vo.mvo_frame(state), np.ascontiguousarray(frame).ctypes.data, mvo.WIDTH * mvo.HEIGHT)
            if k == 0:
                self.vo.mvo_start(state)
            else:
                step = mvo.StepC()
                self.vo.mvo_step(state, ctypes.byref(step))
                kinds.append(mvo.KINDS[step.kind])
                self.assertLess(step.inliers, 0.5 * step.tracks)  # most points are too far for a point
            count = self.vo.mvo_tracks(state, xy, ids, mvo.MAX_FEATURES)
            tracked.append(set(ids[:count]))
        self.assertEqual(kinds, ["essential"] * (len(frames) - 1))
        for before, after in zip(tracked, tracked[1:]):
            self.assertGreater(len(before & after), 0.8 * len(before))

    def test_step_length_keeps_to_the_recent_speed(self):
        # The median depth ratio stays within MVO_SPEED_LIMIT (3) times the median of the last MVO_SPEED_STEPS (10)
        # lengths: on fast turns a median from a few points could be far off and the chain of depths kept that scale.
        # With fewer than MVO_MIN_SCALE_POINTS (8) ratios the length is that median.
        state = ctypes.create_string_buffer(self.vo.mvo_state_size())
        pyramids = [ctypes.create_string_buffer(self.vo.mvo_pyramid_bytes()) for _ in range(2)]
        self.vo.mvo_init(state, ctypes.byref(self.camera), mvo.SEED, pyramids[0], pyramids[1])

        def length(ratio: float, count: int = 8) -> float:
            ratios = (ctypes.c_float * max(count, 1))(*([ratio] * count))
            return self.vo.mvo_step_length(state, ratios, count)

        self.assertEqual(length(5.0, count=0), 1.0)  # the first step sets the unit
        self.assertEqual(length(2.5), 2.5)
        for _ in range(9):
            self.assertEqual(length(2.0), 2.0)
        self.assertEqual(length(10.0), 6.0)  # the last ten: 2.5 and nine 2.0, so at most 3 x 2
        self.assertAlmostEqual(length(0.1), 2.0 / 3.0, places=6)
        self.assertEqual(length(50.0, count=7), 2.0)
        for _ in range(10):
            self.assertEqual(length(5.0), 5.0)  # a real change of speed passes, and the speed follows it
        self.assertEqual(length(20.0), 15.0)

    def test_refinement_jacobian_is_analytic(self):
        # The refinement once took forward differences (a step of +h on every parameter), which biased every refined
        # rotation the same way whenever the tracks were noisy: with exact synthetic geometry and 0.3 px of noise, -0.04
        # degrees of pitch and -0.07 of yaw a step, a steady drift. The analytic Jacobian must match central differences
        # of the residuals, whose own error shrinks as h^2: to about a quarter when h halves.
        program = self.folder / "mvo_jacobian_check.exe"
        units = [str(mvo.CORE_DIR / name) for name in mvo.CORE_FILES if name.endswith(".c") and name != "mono_vo_essential.c"]
        subprocess.run(["gcc", *CFLAGS, *STATIC, f"-I{mvo.CORE_DIR}", str(C_CHECKS / "mvo_jacobian_check.c"), *units,
                        "-o", str(program)], check=True)
        printed = subprocess.run([str(program)], capture_output=True, text=True, check=True).stdout
        values = {key: float(value) for key, value in (pair.split("=") for pair in printed.split())}
        self.assertGreater(values["largest"], 10.0)
        self.assertLess(values["gap_half"], 0.02 * values["largest"])
        self.assertGreater(values["gap_h"], 3.0 * values["gap_half"])

    def test_turn_pull_follows_its_contract(self):
        # The pull of a fast turn's homography rotation (mono_vo.h): the refinement's normal-equation terms for it must
        # be half the gradient of its cost (a factor of two off on the Cayley steps would over- or under-pull every
        # fast turn), its cost must stay under its ceiling far from the homography's rotation along the turn's axis
        # (where that rotation is wrong it must stop pulling) and be nothing across the axis (pulling the whole
        # rotation dragged in the homography's tilt: 10 degrees more pitch over the lake clip), and a slow turn's
        # pairs must leave no pull at all while a fast turn's get one.
        program = self.folder / "mvo_turn_pull_check.exe"
        units = [str(mvo.CORE_DIR / name) for name in mvo.CORE_FILES if name.endswith(".c") and name != "mono_vo_essential.c"]
        subprocess.run(["gcc", *CFLAGS, *STATIC, f"-I{mvo.CORE_DIR}", str(C_CHECKS / "mvo_turn_pull_check.c"), *units,
                        "-o", str(program)], check=True)
        printed = subprocess.run([str(program)], capture_output=True, text=True, check=True).stdout
        values = {key: float(value) for key, value in (pair.split("=") for pair in printed.split())}
        self.assertGreater(values["gradient_largest"], 10.0)
        self.assertLess(values["gradient_gap"], 0.02 * values["gradient_largest"])
        self.assertLessEqual(values["ceiling"], 1.0)
        self.assertGreater(values["ceiling"], 0.9)
        self.assertLess(values["across"], 1e-6)
        self.assertEqual(values["slow_weight"], 0.0)
        self.assertGreater(values["fast_weight"], 0.0)

    def test_tracker_kernels_follow_their_contract(self):
        # mvo_kernels_t in mono_vo.h: what a platform's vector kernels must reproduce (the board checks them against these).
        rng = np.random.default_rng(7)
        side, lanes, stride = 2 * 7 + 1, 16, 40
        image = rng.integers(0, 256, (24, stride), dtype=np.uint8)
        weights = np.array([16384, 0, 0, 0], dtype=np.int16)
        while weights[0] == 16384:
            w = np.sort(rng.integers(0, 16385, 3))
            weights = np.array([w[0], w[1] - w[0], w[2] - w[1], 16384 - w[2]], dtype=np.int16)
        flat = image.ravel().astype(np.int64)

        def interpolate(at: np.ndarray) -> np.ndarray:  # at: flat pixel indices; Q7, each product shifted on its own
            corners = (at, at + 1, at + stride, at + stride + 1)
            return sum((flat[c] * int(w)) >> 7 for c, w in zip(corners, weights))

        j, i = np.mgrid[0:side, 0:side]
        origin = 2 * stride + 3  # the pixel one up and one left of the window
        centre = origin + (j + 1) * stride + (i + 1)
        patch, gx = interpolate(centre), interpolate(centre + 1) - interpolate(centre - 1)
        gy = interpolate(centre + stride) - interpolate(centre - stride)
        window = np.zeros((3, side, lanes), dtype=np.int16)
        g = np.zeros(3, dtype=np.int64)
        self.vo.mvo_lk_template_c(window.ctypes.data, image.ctypes.data + origin, stride, weights.ctypes.data, g.ctypes.data)
        for got, want in zip(window, (patch, gx, gy)):
            np.testing.assert_array_equal(got[:, :side], want)
            np.testing.assert_array_equal(got[:, side:], 0)
        np.testing.assert_array_equal(g, [(gx * gx).sum(), (gx * gy).sum(), (gy * gy).sum()])
        target = origin + stride + 2  # the window's top-left pixel in the other image, here a pixel right and down
        sums = np.zeros(2, dtype=np.int64)
        self.vo.mvo_lk_sums_c(window.ctypes.data, image.ctypes.data + target, stride, weights.ctypes.data, sums.ctypes.data)
        diff = patch - interpolate(target + j * stride + i)
        np.testing.assert_array_equal(sums, [(diff * gx).sum(), (diff * gy).sum()])

    def test_border_block_is_clamped_to_the_image(self):
        rng = np.random.default_rng(11)
        for w, h in ((40, 30), (80, 60), (320, 240)):
            image = rng.integers(0, 256, (h, w), dtype=np.uint8)
            for _ in range(20):
                size = int(rng.choice([16, 18]))
                stride = (size + 15) // 16 * 16
                origin = np.array([rng.integers(-20, w + 20), rng.integers(-20, h + 20)], dtype=np.int32)
                out = np.zeros(size * stride, dtype=np.uint8)
                self.vo.mvo_block_c(out.ctypes.data, size, image.ctypes.data, w, h, origin.ctypes.data)
                rows = np.clip(origin[1] + np.arange(size), 0, h - 1)
                columns = np.clip(origin[0] + np.arange(size), 0, w - 1)
                np.testing.assert_array_equal(out.reshape(size, stride)[:, :size], image[np.ix_(rows, columns)])

    def test_parallel_parts_are_independent(self):
        # Every split part (tracking, detection, direction search, refinements) writes only its own results, so running
        # the second part before the first must change nothing: the board runs them at the same time.
        poses = [(euler_rotation(np.radians(1.5 * k), 0.0, 0.0), np.array([0.30 * k, 0.0, 0.15 * k])) for k in range(6)]
        frames = [render(r, c, self.camera, self.wall, self.floor) for r, c in poses]
        work_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)
        calls = []

        @ctypes.CFUNCTYPE(None, work_type, ctypes.c_void_p)
        def backwards(work, context):
            calls.append(1)
            work(context, 1)
            work(context, 0)

        hashes = []
        for parallel in (None, backwards):
            state = ctypes.create_string_buffer(self.vo.mvo_state_size())
            pyramids = [ctypes.create_string_buffer(self.vo.mvo_pyramid_bytes()) for _ in range(2)]
            self.vo.mvo_init(state, ctypes.byref(self.camera), mvo.SEED, pyramids[0], pyramids[1])
            if parallel is not None:  # mvo_t ends with its two optional pointers: parallel, then kernels
                pointer = ctypes.sizeof(ctypes.c_void_p)
                ctypes.c_void_p.from_buffer(state, len(state) - 2 * pointer).value = ctypes.cast(parallel, ctypes.c_void_p).value
            run = []
            for k, frame in enumerate(frames):
                ctypes.memmove(self.vo.mvo_frame(state), np.ascontiguousarray(frame).ctypes.data, mvo.WIDTH * mvo.HEIGHT)
                if k == 0:
                    self.vo.mvo_start(state)
                    continue
                step = mvo.StepC()
                self.vo.mvo_step(state, ctypes.byref(step))
                run.append(mvo.fnv(mvo.FNV_OFFSET, bytes(step)))
            hashes.append(run)
        self.assertGreaterEqual(len(calls), 3 * (len(frames) - 1))  # tracking, direction search and refinements per step
        self.assertEqual(hashes[0], hashes[1])

    def test_square_root_without_division(self):
        values = np.array([4.0, 2.0, 0.25, 1e-3, 3.7e5, 1e30, 1e-29, 1e-31, 1e-38, 1e-44], dtype=np.float32)
        roots = np.array([self.vo.mvo_sqrt(float(v)) for v in values], dtype=np.float64)
        np.testing.assert_allclose(roots, np.sqrt(values.astype(np.float64)), rtol=3e-7)
        self.assertEqual(self.vo.mvo_sqrt(0.0), 0.0)
        self.assertEqual(self.vo.mvo_sqrt(-1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
