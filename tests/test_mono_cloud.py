"""The mask-free sparse 3D cloud on synthetic camera images and the video/board output contract."""

from importlib.util import find_spec
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    import cv2
    from tests.test_mvo import FLOOR_Y, WALL_Z, render, texture
    from uwslam import mono_cloud, mvo
    from uwslam.experiments.mask_motion import euler_rotation

ROOT = Path(__file__).resolve().parents[1]


class FakeCapture:
    def __init__(self, frames, reported_frames=None):
        self.frames = frames
        self.reported_frames = len(frames) if reported_frames is None else reported_frames
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.reported_frames
        if prop == cv2.CAP_PROP_POS_MSEC:
            return max(0, self.index - 1) * (1000.0 / 30.0)
        return 0.0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        gray = self.frames[self.index]
        self.index += 1
        return True, np.repeat(gray[..., None], 3, axis=2)

    def release(self):
        self.released = True


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs gcc and OpenCV")
class MonoCloudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.vo = mvo.build(cls.folder)
        cls.camera = mvo.camera_for(110.0, 544 / 960)
        wall, floor = texture(1), texture(2)
        cls.poses = [(euler_rotation(np.radians(1.5 * k), 0.0, 0.0), np.array([0.30 * k, 0.0, 0.15 * k]))
                     for k in range(8)]
        cls.frames = [render(rotation, centre, cls.camera, wall, floor) for rotation, centre in cls.poses]
        cls.unit = float(np.linalg.norm(cls.poses[1][1] - cls.poses[0][1]))

    @classmethod
    def tearDownClass(cls):
        del cls.vo
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_original_frames_accumulate_wall_and_floor_without_mask(self):
        result = mono_cloud.run_frames(self.vo, iter(self.frames), self.camera)
        cloud = result["cloud"]
        self.assertGreater(cloud.count, 100)
        self.assertEqual(cloud.continuous, 1)
        self.assertTrue(all(item["kind"] == "essential" for item in result["trajectory"][1:]))
        points = np.array([list(point.xyz) for point in cloud.points[:cloud.count]]) * self.unit
        # Only this known synthetic fixture supplies the unit for comparison; VO has no metric scale.
        nearest_surface = np.minimum(abs(points[:, 2] - WALL_Z), abs(points[:, 1] - FLOOR_Y))
        self.assertLess(float(np.median(nearest_surface)), 0.15)
        self.assertGreater(int(np.sum(abs(points[:, 2] - WALL_Z) < 0.4)), 20)
        self.assertGreater(int(np.sum(abs(points[:, 1] - FLOOR_Y) < 0.2)), 10)

    def test_first_loss_freezes_old_points_after_tracking_resumes(self):
        before = mono_cloud.run_frames(self.vo, iter(self.frames[:2]), self.camera)
        frames = list(self.frames)
        frames[2] = np.full((mvo.HEIGHT, mvo.WIDTH), 128, dtype=np.uint8)
        result = mono_cloud.run_frames(self.vo, iter(frames), self.camera)
        self.assertGreater(before["cloud"].count, 0)
        self.assertEqual([entry["kind"] for entry in result["trajectory"]][2:5],
                         ["lost", "lost", "essential"])
        self.assertEqual(result["first_lost_frame"], 2)
        self.assertEqual(result["cloud"].continuous, 0)
        self.assertEqual(result["cloud"].count, before["cloud"].count)
        self.assertEqual(result["cloud"].dropped, before["cloud"].dropped)
        self.assertEqual([bytes(point) for point in result["cloud"].points[:result["cloud"].count]],
                         [bytes(point) for point in before["cloud"].points[:before["cloud"].count]])
        self.assertTrue(all(item["cloud"]["accepted"] == 0 for item in result["trajectory"][2:]))

    def test_pure_turn_does_not_create_depth(self):
        wall, floor = texture(1), texture(2)
        rotation = euler_rotation(np.radians(5.0), np.radians(-1.5), np.radians(0.5))
        frames = [render(np.eye(3), np.zeros(3), self.camera, wall, floor),
                  render(rotation, np.zeros(3), self.camera, wall, floor)]
        result = mono_cloud.run_frames(self.vo, iter(frames), self.camera)
        self.assertEqual(result["trajectory"][1]["kind"], "rotation")
        self.assertEqual(result["cloud"].count, 0)

    def test_video_exports_ply_trajectory_and_consistent_board_expectations(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            video, output = root / "video.fake", root / "out"
            video.write_bytes(b"known fake decoder input")
            capture = FakeCapture(self.frames, reported_frames=10)
            with patch.object(cv2, "VideoCapture", return_value=capture):
                summary = mono_cloud.run_video(video, output, self.camera, every=2, board_files=True)
            self.assertTrue(capture.released)
            self.assertEqual(summary["frames"], 4)
            self.assertGreater(summary["count"], 0)
            self.assertEqual(summary["source_fps"], 30.0)
            self.assertEqual(summary["source_reported_frame_count"], 10)
            self.assertEqual(summary["source_decoded_frame_count"], 8)
            self.assertFalse(summary["source_frame_count_consistent"])
            self.assertEqual(summary["source_read_termination"], "read_false_eof_or_decode_error")
            trajectory = json.loads((output / "trajectory.json").read_text(encoding="utf-8"))
            expected = json.loads((output / "board_expected.json").read_text(encoding="utf-8"))
            expected_cloud = json.loads((output / "cloud_expected.json").read_text(encoding="utf-8"))
            self.assertEqual([row["source_frame"] for row in trajectory], [0, 2, 4, 6])
            self.assertAlmostEqual(trajectory[2]["source_timestamp_s"], 4 / 30)
            frames_bytes = (output / "frames.bin").read_bytes()
            frame_size = mvo.WIDTH * mvo.HEIGHT
            self.assertEqual(len(frames_bytes), 4 * frame_size)
            self.assertEqual(expected["step_hashes"], expected_cloud["step_hashes"])
            self.assertEqual(expected["frames"], 4)
            self.assertEqual(expected["every"], 1)
            self.assertEqual(expected["rate_hz"], 15.0)
            self.assertEqual(expected_cloud["cloud_policy"], 1)
            self.assertEqual(expected_cloud["frames_sha256"], hashlib.sha256(frames_bytes).hexdigest())
            self.assertEqual(expected_cloud["frame_sha256"],
                             [hashlib.sha256(frames_bytes[i:i + frame_size]).hexdigest()
                              for i in range(0, len(frames_bytes), frame_size)])
            self.assertEqual(expected_cloud["source_sha256"], summary["source_sha256"])
            self.assertEqual(expected_cloud["frame_coordinates"], summary["frame_coordinates"])
            self.assertEqual(expected_cloud["point_units"], summary["point_units"])
            self.assertEqual(expected_cloud["cloud_progress"], [row["cloud"] for row in trajectory[1:]])
            self.assertEqual(expected_cloud["cloud_hash"], summary["cloud_hash"])
            self.assertEqual(expected_cloud["cloud_count"], summary["count"])
            header, vertices = (output / "cloud.ply").read_text(encoding="ascii").split("end_header\n", 1)
            self.assertIn(f"element vertex {summary['count']}", header)
            self.assertIn(f"comment continuous {summary['continuous']}", header)
            self.assertIn(f"comment evicted_points {summary['dropped']}", header)
            self.assertIn(f"comment cloud_sha256 {summary['sources_sha256']['cloud_sha256']}", header)
            self.assertIn(f"comment cloud_header_sha256 {summary['sources_sha256']['cloud_header_sha256']}", header)
            self.assertIn(f"comment source_video_sha256 {summary['source_sha256']}", header)
            self.assertEqual(len(vertices.splitlines()), summary["count"])
            for line in vertices.splitlines():
                values = line.split()
                self.assertEqual(len(values), 6)
                self.assertTrue(all(np.isfinite(float(value)) for value in values[:3]))
                self.assertTrue(np.isfinite(float(values[5])))

    def test_bad_camera_or_frame_and_existing_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "intrinsics"):
            mono_cloud.run_frames(self.vo, iter(self.frames[:1]), mvo.Camera(float("nan"), 1, 0, 0))
        with self.assertRaisesRegex(ValueError, "240x320"):
            mono_cloud.run_frames(self.vo, iter([np.zeros((10, 10), dtype=np.uint8)]), self.camera)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            existing = root / "existing"
            existing.mkdir()
            marker = existing / "keep.txt"
            marker.write_text("do not overwrite", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                mono_cloud.run_video(root / "missing", existing, self.camera)
            self.assertEqual(marker.read_text(encoding="utf-8"), "do not overwrite")
            with self.assertRaisesRegex(ValueError, "every"):
                mono_cloud.run_video(root / "missing", root / "new", self.camera, every=0)

    def test_timestamps_must_not_go_backward_across_missing_values(self):
        frames = [mono_cloud.Frame(self.frames[0], 0, 0.2), mono_cloud.Frame(self.frames[1], 1, None),
                  mono_cloud.Frame(self.frames[2], 2, 0.1)]
        with self.assertRaisesRegex(ValueError, "timestamps must be nondecreasing"):
            mono_cloud.run_frames(self.vo, iter(frames), self.camera)
        frames[2] = mono_cloud.Frame(self.frames[2], 2, 0.2)
        result = mono_cloud.run_frames(self.vo, iter(frames), self.camera)
        self.assertEqual([row["source_timestamp_s"] for row in result["trajectory"]], [0.2, None, 0.2])

    def test_unavailable_container_frame_count_stays_unknown(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            video, output = root / "video.fake", root / "out"
            video.write_bytes(b"fake decoder input")
            capture = FakeCapture(self.frames[:3], reported_frames=0)
            with patch.object(cv2, "VideoCapture", return_value=capture):
                summary = mono_cloud.run_video(video, output, self.camera, every=2)
            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["source_decoded_frame_count"], 3)
            self.assertIsNone(summary["source_reported_frame_count"])
            self.assertIsNone(summary["source_frame_count_consistent"])
            self.assertEqual(summary["source_read_termination"], "read_false_eof_or_decode_error")

    def test_empty_video_has_no_output_and_releases_decoder(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            video, output = root / "empty.fake", root / "out"
            video.write_bytes(b"fake empty video")
            capture = FakeCapture([])
            with patch.object(cv2, "VideoCapture", return_value=capture):
                with self.assertRaisesRegex(ValueError, "no frames"):
                    mono_cloud.run_video(video, output, self.camera)
            self.assertTrue(capture.released)
            self.assertFalse(output.exists())

    def test_decoder_metadata_error_releases_decoder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "video.fake"
            video.write_bytes(b"fake video")
            capture = FakeCapture(self.frames[:1])
            with patch.object(cv2, "VideoCapture", return_value=capture), \
                    patch.object(capture, "get", side_effect=RuntimeError("decoder metadata failed")):
                with self.assertRaisesRegex(RuntimeError, "decoder metadata failed"):
                    mono_cloud.run_video(video, root / "out", self.camera)
            self.assertTrue(capture.released)

    def test_cli_on_a_real_encoded_synthetic_video(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            video, output = root / "synthetic.avi", root / "out"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (mvo.WIDTH, mvo.HEIGHT))
            if not writer.isOpened():
                writer.release()
                self.skipTest("OpenCV MJPG video writer unavailable")
            try:
                for frame in self.frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            finally:
                writer.release()
            command = [sys.executable, "-m", "uwslam.mono_cloud", str(video), "--fx", str(self.camera.fx),
                       "--fy", str(self.camera.fy), "--cx", str(self.camera.cx), "--cy", str(self.camera.cy),
                       "--board-files", "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["frames"], len(self.frames))
            self.assertAlmostEqual(summary["source_fps"], 30.0)
            self.assertGreater(summary["count"], 0)
            self.assertEqual((output / "frames.bin").stat().st_size, len(self.frames) * mvo.WIDTH * mvo.HEIGHT)

    def test_c_cloud_invariants(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            program = Path(folder) / "cloud_check.exe"
            subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                            *(["-static"] if sys.platform == "win32" else []), f"-I{ROOT / 'core'}",
                            str(ROOT / "tests/c/mono_cloud_check.c"), str(ROOT / "core/mono_cloud.c"),
                            "-o", str(program)], check=True)
            subprocess.run([str(program)], check=True)


if __name__ == "__main__":
    unittest.main()
