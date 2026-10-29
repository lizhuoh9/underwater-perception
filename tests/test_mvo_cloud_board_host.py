"""End-to-end PC/firmware-handler replay using real VO and cloud C code, no device."""

import hashlib
import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np

HAS_DEPENDENCIES = shutil.which("gcc") is not None and find_spec("cv2") is not None
if HAS_DEPENDENCIES:
    from tests.test_mvo import render, texture
    from uwslam import mono_cloud, mvo, mvo_stream
    from uwslam.core import CFLAGS
    from uwslam.experiments.mask_motion import euler_rotation

ROOT = Path(__file__).resolve().parents[1]


class ProcessBoard:
    """Implements the host's Board interface with a host-only firmware subprocess."""

    def __init__(self, program: Path | list[str], write_timeout: float = 15.0, close_timeout: float = 15.0):
        command = [str(program)] if isinstance(program, Path) else program
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        self.lines = []
        self.output = queue.Queue()
        self.write_timeout, self.close_timeout = write_timeout, close_timeout
        self.failed, self.closed, self.writer = False, False, None

        def drain():
            for line in iter(self.process.stdout.readline, b""):
                self.output.put(line.decode("ascii", errors="replace").strip())
            self.output.put(None)

        self.reader = threading.Thread(target=drain, daemon=True)
        self.reader.start()

    def send(self, data: bytes) -> None:
        failures = []

        def write():
            try:
                remaining = memoryview(data)
                while remaining:
                    written = self.process.stdin.write(remaining)
                    if not written:
                        raise RuntimeError("The firmware host process closed its input")
                    remaining = remaining[written:]
                self.process.stdin.flush()
            except BaseException as exc:
                failures.append(exc)

        self.writer = threading.Thread(target=write, daemon=True)
        self.writer.start()
        self.writer.join(timeout=self.write_timeout)
        if self.writer.is_alive():
            self.failed = True
            if self.process.poll() is None:
                try:
                    self.process.kill()
                except OSError:
                    pass  # the subprocess may have exited between poll and kill
            self.process.wait(timeout=5)
            self.writer.join(timeout=5)
            raise TimeoutError("Host firmware did not read the frame within the write timeout")
        if failures:
            self.failed = True
            raise RuntimeError("Could not send data to host firmware") from failures[0]

    def reply(self, prefixes: tuple[str, ...], timeout: float) -> str:
        while True:
            try:
                line = self.output.get(timeout=timeout)
            except queue.Empty as exc:
                self.failed = True
                raise TimeoutError(f"No host firmware reply for {prefixes}") from exc
            if line is None:
                self.failed = True
                raise RuntimeError("The firmware host process exited before replying")
            if line:
                self.lines.append(line)
            if line.startswith("error"):
                self.failed = True
                raise RuntimeError(f"Host firmware reported: {line}")
            if line.startswith(prefixes):
                return line

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        suppress = self.failed or sys.exc_info()[0] is not None
        cleanup_error = None
        try:
            self.process.stdin.close()
        except OSError as exc:
            cleanup_error = exc
        try:
            status = self.process.wait(timeout=self.close_timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            status = self.process.wait(timeout=5)
            cleanup_error = RuntimeError("Host firmware did not terminate after input closed")
        if self.writer is not None:
            self.writer.join(timeout=2)
        self.reader.join(timeout=2)
        stderr = self.process.stderr.read().decode("utf-8", errors="replace")
        self.process.stdout.close()
        self.process.stderr.close()
        if not suppress:
            if cleanup_error is not None:
                raise cleanup_error
            if status:
                raise RuntimeError(f"Host firmware exited {status}: {stderr}")


class ProcessBoardTimeoutTests(unittest.TestCase):
    def test_write_timeout_reaps_child_and_unblocks_writer(self):
        board = ProcessBoard(
            [sys.executable, "-u", "-c", "import time; time.sleep(60)"], write_timeout=0.5, close_timeout=0.5
        )
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "write timeout"):
            try:
                board.send(bytes(8 * 1024 * 1024))  # exceed the child pipe buffer while it never reads
            finally:
                board.close()
        self.assertLess(time.monotonic() - started, 6)
        self.assertIsNotNone(board.process.poll())
        self.assertFalse(board.writer.is_alive())
        self.assertFalse(board.reader.is_alive())

    def test_close_does_not_replace_an_active_reply_error(self):
        board = ProcessBoard(
            [sys.executable, "-u", "-c", "import time; print('error simulated'); time.sleep(60)"],
            write_timeout=0.5,
            close_timeout=0.5,
        )
        with self.assertRaisesRegex(RuntimeError, "reported: error simulated"):
            try:
                board.reply(("ready",), timeout=5)
            finally:
                board.close()
        self.assertIsNotNone(board.process.poll())
        self.assertFalse(board.reader.is_alive())


@unittest.skipUnless(HAS_DEPENDENCIES, "Needs gcc and OpenCV")
class MonoCloudBoardHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.vo = mvo.build(cls.folder / "library")
        cls.camera = mvo.camera_for(110.0, 544 / 960)
        include = cls.folder / "include"
        for name in (
            "driver/usb_serial_jtag.h",
            "driver/usb_serial_jtag_vfs.h",
            "esp_cpu.h",
            "esp_heap_caps.h",
            "esp_timer.h",
            "freertos/FreeRTOS.h",
            "freertos/semphr.h",
            "freertos/task.h",
            "sdkconfig.h",
        ):
            header = include / name
            header.parent.mkdir(parents=True, exist_ok=True)
            header.write_text("", encoding="ascii")
        fingerprints = mono_cloud.fingerprints()
        definitions = {
            "MVO_CORE_SHA256": fingerprints["core_sha256"],
            "MVO_HEADER_SHA256": fingerprints["header_sha256"],
            "MC_CORE_SHA256": fingerprints["cloud_sha256"],
            "MC_HEADER_SHA256": fingerprints["cloud_header_sha256"],
            "MM_CORE_SHA256": hashlib.sha256((ROOT / "core/mono_map.c").read_bytes()).hexdigest(),
            "MM_HEADER_SHA256": hashlib.sha256((ROOT / "core/mono_map.h").read_bytes()).hexdigest(),
            "VM_CORE_SHA256": hashlib.sha256((ROOT / "core/voxel_map.c").read_bytes()).hexdigest(),
            "VM_HEADER_SHA256": hashlib.sha256((ROOT / "core/voxel_map.h").read_bytes()).hexdigest(),
            "OD_CORE_SHA256": hashlib.sha256((ROOT / "core/obstacle_dir.c").read_bytes()).hexdigest(),
            "OD_HEADER_SHA256": hashlib.sha256((ROOT / "core/obstacle_dir.h").read_bytes()).hexdigest(),
            "PP_CORE_SHA256": hashlib.sha256((ROOT / "core/path_plan.c").read_bytes()).hexdigest(),
            "PP_HEADER_SHA256": hashlib.sha256((ROOT / "core/path_plan.h").read_bytes()).hexdigest(),
        }
        header = cls.folder / "hashes.h"
        header.write_text("".join(f'#define {key} "{value}"\n' for key, value in definitions.items()), encoding="ascii")
        cls.program = cls.folder / ("mvo_real_replay.exe" if sys.platform == "win32" else "mvo_real_replay")
        command = [
            "gcc",
            *CFLAGS,
            *mvo.config_defines(),
            "-Wno-unused-function",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            *(["-static"] if sys.platform == "win32" else []),
            f"-I{include}",
            f"-I{ROOT / 'core'}",
            "-include",
            str(header),
            str(ROOT / "tests/c/mvo_real_replay.c"),
            *map(str, mvo.SOURCES),
            str(mvo.MAP_SOURCE),
            str(mvo.CLOUD_SOURCE),
            str(mvo.VOXEL_SOURCE),
            str(mvo.DIRECTION_SOURCE),
            str(mvo.PLAN_SOURCE),
            "-o",
            str(cls.program),
        ]
        env = dict(os.environ, TMP=str(cls.folder), TEMP=str(cls.folder))
        subprocess.run(command, check=True, env=env)
        cls.fingerprints = fingerprints
        wall, floor = texture(1), texture(2)
        cls.poses = [
            (euler_rotation(np.radians(1.5 * k), 0.0, 0.0), np.array([0.30 * k, 0.0, 0.15 * k])) for k in range(8)
        ]
        cls.frames = [render(rotation, centre, cls.camera, wall, floor) for rotation, centre in cls.poses]

    @classmethod
    def tearDownClass(cls):
        del cls.vo
        shutil.rmtree(cls.folder, ignore_errors=True)

    def replay(self, frames):
        result = mono_cloud.run_frames(self.vo, iter(frames), self.camera)
        cloud = result["cloud"]
        frame_data = [np.ascontiguousarray(frame).tobytes() for frame in frames]
        camera_bits = [
            struct.unpack("<I", struct.pack("<f", getattr(self.camera, name)))[0] for name in ("fx", "fy", "cx", "cy")
        ]
        expected = {
            "frames": len(frames),
            "every": 1,
            "rate_hz": None,
            "step_hashes": result["step_hashes"],
            "camera_bits": camera_bits,
            "seed": mvo.SEED,
            **{key: self.fingerprints[key] for key in ("core_sha256", "header_sha256")},
            **mvo.build_config(),
        }
        expected_cloud = {
            "frames": len(frames),
            "step_hashes": result["step_hashes"],
            "cloud_policy": 1,
            **{key: self.fingerprints[key] for key in ("cloud_sha256", "cloud_header_sha256")},
            "frames_sha256": hashlib.sha256(b"".join(frame_data)).hexdigest(),
            "frame_sha256": [hashlib.sha256(frame).hexdigest() for frame in frame_data],
            "cloud_progress": result["cloud_progress"],
            "cloud_hash": result["cloud_hash"],
            "cloud_count": int(cloud.count),
            "cloud_dropped": int(cloud.dropped),
            "cloud_continuous": int(cloud.continuous),
            **mvo.build_config(),
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            folder = Path(temp)
            path = folder / "frames.bin"
            path.write_bytes(b"".join(frame_data))
            vo_path, cloud_path = folder / "vo.json", folder / "cloud.json"
            vo_path.write_text(json.dumps(expected), encoding="utf-8")
            cloud_path.write_text(json.dumps(expected_cloud), encoding="utf-8")
            with (
                patch.object(mvo_stream, "Board", side_effect=lambda port, baud: ProcessBoard(self.program)),
                patch("builtins.print"),
            ):
                record, lines = mvo_stream.stream("host-only", path, vo_path, cores=1, expected_cloud_path=cloud_path)
        self.assertTrue(record["all_match"])
        self.assertTrue(record["cloud"]["match"])
        self.assertEqual(record["cloud"]["hash"], result["cloud_hash"])
        self.assertEqual(len(record["cloud"]["points"]), cloud.count)
        for pc, board in zip(cloud.points[: cloud.count], record["cloud"]["points"]):
            self.assertEqual(board["id"], pc.id)
            self.assertEqual(board["observations"], pc.observations)
            self.assertEqual(struct.pack("<3f", *board["xyz"]), struct.pack("<3f", *pc.xyz))
            self.assertEqual(struct.pack("<f", board["spread"]), struct.pack("<f", pc.spread))
        return result, record, lines

    def test_normal_video_real_vo_and_firmware_produce_identical_cloud(self):
        result, record, lines = self.replay(self.frames)
        self.assertTrue(all(item["kind"] == "essential" for item in result["trajectory"][1:]))
        self.assertGreater(record["cloud"]["count"], 100)
        self.assertEqual(record["cloud"]["continuous"], 1)
        self.assertEqual(sum(line.startswith("step ") for line in lines), len(self.frames) - 1)

    def test_blackout_freezes_real_cloud_even_after_tracking_resumes(self):
        frames = list(self.frames)
        frames[2] = np.full((mvo.HEIGHT, mvo.WIDTH), 128, dtype=np.uint8)
        before = mono_cloud.run_frames(self.vo, iter(frames[:2]), self.camera)
        result, record, _ = self.replay(frames)
        self.assertEqual([row["kind"] for row in result["trajectory"]][2:5], ["lost", "lost", "essential"])
        self.assertEqual(record["cloud"]["continuous"], 0)
        self.assertEqual(result["cloud"].count, before["cloud"].count)
        self.assertGreater(result["cloud"].count, 0)
        self.assertEqual(
            [bytes(point) for point in result["cloud"].points[: result["cloud"].count]],
            [bytes(point) for point in before["cloud"].points[: before["cloud"].count]],
        )

    def test_voxel_map_and_directions_match_the_pc_through_the_firmware(self):
        """With masks (one frame without), a blackout and the real handlers: every step's voxels and plan, every
        frame's directions and the final hashes equal the PC's, and the voxel map goes on after the loss."""
        from uwslam import mono_map, obstacle_dir, path_plan, voxel_map

        frames = list(self.frames)
        frames[3] = np.full((mvo.HEIGHT, mvo.WIDTH), 128, dtype=np.uint8)
        masks = []
        for k in range(len(frames)):
            mask = np.ones((60, 80), dtype=np.uint8)
            mask[:24, 4 * k : 4 * k + 30] = 0  # open water up and to the side, drifting right
            masks.append(mask)
        masks[5] = None
        voxels = voxel_map.run_mapping(self.vo, frames, masks, self.camera)
        dirs = obstacle_dir.run_directions(self.vo, masks, self.camera)
        grid = mono_map.run_mapping(self.vo, frames, masks, self.camera)
        plans = path_plan.run_planning(self.vo, frames, masks, self.camera)
        self.assertEqual(plans["plan_hashes"][-1], plans["hash"])
        self.assertEqual(voxels["stats"]["lost"], 2)
        self.assertGreater(sum(item["filed"] for item in voxels["progress"][4:]), 0)
        camera_bits = [
            struct.unpack("<I", struct.pack("<f", getattr(self.camera, name)))[0] for name in ("fx", "fy", "cx", "cy")
        ]
        step_hashes = [mvo.fnv(mvo.FNV_OFFSET, bytes(step)) for step in voxels["steps"]]
        expected = {
            "frames": len(frames),
            "every": 1,
            "rate_hz": 10.0,
            "step_hashes": step_hashes,
            "camera_bits": camera_bits,
            "seed": mvo.SEED,
            **{key: self.fingerprints[key] for key in ("core_sha256", "header_sha256")},
            **mvo.build_config(),
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            folder = Path(temp)
            frames_path, vo_path, masks_path = folder / "frames.bin", folder / "vo.json", folder / "masks.bin"
            frames_path.write_bytes(b"".join(np.ascontiguousarray(frame).tobytes() for frame in frames))
            vo_path.write_text(json.dumps(expected), encoding="utf-8")
            mono_map.write_board_files(folder, masks_path, grid, masks, 10.0)
            voxel_map.write_board_files(folder, voxels, 10.0)
            obstacle_dir.write_board_files(folder, dirs, masks, 10.0)
            path_plan.write_board_files(folder, plans, 10.0)
            with (
                patch.object(mvo_stream, "Board", side_effect=lambda port, baud: ProcessBoard(self.program)),
                patch("builtins.print"),
            ):
                record, _ = mvo_stream.stream(
                    "host-only",
                    frames_path,
                    vo_path,
                    cores=1,
                    masks_path=masks_path,
                    expected_map_path=folder / "board_expected_map_10hz.json",
                    expected_voxels_path=folder / "board_expected_voxels_10hz.json",
                    expected_dirs_path=folder / "board_expected_dirs_10hz.json",
                    expected_plan_path=folder / "board_expected_plan_10hz.json",
                )
        self.assertTrue(record["all_match"])
        self.assertTrue(record["map"]["match"])
        self.assertTrue(record["voxels"]["match"])
        self.assertTrue(record["dirs"]["match"])
        self.assertTrue(record["plan"]["match"])
        self.assertEqual(record["plan"]["frames"], len(frames) - 1)
        # a part of the run is not compared on its final hashes, but a step that differs still fails it
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            folder = Path(temp)
            frames_path, vo_path, masks_path = folder / "frames.bin", folder / "vo.json", folder / "masks.bin"
            frames_path.write_bytes(b"".join(np.ascontiguousarray(frame).tobytes() for frame in frames))
            vo_path.write_text(json.dumps(expected), encoding="utf-8")
            mono_map.write_board_files(folder, masks_path, grid, masks, 10.0)
            path_plan.write_board_files(folder, plans, 10.0)
            plan_path = folder / "board_expected_plan_10hz.json"
            tampered = json.loads(plan_path.read_text(encoding="utf-8"))
            tampered["plan_hashes"][1] ^= 1
            plan_path.write_text(json.dumps(tampered), encoding="utf-8")
            with (
                patch.object(mvo_stream, "Board", side_effect=lambda port, baud: ProcessBoard(self.program)),
                patch("builtins.print"),
            ):
                partial, _ = mvo_stream.stream(
                    "host-only",
                    frames_path,
                    vo_path,
                    limit=4,
                    cores=1,
                    masks_path=masks_path,
                    expected_map_path=folder / "board_expected_map_10hz.json",
                    expected_plan_path=plan_path,
                )
        self.assertIs(partial["plan"]["match"], False)
        self.assertEqual(partial["plan"]["mismatched_steps"], [2])
        self.assertEqual(record["voxels"]["count"], voxels["stats"]["count"])
        self.assertEqual(record["voxels"]["lost"], 2)
        self.assertEqual(record["dirs"]["frames"], len(frames) - 1)

    def test_full_cloud_keeps_updating_with_real_vo_and_identical_evictions(self):
        wall, floor = texture(1), texture(2)
        frames = list(self.frames)
        for k in range(len(frames), 24):
            rotation = euler_rotation(np.radians(1.5 * k), 0.0, 0.0)
            centre = np.array([0.30 * k, 0.0, 0.15 * k])
            frames.append(render(rotation, centre, self.camera, wall, floor))
        result, record, _ = self.replay(frames)
        self.assertTrue(all(item["kind"] == "essential" for item in result["trajectory"][1:]))
        self.assertEqual(record["cloud"]["count"], mono_cloud.MAX_POINTS)
        self.assertGreater(record["cloud"]["dropped"], 0)
        full_index = next(
            i for i, state in enumerate(result["cloud_progress"]) if state["count"] == mono_cloud.MAX_POINTS
        )
        self.assertGreater(result["cloud_progress"][-1]["dropped"], result["cloud_progress"][full_index]["dropped"])


if __name__ == "__main__":
    unittest.main()
