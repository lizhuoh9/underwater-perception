"""The board host's pure parts (uwslam.mvo_stream): parsing the board's lines, the frame hash, and the boards section
it writes into RESULTS.md. No board or serial port needed."""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from uwslam import mvo_stream


def record(cores: int, mismatched: list[int], with_map: bool, simd: bool) -> dict:
    timing = {
        f"{stage}_ms": {"mean": 10.0 * k, "median": 9.0 * k, "p90": 12.0 * k, "max": 15.0 * k}
        for k, stage in enumerate(("total", "track", "motion", "detect"), start=1)
    }
    entry = {
        "cores": cores,
        "simd": simd,
        "frames": 5,
        "steps": 4,
        "mismatched_steps": mismatched,
        "timing": timing,
        "cpu_mhz": 360,
        "config": "p4",
    }
    if with_map:
        entry["timing"]["map_ms"] = {"mean": 0.25, "median": 0.2, "p90": 0.3, "max": 0.4}
        entry["map"] = {"observations": 42, "match": True}
    return entry


class MvoStreamTests(unittest.TestCase):
    def test_fields_reads_key_values_and_numbers(self):
        line = "step index=3 kind=2 hash=4294967295 total_us=-1 core_sha256=ab12"
        self.assertEqual(
            mvo_stream.fields(line), {"index": 3, "kind": 2, "hash": 4294967295, "total_us": -1, "core_sha256": "ab12"}
        )

    def test_fnv_matches_the_reference_values(self):
        self.assertEqual(mvo_stream.fnv(b""), 2166136261)
        self.assertEqual(mvo_stream.fnv(b"a"), 0xE40C292C)  # FNV-1a 32-bit of "a"

    def test_board_section_is_written_once_and_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder)
            (results / "board").mkdir()
            (results / "RESULTS.md").write_text("# Report\n\nBody.\n", encoding="utf-8")
            (results / "board" / "a.json").write_text(json.dumps(record(2, [], True, True)), encoding="utf-8")
            mvo_stream.write_board_section(results)
            (results / "board" / "b.json").write_text(json.dumps(record(1, [3], False, False)), encoding="utf-8")
            mvo_stream.write_board_section(results)
            text = (results / "RESULTS.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Report\n\nBody.\n"))
        self.assertEqual(text.count(mvo_stream.BOARD_HEADING), 1)
        self.assertIn(
            "| `a` | 360 MHz | p4 | 10 Hz | 2 | vector | 5 | 4 of 4 | 42 observations, hash equal to the PC's |", text
        )
        self.assertIn("| `b` | 360 MHz | p4 | 10 Hz | 1 | C | 5 | 3 of 4 | — |", text)

    def test_board_section_does_not_invent_an_unknown_frame_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder)
            (results / "board").mkdir()
            item = record(2, [], False, True)
            item["rate_hz"] = None
            (results / "board" / "unknown.json").write_text(json.dumps(item), encoding="utf-8")
            mvo_stream.write_board_section(results)
            text = (results / "RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("| `unknown` | 360 MHz | p4 | unknown |", text)

    def test_board_section_notes_the_usb_serial_jtag_records(self):
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder)
            (results / "board").mkdir()
            (results / "board" / "p4_360mhz.json").write_text(json.dumps(record(2, [], False, False)), encoding="utf-8")
            mvo_stream.write_board_section(results)
            without = (results / "RESULTS.md").read_text(encoding="utf-8")
            (results / "board" / "p4_360mhz_usj.json").write_text(
                json.dumps(record(2, [], False, False)), encoding="utf-8"
            )
            mvo_stream.write_board_section(results)
            text = (results / "RESULTS.md").read_text(encoding="utf-8")
        self.assertNotIn("USB-Serial-JTAG port, not its USB-UART bridge", without)
        self.assertEqual(
            text.count("Records named `*_usj*` were streamed over the ESP32-P4's own USB-Serial-JTAG port"), 1
        )

    def test_board_section_preserves_other_sections_after_it(self):
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder)
            (results / "board").mkdir()
            (results / "RESULTS.md").write_text(
                "# Report\n\n## ESP32 boards\n\nOld table.\n\n## Personal notes\n\nKeep this.\n", encoding="utf-8"
            )
            mvo_stream.write_board_section(results)
            text = (results / "RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("## Personal notes\n\nKeep this.", text)

    def test_new_board_log_cannot_overwrite_a_concurrent_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = root / "vo.json"
            expected.write_text("{}", encoding="utf-8")
            log = root / "board" / "shared_serial.log"

            def concurrent_stream(*args):
                log.parent.mkdir()
                log.write_text("other session evidence", encoding="utf-8")
                return record(2, [], False, True), ["stream log"]

            argv = [
                "mvo_stream",
                "--port",
                "unused",
                "--frames",
                str(root / "frames.bin"),
                "--expected",
                str(expected),
                "--label",
                "shared",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(mvo_stream, "stream", side_effect=concurrent_stream),
            ):
                with self.assertRaises(FileExistsError):
                    mvo_stream.main()
            self.assertEqual(log.read_text(encoding="utf-8"), "other session evidence")

    def test_cloud_output_cannot_collide_with_new_board_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = root / "vo.json"
            expected.write_text("{}", encoding="utf-8")
            argv = [
                "mvo_stream",
                "--port",
                "unused",
                "--frames",
                str(root / "frames.bin"),
                "--expected",
                str(expected),
                "--expected-cloud",
                str(root / "cloud.json"),
                "--cloud-output",
                str(root / "board" / "same.json"),
                "--label",
                "same",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(mvo_stream, "stream") as replay:
                with self.assertRaisesRegex(ValueError, "distinct"):
                    mvo_stream.main()
                replay.assert_not_called()


class FakeBoard:
    """Serial protocol without a port; loss happens on frame 2, which has no mask."""

    def __init__(self, ready_policy=1, bad_reply=None, include_continuity=True, config="p4", max_features=300):
        self.ready_policy, self.bad_reply, self.include_continuity = ready_policy, bad_reply, include_continuity
        self.config, self.max_features = config, max_features
        self.lines, self.sent, self.pending = [], [], []
        self.frame_index = 0
        self.closed = False

    def close(self):
        self.closed = True

    def send(self, data):
        self.sent.append(data)
        command = data.split(b"\n", 1)[0].decode("ascii")
        if command.startswith("selftest "):
            self.pending.append("selftest template_mismatches=0 sums_mismatches=0 block_mismatches=0")
        elif command.startswith("start "):
            self.pending.append("started state_bytes=10 cores=2 simd=1")
        elif command.startswith("frame "):
            index = int(command.split()[1])
            self.frame_index = index
            if index == 0:
                self.pending.append("first index=0 tracks=3 detect_us=10")
            else:
                kind = (1, 0, 2)[index - 1]
                continuity = int(index < 2)
                line = (
                    f"step index={index} kind={kind} tracks=3 inliers=2 scale_points=0 hash={100 + index} "
                    "detect_us=10 track_us=20 motion_us=30 total_us=60 observed=0 mapped=0 map_us=5"
                )
                self.pending.append(self.add_continuity(line, continuity, f"step {index}"))
        elif command == "end":
            continuity = int(self.frame_index < 2)
            self.pending.append(self.add_continuity("map hash=451 observations=0", continuity, "map"))
            self.pending.append(self.add_continuity("end heap_internal_free=1", continuity, "end"))
        else:
            raise AssertionError(command)

    def add_continuity(self, line, value, where):
        if not self.include_continuity or self.bad_reply == (where, "missing"):
            return line
        if self.bad_reply == (where, "wrong"):
            value = 1 - value
        return f"{line} map_continuous={value}"

    def reply(self, prefixes, timeout):
        if not self.lines:
            built = "" if self.config is None else f"config={self.config} max_features={self.max_features} "
            line = (
                f"ready cpu_mhz=360 {built}core_sha256=core "
                "header_sha256=header map_sha256=map map_header_sha256=mapheader"
            )
            if self.ready_policy is not None:
                line += f" map_policy={self.ready_policy}"
        else:
            line = self.pending.pop(0)
        self.lines.append(line)
        if not line.startswith(prefixes):
            raise AssertionError(f"Unexpected {line!r} for {prefixes!r}")
        return line


class MapProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.frames = self.folder / "frames.bin"
        self.masks = self.folder / "masks.bin"
        self.expected = self.folder / "vo.json"
        self.expected_map = self.folder / "map.json"
        self.frames.write_bytes(bytes(mvo_stream.FRAME_BYTES * 4))
        self.masks.write_bytes(b"".join(bytes([flag]) + bytes(mvo_stream.MASK_RECORD - 1) for flag in (1, 1, 0, 1)))
        sources = dict(core_sha256="core", header_sha256="header", map_sha256="map", map_header_sha256="mapheader")
        build = dict(config="p4", max_features=300)
        self.vo = dict(frames=4, step_hashes=[101, 102, 103], camera_bits=[1, 2, 3, 4], seed=1, **build, **sources)
        self.map = dict(
            **build,
            step_hashes=self.vo["step_hashes"],
            map_policy=1,
            map_continuity=[1, 0, 0],
            map_continuous=0,
            map_hash=451,
            map_observations=0,
            **sources,
        )
        self.write_expectations()

    def write_expectations(self):
        self.expected.write_text(json.dumps(self.vo), encoding="utf-8")
        self.expected_map.write_text(json.dumps(self.map), encoding="utf-8")

    def replay(self, board, limit=None, masks=True):
        with mock.patch.object(mvo_stream, "Board", return_value=board), mock.patch("builtins.print"):
            return mvo_stream.stream(
                "no-port",
                self.frames,
                self.expected,
                limit=limit,
                masks_path=self.masks if masks else None,
                expected_map_path=self.expected_map,
            )

    def test_start_line_carries_the_lens(self):
        board = FakeBoard()
        self.replay(board)
        self.assertIn(b"start 1 2 3 4 1 2 1\n", board.sent)  # a pinhole: the four intrinsics alone
        self.vo["camera_bits"] = [1, 2, 3, 4, 6, 7, 8, 9, 10]
        self.write_expectations()
        board = FakeBoard()
        self.replay(board)
        self.assertIn(b"start 1 2 3 4 1 2 1 6 7 8 9 10\n", board.sent)
        for bad in ([1, 2, 3, 4, 6], [1, 2, 3, 4, -1, 0, 0, 0, 0], [1, 2, 3, 4, 1.5, 0, 0, 0, 0]):
            with self.subTest(camera_bits=bad):
                self.vo["camera_bits"] = bad
                self.write_expectations()
                with self.assertRaisesRegex(RuntimeError, "camera_bits"):
                    self.replay(FakeBoard())

    def test_board_of_another_build_is_refused(self):
        board = FakeBoard(config="other", max_features=250)
        with self.assertRaisesRegex(RuntimeError, "runs the .* build"):
            self.replay(board)
        self.assertFalse(any(item.startswith(b"frame ") for item in board.sent))
        record, _ = self.replay(FakeBoard())  # the matching board
        self.assertTrue(record["all_match"])
        self.assertEqual(record["config"], "p4")

    def test_board_reporting_another_feature_count_or_none_is_refused(self):
        for board in (FakeBoard(max_features=250), FakeBoard(config=None)):
            with self.subTest(config=board.config, max_features=board.max_features):
                with self.assertRaisesRegex(RuntimeError, "runs the .* build"):
                    self.replay(board)
                self.assertFalse(any(item.startswith(b"frame ") for item in board.sent))

    def test_map_expectation_of_another_build_is_refused(self):
        self.map.update(config="other", max_features=250)
        self.write_expectations()
        board = FakeBoard()
        with self.assertRaisesRegex(ValueError, "map expectation: .*made with the 'other' build"):
            self.replay(board)
        self.assertEqual(board.sent, [])  # refused before the board is opened

    def test_expectations_must_name_this_build(self):
        for bad in (
            {"config": ["p4"]},
            {"config": None},
            {"config": "P4"},
            {"max_features": 300},
            {},
            {"config": "p4", "max_features": 250},
            {"config": "other"},
        ):
            with self.subTest(expected=bad), self.assertRaisesRegex(ValueError, "make it again with this tree"):
                mvo_stream.expectation_config(bad)
        self.assertEqual(mvo_stream.expectation_config({"config": "p4", "max_features": 300}), "p4")

    def test_voxel_filing_is_capped_at_the_feature_count(self):
        voxels = dict(
            step_hashes=self.vo["step_hashes"],
            frames=self.vo["frames"],
            voxel_hash=1,
            voxel_progress=[{"count": 1, "filed": 300}] * 3,
            voxel_sha256="a" * 64,
            voxel_header_sha256="b" * 64,
        )
        mvo_stream.validate_expected_voxels(voxels, self.vo)
        voxels["voxel_progress"] = [{"count": 1, "filed": 301}] * 3
        with self.assertRaisesRegex(ValueError, "voxel count and filed"):
            mvo_stream.validate_expected_voxels(voxels, self.vo)  # no step files more than MVO_MAX_FEATURES

    def test_build_follows_the_binding(self):
        from uwslam import mvo

        self.assertEqual((mvo_stream.BUILD, mvo_stream.MAX_FEATURES), (mvo.CONFIG, mvo.MAX_FEATURES))

    def test_full_replay_checks_states_independent_of_equal_map_hash(self):
        board = FakeBoard()
        record, _ = self.replay(board)
        self.assertTrue(record["all_match"])
        self.assertTrue(record["map"]["match"])
        self.assertEqual(record["map"]["observations_expected"], 0)
        self.assertEqual([step["map_continuous"] for step in record["per_step"]], [1, 0, 0])
        self.assertEqual(record["map"]["map_continuous"], 0)
        self.assertEqual(record["memory"]["map_continuous"], 0)
        frames = [item.split(b"\n", 1)[0].split() for item in board.sent if item.startswith(b"frame ")]
        self.assertEqual([len(frame) for frame in frames], [4, 4, 3, 4])
        self.assertTrue(board.closed)

    def test_partial_replay_uses_prefix_and_its_own_final_state(self):
        self.map["map_observations"] = 100  # full-run count is not a prefix comparison
        self.write_expectations()
        for limit, state in ((1, 1), (2, 1), (3, 0)):
            with self.subTest(limit=limit):
                record, _ = self.replay(FakeBoard(), limit=limit)
                self.assertEqual(record["steps"], limit - 1)
                self.assertEqual(record["map"]["map_continuous"], state)
                self.assertEqual(record["map"]["continuous_expected"], state)
                self.assertIsNone(record["map"]["match"])
                self.assertIsNone(record["map"]["observations_expected"])
                self.assertEqual(record["memory"]["map_continuous"], state)
                if limit == 1:
                    self.assertEqual(record["timing"], {})
                    (self.folder / "board").mkdir(exist_ok=True)
                    (self.folder / "board" / "one.json").write_text(json.dumps(record), encoding="utf-8")
                    mvo_stream.write_board_section(self.folder)
                    self.assertIn("0 of 0", (self.folder / "RESULTS.md").read_text(encoding="utf-8"))

    def test_old_or_invalid_expectation_fails_before_opening_board(self):
        for field, value in (
            ("map_policy", None),
            ("map_policy", 2),
            ("map_policy", True),
            ("map_continuity", None),
            ("map_continuity", [1, 0]),
            ("map_continuity", [1, 0, 1]),
            ("map_continuity", [1, False, 0]),
            ("map_continuous", None),
            ("map_continuous", 1),
        ):
            with self.subTest(field=field, value=value):
                original = self.map[field]
                if value is None:
                    del self.map[field]
                else:
                    self.map[field] = value
                self.write_expectations()
                with mock.patch.object(mvo_stream, "Board") as constructor:
                    with self.assertRaisesRegex(ValueError, "map_(policy|continuity|continuous)"):
                        mvo_stream.stream(
                            "no-port",
                            self.frames,
                            self.expected,
                            masks_path=self.masks,
                            expected_map_path=self.expected_map,
                        )
                    constructor.assert_not_called()
                self.map[field] = original
        self.write_expectations()

    def test_old_firmware_rejected_before_any_frame(self):
        for policy in (None, 0, "bad"):
            with self.subTest(policy=policy):
                board = FakeBoard(ready_policy=policy)
                with self.assertRaisesRegex(RuntimeError, "map_policy=1"):
                    self.replay(board)
                self.assertEqual(board.sent, [])
                self.assertTrue(board.closed)

    def test_missing_or_wrong_continuity_never_passes_on_map_hash(self):
        for where in ("step 1", "step 2", "step 3", "map", "end"):
            for failure in ("missing", "wrong"):
                with self.subTest(where=where, failure=failure):
                    board = FakeBoard(bad_reply=(where, failure))
                    with self.assertRaisesRegex(RuntimeError, "map_continuous"):
                        self.replay(board)
                    self.assertTrue(board.closed)

    def test_pure_odometry_accepts_old_protocol(self):
        record, _ = self.replay(FakeBoard(ready_policy=None, include_continuity=False), masks=False)
        self.assertTrue(record["all_match"])
        self.assertNotIn("map", record)

    @unittest.skipUnless(shutil.which("gcc"), "gcc is needed for the firmware host check")
    def test_firmware_freezes_every_map_byte_and_resets_only_on_start(self):
        project = Path(__file__).resolve().parents[1]
        include = self.folder / "include"
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
        program = self.folder / "firmware_check.exe"
        env = dict(os.environ, TMP=str(self.folder), TEMP=str(self.folder))
        subprocess.run(
            [
                "gcc",
                "-std=c99",
                "-O1",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wno-unused-function",
                "-ffunction-sections",
                "-fdata-sections",
                "-Wl,--gc-sections",
                *(["-static"] if sys.platform == "win32" else []),
                f"-I{include}",
                f"-I{project / 'core'}",
                str(project / "tests/c/mvo_map_continuity_check.c"),
                str(project / "core/mono_cloud.c"),
                str(project / "core/voxel_map.c"),
                str(project / "core/obstacle_dir.c"),
                str(project / "core/path_plan.c"),
                "-o",
                str(program),
            ],
            check=True,
            env=env,
        )
        output = subprocess.run([str(program)], check=True, env=env, capture_output=True, text=True).stdout.splitlines()
        steps = [mvo_stream.fields(line)["map_continuous"] for line in output if line.startswith("step ")]
        maps = [mvo_stream.fields(line)["map_continuous"] for line in output if line.startswith("map ")]
        ends = [mvo_stream.fields(line)["map_continuous"] for line in output if line.startswith("end ")]
        self.assertEqual(steps, [1, 1, 0, 0, 0, 0, 0, 1])
        self.assertEqual(maps, [0, 0, 1])
        self.assertEqual(ends, maps)


class FakeCloudBoard(FakeBoard):
    """A point appears on frame 1, then frame 2 loses tracking."""

    def __init__(self, cloud_policy=1, bad_step=False, bad_point=False):
        super().__init__()
        self.cloud_policy, self.bad_step, self.bad_point = cloud_policy, bad_step, bad_point

    def cloud_bytes(self):
        raw = bytearray(512 * 24 + 12)
        count = int(self.frame_index > 0)
        continuous = int(self.frame_index < 2)
        if count:
            struct.pack_into("<6I", raw, 0, 0x3F800000, 0x40000000, 0x40400000, 17, 0x3F000000, 1)
        struct.pack_into("<IIi", raw, 512 * 24, count, 0, continuous)
        return bytes(raw)

    def send(self, data):
        command = data.split(b"\n", 1)[0].decode("ascii")
        if command == "cloud":
            self.sent.append(data)
            count, continuous = int(self.frame_index > 0), int(self.frame_index < 2)
            self.pending.append(
                f"cloud count={count} dropped=0 continuous={continuous} hash={mvo_stream.fnv(self.cloud_bytes())}"
            )
            if count:
                x = 0x3F800001 if self.bad_point else 0x3F800000
                self.pending.append(
                    f"point index=0 id=17 x_bits={x} y_bits=1073741824 z_bits=1077936128 "
                    "spread_bits=1056964608 observations=1"
                )
            self.pending.append(f"cloud_end count={count}")
            return
        super().send(data)
        if command.startswith("frame ") and self.frame_index > 0:
            index = self.frame_index
            count, accepted, continuous = 1, int(index == 1), int(index < 2)
            if self.bad_step and index == 2:
                continuous = 1
            line = self.pending[-1].replace("kind=1 ", "kind=2 ")
            self.pending[-1] = (
                f"{line} cloud_count={count} cloud_accepted={accepted} cloud_dropped=0 "
                f"cloud_continuous={continuous} cloud_us=10"
            )
        elif command == "end":
            count, continuous = int(self.frame_index > 0), int(self.frame_index < 2)
            self.pending[-1] += (
                f" cloud_count={count} cloud_dropped=0 cloud_continuous={continuous} "
                f"cloud_hash={mvo_stream.fnv(self.cloud_bytes())}"
            )

    def reply(self, prefixes, timeout):
        line = super().reply(prefixes, timeout)
        if line.startswith("ready") and self.cloud_policy is not None:
            line += (
                f" cloud_policy={self.cloud_policy} cloud_sha256={'a' * 64} cloud_header_sha256={'b' * 64}"
                " cloud_capacity=512 cloud_bytes=12300"
            )
            self.lines[-1] = line
        return line


class CloudProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        self.frames, self.expected, self.expected_cloud = (
            folder / "frames.bin",
            folder / "vo.json",
            folder / "cloud.json",
        )
        self.frames.write_bytes(bytes(mvo_stream.FRAME_BYTES * 4))
        vo = dict(
            frames=4,
            every=1,
            step_hashes=[101, 102, 103],
            camera_bits=[1, 2, 3, 4],
            seed=1,
            core_sha256="core",
            header_sha256="header",
            config="p4",
            max_features=300,
        )
        cloud = dict(
            config="p4",
            max_features=300,
            frames=4,
            step_hashes=vo["step_hashes"],
            cloud_policy=1,
            cloud_sha256="a" * 64,
            cloud_header_sha256="b" * 64,
            source_sha256="c" * 64,
            cloud_progress=[
                dict(count=1, accepted=1, dropped=0, continuous=1),
                dict(count=1, accepted=0, dropped=0, continuous=0),
                dict(count=1, accepted=0, dropped=0, continuous=0),
            ],
            cloud_count=1,
            cloud_dropped=0,
            cloud_continuous=0,
            cloud_hash=mvo_stream.fnv(FakeCloudBoard().cloud_bytes()),
        )
        cloud["frames_sha256"] = hashlib.sha256(bytes(mvo_stream.FRAME_BYTES * 4)).hexdigest()
        cloud["frame_sha256"] = [hashlib.sha256(bytes(mvo_stream.FRAME_BYTES)).hexdigest()] * 4
        # Hash of the final state has one point and interrupted continuity.
        board = FakeCloudBoard()
        board.frame_index = 3
        cloud["cloud_hash"] = mvo_stream.fnv(board.cloud_bytes())
        self.expected.write_text(json.dumps(vo), encoding="utf-8")
        self.expected_cloud.write_text(json.dumps(cloud), encoding="utf-8")

    def replay(self, board, limit=None):
        with mock.patch.object(mvo_stream, "Board", return_value=board), mock.patch("builtins.print"):
            return mvo_stream.stream(
                "no-port", self.frames, self.expected, limit=limit, expected_cloud_path=self.expected_cloud
            )

    def test_cloud_replay_checks_and_exports_real_point_bits(self):
        board = FakeCloudBoard()
        record, _ = self.replay(board)
        self.assertTrue(record["all_match"])
        self.assertTrue(record["cloud"]["match"])
        self.assertEqual(record["cloud"]["points"][0]["xyz"], [1.0, 2.0, 3.0])
        self.assertEqual(record["cloud"]["points"][0]["id"], 17)
        self.assertEqual(record["cloud"]["continuous"], 0)
        self.assertAlmostEqual(record["timing"]["vo_cloud_map_ms"]["mean"], 0.075)
        commands = [data.split(b"\n", 1)[0].split()[0] for data in board.sent]
        self.assertLess(commands.index(b"start"), commands.index(b"selftest"))
        self.assertTrue(board.closed)
        output = self.frames.parent / "board_cloud.ply"
        mvo_stream.write_cloud_ply(output, record["cloud"])
        ply = output.read_text(encoding="ascii")
        self.assertIn("1 2 3 17 0.5 1\n", ply)
        self.assertIn("comment continuous 0 evicted 0", ply)
        self.assertIn("comment pc_match matched", ply)
        self.assertIn("comment source_sha256 " + "c" * 64, ply)
        with self.assertRaises(FileExistsError):
            mvo_stream.write_cloud_ply(output, record["cloud"])

    def test_cloud_replay_refuses_invalid_state_and_corrupted_export(self):
        for board in (FakeCloudBoard(cloud_policy=None), FakeCloudBoard(bad_step=True), FakeCloudBoard(bad_point=True)):
            with self.subTest(board=board.__dict__):
                with self.assertRaises(RuntimeError):
                    self.replay(board)
                self.assertTrue(board.closed)

    def test_cloud_replay_partial_run_checks_prefix_without_claiming_full_match(self):
        for limit, count, continuous in ((1, 0, 1), (2, 1, 1), (3, 1, 0)):
            with self.subTest(limit=limit):
                record, _ = self.replay(FakeCloudBoard(), limit)
                self.assertEqual(record["cloud"]["count"], count)
                self.assertEqual(record["cloud"]["continuous"], continuous)
                self.assertIsNone(record["cloud"]["match"])
                output = self.frames.parent / f"partial_{limit}.ply"
                mvo_stream.write_cloud_ply(output, record["cloud"])
                self.assertIn("comment pc_match not_compared_partial", output.read_text(encoding="ascii"))

    def test_cloud_replay_refuses_changed_frame_file_before_opening_port(self):
        frame_data = bytearray(self.frames.read_bytes())
        frame_data[0] = 1
        self.frames.write_bytes(frame_data)
        with mock.patch.object(mvo_stream, "Board") as board:
            with self.assertRaisesRegex(ValueError, "frame.*SHA"):
                mvo_stream.stream("no-port", self.frames, self.expected, expected_cloud_path=self.expected_cloud)
            board.assert_not_called()

    def test_cloud_replay_checks_each_frame_again_after_file_hash(self):
        board = FakeCloudBoard()
        original_hash = mvo_stream.sha256_file

        def hash_then_change(path):
            digest = original_hash(path)
            with path.open("r+b") as target:
                target.write(b"\x01")
            return digest

        with mock.patch.object(mvo_stream, "sha256_file", side_effect=hash_then_change):
            with self.assertRaisesRegex(RuntimeError, "Frame 0 SHA"):
                self.replay(board)
        self.assertTrue(board.closed)
        self.assertFalse(any(data.startswith(b"frame ") for data in board.sent))

    def test_cloud_replay_reads_frames_in_bounded_chunks(self):
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read")):
            record, _ = self.replay(FakeCloudBoard(), limit=2)
        self.assertEqual(record["frames"], 2)


if __name__ == "__main__":
    unittest.main()


class GyroFileTests(unittest.TestCase):
    """The gyro readings file and expectation the host checks before it opens a board."""

    def write(self, folder: Path, counts: list[int]) -> Path:
        path = folder / "gyro.bin"
        path.write_bytes(b"".join(struct.pack("<I", n) + bytes(16 * n) for n in counts))
        return path

    def test_readings_file_splits_into_each_steps_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            payloads = mvo_stream.read_gyro_file(self.write(Path(temp), [2, 0, 41]), 3)
        self.assertEqual([len(p) for p in payloads], [32, 0, 41 * 16])

    def test_readings_file_must_fit_the_run_and_the_firmware(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with self.assertRaisesRegex(ValueError, "for 2 steps"):
                mvo_stream.read_gyro_file(self.write(folder, [1, 1]), 3)
            with self.assertRaisesRegex(ValueError, "the firmware takes 256"):
                mvo_stream.read_gyro_file(self.write(folder, [257]), 1)
            truncated = self.write(folder, [3])
            truncated.write_bytes(truncated.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "inside step 0"):
                mvo_stream.read_gyro_file(truncated, 1)

    def test_gyro_expectation_is_checked(self):
        expected = {"frames": 3}
        good = {"mount_bits": [0] * 9, "readings_sha256": "0" * 64, "used": [0, 1], "final_bias_bits": [0, 0, 0]}
        mvo_stream.validate_expected_gyro(good, expected)
        for key, value, message in (
            ("mount_bits", [0] * 8, "nine mount_bits"),
            ("readings_sha256", "x", "SHA-256"),
            ("used", [0, 4], "each step used"),
            ("used", [0], "each step used"),
            ("final_bias_bits", [0.5, 0, 0], "final bias"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                mvo_stream.validate_expected_gyro({**good, key: value}, expected)
