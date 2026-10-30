"""The live camera host (uwslam.live_feed) without a board: prepare's JPEG frames from a video, the frames file, the
board's protocol in check mode (each JPEG's step, then its 320x240 frame read back and verified), what a run's record
counts and times, and check's refusal of a live run or of a board built from other sources.

prepare and check need OpenCV; skipped without it.
"""

import hashlib
import json
import struct
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest import mock

from uwslam import live_feed
from uwslam.mvo_stream import fnv

HAS_CV2 = find_spec("cv2") is not None
END = (
    "map hash=1",
    "voxels count=2 hash=3",
    "dirs choice=-1 hash=4",
    "plan action=1 hash=5",
    "gyro bias_steps=0",
    "imu up_steps=0",
    "live received=3 dropped=0 psram_free=100",
    "end heap_internal_free=7",
)


class FakeLiveBoard:
    """The live build's protocol without a port: each JPEG is stepped at once, its frame filled with its index, and
    `grey` sends that frame back."""

    def __init__(self, live=1, damage_grey=False, selftest_mismatches=0):
        self.live, self.damage_grey, self.selftest_mismatches = live, damage_grey, selftest_mismatches
        self.lines, self.sent, self.pending = [], [], []
        self.port, self.waiting, self.frame = self, b"", b""  # read() stands in for the serial port's

    def close(self):
        pass

    def send(self, data):
        self.sent.append(data)
        command = data.lstrip(b"\n").split(b"\n", 1)[0].decode("ascii")
        if command.startswith("start "):
            self.pending.append("started state_bytes=10 cores=2 simd=1")
        elif command.startswith("selftest "):
            mismatches = self.selftest_mismatches
            self.pending.append(f"selftest template_mismatches={mismatches} sums_mismatches=0 block_mismatches=0")
        elif command.startswith("jpeg "):
            index = int(command.split()[1])
            self.frame = bytes([index]) * live_feed.FRAME_BYTES
            timing = " ".join(f"{key}={1000 * (k + 1)}" for k, key in enumerate(live_feed.WORK))
            kind = "first" if index == 0 else "step"
            self.pending.append(f"{kind} index={index} hash={100 + index} {timing} grey_hash={fnv(self.frame)}")
        elif command == "grey":
            pixels = bytearray(self.frame)
            pixels[0] ^= int(self.damage_grey)
            self.pending.append(f"grey bytes={live_feed.FRAME_BYTES} hash={fnv(self.frame)}")
            self.waiting += bytes(pixels)
        elif command == "end":
            self.pending.extend(END)
        else:
            raise AssertionError(command)

    def read(self, size):
        out, self.waiting = self.waiting[:size], self.waiting[size:]
        return out

    def reply(self, prefixes, timeout):
        if not self.lines:
            line = "ready cpu_mhz=360 config=p4 max_features=300 core_sha256=core header_sha256=header"
            line += f" live={self.live} prep_sha256=prep prep_header_sha256=prepheader" if self.live else ""
        else:
            line = self.pending.pop(0)
        self.lines.append(line)
        if not line.startswith(prefixes):
            raise AssertionError(f"Unexpected {line!r} for {prefixes!r}")
        return line


def write_feed(folder: Path, count: int) -> list[bytes]:
    """A prepared folder as prepare leaves it, with stand-in frames (the fake board decodes nothing)."""
    jpegs = [b"\xff\xd8" + bytes([k]) * (100 + k) for k in range(count)]
    data = b"".join(struct.pack("<I", len(j)) + j for j in jpegs)
    (folder / "frames.jpg.bin").write_bytes(data)
    feed = {
        "rate_hz": 5.0,
        "video_indices": [6 * k for k in range(count)],
        "source_size": [960, 544],
        "jpeg_quality": 80,
        "frames": count,
        "frames_file": "frames.jpg.bin",
        "frames_sha256": hashlib.sha256(data).hexdigest(),
        "jpeg_fnv": [fnv(j) for j in jpegs],
        "jpeg_bytes": {"mean": 101.0},
        "camera_bits": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "seed": 12345,
    }
    (folder / "feed.json").write_text(json.dumps(feed), encoding="utf-8")
    return jpegs


class LiveFeedHostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_the_frames_file_round_trips_and_damage_is_refused(self):
        jpegs = write_feed(self.folder, 3)
        path = self.folder / "frames.jpg.bin"
        self.assertEqual(live_feed.read_jpegs(path, 3), jpegs)
        with self.assertRaises(ValueError):
            live_feed.read_jpegs(path, 4)  # feed.json names another count
        data = path.read_bytes()
        for cut in (len(data) - 1, len(data) - len(jpegs[-1]) - 2):  # inside the last frame, inside its length
            path.write_bytes(data[:cut])
            with self.assertRaises(ValueError):
                live_feed.read_jpegs(path, 3)

    def test_a_frame_message_is_its_line_then_its_bytes(self):
        self.assertEqual(live_feed.frame_message(4, b"abc", 99), b"\njpeg 4 3 99\nabc")

    def test_answers_are_taken_by_frame_index(self):
        lines = [
            (0.1, "first index=0 tracks=3"),
            (0.2, "ready mono_vo"),
            (0.3, "dropped index=1"),
            (0.4, "step index=2 hash=5"),
            (0.5, "error jpeg 3 hash mismatch"),
            (0.6, "error bad jpeg line"),
        ]
        answers = live_feed.answered(lines)
        self.assertEqual(sorted(answers), [0, 1, 2, 3])
        self.assertEqual(answers[1][1], "dropped index=1")

    def test_a_check_run_steps_each_frame_and_reads_its_frame_back(self):
        write_feed(self.folder, 3)
        board = FakeLiveBoard()
        with mock.patch.object(live_feed, "Board", lambda port, baud: board):
            record = live_feed.stream(self.folder, "COMX", "run", check=True)
        self.assertEqual(record["results"], {"first": 1, "step": 2, "dropped": 0, "error": 0, "none": 0})
        self.assertEqual([f["hash"] for f in record["per_frame"][1:]], [101, 102])
        self.assertEqual(record["timing"]["work_ms"]["max"], sum(range(1, len(live_feed.WORK) + 1)))
        self.assertEqual(record["prep_sha256"], "prep")
        self.assertEqual(record["board_received"], 3)
        grey = (self.folder / "board" / "run_grey.bin").read_bytes()
        self.assertEqual(grey, b"".join(bytes([k]) * live_feed.FRAME_BYTES for k in range(3)))
        start = board.sent[0].decode()
        self.assertEqual(start, "start 1 2 3 4 12345 2 1 5 6 7 8 9\n")
        with self.assertRaises(FileExistsError), mock.patch.object(live_feed, "Board", lambda port, baud: board):
            live_feed.stream(self.folder, "COMX", "run", check=True)  # the label's files exist

    def test_a_damaged_frame_read_back_is_refused(self):
        write_feed(self.folder, 2)
        with mock.patch.object(live_feed, "Board", lambda port, baud: FakeLiveBoard(damage_grey=True)):
            with self.assertRaises(RuntimeError):
                live_feed.stream(self.folder, "COMX", "run", check=True)

    def test_the_session_needs_the_live_build_and_matching_kernels(self):
        feed = {"camera_bits": [1, 2, 3, 4], "seed": 7}
        with self.assertRaises(RuntimeError):
            live_feed.open_session(FakeLiveBoard(live=None), feed, 2)
        with self.assertRaises(RuntimeError):
            live_feed.open_session(FakeLiveBoard(selftest_mismatches=1), feed, 2)
        self.assertEqual(live_feed.open_session(FakeLiveBoard(), feed, 1)["live"], 1)

    def test_a_record_counts_drops_and_times_answers(self):
        feed = {"frames": 3, "video_indices": [0, 6, 12], "rate_hz": 5.0, "source_size": [960, 544]}
        feed.update(jpeg_quality=80, jpeg_bytes={})
        timing = " ".join(f"{key}=2000" for key in live_feed.TIMINGS)
        lines = [(0.5, "first index=0 tracks=3"), (0.6, "dropped index=1"), (0.9, f"step index=2 hash=9 {timing}")]
        sent = [{"index": k, "late_s": 0.001 * k, "written_s": 0.2 * k, "bytes": 10} for k in range(3)]
        end = {key.split()[0]: live_feed.fields(key) for key in END}
        record = live_feed.summarise_run(feed, {"live": 1}, sent, lines, end, False, 1.0)
        self.assertEqual(record["results"], {"first": 1, "step": 1, "dropped": 1, "error": 0, "none": 0})
        self.assertAlmostEqual(record["per_frame"][2]["answer_s"], 0.5)
        self.assertAlmostEqual(record["timing"]["work_ms"]["mean"], 2.0 * len(live_feed.WORK))
        self.assertAlmostEqual(record["answer_ms"]["max"], 500.0)
        self.assertAlmostEqual(record["late_ms"]["max"], 2.0)


@unittest.skipUnless(HAS_CV2, "needs OpenCV")
class LiveFeedPcTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_prepare_takes_every_board_frame_as_jpeg(self):
        import cv2
        import numpy as np

        video = self.folder / "clip.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (400, 300))
        rng = np.random.default_rng(1)
        for _ in range(40):
            writer.write(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8))
        writer.release()
        reconstruction = self.folder / "reconstruction"
        reconstruction.mkdir()
        (reconstruction / "images.txt").write_text("0 0 0\n0.3 0 0\n", encoding="utf-8")  # 0.9 s of video
        output = self.folder / "feed"
        summary = live_feed.prepare(video, reconstruction, output, 5.0, 70)
        feed = json.loads((output / "feed.json").read_text(encoding="utf-8"))
        self.assertEqual(feed["video_indices"], [0, 6, 12, 18, 24])
        self.assertEqual((summary["frames"], feed["source_size"], feed["jpeg_quality"]), (5, [400, 300], 70))
        jpegs = live_feed.read_jpegs(output / feed["frames_file"], feed["frames"])
        self.assertEqual(feed["jpeg_fnv"], [fnv(j) for j in jpegs])
        for jpeg in jpegs:
            self.assertEqual(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape, (300, 400, 3))
        self.assertNotIn("core_sha256", feed)  # the frames do not depend on the C sources
        with self.assertRaises(FileExistsError):
            live_feed.prepare(video, reconstruction, output, 5.0, 70)
        with self.assertRaises(ValueError):
            live_feed.prepare(video, reconstruction, self.folder / "other", 7.0, 70)
        with self.assertRaises(ValueError):
            live_feed.prepare(video, reconstruction, self.folder / "other", 5.0, 70, "411")
        for sampling, factor in (("420", 0x22), ("422", 0x21), ("444", 0x11)):  # the Y component's in SOF0
            folder = self.folder / f"feed_{sampling}"
            live_feed.prepare(video, reconstruction, folder, 5.0, 70, sampling)
            jpeg = live_feed.read_jpegs(folder / "frames.jpg.bin", 5)[0]
            self.assertEqual(jpeg[jpeg.index(b"\xff\xc0") + 11], factor, sampling)
        same = live_feed.read_jpegs(self.folder / "feed_420" / "frames.jpg.bin", 5)
        self.assertEqual(same, jpegs)  # 4:2:0 is OpenCV's default

    def test_check_refuses_a_live_run_and_a_board_built_from_other_sources(self):
        from uwslam import mvo

        write_feed(self.folder, 2)
        (self.folder / "board").mkdir()
        sources = {**mvo.sources_sha256(), **mvo.prep_sha256()}
        record = {"mode": "check", **sources, **mvo.build_config()}
        cases = [
            ({"mode": "live"}, "live run"),
            ({"prep_sha256": "other"}, "prep_sha256"),
            ({"prep_header_sha256": None}, "prep_header_sha256"),  # a firmware that did not report it
            ({"config": "x"}, "build"),
        ]
        for change, message in cases:
            with self.subTest(change=change):
                path = self.folder / "board" / "run.json"
                path.write_text(json.dumps({**record, **change}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    live_feed.check(self.folder, "run", self.folder, self.folder / "out")

    def test_a_difference_is_counted_in_grey_levels(self):
        import numpy as np

        board = [np.array([[10, 12], [9, 9]], dtype=np.uint8)]
        pc = [np.array([[10, 11], [10, 9]], dtype=np.uint8)]
        self.assertEqual(
            live_feed.difference(board, pc), {"mean_abs": 0.5, "mean": 0.0, "max_abs": 1, "share_equal": 0.5}
        )


if __name__ == "__main__":
    unittest.main()
