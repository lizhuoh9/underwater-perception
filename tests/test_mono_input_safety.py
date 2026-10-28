"""Input and scoring boundaries of the monocular VO and mapping tools."""

import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from uwslam import mono_map, mvo


class FakeVO:
    def __init__(self, kinds=()):
        self.frame = (ctypes.c_uint8 * (mvo.WIDTH * mvo.HEIGHT))()
        self.map_updates = []
        self.kinds = iter(kinds)

    def mvo_state_size(self): return 1
    def mvo_pyramid_bytes(self): return 1
    def mm_map_size(self): return mono_map.MAP_SIZE ** 2 + 28
    def mvo_init(self, *args): pass
    def mm_init(self, *args): pass
    def mvo_frame(self, *args): return ctypes.addressof(self.frame)
    def mvo_start(self, *args): pass
    def mvo_step(self, state, out): out._obj.kind = next(self.kinds, 2)
    def mm_add(self, state, vo, points, count):
        self.map_updates.append(count)
        return 0


class InputSafetyTests(unittest.TestCase):
    def test_contiguous_copy_lives_through_memmove(self):
        frame = np.zeros((mvo.HEIGHT, mvo.WIDTH * 2), dtype=np.uint8)[:, ::2]
        self.assertFalse(frame.flags.c_contiguous)
        original_contiguous, original_memmove = np.ascontiguousarray, ctypes.memmove
        references, alive = [], []

        def contiguous(*args, **kwargs):
            result = original_contiguous(*args, **kwargs)
            references.append(weakref.ref(result))
            return result

        def copy(*args):
            alive.append(references[-1]() is not None)
            return original_memmove(*args)

        for mapping in (False, True):
            with self.subTest(mapping=mapping), patch.object(np, "ascontiguousarray", side_effect=contiguous), \
                    patch.object(ctypes, "memmove", side_effect=copy):
                if mapping:
                    mono_map.run_mapping(FakeVO(), [frame], [None], mvo.Camera())
                else:
                    mvo.run_frames(FakeVO(), [frame], mvo.Camera())
            self.assertTrue(alive[-1])

    def test_mapping_rejects_wrong_frame_and_mask_counts(self):
        with self.assertRaisesRegex(ValueError, "240x320"):
            mono_map.run_mapping(FakeVO(), [np.zeros((120, 160), dtype=np.uint8)], [None], mvo.Camera())
        with self.assertRaisesRegex(ValueError, "mask"):
            mono_map.run_mapping(FakeVO(), [np.zeros((240, 320), dtype=np.uint8)], [], mvo.Camera())

    def test_mapping_updates_its_pose_when_a_mask_is_missing(self):
        vo = FakeVO()
        frame = np.zeros((240, 320), dtype=np.uint8)
        mono_map.run_mapping(vo, [frame, frame], [None, None], mvo.Camera())
        self.assertEqual(vo.map_updates, [0])

    def test_lost_without_a_mask_freezes_all_later_map_updates(self):
        vo = FakeVO(kinds=(2, 0, 2))
        frame = np.zeros((240, 320), dtype=np.uint8)
        result = mono_map.run_mapping(vo, [frame] * 4, [None] * 4, mvo.Camera())
        self.assertEqual(vo.map_updates, [0])
        self.assertEqual(result["map_continuity"], [1, 0, 0])
        self.assertEqual(result["first_lost_frame"], 2)
        self.assertFalse(result["map_continuous"])

    def test_lost_before_map_initialization_still_marks_it_interrupted(self):
        vo = FakeVO(kinds=(0, 2))
        frame = np.zeros((240, 320), dtype=np.uint8)
        result = mono_map.run_mapping(vo, [frame] * 3, [None] * 3, mvo.Camera())
        self.assertEqual(vo.map_updates, [])
        self.assertEqual(result["map_continuity"], [0, 0])
        self.assertEqual(result["cell"], 0.0)
        self.assertFalse(result["map_continuous"])

    def test_fps_is_checked(self):
        mvo.require_30fps(30.0)
        with self.assertRaisesRegex(ValueError, "30"):
            mvo.require_30fps(60.0)

    def test_board_inputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            existing = root / "board-input.bin"
            existing.write_bytes(b"existing evidence")
            with self.assertRaisesRegex(FileExistsError, "board frames"):
                mvo.compare(root / "video", root, root / "output", existing, 3.0)
            with self.assertRaisesRegex(FileExistsError, "board masks"):
                mono_map.run(root / "video", root, root / "output", 3.0, existing)
            self.assertEqual(existing.read_bytes(), b"existing evidence")

    def test_empty_cloud_is_not_perfect_and_large_cloud_is_chunked(self):
        query = np.array([[1.0, 2.0, 3.0]])
        with self.assertRaisesRegex(ValueError, "cloud"):
            mono_map.nearest_distances(query, np.empty((0, 3)))
        cloud = np.zeros((262144, 3))
        cloud[-1] = query[0]
        np.testing.assert_array_equal(mono_map.nearest_distances(query, cloud), [0.0])

    def test_no_observations_have_no_distance_summary(self):
        result = {"steps": [], "observations": [], "odds": np.zeros((128, 128), dtype=np.int8),
                  "cell": 0.0, "origin": (0, 0), "hits": 0, "updates": 0, "map_hash": 0,
                  "slice_y": 0.0, "height_resets": 0}
        result = mono_map.evaluate_map(result, [0], np.array([np.eye(3)]), np.zeros((1, 3)), np.ones((1, 3)))
        self.assertIsNone(result["distance_to_cloud_m"]["median"])
        self.assertIsNone(result["ate_rmse_m"])
        self.assertIsNone(mono_map.distance_spread([None, None]))
        self.assertEqual(mono_map.distance_spread([None, 1.0])["median"], 1.0)
        result.update(rate_hz=10.0, over_seeds={"distance_median_m": None, "distance_valid_seeds": 0,
                                                "ate_rmse_m": None, "ate_valid_seeds": 0})
        with tempfile.TemporaryDirectory() as folder:
            mono_map.write_report(Path(folder), {"runs": [result]})
            report = (Path(folder) / "RESULTS.md").read_text(encoding="utf-8")
            self.assertEqual(report.count("n/a (0/16 seeds)"), 2)

    def test_interrupted_map_alignment_uses_only_the_continuous_prefix(self):
        def step(centre, kind):
            value = mvo.StepC()
            value.kind = kind
            value.world_centre[:] = centre
            return value

        steps = [step((1, 0, 0), 2), step((1, 1, 0), 2),
                 step((99, 99, 99), 0), step((200, 200, 200), 2)]
        point = np.array([[2.0, 2.0, 2.0]])
        result = {"steps": steps, "observations": [point, point, np.empty((0, 3)), np.empty((0, 3))],
                  "odds": np.zeros((128, 128), dtype=np.int8), "cell": 1.0, "origin": (-64, -64),
                  "hits": 0, "updates": 0, "map_hash": 0, "slice_y": 0.0, "height_resets": 0,
                  "first_lost_frame": 3, "map_continuous": False}
        reference = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [2, 4, 0], [2, 6, 0]], dtype=float)
        evaluated = mono_map.evaluate_map(result, list(range(5)), np.repeat(np.eye(3)[None], 5, axis=0), reference, point * 2)
        self.assertEqual(evaluated["scored_frames"], 3)
        self.assertAlmostEqual(evaluated["alignment_scale_m_per_unit"], 2.0, places=6)
        self.assertAlmostEqual(evaluated["ate_rmse_m"], 0.0, places=6)
        self.assertAlmostEqual(evaluated["distance_to_cloud_m"]["median"], 0.0, places=6)
        self.assertEqual(len(evaluated["_track"]), 3)

    def test_nearly_collinear_alignment_reports_weak_geometry(self):
        steps = []
        for centre in ((1, 0, 0), (2, 0.00001, 0)):
            step = mvo.StepC()
            step.world_centre[:] = centre
            steps.append(step)
        result = {"steps": steps, "observations": [np.empty((0, 3))] * 2,
                  "odds": np.zeros((128, 128), dtype=np.int8), "cell": 1.0, "origin": (0, 0),
                  "hits": 0, "updates": 0, "map_hash": 0, "slice_y": 0.0, "height_resets": 0}
        reference = np.array([[0, 0, 0], [1, 0, 0], [2, 0.00001, 0]], dtype=float)
        evaluated = mono_map.evaluate_map(result, [0, 1, 2], np.repeat(np.eye(3)[None], 3, axis=0),
                                          reference, np.ones((1, 3)))
        self.assertIsNotNone(evaluated["ate_rmse_m"])
        self.assertGreater(evaluated["alignment_spread_ratio"], 0)
        self.assertLess(evaluated["alignment_spread_ratio"], 1e-8)

    def test_short_or_collinear_continuous_prefix_has_no_alignment_score(self):
        frame = np.zeros((240, 320), dtype=np.uint8)
        for kinds in ((0,), (2, 0), (2, 2, 0)):
            with self.subTest(kinds=kinds):
                vo = FakeVO(kinds=kinds)
                result = mono_map.run_mapping(vo, [frame] * (len(kinds) + 1), [None] * (len(kinds) + 1), mvo.Camera())
                n = len(kinds) + 1
                reference = np.column_stack((np.arange(n), np.zeros(n), np.zeros(n))).astype(float)
                evaluated = mono_map.evaluate_map(result, list(range(n)), np.repeat(np.eye(3)[None], n, axis=0),
                                                  reference, np.ones((1, 3)))
                self.assertIsNone(evaluated["alignment_scale_m_per_unit"])
                self.assertIsNone(evaluated["ate_rmse_m"])
                self.assertIsNone(evaluated["alignment_spread_ratio"])
                self.assertIsNone(evaluated["distance_to_cloud_m"]["median"])
                with tempfile.TemporaryDirectory() as folder:
                    mono_map.draw(Path(folder), 5.0, evaluated, np.ones((1, 3)), result)
                    self.assertTrue((Path(folder) / "map_5hz.png").is_file())

    def test_board_expectation_records_the_interruption_policy(self):
        frame = np.zeros((240, 320), dtype=np.uint8)
        masks = [None] * 4
        result = mono_map.run_mapping(FakeVO(kinds=(2, 0, 2)), [frame] * 4, masks, mvo.Camera())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mono_map.write_board_files(root, root / "masks.bin", result, masks, 10.0)
            expected = json.loads((root / "board_expected_map_10hz.json").read_text(encoding="utf-8"))
        self.assertEqual(expected["map_policy"], mono_map.MAP_POLICY)
        self.assertEqual(expected["map_continuity"], [1, 0, 0])
        self.assertEqual(expected["map_continuous"], 0)

    def test_mask_video_identity_and_frame_times_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video, masks = root / "video.bin", root / "frames.npz"
            video.write_bytes(b"same video")
            np.savez(masks, times=np.arange(3) / 30, mask_nonwater=np.zeros((3, 60, 80), dtype=bool))
            (root / "summary.json").write_text(json.dumps({"source": {"sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                                                                         "fps": 30.0}}), encoding="utf-8")
            with patch.object(mono_map, "MASKS", masks):
                self.assertEqual(len(mono_map.load_matched_masks(video, 30.0)), 3)
                video.write_bytes(b"other video")
                with self.assertRaisesRegex(ValueError, "video"):
                    mono_map.load_matched_masks(video, 30.0)
                video.write_bytes(b"same video")
                np.savez(masks, times=np.arange(3) / 20, mask_nonwater=np.zeros((3, 60, 80), dtype=bool))
                with self.assertRaisesRegex(ValueError, "times"):
                    mono_map.load_matched_masks(video, 30.0)


if __name__ == "__main__":
    unittest.main()
