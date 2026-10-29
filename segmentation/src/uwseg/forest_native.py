"""ctypes bridge to the exact allocation-free C forest used for MCU builds."""

import ctypes as ct
from pathlib import Path

import numpy as np

from uwseg.pixel_forest import pack_rgb565


class NativeForest:
    def __init__(self, library):
        self.library = ct.CDLL(str(Path(library).resolve()))
        lib = self.library
        lib.pf_workspace_bytes.argtypes = [ct.c_int, ct.c_int]
        lib.pf_workspace_bytes.restype = ct.c_size_t
        lib.pf_extract_features.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_size_t]
        lib.pf_extract_features.restype = ct.c_int
        lib.pf_predict_rgb565.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
        ]
        lib.pf_predict_rgb565.restype = ct.c_int
        lib.pf_predict_features.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p]
        lib.pf_predict_features.restype = ct.c_int

    def buffers(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected uint8 RGB image")
        height, width = rgb.shape[:2]
        size = self.library.pf_workspace_bytes(width, height)
        if not size:
            raise ValueError("Image must fit 1..160 by 1..120")
        # uint32 ownership guarantees the alignment required by the C API.
        return np.ascontiguousarray(pack_rgb565(rgb)), np.empty((size + 3) // 4, dtype=np.uint32), size

    def features(self, rgb):
        packed, workspace, size = self.buffers(rgb)
        height, width = packed.shape
        features = np.empty((height, width, 16), dtype=np.uint8)
        result = self.library.pf_extract_features(
            packed.ctypes.data, width, height, features.ctypes.data, workspace.ctypes.data, size
        )
        if result:
            raise RuntimeError(f"C feature extraction failed: {result}")
        return features

    def predict(self, rgb):
        packed, workspace, size = self.buffers(rgb)
        height, width = packed.shape
        scores, masks = (np.empty((height, width), dtype=np.uint8) for _ in range(2))
        result = self.library.pf_predict_rgb565(
            packed.ctypes.data, width, height, scores.ctypes.data, masks.ctypes.data, workspace.ctypes.data, size
        )
        if result:
            raise RuntimeError(f"C prediction failed: {result}")
        return scores, masks

    def predict_rows(self, rows):
        if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] != 16 or not len(rows):
            raise ValueError("Expected nonempty uint8 feature rows with sixteen columns")
        rows = np.ascontiguousarray(rows)
        scores = np.empty(len(rows), dtype=np.uint8)
        result = self.library.pf_predict_features(rows.ctypes.data, len(rows), scores.ctypes.data)
        if result:
            raise RuntimeError(f"C feature prediction failed: {result}")
        return scores
