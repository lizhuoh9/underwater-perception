"""Water masks for the camera experiments: the robot camera's input path and an ESP-DL int8 segmentation model.

`device_frame` turns any RGB frame into what the robot's camera feeds its model: stretched to 320x240, 2x2-averaged to
160x120 and truncated to RGB565. `EspdlSegmenter` runs an int8 model exported by esp-ppq on such frames with the
board's own input quantisation and decision threshold, giving the 60x80 non-water mask the board would compute.

A model folder holds water_seg_native (the esp-ppq native graph) and LOCKED.json with mean and std (per channel, in
device units), input_exponent and output_exponent, chosen.threshold_q (score threshold on 0..255),
chosen.threshold_code and artifact_sha256.water_seg_native. Running the model needs torch and esp-ppq.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

INPUT_HEIGHT, INPUT_WIDTH = 120, 160
WORK_HEIGHT, WORK_WIDTH = 2 * INPUT_HEIGHT, 2 * INPUT_WIDTH  # the camera frame before the 2x2 average
RGB565_KEEP = np.array([0xF8, 0xFC, 0xF8], dtype=np.uint8)


def model_frame(work: np.ndarray) -> np.ndarray:
    """320x240 frame -> model input: 2x2 block average, then the device's RGB565 truncation."""
    pooled = work.reshape(INPUT_HEIGHT, 2, INPUT_WIDTH, 2, 3).astype(np.float32).mean(axis=(1, 3))
    return np.rint(pooled).astype(np.uint8) & RGB565_KEEP


def device_frame(rgb: np.ndarray) -> np.ndarray:
    """Any RGB frame -> the robot's 160x120 model input (stretched to 4:3, 2x2-averaged, RGB565-truncated)."""
    return model_frame(np.array(Image.fromarray(rgb).resize((WORK_WIDTH, WORK_HEIGHT), Image.Resampling.BILINEAR)))


def device_codes(frames: np.ndarray, mean: np.ndarray, std: np.ndarray, exponent: int) -> np.ndarray:
    """esp-dl input quantisation on the ESP32-S3, as NCHW int8: float32 (v - mean) * (1/std) * 2^-exp, then floor(x + 0.5)."""
    inv_std = np.float32(1.0) / std.astype(np.float32)
    scaled = (frames.astype(np.float32) - mean.astype(np.float32)) * inv_std * np.float32(2.0 ** -exponent)
    codes = np.clip(np.floor(scaled.astype(np.float64) + 0.5), -128, 127).astype(np.int8)
    return np.ascontiguousarray(codes.transpose(0, 3, 1, 2))


def int8_scores(logits, exponent: int) -> np.ndarray:
    """Fake-quantised logits (torch, N x 1 x H x W) -> int8 output codes shifted to 0..255 (code + 128)."""
    import torch
    codes = torch.round(logits / 2.0 ** exponent).clamp(-128, 127)
    return (codes + 128).to(torch.uint8)[:, 0].cpu().numpy()


class EspdlSegmenter:
    """An exported int8 water-segmentation model at its deployment threshold: the mask the board computes for a frame."""

    def __init__(self, model_dir: Path):
        locked = json.loads((model_dir / "LOCKED.json").read_text(encoding="utf-8"))
        native = model_dir / "water_seg_native"
        if hashlib.sha256(native.read_bytes()).hexdigest() != locked["artifact_sha256"]["water_seg_native"]:
            raise ValueError(f"Changed since locking: {native}")
        from esp_ppq import TorchExecutor
        from esp_ppq.api import load_native_graph
        self.model_dir = model_dir
        self.executor = TorchExecutor(graph=load_native_graph(str(native)), device="cpu")
        self.mean, self.std = (np.array(locked[key], dtype=np.float32) for key in ("mean", "std"))
        self.exponent_in, self.exponent_out = locked["input_exponent"], locked["output_exponent"]
        self.threshold = locked["chosen"]["threshold_q"]
        self.threshold_code = locked["chosen"]["threshold_code"]

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """160x120 device frame -> 60x80 boolean non-water mask."""
        import torch
        codes = device_codes(frame[None], self.mean, self.std, self.exponent_in)
        with torch.no_grad():
            quant = self.executor.forward(inputs=torch.from_numpy(codes.astype(np.float32)) * 2.0 ** self.exponent_in)[0]
        return int8_scores(quant, self.exponent_out)[0] >= self.threshold

    def describe(self) -> dict:
        """What the results record about the model: the hash of its lock file and its decision threshold."""
        return {"locked_sha256": hashlib.sha256((self.model_dir / "LOCKED.json").read_bytes()).hexdigest(),
                "threshold_code": self.threshold_code}
