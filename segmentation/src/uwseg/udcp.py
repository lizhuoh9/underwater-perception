"""Small desktop UDCP baseline; scores are appearance cues, not collision truth."""

import numpy as np
from PIL import Image, ImageFilter


def min_filter(values: np.ndarray, size: int) -> np.ndarray:
    """Square, edge-replicated minimum via two 1-D passes (O(N * size))."""
    if values.ndim != 2 or not values.size or size < 1 or size % 2 != 1:
        raise ValueError("Expected a nonempty 2-D array and a positive odd window")
    radius = size // 2
    rows = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    horizontal = np.lib.stride_tricks.sliding_window_view(rows, size, axis=1).min(axis=-1)
    columns = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(columns, size, axis=0).min(axis=-1)


def estimate_ambient(rgb: np.ndarray, window: int, fraction: float = 0.001) -> np.ndarray:
    """DCP-style top-dark-channel candidates; return RGB background light in [0,1]."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("Expected a nonempty uint8 RGB image")
    if not 0 < fraction <= 1:
        raise ValueError("Ambient candidate fraction must be in (0, 1]")
    dark = min_filter(rgb[..., 1:].min(axis=2), window)
    count = max(1, int(np.ceil(dark.size * fraction)))
    indices = np.argsort(dark.ravel(), kind="stable")[-count:]
    candidates = rgb.reshape(-1, 3)[indices]
    chosen = candidates[candidates.astype(np.uint16).sum(axis=1).argmax()]
    return chosen.astype(np.float32) / 255


def transmission(rgb: np.ndarray, window: int, ambient: np.ndarray) -> np.ndarray:
    """Normalized G/B model. ambient must be positive; values outside [0,1] clip."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("Expected a nonempty normalized RGB image")
    ambient = np.asarray(ambient, dtype=np.float32)
    if ambient.shape != (3,) or not np.isfinite(ambient).all() or np.any(ambient <= 0):
        raise ValueError("Ambient light must contain three positive finite values")
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("RGB values must be finite and normalized to [0,1]")
    normalized = np.minimum(rgb[..., 1] / ambient[1], rgb[..., 2] / ambient[2])
    return np.clip(1 - min_filter(normalized, window), 0, 1)


def udcp_score(rgb: np.ndarray, window: int = 5) -> tuple[np.ndarray, dict]:
    ambient = estimate_ambient(rgb, window)
    floored = np.maximum(ambient, 1 / 255)
    t = transmission(rgb.astype(np.float32) / 255, window, floored)
    # shortcut: fixed 3x3 median only; full paper refinement/control is out of scope.
    score = np.rint(t * 255).astype(np.uint8)
    score = np.array(Image.fromarray(score).filter(ImageFilter.MedianFilter(3)))
    flags = []
    if np.any(ambient[1:] <= 1 / 255):
        flags.append("low_background_light")
    if float(score.std()) < 1:
        flags.append("nearly_constant_score")
    return score, {"ambient_rgb": ambient.tolist(), "flags": flags}


def red_fraction_score(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32)
    fraction = values[..., 0] / np.maximum(values.sum(axis=2), 1)
    return np.rint(fraction * 255).astype(np.uint8)


def rgb565_roundtrip(rgb: np.ndarray) -> np.ndarray:
    """Only channel quantization, not an OV2640 sensor/noise/ISP simulation."""
    values = rgb.astype(np.uint16)
    result = np.empty_like(rgb)
    for channel, shift, maximum in ((0, 3, 31), (1, 2, 63), (2, 3, 31)):
        result[..., channel] = ((values[..., channel] >> shift) * 255 // maximum).astype(np.uint8)
    return result


def decode_labels(mask_rgb: np.ndarray) -> np.ndarray:
    if mask_rgb.dtype != np.uint8 or not mask_rgb.size:
        raise ValueError("Expected a nonempty uint8 SUIM mask")
    if mask_rgb.ndim == 2:
        if mask_rgb.max() > 7:
            raise ValueError("Grayscale SUIM class IDs must be in [0,7]")
        return mask_rgb.copy()
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError("Expected grayscale class IDs or an RGB SUIM mask")
    if mask_rgb.max() <= 7 and np.all(mask_rgb == mask_rgb[..., :1]):
        return mask_rgb[..., 0].copy()
    bits = (mask_rgb > 127).astype(np.uint8)
    return bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]


def confusion(predicted: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    if predicted.shape != truth.shape or predicted.dtype != bool or truth.dtype != bool:
        raise ValueError("Prediction and ground truth must be equally shaped boolean masks")
    return (
        int(np.count_nonzero(predicted & truth)),
        int(np.count_nonzero(predicted & ~truth)),
        int(np.count_nonzero(~predicted & truth)),
        int(np.count_nonzero(~predicted & ~truth)),
    )


def metrics(counts) -> dict:
    tp, fp, fn, tn = map(int, counts)

    def divide(a, b):
        return a / b if b else None

    foreground_iou = divide(tp, tp + fp + fn)
    water_iou = divide(tn, tn + fp + fn)
    ious = [value for value in (foreground_iou, water_iou) if value is not None]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "foreground_iou": foreground_iou,
        "water_iou": water_iou,
        "miou": sum(ious) / len(ious) if ious else None,
        "foreground_precision": divide(tp, tp + fp),
        "foreground_recall": divide(tp, tp + fn),
        "foreground_f1": divide(2 * tp, 2 * tp + fp + fn),
        "water_false_positive_rate": divide(fp, fp + tn),
        "pixel_accuracy": divide(tp + tn, tp + fp + fn + tn),
    }


def otsu_threshold(score: np.ndarray) -> int:
    hist = np.bincount(score.ravel(), minlength=256).astype(float)
    total = hist.sum()
    left_count = hist.cumsum()
    left_sum = (hist * np.arange(256)).cumsum()
    denominator = left_count * (total - left_count)
    numerator = (left_sum * total - left_count * left_sum[-1]) ** 2
    between = np.divide(numerator, denominator, out=np.zeros(256), where=denominator > 0)
    return int(between.argmax()) + 1
