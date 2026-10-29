"""Integer color/texture features and quantized soft-voting pixel forest."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from uwseg.udcp import min_filter, rgb565_roundtrip

FEATURE_NAMES = [
    "red",
    "green",
    "blue",
    "red_fraction",
    "green_fraction",
    "luma",
    "saturation",
    "red_minus_green",
    "green_minus_blue",
    "mean_luma_3",
    "mean_luma_9",
    "gradient",
    "local_deviation",
    "luma_range_5",
    "mean_red_fraction_9",
    "mean_green_fraction_9",
]


def box_mean(values: np.ndarray, size: int) -> np.ndarray:
    if values.ndim != 2 or not values.size or size < 1 or size % 2 != 1:
        raise ValueError("Expected nonempty 2-D values and a positive odd window")
    padded = np.pad(values, size // 2, mode="edge")
    integral = np.pad(padded.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    total = integral[size:, size:] + integral[:-size, :-size] - integral[size:, :-size] - integral[:-size, size:]
    return (total // (size * size)).astype(np.uint8)


def pack_rgb565(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.uint16)
    return ((values[..., 0] >> 3) << 11) | ((values[..., 1] >> 2) << 5) | (values[..., 2] >> 3)


def extract_features(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("Expected a nonempty uint8 RGB image")
    if rgb.shape[0] > 120 or rgb.shape[1] > 160:
        raise ValueError("Feature input must fit within 160x120")
    values = rgb565_roundtrip(rgb).astype(np.int32)
    r, g, b = (values[..., channel] for channel in range(3))
    total = np.maximum(r + g + b, 1)
    red_fraction, green_fraction = r * 255 // total, g * 255 // total
    maximum, minimum = values.max(2), values.min(2)
    luma = ((r + 2 * g + b) // 4).astype(np.uint8)
    mean3, mean9 = box_mean(luma, 3), box_mean(luma, 9)
    padded = np.pad(luma.astype(np.int32), 1, mode="edge")
    gradient = (np.abs(padded[1:-1, 2:] - padded[1:-1, :-2]) + np.abs(padded[2:, 1:-1] - padded[:-2, 1:-1])) // 2
    local_range = -min_filter(-luma.astype(np.int16), 5) - min_filter(luma, 5)
    planes = [
        r,
        g,
        b,
        red_fraction,
        green_fraction,
        luma,
        (maximum - minimum) * 255 // np.maximum(maximum, 1),
        (r - g + 255) // 2,
        (g - b + 255) // 2,
        mean3,
        mean9,
        gradient,
        np.abs(luma.astype(np.int32) - mean3),
        local_range,
        box_mean(red_fraction, 9),
        box_mean(green_fraction, 9),
    ]
    return np.stack(planes, axis=-1).astype(np.uint8)


def quantize_forest(estimator) -> dict:
    if estimator.classes_.tolist() != [0, 1]:
        raise ValueError("Expected binary labels 0,1")
    nodes, roots = [], []
    for tree in estimator.estimators_:
        tree = tree.tree_
        offset = len(nodes)
        roots.append(offset)
        for index in range(tree.node_count):
            weights = tree.value[index, 0]
            probability = int(math.floor(float(weights[1] / weights.sum()) * 255 + 0.5))
            if tree.children_left[index] < 0:
                nodes.append([0, 0, 255, 0, probability])
            else:
                nodes.append(
                    [
                        offset + int(tree.children_left[index]),
                        offset + int(tree.children_right[index]),
                        int(tree.feature[index]),
                        int(math.floor(tree.threshold[index])),
                        probability,
                    ]
                )
    model = {
        "format_version": 1,
        "feature_names": FEATURE_NAMES,
        "nodes": nodes,
        "roots": roots,
        "max_depth": max(tree.tree_.max_depth for tree in estimator.estimators_),
        "median_window": 3,
        "leaf_probability_bits": 8,
    }
    validate_model(model)
    return model


def validate_model(model):
    nodes = model["nodes"]
    if model["feature_names"] != FEATURE_NAMES or not 1 <= len(nodes) < 65536 or not 1 <= len(model["roots"]) <= 16:
        raise ValueError("Unsupported model dimensions or feature schema")
    if not 0 <= model["max_depth"] <= 8:
        raise ValueError("Model exceeds the declared depth limit")
    for root in model["roots"]:
        pending = [(root, 0, frozenset())]
        while pending:
            index, depth, ancestors = pending.pop()
            if not 0 <= index < len(nodes) or index in ancestors or depth > model["max_depth"]:
                raise ValueError("Invalid child, cycle, or model depth")
            left, right, feature, threshold, probability = nodes[index]
            if not 0 <= probability <= 255:
                raise ValueError("Leaf probability is not uint8")
            if feature == 255:
                continue
            if not 0 <= feature < len(FEATURE_NAMES) or not 0 <= threshold <= 255:
                raise ValueError("Invalid feature or threshold")
            pending.extend((child, depth + 1, ancestors | {index}) for child in (left, right))


def predict_features(features: np.ndarray, model: dict) -> np.ndarray:
    if features.dtype != np.uint8 or features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Expected uint8 feature rows with sixteen columns")
    nodes = np.asarray(model["nodes"], dtype=np.int32)
    accumulated = np.zeros(len(features), dtype=np.uint32)
    for root in model["roots"]:
        indices = np.full(len(features), root, dtype=np.int32)
        for _ in range(model["max_depth"] + 1):
            active = np.flatnonzero(nodes[indices, 2] != 255)
            if not len(active):
                break
            current = nodes[indices[active]]
            left = features[active, current[:, 2]] <= current[:, 3]
            indices[active] = np.where(left, current[:, 0], current[:, 1])
        if np.any(nodes[indices, 2] != 255):
            raise ValueError("Forest traversal did not terminate")
        accumulated += nodes[indices, 4].astype(np.uint32)
    count = len(model["roots"])
    return ((accumulated + count // 2) // count).astype(np.uint8)


def score_image(rgb: np.ndarray, model: dict) -> np.ndarray:
    features = extract_features(rgb)
    score = predict_features(features.reshape(-1, len(FEATURE_NAMES)), model).reshape(rgb.shape[:2])
    return np.array(Image.fromarray(score).filter(ImageFilter.MedianFilter(3)))


def write_model(model: dict, directory: Path, threshold: int):
    validate_model(model)
    if not 0 <= threshold <= 256:
        raise ValueError("Score threshold must be in [0,256]")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pixel_forest_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    header = [
        "/* Generated integer forest; feature schema defined in pixel_forest.h. */",
        "#ifndef PIXEL_FOREST_MODEL_H",
        "#define PIXEL_FOREST_MODEL_H",
        '#include "pixel_forest.h"',
        f"#define PF_TREE_COUNT {len(model['roots'])}",
        f"#define PF_NODE_COUNT {len(model['nodes'])}",
        f"#define PF_THRESHOLD_Q {threshold}",
        "#define PF_MEDIAN_SIZE 3",
        "static const uint16_t PF_ROOTS[] = {" + ",".join(map(str, model["roots"])) + "};",
        "static const PixelForestNode PF_NODES[] = {",
    ]
    header.extend("    {" + ",".join(map(str, node)) + ",0}," for node in model["nodes"])
    header.extend(["};", "#endif", ""])
    (directory / "pixel_forest_model.h").write_text("\n".join(header), encoding="utf-8")
