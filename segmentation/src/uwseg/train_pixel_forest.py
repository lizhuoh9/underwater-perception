"""Train bounded pixel forests and lock a model using calibration images only."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from uwseg.paths import DATA_DIR, LOCAL_DEPS_DIR, RESULTS_DIR, resolve

if LOCAL_DEPS_DIR.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS_DIR))

import numpy as np
from PIL import Image, ImageFilter

from uwseg.pixel_forest import FEATURE_NAMES, extract_features, predict_features, quantize_forest, write_model
from uwseg.udcp import confusion, metrics, otsu_threshold, udcp_score
from uwseg.validate import at_native_resolution, dump_json, load_sample

SEED = 20261022
CANDIDATES = [(8, 4), (12, 6), (16, 8)]


def threshold_curve(scored_samples):
    macro, total_counts = [], np.zeros((257, 4), dtype=np.int64)
    for score, labels in scored_samples:
        score = at_native_resolution(score, labels)
        foreground = labels != 0
        positive = np.bincount(score[foreground], minlength=256)
        negative = np.bincount(score[~foreground], minlength=256)
        tp = np.r_[positive[::-1].cumsum()[::-1], 0]
        fp = np.r_[negative[::-1].cumsum()[::-1], 0]
        fn, tn = foreground.sum() - tp, (~foreground).sum() - fp
        total_counts += np.column_stack((tp, fp, fn, tn))
        fg_den, bg_den = tp + fp + fn, tn + fp + fn
        fg = np.divide(tp, fg_den, out=np.zeros(257), where=fg_den > 0)
        bg = np.divide(tn, bg_den, out=np.zeros(257), where=bg_den > 0)
        macro.append((fg + bg) / ((fg_den > 0).astype(int) + (bg_den > 0).astype(int)))
    return np.mean(macro, axis=0), total_counts


def run(output):
    # scikit-learn is imported here, so the threshold helpers above work without it.
    import sklearn
    from sklearn.ensemble import RandomForestClassifier

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training output must be a new empty directory")
    original_path, training_path = DATA_DIR / "manifest.json", DATA_DIR / "forest_training_manifest.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    calibration_samples = [row for row in original["samples"] if row["split"] == "calibration"]
    test_samples = [row for row in original["samples"] if row["split"] == "test"]
    if training["audit"]["original_manifest_sha256"] != hashlib.sha256(original_path.read_bytes()).hexdigest():
        raise ValueError("Training split was prepared against a different baseline manifest")
    hashes = [
        {row["image_sha256"] for row in rows} for rows in (training["samples"], calibration_samples, test_samples)
    ]
    if len(training["samples"]) != 512 or len(calibration_samples) != 80 or len(test_samples) != 110:
        raise ValueError("Expected fixed 512/80/110 train/calibration/test splits")
    if any(hashes[i] & hashes[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Image content leakage across splits")
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "algorithm": "random forest on uint8 RGB565 color and local texture features",
        "candidate_tree_count_depth": CANDIDATES,
        "seed": SEED,
        "training_pixels_per_image": 768,
        "max_features": "sqrt",
        "min_samples_leaf": 32,
        "class_weight": "balanced_subsample",
        "median_window": 3,
        "score_quantization": "uint8 leaf probabilities, rounded integer soft vote",
        "selection_rule": "Prefer thresholds with both aggregate calibration FNR and FPR <= UDCP-Otsu calibration; among these maximize per-image mean IoU. If none qualify, maximize that mean IoU and record failed dominance.",
        "training_manifest_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "baseline_manifest_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "test_policy": "Same previously inspected 110-image comparison benchmark; no test image or label is loaded by this training script and no subsequent test-based retuning is allowed in this run.",
        "feature_names": FEATURE_NAMES,
    }
    dump_json(output / "experiment_plan.json", plan)
    xs, ys = [], []
    for index, sample in enumerate(training["samples"], 1):
        rgb, labels = load_sample(sample)
        features = extract_features(rgb).reshape(-1, len(FEATURE_NAMES))
        small_labels = np.array(
            Image.fromarray(labels).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
        ).ravel()
        image_seed = SEED ^ int(sample["image_sha256"][:8], 16)
        selected = np.random.default_rng(image_seed).choice(len(features), min(768, len(features)), replace=False)
        xs.append(features[selected])
        ys.append((small_labels[selected] != 0).astype(np.uint8))
        if index % 128 == 0:
            print(f"Training features {index}/512", flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    calibration = []
    baseline_counts = np.zeros(4, dtype=np.int64)
    baseline_window = json.loads(
        (RESULTS_DIR / "validation_results/locked_parameters.json").read_text(encoding="utf-8")
    )["udcp"]["window"]
    for sample in calibration_samples:
        rgb, labels = load_sample(sample)
        calibration.append((extract_features(rgb), labels))
        score = udcp_score(rgb, baseline_window)[0]
        native = at_native_resolution((score >= otsu_threshold(score)).astype(np.uint8), labels).astype(bool)
        baseline_counts += confusion(native, labels != 0)
    baseline = metrics(baseline_counts)
    baseline_fnr = 1 - baseline["foreground_recall"]
    baseline_fpr = baseline["water_false_positive_rate"]
    trials, available = [], []
    for count, depth in CANDIDATES:
        estimator = RandomForestClassifier(
            n_estimators=count,
            max_depth=depth,
            min_samples_leaf=32,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=SEED,
        )
        start = time.perf_counter()
        estimator.fit(x, y)
        model = quantize_forest(estimator)
        seconds = time.perf_counter() - start
        scored = []
        for features, labels in calibration:
            score = predict_features(features.reshape(-1, len(FEATURE_NAMES)), model).reshape(features.shape[:2])
            score = np.array(Image.fromarray(score).filter(ImageFilter.MedianFilter(3)))
            scored.append((score, labels))
        objective, counts = threshold_curve(scored)
        fnr = counts[:, 2] / (counts[:, 0] + counts[:, 2])
        fpr = counts[:, 1] / (counts[:, 1] + counts[:, 3])
        eligible = (fnr <= baseline_fnr) & (fpr <= baseline_fpr)
        threshold = int(np.where(eligible, objective, -1).argmax()) if eligible.any() else int(objective.argmax())
        trial = {
            "trees": count,
            "max_depth": depth,
            "node_count": len(model["nodes"]),
            "threshold_q": threshold,
            "macro_image_miou": float(objective[threshold]),
            "calibration_fnr": float(fnr[threshold]),
            "calibration_fpr": float(fpr[threshold]),
            "calibration_dominates_baseline": bool(eligible[threshold]),
            "train_seconds": seconds,
        }
        model_name = f"candidate_{count}_{depth}.json"
        dump_json(output / model_name, model)
        trials.append(trial)
        available.append((trial, model))
        print("Calibration candidate:", json.dumps(trial), flush=True)
    best, model = max(
        available,
        key=lambda pair: (
            pair[0]["calibration_dominates_baseline"],
            pair[0]["macro_image_miou"],
            -pair[0]["node_count"],
        ),
    )
    write_model(model, output / "model", best["threshold_q"])
    probe = x[:4096]
    quantized = predict_features(probe, model).astype(int)
    # Reconstruct the chosen sklearn model deterministically to quantify leaf rounding.
    estimator = RandomForestClassifier(
        n_estimators=best["trees"],
        max_depth=best["max_depth"],
        min_samples_leaf=32,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=2,
        random_state=SEED,
    ).fit(x, y)
    floating = estimator.predict_proba(probe)[:, 1] * 255
    locked = {
        "chosen": best,
        "trials": trials,
        "baseline_calibration_metrics": baseline,
        "train_images": len(training["samples"]),
        "train_pixel_rows": len(x),
        "training_pixel_class_counts": np.bincount(y, minlength=2).tolist(),
        "calibration_images": len(calibration_samples),
        "test_images": len(test_samples),
        "model_array_bytes": len(model["nodes"]) * 8 + len(model["roots"]) * 2,
        "float_vs_uint8_max_score_difference_on_train_probe": float(np.abs(quantized - floating).max()),
        "sklearn_version": sklearn.__version__,
        "plan_sha256": hashlib.sha256((output / "experiment_plan.json").read_bytes()).hexdigest(),
        "model_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (output / "model").iterdir()
        },
        "source_sha256": {
            name: hashlib.sha256(resolve(name).read_bytes()).hexdigest()
            for name in ("pixel_forest.py", "train_pixel_forest.py", "prepare_forest_training.py")
        },
    }
    dump_json(output / "LOCKED.json", locked)
    print("LOCKED:", json.dumps(best), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "forest_comparison")
    run(parser.parse_args().output.resolve())
