"""Reduce calibration FPR with an explicit cap on the increase in FNR."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from uwseg.paths import DATA_DIR, RESULTS_DIR, resolve
from uwseg.pixel_forest import extract_features, predict_features, quantize_forest, write_model

# The audited threshold histogram; scikit-learn is imported in run().
from uwseg.train_pixel_forest import SEED, threshold_curve
from uwseg.validate import dump_json, load_sample


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constrained_threshold(counts: np.ndarray, objective: np.ndarray, fnr_cap: float) -> dict:
    """Choose the lowest FPR under the FNR cap; deterministic ties favor mIoU."""
    positives, negatives = counts[:, 0] + counts[:, 2], counts[:, 1] + counts[:, 3]
    if counts.shape != (257, 4) or np.any(positives == 0) or np.any(negatives == 0):
        raise ValueError("Calibration requires both classes and all 257 thresholds")
    fnr, fpr = counts[:, 2] / positives, counts[:, 1] / negatives
    eligible = np.flatnonzero(fnr <= fnr_cap)
    if not len(eligible):
        raise ValueError("No threshold satisfies the FNR cap")
    q = min(eligible, key=lambda q: (fpr[q], -objective[q], fnr[q], q))
    return {
        "threshold_q": int(q),
        "calibration_fnr": float(fnr[q]),
        "calibration_fpr": float(fpr[q]),
        "macro_image_miou": float(objective[q]),
        "calibration_counts": counts[q].tolist(),
    }


def score_features(features: np.ndarray, model: dict) -> np.ndarray:
    raw = predict_features(features.reshape(-1, 16), model).reshape(features.shape[:2])
    return np.array(Image.fromarray(raw).filter(ImageFilter.MedianFilter(3)))


def sample_training_rows(
    features: np.ndarray, labels: np.ndarray, old_score: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniform rows plus separate hard-water rows; only training labels are used."""
    rng = np.random.default_rng(seed)
    rows = features.reshape(-1, 16)
    foreground = (
        np.array(
            Image.fromarray(labels).resize((features.shape[1], features.shape[0]), Image.Resampling.NEAREST)
        ).ravel()
        != 0
    )
    selected = rng.choice(len(rows), min(1024, len(rows)), replace=False)
    # The extra set targets water already mistaken for foreground by the old model.
    available = np.flatnonzero((~foreground) & (old_score.ravel() >= 57))
    available = np.setdiff1d(available, selected)
    available = rng.permutation(available)
    order = np.argsort(-old_score.ravel()[available].astype(np.int16), kind="stable")
    hard = available[order[:256]]
    augmented = np.r_[selected, hard]
    return (
        rows[selected],
        foreground[selected].astype(np.uint8),
        rows[augmented],
        foreground[augmented].astype(np.uint8),
    )


def run(output: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh empty optimization output directory")
    original_path, training_path = DATA_DIR / "manifest.json", DATA_DIR / "lowfp_training_manifest.json"
    original, training = (json.loads(path.read_text(encoding="utf-8")) for path in (original_path, training_path))
    calibration = [row for row in original["samples"] if row["split"] == "calibration"]
    test = [row for row in original["samples"] if row["split"] == "test"]
    splits = [training["samples"], calibration, test]
    sets = [{row["image_sha256"] for row in rows} for rows in splits]
    if [len(rows) for rows in splits] != [1395, 80, 110] or any(len(s) != len(rows) for s, rows in zip(sets, splits)):
        raise ValueError("Expected unique 1395/80/110 train/calibration/test images")
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Image byte-hash overlap between splits")
    old_run = RESULTS_DIR / "forest_comparison"
    old_locked = json.loads((old_run / "LOCKED.json").read_text(encoding="utf-8"))
    for name, digest in old_locked["model_sha256"].items():
        if sha(old_run / "model" / name) != digest:
            raise ValueError("Original forest changed")
    old_model = json.loads((old_run / "model/pixel_forest_model.json").read_text(encoding="utf-8"))
    candidates = [
        {"name": "threshold_only", "reuse": "model/pixel_forest_model.json"},
        {"name": "old_16_8", "reuse": "candidate_16_8.json"},
        {"name": "uniform_sqrt", "max_features": "sqrt", "hard_water": False},
        {"name": "uniform_8", "max_features": 8, "hard_water": False},
        {"name": "hard_water_8", "max_features": 8, "hard_water": True},
    ]
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "algorithm": "same uint8 color/texture features; larger training set, bounded deeper forest, train-only hard-water mining, recall-constrained threshold",
        "seed": SEED,
        "training_images": 1395,
        "uniform_rows_per_image": 1024,
        "extra_hard_water_rows_max_per_image": 256,
        "hard_water_rule": "Exclude uniform rows; training-GT water with original post-median score>=57, descending score, seeded ties; up to256/image",
        "new_forest": {
            "trees": 16,
            "max_depth": 8,
            "min_samples_leaf": 16,
            "class_weight": "balanced_subsample",
            "n_jobs": 2,
        },
        "candidates": candidates,
        "max_calibration_fnr_increase_absolute": 0.02,
        "selection_rule": "For each candidate minimize aggregate calibration FPR subject to FNR <= original forest calibration FNR +0.02. Overall choose lowest FPR, then highest image-macro mIoU, then smallest model. Same FNR cap applies to the threshold-only control.",
        "hardware_limits": {"trees_max": 16, "depth_max": 8, "features": 16, "work_buffer_change": False},
        "test_policy": "Reused same110-image benchmark; no test pixels/labels loaded during training/calibration. Lock primary model and threshold-only control before test. Report pass/failure of +2pp test FNR criterion without retuning.",
        "manifest_sha256": {
            "data/manifest.json": sha(original_path),
            "data/lowfp_training_manifest.json": sha(training_path),
        },
        "original_locked_sha256": sha(old_run / "LOCKED.json"),
        "reused_candidate_sha256": {c["reuse"]: sha(old_run / c["reuse"]) for c in candidates if "reuse" in c},
    }
    dump_json(output / "experiment_plan.json", plan)
    calibration_features = []
    for sample in calibration:
        rgb, labels = load_sample(sample)
        calibration_features.append((extract_features(rgb), labels))
    old_scored = [(score_features(features, old_model), labels) for features, labels in calibration_features]
    old_objective, old_counts = threshold_curve(old_scored)
    old_q = old_locked["chosen"]["threshold_q"]
    baseline_fnr = float(old_counts[old_q, 2] / (old_counts[old_q, 0] + old_counts[old_q, 2]))
    cap = baseline_fnr + plan["max_calibration_fnr_increase_absolute"]
    trials, models = [], {}

    def calibrate(name: str, model: dict) -> None:
        scored = [(score_features(features, model), labels) for features, labels in calibration_features]
        objective, counts = threshold_curve(scored)
        choice = constrained_threshold(counts, objective, cap)
        choice.update(
            name=name, trees=len(model["roots"]), max_depth=model["max_depth"], node_count=len(model["nodes"])
        )
        trials.append(choice)
        models[name] = model
        dump_json(output / f"{name}_curve.json", {"macro_image_miou": objective.tolist(), "counts": counts.tolist()})
        dump_json(output / f"{name}_model.json", model)
        print("Calibration:", json.dumps(choice), flush=True)

    for candidate in candidates:
        if "reuse" in candidate:
            calibrate(candidate["name"], json.loads((old_run / candidate["reuse"]).read_text(encoding="utf-8")))
    uniform_x, uniform_y, mined_x, mined_y = [], [], [], []
    for index, sample in enumerate(training["samples"], 1):
        rgb, labels = load_sample(sample)
        features = extract_features(rgb)
        score = score_features(features, old_model)
        ux, uy, mx, my = sample_training_rows(features, labels, score, SEED ^ int(sample["image_sha256"][:8], 16))
        uniform_x.append(ux)
        uniform_y.append(uy)
        mined_x.append(mx)
        mined_y.append(my)
        if index % 200 == 0:
            print(f"Training-only mining {index}/1395", flush=True)
    x, y = np.concatenate(uniform_x), np.concatenate(uniform_y)
    xm, ym = np.concatenate(mined_x), np.concatenate(mined_y)
    del uniform_x, uniform_y, mined_x, mined_y
    for candidate in candidates:
        if "reuse" in candidate:
            continue
        params = plan["new_forest"]
        estimator = RandomForestClassifier(
            n_estimators=params["trees"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=candidate["max_features"],
            class_weight=params["class_weight"],
            n_jobs=params["n_jobs"],
            random_state=SEED,
        )
        started = time.perf_counter()
        estimator.fit(xm if candidate["hard_water"] else x, ym if candidate["hard_water"] else y)
        model = quantize_forest(estimator)
        calibrate(candidate["name"], model)
        trials[-1]["train_seconds"] = time.perf_counter() - started
    best = min(trials, key=lambda t: (t["calibration_fpr"], -t["macro_image_miou"], t["node_count"]))
    model = models[best["name"]]
    write_model(model, output / "model", best["threshold_q"])
    locked = {
        "chosen": best,
        "trials": trials,
        "original_calibration_counts": old_counts[old_q].tolist(),
        "original_calibration_fnr": baseline_fnr,
        "calibration_fnr_cap": cap,
        "threshold_only_control": trials[0],
        "uniform_training_rows": len(x),
        "mined_training_rows": len(xm),
        "uniform_class_counts": np.bincount(y, minlength=2).tolist(),
        "mined_class_counts": np.bincount(ym, minlength=2).tolist(),
        "model_array_bytes": len(model["nodes"]) * 8 + len(model["roots"]) * 2,
        "plan_sha256": sha(output / "experiment_plan.json"),
        "model_sha256": {path.name: sha(path) for path in (output / "model").iterdir()},
        "source_sha256": {
            name: sha(resolve(name))
            for name in (
                "optimize_pixel_forest.py",
                "pixel_forest.py",
                "train_pixel_forest.py",
                "prepare_lowfp_training.py",
                "embedded/pixel_forest.c",
                "embedded/pixel_forest.h",
            )
        },
    }
    dump_json(output / "LOCKED.json", locked)
    print("LOCKED:", json.dumps(best), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "forest_lowfp")
    run(parser.parse_args().output.resolve())
