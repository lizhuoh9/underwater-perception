"""Re-select the locked thresholds on calibration images without training neighbours.

Calibration-only diagnostic: no TEST image or label is loaded and nothing is
retrained. The selection rules are those of train_pixel_forest.py (threshold 57)
and optimize_pixel_forest.py (thresholds 71 and 72). Both are first re-run on all
80 calibration images and must reproduce every locked threshold and calibration
metric exactly; only then are they applied to the images that the near-duplicate
audit found no training neighbour for.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from uwseg.compare_dedup import load_audit
from uwseg.optimize_pixel_forest import constrained_threshold, score_features
from uwseg.paths import DATA_DIR, RESULTS_DIR, resolve
from uwseg.pixel_forest import extract_features, validate_model
from uwseg.train_pixel_forest import threshold_curve
from uwseg.udcp import confusion, metrics, otsu_threshold, udcp_score
from uwseg.validate import at_native_resolution, dump_json, load_sample

SEED = 20261022
BOOTSTRAP_DRAWS = 2000
FIRST_ROUND = (
    "forest_comparison/candidate_8_4.json",
    "forest_comparison/candidate_12_6.json",
    "forest_comparison/candidate_16_8.json",
)
LOWFP = {
    name: f"forest_lowfp/{name}_model.json"
    for name in ("threshold_only", "old_16_8", "uniform_sqrt", "uniform_8", "hard_water_8")
}
REFERENCE = "forest_comparison/candidate_12_6.json"
SOURCES = (
    "recheck_calibration.py",
    "train_pixel_forest.py",
    "optimize_pixel_forest.py",
    "pixel_forest.py",
    "compare_dedup.py",
    "udcp.py",
    "validate.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_curve(score: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-image (257, 4) confusion counts and mean IoU per threshold, as in threshold_curve."""
    native = at_native_resolution(score, labels)
    foreground = labels != 0
    tp = np.r_[np.bincount(native[foreground], minlength=256)[::-1].cumsum()[::-1], 0]
    fp = np.r_[np.bincount(native[~foreground], minlength=256)[::-1].cumsum()[::-1], 0]
    fn, tn = foreground.sum() - tp, (~foreground).sum() - fp
    fg_den, bg_den = tp + fp + fn, tn + fp + fn
    fg = np.divide(tp, fg_den, out=np.zeros(257), where=fg_den > 0)
    bg = np.divide(tn, bg_den, out=np.zeros(257), where=bg_den > 0)
    return np.column_stack((tp, fp, fn, tn)), (fg + bg) / ((fg_den > 0).astype(int) + (bg_den > 0).astype(int))


def rates(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return counts[..., 2] / (counts[..., 0] + counts[..., 2]), counts[..., 1] / (counts[..., 1] + counts[..., 3])


def first_round_threshold(objective: np.ndarray, counts: np.ndarray, baseline_fnr, baseline_fpr) -> dict:
    """train_pixel_forest.py: best mean IoU among thresholds no worse than UDCP-Otsu on both error rates."""
    fnr, fpr = rates(counts)
    eligible = (fnr <= baseline_fnr) & (fpr <= baseline_fpr)
    q = int(np.where(eligible, objective, -1).argmax()) if eligible.any() else int(objective.argmax())
    return {
        "threshold_q": q,
        "macro_image_miou": float(objective[q]),
        "calibration_fnr": float(fnr[q]),
        "calibration_fpr": float(fpr[q]),
        "calibration_dominates_baseline": bool(eligible[q]),
    }


def load_models() -> dict:
    paths = dict.fromkeys(FIRST_ROUND) | dict.fromkeys(LOWFP.values())
    models = {}
    for path in paths:
        model = json.loads(resolve(path).read_text(encoding="utf-8"))
        validate_model(model)
        models[path] = model
    return models


def score_calibration(rows: list[dict], models: dict, window: int) -> dict:
    """Per-image curves for each distinct model, plus the UDCP-Otsu baseline counts."""
    distinct = {}
    for path, model in models.items():
        distinct.setdefault(json.dumps(model, sort_keys=True), []).append(path)
    curves = {path: {"counts": [], "macro": [], "scored": []} for path in models}
    udcp = []
    for index, row in enumerate(rows, 1):
        rgb, labels = load_sample(row)
        features = extract_features(rgb)
        for paths in distinct.values():
            score = score_features(features, models[paths[0]])
            counts, macro = image_curve(score, labels)
            for path in paths:
                curves[path]["counts"].append(counts)
                curves[path]["macro"].append(macro)
                curves[path]["scored"].append((score, labels))
        score = udcp_score(rgb, window)[0]
        predicted = at_native_resolution((score >= otsu_threshold(score)).astype(np.uint8), labels).astype(bool)
        udcp.append(confusion(predicted, labels != 0))
        if index % 20 == 0:
            print(f"Scored calibration {index}/{len(rows)}", flush=True)
    return {
        "curves": {
            path: {"counts": np.stack(c["counts"]), "macro": np.stack(c["macro"]), "scored": c["scored"]}
            for path, c in curves.items()
        },
        "udcp": np.array(udcp, dtype=np.int64),
    }


def aggregate(curve: dict, picked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return curve["macro"][picked].mean(axis=0), curve["counts"][picked].sum(axis=0)


def locked_thresholds() -> dict:
    first = json.loads((RESULTS_DIR / "forest_comparison/LOCKED.json").read_text(encoding="utf-8"))
    lowfp = json.loads((RESULTS_DIR / "forest_lowfp/LOCKED.json").read_text(encoding="utf-8"))
    return {
        "57_rule_12x6": first["chosen"]["threshold_q"],
        "71_rule_threshold_only": lowfp["threshold_only_control"]["threshold_q"],
        "72_rule_uniform_sqrt": lowfp["chosen"]["threshold_q"],
    }


def select(scored: dict, models: dict, picked: np.ndarray, reference_q: int) -> dict:
    """Both locked selection procedures on the chosen calibration images.

    optimize_pixel_forest.py caps FNR at the original forest's calibration FNR at
    `reference_q` (its locked threshold) plus 2 points.
    """
    baseline = metrics(scored["udcp"][picked].sum(axis=0))
    baseline_fnr, baseline_fpr = 1 - baseline["foreground_recall"], baseline["water_false_positive_rate"]
    first = []
    for path in FIRST_ROUND:
        model = models[path]
        trial = first_round_threshold(*aggregate(scored["curves"][path], picked), baseline_fnr, baseline_fpr)
        first.append(
            {
                "model": path,
                "trees": len(model["roots"]),
                "max_depth": model["max_depth"],
                "node_count": len(model["nodes"]),
                **trial,
            }
        )
    first_choice = max(
        first, key=lambda t: (t["calibration_dominates_baseline"], t["macro_image_miou"], -t["node_count"])
    )
    reference = next(t for t in first if t["model"] == REFERENCE)
    reference_counts = aggregate(scored["curves"][REFERENCE], picked)[1]
    reference_fnr = float(
        reference_counts[reference_q, 2] / (reference_counts[reference_q, 0] + reference_counts[reference_q, 2])
    )
    cap = reference_fnr + 0.02
    second = []
    for name, path in LOWFP.items():
        objective, counts = aggregate(scored["curves"][path], picked)
        choice = constrained_threshold(counts, objective, cap)
        second.append(
            {
                **choice,
                "name": name,
                "trees": len(models[path]["roots"]),
                "max_depth": models[path]["max_depth"],
                "node_count": len(models[path]["nodes"]),
            }
        )
    second_choice = min(second, key=lambda t: (t["calibration_fpr"], -t["macro_image_miou"], t["node_count"]))
    return {
        "images": int(len(picked)),
        "udcp_otsu_baseline": {"fnr": baseline_fnr, "fpr": baseline_fpr},
        "first_round_trials": first,
        "first_round_choice": first_choice["model"],
        "reference_threshold_q": reference_q,
        "reference_fnr": reference_fnr,
        "fnr_cap": cap,
        "lowfp_trials": second,
        "lowfp_choice": second_choice["name"],
        "thresholds": {
            "57_rule_12x6": reference["threshold_q"],
            "71_rule_threshold_only": next(t["threshold_q"] for t in second if t["name"] == "threshold_only"),
            "72_rule_uniform_sqrt": next(t["threshold_q"] for t in second if t["name"] == "uniform_sqrt"),
        },
    }


def same(value, recorded) -> bool:
    if isinstance(recorded, float):
        return math.isclose(value, recorded, rel_tol=0, abs_tol=1e-12)
    return value == recorded


def verify_locked(result: dict, scored: dict) -> None:
    """The full-calibration rerun must equal every locked threshold and calibration metric."""
    locked = json.loads((RESULTS_DIR / "forest_comparison/LOCKED.json").read_text(encoding="utf-8"))
    baseline = scored["udcp"].sum(axis=0).tolist()
    if baseline != [locked["baseline_calibration_metrics"][key] for key in ("tp", "fp", "fn", "tn")]:
        raise AssertionError("UDCP-Otsu calibration baseline no longer reproduces")
    keys = ("threshold_q", "macro_image_miou", "calibration_fnr", "calibration_fpr", "calibration_dominates_baseline")
    for trial, recorded in zip(result["first_round_trials"], locked["trials"]):
        if (trial["trees"], trial["max_depth"]) != (recorded["trees"], recorded["max_depth"]) or not all(
            same(trial[key], recorded[key]) for key in keys
        ):
            raise AssertionError(f"First-round selection no longer reproduces: {trial['model']}")
    chosen = next(t for t in result["first_round_trials"] if t["model"] == result["first_round_choice"])
    if (chosen["trees"], chosen["max_depth"]) != (locked["chosen"]["trees"], locked["chosen"]["max_depth"]):
        raise AssertionError("First-round model choice no longer reproduces")
    lowfp = json.loads((RESULTS_DIR / "forest_lowfp/LOCKED.json").read_text(encoding="utf-8"))
    if not same(result["fnr_cap"], lowfp["calibration_fnr_cap"]) or result["lowfp_choice"] != lowfp["chosen"]["name"]:
        raise AssertionError("Low-FP cap or model choice no longer reproduces")
    recorded = {trial["name"]: trial for trial in lowfp["trials"]}
    for trial in result["lowfp_trials"]:
        if not all(
            same(trial[key], recorded[trial["name"]][key])
            for key in ("threshold_q", "calibration_fnr", "calibration_fpr", "macro_image_miou", "calibration_counts")
        ):
            raise AssertionError(f"Low-FP selection no longer reproduces: {trial['name']}")
    for path in FIRST_ROUND:
        objective, counts = threshold_curve(scored["curves"][path]["scored"])
        mine = aggregate(scored["curves"][path], np.arange(len(scored["udcp"])))
        if not (np.array_equal(objective, mine[0]) and np.array_equal(counts, mine[1])):
            raise AssertionError("Per-image curves disagree with train_pixel_forest.threshold_curve")


def bootstrap(scored: dict, picked: np.ndarray, locked: dict) -> dict:
    """Selected thresholds over image-level resamples; the 71/72 cap stays anchored at the locked 57."""
    weights = np.random.default_rng(SEED).multinomial(
        len(picked), np.full(len(picked), 1 / len(picked)), size=BOOTSTRAP_DRAWS
    )

    def resampled(path):
        curve = scored["curves"][path]
        return (weights @ curve["macro"][picked]) / len(picked), np.einsum(
            "di,iqk->dqk", weights, curve["counts"][picked]
        )

    objective, counts = resampled(REFERENCE)
    fnr, fpr = rates(counts)
    baseline = weights @ scored["udcp"][picked]
    baseline_fnr, baseline_fpr = rates(baseline)
    eligible = (fnr <= baseline_fnr[:, None]) & (fpr <= baseline_fpr[:, None])
    first = np.where(eligible.any(axis=1), np.where(eligible, objective, -1).argmax(axis=1), objective.argmax(axis=1))
    caps = fnr[:, locked["57_rule_12x6"]] + 0.02
    chosen = {"57_rule_12x6": first}
    for key, name in (("71_rule_threshold_only", "threshold_only"), ("72_rule_uniform_sqrt", "uniform_sqrt")):
        model_objective, model_counts = resampled(LOWFP[name])
        chosen[key] = np.array(
            [
                constrained_threshold(model_counts[d], model_objective[d], caps[d])["threshold_q"]
                for d in range(BOOTSTRAP_DRAWS)
            ]
        )
    result = {}
    for key, values in chosen.items():
        found, frequency = np.unique(values, return_counts=True)
        result[key] = {
            "locked_q": locked[key],
            "share_equal_locked": float(np.mean(values == locked[key])),
            "share_within_2": float(np.mean(np.abs(values - locked[key]) <= 2)),
            "q_95_interval": [int(v) for v in np.percentile(values, [2.5, 97.5], method="nearest")],
            "mode": int(found[frequency.argmax()]),
            "distribution": {str(int(q)): int(n) for q, n in zip(found, frequency)},
        }
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "unit": "calibration image",
        "cap_reference_q": locked["57_rule_12x6"],
        "rules": result,
    }


def curve_summary(curve: dict, picked: np.ndarray) -> dict:
    objective, counts = aggregate(curve, picked)
    fnr, fpr = rates(counts)
    return {"macro_image_miou": objective.tolist(), "fnr": fnr.tolist(), "fpr": fpr.tolist()}


def write_report(output: Path, summary: dict) -> None:
    full, clean, chained = summary["all_80"], summary["clean"], summary["clean_chained"]
    rules = (
        ("57_rule_12x6", "57：原 12×6 森林，FNR/FPR 均不劣于 UDCP-Otsu 时取最高逐图 mIoU"),
        ("71_rule_threshold_only", "71：原森林只调阈值，FNR 不超过参照 +2 个百分点时取最低 FPR"),
        ("72_rule_uniform_sqrt", "72：16×8 低误检森林，同上规则"),
    )
    locked = summary["locked_thresholds"]
    verdict, recorded = clean["thresholds"]["71_rule_threshold_only"], locked["71_rule_threshold_only"]
    lines = [
        "# 标定集去泄漏后的阈值复核",
        "",
        f"只用标定图，不加载任何 TEST 图或标注，不重新训练。80 张标定图中 {summary['excluded_count']} 张在训练集里有近重复或同场景邻居"
        f"（见去重审计），在其余 {clean['images']} 张上按原规则重新选择阈值。",
        "",
        f"**结论：{recorded} 的规则在 {clean['images']} 张干净标定图上选出 {verdict}"
        + ("，与锁定值相同。**" if verdict == recorded else f"，与锁定值 {recorded} 不同。**"),
        "",
        "| 规则 | 全部 80 张（复现锁定值） | 干净标定图 | 干净图上 FNR / FPR | 自助法：等于锁定值 | ±2 以内 | 自助法 95% 区间 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = {
        "57_rule_12x6": next(t for t in clean["first_round_trials"] if t["model"] == REFERENCE),
        "71_rule_threshold_only": next(t for t in clean["lowfp_trials"] if t["name"] == "threshold_only"),
        "72_rule_uniform_sqrt": next(t for t in clean["lowfp_trials"] if t["name"] == "uniform_sqrt"),
    }
    for key, label in rules:
        boot = summary["bootstrap_clean"]["rules"][key]
        trial = rows[key]
        lines.append(
            f"| {label} | {full['thresholds'][key]} | {clean['thresholds'][key]} | "
            f"{trial['calibration_fnr']:.2%} / {trial['calibration_fpr']:.2%} | {boot['share_equal_locked']:.0%} | "
            f"{boot['share_within_2']:.0%} | {boot['q_95_interval'][0]}–{boot['q_95_interval'][1]} |"
        )
    draws = summary["bootstrap_clean"]["draws"]
    lines.extend(
        [
            "",
            "干净标定图自助法最常选出的阈值："
            + "；".join(
                f"{key.split('_')[0]} 的规则 "
                + "、".join(
                    f"{q}（{n / draws:.0%}）"
                    for q, n in sorted(
                        summary["bootstrap_clean"]["rules"][key]["distribution"].items(), key=lambda item: -item[1]
                    )[:3]
                )
                for key, _ in rules
            )
            + "。",
        ]
    )
    full_boot = summary["bootstrap_all_80"]["rules"]
    lines.extend(
        [
            "",
            f"71/72 规则的漏检上限以原森林在锁定阈值 {locked['57_rule_12x6']} 处的标定 FNR 加 2 个百分点为准：全部 80 张为 {full['fnr_cap']:.2%}，"
            f"干净标定图为 {clean['fnr_cap']:.2%}。若参照阈值也改用干净图上重选的 {chained['reference_threshold_q']}，"
            f"71 的规则选出 {chained['thresholds']['71_rule_threshold_only']}，72 的规则选出 {chained['thresholds']['72_rule_uniform_sqrt']}。",
            "",
            f"模型选择：第一轮在干净图上选 {Path(clean['first_round_choice']).stem}（锁定为 candidate_12_6）；"
            f"低误检一轮选 {clean['lowfp_choice']}（锁定为 uniform_sqrt）。",
            "",
            "自助法按标定图重抽样（"
            f"{summary['bootstrap_clean']['draws']} 次，种子 {summary['bootstrap_clean']['seed']}），"
            f"71/72 规则的漏检上限在每次抽样中按锁定参照阈值 {locked['57_rule_12x6']} 重算。作为对照，在全部 80 张上重抽样时等于锁定值的比例为："
            + "，".join(f"{key.split('_')[0]} {full_boot[key]['share_equal_locked']:.0%}" for key, _ in rules)
            + "。",
            "",
            "## 核查",
            "",
            "- 在全部 80 张标定图上重跑两套规则，阈值、标定 FNR/FPR、逐图 mIoU、UDCP 基线计数及模型选择与锁定记录完全一致后，才计算干净子集。",
            "- UDCP 窗口沿用锁定的 9；排除名单来自已校验的去重审计（与 1395 张训练图比较，覆盖两轮模型的训练集）。",
            "",
            "[结果 JSON](summary.json) · [去重审计](../../data/dedup_manifest.json)",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run(output: Path, manifest_path: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite diagnostic evidence; choose a fresh --output")
    audit, _ = load_audit(manifest_path)
    original = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in original["samples"] if row["split"] == "calibration"]
    excluded = set(audit["calibration_leaks"])
    clean = np.array([index for index, row in enumerate(rows) if row["id"] not in excluded])
    models = load_models()
    window = json.loads((RESULTS_DIR / "validation_results/locked_parameters.json").read_text(encoding="utf-8"))[
        "udcp"
    ]["window"]
    scored = score_calibration(rows, models, window)
    locked = locked_thresholds()
    every = np.arange(len(rows))
    full = select(scored, models, every, locked["57_rule_12x6"])
    verify_locked(full, scored)
    print("All 80 calibration images reproduce every locked threshold and metric", flush=True)
    clean_result = select(scored, models, clean, locked["57_rule_12x6"])
    chained = select(scored, models, clean, clean_result["thresholds"]["57_rule_12x6"])
    output.mkdir(parents=True, exist_ok=True)
    dump_json(
        output / "curves.json",
        {
            path: {
                subset: curve_summary(scored["curves"][path], picked)
                for subset, picked in (("all_80", every), ("clean", clean))
            }
            for path in models
        },
    )
    summary = {
        "task": "Calibration-only re-selection of the locked thresholds without calibration images that have training neighbours",
        "excluded_count": len(excluded),
        "excluded_calibration_ids": sorted(excluded),
        "clean_calibration_ids": [rows[i]["id"] for i in clean],
        "locked_thresholds": locked,
        "all_80": full,
        "clean": clean_result,
        "clean_chained": chained,
        "bootstrap_all_80": bootstrap(scored, every, locked),
        "bootstrap_clean": bootstrap(scored, clean, locked),
        "policy": "No TEST image or label is loaded; no model is retrained; UDCP window stays at the locked value.",
        "dedup_manifest_sha256": sha(manifest_path),
        "model_sha256": {path: sha(resolve(path)) for path in models},
        "source_sha256": {name: sha(resolve(name)) for name in SOURCES},
    }
    write_report(output, summary)
    dump_json(output / "summary.json", summary)
    print(
        json.dumps({key: summary[key]["thresholds"] for key in ("all_80", "clean", "clean_chained")}, indent=2),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "calibration_recheck")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "dedup_manifest.json")
    args = parser.parse_args()
    run(args.output.resolve(), args.manifest.resolve())
