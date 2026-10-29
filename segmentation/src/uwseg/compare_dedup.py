"""Re-run the frozen comparison on TEST subsets without near-duplicate or same-scene leaks.

Run after prepare_dedup_split.py. Models, thresholds, compiled libraries and images
are the locked earlier ones; nothing is retrained or retuned. Every per-image
confusion count must first reproduce the recorded evaluations exactly.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from uwseg.compare_lowfp import load_locked as load_lowfp_locked
from uwseg.compare_pixel_forest import check_native
from uwseg.compare_pixel_forest import load_locked as load_forest_locked
from uwseg.forest_native import NativeForest
from uwseg.near_duplicates import leaked_ids
from uwseg.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR, resolve
from uwseg.pixel_forest import score_image
from uwseg.prepare_dedup_split import CODE, leak_bands
from uwseg.udcp import confusion, metrics, otsu_threshold, rgb565_roundtrip, udcp_score
from uwseg.validate import at_native_resolution, dump_json, font, load_sample

# The seven TEST leaks reported before the full screen; kept to show that step alone.
PREVIOUSLY_REPORTED = ("n_l_26_", "f_r_403_", "n_l_102_", "n_l_79_", "f_r_1892_", "n_l_100_", "d_r_384_")
METHODS = ("udcp_fixed", "udcp_otsu", "udcp_otsu_rgb565", "forest_original", "forest_threshold_only", "forest_lowfp")
LABELS = {
    "udcp_fixed": "UDCP 固定阈值 RGB888",
    "udcp_otsu": "UDCP-Otsu RGB888",
    "udcp_otsu_rgb565": "UDCP-Otsu RGB565",
    "forest_original": "原森林，阈值 57",
    "forest_threshold_only": "原森林，阈值 71（推荐）",
    "forest_lowfp": "16×8 森林，阈值 72",
}
SUBSETS = {
    "all": "全部",
    "without_reported_7": "去掉先前报告的 7 张",
    "without_near_duplicates": "去掉全部近重复",
    "without_same_scene": "再去掉已发现的同场景",
}
# Each method must reproduce these recorded per-image confusion counts.
RECORDED = {
    "udcp_fixed": (("validation_results/per_image.csv", "udcp_fixed"),),
    "udcp_otsu": (
        ("validation_results/per_image.csv", "udcp_otsu"),
        ("forest_comparison/evaluation/per_image.csv", "udcp_otsu"),
    ),
    "udcp_otsu_rgb565": (
        ("forest_comparison/evaluation/per_image.csv", "udcp_otsu_rgb565"),
        ("forest_lowfp/evaluation/per_image.csv", "udcp_rgb565"),
    ),
    "forest_original": (
        ("forest_comparison/evaluation/per_image.csv", "pixel_forest_rgb565"),
        ("forest_lowfp/evaluation/per_image.csv", "forest_original"),
        ("forest_lowfp/threshold_control/evaluation/per_image.csv", "forest_original"),
    ),
    "forest_threshold_only": (
        ("forest_lowfp/evaluation/per_image.csv", "forest_threshold_only"),
        ("forest_lowfp/threshold_control/evaluation/per_image.csv", "forest_threshold_only"),
    ),
    "forest_lowfp": (("forest_lowfp/evaluation/per_image.csv", "forest_lowfp"),),
}
COUNT_KEYS = ("tp", "fp", "fn", "tn")
MANIFEST_INPUTS = ("data/manifest.json", "data/forest_training_manifest.json", "data/lowfp_training_manifest.json")
SOURCES = (
    "compare_dedup.py",
    "near_duplicates.py",
    "prepare_dedup_split.py",
    "compare_lowfp.py",
    "compare_pixel_forest.py",
    "forest_native.py",
    "pixel_forest.py",
    "udcp.py",
    "validate.py",
    "embedded/pixel_forest.c",
    "embedded/pixel_forest.h",
)
GOAL_FNR_INCREASE = 0.02
# 7% - 5% is 0.020000000000000004 in floating point; the goal means at most 2 points.
GOAL_TOLERANCE = 1e-12
BOOTSTRAP_DRAWS = 10000
SEED = 20261022


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_input(audit: dict) -> str:
    names = [name for name in audit["inputs"] if name not in MANIFEST_INPUTS]
    if len(names) != 1:
        raise ValueError("The audit must list exactly one review file among its inputs")
    return names[0]


def leak_outline(leaks: dict) -> dict:
    return {i: (leak["tier"], sorted(row["id"] for row in leak["neighbours"])) for i, leak in leaks.items()}


def load_audit(path: Path) -> tuple[dict, dict]:
    """Accept only a manifest whose inputs, code and leak lists all recompute from what it records."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not set(MANIFEST_INPUTS) <= set(audit["inputs"]) or set(audit.get("code_sha256", {})) != set(CODE):
        raise ValueError("The audit must record every input manifest and the screening code hashes")
    for name, digest in {**audit["inputs"], **audit["code_sha256"]}.items():
        if sha(resolve(name)) != digest:
            raise ValueError(f"Near-duplicate audit input or code changed: {name}")
    review_input(audit)
    original = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))["samples"]
    test_ids = {row["id"] for row in original if row["split"] == "test"}
    calibration_ids = {row["id"] for row in original if row["split"] == "calibration"}
    pairs = [pair for group in audit["screened_pairs"].values() for pair in group]
    derived = {
        "test_leaks": leaked_ids(pairs, test_ids),
        "calibration_leaks": leaked_ids(
            [pair for pair in pairs if not {pair["a"], pair["b"]} & test_ids], calibration_ids
        ),
    }
    for key, leaks in derived.items():
        if leak_outline(leaks) != leak_outline(audit[key]):
            raise ValueError(f"Audit {key} disagree with its screened pairs")
    leaks, ids = derived["test_leaks"], audit["test_subsets"]
    if any(leaks.get(i, {}).get("tier") != "near_duplicate" for i in PREVIOUSLY_REPORTED):
        raise ValueError("The audit does not classify every previously reported leak as a near duplicate")
    if (
        ids["all"] != [row["id"] for row in original if row["split"] == "test"]
        or ids["without_near_duplicates"] != [i for i in ids["all"] if leaks.get(i, {}).get("tier") != "near_duplicate"]
        or ids["without_same_scene"] != [i for i in ids["all"] if i not in leaks]
    ):
        raise ValueError("Audit TEST subsets disagree with its leak list")
    subsets = {
        "all": ids["all"],
        "without_reported_7": [i for i in ids["all"] if i not in PREVIOUSLY_REPORTED],
        "without_near_duplicates": ids["without_near_duplicates"],
        "without_same_scene": ids["without_same_scene"],
    }
    return audit, subsets


def load_forests() -> dict:
    """Locked models and compiled libraries, each checked against its recorded hashes."""
    forest_locked, forest_model = load_forest_locked(RESULTS_DIR / "forest_comparison")
    lowfp_locked, lowfp_model = load_lowfp_locked(RESULTS_DIR / "forest_lowfp")
    control = RESULTS_DIR / "forest_lowfp/threshold_control"
    export = json.loads((control / "MODEL_EXPORT.json").read_text(encoding="utf-8"))
    if (
        export["threshold_q"] != lowfp_locked["threshold_only_control"]["threshold_q"]
        or sha(control / "model/pixel_forest_model.h") != export["exported_header_sha256"]
        or sha(control / "model/pixel_forest_model.json") != forest_locked["model_sha256"]["pixel_forest_model.json"]
    ):
        raise ValueError("Threshold-only export differs from its frozen control")
    runs = {
        "forest_original": (RESULTS_DIR / "forest_comparison", forest_model, forest_locked["chosen"]["threshold_q"]),
        "forest_threshold_only": (control, forest_model, export["threshold_q"]),
        "forest_lowfp": (RESULTS_DIR / "forest_lowfp", lowfp_model, lowfp_locked["chosen"]["threshold_q"]),
    }
    forests = {}
    for name, (run, model, threshold) in runs.items():
        recorded = json.loads((run / "evaluation/summary.json").read_text(encoding="utf-8"))["dll_sha256"]
        if sha(run / "pixel_forest.dll") != recorded:
            raise ValueError(f"Compiled model changed since its evaluation: {run}")
        native = NativeForest(run / "pixel_forest.dll")
        check_native(native, model, threshold)
        forests[name] = (native, model, threshold)
    return forests


def recorded_counts() -> dict:
    tables = {}
    for sources in RECORDED.values():
        for path, _ in sources:
            if path not in tables:
                with resolve(path).open(encoding="utf-8") as handle:
                    tables[path] = {
                        (row["id"], row["method"]): tuple(int(row[key]) for key in COUNT_KEYS)
                        for row in csv.DictReader(handle)
                    }
    return {method: [(tables[path], name) for path, name in sources] for method, sources in RECORDED.items()}


def predict(rgb: np.ndarray, udcp: dict, forests: dict) -> dict:
    score = udcp_score(rgb, udcp["window"])[0]
    quantized = udcp_score(rgb565_roundtrip(rgb), udcp["window"])[0]
    predictions = {
        "udcp_fixed": score >= udcp["threshold_q"],
        "udcp_otsu": score >= otsu_threshold(score),
        "udcp_otsu_rgb565": quantized >= otsu_threshold(quantized),
    }
    for name, (native, model, threshold) in forests.items():
        scores, masks = native.predict(rgb)
        np.testing.assert_array_equal(scores, score_image(rgb, model))
        np.testing.assert_array_equal(masks, scores >= threshold)
        predictions[name] = masks.astype(bool)
    return predictions


def summarise(rows: list[dict], ids: list[str]) -> dict:
    selected = set(ids)
    result = {}
    for method in METHODS:
        chosen = [row for row in rows if row["method"] == method and row["id"] in selected]
        m = metrics(np.sum([[row[key] for key in COUNT_KEYS] for row in chosen], axis=0))
        result[method] = {
            **m,
            "fnr": 1 - m["foreground_recall"],
            "images": len(chosen),
            "macro_image_miou": float(np.mean([row["miou"] for row in chosen])),
        }
    return result


def rates(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp, fp, fn, tn = np.moveaxis(counts, -1, 0)
    return fn / (tp + fn), fp / (fp + tn)


def goal(fnr_change, fpr_change):
    return (fnr_change <= GOAL_FNR_INCREASE + GOAL_TOLERANCE) & (fpr_change < 0)


def paired(rows: list[dict], ids: list[str], before: str, after: str) -> dict:
    """Pooled FNR/FPR changes, with an image-level bootstrap of the same pooled statistics."""
    by_key = {(row["id"], row["method"]): row for row in rows}
    counts = np.array(
        [[[by_key[(i, m)][key] for key in COUNT_KEYS] for m in (before, after)] for i in ids], dtype=np.int64
    )
    fnr, fpr = rates(counts.sum(axis=0))
    weights = np.random.default_rng(SEED).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=BOOTSTRAP_DRAWS)
    boot_fnr, boot_fpr = rates((weights @ counts.reshape(len(ids), -1)).reshape(BOOTSTRAP_DRAWS, 2, 4))
    boot_fnr_change, boot_fpr_change = boot_fnr[:, 1] - boot_fnr[:, 0], boot_fpr[:, 1] - boot_fpr[:, 0]
    changes = np.array([by_key[(i, after)]["miou"] - by_key[(i, before)]["miou"] for i in ids])
    return {
        "fnr_percentage_points": float(fnr[1] - fnr[0]) * 100,
        "fpr_percentage_points": float(fpr[1] - fpr[0]) * 100,
        "goal_passed": bool(goal(fnr[1] - fnr[0], fpr[1] - fpr[0])),
        "bootstrap": {
            "unit": "image",
            "draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
            "fnr_percentage_points_95ci": [float(v) for v in np.percentile(boot_fnr_change * 100, [2.5, 97.5])],
            "fpr_percentage_points_95ci": [float(v) for v in np.percentile(boot_fpr_change * 100, [2.5, 97.5])],
            "goal_probability": float(goal(boot_fnr_change, boot_fpr_change).mean()),
        },
        "paired_images": {
            "improved": int((changes > 1e-12).sum()),
            "regressed": int((changes < -1e-12).sum()),
            "tied": int((np.abs(changes) <= 1e-12).sum()),
        },
    }


def leak_sheet(audit: dict, rows_by_id: dict, path: Path) -> None:
    leaks = sorted(audit["test_leaks"].items(), key=lambda item: (item[1]["tier"] != "near_duplicate", item[0]))
    sheet = Image.new("RGB", (640, 250 * len(leaks)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (sample_id, leak) in enumerate(leaks):
        neighbour = leak["neighbours"][0]
        for column, key in enumerate((sample_id, neighbour["id"])):
            with Image.open(REPO_ROOT / rows_by_id[key]["image"]) as image:
                sheet.paste(ImageOps.contain(image.convert("RGB"), (310, 220)), (column * 325, index * 250 + 26))
        draw.text(
            (4, index * 250 + 4),
            f"TEST {sample_id}  |  {neighbour['split']} {neighbour['id']}  |  {leak['tier']}"
            f", hash bits {neighbour['hash_bits']}",
            fill="black",
            font=font(14),
        )
    sheet.save(path, quality=88)


def change_cell(change: dict) -> str:
    low, high = change["bootstrap"]["fnr_percentage_points_95ci"]
    return f"{change['fnr_percentage_points']:+.2f}（{low:+.2f}, {high:+.2f}）"


def write_report(output: Path, summary: dict, audit: dict, manifest_path: Path) -> None:
    results, leaks = summary["subsets"], audit["test_leaks"]
    keyed = ("udcp_otsu_rgb565", "forest_original", "forest_threshold_only", "forest_lowfp")
    review = review_input(audit)
    tiers = {tier: sum(leak["tier"] == tier for leak in leaks.values()) for tier in ("near_duplicate", "same_scene")}
    calibration = {
        tier: sum(leak["tier"] == tier for leak in audit["calibration_leaks"].values())
        for tier in ("near_duplicate", "same_scene")
    }
    bands = leak_bands(leaks)
    extra = sorted(i for i, leak in leaks.items() if leak["tier"] == "near_duplicate" and i not in PREVIOUSLY_REPORTED)
    tentative = sorted(
        i for i, leak in leaks.items() if all("lower confidence" in row["note"] for row in leak["neighbours"])
    )
    strict = summary["subset_ids"]["without_same_scene"]

    def link(path):
        try:
            return Path(os.path.relpath(path, output)).as_posix()
        except ValueError:
            return Path(path).resolve().as_posix()

    def with_neighbour(condition):
        return sum(any(condition(row) for row in leak["neighbours"]) for leak in leaks.values())

    lines = [
        "# 去除测试集泄漏后的复评",
        "",
        "同一批锁定模型、阈值与已编译库，在去掉与标定/训练图近重复或同场景的 TEST 图后重新计分；没有重新训练或调参。"
        "排除名单只依据图像内容（感知哈希、颜色直方图和目视复核），未使用标注或逐图得分。",
        "",
        "## 推荐方案（阈值 57 → 71）",
        "",
        "| 测试子集 | 张数 | 漏检率 57 → 71 | 误检率 57 → 71 | 漏检增加（95% 区间） | 达标概率 | 逐图 改善/退步/持平 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in SUBSETS.items():
        before, after = (results[name]["methods"][key] for key in ("forest_original", "forest_threshold_only"))
        change = results[name]["threshold_71_vs_57"]
        images = change["paired_images"]
        lines.append(
            f"| {label} | {after['images']} | {before['fnr']:.2%} → {after['fnr']:.2%} | "
            f"{before['water_false_positive_rate']:.2%} → {after['water_false_positive_rate']:.2%} | "
            f"{change_cell(change)} pp | {change['bootstrap']['goal_probability']:.0%} | "
            f"{images['improved']}/{images['regressed']}/{images['tied']} |"
        )
    lines.extend(
        [
            "",
            "预设目标：误检率下降，且漏检率增加不超过 2 个百分点。区间与达标概率来自按图片重抽样的自助法"
            f"（{BOOTSTRAP_DRAWS} 次，种子 {SEED}）；漏检增加接近 2 个百分点，单次点估计的“达到/未达到”不稳定。",
            "",
            "## 各方法两类全局 mIoU",
            "",
            "| 方法 | "
            + " | ".join(f"{label}（{len(summary['subset_ids'][name])} 张）" for name, label in SUBSETS.items())
            + " |",
            "|---|" + "---:|" * len(SUBSETS),
        ]
    )
    for method in METHODS:
        lines.append(
            f"| {LABELS[method]} | "
            + " | ".join(f"{results[name]['methods'][method]['miou']:.4f}" for name in SUBSETS)
            + " |"
        )
    lines.extend(
        [
            "",
            f"## 去掉已发现泄漏后（{len(strict)} 张）",
            "",
            "| 方法 | 漏检 FNR | 水体误检 FPR | 两类全局 mIoU | 逐图平均 mIoU |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in keyed:
        m = results["without_same_scene"]["methods"][method]
        lines.append(
            f"| {LABELS[method]} | {m['fnr']:.2%} | {m['water_false_positive_rate']:.2%} | {m['miou']:.4f} | {m['macro_image_miou']:.4f} |"
        )
    lowfp = results["without_same_scene"]["lowfp_vs_original"]
    lines.extend(
        [
            "",
            f"16×8 主实验相对原森林：漏检 {change_cell(lowfp)} 个百分点、误检 {lowfp['fpr_percentage_points']:+.2f} 个百分点，"
            f"达标概率 {lowfp['bootstrap']['goal_probability']:.0%}。",
            "",
            "## 排除的 TEST 图",
            "",
            f"近重复 {tiers['near_duplicate']} 张：同一照片的重编码、缩放、重新裁剪，或相邻视频帧。"
            f"同场景 {tiers['same_scene']} 张：同一段视频或同一次拍摄的不同时刻。先前报告的 7 张均属近重复；"
            + (f"完整筛查另发现 {'、'.join(extra)}。" if extra else "完整筛查没有发现其他近重复。")
            + (
                f"其中 {len(tentative)} 张（{'、'.join(tentative)}）只有低置信的同场景判断，按保守原则排除。"
                if tentative
                else ""
            ),
            "",
            "| TEST 图 | 类别 | 最接近的邻居 | 所在集合 | 哈希距离 | 颜色重合 | 说明 | 在原 512 张训练集 |",
            "|---|---|---|---|---:|---:|---|---|",
        ]
    )
    for sample_id, leak in sorted(leaks.items(), key=lambda item: (item[1]["tier"] != "near_duplicate", item[0])):
        n = leak["neighbours"][0]
        own_training = any(row["in_first_512_training"] for row in leak["neighbours"])
        lines.append(
            f"| {sample_id} | {leak['tier']} | {n['id']} | {n['split']} | {n['hash_bits']} | {n['colour_intersection']:.2f} | "
            f"{n['note'] or '哈希判定'} | {'是' if own_training else '否'} |"
        )
    lines.extend(
        [
            "",
            f"最后一列指是否有邻居在训练原森林（阈值 57/71）的 512 张图中"
            f"（{with_neighbour(lambda row: row['in_first_512_training'])}/{len(leaks)} 张）；"
            f"{with_neighbour(lambda row: row['split'] == 'train')}/{len(leaks)} 张至少有一个邻居在 16×8 森林的 1395 张训练集中，"
            f"{with_neighbour(lambda row: row['split'] == 'calibration')} 张另有标定集邻居。",
            "",
            "## 筛查深度与残留风险",
            "",
            "筛查条件：哈希距离不超过 260/1024，或颜色直方图重合不低于 0.60（TEST 配对）/0.70（标定配对），不再设哈希上限。"
            "各泄漏 TEST 图按最先能找到它的条件计数："
            + "，".join(
                f"{label} {bands[key]} 张"
                for key, label in (
                    ("hash", "哈希"),
                    ("0.75+", "颜色 ≥0.75"),
                    ("0.70-0.75", "0.70–0.75"),
                    ("0.65-0.70", "0.65–0.70"),
                    ("0.60-0.65", "0.60–0.65"),
                )
            )
            + "。"
            "每降一档新增的泄漏图逐渐减少但没有归零，0.60 以下很可能仍有同段视频的帧。",
            "",
            "## 尚未解决的问题",
            "",
            f"- 80 张标定图中有 {calibration['near_duplicate'] + calibration['same_scene']} 张在训练集里有邻居（近重复 "
            f"{calibration['near_duplicate']}、同场景 {calibration['same_scene']}）。阈值 57、71、72 都在这批标定图上选出，本次没有重新标定。",
            "- SUIM 没有视频或拍摄批次编号，筛查只能找出画面或颜色接近的帧；本表最后一个子集是“去掉已发现泄漏”，不能证明场景级独立。",
            f"- 哈希无法判定的配对均已目视复核；复核人、日期和逐条决定见 [{review}]({link(resolve(review))})，可修改后生成新的审计清单。",
            "- 真正独立的检验仍需新的盲测集，最好是自己的 OV2640 实拍数据，并按拍摄批次划分。",
            "",
            "## 一致性核查",
            "",
            f"- 全部 {summary['test_images']} 张图、{len(METHODS)} 种方法的逐图混淆计数与此前各轮记录完全一致；"
            "三个已编译库的 C 分数和掩膜与 Python 参考逐像素一致。",
            "- 模型、阈值、导出头文件、DLL、数据清单、筛查代码与复核记录均按记录的哈希校验；泄漏名单由审计中的配对决定重新推导。",
            "",
            f"[审计清单]({link(manifest_path)}) · [目视复核记录]({link(resolve(review))}) · "
            "[逐图结果](per_image.csv) · [结果 JSON](summary.json)",
            "",
            "![排除的 TEST 图（左）与最接近的标定/训练图（右）](leak_pairs.jpg)",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run(output: Path, manifest_path: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite evaluation evidence; choose a fresh --output")
    audit, subsets = load_audit(manifest_path)
    forests = load_forests()
    udcp = json.loads((RESULTS_DIR / "validation_results/locked_parameters.json").read_text(encoding="utf-8"))["udcp"]
    expected = recorded_counts()
    original = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    samples = [row for row in original["samples"] if row["split"] == "test"]
    tiers = {sample_id: leak["tier"] for sample_id, leak in audit["test_leaks"].items()}
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sample in enumerate(samples, 1):
        rgb, labels = load_sample(sample)
        for method, predicted in predict(rgb, udcp, forests).items():
            counts = confusion(at_native_resolution(predicted.astype(np.uint8), labels).astype(bool), labels != 0)
            for table, recorded in expected[method]:
                if table[(sample["id"], recorded)] != counts:
                    raise AssertionError(f"{method} on {sample['id']} no longer reproduces {recorded}")
            rows.append(
                {"id": sample["id"], "leak_tier": tiers.get(sample["id"], ""), "method": method, **metrics(counts)}
            )
        if index % 20 == 0:
            print(f"Re-evaluated {index}/{len(samples)}; recorded counts reproduced", flush=True)
    results = {}
    for name, ids in subsets.items():
        methods = summarise(rows, ids)
        results[name] = {
            "images": len(ids),
            "methods": methods,
            "threshold_71_vs_57": paired(rows, ids, "forest_original", "forest_threshold_only"),
            "lowfp_vs_original": paired(rows, ids, "forest_original", "forest_lowfp"),
            "forest_57_minus_udcp_otsu_rgb565_miou": methods["forest_original"]["miou"]
            - methods["udcp_otsu_rgb565"]["miou"],
        }
    summary = {
        "task": "Same locked SUIM water/non-water comparison on TEST subsets without detected train/calibration leaks",
        "test_images": len(samples),
        "subset_ids": subsets,
        "excluded": {name: sorted(set(subsets["all"]) - set(ids)) for name, ids in subsets.items()},
        "subsets": results,
        "policy": "No retraining or threshold change; exclusions use image content only, not labels or scores.",
        "reproduced_recorded_counts": {
            method: [f"{path}:{name}" for path, name in RECORDED[method]] for method in METHODS
        },
        "dedup_manifest_sha256": sha(manifest_path),
        "source_sha256": {name: sha(resolve(name)) for name in SOURCES},
    }
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    leak_sheet(
        audit,
        {row["id"]: row for row in original["samples"]}
        | {
            row["id"]: row
            for row in json.loads((DATA_DIR / "lowfp_training_manifest.json").read_text(encoding="utf-8"))["samples"]
        },
        output / "leak_pairs.jpg",
    )
    write_report(output, summary, audit, manifest_path)
    # Written last, so a complete summary.json marks a finished run.
    dump_json(output / "summary.json", summary)
    for name in subsets:
        change = results[name]["threshold_71_vs_57"]
        print(
            f"{name}: {results[name]['images']} images; threshold 71 vs 57 FNR {change_cell(change)}pp, "
            f"FPR {change['fpr_percentage_points']:+.2f}pp, P(goal) {change['bootstrap']['goal_probability']:.2f}",
            flush=True,
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "evaluation_dedup")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "dedup_manifest.json")
    args = parser.parse_args()
    run(args.output.resolve(), args.manifest.resolve())
