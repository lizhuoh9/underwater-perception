"""Evaluate a locked low-FP forest and its threshold-only control on the same TEST."""

import argparse
import csv
import hashlib
import html
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from uwseg.compare_pixel_forest import check_native
from uwseg.forest_native import NativeForest
from uwseg.paths import DATA_DIR, RESULTS_DIR, resolve
from uwseg.pixel_forest import extract_features, score_image, validate_model
from uwseg.udcp import confusion, metrics, otsu_threshold, rgb565_roundtrip, udcp_score
from uwseg.validate import CLASS_NAMES, at_native_resolution, dump_json, font, load_sample, overlay

METHODS = ("udcp_rgb565", "forest_original", "forest_threshold_only", "forest_lowfp")
LABELS = ("UDCP-Otsu RGB565", "上一版森林", "上一版森林只调阈值", "本轮优化森林")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_locked(run: Path) -> tuple[dict, dict]:
    locked = json.loads((run / "LOCKED.json").read_text(encoding="utf-8"))
    if sha(run / "experiment_plan.json") != locked["plan_sha256"]:
        raise ValueError("Experiment plan changed after locking")
    plan = json.loads((run / "experiment_plan.json").read_text(encoding="utf-8"))
    for name, digest in plan["manifest_sha256"].items():
        if sha(resolve(name)) != digest:
            raise ValueError(f"Manifest changed: {name}")
    if sha(RESULTS_DIR / "forest_comparison/LOCKED.json") != plan["original_locked_sha256"]:
        raise ValueError("Original locked run changed")
    for name, digest in locked["model_sha256"].items():
        if sha(run / "model" / name) != digest:
            raise ValueError(f"Model changed: {name}")
    for name, digest in locked["source_sha256"].items():
        if sha(resolve(name)) != digest:
            raise ValueError(f"Source changed since locking: {name}")
    model = json.loads((run / "model/pixel_forest_model.json").read_text(encoding="utf-8"))
    validate_model(model)
    return locked, model


def panel(
    rgb: np.ndarray,
    labels: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    sample_id: str,
    old_miou: float,
    new_miou: float,
) -> Image.Image:
    truth = at_native_resolution((labels != 0).astype(np.uint8), np.empty(rgb.shape[:2])).astype(bool)
    views = [
        Image.fromarray(rgb),
        overlay(rgb, truth, (35, 210, 90)),
        overlay(rgb, old, (255, 110, 30)),
        overlay(rgb, new, (180, 90, 255)),
    ]
    titles = ["Input RGB565", "GT: non-water", "Previous forest", "Low-FP forest"]
    result = Image.new("RGB", (1120, 250), "white")
    draw = ImageDraw.Draw(result)
    draw.text(
        (8, 3),
        f"{sample_id} | native mIoU: previous {old_miou:.3f}, optimized {new_miou:.3f}",
        fill="black",
        font=font(17),
    )
    for index, (image, title) in enumerate(zip(views, titles)):
        image = image.resize((round(image.width * 1.6), round(image.height * 1.6)), Image.Resampling.NEAREST)
        result.paste(image, (index * 280 + (280 - image.width) // 2, 52))
        draw.text((index * 280 + 7, 29), title, fill="black", font=font(16))
    return result


def write_outputs(output: Path, summary: dict, rows: list[dict], image_stats: list[dict]) -> None:
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    methods = summary["methods"]
    md = ["| 方法 | 像素漏检 FNR ↓ | 水体误检 FPR ↓ | 两类全局 mIoU ↑ | 逐图平均 mIoU ↑ |", "|---|---:|---:|---:|---:|"]
    html_rows = []
    for key, label in zip(METHODS, LABELS):
        m = methods[key]
        md.append(
            f"| {label} | {m['fnr']:.2%} | {m['water_false_positive_rate']:.2%} | {m['miou']:.4f} | {m['macro_image_miou']:.4f} |"
        )
        html_rows.append(
            f"<tr><td>{label}</td><td>{m['fnr']:.2%}</td><td>{m['water_false_positive_rate']:.2%}</td><td>{m['miou']:.4f}</td></tr>"
        )
    ordered = sorted(image_stats, key=lambda r: (r["delta_miou"], r["id"]))
    improved = [r for r in ordered if r["delta_miou"] > 0]
    # Representative positive change and worst regression are visible together.
    selected = [improved[len(improved) // 2]["id"] if improved else ordered[-1]["id"], ordered[0]["id"]]
    examples = Image.new("RGB", (1120, 500), "white")
    for index, sample_id in enumerate(selected):
        with Image.open(output / "overlays" / f"{sample_id}.jpg") as image:
            examples.paste(image, (0, 250 * index))
    examples.save(output / "examples.jpg", quality=93)
    # Explicitly revisit the previous round's three worst regressions, without tuning.
    old_selection = json.loads(
        (RESULTS_DIR / "forest_comparison/evaluation/visual_selection.json").read_text(encoding="utf-8")
    )
    revisited = [r["id"] for r in old_selection["worst_forest_minus_udcp_deltas"]]
    canvas = Image.new("RGB", (1120, 750), "white")
    for index, sample_id in enumerate(revisited):
        with Image.open(output / "overlays" / f"{sample_id}.jpg") as image:
            canvas.paste(image, (0, index * 250))
    canvas.save(output / "previous_failures.jpg", quality=93)
    dump_json(
        output / "visual_selection.json",
        {
            "examples": selected,
            "rule": "Median positive per-image mIoU improvement, followed by worst regression; evaluation only",
            "previous_failures": revisited,
        },
    )
    figures = []
    for entry in ordered:
        sample_id = html.escape(entry["id"], quote=True)
        figures.append(
            f'<figure><img loading="lazy" src="overlays/{sample_id}.jpg" alt="{sample_id}: input,GT,previous,optimized"><figcaption>{sample_id}: delta mIoU={entry["delta_miou"]:+.3f}</figcaption></figure>'
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>水下低误检森林优化</title>
<style>body{font:16px/1.6 system-ui;max-width:1160px;margin:24px auto;padding:0 16px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%;height:auto}figure{margin:28px 0}figcaption{color:#555}</style>
<h1>低误检优化：同一批110张图片</h1><p>1395张训练、原80张标定。选择标定漏检率最多增加2个百分点的最低误检点，再冻结模型和阈值。测试是多轮复用比较基准，不是新盲测。按mIoU变化从退步到改善显示全部图片。绿色GT，橙色上一版，紫色本轮。</p>
<p><a href="RESULTS.md">报告</a> · <a href="per_image.csv">逐图CSV</a> · <a href="summary.json">结果JSON</a> · <a href="../LOCKED.json">锁定参数</a></p><table><tr><th>方法</th><th>漏检率</th><th>误检率</th><th>mIoU</th></tr>"""
    (output / "gallery.html").write_text(
        page + "".join(html_rows) + "</table>" + "".join(figures) + "</html>", encoding="utf-8"
    )
    d = summary["delta_vs_previous"]
    outcome = "本轮达到预设目标" if summary["test_goal_passed"] else "本轮未达到全部预设目标，不能宣称优化成功"
    chosen = summary["chosen"]
    report = [
        "# 低误检优化：冻结模型后的同集比较",
        "",
        outcome + "：误检必须下降，测试漏检率增加不超过 2 个百分点。",
        "",
        *md,
        "",
        f"相对上一版森林：误检率变化 {d['fpr_percentage_points']:+.2f} 个百分点，漏检率变化 {d['fnr_percentage_points']:+.2f} 个百分点。",
        f"逐图 mIoU 改善 {summary['paired_images']['improved']} 张、退步 {summary['paired_images']['regressed']} 张、持平 {summary['paired_images']['tied']} 张。",
        "",
        "## 优化内容与验证约束",
        "",
        "- 保留 16 个整数颜色/纹理特征与原 C99 推理；增加训练图到全部 1395 张可用的不重复 train_val 图片。原 512 张是其子集，额外增加 883 张。标定仍为原 80 张，测试仍为原 110 张。",
        "- 图像文件字节 SHA256 三组互斥；未排除重压缩同图或相关场景。TEST 已经被前几轮查看，这是复用基准，不是新盲测。",
        "- 固定五个候选：旧模型只调阈值、上一轮 16×8 候选、全量均匀训练 sqrt/8 特征、另加训练集难水体负样本。难负样本只由训练图标注和旧模型预测生成。",
        "- 所有候选在标定集上选择满足原森林 FNR+0.02 的最低 FPR 点，再选标定 FPR 最低模型；整个计划在训练前保存，主模型和阈值对照在加载 TEST 前锁定。本轮测试后没有调参。",
        f"- 标定原漏检率 {summary['calibration']['original_fnr']:.2%}，容许上限 {summary['calibration']['fnr_cap']:.2%}；最终选择 {chosen['name']}、{chosen['trees']} 棵树、最大深度 {chosen['max_depth']}、阈值 {chosen['threshold_q']}/255。",
        "- FNR=FN/(TP+FN)，FPR=FP/(FP+TN)；全局指标累加所有图片原尺寸像素。另报告每图等权 mIoU，避免混淆两种平均。输入 RGB565，保持比例缩至 160×120 范围，预测最近邻放回原尺寸。",
        "- 优化后的 C 特征/分数/掩膜在全部 TEST 图上逐像素匹配 Python 参考；上一版森林及 UDCP 的混淆计数与旧结果精确一致。",
        "",
        "## MCU",
        "",
        f"- 模型数组 {summary['hardware']['model_array_bytes']:,} 字节；160×120 输入、分数、掩膜和工作区仍合计 298,404 字节（291.41 KiB）。系统、DMA、栈、额外相机缓冲另计，建议带 PSRAM。",
        "- 无动态 C 内存分配、无浮点推理。当前最大预算 16 棵树、深度 8，树比较上界为每像素 128 次，旧模型为 72 次；无法据此直接换算板端帧率。",
        "- [接口、内存及字节序说明](../../../embedded/README.md)。更换模型时使用本轮 model/pixel_forest_model.h。",
        "",
        "## 仍然存在的限制",
        "",
        "这仍是水体／非水体像素分割，未验证真实碰撞距离；水面反光、悬浮物、相机曝光和实际水体差异仍可能导致误检。尚无 OV2640 水池采集、MCU 帧率或运动闭环测试。",
        "",
        "[完整110张对比](gallery.html) · [逐图结果](per_image.csv) · [锁定参数](../LOCKED.json) · [训练计划](../experiment_plan.json)",
        "",
        "![代表性改善与最大退步](examples.jpg)",
        "",
        "上一轮最大退步的三张图：",
        "",
        "![重新查看上一轮失败图](previous_failures.jpg)",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")


def run_comparison(run: Path) -> dict:
    output = run / "evaluation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite evaluation evidence")
    locked, model = load_locked(run)
    native = NativeForest(run / "pixel_forest.dll")
    old_native = NativeForest(RESULTS_DIR / "forest_comparison/pixel_forest.dll")
    check_native(native, model, locked["chosen"]["threshold_q"])
    original = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    samples = [row for row in original["samples"] if row["split"] == "test"]
    if len(samples) != 110:
        raise ValueError("Expected original 110 test images")
    previous = json.loads((RESULTS_DIR / "forest_comparison/evaluation/summary.json").read_text(encoding="utf-8"))
    previous_rows = list(
        csv.DictReader((RESULTS_DIR / "forest_comparison/evaluation/per_image.csv").open(encoding="utf-8"))
    )
    if {row["id"] for row in previous_rows} != {row["id"] for row in samples}:
        raise ValueError("Previous run evaluated different image IDs")
    old_locked = json.loads((RESULTS_DIR / "forest_comparison/LOCKED.json").read_text(encoding="utf-8"))
    old_model = json.loads(
        (RESULTS_DIR / "forest_comparison/model/pixel_forest_model.json").read_text(encoding="utf-8")
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir()
    counts = {method: np.zeros(4, dtype=np.int64) for method in METHODS}
    classes = {method: np.zeros((8, 2), dtype=np.int64) for method in METHODS}
    rows, image_stats, times = [], [], []
    for index, sample in enumerate(samples, 1):
        rgb, labels = load_sample(sample)
        np.testing.assert_array_equal(native.features(rgb), extract_features(rgb))
        start = time.perf_counter()
        new_scores, new_masks = native.predict(rgb)
        times.append((time.perf_counter() - start) * 1000)
        np.testing.assert_array_equal(new_scores, score_image(rgb, model))
        np.testing.assert_array_equal(new_masks, new_scores >= locked["chosen"]["threshold_q"])
        old_scores, old_masks = old_native.predict(rgb)
        np.testing.assert_array_equal(old_scores, score_image(rgb, old_model))
        np.testing.assert_array_equal(old_masks, old_scores >= old_locked["chosen"]["threshold_q"])
        quantized = rgb565_roundtrip(rgb)
        udcp = udcp_score(quantized, 9)[0]
        predictions = {
            "udcp_rgb565": udcp >= otsu_threshold(udcp),
            "forest_original": old_masks.astype(bool),
            "forest_threshold_only": old_scores >= locked["threshold_only_control"]["threshold_q"],
            "forest_lowfp": new_masks.astype(bool),
        }
        image_metrics = {}
        for method in METHODS:
            prediction = at_native_resolution(predictions[method].astype(np.uint8), labels).astype(bool)
            current = confusion(prediction, labels != 0)
            counts[method] += current
            current_metrics = metrics(current)
            rows.append({"id": sample["id"], "method": method, **current_metrics})
            image_metrics[method] = current_metrics
            for class_id in range(8):
                selected = labels == class_id
                classes[method][class_id] += [np.count_nonzero(prediction[selected]), selected.sum()]
        old_miou, new_miou = image_metrics["forest_original"]["miou"], image_metrics["forest_lowfp"]["miou"]
        image_stats.append({"id": sample["id"], "delta_miou": new_miou - old_miou})
        panel(
            quantized,
            labels,
            predictions["forest_original"],
            predictions["forest_lowfp"],
            sample["id"],
            old_miou,
            new_miou,
        ).save(output / "overlays" / f"{sample['id']}.jpg", quality=92)
        if index % 20 == 0:
            print(f"Compared {index}/110; exact C/Python parity", flush=True)
    for method, previous_method in (("forest_original", "pixel_forest_rgb565"), ("udcp_rgb565", "udcp_otsu_rgb565")):
        for index, key in enumerate(("tp", "fp", "fn", "tn")):
            if int(counts[method][index]) != previous["methods"][previous_method][key]:
                raise AssertionError("Previous baseline counts changed")
    methods = {}
    for method in METHODS:
        m = metrics(counts[method])
        methods[method] = {
            **m,
            "fnr": 1 - m["foreground_recall"],
            "macro_image_miou": float(np.mean([r["miou"] for r in rows if r["method"] == method])),
            "per_class_foreground_fraction": {
                name: float(hits / total) if total else None
                for name, (hits, total) in zip(CLASS_NAMES, classes[method])
            },
        }
    before, after = methods["forest_original"], methods["forest_lowfp"]
    delta_fnr, delta_fpr = (
        after["fnr"] - before["fnr"],
        after["water_false_positive_rate"] - before["water_false_positive_rate"],
    )
    changes = np.array([row["delta_miou"] for row in image_stats])
    summary = {
        "task": "Same110 SUIM non-water/water pixel comparison; no collision distance truth",
        "test_images": len(samples),
        "chosen": locked["chosen"],
        "methods": methods,
        "delta_vs_previous": {"fnr_percentage_points": delta_fnr * 100, "fpr_percentage_points": delta_fpr * 100},
        "test_goal_passed": bool(delta_fnr <= 0.02 and delta_fpr < 0),
        "calibration": {"original_fnr": locked["original_calibration_fnr"], "fnr_cap": locked["calibration_fnr_cap"]},
        "paired_images": {
            "improved": int((changes > 1e-12).sum()),
            "regressed": int((changes < -1e-12).sum()),
            "tied": int((abs(changes) <= 1e-12).sum()),
        },
        "parity": {
            "test_images": len(samples),
            "feature_mismatches": 0,
            "score_mismatches": 0,
            "mask_mismatches": 0,
            "previous_baseline_counts_exact": True,
        },
        "hardware": {
            "model_array_bytes": locked["model_array_bytes"],
            "workspace_bytes_160x120": native.library.pf_workspace_bytes(160, 120),
            "total_image_buffers_bytes_160x120": native.library.pf_workspace_bytes(160, 120) + 76800,
            "board_fps_measured": False,
        },
        "desktop_only_timing_ms": {
            "median": float(np.median(times)),
            "p95": float(np.percentile(times, 95)),
            "max": max(times),
            "includes": "ctypes,packing,allocation,C features/prediction/median; no camera",
        },
        "locked_sha256": sha(run / "LOCKED.json"),
        "dll_sha256": sha(run / "pixel_forest.dll"),
        "source_sha256": {
            name: sha(resolve(name))
            for name in (
                "compare_lowfp.py",
                "compare_pixel_forest.py",
                "forest_native.py",
                "pixel_forest.py",
                "embedded/pixel_forest.c",
                "embedded/pixel_forest.h",
                "udcp.py",
                "validate.py",
            )
        },
    }
    dump_json(output / "summary.json", summary)
    write_outputs(output, summary, rows, image_stats)
    print(
        json.dumps({"goal_passed": summary["test_goal_passed"], "methods": methods}, ensure_ascii=False, indent=2),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RESULTS_DIR / "forest_lowfp")
    run_comparison(parser.parse_args().run.resolve())
