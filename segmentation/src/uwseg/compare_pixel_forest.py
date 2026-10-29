"""Frozen comparison of integer pixel forest and UDCP on the original 110 images."""

import argparse
import csv
import hashlib
import html
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from uwseg.forest_native import NativeForest
from uwseg.paths import DATA_DIR, RESULTS_DIR, resolve
from uwseg.pixel_forest import extract_features, predict_features, score_image, validate_model
from uwseg.udcp import confusion, metrics, otsu_threshold, rgb565_roundtrip, udcp_score
from uwseg.validate import CLASS_NAMES, at_native_resolution, dump_json, font, load_sample, overlay

METHODS = ("udcp_otsu", "udcp_otsu_rgb565", "pixel_forest_rgb565")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_locked(run):
    locked = json.loads((run / "LOCKED.json").read_text(encoding="utf-8"))
    if sha(run / "experiment_plan.json") != locked["plan_sha256"]:
        raise ValueError("Experiment plan changed after locking")
    for name, digest in locked["model_sha256"].items():
        if sha(run / "model" / name) != digest:
            raise ValueError(f"Locked model changed: {name}")
    for name, digest in locked["source_sha256"].items():
        if sha(resolve(name)) != digest:
            raise ValueError(f"Training/feature source changed after locking: {name}")
    plan = json.loads((run / "experiment_plan.json").read_text(encoding="utf-8"))
    for name, key in (
        ("manifest.json", "baseline_manifest_sha256"),
        ("forest_training_manifest.json", "training_manifest_sha256"),
    ):
        if sha(DATA_DIR / name) != plan[key]:
            raise ValueError(f"Dataset manifest changed: {name}")
    model = json.loads((run / "model/pixel_forest_model.json").read_text(encoding="utf-8"))
    validate_model(model)
    return locked, model


def check_native(native, model, threshold):
    """Known boundaries, degenerate dimensions, and seeded full-frame parity."""
    rng = np.random.default_rng(20261022)
    images = [
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.full((1, 7, 3), 255, dtype=np.uint8),
        rng.integers(0, 256, (9, 1, 3), dtype=np.uint8),
        rng.integers(0, 256, (7, 11, 3), dtype=np.uint8),
        rng.integers(0, 256, (120, 160, 3), dtype=np.uint8),
    ]
    for image in images:
        features = extract_features(image)
        np.testing.assert_array_equal(native.features(image), features)
        np.testing.assert_array_equal(
            native.predict_rows(features.reshape(-1, 16)), predict_features(features.reshape(-1, 16), model)
        )
        scores, masks = native.predict(image)
        np.testing.assert_array_equal(scores, score_image(image, model))
        np.testing.assert_array_equal(masks, scores >= threshold)
    return len(images)


def make_panel(rgb, labels, predictions, sample_id, values):
    truth = at_native_resolution((labels != 0).astype(np.uint8), np.empty(rgb.shape[:2])).astype(bool)
    views = [
        Image.fromarray(rgb),
        overlay(rgb, truth, (35, 210, 90)),
        overlay(rgb, predictions["udcp_otsu_rgb565"], (180, 90, 255)),
        overlay(rgb, predictions["pixel_forest_rgb565"], (255, 110, 30)),
    ]
    titles = ["Input RGB565", "GT: non-water", "UDCP-Otsu RGB565", "Integer pixel forest"]
    panel = Image.new("RGB", (1120, 250), "white")
    draw = ImageDraw.Draw(panel)
    draw.text(
        (8, 3), f"{sample_id} | native mIoU: UDCP {values[0]:.3f}, forest {values[1]:.3f}", fill="black", font=font(17)
    )
    for index, (view, title) in enumerate(zip(views, titles)):
        view = view.resize((round(view.width * 1.6), round(view.height * 1.6)), Image.Resampling.NEAREST)
        panel.paste(view, (index * 280 + (280 - view.width) // 2, 52))
        draw.text((index * 280 + 7, 29), title, fill="black", font=font(16))
    return panel


def write_outputs(output, summary, rows, panels):
    with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline = summary["methods"]["udcp_otsu_rgb565"]
    forest = summary["methods"]["pixel_forest_rgb565"]
    deltas = {sample_id: values[1] - values[0] for sample_id, values in panels.items()}
    original = json.loads((RESULTS_DIR / "validation_results/contact_sheet_selection.json").read_text(encoding="utf-8"))
    preview_ids = [original[index]["id"] for index in (0, 4, 8)]
    regression_ids = sorted(deltas, key=lambda key: (deltas[key], key))[:3]
    for filename, selected in (("preview.jpg", preview_ids), ("regressions.jpg", regression_ids)):
        canvas = Image.new("RGB", (1120, 250 * len(selected)), "white")
        for index, sample_id in enumerate(selected):
            with Image.open(output / "overlays" / f"{sample_id}.jpg") as panel:
                canvas.paste(panel, (0, index * 250))
        canvas.save(output / filename, quality=93)
    dump_json(
        output / "visual_selection.json",
        {
            "preview": preview_ids,
            "preview_rule": "Original UDCP contact sheet indices 0,4,8; unchanged selection",
            "worst_forest_minus_udcp_deltas": [{"id": key, "delta_miou": deltas[key]} for key in regression_ids],
        },
    )
    table = []
    mdtable = ["| 方法 | 两类 mIoU | 非水体漏检率 FNR | 水体误检率 FPR | 非水体精确率 |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        m = summary["methods"][method]
        label = {
            "udcp_otsu": "原 UDCP-Otsu RGB888",
            "udcp_otsu_rgb565": "UDCP-Otsu RGB565",
            "pixel_forest_rgb565": "整数像素随机森林 RGB565",
        }[method]
        fnr = 1 - m["foreground_recall"]
        mdtable.append(
            f"| {label} | {m['miou']:.4f} | {fnr:.2%} | {m['water_false_positive_rate']:.2%} | {m['foreground_precision']:.2%} |"
        )
        table.append(
            f"<tr><td>{label}</td><td>{m['miou']:.4f}</td><td>{fnr:.2%}</td><td>{m['water_false_positive_rate']:.2%}</td></tr>"
        )
    figures = []
    for sample_id in sorted(deltas, key=lambda key: (deltas[key], key)):
        safe_id = html.escape(sample_id, quote=True)
        figures.append(
            f'<figure><img loading="lazy" src="overlays/{safe_id}.jpg" alt="{safe_id}: input, GT, UDCP, forest"><figcaption>{safe_id}: forest − UDCP mIoU = {deltas[sample_id]:+.3f}</figcaption></figure>'
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>水下像素随机森林对比</title>
<style>body{font:16px/1.6 system-ui;max-width:1160px;margin:24px auto;padding:0 16px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%;height:auto}figure{margin:28px 0}figcaption{color:#555}</style>
<h1>水下像素随机森林：同一批 110 张图片对比</h1><p>512 张训练、80 张标定，三者按图像 SHA256 排除完全重复。模型/阈值在加载测试图之前锁定。此测试集此前已用于 UDCP，属于复用比较基准；没有按本轮测试结果再调参。水体／非水体分割不是碰撞距离验证。</p>
<p>全部图按新方法减旧方法 mIoU 从差到好排列，先显示退步样本。绿色为 GT，紫色为 UDCP，橙色为森林。</p><p><a href="RESULTS.md">报告</a> · <a href="summary.json">结果 JSON</a> · <a href="per_image.csv">逐图 CSV</a> · <a href="../LOCKED.json">锁定参数</a></p><table><tr><th>方法</th><th>mIoU</th><th>漏检 FNR</th><th>误检 FPR</th></tr>"""
    (output / "gallery.html").write_text(
        page + "".join(table) + "</table>" + "".join(figures) + "</html>", encoding="utf-8"
    )
    fnr_old, fnr_new = 1 - baseline["foreground_recall"], 1 - forest["foreground_recall"]
    fpr_old, fpr_new = baseline["water_false_positive_rate"], forest["water_false_positive_rate"]
    report = [
        "# 整数像素随机森林与 UDCP 对比",
        "",
        "同一批 SUIM 110 张图片上的水体／非水体二分类；不是近距离碰撞真值、八类语义识别或 SLAM。",
        "",
        *mdtable,
        "",
        f"以相同 RGB565 输入比较，漏检率变化 {(fnr_new - fnr_old) * 100:+.2f} 个百分点，误检率变化 {(fpr_new - fpr_old) * 100:+.2f} 个百分点。",
        f"逐图 mIoU 改善 {summary['paired_images']['improved']} 张、退步 {summary['paired_images']['regressed']} 张、相同 {summary['paired_images']['tied']} 张；全图累计得分不能保证每张图都改善。",
        "",
        "## 实验约束",
        "",
        "- 512 张训练，另用原 80 张标定；完全排除训练/标定/测试间图像内容重复。缺乏场景序列信息，未做近重复或场景级独立性验证。",
        "- 每张训练图均匀抽 768 个像素，16 个整数颜色与局部纹理特征；随机森林只在 PC 训练。",
        "- 预声明 8×4、12×6、16×8 三种树数/深度组合。在标定集选择 FNR/FPR 均不高于 UDCP 的阈值，再最大化逐图 mIoU。没有根据本轮测试结果重新训练/调阈值。",
        "- 已锁定 12 棵树、最大深度 6、1496 节点、8 位叶概率、3×3 中值和阈值 57/255。",
        "- 复用此前已查看过的 110 张测试图，因此这是同集对比，不是新盲测集。输入保持比例缩至 160×120 范围，预测最近邻放回原始掩膜后计分。",
        "- 所有图片的 C 特征、分数、二值掩膜与 Python 整数参考逐像素完全一致；原 RGB888 UDCP 混淆计数也与之前保存结果完全一致。",
        "",
        "## MCU 移植",
        "",
        "- 提供无动态内存分配、无浮点运算的 C99 推理；PC C 库已实际运行全部测试。",
        f"- 模型数组 {summary['hardware']['model_array_bytes']:,} 字节；160×120 工作区 {summary['hardware']['workspace_bytes_160x120']:,} 字节，加输入、分数、掩膜共 {summary['hardware']['total_image_buffers_bytes_160x120']:,} 字节（{summary['hardware']['total_image_buffers_bytes_160x120'] / 1024:.2f} KiB）。不包含 SDK、DMA、栈和显示/通信缓冲，建议带 PSRAM 的板卡。",
        "- MCU 输出每像素二值掩膜；训练与指标计算、画廊生成留在 PC。接口、内存和字节序见 ../../../embedded/README.md。",
        "- 尚无 MCU + OV2640 实机采图、端到端 FPS、功耗、真实水池或机器人闭环验证。PC 耗时不换算 MCU 帧率。",
        "",
        "## 结果边界",
        "",
        "SUIM 中海底、珊瑚、鱼、潜水员等都算非水体，无论远近。残余误检与漏检仍需实测，颜色/纹理方法也会受水色、照明、悬浮物、相机自动曝光及白平衡变化影响。更换至 OV2640 和实际水体后应重新采集独立训练/标定/盲测样本。",
        "",
        "[完整 110 张对比](gallery.html) · [逐图指标](per_image.csv) · [锁定参数](../LOCKED.json) · [实验计划](../experiment_plan.json) · [原始基线](../../validation_results/RESULTS.md)",
        "",
        "![固定选择的原基线差/中/好样例](preview.jpg)",
        "",
        "以下为新方法相对 UDCP 退步最大的三个样例：",
        "",
        "![最大退步样例](regressions.jpg)",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")


def run_comparison(run):
    output = run / "evaluation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite evaluation evidence; use a fresh run")
    locked, model = load_locked(run)
    native = NativeForest(run / "pixel_forest.dll")
    synthetic_cases = check_native(native, model, locked["chosen"]["threshold_q"])
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    samples = [sample for sample in manifest["samples"] if sample["split"] == "test"]
    if len(samples) != 110:
        raise ValueError("Expected the original 110-image test set")
    previous = json.loads((RESULTS_DIR / "validation_results/summary.json").read_text(encoding="utf-8"))
    window = previous["config"]["udcp"]["window"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir()
    counts = {method: np.zeros(4, dtype=np.int64) for method in METHODS}
    per_class = {method: np.zeros((8, 2), dtype=np.int64) for method in METHODS}
    rows, panels, times = [], {}, []
    for index, sample in enumerate(samples):
        rgb, labels = load_sample(sample)
        rgb565 = rgb565_roundtrip(rgb)
        np.testing.assert_array_equal(native.features(rgb), extract_features(rgb))
        start = time.perf_counter()
        cscore, cmask = native.predict(rgb)
        times.append((time.perf_counter() - start) * 1000)
        reference = score_image(rgb, model)
        np.testing.assert_array_equal(cscore, reference)
        np.testing.assert_array_equal(cmask, reference >= locked["chosen"]["threshold_q"])
        predictions = {"pixel_forest_rgb565": cmask.astype(bool)}
        for method, image in (("udcp_otsu", rgb), ("udcp_otsu_rgb565", rgb565)):
            score = udcp_score(image, window)[0]
            predictions[method] = score >= otsu_threshold(score)
        values = {}
        for method in METHODS:
            prediction = at_native_resolution(predictions[method].astype(np.uint8), labels).astype(bool)
            current = confusion(prediction, labels != 0)
            counts[method] += current
            m = metrics(current)
            values[method] = m["miou"]
            rows.append({"id": sample["id"], "method": method, **m})
            for class_id in range(8):
                selected = labels == class_id
                per_class[method][class_id] += [np.count_nonzero(prediction[selected]), selected.sum()]
        panels[sample["id"]] = [values["udcp_otsu_rgb565"], values["pixel_forest_rgb565"]]
        make_panel(rgb565, labels, predictions, sample["id"], panels[sample["id"]]).save(
            output / "overlays" / f"{sample['id']}.jpg", quality=92
        )
        if (index + 1) % 20 == 0:
            print(f"Compared {index + 1}/110; exact C/Python parity", flush=True)
    for position, key in enumerate(("tp", "fp", "fn", "tn")):
        if int(counts["udcp_otsu"][position]) != previous["methods"]["udcp_otsu"][key]:
            raise AssertionError("Original baseline counts changed")
    methods = {}
    for method in METHODS:
        methods[method] = {
            **metrics(counts[method]),
            "macro_image_miou": float(np.mean([row["miou"] for row in rows if row["method"] == method])),
            "per_class_predicted_foreground_fraction": {
                name: (float(hits / total) if total else None)
                for name, (hits, total) in zip(CLASS_NAMES, per_class[method])
            },
        }
    changes = np.array([after - before for before, after in panels.values()])
    workspace = native.library.pf_workspace_bytes(160, 120)
    summary = {
        "task": "SUIM water/non-water semantic proxy; no collision distance ground truth",
        "test_count": len(samples),
        "methods": methods,
        "paired_images": {
            "improved": int((changes > 1e-12).sum()),
            "regressed": int((changes < -1e-12).sum()),
            "tied": int((abs(changes) <= 1e-12).sum()),
        },
        "parity": {
            "synthetic_cases": synthetic_cases,
            "test_images": len(samples),
            "feature_mismatches": 0,
            "score_mismatches": 0,
            "mask_mismatches": 0,
            "old_baseline_counts_exact": True,
        },
        "hardware": {
            "model_array_bytes": locked["model_array_bytes"],
            "workspace_bytes_160x120": workspace,
            "total_image_buffers_bytes_160x120": workspace + 4 * 160 * 120,
            "board_fps_measured": False,
        },
        "desktop_timing_only_ms": {
            "median": float(np.median(times)),
            "p95": float(np.percentile(times, 95)),
            "max": max(times),
            "includes": "ctypes, RGB565 packing, buffer allocation, C features + inference + median; excludes capture, I/O, and feature parity checks",
            "platform": platform.platform(),
        },
        "locked_sha256": sha(run / "LOCKED.json"),
        "dll_sha256": sha(run / "pixel_forest.dll"),
        "source_sha256": {
            name: sha(resolve(name))
            for name in (
                "compare_pixel_forest.py",
                "forest_native.py",
                "embedded/pixel_forest.c",
                "embedded/pixel_forest.h",
                "udcp.py",
                "validate.py",
            )
        },
    }
    dump_json(output / "summary.json", summary)
    write_outputs(output, summary, rows, panels)
    print(json.dumps({method: methods[method] for method in METHODS}, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RESULTS_DIR / "forest_comparison")
    run_comparison(parser.parse_args().run.resolve())
