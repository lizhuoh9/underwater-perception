"""Calibrate on train_val, lock parameters, then evaluate every SUIM TEST image."""

import argparse
import csv
import hashlib
import html
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from uwseg.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR, resolve
from uwseg.udcp import (
    confusion,
    decode_labels,
    metrics,
    otsu_threshold,
    red_fraction_score,
    rgb565_roundtrip,
    udcp_score,
)

CLASS_NAMES = ["water", "diver", "plants", "wreck", "robot", "reef", "fish", "seafloor_rocks"]
METHODS = ["udcp_fixed", "udcp_otsu", "udcp_rgb565", "red_fraction", "all_foreground"]


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_sample(sample):
    for field in ("image", "mask"):
        path = REPO_ROOT / sample[field]
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != sample[field + "_sha256"]:
            raise ValueError(f"Modified {field}: {path}")
    with Image.open(REPO_ROOT / sample["image"]) as image, Image.open(REPO_ROOT / sample["mask"]) as mask:
        if image.size != mask.size:
            raise ValueError(f"Image/mask geometry mismatch: {sample['id']}")
        labels = decode_labels(np.array(mask.convert("RGB")))
        rgb_image = image.convert("RGB")
        rgb_image.thumbnail((160, 120), Image.Resampling.BILINEAR)
        rgb = np.array(rgb_image)
    return rgb, labels


def at_native_resolution(values, labels):
    return np.array(Image.fromarray(values).resize((labels.shape[1], labels.shape[0]), Image.Resampling.NEAREST))


def choose_threshold(scored_samples):
    """Every candidate uses the same native-resolution ground truth and image weighting."""
    objectives = []
    for score, labels in scored_samples:
        native_score = at_native_resolution(score, labels)
        foreground = labels != 0
        foreground_hist = np.bincount(native_score[foreground], minlength=256)
        water_hist = np.bincount(native_score[~foreground], minlength=256)
        tp = np.r_[foreground_hist[::-1].cumsum()[::-1], 0]
        fp = np.r_[water_hist[::-1].cumsum()[::-1], 0]
        fn, tn = foreground.sum() - tp, (~foreground).sum() - fp
        fg_denominator, bg_denominator = tp + fp + fn, tn + fp + fn
        fg_iou = np.divide(tp, fg_denominator, out=np.zeros(257), where=fg_denominator > 0)
        bg_iou = np.divide(tn, bg_denominator, out=np.zeros(257), where=bg_denominator > 0)
        defined = (fg_denominator > 0).astype(int) + (bg_denominator > 0).astype(int)
        objectives.append((fg_iou + bg_iou) / defined)
    average = np.mean(objectives, axis=0)
    threshold = int(average.argmax())
    return threshold, float(average[threshold]), average


def calibrate(samples, output):
    calibration = [load_sample(sample) for sample in samples]
    if not calibration:
        raise ValueError("Calibration split is empty")
    candidates, curve_rows = [], []
    for window in (3, 5, 9):
        scored = [(udcp_score(rgb, window)[0], labels) for rgb, labels in calibration]
        threshold, objective, curve = choose_threshold(scored)
        candidates.append({"window": window, "threshold_q": threshold, "calibration_macro_miou": objective})
        curve_rows.extend(
            {"method": "udcp", "window": window, "threshold_q": q, "macro_miou": float(value)}
            for q, value in enumerate(curve)
        )
    chosen = max(candidates, key=lambda row: row["calibration_macro_miou"])
    red_threshold, red_objective, red_curve = choose_threshold(
        [(red_fraction_score(rgb), labels) for rgb, labels in calibration]
    )
    curve_rows.extend(
        {"method": "red_fraction", "window": "", "threshold_q": q, "macro_miou": float(value)}
        for q, value in enumerate(red_curve)
    )
    with (output / "calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    config = {
        "input_bounds": [160, 120],
        "preserve_aspect_ratio": True,
        "ambient_candidate_fraction": 0.001,
        "median_window": 3,
        "foreground_rule": "uint8_score >= threshold_q",
        "calibration_objective": "mean per-image mean IoU of water and non-water on native masks",
        "udcp": chosen,
        "udcp_candidates": candidates,
        "red_fraction": {"threshold_q": red_threshold, "calibration_macro_miou": red_objective},
        "calibration_ids": [sample["id"] for sample in samples],
    }
    # This file is written before any test images or labels are loaded.
    dump_json(output / "locked_parameters.json", config)
    return config


def overlay(rgb, mask, color):
    mixed = rgb.astype(float).copy()
    mixed[mask] = mixed[mask] * 0.45 + np.array(color) * 0.55
    return Image.fromarray(np.rint(mixed).astype(np.uint8))


def font(size):
    for path in (Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_panel(rgb, labels, predictions, sample_id, image_miou):
    small_truth = np.array(
        Image.fromarray((labels != 0).astype(np.uint8)).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
    ).astype(bool)
    views = [
        Image.fromarray(rgb),
        overlay(rgb, small_truth, (35, 210, 90)),
        overlay(rgb, predictions["udcp_fixed"], (255, 110, 30)),
        overlay(rgb, predictions["udcp_otsu"], (180, 90, 255)),
    ]
    titles = ["Input (<=160x120)", "GT: non-water", "UDCP: fixed", "UDCP: Otsu"]
    panel = Image.new("RGB", (960, 225), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 3), f"{sample_id} | UDCP native mIoU {image_miou:.3f}", fill="black", font=font(15))
    for index, (view, title) in enumerate(zip(views, titles)):
        view.thumbnail((236, 165), Image.Resampling.NEAREST)
        x = index * 240 + (240 - view.width) // 2
        panel.paste(view, (x, 50))
        draw.text((index * 240 + 6, 29), title, fill="black", font=font(14))
    return panel


def write_gallery(output, entries, summary):
    rows = []
    for name in METHODS:
        values = summary["methods"][name]
        rows.append(
            f"<tr><td>{name}</td><td>{values['miou']:.3f}</td><td>{values['foreground_iou']:.3f}</td><td>{values['foreground_recall']:.3f}</td><td>{values['water_false_positive_rate']:.3f}</td></tr>"
        )
    figures = []
    for entry in sorted(entries, key=lambda row: row["miou"]):
        sample_id = html.escape(entry["id"], quote=True)
        figures.append(
            f'<figure><img loading="lazy" src="overlays/{sample_id}.jpg" alt="{sample_id}: input, ground truth, fixed-threshold UDCP and Otsu UDCP"><figcaption>{sample_id}; fixed-threshold UDCP mIoU={entry["miou"]:.3f}</figcaption></figure>'
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>水下 UDCP 离线验证</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#17202a;background:#fff}h1{font-size:25px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}figure{margin:28px 0}img{width:100%;height:auto}figcaption{color:#555}a{color:#1558a6}.scroll{overflow-x:auto}</style>
<h1>水下 UDCP 离线验证</h1><p>SUIM 110 张测试图；参数仅用另外 80 张校准图选择。水体与非水体二分类，非距离／碰撞真值。非严格复现原论文控制器；无实机帧率结论。</p>
<p>输入保持比例缩小至 160×120 范围；预测最近邻放回原始尺寸后与标注比较。下表为所有测试像素汇总指标；图像按 UDCP mIoU 从差到好排列，完整展示全部样本。</p>
<p><a href="summary.json">机器可读结果</a> · <a href="per_image.csv">逐图指标</a> · <a href="locked_parameters.json">锁定参数</a> · <a href="../data/SOURCES.md">数据来源与编码审计</a></p>
<div class="scroll"><table><thead><tr><th>方法</th><th>两类 mIoU</th><th>非水体 IoU</th><th>非水体召回</th><th>水体误报率</th></tr></thead><tbody>"""
    source_relative = Path(os.path.relpath(DATA_DIR / "SOURCES.md", output)).as_posix()
    page = page.replace("../data/SOURCES.md", html.escape(source_relative, quote=True))
    page += "".join(rows) + "</tbody></table></div>" + "".join(figures) + "</html>"
    (output / "gallery.html").write_text(page, encoding="utf-8")


def write_report(output, summary):
    lines = [
        "# 水下 UDCP 离线验证结果",
        "",
        "本次是已知标注上的水体／非水体二分类验证。它没有验证障碍距离、八种语义类别或自主避障安全性。",
        "",
        "## 数据与方法",
        "",
        "- SUIM 公开镜像；80 张 train_val 图用于选参数，全部 110 张 TEST 图只用于评估。",
        "- 训练部分与测试图文件内容完全重复的 4 张已在选校准集前排除；没有场景级去重信息。",
        "- 镜像实际有 1488 个 train_val 样本（官方描述为 1525）；已审计的 190 张掩膜是灰度类别 ID 0–7。",
        "- 类别 0 是水体，类别 1–7 合并为非水体；不将非水体直接解释成近距离碰撞障碍。",
        "- 输入保持比例缩至 160×120 范围；预测最近邻放回原尺寸，与原始标注比较。",
        "- 简化 UDCP：G/B 暗通道、候选区域估计 A、透射率量化、3×3 中值滤波；不是原论文全流程复现。",
        "",
    ]
    chosen = summary["config"]["udcp"]
    lines.extend(
        [
            f"锁定参数：最小值窗口 **{chosen['window']}×{chosen['window']}**，固定阈值 **t ≥ {chosen['threshold_q']}/255**。Otsu 是预先指定的逐帧无监督阈值对照，不根据 TEST 标注调参。",
            "",
            "## 测试结果",
            "",
            "| 方法 | 两类 mIoU | 非水体 IoU | 非水体召回率 | 水体误报率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in METHODS:
        result = summary["methods"][name]
        values = [result[key] for key in ("miou", "foreground_iou", "foreground_recall", "water_false_positive_rate")]
        lines.append("| " + name + " | " + " | ".join(f"{100 * value:.2f}%" for value in values) + " |")
    otsu = summary["methods"]["udcp_otsu"]
    lines.extend(
        [
            "",
            "上表汇总全部测试像素。mIoU 是水体与非水体 IoU 的平均；召回率和误报率均按像素统计，不是按障碍物数量统计。逐图宏平均值和每类前景覆盖率另存于 summary.json。",
            "",
            f"Otsu 版本仍漏掉 **{100 * (1 - otsu['foreground_recall']):.2f}%** 非水体像素，并把 **{100 * otsu['water_false_positive_rate']:.2f}%** 水体像素标成前景。当前结果不支持把这个方法单独作为可靠障碍检测器。",
            "",
            "固定阈值版本的 RGB565 量化对照得分接近原 RGB 输入，提示当前主要障碍是外观先验与场景泛化；这只隔离了颜色位深，未模拟 OV2640 噪声、ISP、曝光、JPEG 或防水窗。",
            "",
            "## 图像与原始证据",
            "",
            "![固定阈值由差到好的三例，附 Otsu 对照](preview.jpg)",
            "",
            "三例选取自固定阈值 UDCP 的最差、中间排名、最佳测试图；完整 9 个排名分位点见 contact_sheet.jpg，全部 110 图见 [gallery.html](gallery.html)。不是随机抽样或总体成功率展示。",
            "",
            "- [完整逐图指标](per_image.csv)",
            "- [参数选择曲线](calibration.csv)",
            "- [锁定参数](locked_parameters.json)",
            "- [来源、汇总指标及源码哈希](summary.json)",
            "",
            "## MCU",
            "",
            "160×120 整数实现具备移植可行性；计划基本缓冲为 6N=115200 字节（112.5 KiB），另计相机驱动、DMA、RTOS、通信和滤波队列。建议使用带可用 PSRAM 的板卡。当前 Python 实现不能直接搬到 MCU，需改为 C/C++、整数归一化查表及快速最小值滤波。",
            "",
            "没有板端帧率、相机采集、内存峰值或水中闭环实测；不能从 PC 耗时换算 MCU 帧率。",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(manifest_path, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix runs in populated output directory: {output}; choose a fresh --output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = [row for row in manifest["samples"] if row["split"] == "calibration"]
    test = [row for row in manifest["samples"] if row["split"] == "test"]
    if len(test) != 110:
        raise ValueError(f"Expected all 110 test images; got {len(test)}")
    if {s["image_sha256"] for s in calibration} & {s["image_sha256"] for s in test}:
        raise ValueError("Calibration/test image content overlap")
    output.mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir(exist_ok=True)
    config = calibrate(calibration, output)
    print("Locked parameters:", json.dumps(config["udcp"]), flush=True)
    sums = {name: np.zeros(4, dtype=np.int64) for name in METHODS}
    per_class = {name: np.zeros((8, 2), dtype=np.int64) for name in METHODS}
    timings = {name: [] for name in ("udcp_fixed", "udcp_rgb565", "red_fraction")}
    image_metrics = {name: [] for name in METHODS}
    rows, entries, flagged = [], [], []
    window = config["udcp"]["window"]
    threshold = config["udcp"]["threshold_q"]
    for index, sample in enumerate(test, 1):
        rgb, labels = load_sample(sample)
        start = time.perf_counter()
        score, info = udcp_score(rgb, window)
        timings["udcp_fixed"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        quantized_score, _ = udcp_score(rgb565_roundtrip(rgb), window)
        timings["udcp_rgb565"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        red_score = red_fraction_score(rgb)
        timings["red_fraction"].append((time.perf_counter() - start) * 1000)
        predictions = {
            "udcp_fixed": score >= threshold,
            "udcp_otsu": score >= otsu_threshold(score),
            "udcp_rgb565": quantized_score >= threshold,
            "red_fraction": red_score >= config["red_fraction"]["threshold_q"],
            "all_foreground": np.ones(score.shape, dtype=bool),
        }
        if info["flags"]:
            flagged.append({"id": sample["id"], "flags": info["flags"]})
        for name, predicted in predictions.items():
            native = at_native_resolution(predicted.astype(np.uint8), labels).astype(bool)
            counts = confusion(native, labels != 0)
            values = metrics(counts)
            sums[name] += counts
            image_metrics[name].append(values)
            rows.append(
                {
                    "id": sample["id"],
                    "method": name,
                    "input_width": rgb.shape[1],
                    "input_height": rgb.shape[0],
                    "native_width": labels.shape[1],
                    "native_height": labels.shape[0],
                    **values,
                    "ambient_rgb": json.dumps(info["ambient_rgb"]),
                    "flags": ";".join(info["flags"]),
                }
            )
            for class_id in range(8):
                selected = labels == class_id
                per_class[name][class_id] += [np.count_nonzero(native & selected), np.count_nonzero(selected)]
        current_miou = image_metrics["udcp_fixed"][-1]["miou"]
        panel = make_panel(rgb, labels, predictions, sample["id"], current_miou)
        panel.save(output / "overlays" / (sample["id"] + ".jpg"), quality=90)
        entries.append({"id": sample["id"], "miou": current_miou})
        if index % 20 == 0 or index == len(test):
            print(f"Evaluated {index}/{len(test)}", flush=True)
    results = {}
    for name in METHODS:
        result = metrics(sums[name])
        result["macro_image_miou"] = float(np.mean([row["miou"] for row in image_metrics[name]]))
        result["per_class_predicted_foreground_rate"] = {
            CLASS_NAMES[class_id]: {
                "positive_pixels": int(positive),
                "pixels": int(total),
                "rate": float(positive / total) if total else None,
            }
            for class_id, (positive, total) in enumerate(per_class[name])
        }
        if name in timings:
            result["desktop_score_time_ms"] = {
                "median": float(np.median(timings[name])),
                "p95": float(np.percentile(timings[name], 95)),
            }
        results[name] = result
    summary = {
        "task": "SUIM water vs non-water semantic proxy, not collision/metric-depth evaluation",
        "evaluation": "native-resolution masks; nearest-neighbor upsample of <=160x120 predictions",
        "dataset_source": manifest["source"],
        "dataset_counts": manifest["counts"],
        "calibration_count": len(calibration),
        "test_count": len(test),
        "config": config,
        "methods": results,
        "flagged_test_frames": flagged,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "timing_scope": "desktop score calculation only; excludes camera capture, JPEG decode, resize, I/O, plotting, networking; not MCU timing",
        },
        "source_code_sha256": {
            name: hashlib.sha256(resolve(name).read_bytes()).hexdigest()
            for name in ("udcp.py", "validate.py", "prepare_data.py")
        },
    }
    dump_json(output / "summary.json", summary)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted(entries, key=lambda row: row["miou"])
    selected = [ranked[int(round(value))] for value in np.linspace(0, len(ranked) - 1, 9)]
    sheet = Image.new("RGB", (960, 225 * len(selected) + 40), "white")
    ImageDraw.Draw(sheet).text(
        (8, 8),
        "9 test-image rank quantiles: worst -> best; full 110-image gallery available",
        fill="black",
        font=font(17),
    )
    for index, entry in enumerate(selected):
        with Image.open(output / "overlays" / (entry["id"] + ".jpg")) as panel:
            sheet.paste(panel, (0, 40 + index * 225))
    sheet.save(output / "contact_sheet.jpg", quality=91)
    preview = Image.new("RGB", (960, 225 * 3 + 40), "white")
    ImageDraw.Draw(preview).text(
        (8, 8), "Fixed-UDCP worst / median-rank / best; Otsu shown for comparison", fill="black", font=font(17)
    )
    for index, entry in enumerate((selected[0], selected[4], selected[-1])):
        with Image.open(output / "overlays" / (entry["id"] + ".jpg")) as panel:
            preview.paste(panel, (0, 40 + index * 225))
    preview.save(output / "preview.jpg", quality=93)
    dump_json(output / "contact_sheet_selection.json", selected)
    write_gallery(output, entries, summary)
    write_report(output, summary)
    print(
        json.dumps(
            {
                name: {
                    key: value
                    for key, value in result.items()
                    if key
                    in ("miou", "foreground_iou", "foreground_recall", "water_false_positive_rate", "macro_image_miou")
                }
                for name, result in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "manifest.json")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "validation_results")
    args = parser.parse_args()
    run(args.manifest, args.output)
