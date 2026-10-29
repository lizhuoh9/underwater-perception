"""Score the locked ESP-DL CNN on TEST and compare it with the forest on the audited subsets.

Needs the training extras (torch, esp-ppq) and the prepared data; run after train_cnn_espdl. Every locked artifact
and source is hash-checked first. The int8 result runs the saved ESP-PPQ graph on
the device-exact input codes; forest and UDCP per-image counts come from the
compare_dedup.py evaluation built on the same near-duplicate audit.
"""

import argparse
import csv
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from uwseg.cnn_seg import WaterSegNet, float_scores, int8_scores, load_work_frame, model_frame, normalise
from uwseg.optimize_pixel_forest import constrained_threshold
from uwseg.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR, resolve
from uwseg.train_cnn_espdl import SEED, device_codes, exponents
from uwseg.train_pixel_forest import threshold_curve
from uwseg.udcp import confusion, metrics
from uwseg.validate import at_native_resolution, dump_json, font, overlay

COUNT_KEYS = ("tp", "fp", "fn", "tn")
BASELINES = ("udcp_otsu_rgb565", "forest_original", "forest_threshold_only", "forest_lowfp")
LABELS = {
    "cnn_int8": "ESP-DL CNN int8（ESP-PPQ 仿真）",
    "cnn_float": "CNN 浮点参考",
    "udcp_otsu_rgb565": "UDCP-Otsu RGB565",
    "forest_original": "原森林，阈值 57",
    "forest_threshold_only": "原森林，阈值 71（推荐）",
    "forest_lowfp": "16×8 森林，阈值 72",
}
SUBSETS = {"all": "全部", "without_near_duplicates": "去掉近重复", "without_same_scene": "再去掉同场景（最严格）"}
BOOTSTRAP_REPS = 2000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(run: Path, baseline: Path) -> tuple[dict, dict, dict]:
    locked = json.loads((run / "LOCKED.json").read_text(encoding="utf-8"))
    if sha(run / "experiment_plan.json") != locked["plan_sha256"]:
        raise ValueError("Experiment plan changed after locking")
    for name, digest in locked["artifact_sha256"].items():
        if sha(run / name) != digest:
            raise ValueError(f"Locked artifact changed: {name}")
    for name, digest in locked["source_sha256"].items():
        if sha(resolve(name)) != digest:
            raise ValueError(f"Source changed since locking: {name}")
    plan = json.loads((run / "experiment_plan.json").read_text(encoding="utf-8"))
    for name, digest in plan["inputs_sha256"].items():
        if sha(resolve(name)) != digest:
            raise ValueError(f"Input changed since training: {name}")
    audit_path = REPO_ROOT / plan["dedup_manifest"]["path"]
    if sha(audit_path) != plan["dedup_manifest"]["sha256"]:
        raise ValueError("Near-duplicate audit changed since training")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    summary = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    if summary["dedup_manifest_sha256"] != plan["dedup_manifest"]["sha256"]:
        raise ValueError("Baseline evaluation used a different near-duplicate audit")
    check_lock_values(run, locked, plan)
    return locked, plan, audit


def check_lock_values(run: Path, locked: dict, plan: dict) -> None:
    """LOCKED.json may only restate what its hashed curves and generated header imply."""
    if locked["fnr_cap"] != plan["fnr_cap"]:
        raise ValueError("Locked FNR cap differs from the experiment plan")
    curves = json.loads((run / "calibration_curves.json").read_text(encoding="utf-8"))
    for key, name in (("int8", "chosen"), ("float", "float_reference")):
        choice = constrained_threshold(
            np.array(curves[key]["counts"]), np.array(curves[key]["macro_image_miou"]), plan["fnr_cap"]
        )
        if choice["threshold_q"] != locked[name]["threshold_q"]:
            raise ValueError(f"Locked {key} threshold does not follow from the calibration curves")
    header = (run / "water_seg_model_config.h").read_text(encoding="utf-8")

    def constant(name):
        return re.search(rf"\b{name} = \{{?([^;}}]+)\}}?;", header).group(1)

    stated = {
        "kInputExponent": locked["input_exponent"],
        "kOutputExponent": locked["output_exponent"],
        "kThresholdCode": locked["chosen"]["threshold_code"],
    }
    if (
        any(int(constant(name)) != value for name, value in stated.items())
        or locked["chosen"]["threshold_code"] != locked["chosen"]["threshold_q"] - 128
    ):
        raise ValueError("Generated header disagrees with LOCKED.json")
    for name, key in (("kMean", "mean"), ("kStd", "std")):
        values = [np.float32(part.strip().rstrip("f")) for part in constant(name).split(",")]
        if values != [np.float32(v) for v in locked[key]]:
            raise ValueError(f"Generated header {name} disagrees with LOCKED.json")


def verify_graph(graph, executor, locked: dict, calibration: list[dict]) -> None:
    """The reloaded int8 graph must be the one the threshold was chosen on."""
    if exponents(graph) != (locked["input_exponent"], locked["output_exponent"]):
        raise ValueError("Reloaded ESP-PPQ graph has different input/output exponents")
    current = {name: version(name) for name in ("torch", "esp-ppq", "onnx", "numpy", "pillow")}
    if any(current[name] != locked["environment"][name] for name in current):
        raise ValueError(f"Package versions differ from the locked run: {current}")
    mean, std = (np.array(locked[key], dtype=np.float32) for key in ("mean", "std"))
    scored = []
    for sample in calibration:
        work, labels = load_work_frame(sample, REPO_ROOT)
        codes = device_codes(model_frame(work)[None], mean, std, locked["input_exponent"])
        with torch.no_grad():
            quant = executor.forward(
                inputs=torch.from_numpy(codes.astype(np.float32)) * 2.0 ** locked["input_exponent"]
            )[0]
        scored.append((int8_scores(quant, locked["output_exponent"])[0], labels))
    _, counts = threshold_curve(scored)
    if counts[locked["chosen"]["threshold_q"]].tolist() != locked["chosen"]["calibration_counts"]:
        raise ValueError("Calibration replay through the reloaded graph does not reproduce the locked counts")


def baseline_rows(baseline: Path, audit: dict) -> list[dict]:
    """Forest/UDCP rows, checked against the pooled counts their own summary reports."""
    with (baseline / "per_image.csv").open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] in BASELINES]
    rows = [
        {"id": row["id"], "method": row["method"], **metrics([int(row[key]) for key in COUNT_KEYS])} for row in rows
    ]
    summary = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    for name in SUBSETS:
        ids = set(audit["test_subsets"][name])
        for method in BASELINES:
            pooled = [
                sum(row[key] for row in rows if row["method"] == method and row["id"] in ids) for key in COUNT_KEYS
            ]
            if pooled != [summary["subsets"][name]["methods"][method][key] for key in COUNT_KEYS]:
                raise ValueError(f"Baseline per_image.csv disagrees with its summary: {method} on {name}")
    return rows


def summarise(rows: list[dict], ids: list[str], methods) -> dict:
    selected, result = set(ids), {}
    for method in methods:
        chosen = [row for row in rows if row["method"] == method and row["id"] in selected]
        m = metrics(np.sum([[row[key] for key in COUNT_KEYS] for row in chosen], axis=0))
        result[method] = {
            **m,
            "fnr": 1 - m["foreground_recall"],
            "images": len(chosen),
            "macro_image_miou": float(np.mean([row["miou"] for row in chosen])),
        }
    return result


def compare(rows: list[dict], ids: list[str], candidate: str, reference: str) -> dict:
    """Pooled deltas with an image-level paired bootstrap, plus per-image mIoU changes."""
    table = {(row["id"], row["method"]): row for row in rows}
    counts = {
        name: np.array([[table[(i, name)][key] for key in COUNT_KEYS] for i in ids]) for name in (candidate, reference)
    }
    draws = np.random.default_rng(SEED).integers(0, len(ids), (BOOTSTRAP_REPS, len(ids)))

    def rates(sums):
        return sums[..., 2] / (sums[..., 0] + sums[..., 2]), sums[..., 1] / (sums[..., 1] + sums[..., 3])

    (fnr_c, fpr_c), (fnr_r, fpr_r) = rates(counts[candidate].sum(0)), rates(counts[reference].sum(0))
    (boot_fnr_c, boot_fpr_c), (boot_fnr_r, boot_fpr_r) = (
        rates(counts[name][draws].sum(1)) for name in (candidate, reference)
    )
    changes = np.array([table[(i, candidate)]["miou"] - table[(i, reference)]["miou"] for i in ids])
    return {
        "fnr_percentage_points": (fnr_c - fnr_r) * 100,
        "fpr_percentage_points": (fpr_c - fpr_r) * 100,
        "fnr_ci95_pp": (np.percentile(boot_fnr_c - boot_fnr_r, [2.5, 97.5]) * 100).tolist(),
        "fpr_ci95_pp": (np.percentile(boot_fpr_c - boot_fpr_r, [2.5, 97.5]) * 100).tolist(),
        "paired_images": {
            "improved": int((changes > 1e-12).sum()),
            "regressed": int((changes < -1e-12).sum()),
            "tied": int((np.abs(changes) <= 1e-12).sum()),
        },
    }


def panel(frame: np.ndarray, labels: np.ndarray, cnn: np.ndarray, sample_id: str, values: tuple) -> Image.Image:
    small = (frame.shape[0], frame.shape[1])
    truth = at_native_resolution((labels != 0).astype(np.uint8), np.empty(small)).astype(bool)
    upsampled = np.repeat(np.repeat(cnn, 2, axis=0), 2, axis=1).astype(bool)
    views = [Image.fromarray(frame), overlay(frame, truth, (35, 210, 90)), overlay(frame, upsampled, (180, 90, 255))]
    result = Image.new("RGB", (860, 250), "white")
    draw = ImageDraw.Draw(result)
    draw.text(
        (8, 3),
        f"{sample_id} | native mIoU: forest-71 {values[0]:.3f}, CNN int8 {values[1]:.3f}",
        fill="black",
        font=font(17),
    )
    for index, (image, title) in enumerate(zip(views, ("Input (RGB565)", "GT: non-water", "ESP-DL CNN int8"))):
        image = image.resize((256, 192), Image.Resampling.NEAREST)
        result.paste(image, (index * 286 + 15, 52))
        draw.text((index * 286 + 15, 29), title, fill="black", font=font(16))
    return result


def run(run_dir: Path, baseline: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite evaluation evidence; choose a fresh --output")
    locked, plan, audit = load_run(run_dir, baseline)
    from esp_ppq import TorchExecutor
    from esp_ppq.api import load_native_graph

    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model = WaterSegNet(tuple(checkpoint["widths"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    graph = load_native_graph(str(run_dir / "water_seg_native"))
    executor = TorchExecutor(graph=graph, device="cpu")
    mean, std = (np.array(locked[key], dtype=np.float32) for key in ("mean", "std"))
    exponent_in, exponent_out = locked["input_exponent"], locked["output_exponent"]
    int8_threshold, float_threshold = locked["chosen"]["threshold_q"], locked["float_reference"]["threshold_q"]
    original = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    samples = [row for row in original["samples"] if row["split"] == "test"]
    if [row["id"] for row in samples] != audit["test_subsets"]["all"]:
        raise ValueError("Audit TEST order differs from data/manifest.json")
    # Both checks run before any TEST image is opened.
    verify_graph(graph, executor, locked, [row for row in original["samples"] if row["split"] == "calibration"])
    rows, masks = baseline_rows(baseline, audit), {}
    output.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples, 1):
        work, labels = load_work_frame(sample, REPO_ROOT)
        frame = model_frame(work)
        codes = device_codes(frame[None], mean, std, exponent_in)
        with torch.no_grad():
            quant = executor.forward(inputs=torch.from_numpy(codes.astype(np.float32)) * 2.0**exponent_in)[0]
            floating = model(
                normalise(
                    torch.from_numpy(frame[None]).permute(0, 3, 1, 2), torch.from_numpy(mean), torch.from_numpy(std)
                )
            )
        predictions = {
            "cnn_int8": int8_scores(quant, exponent_out)[0] >= int8_threshold,
            "cnn_float": float_scores(floating)[0] >= float_threshold,
        }
        masks[sample["id"]] = (frame, labels, predictions["cnn_int8"])
        for method, predicted in predictions.items():
            native = at_native_resolution(predicted.astype(np.uint8), labels).astype(bool)
            rows.append({"id": sample["id"], "method": method, **metrics(confusion(native, labels != 0))})
        if index % 20 == 0:
            print(f"Scored {index}/{len(samples)} TEST images", flush=True)
    methods = ("cnn_int8", "cnn_float", *BASELINES)
    results = {}
    for name in SUBSETS:
        ids = audit["test_subsets"][name]
        results[name] = {
            "images": len(ids),
            "methods": summarise(rows, ids, methods),
            "cnn_int8_vs_forest_71": compare(rows, ids, "cnn_int8", "forest_threshold_only"),
            "cnn_int8_vs_cnn_float": compare(rows, ids, "cnn_int8", "cnn_float"),
        }
    strict = audit["test_subsets"]["without_same_scene"]
    miou = {(row["id"], row["method"]): row["miou"] for row in rows}
    ordered = sorted(strict, key=lambda i: (miou[(i, "cnn_int8")] - miou[(i, "forest_threshold_only")], i))
    chosen = ordered[:2] + ordered[len(ordered) // 2 : len(ordered) // 2 + 1] + ordered[-2:]
    sheet = Image.new("RGB", (860, 250 * len(chosen)), "white")
    for position, sample_id in enumerate(chosen):
        frame, labels, cnn = masks[sample_id]
        sheet.paste(
            panel(
                frame,
                labels,
                cnn,
                sample_id,
                (miou[(sample_id, "forest_threshold_only")], miou[(sample_id, "cnn_int8")]),
            ),
            (0, 250 * position),
        )
    sheet.save(output / "examples.jpg", quality=90)
    summary = {
        "task": "Locked ESP-DL CNN vs locked forest on the audited SUIM TEST subsets; water/non-water, not collision distance",
        "subsets": results,
        "examples": {
            "ids": chosen,
            "rule": "strict subset: two largest regressions vs forest-71, median, two largest gains",
        },
        "checks": "LOCKED.json values re-derived from hashed curves and header; reloaded graph exponents, package versions and calibration counts reproduced; baseline CSV matches its summary",
        "locked_sha256": sha(run_dir / "LOCKED.json"),
        "baseline_summary_sha256": sha(baseline / "summary.json"),
        "baseline_per_image_sha256": sha(baseline / "per_image.csv"),
        "dedup_manifest_sha256": plan["dedup_manifest"]["sha256"],
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": SEED},
        "source_sha256": {
            name: sha(resolve(name)) for name in ("compare_cnn_espdl.py", "cnn_seg.py", "train_cnn_espdl.py")
        },
    }
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(output, summary, locked)
    dump_json(output / "summary.json", summary)
    return summary


def write_report(output: Path, summary: dict, locked: dict) -> None:
    results = summary["subsets"]
    lines = [
        "# ESP-DL CNN 与随机森林对比",
        "",
        "模型、阈值与 ESP-DL 导出在加载 TEST 前已锁定；int8 结果使用与 ESP32-P4 相同的输入量化码运行 ESP-PPQ 量化图。"
        "森林与 UDCP 的逐图计数来自去泄漏复评。",
        "",
        f"CNN：{locked['macs_per_frame'] / 1e6:.1f} MMAC/帧，{locked['parameters']:,} 个参数，输出 80×60；"
        f"阈值码 {locked['chosen']['threshold_code']}（标定漏检 {locked['chosen']['calibration_fnr']:.2%}，"
        f"误检 {locked['chosen']['calibration_fpr']:.2%}，上限 {locked['fnr_cap']:.2%}）。",
        "",
    ]
    for name, label in SUBSETS.items():
        block = results[name]
        lines.extend(
            [
                f"## {label}（{block['images']} 张）",
                "",
                "| 方法 | 漏检 FNR | 水体误检 FPR | 两类全局 mIoU | 逐图平均 mIoU |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in ("cnn_int8", "cnn_float", *BASELINES):
            m = block["methods"][method]
            lines.append(
                f"| {LABELS[method]} | {m['fnr']:.2%} | {m['water_false_positive_rate']:.2%} | {m['miou']:.4f} | {m['macro_image_miou']:.4f} |"
            )
        delta = block["cnn_int8_vs_forest_71"]
        images = delta["paired_images"]
        lines.extend(
            [
                "",
                f"CNN int8 相对推荐森林：漏检 {delta['fnr_percentage_points']:+.2f} 个百分点"
                f"（95% 区间 {delta['fnr_ci95_pp'][0]:+.2f} ~ {delta['fnr_ci95_pp'][1]:+.2f}），误检 {delta['fpr_percentage_points']:+.2f} 个百分点"
                f"（{delta['fpr_ci95_pp'][0]:+.2f} ~ {delta['fpr_ci95_pp'][1]:+.2f}）；逐图 mIoU 改善 {images['improved']}、退步 {images['regressed']}、持平 {images['tied']}。",
                "",
            ]
        )
    lines.extend(
        [
            "区间为按图像有放回重抽样 2000 次的 95% 百分位区间；测试图多来自少数视频，真实不确定性可能更大。",
            "",
            "## 限制",
            "",
            "- 同一批 SUIM 照片的复用比较，不是 OV2640 实拍；水色、曝光、白平衡与防水窗差异需要实拍数据验证。",
            "- int8 结果是 ESP-PPQ 对 ESP32-P4 的仿真；板上一致性由 water_seg_demo 的自检与参考掩膜比对确认。",
            "- 输出为水体/非水体分割，不是碰撞距离。",
            "",
            "![最严格子集：相对森林退步最大两张、中位一张、改善最大两张](examples.jpg)",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=RESULTS_DIR / "evaluation_dedup")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run.resolve(), args.baseline.resolve(), args.output.resolve())
