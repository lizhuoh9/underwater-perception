"""Score a locked lite ESP-DL CNN run (train_cnn_lite.py) on TEST against run_01 and the forest.

Needs the training extras (torch, esp-ppq) and the prepared data. Before any TEST image is opened it verifies the
locked artifacts, sources, inputs and exploration record, re-derives both
pre-registered thresholds from the hashed calibration curves, checks the generated
header, replays the calibration images through the reloaded ESP-PPQ graph, and checks
the run_01 and forest/UDCP per-image counts against their own summaries. run_01 rows
are its recorded results: the locked threshold (+11) and the post-hoc +5 variant.
"""

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from uwseg.cnn_lite import build
from uwseg.cnn_seg import float_scores, int8_scores, load_work_frame, model_frame, normalise
from uwseg.compare_cnn_espdl import (
    BASELINES,
    BOOTSTRAP_REPS,
    COUNT_KEYS,
    SUBSETS,
    baseline_rows,
    check_lock_values,
    compare,
    sha,
    summarise,
)
from uwseg.optimize_pixel_forest import constrained_threshold
from uwseg.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR, resolve
from uwseg.train_cnn_espdl import SEED, device_codes, exponents
from uwseg.train_pixel_forest import threshold_curve
from uwseg.udcp import confusion, metrics
from uwseg.validate import at_native_resolution, dump_json, font, overlay

# The dense reference run (run_01): its recorded int8 results are compared, not recomputed.
REFERENCE_RUN = RESULTS_DIR / "cnn_espdl_run_01"
CLASS_NAMES = ["water", "diver", "plants", "wreck", "robot", "reef", "fish", "seafloor_rocks"]
CANDIDATES = ("run02_int8_deploy", "run02_int8_compare", "run02_float_deploy", "run02_float_compare")
REFERENCE_METHODS = ("run01_cap5", "run01_locked")
LABELS = {
    "run02_int8_deploy": "run_02 int8，部署阈值（漏检上限 5%）",
    "run02_int8_compare": "run_02 int8，对照阈值（同森林上限）",
    "run02_float_deploy": "run_02 浮点参考，部署阈值",
    "run02_float_compare": "run_02 浮点参考，对照阈值",
    "run01_cap5": "run_01 int8，阈值 +5（事后版本）",
    "run01_locked": "run_01 int8，锁定阈值 +11",
    "udcp_otsu_rgb565": "UDCP-Otsu RGB565",
    "forest_original": "原森林，阈值 57",
    "forest_threshold_only": "原森林，阈值 71（推荐）",
    "forest_lowfp": "16×8 森林，阈值 72",
}
COMPARISONS = {
    "deploy_vs_run01_cap5": ("run02_int8_deploy", "run01_cap5"),
    "compare_vs_run01_locked": ("run02_int8_compare", "run01_locked"),
    "deploy_vs_forest_71": ("run02_int8_deploy", "forest_threshold_only"),
    "compare_vs_forest_71": ("run02_int8_compare", "forest_threshold_only"),
    "deploy_int8_vs_float": ("run02_int8_deploy", "run02_float_deploy"),
}


def load_run(run: Path, baseline: Path) -> tuple[dict, dict, dict]:
    locked = json.loads((run / "LOCKED.json").read_text(encoding="utf-8"))
    plan = json.loads((run / "experiment_plan.json").read_text(encoding="utf-8"))
    if sha(run / "experiment_plan.json") != locked["plan_sha256"]:
        raise ValueError("Experiment plan changed after locking")
    for folder, hashes in (
        (run, locked["artifact_sha256"]),
        (run / "exploration", locked["exploration_sha256"]),
        (run / "exploration", plan["architecture_selection"]["files_sha256"]),
        (None, locked["source_sha256"]),
    ):
        for name, digest in hashes.items():
            path = resolve(name) if folder is None else folder / name  # sources are package modules
            if sha(path) != digest:
                raise ValueError(f"Changed since locking: {path}")
    for name, digest in plan["inputs_sha256"].items():
        if sha(resolve(name)) != digest:
            raise ValueError(f"Input changed since training: {name}")
    audit_path = REPO_ROOT / plan["dedup_manifest"]["path"]
    if sha(audit_path) != plan["dedup_manifest"]["sha256"]:
        raise ValueError("Near-duplicate audit changed since training")
    if (
        json.loads((baseline / "summary.json").read_text(encoding="utf-8"))["dedup_manifest_sha256"]
        != plan["dedup_manifest"]["sha256"]
    ):
        raise ValueError("Baseline evaluation used a different near-duplicate audit")
    check_operating_points(run, locked, plan)
    return locked, plan, json.loads(audit_path.read_text(encoding="utf-8"))


def check_operating_points(run: Path, locked: dict, plan: dict) -> None:
    """Both thresholds must follow from the hashed curves under the plan's caps; the header carries the deployment one."""
    points = plan["operating_points"]
    if not (
        locked["architecture"] == plan["architecture"]
        and points["deployment"]["fnr_cap"] == plan["fnr_cap"]
        and points["comparison"]["fnr_cap"] == locked["comparison"]["fnr_cap"]
    ):
        raise ValueError("Locked architecture or caps differ from the experiment plan")
    check_lock_values(run, locked, plan)  # deployment point, header constants
    curves = json.loads((run / "calibration_curves.json").read_text(encoding="utf-8"))
    for kind in ("int8", "float"):
        choice = constrained_threshold(
            np.array(curves[kind]["counts"]),
            np.array(curves[kind]["macro_image_miou"]),
            points["comparison"]["fnr_cap"],
        )
        if choice["threshold_q"] != locked["comparison"][kind]["threshold_q"]:
            raise ValueError(f"Locked comparison {kind} threshold does not follow from the calibration curves")
    if locked["comparison"]["int8"]["threshold_code"] != locked["comparison"]["int8"]["threshold_q"] - 128:
        raise ValueError("Comparison threshold code disagrees with its threshold")


def verify_graph(graph, executor, locked: dict, calibration: list[dict]) -> None:
    """The reloaded int8 graph must reproduce the calibration counts at both locked thresholds."""
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
    for choice in (locked["chosen"], locked["comparison"]["int8"]):
        if counts[choice["threshold_q"]].tolist() != choice["calibration_counts"]:
            raise ValueError("Calibration replay through the reloaded graph does not reproduce the locked counts")


def reference_rows(audit: dict, plan: dict) -> tuple[list[dict], dict]:
    """run_01's recorded int8 rows (+11 locked, +5 variant), checked against their own summaries and lock."""
    locked_sha = sha(REFERENCE_RUN / "LOCKED.json")
    reference_plan = json.loads((REFERENCE_RUN / "experiment_plan.json").read_text(encoding="utf-8"))
    if reference_plan["dedup_manifest"]["sha256"] != plan["dedup_manifest"]["sha256"]:
        raise ValueError("run_01 was trained against a different near-duplicate audit")
    export_path = REFERENCE_RUN / "threshold_cap5/MODEL_EXPORT.json"
    summaries = {
        "run01_locked": json.loads((REFERENCE_RUN / "evaluation/summary.json").read_text(encoding="utf-8")),
        "run01_cap5": json.loads(
            (REFERENCE_RUN / "threshold_cap5/evaluation/summary.json").read_text(encoding="utf-8")
        ),
    }
    if (
        summaries["run01_locked"]["locked_sha256"] != locked_sha
        or summaries["run01_cap5"]["export_sha256"] != sha(export_path)
        or json.loads(export_path.read_text(encoding="utf-8"))["locked_sha256"] != locked_sha
    ):
        raise ValueError("run_01 evaluations do not belong to run_01's lock")
    rows = []
    for name, folder, method in (
        ("run01_locked", REFERENCE_RUN / "evaluation", "cnn_int8"),
        ("run01_cap5", REFERENCE_RUN / "threshold_cap5/evaluation", "cnn_int8_variant"),
    ):
        with (folder / "per_image.csv").open(encoding="utf-8") as handle:
            chosen = [
                {"id": row["id"], "method": name, **metrics([int(row[key]) for key in COUNT_KEYS])}
                for row in csv.DictReader(handle)
                if row["method"] == method
            ]
        for subset in SUBSETS:
            ids = set(audit["test_subsets"][subset])
            pooled = [sum(row[key] for row in chosen if row["id"] in ids) for key in COUNT_KEYS]
            if pooled != [summaries[name]["subsets"][subset]["methods"][method][key] for key in COUNT_KEYS]:
                raise ValueError(f"run_01 per_image.csv disagrees with its summary: {name} on {subset}")
        rows.extend(chosen)
    rates = summaries["run01_cap5"]["strict_subset_class_rates"]
    hashes = {
        "run01_locked_sha256": locked_sha,
        "run01_evaluation_summary_sha256": sha(REFERENCE_RUN / "evaluation/summary.json"),
        "run01_cap5_summary_sha256": sha(REFERENCE_RUN / "threshold_cap5/evaluation/summary.json"),
    }
    return rows, {
        "rates": {
            "run01_cap5": rates["cnn_int8_variant"],
            "run01_locked": rates["cnn_int8"],
            "forest_threshold_only": rates["forest_threshold_only"],
        },
        "sha256": hashes,
    }


def panel(frame: np.ndarray, labels: np.ndarray, mask: np.ndarray, sample_id: str, values: tuple) -> Image.Image:
    truth = at_native_resolution((labels != 0).astype(np.uint8), np.empty(frame.shape[:2])).astype(bool)
    upsampled = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1).astype(bool)
    views = [Image.fromarray(frame), overlay(frame, truth, (35, 210, 90)), overlay(frame, upsampled, (180, 90, 255))]
    result = Image.new("RGB", (860, 250), "white")
    draw = ImageDraw.Draw(result)
    draw.text(
        (8, 3),
        f"{sample_id} | native mIoU: run_01 +5 {values[0]:.3f}, run_02 deploy {values[1]:.3f}",
        fill="black",
        font=font(17),
    )
    for index, (image, title) in enumerate(zip(views, ("Input (RGB565)", "GT: non-water", "run_02 int8 (deploy)"))):
        result.paste(image.resize((256, 192), Image.Resampling.NEAREST), (index * 286 + 15, 52))
        draw.text((index * 286 + 15, 29), title, fill="black", font=font(16))
    return result


def run(run_dir: Path, baseline: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite evaluation evidence; choose a fresh --output")
    locked, plan, audit = load_run(run_dir, baseline)
    from esp_ppq import TorchExecutor
    from esp_ppq.api import load_native_graph

    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != locked["architecture"]:
        raise ValueError("Checkpoint architecture differs from the lock")
    model = build(checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    graph = load_native_graph(str(run_dir / "water_seg_native"))
    executor = TorchExecutor(graph=graph, device="cpu")
    mean, std = (np.array(locked[key], dtype=np.float32) for key in ("mean", "std"))
    exponent_in, exponent_out = locked["input_exponent"], locked["output_exponent"]
    thresholds = {
        "run02_int8_deploy": locked["chosen"]["threshold_q"],
        "run02_int8_compare": locked["comparison"]["int8"]["threshold_q"],
        "run02_float_deploy": locked["float_reference"]["threshold_q"],
        "run02_float_compare": locked["comparison"]["float"]["threshold_q"],
    }
    original = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    samples = [row for row in original["samples"] if row["split"] == "test"]
    if [row["id"] for row in samples] != audit["test_subsets"]["all"]:
        raise ValueError("Audit TEST order differs from data/manifest.json")
    # All checks run before any TEST image is opened.
    verify_graph(graph, executor, locked, [row for row in original["samples"] if row["split"] == "calibration"])
    rows = baseline_rows(baseline, audit)
    previous, reference = reference_rows(audit, plan)
    rows.extend(previous)
    strict = set(audit["test_subsets"]["without_same_scene"])
    per_class = {
        name: np.zeros((len(CLASS_NAMES), 2), dtype=np.int64) for name in ("run02_int8_deploy", "run02_int8_compare")
    }
    deploy_masks = {}
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
        scores = {"int8": int8_scores(quant, exponent_out)[0], "float": float_scores(floating)[0]}
        for method, threshold in thresholds.items():
            predicted = scores["int8" if "int8" in method else "float"] >= threshold
            native = at_native_resolution(predicted.astype(np.uint8), labels).astype(bool)
            rows.append({"id": sample["id"], "method": method, **metrics(confusion(native, labels != 0))})
            if method in per_class and sample["id"] in strict:
                for class_id in range(len(CLASS_NAMES)):
                    selected = labels == class_id
                    per_class[method][class_id] += [np.count_nonzero(native[selected]), np.count_nonzero(selected)]
            if method == "run02_int8_deploy":
                deploy_masks[sample["id"]] = (frame, labels, predicted)
        if index % 20 == 0:
            print(f"Scored {index}/{len(samples)} TEST images", flush=True)
    methods = (*CANDIDATES, *REFERENCE_METHODS, *BASELINES)
    results = {
        name: {
            "images": len(audit["test_subsets"][name]),
            "methods": summarise(rows, audit["test_subsets"][name], methods),
            **{key: compare(rows, audit["test_subsets"][name], *pair) for key, pair in COMPARISONS.items()},
        }
        for name in SUBSETS
    }
    rates = {
        **{
            name: {CLASS_NAMES[c]: float(hits / total) for c, (hits, total) in enumerate(table) if total}
            for name, table in per_class.items()
        },
        **reference["rates"],
    }
    miou = {(row["id"], row["method"]): row["miou"] for row in rows}
    ordered = sorted(strict, key=lambda i: (miou[(i, "run02_int8_deploy")] - miou[(i, "run01_cap5")], i))
    chosen = ordered[:2] + ordered[len(ordered) // 2 : len(ordered) // 2 + 1] + ordered[-2:]
    sheet = Image.new("RGB", (860, 250 * len(chosen)), "white")
    for position, sample_id in enumerate(chosen):
        frame, labels, mask = deploy_masks[sample_id]
        sheet.paste(
            panel(
                frame,
                labels,
                mask,
                sample_id,
                (miou[(sample_id, "run01_cap5")], miou[(sample_id, "run02_int8_deploy")]),
            ),
            (0, 250 * position),
        )
    sheet.save(output / "examples.jpg", quality=90)
    summary = {
        "task": "Locked lite ESP-DL CNN (two pre-registered thresholds) vs run_01 and the locked forest on the audited SUIM TEST subsets; water/non-water, not collision distance",
        "architecture": locked["architecture"],
        "macs_per_frame": locked["macs_per_frame"],
        "conv_layers": locked["conv_layers"],
        "thresholds": {method: q - 128 if "int8" in method else q for method, q in thresholds.items()},
        "subsets": results,
        "strict_subset_class_rates": rates,
        "examples": {
            "ids": chosen,
            "rule": "strict subset: two largest regressions vs run_01 +5, median, two largest gains",
        },
        "checks": "Locked artifacts, sources, inputs and exploration verified; both thresholds re-derived from hashed curves; header checked; "
        "reloaded graph exponents, package versions and calibration counts at both thresholds reproduced; "
        "forest/UDCP and run_01 per-image counts match their own summaries",
        "locked_sha256": sha(run_dir / "LOCKED.json"),
        "baseline_summary_sha256": sha(baseline / "summary.json"),
        "baseline_per_image_sha256": sha(baseline / "per_image.csv"),
        **reference["sha256"],
        "dedup_manifest_sha256": plan["dedup_manifest"]["sha256"],
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": SEED},
        "source_sha256": {
            name: sha(resolve(name))
            for name in (
                "compare_cnn_lite.py",
                "compare_cnn_espdl.py",
                "cnn_lite.py",
                "cnn_seg.py",
                "train_cnn_lite.py",
                "train_cnn_espdl.py",
            )
        },
    }
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(output, summary, locked)
    dump_json(output / "summary.json", summary)
    return summary


def delta_text(delta: dict) -> str:
    return (
        f"漏检 {delta['fnr_percentage_points']:+.2f} 个百分点（95% 区间 {delta['fnr_ci95_pp'][0]:+.2f} ~ {delta['fnr_ci95_pp'][1]:+.2f}），"
        f"误检 {delta['fpr_percentage_points']:+.2f} 个百分点（{delta['fpr_ci95_pp'][0]:+.2f} ~ {delta['fpr_ci95_pp'][1]:+.2f}）；"
        f"逐图 mIoU 改善 {delta['paired_images']['improved']}、退步 {delta['paired_images']['regressed']}、持平 {delta['paired_images']['tied']}"
    )


def write_report(output: Path, summary: dict, locked: dict) -> None:
    chosen, comparison = locked["chosen"], locked["comparison"]["int8"]
    lines = [
        f"# 轻量 CNN（{locked['architecture']}）锁定评估",
        "",
        "网络结构由只用标定集的探索选出；模型、两个阈值与 ESP-DL 导出都在加载 TEST 前锁定，TEST 只评估这一次。"
        "int8 结果使用与 ESP32-P4 相同的输入量化码运行 ESP-PPQ 量化图。run_01 与森林、UDCP 的数字来自它们已记录的逐图结果。",
        "",
        f"run_02：{locked['macs_per_frame'] / 1e6:.1f} MMAC/帧（run_01 为 69.7），{locked['parameters']:,} 个参数，{locked['conv_layers']} 个卷积层，输出 80×60。",
        "",
        f"- 部署阈值码 {chosen['threshold_code']:+d}（预先登记：标定漏检上限 {locked['fnr_cap']:.0%}；标定漏检 {chosen['calibration_fnr']:.2%}，"
        f"误检 {chosen['calibration_fpr']:.2%}）。固件头文件与自检帧用这个阈值。",
        f"- 对照阈值码 {comparison['threshold_code']:+d}（与森林阈值 71、run_01 锁定阈值相同的上限 {locked['comparison']['fnr_cap']:.2%}；"
        f"标定漏检 {comparison['calibration_fnr']:.2%}，误检 {comparison['calibration_fpr']:.2%}）。",
        "",
        "run_01 的 +5 阈值是看过 TEST 后选的，它的 TEST 数字不是独立检验；这里列出只为同上限对照。",
        "",
    ]
    for name, label in SUBSETS.items():
        block = summary["subsets"][name]
        lines.extend(
            [
                f"## {label}（{block['images']} 张）",
                "",
                "| 方法 | 漏检 FNR | 水体误检 FPR | 两类全局 mIoU | 逐图平均 mIoU |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in (*CANDIDATES, *REFERENCE_METHODS, *BASELINES):
            m = block["methods"][method]
            lines.append(
                f"| {LABELS[method]} | {m['fnr']:.2%} | {m['water_false_positive_rate']:.2%} | {m['miou']:.4f} | {m['macro_image_miou']:.4f} |"
            )
        lines.extend(
            [
                "",
                f"- 部署阈值相对 run_01 +5（同为 5% 上限）：{delta_text(block['deploy_vs_run01_cap5'])}。",
                f"- 对照阈值相对 run_01 锁定 +11（同为森林上限）：{delta_text(block['compare_vs_run01_locked'])}。",
                f"- 部署阈值相对推荐森林：{delta_text(block['deploy_vs_forest_71'])}。",
                "",
            ]
        )
    rates = summary["strict_subset_class_rates"]
    lines.extend(
        [
            "## 最严格子集的逐类判为非水体比例",
            "",
            "水体一行是误检率，其余是召回率。",
            "",
            "| 类别 | run_02 部署 | run_02 对照 | run_01 +5 | run_01 +11 | 森林阈值 71 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    columns = ("run02_int8_deploy", "run02_int8_compare", "run01_cap5", "run01_locked", "forest_threshold_only")
    for name in CLASS_NAMES:
        lines.append(f"| {name} | " + " | ".join(f"{rates[column][name]:.1%}" for column in columns) + " |")
    lines.extend(
        [
            "",
            "区间为按图像有放回重抽样 2000 次的 95% 百分位区间；测试图多来自少数视频，真实不确定性可能更大。",
            "",
            "## 限制",
            "",
            "- 同一批 SUIM 照片的复用比较，不是 OV2640 实拍；水色、曝光、白平衡与防水窗差异需要实拍数据验证。",
            "- 结构选择用了与阈值相同的 80 张标定图，标定数字略偏乐观；TEST 未参与任何选择。",
            "- int8 结果是 ESP-PPQ 对 ESP32-P4 的仿真；板上一致性与速度由 water_seg_demo 实测确认。",
            "- 输出为水体/非水体分割，不是碰撞距离。",
            "",
            "![最严格子集：相对 run_01 +5 退步最大两张、中位一张、改善最大两张](examples.jpg)",
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
