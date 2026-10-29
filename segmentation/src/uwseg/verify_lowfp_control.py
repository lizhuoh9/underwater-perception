"""Verify the pre-frozen threshold control as a separately compiled C model."""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from uwseg.compare_lowfp import panel, sha
from uwseg.compare_pixel_forest import check_native
from uwseg.forest_native import NativeForest
from uwseg.paths import DATA_DIR, RESULTS_DIR
from uwseg.pixel_forest import score_image
from uwseg.udcp import confusion, metrics, rgb565_roundtrip
from uwseg.validate import at_native_resolution, dump_json, load_sample


def run(parent: Path) -> dict:
    control = parent / "threshold_control"
    output = control / "evaluation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite control verification")
    locked = json.loads((parent / "LOCKED.json").read_text(encoding="utf-8"))
    export = json.loads((control / "MODEL_EXPORT.json").read_text(encoding="utf-8"))
    parent_summary = json.loads((parent / "evaluation/summary.json").read_text(encoding="utf-8"))
    if (
        sha(parent / "LOCKED.json") != export["calibration_lock_sha256"]
        or sha(parent / "LOCKED.json") != parent_summary["locked_sha256"]
    ):
        raise ValueError("Calibration lock differs from the evaluated/exported control")
    q = locked["threshold_only_control"]["threshold_q"]
    if q != export["threshold_q"] or sha(control / "model/pixel_forest_model.h") != export["exported_header_sha256"]:
        raise ValueError("Control export changed")
    old = RESULTS_DIR / "forest_comparison"
    old_lock = json.loads((old / "LOCKED.json").read_text(encoding="utf-8"))
    old_model_hash = old_lock["model_sha256"]["pixel_forest_model.json"]
    if (
        sha(old / "model/pixel_forest_model.json") != old_model_hash
        or sha(control / "model/pixel_forest_model.json") != old_model_hash
    ):
        raise ValueError("Control tree model differs from original")
    model = json.loads((control / "model/pixel_forest_model.json").read_text(encoding="utf-8"))
    native = NativeForest(control / "pixel_forest.dll")
    check_native(native, model, q)
    original = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((parent / "experiment_plan.json").read_text(encoding="utf-8"))
    if sha(DATA_DIR / "manifest.json") != plan["manifest_sha256"]["data/manifest.json"]:
        raise ValueError("Original image manifest changed")
    samples = [row for row in original["samples"] if row["split"] == "test"]
    with (parent / "evaluation/per_image.csv").open(encoding="utf-8") as handle:
        expected = {(row["id"], row["method"]): row for row in csv.DictReader(handle)}
    expected_ids = {sample_id for sample_id, method in expected if method == "forest_threshold_only"}
    if len(samples) != 110 or {row["id"] for row in samples} != expected_ids:
        raise ValueError("Control comparison does not contain the same110 IDs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir()
    counts = {name: np.zeros(4, dtype=np.int64) for name in ("forest_original", "forest_threshold_only")}
    rows, changes = [], []
    for sample in samples:
        rgb, labels = load_sample(sample)
        scores, masks = native.predict(rgb)
        np.testing.assert_array_equal(scores, score_image(rgb, model))
        np.testing.assert_array_equal(masks, scores >= q)
        predictions = {
            "forest_original": scores >= old_lock["chosen"]["threshold_q"],
            "forest_threshold_only": masks.astype(bool),
        }
        image_miou = []
        for method, predicted in predictions.items():
            current = confusion(at_native_resolution(predicted.astype(np.uint8), labels).astype(bool), labels != 0)
            target = expected[(sample["id"], method)]
            if tuple(current) != tuple(int(target[key]) for key in ("tp", "fp", "fn", "tn")):
                raise AssertionError("Compiled control mask changed previously recorded counts")
            counts[method] += current
            m = metrics(current)
            rows.append({"id": sample["id"], "method": method, **m})
            image_miou.append(m["miou"])
        panel(
            rgb565_roundtrip(rgb),
            labels,
            predictions["forest_original"],
            predictions["forest_threshold_only"],
            sample["id"],
            *image_miou,
        ).save(output / "overlays" / f"{sample['id']}.jpg", quality=92)
        changes.append({"id": sample["id"], "delta_miou": image_miou[1] - image_miou[0]})
    methods = {
        name: {
            **metrics(current),
            "macro_image_miou": float(np.mean([row["miou"] for row in rows if row["method"] == name])),
        }
        for name, current in counts.items()
    }
    before, after = methods.values()
    delta_fnr = before["foreground_recall"] - after["foreground_recall"]
    delta_fpr = after["water_false_positive_rate"] - before["water_false_positive_rate"]
    summary = {
        "role": "Previously frozen threshold-only control; engineering recommendation made after paired comparison, not preselected primary",
        "primary_experiment_goal_passed": parent_summary["test_goal_passed"],
        "threshold_q": q,
        "test_images": len(samples),
        "methods": methods,
        "control_goal_passed": bool(delta_fnr <= 0.02 and delta_fpr < 0),
        "fnr_increase_percentage_points": delta_fnr * 100,
        "fpr_change_percentage_points": delta_fpr * 100,
        "paired_images": {
            "improved": sum(r["delta_miou"] > 1e-12 for r in changes),
            "regressed": sum(r["delta_miou"] < -1e-12 for r in changes),
            "tied": sum(abs(r["delta_miou"]) <= 1e-12 for r in changes),
        },
        "model_arrays_bytes": 11992,
        "total_image_buffers_bytes_160x120": native.library.pf_workspace_bytes(160, 120) + 76800,
        "pixel_score_and_mask_mismatches": 0,
        "per_image_confusion_counts_match_frozen_control": True,
        "dll_sha256": sha(control / "pixel_forest.dll"),
        "source_sha256": sha(Path(__file__)),
        "calibration_lock_sha256": sha(parent / "LOCKED.json"),
    }
    dump_json(output / "summary.json", summary)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ordered = sorted(changes, key=lambda r: (r["delta_miou"], r["id"]))
    positive = [r for r in ordered if r["delta_miou"] > 0]
    ids = [positive[len(positive) // 2]["id"] if positive else ordered[-1]["id"], ordered[0]["id"]]
    canvas = Image.new("RGB", (1120, 500), "white")
    for index, sample_id in enumerate(ids):
        with Image.open(output / "overlays" / f"{sample_id}.jpg") as image:
            canvas.paste(image, (0, index * 250))
    canvas.save(output / "examples.jpg", quality=93)
    dump_json(
        output / "visual_selection.json",
        {"ids": ids, "rule": "Median positive mIoU delta followed by worst regression; visualization only"},
    )
    figures = []
    for row in ordered:
        safe = html.escape(row["id"], quote=True)
        figures.append(
            f'<figure><img loading="lazy" src="overlays/{safe}.jpg" alt="{safe}:input,GT,threshold57,threshold71"><figcaption>{safe}: delta mIoU={row["delta_miou"]:+.3f}</figcaption></figure>'
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>推荐阈值方案</title><style>body{font:16px/1.6 system-ui;max-width:1160px;margin:24px auto;padding:0 16px}img{width:100%;height:auto}figure{margin:24px 0}</style><h1>推荐方案：原森林阈值57→71</h1><p>阈值在标定阶段预先冻结；本页展示单独编译的C模型在同110张图上的复核。原模型不重训，不增加模型/图像缓冲和树推理比较次数。绿色GT，橙色旧阈值，紫色新阈值；全部图按退步到改善排序。</p><p>像素漏检5.93%→7.74%，水体误检27.44%→21.52%；这是多轮复用比较集，非独立盲测或碰撞距离验证。<a href="summary.json">结果JSON</a> · <a href="per_image.csv">逐图CSV</a> · <a href="../../evaluation/RESULTS.md">全实验报告</a></p>"""
    (output / "gallery.html").write_text(page + "".join(figures) + "</html>", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RESULTS_DIR / "forest_lowfp")
    run(parser.parse_args().run.resolve())
