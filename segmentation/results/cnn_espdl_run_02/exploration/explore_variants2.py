"""Exploration (calibration only, TEST untouched): lite depthwise model and obstacle-weighted loss.

Same data, recipe and seed as cnn_espdl_run_01; nothing here is locked or reported as a test result.

Record of the architecture exploration behind cnn_espdl_run_02 (variants2.json is its output, quantized
for the ESP32-S3 at the time). Ported to the uwseg package: it now quantizes through train_cnn_espdl, i.e. for
the ESP32-P4. Usage: python explore_variants2.py <fresh output folder> (needs CUDA and the prepared data).
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from uwseg import train_cnn_espdl as t
from uwseg.cnn_seg import (
    RGB565_KEEP,
    WIDTHS,
    WaterSegNet,
    count_macs,
    float_scores,
    int8_scores,
    load_work_frame,
    model_frame,
    normalise,
)
from uwseg.optimize_pixel_forest import constrained_threshold
from uwseg.paths import DATA_DIR, REPO_ROOT
from uwseg.train_pixel_forest import threshold_curve
from uwseg.validate import at_native_resolution

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
CLASS_WEIGHTS = torch.tensor(
    [1.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 1.0]
)  # water diver plants wreck robot reef fish seafloor


def dw_separable(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, 1, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )


def dw_down(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, 2, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin),
        nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class LiteWaterSegNet(WaterSegNet):
    """Same graph; the five stride-1 3x3 convs become depthwise 3x3 + pointwise 1x1."""

    def __init__(self, widths=WIDTHS, dw_down_layers=False):
        super().__init__(widths)
        c1, c2, c3 = widths
        self.block2, self.fuse2, self.fuse1 = dw_separable(c2), dw_separable(c2), dw_separable(c1)
        self.block3 = nn.Sequential(dw_separable(c3), dw_separable(c3))
        if dw_down_layers:
            self.down2, self.down3 = dw_down(c1, c2), dw_down(c2, c3)


def augment_classes(images, classes, mean, std, generator):
    """train_cnn_espdl.augment with class IDs kept, so pixels can be weighted by class."""
    count, device = images.shape[0], images.device

    def uniform(low, high, *shape):
        return low + (high - low) * torch.rand(*shape, generator=generator, device=device)

    scale, aspect = uniform(0.6, 1.0, count), torch.exp(uniform(-0.15, 0.15, count))
    sx, sy = (scale * aspect).clamp(max=1.0), (scale / aspect).clamp(max=1.0)
    flip = torch.where(uniform(0, 1, count) < 0.5, -1.0, 1.0)
    theta = torch.zeros(count, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 2] = sx * flip, uniform(-1, 1, count) * (1 - sx)
    theta[:, 1, 1], theta[:, 1, 2] = sy, uniform(-1, 1, count) * (1 - sy)
    grid = F.affine_grid(theta, (count, 1, t.WORK_HEIGHT, t.WORK_WIDTH), align_corners=False)
    image = F.avg_pool2d(F.grid_sample(images.float(), grid, "bilinear", "border", align_corners=False), 2)
    ids = F.grid_sample(classes.float(), grid, "nearest", "border", align_corners=False).long()
    target = F.avg_pool2d((ids != 0).float(), 4)
    weight = F.avg_pool2d(CLASS_WEIGHTS.to(device)[ids], 4)
    level = image.mean(dim=(1, 2, 3), keepdim=True)
    image = ((image - level) * uniform(0.7, 1.3, count, 1, 1, 1) + level) * uniform(0.7, 1.3, count, 1, 1, 1)
    image = image * uniform(0.85, 1.15, count, 3, 1, 1)
    image = image + torch.randn(image.shape, generator=generator, device=device) * uniform(0, 4, count, 1, 1, 1)
    frames = image.clamp(0, 255).round().to(torch.uint8) & torch.tensor(RGB565_KEEP, device=device).view(1, 3, 1, 1)
    return normalise(frames, mean, std), target, weight


def train(model, images, classes, mean, std, weighted):
    generator = torch.Generator(device=images.device).manual_seed(t.SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.LEARNING_RATE, weight_decay=t.WEIGHT_DECAY)
    steps = math.ceil(len(images) / t.BATCH)
    total, warmup = t.EPOCHS * steps, t.WARMUP_EPOCHS * steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup))),
    )
    for _epoch in range(t.EPOCHS):
        model.train()
        order = torch.randperm(len(images), generator=generator, device=images.device)
        for start in range(0, len(images), t.BATCH):
            index = order[start : start + t.BATCH]
            inputs, target, weight = augment_classes(images[index], classes[index], mean, std, generator)
            loss = F.binary_cross_entropy_with_logits(model(inputs), target, reduction="none")
            loss = (loss * weight).sum() / weight.sum() if weighted else loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()


def evaluate(scored, labels):
    objective, counts = threshold_curve(list(zip(scored, labels)))
    counts = counts.astype(float)
    fnr, fpr = counts[:, 2] / (counts[:, 0] + counts[:, 2]), counts[:, 1] / (counts[:, 1] + counts[:, 3])
    x, y = fpr[::-1], (1 - fnr)[::-1]
    choice = constrained_threshold(counts.astype(np.int64), objective, 0.05)
    recall = {}
    for name, cid in (("diver", 1), ("robot", 4), ("fish", 6), ("wreck", 3), ("reef", 5), ("seafloor", 7)):
        hit = tot = 0
        for score, lab in zip(scored, labels):
            native = at_native_resolution((score >= choice["threshold_q"]).astype(np.uint8), lab).astype(bool)
            sel = lab == cid
            hit, tot = hit + np.count_nonzero(native[sel]), tot + np.count_nonzero(sel)
        recall[name] = round(hit / tot, 4) if tot else None
    return {
        "auc": round(float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2)), 4),
        "fpr_at_fnr5": round(float(np.interp(0.05, fnr, fpr)), 4),
        "cap5_fnr": round(choice["calibration_fnr"], 4),
        "cap5_fpr": round(choice["calibration_fpr"], 4),
        "cap5_recall": recall,
    }


train_rows, calibration, excluded, _ = t.load_inputs(DATA_DIR / "dedup_manifest.json")
device = torch.device("cuda")
works, class_maps = [], []
for row in train_rows:
    work, native = load_work_frame(row, REPO_ROOT)
    works.append(work)
    class_maps.append(np.array(Image.fromarray(native).resize((t.WORK_WIDTH, t.WORK_HEIGHT), Image.Resampling.NEAREST)))
frames = np.stack([model_frame(w) for w in works])
mean, std = frames.reshape(-1, 3).mean(0).astype(np.float32), frames.reshape(-1, 3).std(0).astype(np.float32)
images = torch.from_numpy(np.stack(works)).permute(0, 3, 1, 2).contiguous().to(device)
classes = torch.from_numpy(np.stack(class_maps)).unsqueeze(1).to(device)
del works, class_maps
mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
cal = [load_work_frame(row, REPO_ROOT) for row in calibration]
cal_frames = np.stack([model_frame(w) for w, _ in cal])
cal_labels = [lab for _, lab in cal]
cal_inputs = normalise(torch.from_numpy(cal_frames).permute(0, 3, 1, 2), torch.from_numpy(mean), torch.from_numpy(std))
print(f"train {len(train_rows)} images, calibration {len(cal)}; mean {mean}, std {std}", flush=True)

results = {}
for name, build, weighted, quantize in (
    ("lite_dw_down", lambda: LiteWaterSegNet(dw_down_layers=True), False, True),
    ("lite_wide", lambda: LiteWaterSegNet((16, 48, 96), dw_down_layers=True), False, True),
):
    torch.manual_seed(t.SEED)
    model = build().to(device)
    macs = count_macs(model)
    started = time.perf_counter()
    train(model, images, classes, mean_t, std_t, weighted)
    model = model.cpu().eval()
    with torch.no_grad():
        logits = model(cal_inputs)
    entry = {
        "macs": macs,
        "params": sum(p.numel() for p in model.parameters()),
        "train_seconds": round(time.perf_counter() - started, 1),
        "float": evaluate(list(float_scores(logits)), cal_labels),
    }
    if quantize:
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        onnx_path = folder / "water_seg.onnx"
        torch.onnx.export(
            model,
            torch.zeros(1, 3, 120, 160),
            onnx_path,
            input_names=["image"],
            output_names=["logits"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
        index = np.linspace(0, len(frames) - 1, t.PPQ_CALIBRATION_IMAGES).round().astype(int)
        graph = t.quantize(
            onnx_path,
            folder / "water_seg.espdl",
            normalise(
                torch.from_numpy(frames[index]).permute(0, 3, 1, 2), torch.from_numpy(mean), torch.from_numpy(std)
            ),
        )
        e_in, e_out = t.exponents(graph)
        from esp_ppq import TorchExecutor

        ex = TorchExecutor(graph=graph, device="cpu")
        codes = t.device_codes(cal_frames, mean, std, e_in)
        with torch.no_grad():
            quant = torch.cat(
                [ex.forward(inputs=torch.from_numpy(c[None].astype(np.float32)) * 2.0**e_in)[0] for c in codes]
            )
        entry["int8"] = evaluate(list(int8_scores(quant, e_out)), cal_labels)
        entry["espdl_bytes"] = (folder / "water_seg.espdl").stat().st_size
    results[name] = entry
    print(name, json.dumps(entry), flush=True)
(OUT / "variants2.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
