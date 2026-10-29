"""Train the ESP-DL water segmenter, quantize it with ESP-PPQ and lock it on calibration only.

Needs the training extras (torch, esp-ppq, onnx) and the prepared data. In a fresh
--output directory it writes experiment_plan.json before training, then
model.pt, the ONNX and ESP-DL exports, the firmware config header, the
compile_req.yml for esp-dl's operator stripping, a self-test frame header,
calibration curves and LOCKED.json. No TEST image or label is loaded.
"""

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from uwseg.cnn_seg import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    RGB565_KEEP,
    WIDTHS,
    WORK_HEIGHT,
    WORK_WIDTH,
    WaterSegNet,
    count_macs,
    float_scores,
    int8_scores,
    load_work_frame,
    model_frame,
    normalise,
    work_labels,
)
from uwseg.optimize_pixel_forest import constrained_threshold
from uwseg.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR, resolve
from uwseg.train_pixel_forest import threshold_curve
from uwseg.validate import dump_json

SEED = 20261022
EPOCHS, BATCH, LEARNING_RATE, WEIGHT_DECAY, WARMUP_EPOCHS = 120, 32, 3e-3, 1e-4, 3
PPQ_CALIBRATION_IMAGES = 128
OPSET = 18
# ESP-PPQ target of the quantized graph: the board the firmware runs on.
TARGET = "esp32p4"
# Every int8 ESP32-P4 conv kernel in esp-dl 3.3.12 spec/kernels.yml (all names matching
# dl_esp32p4_s8_*conv2d*, as export_p4 reads them). Listing them all means a different ESP-PPQ
# fusion can never strip a kernel this model selects at runtime.
P4_INT8_CONV_KERNELS = [
    f"dl_esp32p4_s8_{name}"
    for name in (
        "conv2d_11cn",
        "conv2d_11cn_bias",
        "conv2d_11cn_bias_prelu",
        "conv2d_11cn_bias_relu",
        "conv2d_11cn_prelu",
        "conv2d_11cn_relu",
        "conv2d_33cn",
        "conv2d_33cn_bias",
        "conv2d_33cn_bias_prelu",
        "conv2d_33cn_bias_relu",
        "conv2d_33cn_prelu",
        "conv2d_33cn_relu",
        "conv2d_hwcn",
        "conv2d_hwcn_bias",
        "conv2d_hwcn_bias_prelu",
        "conv2d_hwcn_bias_relu",
        "conv2d_hwcn_prelu",
        "conv2d_hwcn_relu",
        "conv2d_per_channel_11cn",
        "conv2d_per_channel_11cn_bias",
        "conv2d_per_channel_11cn_bias_relu",
        "conv2d_per_channel_11cn_relu",
        "conv2d_per_channel_33cn",
        "conv2d_per_channel_33cn_bias",
        "conv2d_per_channel_33cn_bias_relu",
        "conv2d_per_channel_33cn_relu",
        "conv2d_per_channel_hwcn",
        "conv2d_per_channel_hwcn_bias",
        "conv2d_per_channel_hwcn_bias_relu",
        "conv2d_per_channel_hwcn_relu",
        "depthwise_conv2d_33c1",
        "depthwise_conv2d_33c1_bias",
        "depthwise_conv2d_33c1_bias_prelu",
        "depthwise_conv2d_33c1_bias_relu",
        "depthwise_conv2d_33c1_prelu",
        "depthwise_conv2d_33c1_relu",
        "depthwise_conv2d_hwc1",
        "depthwise_conv2d_hwc1_bias",
        "depthwise_conv2d_hwc1_bias_prelu",
        "depthwise_conv2d_hwc1_bias_relu",
        "depthwise_conv2d_hwc1_prelu",
        "depthwise_conv2d_hwc1_relu",
        "depthwise_conv2d_per_channel_33c1",
        "depthwise_conv2d_per_channel_33c1_bias",
        "depthwise_conv2d_per_channel_33c1_bias_relu",
        "depthwise_conv2d_per_channel_33c1_relu",
        "depthwise_conv2d_per_channel_hwc1",
        "depthwise_conv2d_per_channel_hwc1_bias",
        "depthwise_conv2d_per_channel_hwc1_bias_relu",
        "depthwise_conv2d_per_channel_hwc1_relu",
        "unaligned_conv2d_11cn",
        "unaligned_conv2d_11cn_bias",
        "unaligned_conv2d_11cn_bias_leakyrelu",
        "unaligned_conv2d_11cn_bias_prelu",
        "unaligned_conv2d_11cn_leakyrelu",
        "unaligned_conv2d_11cn_prelu",
        "unaligned_conv2d_33cn",
        "unaligned_conv2d_33cn_bias",
        "unaligned_conv2d_33cn_bias_leakyrelu",
        "unaligned_conv2d_33cn_bias_prelu",
        "unaligned_conv2d_33cn_leakyrelu",
        "unaligned_conv2d_33cn_prelu",
        "unaligned_conv2d_hwcn",
        "unaligned_conv2d_hwcn_bias",
        "unaligned_conv2d_hwcn_bias_leakyrelu",
        "unaligned_conv2d_hwcn_bias_prelu",
        "unaligned_conv2d_hwcn_leakyrelu",
        "unaligned_conv2d_hwcn_prelu",
        "unaligned_conv2d_per_channel_11cn",
        "unaligned_conv2d_per_channel_11cn_bias",
        "unaligned_conv2d_per_channel_11cn_bias_leakyrelu",
        "unaligned_conv2d_per_channel_11cn_leakyrelu",
        "unaligned_conv2d_per_channel_33cn",
        "unaligned_conv2d_per_channel_33cn_bias",
        "unaligned_conv2d_per_channel_33cn_bias_leakyrelu",
        "unaligned_conv2d_per_channel_33cn_leakyrelu",
        "unaligned_conv2d_per_channel_hwcn",
        "unaligned_conv2d_per_channel_hwcn_bias",
        "unaligned_conv2d_per_channel_hwcn_bias_leakyrelu",
        "unaligned_conv2d_per_channel_hwcn_leakyrelu",
        "unaligned_depthwise_conv2d_33c1",
        "unaligned_depthwise_conv2d_33c1_bias",
        "unaligned_depthwise_conv2d_33c1_bias_relu",
        "unaligned_depthwise_conv2d_33c1_relu",
        "unaligned_depthwise_conv2d_hwc1",
        "unaligned_depthwise_conv2d_hwc1_bias",
        "unaligned_depthwise_conv2d_hwc1_bias_relu",
        "unaligned_depthwise_conv2d_hwc1_relu",
        "unaligned_depthwise_conv2d_per_channel_33c1",
        "unaligned_depthwise_conv2d_per_channel_33c1_bias",
        "unaligned_depthwise_conv2d_per_channel_33c1_bias_relu",
        "unaligned_depthwise_conv2d_per_channel_33c1_relu",
        "unaligned_depthwise_conv2d_per_channel_hwc1",
        "unaligned_depthwise_conv2d_per_channel_hwc1_bias",
        "unaligned_depthwise_conv2d_per_channel_hwc1_bias_relu",
        "unaligned_depthwise_conv2d_per_channel_hwc1_relu",
    )
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(dedup_path: Path) -> tuple[list, list, list, dict]:
    """Calibration and training samples; training drops every audited neighbour of TEST or calibration."""
    dedup = json.loads(dedup_path.read_text(encoding="utf-8"))
    for name, digest in dedup["inputs"].items():
        if sha(resolve(name)) != digest:
            raise ValueError(f"Near-duplicate audit input changed: {name}")
    original = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    training = json.loads((DATA_DIR / "lowfp_training_manifest.json").read_text(encoding="utf-8"))
    calibration = [row for row in original["samples"] if row["split"] == "calibration"]
    neighbours = {
        n["id"]
        for leaks in (dedup["test_leaks"], dedup["calibration_leaks"])
        for leak in leaks.values()
        for n in leak["neighbours"]
        if n["split"] == "train"
    }
    train = [row for row in training["samples"] if row["id"] not in neighbours]
    if len(calibration) != 80 or len(training["samples"]) != 1395 or not train:
        raise ValueError("Expected the frozen 80 calibration and 1395 training samples")
    return train, calibration, sorted(neighbours), dedup


def augment(
    images: torch.Tensor, labels: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crop/zoom, flip and camera-like photometric jitter; soft 60x80 targets by area."""
    count, device = images.shape[0], images.device

    def uniform(low, high, *shape):
        return low + (high - low) * torch.rand(*shape, generator=generator, device=device)

    scale, aspect = uniform(0.6, 1.0, count), torch.exp(uniform(-0.15, 0.15, count))
    sx, sy = (scale * aspect).clamp(max=1.0), (scale / aspect).clamp(max=1.0)
    flip = torch.where(uniform(0, 1, count) < 0.5, -1.0, 1.0)
    theta = torch.zeros(count, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 2] = sx * flip, uniform(-1, 1, count) * (1 - sx)
    theta[:, 1, 1], theta[:, 1, 2] = sy, uniform(-1, 1, count) * (1 - sy)
    grid = F.affine_grid(theta, (count, 1, WORK_HEIGHT, WORK_WIDTH), align_corners=False)
    image = F.avg_pool2d(F.grid_sample(images.float(), grid, "bilinear", "border", align_corners=False), 2)
    target = F.avg_pool2d(F.grid_sample(labels.float(), grid, "nearest", "border", align_corners=False), 4)
    level = image.mean(dim=(1, 2, 3), keepdim=True)
    image = ((image - level) * uniform(0.7, 1.3, count, 1, 1, 1) + level) * uniform(0.7, 1.3, count, 1, 1, 1)
    image = image * uniform(0.85, 1.15, count, 3, 1, 1)
    image = image + torch.randn(image.shape, generator=generator, device=device) * uniform(0, 4, count, 1, 1, 1)
    keep = torch.tensor(RGB565_KEEP, device=device).view(1, 3, 1, 1)
    frames = image.clamp(0, 255).round().to(torch.uint8) & keep
    return normalise(frames, mean, std), target


def train_model(model, images, labels, mean, std, generator) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = math.ceil(len(images) / BATCH)
    total, warmup = EPOCHS * steps, WARMUP_EPOCHS * steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
        ),
    )
    history = []
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(images), generator=generator, device=images.device)
        running = 0.0
        for start in range(0, len(images), BATCH):
            index = order[start : start + BATCH]
            inputs, target = augment(images[index], labels[index], mean, std, generator)
            loss = F.binary_cross_entropy_with_logits(model(inputs), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(index)
        history.append(running / len(images))
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch + 1}/{EPOCHS} loss {history[-1]:.4f}", flush=True)
    return history


def device_codes(frames: np.ndarray, mean: np.ndarray, std: np.ndarray, exponent: int) -> np.ndarray:
    """esp-dl's NormQuant input LUT on the ESP32-P4: float32 (v - mean) * (1/std) * 2^-exp, rounded half to even.

    esp-dl 3.3.12 rounds half to even on the ESP32-P4 (dl::tool::round, CONFIG_ROUND_HALF_EVEN_ENABLED); it rounded
    half up, floor(x + 0.5), on the ESP32-S3 the committed CNN runs were quantized for.
    """
    inv_std = np.float32(1.0) / std.astype(np.float32)
    scaled = (frames.astype(np.float32) - mean.astype(np.float32)) * inv_std * np.float32(2.0**-exponent)
    codes = np.clip(np.rint(scaled.astype(np.float64)), -128, 127).astype(np.int8)
    return np.ascontiguousarray(codes.transpose(0, 3, 1, 2))


def selection(scores: list[np.ndarray], labels: list[np.ndarray], cap: float) -> tuple[dict, dict]:
    objective, counts = threshold_curve(list(zip(scores, labels)))
    return constrained_threshold(counts, objective, cap), {
        "macro_image_miou": objective.tolist(),
        "counts": counts.tolist(),
    }


def write_header(path: Path, mean, std, input_exponent: int, output_exponent: int, threshold_code: int) -> None:
    def floats(values):
        # "%.9g" round-trips float32; integral values need ".0" before the f suffix to stay C++ literals.
        texts = (f"{float(value):.9g}" for value in values)
        return ", ".join(f"{text}{'' if '.' in text or 'e' in text else '.0'}f" for text in texts)

    path.write_text(
        "// Generated by train_cnn_espdl.py together with water_seg.espdl; keep them paired.\n"
        "#pragma once\n#include <array>\n#include <cstdint>\n\nnamespace water_seg_config {\n"
        f"constexpr int kInputWidth = {INPUT_WIDTH};\nconstexpr int kInputHeight = {INPUT_HEIGHT};\n"
        f"constexpr int kOutputWidth = {OUTPUT_WIDTH};\nconstexpr int kOutputHeight = {OUTPUT_HEIGHT};\n"
        "// Device units: RGB565 channels expanded by left shift, as esp-dl's ImagePreprocessor does.\n"
        f"constexpr std::array<float, 3> kMean = {{{floats(mean)}}};\n"
        f"constexpr std::array<float, 3> kStd = {{{floats(std)}}};\n"
        f"constexpr int kInputExponent = {input_exponent};\nconstexpr int kOutputExponent = {output_exponent};\n"
        "// Non-water where the int8 output code >= kThresholdCode (128 never fires).\n"
        f"constexpr int kThresholdCode = {threshold_code};\n}}  // namespace water_seg_config\n",
        encoding="utf-8",
    )


def write_test_frame(path: Path, frame: np.ndarray, expected: np.ndarray, sample_id: str) -> None:
    packed = (
        ((frame[..., 0].astype(np.uint16) >> 3) << 11)
        | ((frame[..., 1].astype(np.uint16) >> 2) << 5)
        | (frame[..., 2] >> 3)
    )
    pixels = ",".join(str(int(v)) for v in packed.ravel())
    mask = ",".join(str(int(v)) for v in expected.ravel())
    path.write_text(
        f"// Calibration image {sample_id} as an RGB565 value array (load as DL_IMAGE_PIX_TYPE_RGB565LE)\n"
        "// and the mask the locked int8 model must produce from it.\n#pragma once\n#include <cstdint>\n\n"
        f'static const char kTestFrameId[] = "{sample_id}";\n'
        f"static const uint16_t kTestFrameRgb565[{packed.size}] = {{{pixels}}};\n"
        f"static const uint8_t kTestFrameExpectedMask[{expected.size}] = {{{mask}}};\n",
        encoding="utf-8",
    )


def write_compile_req(path: Path, info_text: str) -> list[str]:
    ops = sorted(set(re.findall(r"= (\w+)\[", info_text)) | {"RequantizeLinear"})
    lines = [
        "# Generated by train_cnn_espdl.py from water_seg.info for esp-dl 3.3.12 operator stripping.",
        "# Register it with esp_dl_compile_add_yml() wherever the model is linked.",
        "kernel_abi: 1",
        "ops:",
        *(f"  - {op}" for op in ops),
        "kernels:",
        *(f"  - {k}" for k in P4_INT8_CONV_KERNELS),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ops


def quantize(onnx_path: Path, espdl_path: Path, inputs: torch.Tensor):
    from esp_ppq import TargetPlatform
    from esp_ppq.api import espdl_quantize_onnx, export_ppq_graph

    graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=DataLoader(TensorDataset(inputs), batch_size=1, shuffle=False),
        calib_steps=len(inputs),
        input_shape=[1, 3, INPUT_HEIGHT, INPUT_WIDTH],
        target=TARGET,
        num_of_bits=8,
        collate_fn=lambda batch: batch[0],
        device="cpu",
        error_report=True,
        export_test_values=True,
        verbose=1,
    )
    export_ppq_graph(graph, platform=TargetPlatform.NATIVE, graph_save_to=str(espdl_path.with_name("water_seg_native")))
    return graph


def exponents(graph) -> tuple[int, int]:
    image = graph.variables["image"]
    consumer = image.dest_ops[0]
    input_scale = float(consumer.input_quant_config[consumer.inputs.index(image)].scale.item())
    logits = graph.outputs["logits"]
    output_scale = float(logits.source_op.output_quant_config[logits.source_op.outputs.index(logits)].scale.item())
    result = tuple(int(round(math.log2(scale))) for scale in (input_scale, output_scale))
    if any(2.0**e != s for e, s in zip(result, (input_scale, output_scale))):
        raise ValueError("ESP-DL tensors must use power-of-two scales")
    return result


def run(output: Path, dedup_path: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh empty --output directory")
    train, calibration, excluded, dedup = load_inputs(dedup_path)
    cap = json.loads((RESULTS_DIR / "forest_lowfp/LOCKED.json").read_text(encoding="utf-8"))["calibration_fnr_cap"]
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaterSegNet().to(device)
    macs, parameters = count_macs(model), sum(p.numel() for p in model.parameters())
    output.mkdir(parents=True, exist_ok=True)
    dump_json(
        output / "experiment_plan.json",
        {
            "algorithm": "WaterSegNet: strided conv encoder (16/32/64 ch, stride 8) and nearest-upsampling decoder with additive skips; non-water logits at 60x80",
            "macs_per_frame": macs,
            "parameters": parameters,
            "input": [1, 3, INPUT_HEIGHT, INPUT_WIDTH],
            "output": [1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH],
            "input_pipeline": "image stretched to 320x240 (bilinear), 2x2 average to 160x120, RGB565 truncation R&0xF8 G&0xFC B&0xF8 (esp-dl shift expansion), per-channel (v-mean)/std",
            "training": {
                "epochs": EPOCHS,
                "batch": BATCH,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "schedule": f"linear warmup {WARMUP_EPOCHS} epochs then cosine to zero; final weights, no epoch selection",
                "loss": "BCE with logits against area-averaged 60x80 soft non-water targets",
                "augmentation": "zoom-crop 0.6-1.0, aspect +-15%, horizontal flip, contrast/brightness 0.7-1.3, channel gain 0.85-1.15, Gaussian noise sigma 0-4, then RGB565 truncation",
                "seed": SEED,
            },
            "training_images": len(train),
            "excluded_training_neighbours": excluded,
            "normalisation": "per-channel mean/std of the un-augmented training model frames (device units)",
            "quantization": {
                "tool": "esp-ppq espdl_quantize_onnx",
                "target": TARGET,
                "bits": 8,
                "opset": OPSET,
                "calibration_images": f"{PPQ_CALIBRATION_IMAGES} evenly spaced training images, float inputs",
            },
            "threshold_rule": "On the 80 calibration images, with device-exact int8 input codes through the quantized graph, minimize aggregate FPR subject to FNR <= cap (optimize_pixel_forest.constrained_threshold), the rule and cap used for the recommended forest threshold 71.",
            "fnr_cap": cap,
            "test_policy": "No TEST image or label is loaded here; model, threshold and exports are locked before compare_cnn_espdl.py touches TEST.",
            "inputs_sha256": {
                name: sha(resolve(name))
                for name in ("data/manifest.json", "data/lowfp_training_manifest.json", "forest_lowfp/LOCKED.json")
            },
            "dedup_manifest": {"path": dedup_path.relative_to(REPO_ROOT).as_posix(), "sha256": sha(dedup_path)},
        },
    )
    print(
        f"Training on {len(train)} images ({len(excluded)} audited neighbours excluded); {macs / 1e6:.1f} MMAC",
        flush=True,
    )
    work_images, work_masks = [], []
    for row in train:
        image, labels_native = load_work_frame(row, REPO_ROOT)
        work_images.append(image)
        work_masks.append(work_labels(labels_native))
    frames = np.stack([model_frame(image) for image in work_images])
    mean = frames.reshape(-1, 3).mean(axis=0).astype(np.float32)
    std = frames.reshape(-1, 3).std(axis=0).astype(np.float32)
    images = torch.from_numpy(np.stack(work_images)).permute(0, 3, 1, 2).contiguous().to(device)
    labels = torch.from_numpy(np.stack(work_masks)).unsqueeze(1).to(device)
    del work_images, work_masks
    mean_t, std_t = (torch.from_numpy(v).to(device) for v in (mean, std))
    started = time.perf_counter()
    history = train_model(model, images, labels, mean_t, std_t, torch.Generator(device=device).manual_seed(SEED))
    train_seconds = time.perf_counter() - started
    del images, labels
    model = model.cpu().eval()
    torch.save(
        {"state_dict": model.state_dict(), "widths": WIDTHS, "mean": mean.tolist(), "std": std.tolist()},
        output / "model.pt",
    )

    calibration_work = [load_work_frame(row, REPO_ROOT) for row in calibration]
    calibration_frames = np.stack([model_frame(image) for image, _ in calibration_work])
    calibration_labels = [labels_native for _, labels_native in calibration_work]
    calibration_float = normalise(
        torch.from_numpy(calibration_frames).permute(0, 3, 1, 2), torch.from_numpy(mean), torch.from_numpy(std)
    )
    with torch.no_grad():
        float_logits = model(calibration_float)
    float_choice, float_curve = selection(list(float_scores(float_logits)), calibration_labels, cap)

    onnx_path, espdl_path = output / "water_seg.onnx", output / "water_seg.espdl"
    torch.onnx.export(
        model,
        torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH),
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    ppq_index = np.linspace(0, len(frames) - 1, PPQ_CALIBRATION_IMAGES).round().astype(int)
    ppq_inputs = normalise(
        torch.from_numpy(frames[ppq_index]).permute(0, 3, 1, 2), torch.from_numpy(mean), torch.from_numpy(std)
    )
    graph = quantize(onnx_path, espdl_path, ppq_inputs)
    input_exponent, output_exponent = exponents(graph)

    from esp_ppq import TorchExecutor

    executor = TorchExecutor(graph=graph, device="cpu")
    codes = device_codes(calibration_frames, mean, std, input_exponent)
    with torch.no_grad():
        quant_logits = torch.cat(
            [
                executor.forward(inputs=torch.from_numpy(code[None].astype(np.float32)) * 2.0**input_exponent)[0]
                for code in codes
            ]
        )
    int8 = int8_scores(quant_logits, output_exponent)
    int8_choice, int8_curve = selection(list(int8), calibration_labels, cap)
    threshold_code = int8_choice["threshold_q"] - 128
    float_masks = float_scores(float_logits) >= float_choice["threshold_q"]
    int8_masks = int8 >= int8_choice["threshold_q"]

    write_header(output / "water_seg_model_config.h", mean, std, input_exponent, output_exponent, threshold_code)
    write_test_frame(
        output / "water_seg_test_frame.h", calibration_frames[0], int8_masks[0].astype(np.uint8), calibration[0]["id"]
    )
    ops = write_compile_req(
        output / "water_seg.compile_req.yml", (output / "water_seg.info").read_text(encoding="utf-8")
    )
    dump_json(output / "calibration_curves.json", {"float": float_curve, "int8": int8_curve})
    artifacts = sorted(path.name for path in output.iterdir() if path.name not in ("LOCKED.json",))
    from importlib.metadata import version

    locked = {
        "chosen": {**int8_choice, "threshold_code": threshold_code},
        "float_reference": float_choice,
        "calibration_float_vs_int8_mask_agreement": float((float_masks == int8_masks).mean()),
        "fnr_cap": cap,
        "input_exponent": input_exponent,
        "output_exponent": output_exponent,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "macs_per_frame": macs,
        "parameters": parameters,
        "compile_req_ops": ops,
        "training_images": len(train),
        "train_seconds": train_seconds,
        "loss_history": history,
        "test_frame_id": calibration[0]["id"],
        "plan_sha256": sha(output / "experiment_plan.json"),
        "artifact_sha256": {name: sha(output / name) for name in artifacts if (output / name).is_file()},
        "source_sha256": {
            name: sha(resolve(name))
            for name in (
                "cnn_seg.py",
                "train_cnn_espdl.py",
                "optimize_pixel_forest.py",
                "train_pixel_forest.py",
                "udcp.py",
                "validate.py",
            )
        },
        "environment": {
            **{name: version(name) for name in ("torch", "esp-ppq", "onnx", "numpy", "pillow")},
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
    }
    dump_json(output / "LOCKED.json", locked)
    print(
        "LOCKED:",
        json.dumps({key: locked[key] for key in ("chosen", "float_reference", "input_exponent", "output_exponent")}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dedup-manifest", type=Path, default=DATA_DIR / "dedup_manifest.json")
    args = parser.parse_args()
    run(args.output.resolve(), args.dedup_manifest.resolve())
