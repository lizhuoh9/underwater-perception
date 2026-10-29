"""Train a depthwise-separable ESP-DL water segmenter and lock two calibration operating points.

Needs the training extras (torch, esp-ppq, onnx) and the prepared data. Reuses the
data, augmentation, schedule, seed, ESP-PPQ quantization and device-exact
calibration replay of train_cnn_espdl.py; it differs in three ways:
- the architecture comes from cnn_lite.ARCHITECTURES and is recorded together with
  the calibration-only exploration that selected it (copied into exploration/);
- two thresholds are pre-registered on calibration: the deployment point (miss cap
  5%, for collision avoidance; written to the firmware header and self-test frame)
  and the comparison point (the forest's cap, like-for-like with forest-71 and run_01);
- LOCKED.json keeps run_01's layout ("chosen" = deployment) plus a "comparison" block.
No TEST image or label is loaded.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from uwseg.cnn_lite import ARCHITECTURES, build
from uwseg.cnn_seg import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
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
from uwseg.train_cnn_espdl import (
    BATCH,
    EPOCHS,
    LEARNING_RATE,
    OPSET,
    PPQ_CALIBRATION_IMAGES,
    SEED,
    TARGET,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    device_codes,
    exponents,
    load_inputs,
    quantize,
    sha,
    train_model,
    write_compile_req,
    write_header,
    write_test_frame,
)
from uwseg.train_pixel_forest import threshold_curve
from uwseg.validate import dump_json

DEPLOYMENT_FNR_CAP = 0.05
EXPLORATION_FILES = ("selection.json", "explore_variants.py", "explore_variants2.py", "variants.json", "variants2.json")
SOURCES = (
    "cnn_lite.py",
    "train_cnn_lite.py",
    "cnn_seg.py",
    "train_cnn_espdl.py",
    "optimize_pixel_forest.py",
    "train_pixel_forest.py",
    "udcp.py",
    "validate.py",
)


def conv_layers(model: torch.nn.Module) -> int:
    return sum(isinstance(module, torch.nn.Conv2d) for module in model.modules())


def load_selection(exploration: Path, architecture: str, macs: int, layers: int) -> dict:
    """The exploration record must name this architecture and agree with the module's cost."""
    missing = [name for name in EXPLORATION_FILES if not (exploration / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Exploration record incomplete: {missing}")
    record = json.loads((exploration / "selection.json").read_text(encoding="utf-8"))
    if record["chosen"] != architecture:
        raise ValueError(f"The exploration selected {record['chosen']!r}, not {architecture!r}")
    stated = record["candidates"][architecture]
    if (stated["macs"], stated["conv_layers"]) != (macs, layers):
        raise ValueError("Exploration costs disagree with cnn_lite's architecture")
    return record


def calibration_scores(
    model, graph, frames: np.ndarray, mean: np.ndarray, std: np.ndarray, input_exponent: int, output_exponent: int
) -> tuple[np.ndarray, np.ndarray]:
    """Float sigmoid scores and int8 output codes (+128) through the quantized graph from device-exact inputs."""
    from esp_ppq import TorchExecutor

    executor = TorchExecutor(graph=graph, device="cpu")
    with torch.no_grad():
        floating = model(
            normalise(torch.from_numpy(frames).permute(0, 3, 1, 2), torch.from_numpy(mean), torch.from_numpy(std))
        )
        quant = torch.cat(
            [
                executor.forward(inputs=torch.from_numpy(code[None].astype(np.float32)) * 2.0**input_exponent)[0]
                for code in device_codes(frames, mean, std, input_exponent)
            ]
        )
    return float_scores(floating), int8_scores(quant, output_exponent)


def run(output: Path, dedup_path: Path, architecture: str, exploration: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh empty --output directory")
    train, calibration, excluded, dedup = load_inputs(dedup_path)
    comparison_cap = json.loads((RESULTS_DIR / "forest_lowfp/LOCKED.json").read_text(encoding="utf-8"))[
        "calibration_fnr_cap"
    ]
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(architecture).to(device)
    macs, parameters, layers = count_macs(model), sum(p.numel() for p in model.parameters()), conv_layers(model)
    record = load_selection(exploration, architecture, macs, layers)
    output.mkdir(parents=True, exist_ok=True)
    (output / "exploration").mkdir()
    for name in EXPLORATION_FILES:
        shutil.copyfile(exploration / name, output / "exploration" / name)
    dump_json(
        output / "experiment_plan.json",
        {
            "algorithm": f"LiteWaterSegNet '{architecture}' (cnn_lite.py): WaterSegNet's stride-8 encoder and nearest-upsampling decoder "
            "with additive skips, stride-1 3x3 convs as depthwise 3x3 + pointwise 1x1"
            + (", stride-2 downsamplers too" if ARCHITECTURES[architecture]["separable_downsampling"] else "")
            + "; non-water logits at 60x80",
            "architecture": architecture,
            "architecture_config": ARCHITECTURES[architecture],
            "macs_per_frame": macs,
            "parameters": parameters,
            "conv_layers": layers,
            "input": [1, 3, INPUT_HEIGHT, INPUT_WIDTH],
            "output": [1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH],
            "architecture_selection": {
                **record,
                "files_sha256": {name: sha(output / "exploration" / name) for name in EXPLORATION_FILES},
            },
            "input_pipeline": "image stretched to 320x240 (bilinear), 2x2 average to 160x120, RGB565 truncation R&0xF8 G&0xFC B&0xF8 (esp-dl shift expansion), per-channel (v-mean)/std",
            "training": {
                "epochs": EPOCHS,
                "batch": BATCH,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "schedule": f"linear warmup {WARMUP_EPOCHS} epochs then cosine to zero; final weights, no epoch selection",
                "loss": "BCE with logits against area-averaged 60x80 soft non-water targets",
                "augmentation": "train_cnn_espdl.augment (as cnn_espdl_run_01)",
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
            "threshold_rule": "On the 80 calibration images, with device-exact int8 input codes through the quantized graph, minimize aggregate FPR subject to FNR <= cap (optimize_pixel_forest.constrained_threshold), once per operating point.",
            "fnr_cap": DEPLOYMENT_FNR_CAP,
            "operating_points": {
                "deployment": {
                    "fnr_cap": DEPLOYMENT_FNR_CAP,
                    "use": "firmware header kThresholdCode and self-test frame; collision avoidance favours fewer misses",
                },
                "comparison": {
                    "fnr_cap": comparison_cap,
                    "use": "same cap as forest threshold 71 and the run_01 locked threshold, for like-for-like comparison",
                },
            },
            "test_policy": "No TEST image or label is loaded here; model, both thresholds and exports are locked before compare_cnn_lite.py touches TEST.",
            "inputs_sha256": {
                name: sha(resolve(name))
                for name in ("data/manifest.json", "data/lowfp_training_manifest.json", "forest_lowfp/LOCKED.json")
            },
            "dedup_manifest": {"path": dedup_path.relative_to(REPO_ROOT).as_posix(), "sha256": sha(dedup_path)},
        },
    )
    print(
        f"Training {architecture} on {len(train)} images ({len(excluded)} audited neighbours excluded); "
        f"{macs / 1e6:.1f} MMAC, {layers} conv layers",
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
        {"state_dict": model.state_dict(), "architecture": architecture, "mean": mean.tolist(), "std": std.tolist()},
        output / "model.pt",
    )

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

    calibration_work = [load_work_frame(row, REPO_ROOT) for row in calibration]
    calibration_frames = np.stack([model_frame(image) for image, _ in calibration_work])
    calibration_labels = [labels_native for _, labels_native in calibration_work]
    floating, int8 = calibration_scores(model, graph, calibration_frames, mean, std, input_exponent, output_exponent)
    choices, curves = {}, {}
    for kind, scores in (("int8", int8), ("float", floating)):
        objective, counts = threshold_curve(list(zip(scores, calibration_labels)))
        curves[kind] = {"macro_image_miou": objective.tolist(), "counts": counts.tolist()}
        for point, cap in (("deployment", DEPLOYMENT_FNR_CAP), ("comparison", comparison_cap)):
            choices[point, kind] = constrained_threshold(counts, objective, cap)
    deployment_q = choices["deployment", "int8"]["threshold_q"]
    int8_masks = int8 >= deployment_q
    float_masks = floating >= choices["deployment", "float"]["threshold_q"]

    write_header(output / "water_seg_model_config.h", mean, std, input_exponent, output_exponent, deployment_q - 128)
    write_test_frame(
        output / "water_seg_test_frame.h", calibration_frames[0], int8_masks[0].astype(np.uint8), calibration[0]["id"]
    )
    ops = write_compile_req(
        output / "water_seg.compile_req.yml", (output / "water_seg.info").read_text(encoding="utf-8")
    )
    dump_json(output / "calibration_curves.json", curves)
    artifacts = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "LOCKED.json")
    from importlib.metadata import version

    comparison_q = choices["comparison", "int8"]["threshold_q"]
    locked = {
        "architecture": architecture,
        "chosen": {**choices["deployment", "int8"], "threshold_code": deployment_q - 128},
        "float_reference": choices["deployment", "float"],
        "fnr_cap": DEPLOYMENT_FNR_CAP,
        "comparison": {
            "fnr_cap": comparison_cap,
            "int8": {**choices["comparison", "int8"], "threshold_code": comparison_q - 128},
            "float": choices["comparison", "float"],
        },
        "calibration_float_vs_int8_mask_agreement": float((float_masks == int8_masks).mean()),
        "input_exponent": input_exponent,
        "output_exponent": output_exponent,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "macs_per_frame": macs,
        "parameters": parameters,
        "conv_layers": layers,
        "compile_req_ops": ops,
        "training_images": len(train),
        "train_seconds": train_seconds,
        "loss_history": history,
        "test_frame_id": calibration[0]["id"],
        "plan_sha256": sha(output / "experiment_plan.json"),
        "artifact_sha256": {name: sha(output / name) for name in artifacts},
        "exploration_sha256": {name: sha(output / "exploration" / name) for name in EXPLORATION_FILES},
        "source_sha256": {name: sha(resolve(name)) for name in SOURCES},
        "environment": {
            **{name: version(name) for name in ("torch", "esp-ppq", "onnx", "numpy", "pillow")},
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
    }
    dump_json(output / "LOCKED.json", locked)
    print(
        "LOCKED:",
        json.dumps(
            {key: locked[key] for key in ("architecture", "chosen", "comparison", "input_exponent", "output_exponent")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument(
        "--exploration",
        type=Path,
        required=True,
        help="folder with selection.json and the calibration-only exploration scripts and results",
    )
    parser.add_argument("--dedup-manifest", type=Path, default=DATA_DIR / "dedup_manifest.json")
    args = parser.parse_args()
    run(args.output.resolve(), args.dedup_manifest.resolve(), args.architecture, args.exploration.resolve())
