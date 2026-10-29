"""Quantize a locked CNN run for the ESP32-P4 and write the model files the water_seg firmware component embeds.

The run's ONNX graph (default: results/cnn_espdl_run_02, the deployed lite CNN) is quantized with ESP-PPQ for target
esp32p4 the way train_cnn_espdl quantizes during training: 128 evenly spaced training frames, float inputs, 8 bits.
Into a fresh --output folder it writes water_seg.espdl (with ESP-PPQ test values for the board's self-test),
water_seg.info and water_seg.json (ESP-PPQ's dumps), water_seg_model_config.h, water_seg.compile_req.yml (the graph's
operators and every int8 ESP32-P4 conv kernel that esp-dl's spec/kernels.yml lists), water_seg_test_frame.h (the
first calibration frame and the mask the P4 graph gives it) and export.json.

The threshold stays the run's locked one: LOCKED.json's threshold code, rescaled if the P4 graph's output exponent
differs. That threshold was chosen on the run's own quantized calibration curve and is not re-selected on the P4
graph, and nothing here measures accuracy on TEST.

usage: python -m uwseg.export_p4 --kernels <esp-dl>/spec/kernels.yml --output <fresh folder> [--run <run folder>]
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from uwseg.cnn_seg import INPUT_HEIGHT, INPUT_WIDTH, load_work_frame, model_frame, normalise
from uwseg.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR
from uwseg.train_cnn_espdl import (
    PPQ_CALIBRATION_IMAGES,
    TARGET,
    device_codes,
    exponents,
    load_inputs,
    write_header,
    write_test_frame,
)

DEFAULT_RUN = RESULTS_DIR / "cnn_espdl_run_02"
# The run artifacts the export reads; their hashes must still equal the ones the run's LOCKED.json recorded.
EXPORTED_ARTIFACTS = ("model.pt", "water_seg.onnx")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p4_conv_kernels(spec_text: str) -> list[str]:
    """Every int8 ESP32-P4 conv kernel named in esp-dl's spec/kernels.yml, sorted and without repeats."""
    return sorted(set(re.findall(r"dl_esp32p4_s8_\w*conv2d\w*", spec_text)))


def graph_ops(info_text: str) -> list[str]:
    """The operators of an ESP-PPQ .info dump, plus RequantizeLinear (as train_cnn_espdl.write_compile_req)."""
    return sorted(set(re.findall(r"= (\w+)\[", info_text)) | {"RequantizeLinear"})


def locked_run(run: Path) -> dict:
    locked = json.loads((run / "LOCKED.json").read_text(encoding="utf-8"))
    for name in EXPORTED_ARTIFACTS:
        if sha(run / name) != locked["artifact_sha256"][name]:
            raise ValueError(f"{run.name}/{name} differs from the file its LOCKED.json recorded")
    return locked


def rescaled_threshold(code: int, exponent: int, output_exponent: int) -> int:
    """The same logit threshold as an int8 code at another output exponent."""
    return int(round(code * 2.0 ** (exponent - output_exponent)))


def export(run: Path, kernels_spec: Path, dedup_path: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh empty --output folder")
    locked = locked_run(run)
    kernels = p4_conv_kernels(kernels_spec.read_text(encoding="utf-8"))
    if not kernels:
        raise ValueError(f"No int8 ESP32-P4 conv kernel in {kernels_spec}")
    out.mkdir(parents=True, exist_ok=True)
    train, calibration, _, _ = load_inputs(dedup_path)
    saved = torch.load(run / "model.pt", map_location="cpu")
    mean, std = (np.array(saved[k], dtype=np.float32) for k in ("mean", "std"))
    index = np.linspace(0, len(train) - 1, PPQ_CALIBRATION_IMAGES).round().astype(int)
    frames = np.stack([model_frame(load_work_frame(train[i], REPO_ROOT)[0]) for i in index])
    inputs = normalise(torch.from_numpy(frames).permute(0, 3, 1, 2), torch.from_numpy(mean), torch.from_numpy(std))

    from esp_ppq.api import espdl_quantize_onnx

    graph = espdl_quantize_onnx(
        onnx_import_file=str(run / "water_seg.onnx"),
        espdl_export_file=str(out / "water_seg.espdl"),
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
    input_exponent, output_exponent = exponents(graph)
    locked_code, locked_exponent = locked["chosen"]["threshold_code"], locked["output_exponent"]
    threshold = rescaled_threshold(locked_code, locked_exponent, output_exponent)

    from esp_ppq import TorchExecutor

    executor = TorchExecutor(graph=graph, device="cpu")
    first = model_frame(load_work_frame(calibration[0], REPO_ROOT)[0])
    code = device_codes(first[None], mean, std, input_exponent)[0]
    with torch.no_grad():
        logits = executor.forward(inputs=torch.from_numpy(code[None].astype(np.float32)) * 2.0**input_exponent)[0]
    logit_codes = torch.clamp(torch.round(logits / 2.0**output_exponent), -128, 127).numpy().reshape(-1)
    mask = (logit_codes >= threshold).astype(np.uint8).reshape(logits.shape[-2], logits.shape[-1])

    write_header(out / "water_seg_model_config.h", mean, std, input_exponent, output_exponent, threshold)
    write_test_frame(out / "water_seg_test_frame.h", first, mask, calibration[0]["id"])
    ops = graph_ops((out / "water_seg.info").read_text(encoding="utf-8"))
    (out / "water_seg.compile_req.yml").write_text(
        f"# Generated by export_p4.py: {run.name} quantized for the ESP32-P4 (esp-dl 3.3.12 operator stripping).\n"
        "kernel_abi: 1\nops:\n"
        + "".join(f"  - {op}\n" for op in ops)
        + "kernels:\n"
        + "".join(f"  - {k}\n" for k in kernels),
        encoding="utf-8",
    )
    summary = {
        "onnx": f"{run.name}/water_seg.onnx",
        "target": TARGET,
        "bits": 8,
        "calibration": f"{PPQ_CALIBRATION_IMAGES} evenly spaced training frames (as train_cnn_espdl.py)",
        "input_exponent": input_exponent,
        "output_exponent": output_exponent,
        "threshold_code": threshold,
        "locked_threshold": f"code {locked_code} at output exponent {locked_exponent} ({run.name}/LOCKED.json)",
        "ops": ops,
        "kernels": len(kernels),
        "test_frame": calibration[0]["id"],
        "test_mask_non_water": int(mask.sum()),
    }
    (out / "export.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run", type=Path, default=DEFAULT_RUN, help="locked run folder with model.pt and water_seg.onnx"
    )
    parser.add_argument(
        "--kernels",
        type=Path,
        required=True,
        help="esp-dl's spec/kernels.yml (under managed_components/espressif__esp-dl once ESP-IDF has resolved esp-dl)",
    )
    parser.add_argument("--dedup-manifest", type=Path, default=DATA_DIR / "dedup_manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="fresh folder for the exported files")
    args = parser.parse_args()
    summary = export(args.run.resolve(), args.kernels.resolve(), args.dedup_manifest.resolve(), args.output.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
