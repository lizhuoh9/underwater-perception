"""Small CNN water/non-water segmenter for ESP-DL, shared by training, export and evaluation.

Model inputs reproduce what esp-dl's ImagePreprocessor feeds the network from an
OV2640 RGB565 frame: each channel expanded by left shift (R<<3, G<<2, B<<3, no
bit replication), then (value - mean) / std. Outputs are non-water logits at
half the input resolution (60x80 for a 160x120 frame).
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from uwseg.udcp import decode_labels

INPUT_HEIGHT, INPUT_WIDTH = 120, 160
# Augmentation samples a 2x working copy and averages 2x2 blocks into the model input.
WORK_HEIGHT, WORK_WIDTH = 2 * INPUT_HEIGHT, 2 * INPUT_WIDTH
OUTPUT_HEIGHT, OUTPUT_WIDTH = INPUT_HEIGHT // 2, INPUT_WIDTH // 2
RGB565_KEEP = np.array([0xF8, 0xFC, 0xF8], dtype=np.uint8)
WIDTHS = (16, 32, 64)


def load_work_frame(sample: dict, root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Hash-checked 320x240 working copy (stretched, like a 4:3 camera frame) and native labels."""
    for field in ("image", "mask"):
        path = root / sample[field]
        if hashlib.sha256(path.read_bytes()).hexdigest() != sample[field + "_sha256"]:
            raise ValueError(f"Modified {field}: {path}")
    with Image.open(root / sample["image"]) as image, Image.open(root / sample["mask"]) as mask:
        if image.size != mask.size:
            raise ValueError(f"Image/mask geometry mismatch: {sample['id']}")
        labels = decode_labels(np.array(mask.convert("RGB")))
        work = np.array(image.convert("RGB").resize((WORK_WIDTH, WORK_HEIGHT), Image.Resampling.BILINEAR))
    return work, labels


def work_labels(labels: np.ndarray) -> np.ndarray:
    """Binary non-water labels sampled (nearest) onto the 320x240 working grid."""
    binary = Image.fromarray((labels != 0).astype(np.uint8))
    return np.array(binary.resize((WORK_WIDTH, WORK_HEIGHT), Image.Resampling.NEAREST))


def model_frame(work: np.ndarray) -> np.ndarray:
    """Un-augmented model input: 2x2 block average, then the device's RGB565 truncation."""
    pooled = work.reshape(INPUT_HEIGHT, 2, INPUT_WIDTH, 2, 3).astype(np.float32).mean(axis=(1, 3))
    return np.rint(pooled).astype(np.uint8) & RGB565_KEEP


def normalise(frames: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW frames in device units -> float network input."""
    return (frames.float() - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def conv_bn(cin: int, cout: int, kernel: int = 3, stride: int = 1, relu: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, kernel, stride, kernel // 2, bias=False), nn.BatchNorm2d(cout)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class WaterSegNet(nn.Module):
    """Encoder at strides 2/4/8 and a nearest-upsampling decoder with additive skips.

    Only Conv(+BN folded, ReLU fused), Add and nearest Resize, all supported by
    esp-dl 3.3; channel widths are multiples of 16 for esp-dl's aligned int8 kernels.
    """

    def __init__(self, widths: tuple[int, int, int] = WIDTHS):
        super().__init__()
        c1, c2, c3 = widths
        self.stem = conv_bn(3, c1, stride=2)
        self.down2 = conv_bn(c1, c2, stride=2)
        self.block2 = conv_bn(c2, c2)
        self.down3 = conv_bn(c2, c3, stride=2)
        self.block3 = nn.Sequential(conv_bn(c3, c3), conv_bn(c3, c3))
        self.lateral3 = conv_bn(c3, c2, kernel=1, relu=False)
        self.fuse2 = conv_bn(c2, c2)
        self.lateral2 = conv_bn(c2, c1, kernel=1, relu=False)
        self.fuse1 = conv_bn(c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.stem(x)
        e2 = self.block2(self.down2(e1))
        e3 = self.block3(self.down3(e2))
        d2 = self.fuse2(F.interpolate(self.lateral3(e3), scale_factor=2.0, mode="nearest") + e2)
        d1 = self.fuse1(F.interpolate(self.lateral2(d2), scale_factor=2.0, mode="nearest") + e1)
        return self.head(d1)


def count_macs(model: nn.Module) -> int:
    """Multiply-accumulates of every convolution for one 160x120 frame."""
    total = 0

    def hook(module, _inputs, output):
        nonlocal total
        per_output = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        total += output[0].numel() * per_output

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device))
    for handle in handles:
        handle.remove()
    return total


def float_scores(logits: torch.Tensor) -> np.ndarray:
    """N x 1 x H x W logits -> uint8 probabilities on 0..255."""
    return torch.sigmoid(logits).mul(255).round().clamp(0, 255).to(torch.uint8)[:, 0].cpu().numpy()


def int8_scores(logits: torch.Tensor, exponent: int) -> np.ndarray:
    """Fake-quantized logits -> int8 output codes shifted to 0..255 (code + 128)."""
    codes = torch.round(logits / 2.0**exponent).clamp(-128, 127)
    return (codes + 128).to(torch.uint8)[:, 0].cpu().numpy()
