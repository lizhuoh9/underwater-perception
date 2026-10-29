"""Depthwise-separable variants of the ESP-DL water segmenter, for faster on-device inference.

Same input contract, encoder/decoder layout and 60x80 non-water logits as
cnn_seg.WaterSegNet (left unchanged: cnn_espdl_run_01 pins it). The stride-1 3x3
convolutions become depthwise 3x3 + pointwise 1x1, optionally the stride-2 ones
too. Still only Conv (grouped), Add and nearest Resize, all supported by esp-dl
3.3; channel widths stay multiples of 16 for esp-dl's aligned int8 kernels.
"""

from torch import nn

from uwseg.cnn_seg import WIDTHS, WaterSegNet

# Explored on the 80 calibration images only (see results/cnn_espdl_run_02/exploration).
ARCHITECTURES = {
    "lite": {"widths": WIDTHS, "separable_downsampling": False},
    "lite_dw_down": {"widths": WIDTHS, "separable_downsampling": True},
    "lite_wide": {"widths": (16, 48, 96), "separable_downsampling": True},
}


def separable(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    """Depthwise 3x3 then pointwise 1x1, each with BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin),
        nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class LiteWaterSegNet(WaterSegNet):
    """WaterSegNet with separable blocks; the forward pass is inherited unchanged.

    The parent builds its dense layers first and they are then replaced, in the order the
    calibration exploration used, so a given seed starts from the explored initial weights.
    """

    def __init__(self, widths: tuple[int, int, int] = WIDTHS, separable_downsampling: bool = False):
        super().__init__(widths)
        c1, c2, c3 = widths
        self.block2, self.fuse2, self.fuse1 = separable(c2, c2), separable(c2, c2), separable(c1, c1)
        self.block3 = nn.Sequential(separable(c3, c3), separable(c3, c3))
        if separable_downsampling:
            self.down2, self.down3 = separable(c1, c2, stride=2), separable(c2, c3, stride=2)


def build(architecture: str) -> LiteWaterSegNet:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {sorted(ARCHITECTURES)}")
    config = ARCHITECTURES[architecture]
    return LiteWaterSegNet(tuple(config["widths"]), config["separable_downsampling"])
