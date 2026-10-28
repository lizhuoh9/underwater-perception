"""Small helpers shared by the tools: JSON output and a font for figure labels."""

import json
from pathlib import Path

from PIL import ImageFont

FONTS = (Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/System/Library/Fonts/Supplemental/Arial.ttf"))


def dump_json(path: Path, data) -> None:
    """Indented UTF-8 JSON; NaN and infinity are refused rather than written as invalid JSON."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def font(size: int) -> ImageFont.ImageFont:
    """Arial or DejaVu Sans at `size` pixels, else Pillow's built-in bitmap font."""
    for path in FONTS:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()

