"""Where the repository keeps its sources, inputs and results. Command-line defaults start here, so the tools are
meant to run from a source checkout (pip install -e .)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "core"
FIRMWARE_DIR = REPO_ROOT / "firmware"
RESULTS_DIR = REPO_ROOT / "results"
CLIP_RESULTS = RESULTS_DIR / "lake_clip"
