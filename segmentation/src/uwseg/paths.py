"""Where the repository keeps its data, sources and results. Command-line defaults start here, so the tools are
meant to run from a source checkout (pip install -e .).

Manifests and locks name files relative to the project root as the research layout had them: data under
`data/`, sources by module file name (`udcp.py`) or under `embedded/`, result sets by their folder name
(`forest_lowfp/LOCKED.json`). `resolve` maps such a name to its file in this layout, where the result sets live
under `results/` and the modules in this package.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
EMBEDDED_DIR = REPO_ROOT / "embedded"
FIRMWARE_DIR = REPO_ROOT / "firmware"
# Optional local installs of the forest-training packages (scikit-learn and its dependencies).
LOCAL_DEPS_DIR = REPO_ROOT / ".deps"


def resolve(name: str) -> Path:
    """The file a recorded name refers to: data/... and embedded/... from the project root, a module (no
    folder, .py) from this package, anything else from results/."""
    if name.startswith(("data/", "embedded/")):
        return REPO_ROOT / name
    if "/" not in name and name.endswith(".py"):
        return PACKAGE_DIR / name
    return RESULTS_DIR / name
