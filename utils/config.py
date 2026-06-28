import os
import sys
import importlib.util
from pathlib import Path

# ── Paths derived from repo layout ───────────────────────────────────────────
_EYETOOLS_ROOT = Path(__file__).resolve().parent.parent   # .../eyetools/
_BS_DIR = _EYETOOLS_ROOT.parent / "bs"                   # sibling bs/ repo

if not _BS_DIR.exists():
    import warnings
    warnings.warn(
        f"Expected 'bs' repo not found at {_BS_DIR} — "
        "imports from python_code.* will fail.",
        stacklevel=2,
    )

# ── Optional local_config.py (gitignored, for paths that can't be auto-detected)
_local_cfg: dict = {}
_local_file = _EYETOOLS_ROOT / "local_config.py"
if _local_file.exists():
    _spec = importlib.util.spec_from_file_location("local_config", _local_file)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _local_cfg = vars(_mod)

# ── Resolve EYETOOLS_DATA_DIR ─────────────────────────────────────────────────
_RELATIVE_DATA = Path("projects/VisBehavDev/data/analyzable_outputs")

def _find_dropbox_data_dir() -> str | None:
    cloudstorage = Path.home() / "Library" / "CloudStorage"
    candidates = list(cloudstorage.glob("Dropbox*")) if cloudstorage.exists() else []
    candidates.append(Path.home() / "Dropbox")
    for root in candidates:
        candidate = root / _RELATIVE_DATA
        if candidate.exists():
            return str(candidate)
    return None

_data_dir = (
    os.environ.get("EYETOOLS_DATA_DIR")
    or _local_cfg.get("EYETOOLS_DATA_DIR")
    or _find_dropbox_data_dir()
)

if not _data_dir:
    _cloudstorage = Path.home() / "Library" / "CloudStorage"
    raise EnvironmentError(
        "EYETOOLS_DATA_DIR could not be auto-detected.\n"
        f"Could not find '{_RELATIVE_DATA}' under any of:\n"
        f"  {_cloudstorage / 'Dropbox*'}\n"
        f"  ~/Dropbox\n\n"
        "To fix: create a file called local_config.py in the eyetools project root:\n"
        "    EYETOOLS_DATA_DIR = '/path/to/VisBehavDev/data/analyzable_outputs'\n"
        "    SAVELOC           = '/path/to/rawfigures'    # optional\n\n"
        "local_config.py is gitignored and will not be committed."
    )

DATA_DIR = Path(_data_dir)

# Expose SAVELOC from local_config.py if provided (used for saving figures)
SAVELOC = Path(_local_cfg["SAVELOC"]) if _local_cfg.get("SAVELOC") else None

# ── Add bs/ to sys.path so python_code.* imports work ────────────────────────
_bs = str(_BS_DIR)
if _bs not in sys.path:
    sys.path.insert(0, _bs)
