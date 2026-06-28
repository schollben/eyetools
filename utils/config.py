import sys
import warnings
from pathlib import Path

_EYETOOLS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EYETOOLS_ROOT))

import local_config  # type: ignore  # gitignored — copy local_config.py.example and fill in paths

DATA_DIR = Path(local_config.EYETOOLS_DATA_DIR)
SAVELOC  = Path(local_config.SAVELOC) if hasattr(local_config, "SAVELOC") else None

if not DATA_DIR.exists():
    warnings.warn(f"EYETOOLS_DATA_DIR not found: {DATA_DIR}\nUpdate local_config.py.")

_bs = local_config.EYETOOLS_PYTHON_CODE_DIR
if not Path(_bs).exists():
    warnings.warn(f"EYETOOLS_PYTHON_CODE_DIR not found: {_bs}\nUpdate local_config.py. Imports from python_code.* will fail.")
elif _bs not in sys.path:
    sys.path.insert(0, _bs)

