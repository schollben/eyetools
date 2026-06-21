import os
import sys
from pathlib import Path

_data_dir = os.environ.get("EYETOOLS_DATA_DIR")
if not _data_dir:
    raise EnvironmentError(
        "EYETOOLS_DATA_DIR is not set.\n"
        "Set it to the path of your 'analyzable_outputs' directory, e.g.:\n"
        "  os.environ['EYETOOLS_DATA_DIR'] = '/path/to/analyzable_outputs'"
    )

_python_code_dir = os.environ.get("EYETOOLS_PYTHON_CODE_DIR")
if not _python_code_dir:
    raise EnvironmentError(
        "EYETOOLS_PYTHON_CODE_DIR is not set.\n"
        "Set it to the directory containing the 'python_code' package, e.g.:\n"
        "  os.environ['EYETOOLS_PYTHON_CODE_DIR'] = '/path/to/bs'"
    )

DATA_DIR = Path(_data_dir)

if _python_code_dir not in sys.path:
    sys.path.insert(0, _python_code_dir)
