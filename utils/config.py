import os
from pathlib import Path

_data_dir = os.environ.get("EYETOOLS_DATA_DIR")
if not _data_dir:
    raise EnvironmentError(
        "EYETOOLS_DATA_DIR is not set.\n"
        "Set it to the path of your 'analyzable_outputs' directory, e.g.:\n"
        "  export EYETOOLS_DATA_DIR=/path/to/analyzable_outputs\n"
        "Or add it to a .env file and load it before importing eyetools."
    )

DATA_DIR = Path(_data_dir)
