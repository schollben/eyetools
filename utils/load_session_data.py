import sys
from pathlib import Path

# Add .utils to path so sub-loaders can be imported directly
_utils_dir = Path(__file__).parent
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))

from config import DATA_DIR
from load_skull_data import load_skull_data
from load_eye_quality import load_eye_quality
from load_eye_kinematics import load_eye_kinematics
from load_gaze_kinematics import load_gaze_kinematics
from load_toy_data import load_toy_data
from parse_session_name import parse_session_name
from session_data import SessionData


def load_session_data(session: str) -> SessionData:

    analyzable_output_dir = DATA_DIR / session

    assert analyzable_output_dir.exists(), f"Session directory not found: {analyzable_output_dir}"

    d = {}
    d.update(parse_session_name(session))
    d.update(load_skull_data(analyzable_output_dir))
    d.update(load_eye_quality(analyzable_output_dir, threshold="low"))
    d.update(load_eye_kinematics(analyzable_output_dir, eye="left"))
    d.update(load_eye_kinematics(analyzable_output_dir, eye="right"))
    d.update(load_gaze_kinematics(analyzable_output_dir, eye="left"))
    d.update(load_gaze_kinematics(analyzable_output_dir, eye="right"))
    d.update(load_toy_data(analyzable_output_dir))

    # Frame count checks
    print(f"skull_timestamps : {len(d['skull_timestamps'])} frames")
    print(f"LE_timestamps    : {len(d['LE_timestamps'])} frames")
    print(f"EQtimestamps     : {len(d['EQtimestamps'])} frames")

    return SessionData(**{k: v for k, v in d.items() if k in SessionData.__dataclass_fields__})
