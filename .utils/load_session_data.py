import sys
from pathlib import Path

sys.path.insert(0, "/Users/benjaminscholl/Documents/bs")

DATA_DIR = Path(
    "/Users/benjaminscholl/Library/CloudStorage/Dropbox/projects/VisBehavDev"
    "/data/analyzable_outputs"
)

# Add .utils to path so sub-loaders can be imported directly
_utils_dir = Path(__file__).parent
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))

from load_skull_data import load_skull_data
from load_eye_quality import load_eye_quality
from load_eye_kinematics import load_eye_kinematics
from load_gaze_kinematics import load_gaze_kinematics
from load_toy_data import load_toy_data
from parse_session_name import parse_session_name


def load_session_data(session: str) -> dict:

    analyzable_output_dir = DATA_DIR / session
    
    assert analyzable_output_dir.exists(), f"Session directory not found: {analyzable_output_dir}"

    results = {}
    results.update(parse_session_name(session))
    results.update(load_skull_data(analyzable_output_dir))
    results.update(load_eye_quality(analyzable_output_dir))
    results.update(load_eye_kinematics(analyzable_output_dir, eye="left"))
    results.update(load_eye_kinematics(analyzable_output_dir, eye="right"))
    results.update(load_gaze_kinematics(analyzable_output_dir, eye="left"))
    results.update(load_gaze_kinematics(analyzable_output_dir, eye="right"))
    results.update(load_toy_data(analyzable_output_dir))

    # Frame count checks
    print(f"skull_timestamps : {len(results['skull_timestamps'])} frames")
    print(f"LE_timestamps    : {len(results['LE_timestamps'])} frames")
    print(f"EQtimestamps     : {len(results['EQtimestamps'])} frames")

    return results
