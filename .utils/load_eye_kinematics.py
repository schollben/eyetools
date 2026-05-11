import numpy as np
import pandas as pd
from pathlib import Path
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import load_ferret_eye_kinematics_from_directory


def load_eye_kinematics(analyzable_output_dir: Path, eye: str) -> dict:
    """
    eye: 'left' or 'right'
    """
    prefix = f"{eye}_eye"
    eye_dir = analyzable_output_dir / f"{prefix}_kinematics"

    ek = load_ferret_eye_kinematics_from_directory(input_directory=eye_dir, eye_name=prefix)

    df = pd.read_csv(eye_dir / f"{prefix}_kinematics.csv")
    pupil_major = df.loc[(df.trajectory == "pupil_axis") & (df.component == "major"), "value"].values

    p = eye[0].upper() + "E"  # "LE" or "RE"
    result = {
        f"{p}_x": np.rad2deg(ek.adduction_angle.values),
        f"{p}_y": np.rad2deg(ek.elevation_angle.values),
        f"{p}_vx": np.rad2deg(ek.eyeball.angular_velocity_local[:, 0]),
        f"{p}_vy": np.rad2deg(ek.eyeball.angular_velocity_local[:, 1]),
        f"{p}_ax": np.rad2deg(ek.eyeball.angular_acceleration_local[:, 0]),
        f"{p}_ay": np.rad2deg(ek.eyeball.angular_acceleration_local[:, 1]),
        f"{p}_pupil_major": pupil_major,
    }

    if eye == "left":
        result["LE_timestamps"] = ek.timestamps
        result["LE_n_frames"]   = ek.n_frames

    return result
