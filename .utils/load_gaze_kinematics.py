import numpy as np
from pathlib import Path
from python_code.ferret_gaze.calculate_gaze.ferret_gaze_kinematics import FerretGazeKinematics


def load_gaze_kinematics(analyzable_output_dir: Path, eye: str) -> dict:
    """
    eye: 'left' or 'right'
    """
    gaze_name = f"{eye}_gaze"
    gaze = FerretGazeKinematics.load_from_directory(
        eye_name=gaze_name,
        input_directory=analyzable_output_dir / "gaze_kinematics",
    )

    ang_vel = gaze.kinematics.angular_velocity_local  # (N, 3)
    p = eye[0].upper() + "E"  # "LE" or "RE"

    return {
        f"{p}_gaze_timestamps":     gaze.kinematics.timestamps,
        f"{p}_gaze_horizontal_deg": gaze.horizontal_degrees,
        f"{p}_gaze_vertical_deg":   gaze.vertical_degrees,
        f"{p}_ang_vel_local_x_deg_s": np.rad2deg(ang_vel[:, 0]),
        f"{p}_ang_vel_local_y_deg_s": np.rad2deg(ang_vel[:, 1]),
    }
