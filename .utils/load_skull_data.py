import numpy as np
import pandas as pd
from pathlib import Path
from python_code.kinematics_core.kinematics_serialization import load_kinematics

def load_skull_data(analyzable_output_dir: Path) -> dict:
    skull_dir = analyzable_output_dir / "skull_kinematics"

    skull = load_kinematics(
        reference_geometry_path=skull_dir / "skull_reference_geometry.json",
        kinematics_csv_path=skull_dir / "skull_kinematics.csv",
    )

    df = pd.read_csv(skull_dir / "skull_kinematics.csv")

    return {
        "skull_timestamps": skull.timestamps,
        "position_x": df.loc[(df.trajectory == "position") & (df.component == "x"), "value"].values,
        "position_y": df.loc[(df.trajectory == "position") & (df.component == "y"), "value"].values,
        "linearVel_x": df.loc[(df.trajectory == "linear_velocity") & (df.component == "x"), "value"].values,
        "linearVel_y": df.loc[(df.trajectory == "linear_velocity") & (df.component == "y"), "value"].values,
        "roll":   np.rad2deg(skull.roll.values),
        "pitch":  np.rad2deg(skull.pitch.values),
        "yaw":    np.rad2deg(skull.yaw.values),
        "roll_v":  df.loc[(df.trajectory == "angular_velocity_local")     & (df.component == "roll"),  "value"].values,
        "pitch_v": df.loc[(df.trajectory == "angular_velocity_local")     & (df.component == "pitch"), "value"].values,
        "yaw_v":   df.loc[(df.trajectory == "angular_velocity_local")     & (df.component == "yaw"),   "value"].values,
        "roll_a":  df.loc[(df.trajectory == "angular_acceleration_local") & (df.component == "roll"),  "value"].values,
        "pitch_a": df.loc[(df.trajectory == "angular_acceleration_local") & (df.component == "pitch"), "value"].values,
        "yaw_a":   df.loc[(df.trajectory == "angular_acceleration_local") & (df.component == "yaw"),   "value"].values,
    }
