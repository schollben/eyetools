# %%
# Setup: point Python at the bs repo so python_code imports work

import sys
sys.path.insert(0, "/Users/benjaminscholl/Documents/bs")

from pathlib import Path
import numpy as np
import plotly.express as px

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import load_ferret_eye_kinematics_from_directory
from python_code.kinematics_core.kinematics_serialization import load_kinematics


DATA_DIR = Path(
    "/Users/benjaminscholl/Library/CloudStorage/Dropbox/projects/VisBehavDev"
    "/data/analyzable_outputs"
)
SESSION = "session_2025-06-28_ferret_753_EyeCameras_P30_EO2_analyzable_output"
ANALYZABLE_OUTPUT_DIR = DATA_DIR / SESSION
print(f"Path exists: {ANALYZABLE_OUTPUT_DIR.exists()}")

# %% load up data

left_eye = load_ferret_eye_kinematics_from_directory(
    input_directory=ANALYZABLE_OUTPUT_DIR / "left_eye_kinematics",
    eye_name="left_eye",
)

right_eye = load_ferret_eye_kinematics_from_directory(
    input_directory=ANALYZABLE_OUTPUT_DIR / "right_eye_kinematics",
    eye_name="right_eye",
)


# %%
# Plot left and right eye rotations (adduction + elevation)

# Build time axes starting from 0
t_left  = left_eye.eyeball.timestamps  - left_eye.eyeball.timestamps[0]
t_right = right_eye.eyeball.timestamps - right_eye.eyeball.timestamps[0]

left_adduction  = np.degrees(left_eye.adduction_angle.values)
left_elevation  = np.degrees(left_eye.elevation_angle.values)
right_adduction = np.degrees(right_eye.adduction_angle.values)
right_elevation = np.degrees(right_eye.elevation_angle.values)

fig = px.line(
    x=t_left,
    y=[left_adduction, right_adduction],
    labels={"x": "Time (s)", "y": "Degrees"},
    title="Eye Adduction (+ = toward nose)",
)
fig.show()

# Adduction
fig = px.line(
    x=t_left,
    y=[left_adduction, right_adduction],
    labels={"x": "Time (s)", "y": "Degrees"},
    title="Eye Adduction (+ = toward nose)",
)
fig.show()

# axes[0].set_title("Eye Adduction (+ = toward nose)")
# axes[0].legend()

# Elevation
fig = px.line(
    x=t_left,
    y=[left_elevation, right_elevation],
    labels={"x": "Time (s)", "y": "Degrees"},
    title="Eye Elevation (+ = up)",
)
fig.show()


# %%
# Load skull kinematics (head rotations)

from python_code.kinematics_core.kinematics_serialization import load_kinematics

skull = load_kinematics(
    reference_geometry_path=ANALYZABLE_OUTPUT_DIR / "skull_kinematics" / "skull_reference_geometry.json",
    kinematics_csv_path=ANALYZABLE_OUTPUT_DIR / "skull_kinematics" / "skull_kinematics.csv",
)

t_skull    = skull.timestamps - skull.timestamps[0]
head_roll  = np.degrees(skull.roll.values)
head_pitch = np.degrees(skull.pitch.values)
head_yaw   = np.degrees(skull.yaw.values)

print(f"Skull frames: {skull.timestamps.shape[0]}")


# %%
# Load eye data quality flags
# Long-format: one row per (frame, eye, threshold). Value 0 = bad frame.

import pandas as pd

quality_raw = pd.read_csv(ANALYZABLE_OUTPUT_DIR / "eye_data_quality.csv")

# Pivot to wide: rows=frames, columns=(trajectory, component)
quality = quality_raw.pivot_table(
    index=["frame", "timestamp_s"],
    columns=["trajectory", "component"],
    values="value",
)
# Flatten MultiIndex before reset_index -> e.g. "left_eye_data_quality_low_threshold"
quality.columns = ["_".join(c) for c in quality.columns]
quality = quality.reset_index()

# Convenience boolean masks: True = good frame, False = bad
left_good  = quality["left_eye_data_quality_low_threshold"].astype(bool)
right_good = quality["right_eye_data_quality_low_threshold"].astype(bool)

print(f"Quality rows: {len(quality)}")
print(f"Left  bad frames: {(~left_good).sum()} / {len(left_good)}")
print(f"Right bad frames: {(~right_good).sum()} / {len(right_good)}")
print(quality.head())

# %%
