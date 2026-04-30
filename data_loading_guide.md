# Data Loading Guide — Ferret Kinematics

How to load and work with ferret eye and gaze data from any Python environment.

---

## 1. Pointing Python at the codebase

All the data loaders live in `python_code/` inside the `bs` repo. To import them from any environment, add the repo root to your Python path at the top of your script:

```python
import sys
sys.path.insert(0, "/Users/benjaminscholl/Documents/bs")
```

After this, `from python_code.xxx import yyy` will work regardless of which Python environment you're running.

---

## 2. Data folder structure

Each session's processed data lives in an `analyzable_output` folder:

```
{session}_analyzable_output/
├── left_eye_kinematics/
│   ├── left_eye_kinematics.csv       ← time-varying pose data
│   └── left_eye_reference_geometry.json  ← static keypoint geometry
├── right_eye_kinematics/
│   ├── right_eye_kinematics.csv
│   └── right_eye_reference_geometry.json
├── gaze_kinematics/
│   ├── left_gaze_kinematics.csv      ← eye + head combined, world space
│   ├── right_gaze_kinematics.csv
│   ├── left_gaze_reference_geometry.json
│   └── right_gaze_reference_geometry.json
└── skull_kinematics/
    ├── skull_kinematics.csv
    └── skull_reference_geometry.json
```

Point `ANALYZABLE_OUTPUT_DIR` at this folder and build sub-paths from it.

---

## 3. Key data classes

### `RigidBodyKinematics`
Generic rigid body motion. The base class underlying everything.

```
from python_code.kinematics_core.kinematics_serialization import load_kinematics
```

| Property | Shape | Description |
|---|---|---|
| `.timestamps` | (N,) | Time in seconds |
| `.position_xyz` | (N, 3) | Position in mm |
| `.quaternions_wxyz` | (N, 4) | Orientation as [w, x, y, z] |
| `.velocity_xyz` | (N, 3) | Linear velocity mm/s |
| `.acceleration_xyz` | (N, 3) | Linear acceleration mm/s² |
| `.angular_velocity_local` | (N, 3) | Angular velocity in body frame rad/s |
| `.angular_velocity_global` | (N, 3) | Angular velocity in world frame rad/s |
| `.roll`, `.pitch`, `.yaw` | Timeseries | Euler angles in radians |

All derived quantities (velocity, acceleration, Euler angles) are computed lazily on first access.

---

### `FerretEyeKinematics`
Eye rotation in the **eye camera frame** — purely the eye, not combined with head movement.

```python
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import (
    load_ferret_eye_kinematics_from_directory,
)

left_eye = load_ferret_eye_kinematics_from_directory(
    eye_name="left_eye",
    directory=ANALYZABLE_OUTPUT_DIR / "left_eye_kinematics",
)
```

| Property | Type | Description |
|---|---|---|
| `.eyeball` | RigidBodyKinematics | Underlying rigid body (use for `.timestamps`) |
| `.eye_side` | str | `'left'` or `'right'` |
| `.n_frames` | int | Number of frames |
| `.adduction_angle` | Timeseries | + = toward nose, − = away (radians) |
| `.elevation_angle` | Timeseries | + = up, − = down (radians) |
| `.torsion_angle` | Timeseries | + = extorsion, − = intorsion (radians) |
| `.adduction_velocity` | Timeseries | rad/s |
| `.elevation_velocity` | Timeseries | rad/s |
| `.torsion_velocity` | Timeseries | rad/s |
| `.azimuth_degrees` | (N,) array | Horizontal angle in degrees |
| `.elevation_degrees` | (N,) array | Vertical angle in degrees |

**Timeseries** objects have two properties:
- `.values` — numpy array of values
- `.timestamps` — numpy array of timestamps in seconds

To convert to degrees: `np.degrees(left_eye.adduction_angle.values)`

---

### `FerretGazeKinematics`
World-space gaze — eye rotation **combined with head movement**. Use this when you want to know where the ferret is actually looking in the room.

```python
from python_code.ferret_gaze.calculate_gaze.ferret_gaze_kinematics import FerretGazeKinematics

left_gaze = FerretGazeKinematics.load_from_directory(
    eye_name="left_gaze",
    input_directory=ANALYZABLE_OUTPUT_DIR / "gaze_kinematics",
)
```

| Property | Type | Description |
|---|---|---|
| `.kinematics` | RigidBodyKinematics | Underlying rigid body in world space |
| `.eye_side` | str | `'left'` or `'right'` |
| `.horizontal_degrees` | (N,) array | World-space horizontal gaze angle |
| `.vertical_degrees` | (N,) array | World-space vertical gaze angle |
| `.horizontal_velocity` | Timeseries | rad/s |
| `.vertical_velocity` | Timeseries | rad/s |
| `.horizontal_acceleration` | Timeseries | rad/s² |
| `.vertical_acceleration` | Timeseries | rad/s² |
| `.gaze_directions` | (N, 3) array | Unit vectors in world space |

---

## 4. Color conventions

Consistent across all plots in this codebase:

| Eye | Channel | Color |
|---|---|---|
| Left | Primary (adduction / horizontal) | `#0096FF` (blue) |
| Left | Secondary (elevation / vertical) | `#64B4FF` (light blue) |
| Right | Primary (adduction / horizontal) | `#FF6400` (orange) |
| Right | Secondary (elevation / vertical) | `#FFA050` (light orange) |

---

## 5. Minimal working example

```python
import sys
sys.path.insert(0, "/Users/benjaminscholl/Documents/bs")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import (
    load_ferret_eye_kinematics_from_directory,
)

DATA_DIR = Path("/path/to/analyzable_outputs")
SESSION  = "session_2025-06-28_ferret_753_EyeCameras_P30_EO2_analyzable_output"
ANALYZABLE_OUTPUT_DIR = DATA_DIR / SESSION

left_eye  = load_ferret_eye_kinematics_from_directory("left_eye",  ANALYZABLE_OUTPUT_DIR / "left_eye_kinematics")
right_eye = load_ferret_eye_kinematics_from_directory("right_eye", ANALYZABLE_OUTPUT_DIR / "right_eye_kinematics")

t = left_eye.eyeball.timestamps - left_eye.eyeball.timestamps[0]

plt.plot(t, np.degrees(left_eye.adduction_angle.values),  color="#0096FF", label="Left adduction")
plt.plot(t, np.degrees(right_eye.adduction_angle.values), color="#FF6400", label="Right adduction")
plt.xlabel("Time (s)")
plt.ylabel("Degrees")
plt.legend()
plt.show()
```

---

## 6. What's next

Once eye position is working, velocity is already computed on the same object:

- **Velocity**: `.adduction_velocity.values`, `.elevation_velocity.values` (rad/s)
- **Gaze** (eye + head): load `FerretGazeKinematics` from `gaze_kinematics/` — same Timeseries pattern, gives world-space horizontal/vertical and their derivatives
