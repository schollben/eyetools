from dataclasses import dataclass
import numpy as np

@dataclass
class SessionData:
    # --- session metadata ---
    session: str
    date:    str
    id:      int
    age:     int
    eo:      int

    # --- skull kinematics ---
    skull_timestamps: np.ndarray
    roll:   np.ndarray
    pitch:  np.ndarray
    yaw:    np.ndarray
    roll_v: np.ndarray
    pitch_v: np.ndarray
    yaw_v:  np.ndarray
    roll_a: np.ndarray
    pitch_a: np.ndarray
    yaw_a:  np.ndarray

    # --- eye data quality ---
    EQtimestamps: np.ndarray
    EQframes:     np.ndarray
    LEQ:          np.ndarray
    REQ:          np.ndarray

    # --- left eye kinematics ---
    LE_timestamps: np.ndarray
    LE_n_frames:   int
    LE_x:  np.ndarray
    LE_y:  np.ndarray
    LE_vx: np.ndarray
    LE_vy: np.ndarray
    LE_ax: np.ndarray
    LE_ay: np.ndarray
    LE_pupil_major: np.ndarray

    # --- right eye kinematics ---
    RE_x:  np.ndarray
    RE_y:  np.ndarray
    RE_vx: np.ndarray
    RE_vy: np.ndarray
    RE_ax: np.ndarray
    RE_ay: np.ndarray
    RE_pupil_major: np.ndarray

    # --- left gaze ---
    LE_gaze_timestamps:     np.ndarray
    LE_gaze_horizontal_deg: np.ndarray
    LE_gaze_vertical_deg:   np.ndarray
    LE_ang_vel_local_x_deg_s: np.ndarray
    LE_ang_vel_local_y_deg_s: np.ndarray

    # --- right gaze ---
    RE_gaze_timestamps:     np.ndarray
    RE_gaze_horizontal_deg: np.ndarray
    RE_gaze_vertical_deg:   np.ndarray
    RE_ang_vel_local_x_deg_s: np.ndarray
    RE_ang_vel_local_y_deg_s: np.ndarray

    # --- toy ---
    toy_x: np.ndarray
    toy_y: np.ndarray
