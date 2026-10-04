import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

_EYETOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(_EYETOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EYETOOLS_ROOT))

from utils.config import DATA_DIR  # also puts the bs repo on sys.path
from utils.load_skull_data import load_skull_data
from utils.parse_session_name import parse_session_name
from utils.load_session_data import load_session_data
from utils.removeBadData import removeBadData
from utils.eye_velocity import eye_velocity


@dataclass
class Session:
    # --- required ---
    session_id: str
    animal_id:  int
    eo:         int
    t:          np.ndarray   # (T,) s, monotonic
    fs:         float        # Hz
    yaw_deg:    np.ndarray   # (T,) global, 0–360
    pos_xy:     np.ndarray   # (T, 2) mm, world
    valid:      np.ndarray   # (T,) bool

    # --- optional ---
    pitch_deg:   np.ndarray | None = None
    roll_deg:    np.ndarray | None = None
    vel_global:  np.ndarray | None = None   # (T, 2) mm/s, world frame
    omega_local: np.ndarray | None = None   # (T, 3) rad/s, head frame [roll, pitch, yaw]
    neural:      object | None = None
    # eye (both eyes averaged, head frame: + horizontal = head yaw +); deg, deg/s
    eye_x:  np.ndarray | None = None
    eye_y:  np.ndarray | None = None
    eye_vx: np.ndarray | None = None
    eye_vy: np.ndarray | None = None
    pupil_eyes: np.ndarray | None = None   # (T, 2) % change from session median, NaN where not trusted
    pupil:  np.ndarray | None = None       # both eyes averaged, gaps interpolated, smoothed


def _mean2(a, b):
    """Mean of two eyes; the one finite value where only one eye is finite."""
    return np.where(np.isnan(a), b, np.where(np.isnan(b), a, (a + b) / 2))


def load_session(session_id: str, max_speed: float = 800, max_abs_pitch: float = 75,
                 max_ang_speed: float = 1500, pad: int = 6, eye: dict | None = None) -> Session:
    """eye: None loads head only; else dict(max_eye_deg, pupil_max_gap_s, pupil_smooth_s) also loads eye + pupil."""
    if eye is None:
        meta = load_skull_data(DATA_DIR / session_id)
    else:
        R = load_session_data(session_id)
        removeBadData(R)   # NaN eye data where LEQ / REQ quality is bad
        meta = vars(R)
    info = parse_session_name(session_id)

    t = meta["skull_timestamps"]
    pos_xy = np.column_stack([meta["position_x"], meta["position_y"]])
    vel_global = np.column_stack([meta["linearVel_x"], meta["linearVel_y"]])
    omega_local = np.column_stack([meta["roll_v"], meta["pitch_v"], meta["yaw_v"]])

    # same head-speed criterion as utils/process_session.py
    speed = np.linalg.norm(vel_global, axis=1)
    valid = np.isfinite(meta["yaw"]) & np.isfinite(pos_xy).all(axis=1) & (speed < max_speed)
    # Euler yaw is ill-defined near |pitch| = 90 (gimbal lock)
    valid &= np.abs(meta["pitch"]) < max_abs_pitch
    # tracking glitches: head angular speed (deg/s) far above real movement
    valid &= np.degrees(np.linalg.norm(omega_local, axis=1)) < max_ang_speed
    # pad excluded stretches, as in utils/removeBadData.py
    valid = ~binary_dilation(~valid, iterations=pad)

    ek = {}
    if eye is not None:
        fs = 1 / np.median(np.diff(t))
        ek["eye_x"] = _mean2(-R.LE_x, R.RE_x)
        ek["eye_y"] = _mean2(R.LE_y, R.RE_y)
        ek["eye_vx"] = _mean2(eye_velocity(R, "LE", "vx", head_frame=True), eye_velocity(R, "RE", "vx", head_frame=True))
        ek["eye_vy"] = _mean2(eye_velocity(R, "LE", "vy"), eye_velocity(R, "RE", "vy"))

        # pupil is trusted only with the eye near centre; % change from each eye's session median
        pe = []
        for e in ("LE", "RE"):
            x, y, p = getattr(R, f"{e}_x"), getattr(R, f"{e}_y"), getattr(R, f"{e}_pupil")
            p = np.where((np.abs(x) < eye["max_eye_deg"]) & (np.abs(y) < eye["max_eye_deg"]), p, np.nan)
            pe.append(100 * (p / np.nanmedian(p) - 1))
        ek["pupil_eyes"] = np.column_stack(pe)

        # interpolate gaps up to pupil_max_gap_s, then centred moving average
        p = pd.Series(_mean2(*pe))
        gap = p.isna().groupby(p.notna().cumsum()).transform("sum")   # length of the NaN run each frame is in
        filled = p.interpolate(limit_area="inside").mask(p.isna() & (gap > eye["pupil_max_gap_s"] * fs))
        smooth = filled.rolling(int(round(eye["pupil_smooth_s"] * fs)), center=True, min_periods=1).mean()
        ek["pupil"] = smooth.mask(filled.isna()).to_numpy()

        valid &= np.isfinite(np.column_stack([ek["eye_x"], ek["eye_y"], ek["eye_vx"], ek["eye_vy"], ek["pupil"]])).all(axis=1)

    return Session(
        session_id=session_id,
        animal_id=info["id"],
        eo=info["eo"],
        t=t,
        fs=1 / np.median(np.diff(t)),
        yaw_deg=meta["yaw"] % 360,
        pos_xy=pos_xy,
        valid=valid,
        pitch_deg=meta["pitch"],
        roll_deg=meta["roll"],
        vel_global=vel_global,
        omega_local=omega_local,
        **ek,
    )
