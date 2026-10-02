import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

_EYETOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(_EYETOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EYETOOLS_ROOT))

from utils.config import DATA_DIR  # also puts the bs repo on sys.path
from utils.load_skull_data import load_skull_data
from utils.parse_session_name import parse_session_name


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


def load_session(session_id: str, max_speed: float = 800) -> Session:
    meta = load_skull_data(DATA_DIR / session_id)
    info = parse_session_name(session_id)

    t = meta["skull_timestamps"]
    pos_xy = np.column_stack([meta["position_x"], meta["position_y"]])
    vel_global = np.column_stack([meta["linearVel_x"], meta["linearVel_y"]])
    omega_local = np.column_stack([meta["roll_v"], meta["pitch_v"], meta["yaw_v"]])

    # same head-speed criterion as utils/process_session.py
    speed = np.linalg.norm(vel_global, axis=1)
    valid = np.isfinite(meta["yaw"]) & np.isfinite(pos_xy).all(axis=1) & (speed < max_speed)

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
    )
