import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.session_data import SessionData
from utils.saccade_triggered_average import _plot_mean_se, LE_COLOR, HEAD_COLOR

EYE_VEL_COLOR = "#5aae61"

SACCADE_TYPES = ["counter_rotation", "gaze_shift", "eye_only"]
TYPE_LABELS   = {"counter_rotation": "Counter-rotation", "gaze_shift": "Gaze-shift", "eye_only": "Eye-only"}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def classify_saccades(
    df_eye: pd.DataFrame,
    session: SessionData,
    eye: str = "LE",
    pre_window: int = 20,
    pre_offset: int = 3,
    head_vel_threshold: float = 2.0,
    drift_alignment_threshold: float = 0.3,
    plot: bool = False,
    window: int = 60,
    pre_avg: int = 20,
):
    """Classify eye saccades into counter_rotation, gaze_shift, or eye_only.

    Uses the pre-saccade eye drift direction relative to head velocity to classify.
    Adds columns: saccade_type, head_direction, canonical_sign.

    Args:
        head_vel_threshold: deg/s below which head is considered stationary.
            Should match velocity_threshold used in extract_saccades(..., 'skull').
        drift_alignment_threshold: cosine similarity above which pre-saccade eye
            drift is considered aligned with head (→ counter_rotation).
        plot: if True, also return a 3×3 triggered average figure.
        window: frames after onset for triggered averages.
        pre_avg: frames before onset for triggered averages.
    """
    vx_key = f"{eye}_vx"
    vy_key = f"{eye}_vy"
    x_key  = f"{eye}_x"

    eye_vx = np.array(getattr(session, vx_key), dtype=float)
    eye_vy = np.array(getattr(session, vy_key), dtype=float)
    eye_x  = np.array(getattr(session, x_key),  dtype=float)
    yaw_v  = np.array(session.yaw_v,  dtype=float)
    pitch_v = np.array(session.pitch_v, dtype=float)
    n_frames = len(eye_vx)

    saccade_types  = []
    head_directions = []
    canonical_signs = []

    for _, row in df_eye.iterrows():
        onset = int(row["onset"])
        start = onset - pre_window
        end   = onset - pre_offset

        # bounds / NaN guard
        if start < 0 or end <= start or end > n_frames:
            saccade_types.append("unclassified")
            head_directions.append("unknown")
            canonical_signs.append(np.nan)
            continue

        pre_eye_vx  = eye_vx[start:end]
        pre_eye_vy  = eye_vy[start:end]
        pre_yaw_v   = yaw_v[start:end]
        pre_pitch_v = pitch_v[start:end]

        if not (np.all(np.isfinite(pre_eye_vx)) and np.all(np.isfinite(pre_yaw_v))):
            saccade_types.append("unclassified")
            head_directions.append("unknown")
            canonical_signs.append(np.nan)
            continue

        head_speed = np.mean(np.sqrt(pre_yaw_v ** 2 + pre_pitch_v ** 2))
        mean_yaw_v = float(np.mean(pre_yaw_v))

        # head direction label
        if head_speed < head_vel_threshold:
            head_dir = "stationary"
        else:
            head_dir = "CCW" if mean_yaw_v > 0 else "CW"

        head_directions.append(head_dir)

        if head_speed < head_vel_threshold:
            saccade_types.append("eye_only")
            # canonical sign from eye velocity direction at onset
            onset_vx = eye_vx[onset] if np.isfinite(eye_vx[onset]) else np.mean(pre_eye_vx)
            canonical_signs.append(1 if onset_vx >= 0 else -1)
            continue

        # cosine similarity between pre-saccade eye drift and head velocity
        eye_drift  = np.array([np.mean(pre_eye_vx), np.mean(pre_eye_vy)])
        head_vel   = np.array([np.mean(pre_yaw_v),  np.mean(pre_pitch_v)])
        cos_sim    = _cosine_similarity(eye_drift, head_vel)

        if cos_sim > drift_alignment_threshold:
            saccade_types.append("counter_rotation")
        else:
            saccade_types.append("gaze_shift")

        # canonical sign from head direction for counter_rotation and gaze_shift
        canonical_signs.append(1 if mean_yaw_v >= 0 else -1)

    df = df_eye.copy()
    df["saccade_type"]   = saccade_types
    df["head_direction"] = head_directions
    df["canonical_sign"] = canonical_signs

    if not plot:
        return df

    fig = _plot_triggered_averages(df, session, eye, eye_x, eye_vx, yaw_v, window, pre_avg)
    return df, fig


def _build_traces(onsets, signs, x_arr, pre, win, n_frames):
    """Extract direction-normalized, baseline-subtracted traces."""
    baseline = slice(pre - min(5, pre), pre)
    traces = []
    for onset, sign in zip(onsets, signs):
        if np.isnan(sign):
            continue
        s, e = onset - pre, onset + win
        if s < 0 or e > n_frames:
            continue
        seg = np.array(x_arr[s:e], dtype=float)
        if not np.all(np.isfinite(seg)):
            continue
        seg -= np.mean(seg[baseline])
        seg *= sign
        traces.append(seg)
    return traces


def _plot_triggered_averages(df, session, eye, eye_x, eye_vx, yaw_v, window, pre_avg):
    yaw = np.unwrap(np.array(session.yaw, dtype=float), period=360)
    n_frames = len(eye_x)
    t = np.arange(-pre_avg, window) / 120.0

    fig, axes = plt.subplots(3, 3, figsize=(8, 6), sharex=True)
    fig.tight_layout(pad=3)

    row_labels = ["Eye position (deg)", "Head yaw (deg)", "Eye velocity (deg/s)"]
    signals    = [eye_x, yaw, eye_vx]
    colors     = [LE_COLOR, HEAD_COLOR, EYE_VEL_COLOR]

    for col_idx, stype in enumerate(SACCADE_TYPES):
        sub = df[df["saccade_type"] == stype]
        onsets = sub["onset"].values
        signs  = sub["canonical_sign"].values

        for row_idx, (arr, color, ylabel) in enumerate(zip(signals, colors, row_labels)):
            ax = axes[row_idx, col_idx]
            traces = _build_traces(onsets, signs, arr, pre_avg, window, n_frames)
            _plot_mean_se(ax, t, traces, color, f"n={len(traces)}")
            ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=6)
            if row_idx == 0:
                ax.set_title(TYPE_LABELS[stype], fontsize=7)
            if row_idx == 2:
                ax.set_xlabel("Time (s)", fontsize=6)
            ax.legend(fontsize=5, frameon=False)

    return fig
