import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from utils.session_data import SessionData

_COLS = ['onset', 'peak', 'amplitude_deg', 'direction_deg']

def _get_arrays(session: SessionData, data_type: str, eye: str | None):
    """Return (vx, vy, ax, ay, x, y); ax/ay are None for gaze (no stored accel)."""
    if data_type == 'eye':
        if eye == 'LE':
            return session.LE_vx, session.LE_vy, session.LE_ax, session.LE_ay, session.LE_x, session.LE_y
        else:
            return session.RE_vx, session.RE_vy, session.RE_ax, session.RE_ay, session.RE_x, session.RE_y
    elif data_type == 'gaze':
        if eye == 'LE':
            return (session.LE_ang_vel_local_x_deg_s, session.LE_ang_vel_local_y_deg_s,
                    None, None,
                    session.LE_gaze_horizontal_deg, session.LE_gaze_vertical_deg)
        else:
            return (session.RE_ang_vel_local_x_deg_s, session.RE_ang_vel_local_y_deg_s,
                    None, None,
                    session.RE_gaze_horizontal_deg, session.RE_gaze_vertical_deg)
    else:  # skull
        return session.yaw_v, session.pitch_v, session.yaw_a, session.pitch_a, session.yaw, session.pitch


def _detect(vx, vy, ax, ay, x, y, velocity_threshold, min_duration, max_duration, nan_margin):

    speed = np.sqrt(vx ** 2 + vy ** 2)
    nan_mask = ~np.isfinite(speed) | ~np.isfinite(x) | ~np.isfinite(y)

    # Tangential acceleration: signed component of acceleration along velocity direction.
    # Positive = speeding up, negative = slowing down.
    if ax is not None and ay is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            a_tang = np.where(speed > 0, (vx * ax + vy * ay) / speed, 0.0)
    else:
        a_tang = np.gradient(speed)
    a_tang = np.where(np.isfinite(a_tang), a_tang, 0.0)

    # Use 0 in place of NaN so find_peaks skips bad frames cleanly.
    speed_clean = np.where(np.isfinite(speed), speed, 0.0)
    peaks, _ = find_peaks(speed_clean, height=velocity_threshold, distance=min_duration)

    n = len(speed)
    
    events = []

    for p in peaks:

        # onset: walk backward to last frame where tangential accel <= 0
        onset = max(0, p - max_duration)
        for i in range(p - 1, max(0, p - max_duration) - 1, -1):
            if a_tang[i] <= 0:
                onset = i
                break

        # peak (position): walk forward to first frame where tangential accel >= 0
        # — this is where the eye reaches maximum displacement
        pos_peak = min(n - 1, p + max_duration)
        for i in range(p + 1, min(n, p + max_duration + 1)):
            if a_tang[i] >= 0:
                pos_peak = i
                break

        if (pos_peak - onset) > max_duration:
            continue

        if np.any(~np.isfinite(speed[onset:pos_peak + 1])):
            continue

        lo_on = max(0, onset - nan_margin)
        hi_on = min(n, onset + nan_margin + 1)
        lo_pk = max(0, pos_peak - nan_margin)
        hi_pk = min(n, pos_peak + nan_margin + 1)
        if np.any(nan_mask[lo_on:hi_on]) or np.any(nan_mask[lo_pk:hi_pk]):
            continue

        dx = float(x[pos_peak] - x[onset])
        dy = float(y[pos_peak] - y[onset])

        events.append({
            'onset': int(onset),
            'peak': int(pos_peak),
            'amplitude_deg': float(np.sqrt(dx ** 2 + dy ** 2)),
            'direction_deg': float(np.degrees(np.arctan2(dy, dx))),
        })

    return events


def _filter_inter_event(df: pd.DataFrame, min_inter_event: int) -> pd.DataFrame:
    """Drop events where onset[i+1] - peak[i] < min_inter_event frames."""
    if len(df) < 2:
        return df
    onsets = df['onset'].to_numpy()
    peaks = df['peak'].to_numpy()
    keep = [True] * len(df)
    for i in range(len(df) - 1):
        if onsets[i + 1] - peaks[i] < min_inter_event:
            keep[i + 1] = False
    return df[keep].reset_index(drop=True)


def extract_saccades(
    session: SessionData,
    data_type: str,
    eye: str = 'LE',
    velocity_threshold: float = 30.0,
    min_duration: int = 3,
    max_duration: int = 60,
    min_inter_event: int = 12,
    nan_margin: int = 12,
) -> pd.DataFrame:
    """Detect saccades (eye/gaze) or shifts (skull) from a SessionData object.

    Args:
        session: Loaded SessionData.
        data_type: 'eye', 'gaze', or 'skull'.
        eye: 'LE' or 'RE'. Ignored when data_type='skull'.
        velocity_threshold: 2D speed (deg/s) a peak must exceed to be a candidate.
        min_duration: Minimum frame separation between adjacent velocity peaks.
        max_duration: Maximum frames allowed between onset and position peak.
        min_inter_event: Minimum frames required between peak of one event and
            onset of the next. Removes rapid return/corrective movements.
        nan_margin: Number of frames around onset/peak to check for NaNs. Events
            with NaNs in this window are discarded.

    Returns:
        DataFrame with columns: onset, peak (frame indices of movement start and
        position maximum), amplitude_deg, direction_deg.
    """
    if data_type not in ('eye', 'gaze', 'skull'):
        raise ValueError(f"data_type must be 'eye', 'gaze', or 'skull', got '{data_type}'")
    if data_type != 'skull' and eye not in ('LE', 'RE'):
        raise ValueError(f"eye must be 'LE' or 'RE', got '{eye}'")

    args = _get_arrays(session, data_type, eye if data_type != 'skull' else None)
    events = _detect(*args, velocity_threshold, min_duration, max_duration, nan_margin)
    if not events:
        return pd.DataFrame(columns=_COLS)
    df = pd.DataFrame(events, columns=_COLS)
    return _filter_inter_event(df, min_inter_event)
