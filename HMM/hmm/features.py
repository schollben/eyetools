import numpy as np
from scipy.signal import savgol_filter, butter, sosfiltfilt

FEATURE_NAMES = ["omega_yaw", "log_speed", "pitch", "v_fwd", "v_lat"]
SIGNED = [0, 4]   # columns that flip sign under left/right mirroring


def _valid_runs(valid: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """(start, stop) index pairs of contiguous True runs at least min_len long."""
    edges = np.diff(np.concatenate([[0], valid.astype(int), [0]]))
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_len]


def session_features(s, fs_out=30, min_run_s=2, sg_win_s=None, lowpass_hz=None) -> list[dict]:
    """Features for each valid run of a Session, resampled to a uniform fs_out grid.
    sg_win_s / lowpass_hz = None uses the loaded (already smoothed upstream) data as is.
    Returns a list of segments: dict(session_id, animal_id, eo, t, X)."""
    segs = []
    for a, b in _valid_runs(s.valid, int(min_run_s * s.fs)):
        t = s.t[a:b]
        yaw = np.unwrap(np.radians(s.yaw_deg[a:b]))
        if sg_win_s is None:
            omega = np.gradient(yaw, t)                                               # rad/s
        else:
            omega = savgol_filter(yaw, int(round(sg_win_s * s.fs)) | 1, 2, deriv=1, delta=1 / s.fs)
        log_speed = np.log(np.linalg.norm(s.vel_global[a:b], axis=1) + 1)           # log(mm/s + 1)
        pitch = np.radians(s.pitch_deg[a:b])                                          # rad
        vx, vy = s.vel_global[a:b].T
        v_fwd = vx * np.cos(yaw) + vy * np.sin(yaw)                                  # head frame, mm/s
        v_lat = -vx * np.sin(yaw) + vy * np.cos(yaw)
        slog = lambda v: np.sign(v) * np.log(np.abs(v) + 1)                          # signed log(mm/s + 1)
        raw = np.column_stack([omega, log_speed, pitch, slog(v_fwd), slog(v_lat)])

        # optional anti-alias, then interpolate onto a common-rate grid
        if lowpass_hz is not None:
            raw = sosfiltfilt(butter(4, lowpass_hz, fs=s.fs, output="sos"), raw, axis=0)
        t_out = np.arange(t[0], t[-1], 1 / fs_out)
        X = np.column_stack([np.interp(t_out, t, raw[:, j]) for j in range(raw.shape[1])])

        segs.append(dict(session_id=s.session_id, animal_id=s.animal_id, eo=s.eo, t=t_out, X=X))
    return segs


def fit_scaler(Xs: list[np.ndarray]) -> dict:
    """Robust z-score stats (median / IQR) pooled over training segments."""
    allX = np.concatenate(Xs)
    q25, med, q75 = np.percentile(allX, [25, 50, 75], axis=0)
    return dict(median=med, iqr=q75 - q25)


def apply_scaler(Xs: list[np.ndarray], scaler: dict) -> list[np.ndarray]:
    return [(X - scaler["median"]) / scaler["iqr"] for X in Xs]


def mirror(X: np.ndarray) -> np.ndarray:
    """Left/right mirror: flip the sign of signed features."""
    Xm = X.copy()
    Xm[:, SIGNED] *= -1
    return Xm
