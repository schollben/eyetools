import numpy as np

from utils import non_saccade_mask

FS = 120.0
LOCO_COLORS = {"all": "#444444", "stationary": "#725EE7", "running": "#E93115"}


def running_mask(R, speed_threshold=100, min_bout=30):
    """True = running. Threshold crossing plus a minimum bout length in frames."""
    m = np.asarray(R.speed, float) >= speed_threshold
    edges = np.diff(np.concatenate(([0], m.astype(int), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a < min_bout:
            m[a:b] = False
    return m


def saccade_frames(R):
    """True = frame inside an eye-saccade window [onset, peak], either eye."""
    n = len(R.LE_vx)
    sacc = np.zeros(n, bool)
    for df in (R.df_LE, R.df_RE):
        for onset, peak in zip(df["onset"].to_numpy(), df["peak"].to_numpy()):
            sacc[onset:min(n, peak + 1)] = True
    return sacc


def frame_mask(R, sacc_subset="non_saccade", loco_subset="all",
               speed_threshold=100, min_bout=30):
    """Per-frame selection combining saccade state and locomotor state."""
    if sacc_subset == "non_saccade":
        m = non_saccade_mask(R)
    elif sacc_subset == "saccade":
        m = saccade_frames(R)
    else:
        m = np.ones(len(R.LE_vx), bool)

    if loco_subset != "all":
        run = running_mask(R, speed_threshold, min_bout)
        m = m & (run if loco_subset == "running" else ~run & np.isfinite(R.speed))

    return m


def head_signal(R, name):
    """One head signal in degrees. Velocities are stored as rad/s, so convert."""
    if name == "speed":
        return np.rad2deg(np.asarray(R.angVelocities, float))
    v = np.asarray(getattr(R, name), float)
    return np.rad2deg(v) if name.endswith("_v") else v


def event_condition(R, df, condition, speed_threshold=100, min_bout=30,
                    head_still_thresh=50):
    """Boolean per EVENT: does this event's onset frame fall in `condition`?

    The per-event analogue of frame_mask. condition is "all", or any combination of
    "stationary" and "head_still" (e.g. "stationary_and_head_still").
    """
    onsets = df["onset"].to_numpy().astype(int) if len(df) else np.array([], int)
    keep = np.ones(len(onsets), bool)
    if condition == "all" or not len(onsets):
        return keep
    if "stationary" in condition:
        keep &= ~running_mask(R, speed_threshold, min_bout)[onsets]
    if "head_still" in condition:
        keep &= head_signal(R, "speed")[onsets] < head_still_thresh
    return keep


def eye_signal(R, eye, key, flip_eye=None):
    """One eye signal. 'speed' is sqrt(vx^2 + vy^2).

    flip_eye: None keeps each eye in its own nasal/temporal frame (correct for
    integrator/drift, where the eye drifts toward its own orbital center). "LE" or "RE"
    negates that eye's horizontal signal to put both eyes in a common conjugate frame
    (needed for VOR and binocular measures). Which eye is flipped only sets a global sign
    on the shared axis.
    """
    if key == "speed":
        vx = np.asarray(getattr(R, f"{eye}_vx"), float)
        vy = np.asarray(getattr(R, f"{eye}_vy"), float)
        return np.sqrt(vx ** 2 + vy ** 2)
    v = np.asarray(getattr(R, f"{eye}_{key}"), float)
    return -v if eye == flip_eye and key.endswith("x") else v


def head_eye_pairs(group, head_name, eye_key, sacc_subset="non_saccade", loco_subset="all",
                   flip_eye=None, speed_threshold=100, min_bout=30):
    """Pool a head signal against both eyes, frame-aligned and masked.

    head_name: "speed" (total angular speed), "yaw_v", "pitch_v", "pitch", "roll"
    eye_key:   "speed", "vx", "vy", "x", "y"
    """
    hs, es = [], []
    for R in group:
        h = head_signal(R, head_name)
        m = frame_mask(R, sacc_subset, loco_subset, speed_threshold, min_bout)
        for eye in ("LE", "RE"):
            hs.append(h[m])
            es.append(eye_signal(R, eye, eye_key, flip_eye)[m])
    h = np.concatenate(hs)
    e = np.concatenate(es)
    inds = np.isfinite(h) & np.isfinite(e)
    return h[inds], e[inds]


def drift_frames(R, eye, pad_post=24, pad_pre=3, vel_ceiling=20, flip_eye=None):
    """Saccade-free (position, velocity) frames for one eye. Horizontal only."""
    x = eye_signal(R, eye, "x", flip_eye)
    v = eye_signal(R, eye, "vx", flip_eye)
    m = non_saccade_mask(R, pad_pre=pad_pre, pad_post=pad_post)
    m = m & np.isfinite(x) & np.isfinite(v)
    if vel_ceiling:
        m = m & (np.abs(v) < vel_ceiling)
    return x[m], v[m], m


def drift_by_position(group, eyes, bins, pad_post=24, pad_pre=3, vel_ceiling=20,
                      flip_eye=None, center=True):
    """Mean drift velocity per signed eye-position bin.

    Returns centers, means, sems, counts — one entry per bin, NaN where a bin is empty.
    """
    xs, vs = [], []
    for R in group:
        for eye in eyes:
            x, v, _ = drift_frames(R, eye, pad_post, pad_pre, vel_ceiling, flip_eye)
            if center and len(x):
                x = x - np.median(x)
            xs.append(x)
            vs.append(v)
    x = np.concatenate(xs) if xs else np.array([])
    v = np.concatenate(vs) if vs else np.array([])

    centers = (bins[:-1] + bins[1:]) / 2
    means = np.full(len(centers), np.nan)
    sems = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), int)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        sel = (x >= lo) & (x < hi)
        counts[i] = sel.sum()
        if counts[i] > 1:
            means[i] = v[sel].mean()
            sems[i] = v[sel].std() / np.sqrt(counts[i])
    return centers, means, sems, counts


def clean_runs(mask, min_len):
    """Start/stop indices of contiguous True runs at least min_len long."""
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_len]


def run_xcorr(x, y, max_lag):
    """Cross-correlation of two z-scored NaN-free segments, lags -max_lag..+max_lag.

    Positive lag = y lags x. Returns None if either segment is constant.
    """
    if x.std() == 0 or y.std() == 0:
        return None
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    cc = np.correlate(y, x, mode="full") / len(x)
    mid = len(x) - 1
    return cc[mid - max_lag: mid + max_lag + 1]


def eo_groups(Results, pool_by_eo, eo_bins):
    """Session groups and panel titles, pooled by EO bin or one per session."""
    if pool_by_eo:
        groups = [[R for R in Results if lo <= R.eo <= hi] for lo, hi in eo_bins]
        titles = [f"EO {lo}-{hi}" for lo, hi in eo_bins]
    else:
        groups = [[R] for R in Results]
        titles = [f"Ferret {R.id} EO{R.eo}" for R in Results]
    return groups, titles


def fit_line(ax, x, y, color="k", min_n=500):
    """Overlay a linear fit; returns (nan, nan) when too few points to be meaningful."""
    if len(x) < min_n:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    xl = np.array([x.min(), x.max()])
    ax.plot(xl, slope * xl + intercept, color=color, lw=1)
    return slope, intercept
