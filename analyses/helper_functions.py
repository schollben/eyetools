import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from utils import non_saccade_mask
from utils.extract_saccades import _detect, _COLS

FS = 120.0
LOCO_COLORS = {"all": "#444444", "stationary": "#725EE7", "running": "#E93115"}

# Eye and head traces, shared by every script. EYE_COLOR is the default when LE and RE
# are not being compared.
LE_COLOR, RE_COLOR, HEAD_COLOR = "#6BAED6", "#74C476", "#9E9AC8"
EYE_COLOR = RE_COLOR
EYE_COLORS = {"LE": LE_COLOR, "RE": RE_COLOR}


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


def event_dfs(R, signal):
    """The two per-eye event tables: 'eye' -> eye saccades, 'gaze' -> gaze shifts."""
    return (R.df_LE, R.df_RE) if signal == "eye" else (R.df_LEgaze, R.df_REgaze)


def pooled_events(group, signal, condition, column,
                  speed_threshold=100, min_bout=30, head_still_thresh=50):
    """One scalar column (e.g. 'amplitude_deg') pooled over both eyes and all sessions."""
    out = []
    for R in group:
        for df in event_dfs(R, signal):
            if not len(df):
                continue
            keep = event_condition(R, df, condition, speed_threshold, min_bout,
                                   head_still_thresh)
            out.append(df[column].to_numpy().astype(float)[keep])
    return np.concatenate(out) if out else np.array([])


def pooled_intervals(group, signal, condition="all", speed_threshold=100,
                     min_bout=30, head_still_thresh=50):
    """Quiescent intervals (ms) between consecutive events, pooled over eyes and sessions.

    Measured peak[i] -> onset[i+1]: the gap between the end of one movement and the start
    of the next. Both events must satisfy `condition`, and the gap must be free of NaNs —
    4.5% of gaps span a tracking dropout and would otherwise dominate the tail (median
    1425 ms vs 275 ms for clean gaps, max 119.8 s).

    Note: extraction enforces min_inter_event frames between events, so this distribution
    is truncated at min_inter_event / FS seconds (100 ms at the default 12 frames).
    """
    out = []
    for R in group:
        for eye, df in zip(("LE", "RE"), event_dfs(R, signal)):
            if len(df) < 2:
                continue
            keep = event_condition(R, df, condition, speed_threshold, min_bout,
                                   head_still_thresh)
            onsets = df["onset"].to_numpy().astype(int)
            peaks = df["peak"].to_numpy().astype(int)
            pos = np.asarray(getattr(R, f"{eye}_x" if signal == "eye"
                                     else f"{eye}_gaze_horizontal_deg"), float)
            for i in range(len(df) - 1):
                a, b = peaks[i], onsets[i + 1]
                if not (keep[i] and keep[i + 1]) or b <= a:
                    continue
                if np.all(np.isfinite(pos[a:b + 1])):
                    out.append((b - a) / FS * 1000)
    return np.array(out)


def event_traces(R, signal, kind, lo, hi, bin_col, condition, pre, post,
                 flip_eye=None, speed_threshold=100, min_bout=30, head_still_thresh=50):
    """Onset-aligned traces for one session's events falling in one bin.

    kind: "speed" -> sqrt(vx^2 + vy^2); "displacement" -> distance from position at onset.
    Returns (traces, nominal_amplitudes). Events too near a recording edge, or whose window
    contains NaN, are dropped.

    NOTE: gaze angular-velocity fields are already rad2deg-converted at load, unlike the
    skull _v fields — they are read directly here and must never go through head_signal().
    """
    traces, nominal = [], []

    for eye, df in zip(("LE", "RE"), event_dfs(R, signal)):
        if not len(df):
            continue

        if signal == "eye":
            vx = eye_signal(R, eye, "vx", flip_eye)
            vy = eye_signal(R, eye, "vy", flip_eye)
            px = eye_signal(R, eye, "x", flip_eye)
            py = eye_signal(R, eye, "y", flip_eye)
        else:
            vx = np.asarray(getattr(R, f"{eye}_ang_vel_local_x_deg_s"), float)
            vy = np.asarray(getattr(R, f"{eye}_ang_vel_local_y_deg_s"), float)
            px = np.asarray(getattr(R, f"{eye}_gaze_horizontal_deg"), float)
            py = np.asarray(getattr(R, f"{eye}_gaze_vertical_deg"), float)

        speed = np.sqrt(vx ** 2 + vy ** 2) if kind == "speed" else None

        keep = event_condition(R, df, condition, speed_threshold, min_bout,
                               head_still_thresh)
        amp = df[bin_col].to_numpy().astype(float)
        onsets = df["onset"].to_numpy().astype(int)
        sel = keep & (amp >= lo) & (amp < hi)

        for o, a_nom in zip(onsets[sel], amp[sel]):
            start, stop = o - pre, o + post
            if start < 0 or stop > len(px):
                continue
            if kind == "speed":
                tr = speed[start:stop]
            else:
                tr = np.sqrt((px[start:stop] - px[o]) ** 2
                             + (py[start:stop] - py[o]) ** 2)
            if not np.all(np.isfinite(tr)):
                continue
            traces.append(tr)
            nominal.append(a_nom)

    return traces, nominal


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


def head_saccades(R, velocity_threshold=50, min_duration=12, max_duration=60,
                  min_inter_event=12, smooth=11):
    """Head-saccade table, extracted here because R.df_head is not usable.

    Same columns as R.df_LE. Thresholds are in deg/s.

    TODO — UPSTREAM FIX. R.df_head is unusable because of three bugs in utils/:
      1. extract_saccades._get_arrays smooths skull velocity with savgol_filter(L=121)
         = 1.01 s at 120 Hz, while the eye branch uses raw velocity (corr = 0.715).
      2. process_session passes max_duration=600 (5 s) for skull vs 60 for eyes.
      3. velocity_threshold_head=2 is rad/s (~115 deg/s), not deg/s.
    Measured on 402/420: current df_head = 449 events, median 692 ms, 47 deg.
    This function = 2363 events, median 225 ms, 18 deg. Fixing utils/ would change
    df_head for every existing script, so the correction lives here for now.
    """
    sg = lambda a: np.rad2deg(savgol_filter(np.asarray(a, float), smooth, 3))
    events = _detect(sg(R.yaw_v), sg(R.pitch_v), sg(R.yaw_a), sg(R.pitch_a),
                     np.asarray(R.yaw, float), np.asarray(R.pitch, float),
                     velocity_threshold, min_duration, max_duration, min_inter_event)
    return pd.DataFrame(events, columns=_COLS)


def _aligned_windows(R, eye, head_df, flip_eye, pre, post):
    """Yield (onset, amplitude, head_pos, eye_pos, head_vel, eye_vel, sign, lag) per
    eye saccade. Traces are baseline-subtracted at onset and sign-aligned so the head's
    dominant rotation is positive — the convention Wallace Fig 4D requires."""
    df = R.df_LE if eye == "LE" else R.df_RE
    px = eye_signal(R, eye, "x", flip_eye)
    vx = eye_signal(R, eye, "vx", flip_eye)
    yaw = np.asarray(R.yaw, float)
    yv = np.rad2deg(savgol_filter(np.asarray(R.yaw_v, float), 11, 3))
    h_on = head_df["onset"].to_numpy() if len(head_df) else np.array([], int)

    for onset, amp in zip(df["onset"].to_numpy().astype(int),
                          df["amplitude_deg"].to_numpy().astype(float)):
        a, b = onset - pre, onset + post
        if a < 0 or b > len(px):
            continue
        hp, ep, hv, ev = yaw[a:b], px[a:b], yv[a:b], vx[a:b]
        if not (np.all(np.isfinite(hp)) and np.all(np.isfinite(ep))
                and np.all(np.isfinite(hv)) and np.all(np.isfinite(ev))):
            continue
        sign = np.sign(hv[np.argmax(np.abs(hv))]) or 1.0
        lag = np.nan if not len(h_on) else float(h_on[np.argmin(np.abs(h_on - onset))] - onset)
        yield (onset, amp, (hp - hp[pre]) * sign, (ep - ep[pre]) * sign,
               hv * sign, ev * sign, sign, lag)


def head_eye_events(R, head_df, flip_eye=None, pre=24, post=72):
    """One row per eye saccade with its head context.

    Columns: onset, eye, amplitude_deg, peak_velocity_deg_s, lag_ms (head onset minus eye
    onset; NaN if the session has no head saccades), same_direction, head_peak_pos_vel,
    eye_peak_neg_vel. Velocities are sign-aligned to the head's dominant direction, so
    "peak positive head" and "peak negative eye" are well defined (Wallace Fig 4D).
    """
    rows = []
    for eye in ("LE", "RE"):
        df = R.df_LE if eye == "LE" else R.df_RE
        pkv = dict(zip(df["onset"].to_numpy().astype(int),
                       df["peak_velocity_deg_s"].to_numpy().astype(float)))
        for onset, amp, _, _, hv, ev, _, lag in _aligned_windows(
                R, eye, head_df, flip_eye, pre, post):
            rows.append((onset, eye, amp, pkv.get(onset, np.nan),
                         lag / FS * 1000 if np.isfinite(lag) else np.nan,
                         np.sign(ev[np.argmax(np.abs(ev))]) > 0,
                         hv.max(), ev.min()))
    return pd.DataFrame(rows, columns=["onset", "eye", "amplitude_deg",
                                       "peak_velocity_deg_s", "lag_ms", "same_direction",
                                       "head_peak_pos_vel", "eye_peak_neg_vel"])


def head_eye_traces(R, head_df, lo, hi, flip_eye=None, pair_window=30, pre=24, post=72):
    """Onset-aligned traces for eye saccades with amplitude in [lo, hi), paired with a head
    saccade within pair_window frames. Returns a dict of lists keyed
    head_pos/LE_pos/RE_pos/head_vel/LE_vel/RE_vel — head traces are collected once per
    eye saccade, so head_pos aligns with whichever eye contributed it (Wallace Fig 4A/4C).
    """
    out = {k: [] for k in ("head_pos", "LE_pos", "RE_pos",
                           "head_vel", "LE_vel", "RE_vel")}
    for eye in ("LE", "RE"):
        for _, amp, hp, ep, hv, ev, _, lag in _aligned_windows(
                R, eye, head_df, flip_eye, pre, post):
            if not (lo <= amp < hi) or not np.isfinite(lag) or abs(lag) > pair_window:
                continue
            out["head_pos"].append(hp)
            out["head_vel"].append(hv)
            out[f"{eye}_pos"].append(ep)
            out[f"{eye}_vel"].append(ev)
    return out
