import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

from utils import non_saccade_mask, load_session_data, process_session, removeBadData, getSesh
from utils.config import SAVELOC
from utils.eye_velocity import eye_velocity

FS = 120.0

FERRETS = (402, 405, 407, 420) 
# delayed experience: 411, 416, 403
# TO ADD (once fixed): 753, 757
EO_BINS = [(0, 3), (4, 7), (8, 20)]

EXTRACT = dict(window_in_sec=5, velocity_threshold_eye=40, velocity_threshold_gaze=40,
               velocity_threshold_head=1, min_duration=8, min_inter_event=8)

FERRET_MARKERS = ("o", "s", "^", "D", "v", "P")

LOCO_COLORS = {"all": "#444444", "stationary": "#725EE7", "running": "#E93115"}

# Eye and head traces, shared by every script. EYE_COLOR is the default when LE and RE
# are not being compared.
LE_COLOR, RE_COLOR, HEAD_COLOR = "#6BAED6", "#74C476", "#9E9AC8"

EYE_COLOR = RE_COLOR

EYE_COLORS = {"LE": LE_COLOR, "RE": RE_COLOR}

AGE_COLORS = ["#989898", "#666666", "#222222"]


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


def condition_frames(R, condition, speed_threshold=100, min_bout=30,
                     head_still_thresh=50):
    """Boolean per FRAME: is the animal in `condition` on this frame?

    The per-frame analogue of event_condition, which tests only event onsets. Used to
    measure how much TIME a condition covers, which is what a rate needs for its
    denominator.
    """
    m = np.ones(len(R.LE_vx), bool)
    if condition == "all":
        return m
    if "stationary" in condition:
        m &= ~running_mask(R, speed_threshold, min_bout) & np.isfinite(R.speed)
    if "head_still" in condition:
        m &= head_signal(R, "speed") < head_still_thresh
    return m


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
    is truncated at min_inter_event / FS seconds (67 ms at min_inter_event=8).
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
    """One eye signal. 'speed' is sqrt(vx^2 + vy^2). vx / vy come from utils.eye_velocity (stored channels are swapped).

    flip_eye: None keeps each eye in its own nasal/temporal frame (correct for
    integrator/drift, where the eye drifts toward its own orbital center). "LE" or "RE"
    negates that eye's horizontal signal to put both eyes in a common conjugate frame.
    For LE-vs-RE measures the choice only sets a global sign. Against the HEAD it matters:
    "LE" is the head frame (eye + = head yaw +, world gaze = yaw + eye); "RE" points the
    other way. Head-vs-eye measures (VOR, eye_head) use "LE".
    """
    if key == "speed":
        vx = np.asarray(getattr(R, f"{eye}_vx"), float)
        vy = np.asarray(getattr(R, f"{eye}_vy"), float)
        return np.sqrt(vx ** 2 + vy ** 2)

    if key in ("vx", "vy"):
        v = eye_velocity(R, eye, key)
    else:
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


def drift_frames(R, eye, pad_post=24, pad_pre=3, vel_ceiling=20, flip_eye=None,
                 head_rot_thresh=None, head_trans_thresh=None):
    """Saccade-free (position, velocity) frames for one eye. Horizontal only."""
    x = eye_signal(R, eye, "x", flip_eye)
    v = eye_signal(R, eye, "vx", flip_eye)
    m = non_saccade_mask(R, pad_pre=pad_pre, pad_post=pad_post)
    m = m & np.isfinite(x) & np.isfinite(v)
    if vel_ceiling:
        m = m & (np.abs(v) < vel_ceiling)
    if head_rot_thresh:
        m = m & (np.abs(head_signal(R, "speed")) < head_rot_thresh)
    if head_trans_thresh:
        m = m & (np.asarray(R.speed, float) < head_trans_thresh)
    return x[m], v[m], m


def drift_by_position(group, eyes, bins, pad_post=24, pad_pre=3, vel_ceiling=20,
                      flip_eye=None, center=True, head_rot_thresh=None, head_trans_thresh=None):
    """Mean drift velocity per signed eye-position bin.

    Returns centers, means, sems, counts — one entry per bin, NaN where a bin is empty.
    """
    xs, vs = [], []
    for R in group:
        for eye in eyes:
            x, v, _ = drift_frames(R, eye, pad_post, pad_pre, vel_ceiling, flip_eye,
                                   head_rot_thresh, head_trans_thresh)
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


def unwrap_deg(a):
    """Unwrap an angle trace (deg) across the +-180 seam, skipping NaNs."""
    a = np.asarray(a, float).copy()
    ok = np.isfinite(a)
    a[ok] = np.unwrap(a[ok], period=360)
    return a


def in_head_saccade(onsets, head_df, pair_window=30):
    """Index of the head saccade each onset falls in, or -1.

    "In" = onset within [head onset - pair_window, head peak], so the eye may lead the head
    by up to pair_window frames. Windows do not overlap while pair_window < the head
    min_inter_event (60 frames).
    """
    onsets = np.asarray(onsets, int)
    if not len(head_df):
        return np.full(len(onsets), -1)
    h_on = head_df["onset"].to_numpy().astype(int)
    h_pk = head_df["peak"].to_numpy().astype(int)
    i = np.searchsorted(h_on - pair_window, onsets, side="right") - 1
    ok = (i >= 0) & (onsets <= h_pk[np.clip(i, 0, None)])
    return np.where(ok, i, -1)


def eye_head_coupling(R, head_df, pair_window=30, n_shift=100, seed=0):
    """(frac_eye, chance_eye, frac_head, chance_head) for one session.

    frac_eye: fraction of eye saccades (both eyes) starting inside a head saccade.
    frac_head: fraction of head saccades containing at least one. Chance circularly shifts
    the eye onsets against the head by random offsets >= 10 s, keeping both rates.
    """
    n = len(R.LE_vx)
    e = np.concatenate([R.df_LE["onset"].to_numpy(), R.df_RE["onset"].to_numpy()]).astype(int)
    if not len(e) or not len(head_df):
        return np.nan, np.nan, np.nan, np.nan

    def fracs(onsets):
        idx = in_head_saccade(onsets, head_df, pair_window)
        return (idx >= 0).mean(), len(np.unique(idx[idx >= 0])) / len(head_df)

    obs_e, obs_h = fracs(e)
    shifts = np.random.default_rng(seed).integers(int(10 * FS), n - int(10 * FS), n_shift)
    chance = np.array([fracs((e + s) % n) for s in shifts])
    return obs_e, chance[:, 0].mean(), obs_h, chance[:, 1].mean()


def onset_correlogram(R, head_df, max_lag=60, bin_frames=3):
    """Eye onsets (both eyes) around each head onset.

    Lag = eye onset - head onset (positive = eye followed), -max_lag..+max_lag frames. Returns
    bin centers (ms), counts, and counts expected if eye onsets were unrelated to the head.
    """
    edges = np.arange(-max_lag, max_lag + bin_frames, bin_frames)
    h = head_df["onset"].to_numpy().astype(int) if len(head_df) else np.array([], int)
    e = np.concatenate([R.df_LE["onset"].to_numpy(), R.df_RE["onset"].to_numpy()]).astype(int)
    counts = np.histogram((e[None, :] - h[:, None]).ravel(), edges)[0]
    expected = len(h) * len(e) / len(R.LE_vx) * np.diff(edges)
    return (edges[:-1] + edges[1:]) / 2 / FS * 1000, counts, expected


def head_eye_windows(R, head_df, flip_eye="LE", pre=24, post=72, pair_window=30, pscr_win=24):
    """One row per eye saccade with its head context (E), plus onset-aligned traces (W).

    E keeps EVERY eye saccade, so timing measures are not filtered by trace quality;
    trace-derived columns are NaN unless the window is clean. Columns:
      onset, peak, eye, amplitude_deg, peak_velocity_deg_s   as extracted
      amp_h, eye_disp    horizontal |displacement| and signed displacement, peak - onset
      head_idx, paired   head saccade the onset falls in (in_head_saccade); -1 / False
      lag_ms             eye onset - head onset (positive = eye followed); NaN if unpaired
      phase              (eye onset - head onset) / head duration: 0 = head onset, 1 = end
      head_sign          sign of the paired head saccade's unwrapped yaw displacement
                         (unpaired: dominant head velocity in the window)
      same_direction     1 = eye moved with the paired head saccade, 0 = against, else NaN
      clean              window [onset-pre, onset+post) in range and NaN-free
      head_peak_vel      peak head velocity after eye onset (sign-aligned, deg/s)
      eye_cr_vel         most negative eye velocity in the pscr_win frames after the eye
                         saccade ends: the counter-rotation (Wallace Fig 4D y)
      pscr_gain          mean eye / mean head velocity over that window when head > 30
                         deg/s there; -1 = full compensation
    W: head_pos, eye_pos, head_vel, eye_vel (n x pre+post) in E's row order, baseline-
    subtracted at onset and sign-aligned so the head turns positive; NaN rows where not
    clean. Head and eye velocity get the same savgol(11, 3). Use flip_eye="LE" (head frame).
    """
    yaw = unwrap_deg(R.yaw)
    yv = np.rad2deg(savgol_filter(np.asarray(R.yaw_v, float), 11, 3))
    h_on = head_df["onset"].to_numpy().astype(int) if len(head_df) else np.array([], int)
    h_pk = head_df["peak"].to_numpy().astype(int) if len(head_df) else np.array([], int)
    h_sign = np.sign(yaw[h_pk] - yaw[h_on])
    win = pre + post
    rows, traces = [], {k: [] for k in ("head_pos", "eye_pos", "head_vel", "eye_vel")}

    for eye, df in (("LE", R.df_LE), ("RE", R.df_RE)):
        px = eye_signal(R, eye, "x", flip_eye)
        vx = savgol_filter(eye_signal(R, eye, "vx", flip_eye), 11, 3)
        onsets = df["onset"].to_numpy().astype(int)
        peaks = df["peak"].to_numpy().astype(int)
        idx = in_head_saccade(onsets, head_df, pair_window)

        for o, p, i, amp, pkv in zip(onsets, peaks, idx, df["amplitude_deg"].to_numpy(float),
                                     df["peak_velocity_deg_s"].to_numpy(float)):
            a, b = o - pre, o + post
            clean = bool(a >= 0 and b <= len(px) and np.all(np.isfinite(yaw[a:b]))
                         and np.all(np.isfinite(yv[a:b])) and np.all(np.isfinite(px[a:b]))
                         and np.all(np.isfinite(vx[a:b])))
            if i >= 0:
                s = h_sign[i] or 1.0
            elif clean:
                s = np.sign(yv[a:b][np.argmax(np.abs(yv[a:b]))]) or 1.0
            else:
                s = np.nan

            if clean:
                hp, ep = (yaw[a:b] - yaw[o]) * s, (px[a:b] - px[o]) * s
                hv, ev = yv[a:b] * s, vx[a:b] * s
                k = p - o + pre
                w = slice(k + 1, min(k + 1 + pscr_win, win))
                hpk = hv[pre:].max()
                cr = ev[w].min() if w.stop > w.start else np.nan
                gain = (ev[w].mean() / hv[w].mean()
                        if w.stop > w.start and hv[w].mean() > 30 else np.nan)
            else:
                hp = ep = hv = ev = np.full(win, np.nan)
                hpk = cr = gain = np.nan

            disp = px[p] - px[o]
            paired = i >= 0
            rows.append((o, p, eye, amp, pkv, abs(disp), disp, i, paired,
                         (o - h_on[i]) / FS * 1000 if paired else np.nan,
                         (o - h_on[i]) / max(h_pk[i] - h_on[i], 1) if paired else np.nan,
                         s,
                         float(np.sign(disp) == s) if paired and np.isfinite(disp) and disp
                         else np.nan,
                         clean, hpk, cr, gain))
            for key, tr in zip(traces, (hp, ep, hv, ev)):
                traces[key].append(tr)

    E = pd.DataFrame(rows, columns=["onset", "peak", "eye", "amplitude_deg",
                                    "peak_velocity_deg_s", "amp_h", "eye_disp", "head_idx",
                                    "paired", "lag_ms", "phase", "head_sign",
                                    "same_direction", "clean", "head_peak_vel", "eye_cr_vel",
                                    "pscr_gain"]).assign(id=R.id, eo=R.eo)
    W = {k: np.array(v).reshape(-1, win) for k, v in traces.items()}
    return E, W


def head_triggered_windows(R, head_df, flip_eye="LE", pre=24, post=144):
    """Head-onset-aligned traces, one row per head saccade.

    Dict of (n_head x pre+post) arrays head_pos, LE_pos, RE_pos, baseline-subtracted at head
    onset and sign-aligned so the head turns positive. NaN rows where the window runs off
    the recording; eye rows may contain NaN. In the head frame (flip_eye="LE"),
    head_pos + eye_pos is gaze.
    """
    yaw = unwrap_deg(R.yaw)
    eyes = {eye: eye_signal(R, eye, "x", flip_eye) for eye in ("LE", "RE")}
    out = {k: np.full((len(head_df), pre + post), np.nan) for k in ("head_pos", "LE_pos", "RE_pos")}
    for i, (o, p) in enumerate(zip(head_df["onset"].to_numpy().astype(int),
                                   head_df["peak"].to_numpy().astype(int))):
        a, b = o - pre, o + post
        if a < 0 or b > len(yaw):
            continue
        s = np.sign(yaw[p] - yaw[o]) or 1.0
        out["head_pos"][i] = (yaw[a:b] - yaw[o]) * s
        for eye, x in eyes.items():
            out[f"{eye}_pos"][i] = (x[a:b] - x[o]) * s
    return out


def plot_mean_se(ax, t, arr, color, label, min_n=10):
    """Mean +- SE of the rows of arr; skipped when fewer than min_n rows."""
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or len(arr) < min_n:
        return
    m = arr.mean(axis=0)
    se = arr.std(axis=0) / np.sqrt(len(arr))
    ax.plot(t, m, color=color, lw=1, label=f"{label} (n={len(arr)})")
    ax.fill_between(t, m - se, m + se, color=color, alpha=0.25)


def paired_saccades(R, pair_window=12, flip_eye="RE", axis="horizontal"):
    """One row per LE saccade matched to its nearest RE saccade by onset.

    LE and RE saccades are detected independently and do NOT co-occur reliably (only 0.39
    of LE saccades have an RE partner within 100 ms on ferrets 402/420, range 0.02-0.76
    across sessions), so binocular measures must pair explicitly. Failure to pair is mostly
    tracking dropout: of unpaired LE saccades the fraction where RE is NaN rises
    0.10 -> 0.34 -> 0.58 across EO 0-4 / 5-9 / 10-20, and the unpaired eye still exceeds
    40 deg/s in 74-89% of cases. `paired` is a data-quality flag, not monocular movement.

    Greedy nearest-onset matching within pair_window FRAMES, each RE saccade used once.
    axis: "horizontal" -> x | "vertical" -> y | "total" -> 2D magnitude.
    eye_signal flips only keys ending in "x", so flip_eye is a no-op for "vertical" and is
    bypassed for "total" — correct, since those axes are already shared.

    Columns: LE_onset, RE_onset, lag_ms, paired, LE_disp, RE_disp (signed displacement
    pos[peak]-pos[onset], deg), LE_amp, RE_amp, LE_pkv, RE_pkv, other_nan, other_max_speed.
    Unpaired rows keep their LE columns and carry NaN for the RE ones.
    """
    key = {"horizontal": "x", "vertical": "y", "total": "x"}[axis]
    lx = eye_signal(R, "LE", key, flip_eye)
    rx = eye_signal(R, "RE", key, flip_eye)
    rs = eye_signal(R, "RE", "speed")
    a, b = R.df_LE, R.df_RE
    ao = a["onset"].to_numpy().astype(int)
    ap = a["peak"].to_numpy().astype(int)
    bo = b["onset"].to_numpy().astype(int)
    bp = b["peak"].to_numpy().astype(int)

    taken = set()
    rows = []
    for i in range(len(ao)):
        j, lag = -1, np.nan
        if len(bo):
            d = bo - ao[i]
            order = np.argsort(np.abs(d))
            for k in order:
                if abs(d[k]) <= pair_window and k not in taken:
                    j, lag = k, d[k]
                    taken.add(k)
                    break
        w = rs[ao[i]:ap[i] + 1]
        rows.append((ao[i], bo[j] if j >= 0 else np.nan,
                     lag / FS * 1000 if np.isfinite(lag) else np.nan, j >= 0,
                     lx[ap[i]] - lx[ao[i]],
                     rx[bp[j]] - rx[bo[j]] if j >= 0 else np.nan,
                     a["amplitude_deg"].to_numpy()[i],
                     b["amplitude_deg"].to_numpy()[j] if j >= 0 else np.nan,
                     a["peak_velocity_deg_s"].to_numpy()[i],
                     b["peak_velocity_deg_s"].to_numpy()[j] if j >= 0 else np.nan,
                     not np.all(np.isfinite(w)) if len(w) else True,
                     np.nanmax(w) if len(w) and np.any(np.isfinite(w)) else np.nan))

    return pd.DataFrame(rows, columns=["LE_onset", "RE_onset", "lag_ms", "paired",
                                       "LE_disp", "RE_disp", "LE_amp", "RE_amp",
                                       "LE_pkv", "RE_pkv", "other_nan", "other_max_speed"])


def pooled_pairs(group, pair_window=12, flip_eye="RE", axis="horizontal"):
    """paired_saccades concatenated over a session group, plus `id` and `eo` columns so
    per-session spread stays visible inside a pooled EO panel."""
    out = []
    for R in group:
        P = paired_saccades(R, pair_window, flip_eye, axis)
        out.append(P.assign(id=R.id, eo=R.eo))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def conjugate_samples(group, data_mode="events", signal="velocity", flip_eye="RE",
                      axis="horizontal", pair_window=12, frame_sacc="saccade",
                      frame_loco="all", head_range=None, speed_threshold=100, min_bout=30):
    """(LE, RE) sample pairs for a session group, selected by data_mode.

    The one place the events-vs-continuous choice is resolved, so every cell asks the same
    question of the same data and only the mode differs.

    data_mode "events"     -> one pair per paired saccade: signed displacement
                              pos[peak]-pos[onset] (signal ignored; a displacement is a
                              position DIFFERENCE and so is registration-free).
              "continuous" -> one pair per retained frame, from frame_mask(frame_sacc,
                              frame_loco), further gated by head_range.
              "both"       -> (ev_LE, ev_RE, co_LE, co_RE).

    signal "velocity" -> vx (recommended) | "position" -> x. Continuous POSITION is known
    not to work: per-session corr(LE_x, RE_x) spans -0.35..+0.21 and flips sign within one
    animal across consecutive days, and no curation tried removes it (non-saccade +
    stationary + head-still, head-moving, and saccade-only frames all sit near zero). The
    registration error is a per-session DC offset, which differencing removes and averaging
    does not — hence velocity works and position does not. Kept as an option so the
    assumption can be re-tested after any loading or threshold change.

    head_range: None, or (lo, hi) in deg/s on head_signal(R, "speed"), continuous only.
    Non-finite pairs are dropped. Returns equal-length float arrays.
    """
    ax_key = {"horizontal": "x", "vertical": "y", "total": "x"}[axis]
    key = ax_key if signal == "position" else "v" + ax_key

    ev_l, ev_r, co_l, co_r = [], [], [], []

    for R in group:
        if data_mode in ("events", "both"):
            P = paired_saccades(R, pair_window, flip_eye, axis)
            P = P[P.paired]
            ev_l.append(P.LE_disp.to_numpy().astype(float))
            ev_r.append(P.RE_disp.to_numpy().astype(float))

        if data_mode in ("continuous", "both"):
            m = frame_mask(R, frame_sacc, frame_loco, speed_threshold, min_bout)
            if head_range is not None:
                hs = head_signal(R, "speed")
                m = m & (hs >= head_range[0]) & (hs < head_range[1])
            co_l.append(eye_signal(R, "LE", key, flip_eye)[m])
            co_r.append(eye_signal(R, "RE", key, flip_eye)[m])

    def finite(ls, rs):
        if not ls:
            return np.array([]), np.array([])
        x, y = np.concatenate(ls), np.concatenate(rs)
        ok = np.isfinite(x) & np.isfinite(y)
        return x[ok], y[ok]

    if data_mode == "both":
        return finite(ev_l, ev_r) + finite(co_l, co_r)
    return finite(ev_l, ev_r) if data_mode == "events" else finite(co_l, co_r)


def logamp_logvel(group):
    """log10 amplitude and log10 peak velocity, both eyes pooled over a session group."""
    amp = np.concatenate([np.concatenate([R.df_LE["amplitude_deg"].to_numpy(),
                                          R.df_RE["amplitude_deg"].to_numpy()]) for R in group]).astype(float)
    pkv = np.concatenate([np.concatenate([R.df_LE["peak_velocity_deg_s"].to_numpy(),
                                          R.df_RE["peak_velocity_deg_s"].to_numpy()]) for R in group]).astype(float)
    x = np.log10(abs(amp))
    y = np.log10(abs(pkv))
    inds = np.isfinite(x) & np.isfinite(y)
    return x[inds], y[inds]


def session_rate(R, signal, condition="all", speed_threshold=100, min_bout=30,
                 head_still_thresh=50, min_exposure_sec=5.0):
    """One session's event rate (Hz): in-condition events / in-condition time.

    Per eye, counting only frames on which that eye is tracked (finite position), then
    averaged over eyes with >= min_exposure_sec. Returns nan when neither eye has enough,
    so callers can align rates across conditions by session index.
    """
    m = condition_frames(R, condition, speed_threshold, min_bout, head_still_thresh)
    rates = []
    for eye, df in zip(("LE", "RE"), event_dfs(R, signal)):
        pos = np.asarray(getattr(R, f"{eye}_x" if signal == "eye"
                                 else f"{eye}_gaze_horizontal_deg"), float)
        me = m & np.isfinite(pos)
        exposure = me.sum() / FS
        if exposure < min_exposure_sec:
            continue
        onsets = df["onset"].to_numpy().astype(int) if len(df) else np.array([], int)
        rates.append(me[onsets].sum() / exposure)
    return np.mean(rates) if rates else np.nan


def session_rates(group, signal, condition="all", speed_threshold=100, min_bout=30,
                  head_still_thresh=50, min_exposure_sec=5.0):
    """Per-session event rates (Hz) for a group, dropping sessions with too little
    exposure.

    Replaces a sliding window for fragmented conditions. "stationary_and_head_still"
    bouts have a median length of ~0.03-0.06 s, so no window both fits inside the
    condition and is long enough to estimate a ~1 Hz rate; measuring total exposure
    sidesteps windowing entirely.
    """
    r = np.array([session_rate(R, signal, condition, speed_threshold, min_bout,
                               head_still_thresh, min_exposure_sec) for R in group])
    return r[np.isfinite(r)]


def load_results(ferrets=FERRETS, **params):
    """Load, clean and extract every session of the given ferrets with one shared set of
    extraction params (EXTRACT, overridable per call)."""
    Results = []
    for session in getSesh.by_ferret(*ferrets):
        R = load_session_data(session)
        removeBadData(R)
        process_session(R, **{**EXTRACT, **params})
        Results.append(R)
    print(len(Results), "sessions loaded")
    return Results


def set_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 6
    plt.rcParams['svg.fonttype'] = 'none'


def save_fig(fig, name):
    """Save fig as SVG in SAVELOC (no-op when SAVELOC is not set)."""
    if SAVELOC is not None:
        fig.savefig(SAVELOC / f"{name}.svg", format="svg", bbox_inches="tight")


def session_trend(df, col, label=None):
    """EO trend of one per-session measure (df needs columns id, eo, col).

    Prints pooled-session Spearman, a mixed model (EO fixed, ferret random intercept) and
    per-ferret Spearman for ferrets with >= 4 sessions. Sessions are the unit.
    """
    d = df[["id", "eo", col]].dropna()
    label = label or col
    rho, p = spearmanr(d.eo, d[col])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # 4 ferrets: convergence warnings are routine
        fit = smf.mixedlm(f"{col} ~ eo", d, groups=d["id"]).fit()
    print(f"{label:24s} n={len(d):3d}  rho={rho:+.3f} p={p:.3g}  |  mixedlm "
          f"{fit.params['eo']:+.4f}/EO day  SE={fit.bse['eo']:.4f}  p={fit.pvalues['eo']:.3g}")
    for fid, g in d.groupby("id"):
        if len(g) >= 4:
            r, pp = spearmanr(g.eo, g[col])
            print(f"{'':24s} F{fid} n={len(g):2d}  rho={r:+.3f} p={pp:.3g}")
    return fit


def plot_vs_eo(ax, df, col, color="k"):
    """One line per ferret of a per-session measure against EO."""
    for i, (fid, g) in enumerate(df.groupby("id")):
        g = g.sort_values("eo")
        ax.plot(g.eo, g[col], marker=FERRET_MARKERS[i % len(FERRET_MARKERS)], ms=3, lw=0.8,
                color=color, label=f"F{fid}")
    ax.set_xlabel("EO (days)")
    ax.set_ylabel(col)
