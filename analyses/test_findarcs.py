# %%
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
from utils import load_session_data
from analyses.helper_functions import unwrap_deg


# ============================== SETTINGS ==============================
SESSION = 'session_2026-03-14_ferret_407_P47_E14_analyzable_output'   # head position_x / position_y (mm)
CHECK_SESSIONS = ('session_2025-10-22_ferret_420_EO13_analyzable_output',   # held-out checks (cell 6),
                  'session_2026-02-28_ferret_405_EO0_analyzable_output',    # incl. young / slow animals
                  'session_2025-10-11_ferret_402_E02_analyzable_output')
FS = 120                         # frame rate (Hz)

# Two scales. Light smoothing keeps the edges of tight arcs; large arcs only rise above the head
# wiggle after heavy smoothing. Coarse-scale arcs are added where no fine-scale arc of the same
# direction is.
# fine scale: tight arcs
SIGMA_S = 0.08       # Gaussian smoothing sigma (s)
KAPPA_HI = 0.015     # 1/mm; an arc must reach |curvature| above this (radius < 1/KAPPA_HI) ...
KAPPA_LO = 0.01125   # 1/mm; ... and extends while |curvature| stays above this (0.75 x KAPPA_HI)
EDGE_FRAC = 0.0      # arc edges: where |curvature| falls below this fraction of the arc's own peak (0 = off)
# coarse scale: large arcs
SIGMA_S_COARSE = 0.24
KAPPA_HI_COARSE = 0.001      # radius < 1000 mm; liberal - MIN_TURN_DEG does most of the selecting
KAPPA_LO_COARSE = 0.00075
EDGE_FRAC_COARSE = 0.33
MERGE_GAP_S_COARSE = 0.4     # bridges the short slow dips (speed < SPEED_MIN) that split large arcs
# both scales
MAX_GAP_S = 0.2      # interpolate tracking gaps up to this long; longer gaps stay NaN
SPEED_MIN = 50.0     # mm/s; set above the speed noise floor when the animal is still
MERGE_GAP_S = 0.0    # same-direction arcs closer than this are joined (fine scale)
MIN_DUR_S = 0.3      # shortest arc kept (s), applied after merging
MIN_TURN_DEG = 30    # an arc must turn the heading at least this much in total (any radius); liberal,
                     # stricter cutoffs can be applied later to the table (cell 3, reliability)

PLOT_WINDOW_S = (0, 60)          # time range drawn in the figures (s)

# synthetic ground truth (cells 2, 3, 5)
NOISE_SIGMA_S = 0.1              # real wiggle + jitter = raw - Gaussian(raw, this) is added to the synthetic path
ARC_RATE_PER_MIN = 100           # planted arcs per minute of moving time (target; placement saturates)
RADIUS_RANGE = (20, 1000)        # mm, planted radius (log-uniform within RADIUS_BINS, bins in turn)
TURN_RANGE_DEG = (45, 180)       # planted turn angle (uniform)
TRUTH_MIN_DUR_S = 0.3            # planted arcs are at least this long
BG_RADIUS = 1500                 # mm; background heading wander mostly has radius > this (= not an arc)
BG_SMOOTH_S = 1.0                # s; time scale of the background wander
N_SYNTH = 5                      # synthetic realizations per score
RADIUS_BINS = (20, 50, 100, 200, 400, 700, 1000)   # mm, planting cycles through these; recall reported per bin

# selection (cell 5): among coarse settings whose detections are this reliable, take the highest recall
PREC_MIN = 0.90                  # fraction of detected arcs that are real
FRAME_PREC_MIN = 0.83            # fraction of detected arc frames that lie inside a real arc
# ======================================================================

P = dict(sigma_s=SIGMA_S, max_gap_s=MAX_GAP_S, speed_min=SPEED_MIN, kappa_hi=KAPPA_HI,
         kappa_lo=KAPPA_LO, merge_gap_s=MERGE_GAP_S, min_dur_s=MIN_DUR_S, min_turn_deg=MIN_TURN_DEG,
         edge_frac=EDGE_FRAC)
PC = dict(P, sigma_s=SIGMA_S_COARSE, kappa_hi=KAPPA_HI_COARSE, kappa_lo=KAPPA_LO_COARSE, edge_frac=EDGE_FRAC_COARSE,
          merge_gap_s=MERGE_GAP_S_COARSE)
LEFT_C, RIGHT_C, COARSE_C = '#2a78d6', '#eb6834', '#8a8984'


def fill_gaps(v, max_gap):
    """Linearly interpolate NaN runs; runs longer than max_gap frames are returned as NaN."""
    v = np.asarray(v, float)
    bad = np.isnan(v)
    idx = np.arange(len(v))
    filled = np.interp(idx, idx[~bad], v[~bad])
    d = np.diff(bad.astype(int), prepend=0, append=0)
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if e - s > max_gap:
            filled[s:e] = np.nan
    return filled


def smooth(v, fs, sigma_s, max_gap_s):
    """Fill gaps, Gaussian-smooth, then restore long gaps as NaN."""
    f = fill_gaps(v, max_gap_s * fs)
    long_gap = np.isnan(f)
    s = gaussian_filter1d(np.nan_to_num(f, nan=np.nanmean(f)), sigma_s * fs)
    s[long_gap] = np.nan
    return s


def kinematics(x, y, fs):
    dt = 1 / fs
    dx, dy = np.gradient(x, dt), np.gradient(y, dt)
    ddx, ddy = np.gradient(dx, dt), np.gradient(dy, dt)
    speed = np.hypot(dx, dy)
    kappa = (dx * ddy - dy * ddx) / np.maximum(speed, 1e-9) ** 3   # signed: + = left (CCW)
    return speed, kappa


def runs(m):
    """Start and end (exclusive) frames of the True runs in m."""
    d = np.diff(m.astype(int), prepend=0, append=0)
    return np.where(d == 1)[0], np.where(d == -1)[0]


def arc_segments(speed, kappa, fs, speed_min, kappa_hi, kappa_lo, merge_gap_s, min_dur_s, min_turn_deg, edge_frac):
    """(start, end, sign) per arc. Left (+1) and right (-1) are found separately: runs above
    kappa_lo that reach kappa_hi. Each run's edges are then pulled in to where curvature is above
    edge_frac x the run's peak (which also splits neighbouring arcs at a dip between them).
    Arcs are joined across gaps <= merge_gap_s, then kept if >= min_dur_s and turning
    >= min_turn_deg in total (heading change = sum of kappa * speed * dt)."""
    moving = speed > speed_min                                    # NaN compares False
    segs = []
    for sign in (1, -1):
        k = sign * kappa
        hi = moving & (k > kappa_hi)
        kept = []
        for s0, e0 in zip(*runs(moving & (k > kappa_lo))):
            if not hi[s0:e0].any():
                continue
            thr = max(kappa_lo, edge_frac * k[s0:e0].max())
            for s, e in zip(*runs(k[s0:e0] > thr)):
                s, e = s + s0, e + s0
                if not hi[s:e].any():
                    continue
                if kept and s - kept[-1][1] <= merge_gap_s * fs:
                    kept[-1] = (kept[-1][0], e)
                else:
                    kept.append((s, e))
        segs += [(s, e, sign) for s, e in kept if (e - s) / fs >= min_dur_s
                 and np.rad2deg(np.nansum(k[s:e] * speed[s:e]) / fs) >= min_turn_deg]
    return sorted(segs)


def find_arcs(x_raw, y_raw, fs, sigma_s, max_gap_s, speed_min, kappa_hi, kappa_lo, merge_gap_s, min_dur_s, min_turn_deg,
              edge_frac):
    """Raw 2D position -> arcs at one smoothing scale. Returns segs [(start, end, sign)], smoothed x, y, speed, kappa."""
    x = smooth(x_raw, fs, sigma_s, max_gap_s)
    y = smooth(y_raw, fs, sigma_s, max_gap_s)
    speed, kappa = kinematics(x, y, fs)
    segs = arc_segments(speed, kappa, fs, speed_min, kappa_hi, kappa_lo, merge_gap_s, min_dur_s, min_turn_deg,
                        edge_frac)
    return segs, x, y, speed, kappa


def find_arcs_2scale(x_raw, y_raw, fs, fine, coarse=None):
    """Fine-scale arcs, plus coarse-scale arcs where no fine arc of the same direction is
    (coarse=None: fine only). Returns segs [(start, end, sign, scale)] and
    K = {scale: (x, y, speed, kappa)}."""
    segs_f, *k_fine = find_arcs(x_raw, y_raw, fs, **fine)
    K = {'fine': tuple(k_fine)}
    segs = [(s, e, sign, 'fine') for s, e, sign in segs_f]
    if coarse is not None:
        segs_c, *k_coarse = find_arcs(x_raw, y_raw, fs, **coarse)
        K['coarse'] = tuple(k_coarse)
        in_fine = {1: np.zeros(len(x_raw), bool), -1: np.zeros(len(x_raw), bool)}
        for s, e, sign in segs_f:
            in_fine[sign][s:e] = True
        segs += [(s, e, sign, 'coarse') for s, e, sign in segs_c if not in_fine[sign][s:e].any()]
    return sorted(segs), K


def arc_stats(k, s, e, fs):
    """Turn (deg, + = left; = sum of kappa * speed * dt), path length (mm) and mean radius
    (path length / turn) over frames s:e of one scale k = (x, y, speed, kappa)."""
    _, _, speed, kappa = k
    turn = np.nansum(kappa[s:e] * speed[s:e]) / fs
    length = np.nansum(speed[s:e]) / fs
    return np.rad2deg(turn), length, length / max(abs(turn), 1e-9)


def load(session):
    """Head position (NaN where head speed >= 800, as in utils/process_session.py), speed, yaw rate."""
    R = load_session_data(session)
    vv = np.hypot(R.linearVel_x, R.linearVel_y)
    bad = ~(vv < 800)
    x_raw = np.where(bad, np.nan, R.position_x).astype(float)
    y_raw = np.where(bad, np.nan, R.position_y).astype(float)
    w_yaw = np.gradient(unwrap_deg(np.where(bad, np.nan, R.yaw))) * FS   # world head yaw rate (deg/s, + = left)
    return R, x_raw, y_raw, np.where(bad, np.nan, vv), w_yaw


def arc_table(segs, K, w_yaw, fs):
    """One row per arc, measured on the arc's own scale. yaw_sign_ok / yaw_r compare path turn
    rate (speed * kappa) with head yaw rate."""
    rows = []
    for s, e, sign, scale in segs:
        _, _, speed, kappa = K[scale]
        turn, length, radius = arc_stats(K[scale], s, e, fs)
        w_path = np.rad2deg(speed[s:e] * kappa[s:e])
        rows.append(dict(start_frame=s, end_frame=e, start_s=s / fs, end_s=e / fs, duration_s=(e - s) / fs,
                         direction='left' if sign > 0 else 'right', scale=scale,
                         radius=radius, turn_deg=turn, mean_speed=np.nanmean(speed[s:e]), path_length=length,
                         yaw_sign_ok=np.sign(np.nanmean(w_yaw[s:e])) == sign,
                         yaw_r=pd.Series(w_path).corr(pd.Series(w_yaw[s:e]))))
    return pd.DataFrame(rows)


def make_synth(v, resid_x, resid_y, nan_mask, fs, rng, arc_rate_per_min):
    """Synthetic path with the real speed trace, real wiggle/jitter (resid) and real gaps.
    Heading = background wander + planted arcs of known radius and turn.
    Returns x_raw, y_raw, truth [(start, end, sign, radius, turn_deg)]."""
    v = np.nan_to_num(v)
    n = len(v)
    moving = v > SPEED_MIN
    dist = np.cumsum(v) / fs
    k_bg = gaussian_filter1d(rng.standard_normal(n), BG_SMOOTH_S * fs)
    kappa = k_bg / k_bg.std() * 0.5 / BG_RADIUS                  # ~95% of frames: radius > BG_RADIUS
    n_arcs = int(arc_rate_per_min * moving.sum() / fs / 60)
    gap = int(0.5 * fs)                                          # planted arcs stay >= 0.5 s apart
    taken = np.zeros(n, bool)
    truth = []
    # radius bins in turn, largest first, so large arcs (which need long moving stretches) get placed
    bins = [(max(lo, RADIUS_RANGE[0]), min(hi, RADIUS_RANGE[1])) for lo, hi in zip(RADIUS_BINS[:-1], RADIUS_BINS[1:])
            if lo < RADIUS_RANGE[1]][::-1]
    for i in range(n_arcs):
        for _ in range(500):
            s = rng.integers(n)
            r = np.exp(rng.uniform(*np.log(bins[i % len(bins)])))
            turn = rng.uniform(*TURN_RANGE_DEG)
            e = np.searchsorted(dist, dist[s] + r * np.deg2rad(turn))
            if (e >= n or (e - s) / fs < TRUTH_MIN_DUR_S or not (moving[s] and moving[e - 1])
                    or moving[s:e].mean() < 0.8 or taken[max(0, s - gap):e + gap].any()):
                continue
            sign = rng.choice([1, -1])
            kappa[s:e] = sign / r
            taken[s:e] = True
            truth.append((s, e, sign, r, turn))
            break
    heading = np.cumsum(v * kappa) / fs
    x = np.cumsum(v * np.cos(heading)) / fs + np.nan_to_num(resid_x)
    y = np.cumsum(v * np.sin(heading)) / fs + np.nan_to_num(resid_y)
    x[nan_mask], y[nan_mask] = np.nan, np.nan
    return x, y, sorted(truth)


def score(segs, truth, K, fs):
    """Truth arc found = >= 50% of its frames inside same-direction detections.
    Detection correct = >= 50% of its frames inside same-direction truth.
    Errors are from the detection overlapping each found truth arc most."""
    n = len(K['fine'][0])
    det = {1: np.zeros(n, bool), -1: np.zeros(n, bool)}
    tru = {1: np.zeros(n, bool), -1: np.zeros(n, bool)}
    for s, e, sign, _ in segs:
        det[sign][s:e] = True
    for s, e, sign, *_ in truth:
        tru[sign][s:e] = True
    found, r_true, rad_err, turn_err, on_err, off_err = [], [], [], [], [], []
    for ts, te, tsign, tr, tturn in truth:
        found.append(det[tsign][ts:te].mean() >= 0.5)
        r_true.append(tr)
        if not found[-1]:
            continue
        s, e, _, scale = max((d for d in segs if d[2] == tsign), key=lambda d: min(d[1], te) - max(d[0], ts))
        turn, _, radius = arc_stats(K[scale], s, e, fs)
        rad_err.append(abs(np.log(radius / tr)))
        turn_err.append(abs(turn) - tturn)
        on_err.append((s - ts) / fs * 1000)
        off_err.append((e - te) / fs * 1000)
    found, r_true = np.array(found, bool), np.array(r_true)
    correct = [tru[sign][s:e].mean() >= 0.5 for s, e, sign, _ in segs]
    recall = found.mean() if len(found) else np.nan
    precision = np.mean(correct) if correct else np.nan
    n_det_frames = det[1].sum() + det[-1].sum()
    out = dict(n_truth=len(truth), n_det=len(segs), recall=recall, precision=precision,
               frame_prec=((det[1] & tru[1]).sum() + (det[-1] & tru[-1]).sum()) / n_det_frames if n_det_frames else np.nan,
               rec_large=found[r_true >= 100].mean() if (r_true >= 100).any() else np.nan,   # recall, radius >= 100 mm
               rad_err=np.median(rad_err) if rad_err else np.nan,       # median |log(R_det / R_true)|
               turn_err=np.median(turn_err) if turn_err else np.nan,    # median (|det turn| - true turn), deg
               onset_ms=np.median(on_err) if on_err else np.nan,
               offset_ms=np.median(off_err) if off_err else np.nan)
    for lo, hi in zip(RADIUS_BINS[:-1], RADIUS_BINS[1:]):
        m = (r_true >= lo) & (r_true < hi)
        out[f'rec_R{lo}-{hi}'] = found[m].mean() if m.any() else np.nan
    return out


def synth_set(session_data, n_synth, arc_rate_per_min, seed=0):
    """n_synth synthetic realizations built from one real session."""
    R, x_raw, y_raw, vv, _ = session_data
    resid_x = x_raw - smooth(x_raw, FS, NOISE_SIGMA_S, MAX_GAP_S)
    resid_y = y_raw - smooth(y_raw, FS, NOISE_SIGMA_S, MAX_GAP_S)
    nan_mask = np.isnan(x_raw) | np.isnan(y_raw)
    rng = np.random.default_rng(seed)
    return [make_synth(vv, resid_x, resid_y, nan_mask, FS, rng, arc_rate_per_min) for _ in range(n_synth)]


def synth_score(synths, fine, coarse=None):
    """Mean score over synthetic realizations."""
    out = []
    for xs, ys, truth in synths:
        segs, K = find_arcs_2scale(xs, ys, FS, fine, coarse)
        out.append(score(segs, truth, K, FS))
    return pd.DataFrame(out).mean()


def real_summary(session_data, fine, coarse=None):
    """Real-data statistics (no ground truth: counts, sizes, yaw agreement)."""
    R, x_raw, y_raw, _, w_yaw = session_data
    segs, K = find_arcs_2scale(x_raw, y_raw, FS, fine, coarse)
    arcs = arc_table(segs, K, w_yaw, FS)
    moving = K['fine'][2] > fine['speed_min']
    in_arc = np.zeros(len(moving), bool)
    for s, e, *_ in segs:
        in_arc[s:e] = True
    return dict(n_arcs=len(arcs),
                n_coarse=int((arcs.scale == 'coarse').sum()) if len(arcs) else 0,
                med_dur_s=arcs.duration_s.median() if len(arcs) else np.nan,
                med_radius=arcs.radius.median() if len(arcs) else np.nan,
                pct_moving_in_arcs=100 * (in_arc & moving).sum() / moving.sum(),
                pct_yaw_sign_ok=100 * arcs.yaw_sign_ok.mean() if len(arcs) else np.nan,
                med_yaw_r=arcs.yaw_r.median() if len(arcs) else np.nan)


def plot_arcs(x_raw, y_raw, segs, K, fine, coarse, title, truth=None):
    """Trajectory with arcs (left) and speed / |kappa| / kappa traces (right), PLOT_WINDOW_S only.
    Fine-scale arcs narrow, coarse-scale arcs wide and light; fine traces black, coarse grey.
    truth (synthetic): thin colored lines on the trajectory, bars at the bottom of the traces."""
    x, y, speed, kappa = K['fine']
    t = np.arange(len(x)) / FS
    w0, w1 = int(PLOT_WINDOW_S[0] * FS), int(PLOT_WINDOW_S[1] * FS)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.6], hspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(x_raw[w0:w1], y_raw[w0:w1], '.', ms=1, color='#b9b8b2', label='raw')
    ax.plot(x[w0:w1], y[w0:w1], color='#0b0b0b', lw=0.8, label='smoothed (fine)')
    for s, e, sign, scale in segs:
        if e > w0 and s < w1:
            ax.plot(x[s:e], y[s:e], lw=4 if scale == 'fine' else 8, alpha=0.6 if scale == 'fine' else 0.3,
                    color=LEFT_C if sign > 0 else RIGHT_C)
    for s, e, sign, *_ in truth or []:
        if e > w0 and s < w1:
            ax.plot(x[s:e], y[s:e], lw=1.2, color=LEFT_C if sign > 0 else RIGHT_C)
    ax.plot([], [], lw=4, color=LEFT_C, label='left arc'); ax.plot([], [], lw=4, color=RIGHT_C, label='right arc')
    ax.plot([], [], lw=8, alpha=0.3, color='0.4', label='coarse-scale arc')
    if truth is not None:
        ax.plot([], [], lw=1.2, color='0.3', label='truth (thin)')
    ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.legend(loc='best', fontsize=7)
    ax.set_title(title)

    a1 = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    a1.plot(t, speed, color='#0b0b0b', lw=1); a1.axhline(fine['speed_min'], color='#e34948', ls='--', lw=1)
    a1.set_ylabel('speed (mm/s)')
    scales = [(K['fine'], fine, '#0b0b0b', '#e34948')]
    if coarse is not None:
        scales.append((K['coarse'], coarse, COARSE_C, COARSE_C))
    for (_, _, sp, k), p, c, c_thr in scales:
        a2.semilogy(t, np.where(sp > p['speed_min'], np.abs(k), np.nan), color=c, lw=1)
        a2.axhline(p['kappa_hi'], color=c_thr, ls='--', lw=1); a2.axhline(p['kappa_lo'], color=c_thr, ls=':', lw=1)
        a3.plot(t, np.where(sp > p['speed_min'], k, np.nan), color=c, lw=1)
    a2.set_ylabel('|curvature| (1/mm)')
    a3.axhline(0, color='#8a8984', lw=0.6)
    a3.set_ylabel('curvature (+ left)'); a3.set_xlabel('time (s)')
    for a in (a1, a2, a3):
        for s, e, sign, scale in segs:
            a.axvspan(s / FS, e / FS, color=LEFT_C if sign > 0 else RIGHT_C, alpha=0.15 if scale == 'fine' else 0.07, lw=0)
        for s, e, sign, *_ in truth or []:
            a.axvspan(s / FS, e / FS, ymax=0.06, color=LEFT_C if sign > 0 else RIGHT_C, lw=0)
        a.spines[['top', 'right']].set_visible(False)
    a1.set_xlim(PLOT_WINDOW_S)
    return fig


# %% 1. real session: arcs, table, figure
D1 = load(SESSION)
R, x_raw, y_raw, vv, w_yaw = D1
OUT_CSV = f'arcs_{R.id}_EO{R.eo}.csv'
OUT_PNG = f'arcs_{R.id}_EO{R.eo}.png'

segs, K = find_arcs_2scale(x_raw, y_raw, FS, P, PC)
arcs = arc_table(segs, K, w_yaw, FS)
arcs.to_csv(OUT_CSV, index=False)
print(f'{len(arcs)} arcs found ({(arcs.scale == "fine").sum()} fine, {(arcs.scale == "coarse").sum()} coarse) -> {OUT_CSV}')
print(arcs.round(2).to_string(index=False))

fig = plot_arcs(x_raw, y_raw, segs, K, P, PC, f'{R.id} EO{R.eo}: trajectory and detected arcs')
fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
print(f'plot -> {OUT_PNG}')


# %% 2. noise floors
# SPEED_MIN: the speed histogram has no separate stationary peak, so read off how much of
# the slow tail SPEED_MIN cuts.
# KAPPA thresholds: |curvature| of a synthetic path with NO planted arcs (real speed, real
# wiggle, background wander only) is the floor that any threshold has to clear, per scale.
log_v = np.log10(vv[vv > 1])
print(f'{100 * np.mean(vv[np.isfinite(vv)] < SPEED_MIN):.0f}% of frames below SPEED_MIN = {SPEED_MIN} mm/s; '
      'speed p10/p25/p50 = ' + ' / '.join(f'{v:.0f}' for v in np.nanpercentile(vv, [10, 25, 50])))

straight = synth_set(D1, N_SYNTH, arc_rate_per_min=0)
for sig in (0.08, 0.16, 0.24, 0.32):
    ks = []
    for xs, ys, _ in straight:
        _, _, _, sp, k = find_arcs(xs, ys, FS, **{**P, 'sigma_s': sig})
        ks.append(np.abs(k[sp > SPEED_MIN]))
    print(f'sigma {sig:.2f} s: |kappa| on arc-free synthetic, p50/p90/p95/p99 = '
          + ' / '.join(f'{v:.4f}' for v in np.percentile(np.concatenate(ks), [50, 90, 95, 99])) + '  1/mm')

fig, axes = plt.subplots(1, 3, figsize=(13, 3))
axes[0].hist(log_v, 60, color='0.4')
axes[0].axvline(np.log10(SPEED_MIN), color='#e34948', ls='--', lw=1, label='SPEED_MIN')
axes[0].set_xlabel('log10 speed (mm/s)'); axes[0].legend(fontsize=7)
for a, p, name in ((axes[1], P, 'fine'), (axes[2], PC, 'coarse')):
    _, _, _, sp, k = find_arcs(x_raw, y_raw, FS, **p)
    a.hist(np.log10(np.abs(k[sp > SPEED_MIN])), 80, histtype='step', density=True, color='k', label='real')
    _, _, _, sp, k = find_arcs(*straight[0][:2], FS, **p)
    a.hist(np.log10(np.abs(k[sp > SPEED_MIN])), 80, histtype='step', density=True, color='#2a78d6', label='arc-free synthetic')
    a.axvline(np.log10(p['kappa_hi']), color='#e34948', ls='--', lw=1)
    a.axvline(np.log10(p['kappa_lo']), color='#e34948', ls=':', lw=1)
    a.set_xlabel('log10 |curvature| (1/mm), moving')
    a.set_title(f'{name} scale (sigma {p["sigma_s"]} s)', fontsize=8); a.legend(fontsize=6)
for a in axes:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()


# %% 3. synthetic ground truth: realism check, then score the current settings
synths = synth_set(D1, N_SYNTH, ARC_RATE_PER_MIN)
xs, ys, truth = synths[0]
print(f'{len(truth)} planted arcs per realization, {N_SYNTH} realizations')

# realism: real and synthetic should be hard to tell apart
_, xr, yr, spr, kr = find_arcs(x_raw, y_raw, FS, **P)
_, xq, yq, spq, kq = find_arcs(xs, ys, FS, **P)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1.2, 1.2]))
w0, w1 = int(PLOT_WINDOW_S[0] * FS), int(PLOT_WINDOW_S[1] * FS)
for a, (px, py, lbl) in zip(axes[:2], ((xr, yr, 'real'), (xq, yq, 'synthetic'))):
    a.plot(px[w0:w1], py[w0:w1], color='k', lw=0.7)
    a.set_aspect('equal'); a.set_title(f'{lbl}, {PLOT_WINDOW_S[0]}-{PLOT_WINDOW_S[1]} s', fontsize=8)
for k, sp, lbl, c in ((kr, spr, 'real', 'k'), (kq, spq, 'synthetic', '#2a78d6')):
    m = sp > SPEED_MIN
    axes[2].hist(np.log10(np.abs(k[m])), 80, histtype='step', density=True, color=c, label=lbl)
    axes[3].hist(np.rad2deg(sp[m] * k[m]), np.linspace(-600, 600, 121), histtype='step', density=True, color=c, label=lbl)
axes[2].set_xlabel('log10 |curvature| (1/mm), moving')
m, mq = spr > SPEED_MIN, spq > SPEED_MIN
print(f'real vs synthetic (Wasserstein; lower = more alike): '
      f'log10|curvature| {wasserstein_distance(np.log10(np.abs(kr[m])), np.log10(np.abs(kq[mq]))):.3f}, '
      f'turn rate {wasserstein_distance(np.rad2deg(spr[m] * kr[m]), np.rad2deg(spq[mq] * kq[mq])):.1f} deg/s')
axes[3].set_xlabel('path turn rate (deg/s), moving'); axes[3].set_yscale('log')
for a in axes[2:]:
    a.legend(fontsize=7); a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()

# one realization with truth vs detections
segs_q, K_q = find_arcs_2scale(xs, ys, FS, P, PC)
plot_arcs(xs, ys, segs_q, K_q, P, PC, 'synthetic: detected (thick) vs truth (thin)', truth)

print('mean over realizations:')
print(pd.DataFrame({'fine only': synth_score(synths, P), 'two-scale': synth_score(synths, P, PC)}).round(3).to_string())

# reliability for later cutoffs: detection is liberal, so how precise are the arcs that pass a
# stricter cutoff on measured |turn| and duration? (correct = >= 50% of its frames inside a
# same-direction planted arc, as in score)
det = []
for xs, ys, truth in synths:
    segs_s, K_s = find_arcs_2scale(xs, ys, FS, P, PC)
    tbl = arc_table(segs_s, K_s, np.full(len(xs), np.nan), FS)
    tru = {1: np.zeros(len(xs), bool), -1: np.zeros(len(xs), bool)}
    for s, e, sign, *_ in truth:
        tru[sign][s:e] = True
    tbl['correct'] = [tru[sign][s:e].mean() >= 0.5 for s, e, sign, _ in segs_s]
    det.append(tbl)
det = pd.concat(det, ignore_index=True)
rows = []
for min_turn in (30, 45, 60, 90):
    for min_dur in (0.3, 0.5, 1.0):
        sel = (det.turn_deg.abs() >= min_turn) & (det.duration_s >= min_dur)
        real_sel = (arcs.turn_deg.abs() >= min_turn) & (arcs.duration_s >= min_dur)
        rows.append(dict(min_turn_deg=min_turn, min_dur_s=min_dur,
                         synth_precision=det.correct[sel].mean(), synth_pct_kept=100 * sel.mean(),
                         fine_precision=det.correct[sel & (det.scale == 'fine')].mean(),
                         coarse_precision=det.correct[sel & (det.scale == 'coarse')].mean(),
                         real_arcs_kept=int(real_sel.sum())))
print(f'reliability after a later cutoff (synthetic: {len(det)} detections; real {R.id} EO{R.eo}: {len(arcs)} arcs):')
print(pd.DataFrame(rows).round(3).to_string(index=False))


# %% 4. yaw-rate check on real data
# Path turn rate (speed * curvature) should follow head yaw rate while the animal turns.
# Per arc: does mean yaw rate have the arc's sign, and how well do the two rates correlate?
_, _, speed, kappa = K['fine']
w_path = np.rad2deg(speed * kappa)
moving = speed > SPEED_MIN
in_arc = np.zeros(len(speed), bool)
for s, e, *_ in segs:
    in_arc[s:e] = True
ok = np.isfinite(w_path) & np.isfinite(w_yaw)
for scale in ('fine', 'coarse'):
    a_s = arcs[arcs.scale == scale]
    print(f'{scale}: yaw sign agrees in {100 * a_s.yaw_sign_ok.mean():.0f}% of {len(a_s)} arcs; '
          f'median within-arc r = {a_s.yaw_r.median():+.2f}')
print(f'r(path turn rate, yaw rate): in arcs {np.corrcoef(w_path[ok & in_arc], w_yaw[ok & in_arc])[0, 1]:+.2f}, '
      f'all moving frames {np.corrcoef(w_path[ok & moving], w_yaw[ok & moving])[0, 1]:+.2f}')

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
axes[0].plot(w_yaw[ok & moving & ~in_arc], w_path[ok & moving & ~in_arc], '.', ms=1, color='0.75', label='moving, no arc')
axes[0].plot(w_yaw[ok & in_arc], w_path[ok & in_arc], '.', ms=1, color='#2a78d6', label='in arc')
axes[0].plot([-600, 600], [-600, 600], color='0.4', lw=0.6)
axes[0].set(xlim=(-600, 600), ylim=(-600, 600), xlabel='head yaw rate (deg/s)', ylabel='path turn rate (deg/s)')
axes[0].legend(fontsize=7, markerscale=6)
axes[1].hist(arcs.yaw_r.dropna(), np.linspace(-1, 1, 21), color='0.4')
axes[1].set_xlabel('within-arc r (path vs yaw rate)')
for a in axes:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()


# %% 5. parameter sweep of the coarse scale (fine scale fixed): synthetic score + real-data statistics
grid = [dict(PC, sigma_s=sig, kappa_hi=khi, kappa_lo=khi * ratio, merge_gap_s=mg, min_turn_deg=mt)
        for sig in (0.16, 0.24, 0.32)
        for khi in (0.001, 0.0015, 0.002, 0.003)
        for ratio in (0.75,)
        for mg in (0.2, 0.4)
        for mt in (30, 45)]
rows = []
for p in grid:
    rows.append({**{k: p[k] for k in ('sigma_s', 'kappa_hi', 'kappa_lo', 'merge_gap_s', 'min_turn_deg')},
                 **synth_score(synths, P, p), **real_summary(D1, P, p)})
SWEEP = pd.DataFrame(rows)
SWEEP['ratio'] = (SWEEP.kappa_lo / SWEEP.kappa_hi).round(2)
cols = ['sigma_s', 'kappa_hi', 'kappa_lo', 'merge_gap_s', 'min_turn_deg', 'precision', 'frame_prec', 'recall', 'rec_large',
        'rad_err', 'onset_ms', 'offset_ms', 'n_arcs', 'n_coarse', 'med_dur_s', 'med_radius', 'pct_moving_in_arcs',
        'pct_yaw_sign_ok']
reliable = SWEEP[(SWEEP.precision >= PREC_MIN) & (SWEEP.frame_prec >= FRAME_PREC_MIN)]
print(f'{len(reliable)} of {len(SWEEP)} settings with precision >= {PREC_MIN} and frame precision >= {FRAME_PREC_MIN}; '
      'top 15 by recall:')
print(reliable.sort_values('recall', ascending=False)[cols].head(15).round(3).to_string(index=False))
cur = SWEEP[np.isclose(SWEEP.sigma_s, SIGMA_S_COARSE) & np.isclose(SWEEP.kappa_hi, KAPPA_HI_COARSE)
            & np.isclose(SWEEP.kappa_lo, KAPPA_LO_COARSE) & np.isclose(SWEEP.merge_gap_s, MERGE_GAP_S_COARSE)
            & np.isclose(SWEEP.min_turn_deg, MIN_TURN_DEG)]
print('current settings:')
print(cur[cols].round(3).to_string(index=False))

# heatmaps over sigma x kappa_hi at the chosen setting's ratio / merge gap / turn
best = reliable.loc[reliable.recall.idxmax()]
sub = SWEEP[(SWEEP.ratio == best.ratio) & (SWEEP.merge_gap_s == best.merge_gap_s) & (SWEEP.min_turn_deg == best.min_turn_deg)]
fig, axes = plt.subplots(1, 4, figsize=(17, 3.2))
for a, col, lbl in zip(axes, ('precision', 'frame_prec', 'recall', 'rec_large'),
                       ('synthetic precision (arcs)', 'synthetic frame precision', 'synthetic recall',
                        'synthetic recall, radius >= 100 mm')):
    tab = sub.pivot(index='sigma_s', columns='kappa_hi', values=col)
    im = a.imshow(tab.to_numpy(), aspect='auto', origin='lower', cmap='viridis')
    a.set_xticks(range(tab.shape[1]), tab.columns); a.set_yticks(range(tab.shape[0]), tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            a.text(j, i, f'{tab.to_numpy()[i, j]:.2f}', ha='center', va='center', fontsize=6, color='w')
    a.set_xlabel('KAPPA_HI_COARSE (1/mm)'); a.set_ylabel('SIGMA_S_COARSE (s)')
    a.set_title(f'{lbl}\nKAPPA_LO = {best.ratio} x HI, merge {best.merge_gap_s} s, turn >= {best.min_turn_deg:.0f} deg', fontsize=8)
    fig.colorbar(im, ax=a)
fig.tight_layout()


# %% 6. held-out sessions (incl. young / slow animals): fine only vs two-scale
# Each session gets its own synthetic data (its own speed trace and wiggle). Arcs are only
# found while speed > SPEED_MIN, so arcs_per_moving_min is the rate to compare across ages.
BEST = dict(PC, sigma_s=best.sigma_s, kappa_hi=best.kappa_hi, kappa_lo=best.kappa_lo, merge_gap_s=best.merge_gap_s,
            min_turn_deg=best.min_turn_deg)
print('sweep choice (coarse):', {k: BEST[k] for k in ('sigma_s', 'kappa_hi', 'kappa_lo', 'merge_gap_s', 'min_turn_deg')})
settings = [('fine only', None), ('two-scale', PC)]
if any(not np.isclose(BEST[k], PC[k]) for k in BEST):
    settings.append(('two-scale, sweep choice', BEST))
rows = []
for i, session in enumerate((SESSION,) + CHECK_SESSIONS):
    d = D1 if session == SESSION else load(session)
    sy = synths if session == SESSION else synth_set(d, N_SYNTH, ARC_RATE_PER_MIN, seed=i)
    moving_min = np.sum(d[3] > SPEED_MIN) / FS / 60
    for label, pc in settings:
        rs = real_summary(d, P, pc)
        rows.append(dict(session=f'{d[0].id} EO{d[0].eo}', setting=label, pct_frames_moving=100 * np.nanmean(d[3] > SPEED_MIN),
                         **synth_score(sy, P, pc), **rs, arcs_per_moving_min=rs['n_arcs'] / moving_min))
    if session == CHECK_SESSIONS[1]:
        R2, x2_raw, y2_raw, _, _ = d
        segs2, K2 = find_arcs_2scale(x2_raw, y2_raw, FS, P, PC)
        plot_arcs(x2_raw, y2_raw, segs2, K2, P, PC, f'{R2.id} EO{R2.eo}: current settings')
print(pd.DataFrame(rows)[['session', 'setting', 'pct_frames_moving', 'precision', 'frame_prec', 'recall', 'rec_large',
                          'onset_ms', 'offset_ms', 'n_arcs', 'n_coarse', 'arcs_per_moving_min', 'med_dur_s', 'med_radius',
                          'pct_yaw_sign_ok']].round(3).to_string(index=False))

plt.show()
