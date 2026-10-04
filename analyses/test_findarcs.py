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

SIGMA_S = 0.08       # Gaussian smoothing sigma (s)
MAX_GAP_S = 0.2      # interpolate tracking gaps up to this long; longer gaps stay NaN
SPEED_MIN = 50.0     # mm/s; set above the speed noise floor when the animal is still
KAPPA_HI = 0.015     # 1/mm; an arc must reach |curvature| above this (radius < 1/KAPPA_HI) ...
KAPPA_LO = 0.01125   # 1/mm; ... and extends while |curvature| stays above this (0.75 x KAPPA_HI)
MERGE_GAP_S = 0.0    # same-direction arcs closer than this are joined
MIN_DUR_S = 0.3      # shortest arc kept (s), applied after merging

PLOT_WINDOW_S = (0, 60)          # time range drawn in the figures (s)

# synthetic ground truth (cells 2, 3, 5)
NOISE_SIGMA_S = 0.1              # real wiggle + jitter = raw - Gaussian(raw, this) is added to the synthetic path
ARC_RATE_PER_MIN = 40            # planted arcs per minute of moving time (40 / 300 mm: closest to real |curvature| and turn rate)
RADIUS_RANGE = (20, 300)         # mm, planted radius (log-uniform)
TURN_RANGE_DEG = (45, 180)       # planted turn angle (uniform)
TRUTH_MIN_DUR_S = 0.3            # planted arcs are at least this long
BG_RADIUS = 300                  # mm; background heading wander mostly has radius > this (= not an arc)
BG_SMOOTH_S = 1.0                # s; time scale of the background wander
N_SYNTH = 5                      # synthetic realizations per score

# conservative selection (cell 5): among settings whose detections are this reliable, take the highest recall
PREC_MIN = 0.95                  # fraction of detected arcs that are real
FRAME_PREC_MIN = 0.85            # fraction of detected arc frames that lie inside a real arc
# ======================================================================

P = dict(sigma_s=SIGMA_S, max_gap_s=MAX_GAP_S, speed_min=SPEED_MIN, kappa_hi=KAPPA_HI,
         kappa_lo=KAPPA_LO, merge_gap_s=MERGE_GAP_S, min_dur_s=MIN_DUR_S)
LEFT_C, RIGHT_C = '#2a78d6', '#eb6834'


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


def arc_segments(speed, kappa, fs, speed_min, kappa_hi, kappa_lo, merge_gap_s, min_dur_s):
    """(start, end, sign) per arc. Left (+1) and right (-1) are found separately: runs above
    kappa_lo that reach kappa_hi, joined across gaps <= merge_gap_s, then >= min_dur_s."""
    moving = speed > speed_min                                    # NaN compares False
    segs = []
    for sign in (1, -1):
        k = sign * kappa
        hi = moving & (k > kappa_hi)
        kept = []
        for s, e in zip(*runs(moving & (k > kappa_lo))):
            if not hi[s:e].any():
                continue
            if kept and s - kept[-1][1] <= merge_gap_s * fs:
                kept[-1] = (kept[-1][0], e)
            else:
                kept.append((s, e))
        segs += [(s, e, sign) for s, e in kept if (e - s) / fs >= min_dur_s]
    return sorted(segs)


def find_arcs(x_raw, y_raw, fs, sigma_s, max_gap_s, speed_min, kappa_hi, kappa_lo, merge_gap_s, min_dur_s):
    """Raw 2D position -> arcs. Returns segs [(start, end, sign)], smoothed x, y, speed, kappa."""
    x = smooth(x_raw, fs, sigma_s, max_gap_s)
    y = smooth(y_raw, fs, sigma_s, max_gap_s)
    speed, kappa = kinematics(x, y, fs)
    segs = arc_segments(speed, kappa, fs, speed_min, kappa_hi, kappa_lo, merge_gap_s, min_dur_s)
    return segs, x, y, speed, kappa


def turn_deg(x, y, s, e):
    """Heading change of the path over frames s:e (deg, + = left)."""
    heading = np.unwrap(np.arctan2(np.gradient(y[s:e]), np.gradient(x[s:e])))
    return np.rad2deg(heading[-1] - heading[0])


def load(session):
    """Head position (NaN where head speed >= 800, as in utils/process_session.py), speed, yaw rate."""
    R = load_session_data(session)
    vv = np.hypot(R.linearVel_x, R.linearVel_y)
    bad = ~(vv < 800)
    x_raw = np.where(bad, np.nan, R.position_x).astype(float)
    y_raw = np.where(bad, np.nan, R.position_y).astype(float)
    w_yaw = np.gradient(unwrap_deg(np.where(bad, np.nan, R.yaw))) * FS   # world head yaw rate (deg/s, + = left)
    return R, x_raw, y_raw, np.where(bad, np.nan, vv), w_yaw


def arc_table(segs, x, y, speed, kappa, w_yaw, fs):
    """One row per arc. yaw_sign_ok / yaw_r compare path turn rate (speed * kappa) with head yaw rate."""
    w_path = np.rad2deg(speed * kappa)
    rows = []
    for s, e, sign in segs:
        rows.append(dict(start_frame=s, end_frame=e, start_s=s / fs, end_s=e / fs, duration_s=(e - s) / fs,
                         direction='left' if sign > 0 else 'right',
                         median_radius=1 / np.median(np.abs(kappa[s:e])),
                         turn_deg=turn_deg(x, y, s, e),
                         mean_speed=speed[s:e].mean(),
                         path_length=np.sum(np.hypot(np.diff(x[s:e]), np.diff(y[s:e]))),
                         yaw_sign_ok=np.sign(np.nanmean(w_yaw[s:e])) == sign,
                         yaw_r=pd.Series(w_path[s:e]).corr(pd.Series(w_yaw[s:e]))))
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
    for _ in range(200 * n_arcs):
        if len(truth) == n_arcs:
            break
        s = rng.integers(n)
        r = np.exp(rng.uniform(*np.log(RADIUS_RANGE)))
        turn = rng.uniform(*TURN_RANGE_DEG)
        e = np.searchsorted(dist, dist[s] + r * np.deg2rad(turn))
        if (e >= n or (e - s) / fs < TRUTH_MIN_DUR_S or not (moving[s] and moving[e - 1])
                or moving[s:e].mean() < 0.8 or taken[max(0, s - gap):e + gap].any()):
            continue
        sign = rng.choice([1, -1])
        kappa[s:e] = sign / r
        taken[s:e] = True
        truth.append((s, e, sign, r, turn))
    heading = np.cumsum(v * kappa) / fs
    x = np.cumsum(v * np.cos(heading)) / fs + np.nan_to_num(resid_x)
    y = np.cumsum(v * np.sin(heading)) / fs + np.nan_to_num(resid_y)
    x[nan_mask], y[nan_mask] = np.nan, np.nan
    return x, y, sorted(truth)


def score(segs, truth, x, y, kappa, fs):
    """Truth arc found = >= 50% of its frames inside same-direction detections.
    Detection correct = >= 50% of its frames inside same-direction truth.
    Errors are from the detection overlapping each found truth arc most."""
    n = len(kappa)
    det = {1: np.zeros(n, bool), -1: np.zeros(n, bool)}
    tru = {1: np.zeros(n, bool), -1: np.zeros(n, bool)}
    for s, e, sign in segs:
        det[sign][s:e] = True
    for s, e, sign, *_ in truth:
        tru[sign][s:e] = True
    found, rad_err, turn_err, on_err, off_err = [], [], [], [], []
    for ts, te, tsign, tr, tturn in truth:
        found.append(det[tsign][ts:te].mean() >= 0.5)
        if not found[-1]:
            continue
        s, e, _ = max((d for d in segs if d[2] == tsign), key=lambda d: min(d[1], te) - max(d[0], ts))
        rad_err.append(abs(np.log(1 / np.median(np.abs(kappa[s:e])) / tr)))
        turn_err.append(abs(turn_deg(x, y, s, e)) - tturn)
        on_err.append((s - ts) / fs * 1000)
        off_err.append((e - te) / fs * 1000)
    correct = [tru[sign][s:e].mean() >= 0.5 for s, e, sign in segs]
    recall = np.mean(found) if found else np.nan
    precision = np.mean(correct) if correct else np.nan
    n_det_frames = det[1].sum() + det[-1].sum()
    return dict(n_truth=len(truth), n_det=len(segs), recall=recall, precision=precision,
                frame_prec=((det[1] & tru[1]).sum() + (det[-1] & tru[-1]).sum()) / n_det_frames if n_det_frames else np.nan,
                f1=2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0,
                rad_err=np.median(rad_err) if rad_err else np.nan,       # median |log(R_det / R_true)|
                turn_err=np.median(turn_err) if turn_err else np.nan,    # median (|det turn| - true turn), deg
                onset_ms=np.median(on_err) if on_err else np.nan,
                offset_ms=np.median(off_err) if off_err else np.nan)


def synth_set(session_data, n_synth, arc_rate_per_min, seed=0):
    """n_synth synthetic realizations built from one real session."""
    R, x_raw, y_raw, vv, _ = session_data
    resid_x = x_raw - smooth(x_raw, FS, NOISE_SIGMA_S, MAX_GAP_S)
    resid_y = y_raw - smooth(y_raw, FS, NOISE_SIGMA_S, MAX_GAP_S)
    nan_mask = np.isnan(x_raw) | np.isnan(y_raw)
    rng = np.random.default_rng(seed)
    return [make_synth(vv, resid_x, resid_y, nan_mask, FS, rng, arc_rate_per_min) for _ in range(n_synth)]


def synth_score(synths, p):
    """Mean score over synthetic realizations for parameter dict p."""
    out = []
    for xs, ys, truth in synths:
        segs, x, y, _, kappa = find_arcs(xs, ys, FS, **p)
        out.append(score(segs, truth, x, y, kappa, FS))
    return pd.DataFrame(out).mean()


def real_summary(session_data, p):
    """Real-data statistics for parameter dict p (no ground truth: counts, sizes, yaw agreement)."""
    R, x_raw, y_raw, _, w_yaw = session_data
    segs, x, y, speed, kappa = find_arcs(x_raw, y_raw, FS, **p)
    arcs = arc_table(segs, x, y, speed, kappa, w_yaw, FS)
    moving = speed > p['speed_min']
    in_arc = np.zeros(len(speed), bool)
    for s, e, _ in segs:
        in_arc[s:e] = True
    return dict(n_arcs=len(arcs),
                med_dur_s=arcs.duration_s.median() if len(arcs) else np.nan,
                med_radius=arcs.median_radius.median() if len(arcs) else np.nan,
                pct_moving_in_arcs=100 * (in_arc & moving).sum() / moving.sum(),
                pct_yaw_sign_ok=100 * arcs.yaw_sign_ok.mean() if len(arcs) else np.nan,
                med_yaw_r=arcs.yaw_r.median() if len(arcs) else np.nan)


def plot_arcs(x_raw, y_raw, x, y, speed, kappa, segs, p, title, truth=None):
    """Trajectory with arcs (left) and speed / |kappa| / kappa traces (right), PLOT_WINDOW_S only.
    truth (synthetic): thin colored lines on the trajectory, hatched spans on the traces."""
    t = np.arange(len(x)) / FS
    w0, w1 = int(PLOT_WINDOW_S[0] * FS), int(PLOT_WINDOW_S[1] * FS)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.6], hspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(x_raw[w0:w1], y_raw[w0:w1], '.', ms=1, color='#b9b8b2', label='raw')
    ax.plot(x[w0:w1], y[w0:w1], color='#0b0b0b', lw=0.8, label='smoothed')
    for s, e, sign in segs:
        if e > w0 and s < w1:
            ax.plot(x[s:e], y[s:e], lw=4, alpha=0.6, color=LEFT_C if sign > 0 else RIGHT_C)
    for s, e, sign, *_ in truth or []:
        if e > w0 and s < w1:
            ax.plot(x[s:e], y[s:e], lw=1.2, color=LEFT_C if sign > 0 else RIGHT_C)
    ax.plot([], [], lw=4, color=LEFT_C, label='left arc'); ax.plot([], [], lw=4, color=RIGHT_C, label='right arc')
    if truth is not None:
        ax.plot([], [], lw=1.2, color='0.3', label='truth (thin)')
    ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.legend(loc='best', fontsize=8)
    ax.set_title(title)

    a1 = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    a1.plot(t, speed, color='#0b0b0b', lw=1); a1.axhline(p['speed_min'], color='#e34948', ls='--', lw=1)
    a1.set_ylabel('speed (mm/s)')
    k_plot = np.where(speed > p['speed_min'], np.abs(kappa), np.nan)
    a2.semilogy(t, k_plot, color='#0b0b0b', lw=1)
    a2.axhline(p['kappa_hi'], color='#e34948', ls='--', lw=1); a2.axhline(p['kappa_lo'], color='#e34948', ls=':', lw=1)
    a2.set_ylabel('|curvature| (1/mm)')
    a3.plot(t, np.where(speed > p['speed_min'], kappa, np.nan), color='#0b0b0b', lw=1); a3.axhline(0, color='#8a8984', lw=0.6)
    a3.set_ylabel('curvature (+ left)'); a3.set_xlabel('time (s)')
    for a in (a1, a2, a3):
        for s, e, sign in segs:
            a.axvspan(s / FS, e / FS, color=LEFT_C if sign > 0 else RIGHT_C, alpha=0.15, lw=0)
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

segs, x, y, speed, kappa = find_arcs(x_raw, y_raw, FS, **P)
arcs = arc_table(segs, x, y, speed, kappa, w_yaw, FS)
arcs.to_csv(OUT_CSV, index=False)
print(f'{len(arcs)} arcs found -> {OUT_CSV}')
print(arcs.round(2).to_string(index=False))

fig = plot_arcs(x_raw, y_raw, x, y, speed, kappa, segs, P, f'{R.id} EO{R.eo}: trajectory and detected arcs')
fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
print(f'plot -> {OUT_PNG}')


# %% 2. noise floors
# SPEED_MIN: the speed histogram has no separate stationary peak, so read off how much of
# the slow tail SPEED_MIN cuts.
# KAPPA_LO / KAPPA_HI: |curvature| of a synthetic path with NO planted arcs (real speed, real
# wiggle, background wander only) is the floor that any threshold has to clear.
log_v = np.log10(vv[vv > 1])
print(f'{100 * np.mean(vv[np.isfinite(vv)] < SPEED_MIN):.0f}% of frames below SPEED_MIN = {SPEED_MIN} mm/s; '
      'speed p10/p25/p50 = ' + ' / '.join(f'{v:.0f}' for v in np.nanpercentile(vv, [10, 25, 50])))

straight = synth_set(D1, N_SYNTH, arc_rate_per_min=0)
k_floor = {}
for sig in (0.04, 0.08, 0.12, 0.16):
    ks = []
    for xs, ys, _ in straight:
        _, _, _, sp, k = find_arcs(xs, ys, FS, **{**P, 'sigma_s': sig})
        ks.append(np.abs(k[sp > SPEED_MIN]))
    k_floor[sig] = np.percentile(np.concatenate(ks), [50, 90, 95, 99])
    print(f'sigma {sig:.2f} s: |kappa| on arc-free synthetic, p50/p90/p95/p99 = '
          + ' / '.join(f'{v:.4f}' for v in k_floor[sig]) + '  1/mm')

fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].hist(log_v, 60, color='0.4')
axes[0].axvline(np.log10(SPEED_MIN), color='#e34948', ls='--', lw=1, label='SPEED_MIN')
axes[0].set_xlabel('log10 speed (mm/s)'); axes[0].legend(fontsize=7)
_, _, _, sp, k = find_arcs(x_raw, y_raw, FS, **P)
axes[1].hist(np.log10(np.abs(k[sp > SPEED_MIN])), 80, histtype='step', density=True, color='k', label='real')
axes[1].hist(np.log10(np.concatenate(ks)), 80, histtype='step', density=True, color='0.6', label='arc-free synthetic (sigma 0.16)')
_, _, _, sp, k = find_arcs(*straight[0][:2], FS, **P)
axes[1].hist(np.log10(np.abs(k[sp > SPEED_MIN])), 80, histtype='step', density=True, color='#2a78d6',
             label=f'arc-free synthetic (sigma {SIGMA_S})')
for v, ls in ((KAPPA_HI, '--'), (KAPPA_LO, ':')):
    axes[1].axvline(np.log10(v), color='#e34948', ls=ls, lw=1)
axes[1].set_xlabel('log10 |curvature| (1/mm), moving'); axes[1].legend(fontsize=6)
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
segs_q, xq, yq, spq, kq = find_arcs(xs, ys, FS, **P)
plot_arcs(xs, ys, xq, yq, spq, kq, segs_q, P, 'synthetic: detected (thick) vs truth (thin)', truth)

print('current settings, mean over realizations:')
print(synth_score(synths, P).round(3).to_string())


# %% 4. yaw-rate check on real data
# Path turn rate (speed * curvature) should follow head yaw rate while the animal turns.
# Per arc: does mean yaw rate have the arc's sign, and how well do the two rates correlate?
w_path = np.rad2deg(speed * kappa)
moving = speed > SPEED_MIN
in_arc = np.zeros(len(speed), bool)
for s, e, _ in segs:
    in_arc[s:e] = True
ok = np.isfinite(w_path) & np.isfinite(w_yaw)
print(f'yaw sign agrees in {100 * arcs.yaw_sign_ok.mean():.0f}% of {len(arcs)} arcs; '
      f'median within-arc r = {arcs.yaw_r.median():+.2f}')
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


# %% 5. parameter sweep: synthetic score + real-data statistics per setting
grid = [dict(P, sigma_s=sig, kappa_hi=khi, kappa_lo=khi * ratio, merge_gap_s=mg)
        for sig in (0.08, 0.12, 0.16)
        for khi in (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)
        for ratio in (0.5, 0.75, 1.0)
        for mg in (0.0, 0.1)]
rows = []
for p in grid:
    rows.append({**{k: p[k] for k in ('sigma_s', 'kappa_hi', 'kappa_lo', 'merge_gap_s')},
                 **synth_score(synths, p), **real_summary(D1, p)})
SWEEP = pd.DataFrame(rows)
SWEEP['ratio'] = (SWEEP.kappa_lo / SWEEP.kappa_hi).round(2)
cols = ['sigma_s', 'kappa_hi', 'kappa_lo', 'merge_gap_s', 'precision', 'frame_prec', 'recall', 'rad_err', 'turn_err',
        'onset_ms', 'offset_ms', 'n_arcs', 'med_dur_s', 'med_radius', 'pct_moving_in_arcs', 'pct_yaw_sign_ok']
reliable = SWEEP[(SWEEP.precision >= PREC_MIN) & (SWEEP.frame_prec >= FRAME_PREC_MIN)]
print(f'{len(reliable)} of {len(SWEEP)} settings with precision >= {PREC_MIN} and frame precision >= {FRAME_PREC_MIN}; '
      'top 15 by recall:')
print(reliable.sort_values('recall', ascending=False)[cols].head(15).round(3).to_string(index=False))
cur = SWEEP[np.isclose(SWEEP.sigma_s, SIGMA_S) & np.isclose(SWEEP.kappa_hi, KAPPA_HI)
            & np.isclose(SWEEP.kappa_lo, KAPPA_LO) & np.isclose(SWEEP.merge_gap_s, MERGE_GAP_S)]
print('current settings:')
print(cur[cols].round(3).to_string(index=False))

# heatmaps over sigma x kappa_hi at the chosen setting's ratio / merge gap
best = reliable.loc[reliable.recall.idxmax()]
sub = SWEEP[(SWEEP.ratio == best.ratio) & (SWEEP.merge_gap_s == best.merge_gap_s)]
fig, axes = plt.subplots(1, 4, figsize=(17, 3.2))
for a, col, lbl in zip(axes, ('precision', 'frame_prec', 'recall', 'pct_yaw_sign_ok'),
                       ('synthetic precision (arcs)', 'synthetic frame precision', 'synthetic recall',
                        'real: % arcs with yaw sign agreeing')):
    tab = sub.pivot(index='sigma_s', columns='kappa_hi', values=col)
    im = a.imshow(tab.to_numpy(), aspect='auto', origin='lower', cmap='viridis')
    a.set_xticks(range(tab.shape[1]), tab.columns); a.set_yticks(range(tab.shape[0]), tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            a.text(j, i, f'{tab.to_numpy()[i, j]:.2f}', ha='center', va='center', fontsize=6, color='w')
    a.set_xlabel('KAPPA_HI (1/mm)'); a.set_ylabel('SIGMA_S (s)')
    a.set_title(f'{lbl}\nKAPPA_LO = {best.ratio} x HI, merge {best.merge_gap_s} s', fontsize=8)
    fig.colorbar(im, ax=a)
fig.tight_layout()


# %% 6. held-out sessions (incl. young / slow animals): current settings vs sweep choice
# Each session gets its own synthetic data (its own speed trace and wiggle). Arcs are only
# found while speed > SPEED_MIN, so arcs_per_moving_min is the rate to compare across ages.
BEST = dict(P, sigma_s=best.sigma_s, kappa_hi=best.kappa_hi, kappa_lo=best.kappa_lo, merge_gap_s=best.merge_gap_s)
print('sweep choice:', {k: BEST[k] for k in ('sigma_s', 'kappa_hi', 'kappa_lo', 'merge_gap_s')})
rows = []
for i, session in enumerate((SESSION,) + CHECK_SESSIONS):
    d = D1 if session == SESSION else load(session)
    sy = synths if session == SESSION else synth_set(d, N_SYNTH, ARC_RATE_PER_MIN, seed=i)
    moving_min = np.sum(d[3] > SPEED_MIN) / FS / 60
    for label, p in (('current', P), ('sweep choice', BEST)):
        rs = real_summary(d, p)
        rows.append(dict(session=f'{d[0].id} EO{d[0].eo}', setting=label, pct_frames_moving=100 * np.nanmean(d[3] > SPEED_MIN),
                         **synth_score(sy, p), **rs, arcs_per_moving_min=rs['n_arcs'] / moving_min))
    if session == CHECK_SESSIONS[1]:
        R2, x2_raw, y2_raw, _, _ = d
        segs2, x2, y2, sp2, k2 = find_arcs(x2_raw, y2_raw, FS, **P)
        plot_arcs(x2_raw, y2_raw, x2, y2, sp2, k2, segs2, P, f'{R2.id} EO{R2.eo}: current settings')
print(pd.DataFrame(rows)[['session', 'setting', 'pct_frames_moving', 'precision', 'frame_prec', 'recall', 'onset_ms',
                          'offset_ms', 'n_arcs', 'arcs_per_moving_min', 'med_dur_s', 'med_radius',
                          'pct_yaw_sign_ok']].round(3).to_string(index=False))

plt.show()
