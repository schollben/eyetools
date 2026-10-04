"""Find arcs in a tracked 2D trajectory: load -> fill gaps -> Gaussian smooth -> curvature -> arcs -> table + plot.
Edit the SETTINGS block, then run from the repo root:  python3 analyses/test_findarcs.py
"""

# %%
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from utils import load_session_data


# ============================== SETTINGS ==============================
SESSION = 'session_2026-03-14_ferret_407_P47_E14_analyzable_output'   # head position_x / position_y (mm)
FS = 120                         # frame rate (Hz)

SIGMA_S = 0.08       # Gaussian smoothing sigma (s)
MAX_GAP_S = 0.2      # interpolate tracking gaps up to this long; longer gaps stay NaN
SPEED_MIN = 50.0     # mm/s; set above the speed noise floor when the animal is still
KAPPA_MIN = 0.01     # 1/mm; |curvature| above this counts as arcing (radius < 1/KAPPA_MIN)
MIN_DUR_S = 0.3      # shortest arc kept (s)

PLOT_WINDOW_S = (0, 60)          # time range drawn in the figure (s)
# ======================================================================


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


def arc_segments(speed, kappa, fs, speed_min, kappa_min, min_dur_s):
    arcing = (speed > speed_min) & (np.abs(kappa) > kappa_min)   # NaN compares False
    d = np.diff(arcing.astype(int), prepend=0, append=0)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if (e - s) / fs >= min_dur_s]


# ---------------------------------------------------------------- run
R = load_session_data(SESSION)
vv = np.hypot(R.linearVel_x, R.linearVel_y)
bad = ~(vv < 800)                      # same head-speed mask as utils/process_session.py
x_raw = np.where(bad, np.nan, R.position_x).astype(float)
y_raw = np.where(bad, np.nan, R.position_y).astype(float)
yaw = np.rad2deg(np.unwrap(np.deg2rad(R.yaw)))   # head yaw, for the sanity column
t = np.arange(len(x_raw)) / FS
OUT_CSV = f'arcs_{R.id}_EO{R.eo}.csv'
OUT_PNG = f'arcs_{R.id}_EO{R.eo}.png'

x = smooth(x_raw, FS, SIGMA_S, MAX_GAP_S)
y = smooth(y_raw, FS, SIGMA_S, MAX_GAP_S)
speed, kappa = kinematics(x, y, FS)
segs = arc_segments(speed, kappa, FS, SPEED_MIN, KAPPA_MIN, MIN_DUR_S)

rows = []
for s, e in segs:
    k = kappa[s:e]
    heading = np.unwrap(np.arctan2(np.gradient(y[s:e]), np.gradient(x[s:e])))
    rows.append(dict(start_frame=s, end_frame=e, start_s=s / FS, end_s=e / FS, duration_s=(e - s) / FS,
                     direction='left' if np.median(k) > 0 else 'right',
                     median_radius=1 / np.median(np.abs(k)),
                     turn_deg=np.rad2deg(heading[-1] - heading[0]),
                     yaw_change_deg=yaw[e - 1] - yaw[s],
                     mean_speed=speed[s:e].mean(),
                     path_length=np.sum(np.hypot(np.diff(x[s:e]), np.diff(y[s:e])))))
arcs = pd.DataFrame(rows)
arcs.to_csv(OUT_CSV, index=False)
print(f'{len(arcs)} arcs found -> {OUT_CSV}')
print(arcs.round(2).to_string(index=False))
print(f'corr(turn_deg, yaw_change_deg) = {arcs.turn_deg.corr(arcs.yaw_change_deg):+.3f}  (negative = left/right flipped)')

# ---------------------------------------------------------------- plot
w0, w1 = int(PLOT_WINDOW_S[0] * FS), int(PLOT_WINDOW_S[1] * FS)
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.6], hspace=0.3)
ax = fig.add_subplot(gs[:, 0])
ax.plot(x_raw[w0:w1], y_raw[w0:w1], '.', ms=1, color='#b9b8b2', label='raw')
ax.plot(x[w0:w1], y[w0:w1], color='#0b0b0b', lw=0.8, label='smoothed')
for s, e in segs:
    if e > w0 and s < w1:
        ax.plot(x[s:e], y[s:e], lw=3, color='#2a78d6' if np.median(kappa[s:e]) > 0 else '#eb6834')
ax.plot([], [], lw=3, color='#2a78d6', label='left arc'); ax.plot([], [], lw=3, color='#eb6834', label='right arc')
ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.legend(loc='best', fontsize=8)
ax.set_title('Trajectory and detected arcs')

a1 = fig.add_subplot(gs[0, 1])
a2 = fig.add_subplot(gs[1, 1], sharex=a1)
a3 = fig.add_subplot(gs[2, 1], sharex=a1)
a1.plot(t, speed, color='#0b0b0b', lw=1); a1.axhline(SPEED_MIN, color='#e34948', ls='--', lw=1)
a1.set_ylabel('speed')
k_plot = np.where(speed > SPEED_MIN, np.abs(kappa), np.nan)
a2.semilogy(t, k_plot, color='#0b0b0b', lw=1); a2.axhline(KAPPA_MIN, color='#e34948', ls='--', lw=1)
a2.set_ylabel('|curvature|')
a3.plot(t, np.where(speed > SPEED_MIN, kappa, np.nan), color='#0b0b0b', lw=1); a3.axhline(0, color='#8a8984', lw=0.6)
a3.set_ylabel('curvature (+ left)'); a3.set_xlabel('time (s)')
for a in (a1, a2, a3):
    for s, e in segs:
        a.axvspan(s / FS, e / FS, color='#2a78d6' if np.median(kappa[s:e]) > 0 else '#eb6834', alpha=0.15, lw=0)
    a.spines[['top', 'right']].set_visible(False)
a1.set_xlim(PLOT_WINDOW_S)
fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
print(f'plot -> {OUT_PNG}')
plt.show()
