# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started): cd /Users/benjaminscholl/Documents/eyetools/
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import non_saccade_mask
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, FS, EO_BINS, AGE_COLORS, HEAD_COLOR,
                                       EYE_COLORS, eo_groups, head_signal, eye_signal, running_mask, clean_runs,
                                       set_style, save_fig)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False   # True: normal EO 8-20 vs. all delayed sessions

# OTHER SETTINGS
# A leaky integrator lets the eye drift back toward center: drift velocity vs eye position has slope
# -1/tau (leak). Frame: flip_eye = None keeps each eye in its own frame (x: nasal + / temporal -,
# y: up + / down -), since each eye drifts toward its own orbital center. Position 0 = neutral from
# the 3D model fit; no per-session re-centering (median clean position is printed as a check).
flip_eye = None
AXES = {"x": ("nasal", "temporal"), "y": ("up", "down")}
sacc_pad = (12, 20)      # frames excluded before eye-saccade onset / after its end (as vor_v2)
nan_pad = 6              # frames excluded on each side of untracked frames
eye_ceiling = 30         # deg/s, max 2D eye speed (either eye) in a clean frame (missed saccades)
head_ceiling = 40        # deg/s, max head angular speed in a clean frame
speed_threshold = 50     # mm/s, running_mask: stationary = not in a running bout
min_bout = 60            # frames, shortest run of frames counted as running
min_period = 60          # frames (0.5 s), shortest clean period
min_periods = 10         # minimum clean periods for a session's per-session measures
min_ecc = 1.0            # deg, periods closer to neutral than this have no defined centripetal direction
pos_edges = np.arange(-12, 13, 4)   # deg, signed position bins (drift vs position cell)
example_session = 0      # index into Results for the clean-period QC figure
qc_window = (60, 120)    # s, time window shown in the QC figure

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals")


# %% clean fixation periods
# METHODS: drift is measured in CLEAN periods: contiguous runs of >= min_period frames (0.5 s), the same
# frames for both eyes. A frame is excluded if it is
#   - inside a detected eye saccade of either eye, padded sacc_pad[0] frames before onset and
#     sacc_pad[1] frames after the end (6 Hz filter ringing, measured in vor_v2);
#   - within nan_pad frames of a frame where either eye's x, y, vx or vy is untracked (NaN);
#   - "fast": 2D eye speed of either eye >= eye_ceiling, or head angular speed >= head_ceiling (or NaN),
#     padded like a saccade. This removes small saccades below the detector threshold (after the 6 Hz
#     filter) and head movements as whole events, not just their peak frames;
#   - running (running_mask: body speed > speed_threshold for >= min_bout frames) or body speed NaN.
# CAVEAT: with head speed < head_ceiling and VOR gain ~ -0.08 (vor_v2), residual VOR is < ~3 deg/s.
CLEAN, PERIODS = {}, {}
for R in Results:
    n = len(R.LE_vx)
    bad = ~non_saccade_mask(R, pad_pre=sacc_pad[0], pad_post=sacc_pad[1])

    nan = np.zeros(n, bool)
    for eye in ("LE", "RE"):
        for key in ("x", "y", "vx", "vy"):
            nan |= ~np.isfinite(eye_signal(R, eye, key))
    for a, b in clean_runs(nan, 1):
        bad[max(0, a - nan_pad):b + nan_pad] = True

    eye_speed = np.fmax(eye_signal(R, "LE", "speed"), eye_signal(R, "RE", "speed"))
    head_speed = head_signal(R, "speed")
    fast = (eye_speed >= eye_ceiling) | ~(head_speed < head_ceiling)
    for a, b in clean_runs(fast, 1):
        bad[max(0, a - sacc_pad[0]):b + sacc_pad[1]] = True

    bad |= running_mask(R, speed_threshold, min_bout) | ~np.isfinite(np.asarray(R.speed, float))
    CLEAN[id(R)] = ~bad
    PERIODS[id(R)] = clean_runs(~bad, min_period)

rows = []
for R in Results:
    m = np.zeros(len(R.LE_vx), bool)
    for a, b in PERIODS[id(R)]:
        m[a:b] = True
    rows.append(dict(id=R.id, eo=R.eo, n_periods=len(PERIODS[id(R)]), clean_s=m.sum() / FS,
                     **{f"{eye}_{ax}_med": np.median(eye_signal(R, eye, ax, flip_eye)[m]) if m.any() else np.nan
                        for ax in AXES for eye in ("LE", "RE")}))
print(pd.DataFrame(rows).round(1).to_string(index=False))

# check: nothing fast inside a period
mx_eye = max([np.fmax(eye_signal(R, "LE", "speed"), eye_signal(R, "RE", "speed"))[a:b].max()
              for R in Results for a, b in PERIODS[id(R)]], default=np.nan)
mx_head = max([head_signal(R, "speed")[a:b].max() for R in Results for a, b in PERIODS[id(R)]], default=np.nan)
print(f"max eye speed in periods {mx_eye:.1f} deg/s, max head speed {mx_head:.1f} deg/s")

# QC figure: one session, eye position, max eye speed, head speed, clean periods shaded
R = Results[example_session]
t = np.arange(len(R.LE_vx)) / FS
w = (t >= qc_window[0]) & (t < qc_window[1])
fig, axes = plt.subplots(4, 1, figsize=(7, 5), sharex=True)
for ax, key in zip(axes[:2], AXES):
    for eye in ("LE", "RE"):
        ax.plot(t[w], eye_signal(R, eye, key, flip_eye)[w], lw=0.6, color=EYE_COLORS[eye], label=eye)
    ax.set_ylabel(f"eye {key} (deg)")
axes[0].legend(fontsize=5)
axes[2].plot(t[w], np.fmax(eye_signal(R, "LE", "speed"), eye_signal(R, "RE", "speed"))[w], lw=0.6, color="k")
axes[2].axhline(eye_ceiling, color="r", lw=0.5, ls="--")
axes[2].set_ylim(0, 4 * eye_ceiling)
axes[2].set_ylabel("max eye speed\n(deg/s)")
axes[3].plot(t[w], head_signal(R, "speed")[w], lw=0.6, color=HEAD_COLOR)
axes[3].axhline(head_ceiling, color="r", lw=0.5, ls="--")
axes[3].set_ylim(0, 4 * head_ceiling)
axes[3].set_ylabel("head speed\n(deg/s)")
axes[3].set_xlabel("time (s)")
for a, b in PERIODS[id(R)]:
    if t[a] < qc_window[1] and t[b - 1] >= qc_window[0]:
        for ax in axes:
            ax.axvspan(t[a], t[b - 1], color="0.85", lw=0, zorder=0)
axes[0].set_title(f"F{R.id} EO{R.eo}: clean periods (grey), {len(PERIODS[id(R)])} in session", fontsize=6)
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "integrator clean periods QC")
