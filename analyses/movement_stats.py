# %% init
%load_ext autoreload
%autoreload 2
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
from utils import create_subplot_grid, load_session_data, process_session, removeBadData, getSesh
from utils import create_subplot_grid
import numpy as np
from scipy.stats import mannwhitneyu as mwu
from utils.config import SAVELOC
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import (EYE_COLOR, AGE_COLORS, FS, eo_groups, pooled_events, pooled_intervals,
                                       event_traces, pooled_rates)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 6
plt.rcParams['svg.fonttype'] = 'none'

# LOAD DATA
SESSION = getSesh.by_ferret(402, 405, 407, 420)
Results = []
for session in SESSION:
    R = load_session_data(session)
    removeBadData(R)
    process_session(R, window_in_sec=5,
                    velocity_threshold_eye=40, velocity_threshold_gaze=40,
                    velocity_threshold_head=1, min_duration=8, min_inter_event=8)
    Results.append(R)
n_sesh = len(Results)
print(n_sesh, "sessions loaded")

# settings for every plot below

# TODO — HEAD SACCADES (revisit later)
# The "head_still" condition gates on head angular speed directly
# (np.rad2deg(R.angVelocities) < head_still_thresh), NOT on R.df_head, because df_head is
# not currently a usable head-saccade table: 
#   process_session(velocity_threshold_head=2) feeds extract_saccades(R, 'skull', ...),
#    which thresholds sqrt(yaw_v^2 + pitch_v^2). Those fields are rad/s in the csv and the
#    loaders do not convert them, so the threshold acts as 2 rad/s ~= 115 deg/s.
# To use real head saccades later, re-extract with a threshold in rad/s that corresponds to
# the deg/s value intended (e.g. 50 deg/s -> velocity_threshold_head=0.87), or convert the
# skull _v fields before extraction. Then this condition can gate on df_head event windows
# the way non_saccade_mask gates on df_LE/df_RE.

# NOTE: gaze angular velocity fields ARE already rad2deg-converted at load, unlike the
# skull _v fields. Do NOT pass them through head_signal() — that would double-convert.


# how panels are split: False = one panel per session, True = one panel per EO range
pool_by_eo = True
eo_bins = [(0, 3), (4, 7), (8, 20)]

flip_eye = "RE"     
amp_bins = [0, 4, 8, 12, 20, 45]

speed_threshold = 50    # mm/s, locomotion
min_bout = 30            # frames
head_still_thresh = 10   # deg/s

conditions = ["all", "stationary", "head_still", "stationary_and_head_still"]
COND_COLORS = dict(zip(conditions, ["#444444", "#725EE7", "#E93115", "#1B9E77"]))

pre, post = 12, 48       # frames: -100 to +400 ms from onset (cell 4 only)
bin_by = "amplitude"     # "amplitude" | "peak_velocity" (cell 4 only)

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)


for R in Results:
    counts = [len(pooled_events([R], "eye", c, "amplitude_deg",
                                speed_threshold, min_bout, head_still_thresh))
              for c in conditions]


# %% 3. amplitude vs peak velocity by condition

signal = "eye"

fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    for cond in conditions:
        amp = pooled_events(group, signal, cond, "amplitude_deg",
                            speed_threshold, min_bout, head_still_thresh)
        pkv = pooled_events(group, signal, cond, "peak_velocity_deg_s",
                            speed_threshold, min_bout, head_still_thresh)
        if len(amp) < 20:
            continue

        x = np.log10(np.abs(amp))
        y = np.log10(np.abs(pkv))
        ok = np.isfinite(x) & np.isfinite(y)

        sns.scatterplot(ax=ax, x=x[ok], y=y[ok], s=2, alpha=0.15,
                        color=COND_COLORS[cond])
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        xl = np.array([x[ok].min(), x[ok].max()])
        ax.plot(xl, slope * xl + intercept, color=COND_COLORS[cond], lw=1,
                label=f"{cond} ({slope:.2f})")

    ax.set_title(title)
    ax.set_xlabel("log10 amplitude (deg)")
    ax.set_ylabel("log10 peak velocity (deg/s)")
    ax.legend(fontsize=4)

# %% 4. mean kinematic traces
# Onset-aligned, so displacement traces start at zero by construction.

signal = "eye"      # "eye" | "gaze"
condition = "all" #"head_still"

t_ms = np.arange(-pre, post) / FS * 1000
bin_col = "amplitude_deg" if bin_by == "amplitude" else "peak_velocity_deg_s"
bin_edges = amp_bins if bin_by == "amplitude" else [0, 50, 100, 200, 400, 1000]

for kind in ("speed", "displacement"):

    fig, axes = create_subplot_grid(len(groups))

    for ax, group, title in zip(axes, groups, titles):

        if not group:
            ax.set_title(title)
            continue

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):

            traces, nominal = [], []
            for R in group:
                t, a = event_traces(R, signal, kind, lo, hi, bin_col, condition,
                                    pre, post, flip_eye,
                                    speed_threshold, min_bout, head_still_thresh)
                traces += t
                nominal += a

            arr = np.array(traces)
            m = arr.mean(axis=0)
            se = arr.std(axis=0) / np.sqrt(len(arr))
            line, = ax.plot(t_ms, m, lw=1, label=f"{lo}-{hi} (n={len(arr)})")
            ax.fill_between(t_ms, m - se, m + se, alpha=0.25)

        ax.set_title(f"{signal} {title}")
        ax.set_xlabel("time from onset (ms)")
        ax.set_ylabel("speed (deg/s)" if kind == "speed" else "displacement (deg)")
        ax.legend(fontsize=4)


# %% 5. inter-event interval distributions (timing, not magnitude)
# Interval = peak of one event -> onset of the next: the quiescent gap between movements.

# Two features of this distribution are extraction artifacts, not biology:
# 1. HARD FLOOR. process_session(min_inter_event=12) drops any event starting within 12
#    frames of the previous peak, so no interval below 100 ms can exist and 3-5% sit
#    exactly on it. The dotted line marks it — it is not a real mode.
# 2. NaN GAPS. pooled_intervals requires the gap to be NaN-free. Gaps spanning a tracking
#    dropout are 4.5% of pairs with median 1425 ms vs 275 ms, the largest 119.8 s (a
#    recording gap, not a fixation). Keeping them would dominate the tail.
# Bins are log-spaced: intervals span 100 ms to ~8 s, so linear bins collapse the
# distribution into the leftmost few.

condition = "head_still"   # "all" | "stationary" | "head_still" | "stationary_and_head_still"

floor_ms = 1000 * Results[0].min_inter_event / FS
bins = np.logspace(np.log10(floor_ms), np.log10(10000), 40)

fig, axes = plt.subplots(1, 2, figsize=(6, 2))

for ax, signal in zip(axes, ("eye", "gaze")):

    for group, title in zip(groups, titles):

        if not group:
            continue

        isi = pooled_intervals(group, signal, condition,
                               speed_threshold, min_bout, head_still_thresh)

        sns.histplot(ax=ax, x=isi, bins=bins, element="step", fill=False,
                     stat="probability", label=f"{title} (n={len(isi)})")

        print(f"{signal:5s} {title:10s} n={len(isi):6d}  "
              f"median={np.median(isi):7.0f} ms  "
              f"IQR={np.subtract(*np.percentile(isi, [75, 25])):7.0f}  "
              f"at floor={np.mean(isi <= floor_ms + 0.5) * 100:4.1f}%")

    ax.set_xscale("log")
    ax.set_xlabel(f"{signal} inter-event interval (ms)")
    ax.legend(fontsize=4)

sns.despine(fig)
fig.tight_layout()


# %% 6. event rate distributions (sliding window)
# Cell 5 asks how long the gaps between movements are; this asks how many movements fall
# in a fixed stretch of time. A win_sec window stepped by step_sec gives one rate per
# window, so the spread shows whether a session alternates bursts and quiet periods or
# holds a steady rate — which a single mean events/s hides.
#
# Windows containing any NaN in the position trace are dropped, for the same reason cell 5
# drops NaN-spanning gaps: a tracking dropout would otherwise be counted as a window with
# no events. win_sec sets the trade-off — short windows resolve bursts but quantize rate
# coarsely (a 5 s window can only report multiples of 0.2 Hz), long windows smooth them away.

condition = "head_still"   # "all" | "stationary" | "head_still" | "stationary_and_head_still"
win_sec, step_sec = 10.0, 5.0

bins = np.linspace(0, 4, 40)

fig, axes = plt.subplots(1, 2, figsize=(6, 2))

for ax, signal in zip(axes, ("eye", "gaze")):

    for group, title in zip(groups, titles):

        if not group:
            continue

        rate = pooled_rates(group, signal, condition, win_sec, step_sec,
                            speed_threshold, min_bout, head_still_thresh)
        if not len(rate):
            continue

        sns.histplot(ax=ax, x=rate, bins=bins, element="step", fill=False,
                     stat="probability", label=f"{title} (n={len(rate)})")

        print(f"{signal:5s} {title:10s} n={len(rate):6d}  "
              f"median={np.median(rate):5.2f} Hz  "
              f"IQR={np.subtract(*np.percentile(rate, [75, 25])):5.2f}  "
              f"zero={np.mean(rate == 0) * 100:4.1f}%")

    ax.set_xlabel(f"{signal} event rate (Hz, {win_sec:.0f} s window)")
    ax.legend(fontsize=4)

sns.despine(fig)
fig.tight_layout()
