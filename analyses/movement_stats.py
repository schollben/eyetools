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
condition = "all" # "all", "stationary", "head_still", "stationary_and_head_still"
# "head_still" condition gates on head angular speed directly

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


# %% event rate distributions (sliding window)

condition = "all"   # "all" | "stationary" | "head_still" | "stationary_and_head_still"
win_sec, step_sec = 10.0, 5.0

bins = np.linspace(0, 4, 40)

fig, axes = plt.subplots(1, 2, figsize=(6, 2))

for ax, signal in zip(axes, ("eye", "gaze")):

    for i, (group, title) in enumerate(zip(groups, titles)):

        if not group:
            continue

        rate = pooled_rates(group, signal, condition, win_sec, step_sec,
                            speed_threshold, min_bout, head_still_thresh)
        if not len(rate):
            continue

        sns.histplot(ax=ax, x=rate, bins=bins, element="step", fill=False,
                     stat="probability", color=AGE_COLORS[i],
                     label=f"{title} (n={len(rate)})")

        print(f"{signal:5s} {title:10s} n={len(rate):6d}  "
              f"median={np.median(rate):5.2f} Hz  "
              f"IQR={np.subtract(*np.percentile(rate, [75, 25])):5.2f}  "
              f"zero={np.mean(rate == 0) * 100:4.1f}%")

    ax.set_xlabel(f"{signal} event rate (Hz, {win_sec:.0f} s window)")
    ax.legend(fontsize=6)

sns.despine(fig)
fig.tight_layout()


# %% inter-event interval distributions (timing, not magnitude)
# Interval = peak of one event -> onset of the next: the quiescent gap between movements.

condition = "all"   # "all" | "stationary" | "head_still" | "stationary_and_head_still"

bins = np.logspace(np.log10(10), np.log10(10000), 30)

fig, axes = plt.subplots(1, 2, figsize=(6, 2))

for ax, signal in zip(axes, ("eye", "gaze")):

    for i, (group, title) in enumerate(zip(groups, titles)):

        if not group:
            continue

        isi = pooled_intervals(group, signal, condition,
                               speed_threshold, min_bout, head_still_thresh)

        sns.histplot(ax=ax, x=isi, bins=bins, element="step", fill=False,
                     stat="probability", color=AGE_COLORS[i],
                     label=f"{title} (n={len(isi)})")

        print(f"{signal:5s} {title:10s} n={len(isi):6d}  "
              f"median={np.median(isi):7.0f} ms  "
              f"IQR={np.subtract(*np.percentile(isi, [75, 25])):7.0f}  ")

    ax.set_xscale("log")
    ax.set_xlabel(f"{signal} inter-event interval (ms)")
    ax.legend(fontsize=6)

sns.despine(fig)
fig.tight_layout()

