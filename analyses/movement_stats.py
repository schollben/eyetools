# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
# tools
from utils import create_subplot_grid, load_session_data, process_session, removeBadData, getSesh
import numpy as np
# plotting setup
from utils.config import SAVELOC
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 6
plt.rcParams['svg.fonttype'] = 'none'

# LOAD DATA

SESSION = getSesh.by_ferret(402, 420)     # multiple — preserves order by ferret
#SESSION = getSesh.by_ferret(753)         # or load sessions from an individual ID
#SESSION = getSesh.by_eo(10,20)           # or load sessions by an EO range (inclusive)

Results = []
for session in SESSION:

    R = load_session_data(session)
    removeBadData(R)
    process_session(R, window_in_sec=5,
                    velocity_threshold_eye=40, velocity_threshold_gaze=2,
                    velocity_threshold_head=2, min_duration=12, min_inter_event=12)
    Results.append(R)

n_sesh = len(Results)
print(n_sesh, "sessions loaded")


# %% settings for every plot below
# Quantifies degree and frequency of eye (eye-in-head) and gaze (eye+head) movements.
# Cells 1-3 use only the per-event scalars already in the dataframes (amplitude_deg,
# peak_velocity_deg_s) — no window, no alignment. Cell 4 is the only one that averages
# kinematic traces, so it is the only one that needs a window.

# TODO — HEAD SACCADES (revisit later)
# The "head_still" condition gates on head angular speed directly
# (np.rad2deg(R.angVelocities) < head_still_thresh), NOT on R.df_head, because df_head is
# not currently a usable head-saccade table:
#
# 1. process_session(velocity_threshold_head=2) feeds extract_saccades(R, 'skull', ...),
#    which thresholds sqrt(yaw_v^2 + pitch_v^2). Those fields are rad/s in the csv and the
#    loaders do not convert them, so the threshold acts as 2 rad/s ~= 115 deg/s.
# 2. Measured on ferret 402 EO5: df_head holds only 48 events, median duration 1017 ms,
#    amplitudes 58-230 deg, covering 13% of frames. Those are long large movements, not
#    discrete head saccades.
# 3. So gating on df_head would exclude almost nothing, and what it did exclude would be
#    the wrong events.
#
# To use real head saccades later, re-extract with a threshold in rad/s that corresponds to
# the deg/s value intended (e.g. 50 deg/s -> velocity_threshold_head=0.87), or convert the
# skull _v fields before extraction. Then this condition can gate on df_head event windows
# the way non_saccade_mask gates on df_LE/df_RE.

# NOTE: gaze angular velocity fields ARE already rad2deg-converted at load, unlike the
# skull _v fields. Do NOT pass them through head_signal() — that would double-convert.

from analyses.helper_functions import (FS, eo_groups, pooled_events, event_traces)

# how panels are split: False = one panel per session, True = one panel per EO range
pool_by_eo = True
eo_bins = [(0, 4), (5, 9), (10, 20)]  # early / middle / late, inclusive

flip_eye = "RE"                   # conjugate frame: these are movement magnitudes
amp_bins = [0, 4, 8, 12, 20, 45]  # deg, fixed edges

speed_threshold = 100    # mm/s, locomotion
min_bout = 30            # frames
head_still_thresh = 50   # deg/s

conditions = ["all", "stationary", "head_still", "stationary_and_head_still"]
COND_COLORS = dict(zip(conditions, ["#444444", "#725EE7", "#E93115", "#1B9E77"]))

pre, post = 12, 48       # frames: -100 to +400 ms from onset (cell 4 only)
bin_by = "amplitude"     # "amplitude" | "peak_velocity" (cell 4 only)

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)


for R in Results:
    counts = [len(pooled_events([R], "eye", c, "amplitude_deg",
                                speed_threshold, min_bout, head_still_thresh))
              for c in conditions]
    print(f"ferret {R.id} EO{R.eo:<3d} eye events by condition  " +
          "  ".join(f"{c}={n}" for c, n in zip(conditions, counts)))


# %% 1. amplitude and peak velocity distributions

fig, axes = plt.subplots(1, 2, figsize=(6, 2))

for signal, ls in (("eye", "-"), ("gaze", "--")):
    for group, title in zip(groups, titles):

        if not group:
            continue

        amp = pooled_events(group, signal, "all", "amplitude_deg",
                            speed_threshold, min_bout, head_still_thresh)
        pkv = pooled_events(group, signal, "all", "peak_velocity_deg_s",
                            speed_threshold, min_bout, head_still_thresh)
        if len(amp) < 20:
            continue

        sns.histplot(ax=axes[0], x=amp, bins=40, element="step", fill=False,
                     stat="density", linestyle=ls, label=f"{signal} {title}")
        sns.histplot(ax=axes[1], x=pkv, bins=40, element="step", fill=False,
                     stat="density", linestyle=ls, label=f"{signal} {title}")

        print(f"{signal:5s} {title:10s} n={len(amp):6d}  "
              f"amp median={np.median(amp):6.2f} IQR={np.subtract(*np.percentile(amp, [75, 25])):6.2f}  "
              f"pkv median={np.median(pkv):7.1f}")

axes[0].set_xlabel("amplitude (deg)")
axes[1].set_xlabel("peak velocity (deg/s)")
axes[0].legend(fontsize=5)
sns.despine(fig)
fig.tight_layout()


# %% 2. condition comparison (amplitude, peak velocity, rate)
# The "all" bars here are the unconditioned rate/medians, so there is no separate rate cell.

fig, axes = plt.subplots(1, 3, figsize=(8, 2))

for signal in ("eye",):   # set to ("eye", "gaze") to compare both
    rows = []
    for group, title in zip(groups, titles):

        if not group:
            continue

        seconds = sum(len(R.LE_vx) for R in group) / FS
        for cond in conditions:
            amp = pooled_events(group, signal, cond, "amplitude_deg",
                                speed_threshold, min_bout, head_still_thresh)
            pkv = pooled_events(group, signal, cond, "peak_velocity_deg_s",
                                speed_threshold, min_bout, head_still_thresh)
            if len(amp) < 20:
                print(f"{title:10s} {cond:26s} only {len(amp)} events, skipped")
                continue
            rows.append((title, cond, np.median(amp), np.median(pkv),
                         len(amp) / 2 / seconds))
            print(f"{title:10s} {cond:26s} n={len(amp):6d}  "
                  f"amp={np.median(amp):6.2f}  pkv={np.median(pkv):7.1f}  "
                  f"rate={len(amp) / 2 / seconds:.2f}/s")

    x = [r[0] for r in rows]
    hue = [r[1] for r in rows]
    for ax, col, lbl in zip(axes, (2, 3, 4),
                            ("median amplitude (deg)", "median peak velocity (deg/s)",
                             "rate (events/s)")):
        sns.barplot(ax=ax, x=x, y=[r[col] for r in rows], hue=hue,
                    palette=COND_COLORS, legend=(col == 2))
        ax.set_ylabel(lbl)
        ax.tick_params(axis="x", rotation=45)

axes[0].legend(fontsize=4)
sns.despine(fig)
fig.tight_layout()


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


# %% 4. mean kinematic traces (the only cell needing a window)
# Onset-aligned, so displacement traces start at zero by construction.
#
# NOTE on displacement magnitude: traces are read at a FIXED offset from onset, while
# amplitude_deg is the distance from onset to that event's own `peak` frame. Measured at
# each event's peak the two agree exactly (corr = 1.0000 on ferret 402 EO5), but at +400 ms
# the mean trace sits BELOW the bin's nominal amplitude because the eye has partly drifted
# back — e.g. the 8-12 deg bin reaches 9.7 deg at peak but 6.2 deg at +400 ms. That gap is
# post-saccadic drift, not a trace-alignment error. The dashed line marks each bin's mean
# amplitude so the two are comparable on the plot.

signal = "eye"      # "eye" | "gaze"
condition = "all"

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

            if len(traces) < 10:
                continue

            arr = np.array(traces)
            m = arr.mean(axis=0)
            se = arr.std(axis=0) / np.sqrt(len(arr))
            line, = ax.plot(t_ms, m, lw=1, label=f"{lo}-{hi} (n={len(arr)})")
            ax.fill_between(t_ms, m - se, m + se, alpha=0.25)

            # mean amplitude of the events in this bin, for comparison with the plateau
            if kind == "displacement":
                ax.axhline(np.mean(nominal), color=line.get_color(), ls=":", lw=0.5)

        ax.axvline(0, color="0.8", lw=0.5)
        ax.set_title(f"{signal} {title}")
        ax.set_xlabel("time from onset (ms)")
        ax.set_ylabel("speed (deg/s)" if kind == "speed" else "displacement (deg)")
        ax.legend(fontsize=4)
