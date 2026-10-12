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
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, FS, EO_BINS, EYE_COLOR, AGE_COLORS, HEAD_COLOR,
                                       eo_groups, head_signal, eye_signal, running_mask, clean_runs, run_xcorr,
                                       set_style, save_fig, plot_mean_se)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False   # True: normal EO 8-20 vs. all delayed sessions

# OTHER SETTINGS
flip_eye = "LE"          # LE horizontal is flipped to match RE, so both eyes can be pooled
gaze_pad = (12, 20)      # frames excluded before gaze-event onset / after its end (see fixation cell)
nan_margin = 36          # frames excluded around each eye's tracking gaps (NaN), per eye
speed_threshold = 50     # mm/s, running_mask: stationary = not in a running bout
min_bout = 60            # frames, shortest run of frames counted as running
vel_edges = np.arange(-300, 301, 20)    # deg/s, head velocity bins
pitch_edges = np.arange(-90, 91, 10)    # deg, head pitch bins
min_bin = 20             # minimum frames for a bin's median
min_frames = 500         # minimum fixation frames for a session's cross-correlation
max_lag = 15             # frames (125 ms), eye-head cross-correlation
min_run = 4 * max_lag    # frames, shortest fixation run worth correlating

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals")


# %% fixation frames (outside gaze shifts) and locomotor state
# METHODS: every eye-head measure uses FIXATION frames only: frames outside every detected gaze event
# (gaze shift) of either eye, padded by gaze_pad[0] frames before onset and gaze_pad[1] frames after the
# event end (`peak`). Per eye, frames within nan_margin frames (300 ms) of that eye's tracking gaps (NaN)
# are also excluded: events with NaN near onset / end are dropped by extraction (never labeled), and gap
# edges carry transients. FIX[id(R)][eye] = that eye's fixation frames.
# METHODS: running = body speed > speed_threshold for >= min_bout frames (running_mask); stationary =
# body speed tracked and not running.
FIX, RUN, STAT = {}, {}, {}
for R in Results:
    gs = np.zeros(len(R.LE_vx), bool)
    for df in (R.df_LEgaze, R.df_REgaze):
        for o, p in zip(df.onset, df.peak):
            gs[max(0, o - gaze_pad[0]):p + gaze_pad[1] + 1] = True
    FIX[id(R)] = {eye: ~gs & (np.convolve(~np.isfinite(eye_signal(R, eye, "vx")), np.ones(2 * nan_margin + 1), "same") == 0)
                  for eye in ("LE", "RE")}
    RUN[id(R)] = running_mask(R, speed_threshold, min_bout)
    STAT[id(R)] = np.isfinite(np.asarray(R.speed, float)) & ~RUN[id(R)]

print(pd.DataFrame([dict(id=R.id, eo=R.eo, **{f"fix_{eye}_pct": 100 * FIX[id(R)][eye].mean() for eye in ("LE", "RE")},
                         **{f"fix_{eye}_stat_pct": 100 * (FIX[id(R)][eye] & STAT[id(R)]).mean() for eye in ("LE", "RE")},
                         **{f"fix_{eye}_run_pct": 100 * (FIX[id(R)][eye] & RUN[id(R)]).mean() for eye in ("LE", "RE")})
                    for R in Results]).round(1).to_string(index=False))


# %% horizontal: head yaw velocity vs horizontal eye velocity (fixation frames)
# Top: fixation frames pooled over the group's sessions and both eyes. Bottom: median eye velocity per
# head velocity bin (bins with >= min_bin frames), one line per group.
centers = (vel_edges[:-1] + vel_edges[1:]) / 2
fig, axes = plt.subplots(2, len(groups), figsize=(2.6 * len(groups), 4.8), squeeze=False)
for j, (group, title, c) in enumerate(zip(groups, titles, GROUP_COLORS)):
    h = np.concatenate([head_signal(R, "yaw_v")[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    e = np.concatenate([eye_signal(R, eye, "vx", flip_eye)[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    k = np.isfinite(h) & np.isfinite(e)
    h, e = h[k], e[k]
    axes[0, j].scatter(h, e, s=1, alpha=0.05, color=EYE_COLOR, lw=0, rasterized=True)
    axes[0, j].set(xlim=(-300, 300), ylim=(-100, 100), title=f"{title} ({len(h)} frames)",
                   xlabel="head yaw velocity (deg/s)", ylabel="eye horizontal velocity (deg/s)")
    b = np.digitize(h, vel_edges) - 1
    med = [np.median(e[b == i]) if (b == i).sum() >= min_bin else np.nan for i in range(len(centers))]
    axes[1, 0].plot(centers, med, color=c, marker="o", ms=2, lw=1, label=title)
axes[1, 0].axhline(0, color="0.8", lw=0.5)
axes[1, 0].set(xlabel="head yaw velocity (deg/s)", ylabel="median eye horizontal velocity (deg/s)")
axes[1, 0].legend(fontsize=5)
for ax in axes[1, 1:]:
    ax.axis("off")
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "VOR horizontal fixation")


# %% vertical: head pitch velocity vs vertical eye velocity (fixation frames)
# Same layout as the horizontal cell.
fig, axes = plt.subplots(2, len(groups), figsize=(2.6 * len(groups), 4.8), squeeze=False)
for j, (group, title, c) in enumerate(zip(groups, titles, GROUP_COLORS)):
    h = np.concatenate([head_signal(R, "pitch_v")[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    e = np.concatenate([eye_signal(R, eye, "vy", flip_eye)[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    k = np.isfinite(h) & np.isfinite(e)
    h, e = h[k], e[k]
    axes[0, j].scatter(h, e, s=1, alpha=0.05, color=EYE_COLOR, lw=0, rasterized=True)
    axes[0, j].set(xlim=(-300, 300), ylim=(-100, 100), title=f"{title} ({len(h)} frames)",
                   xlabel="head pitch velocity (deg/s)", ylabel="eye vertical velocity (deg/s)")
    b = np.digitize(h, vel_edges) - 1
    med = [np.median(e[b == i]) if (b == i).sum() >= min_bin else np.nan for i in range(len(centers))]
    axes[1, 0].plot(centers, med, color=c, marker="o", ms=2, lw=1, label=title)
axes[1, 0].axhline(0, color="0.8", lw=0.5)
axes[1, 0].set(xlabel="head pitch velocity (deg/s)", ylabel="median eye vertical velocity (deg/s)")
axes[1, 0].legend(fontsize=5)
for ax in axes[1, 1:]:
    ax.axis("off")
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "VOR vertical fixation")


# %% counter-pitch: head pitch vs vertical eye position (fixation frames; static, not VOR)
# Same layout as the horizontal cell, binned by head pitch angle.
pcenters = (pitch_edges[:-1] + pitch_edges[1:]) / 2
fig, axes = plt.subplots(2, len(groups), figsize=(2.6 * len(groups), 4.8), squeeze=False)
for j, (group, title, c) in enumerate(zip(groups, titles, GROUP_COLORS)):
    h = np.concatenate([head_signal(R, "pitch")[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    e = np.concatenate([eye_signal(R, eye, "y", flip_eye)[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    k = np.isfinite(h) & np.isfinite(e)
    h, e = h[k], e[k]
    axes[0, j].scatter(h, e, s=1, alpha=0.05, color=EYE_COLOR, lw=0, rasterized=True)
    axes[0, j].set(xlim=(-90, 90), ylim=(-20, 20), title=f"{title} ({len(h)} frames)",
                   xlabel="head pitch (deg)", ylabel="eye vertical position (deg)")
    b = np.digitize(h, pitch_edges) - 1
    med = [np.median(e[b == i]) if (b == i).sum() >= min_bin else np.nan for i in range(len(pcenters))]
    axes[1, 0].plot(pcenters, med, color=c, marker="o", ms=2, lw=1, label=title)
axes[1, 0].axhline(0, color="0.8", lw=0.5)
axes[1, 0].set(xlabel="head pitch (deg)", ylabel="median eye vertical position (deg)")
axes[1, 0].legend(fontsize=5)
for ax in axes[1, 1:]:
    ax.axis("off")
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "counter-pitch fixation")


# %% stationary vs running: binned eye velocity vs head velocity (fixation frames)
# Median eye velocity per head velocity bin; colour = group, solid = stationary, dashed = running.
fig, axes = plt.subplots(1, 2, figsize=(6, 2.4))
for ax, hname, ekey, lab in ((axes[0], "yaw_v", "vx", "horizontal"), (axes[1], "pitch_v", "vy", "vertical")):
    for group, title, c in zip(groups, titles, GROUP_COLORS):
        for state, ls, M in (("stationary", "-", STAT), ("running", "--", RUN)):
            h = np.concatenate([head_signal(R, hname)[FIX[id(R)][eye] & M[id(R)]] for R in group for eye in ("LE", "RE")])
            e = np.concatenate([eye_signal(R, eye, ekey, flip_eye)[FIX[id(R)][eye] & M[id(R)]] for R in group for eye in ("LE", "RE")])
            k = np.isfinite(h) & np.isfinite(e)
            b = np.digitize(h[k], vel_edges) - 1
            med = [np.median(e[k][b == i]) if (b == i).sum() >= min_bin else np.nan for i in range(len(centers))]
            ax.plot(centers, med, color=c, ls=ls, lw=1, label=f"{title} {state}")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set(xlabel=f"head {'yaw' if lab == 'horizontal' else 'pitch'} velocity (deg/s)",
           ylabel=f"median eye {lab} velocity (deg/s)")
axes[0].legend(fontsize=4)
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "VOR stationary vs running")


# %% head angular speed in fixation frames (fixation in either eye)
# Left: head angular speed distributions per group (solid = stationary, dashed = running). Right: head
# angular speed vs running speed in running fixation frames, one panel per group.
fig, axes = plt.subplots(1, len(groups) + 1, figsize=(2.6 * (len(groups) + 1), 2.2), squeeze=False)
for group, title, c in zip(groups, titles, GROUP_COLORS):
    for state, ls, M in (("stationary", "-", STAT), ("running", "--", RUN)):
        vals = np.concatenate([head_signal(R, "speed")[(FIX[id(R)]["LE"] | FIX[id(R)]["RE"]) & M[id(R)]] for R in group])
        vals = vals[np.isfinite(vals)]
        if len(vals) >= min_frames:
            axes[0, 0].hist(vals, bins=np.arange(0, 300, 5), histtype="step", density=True, color=c, ls=ls,
                            label=f"{title} {state}")
axes[0, 0].set_xlabel("head angular speed (deg/s)")
axes[0, 0].legend(fontsize=4)
for ax, group, title in zip(axes[0, 1:], groups, titles):
    s = np.concatenate([np.asarray(R.speed, float)[(FIX[id(R)]["LE"] | FIX[id(R)]["RE"]) & RUN[id(R)]] for R in group])
    h = np.concatenate([head_signal(R, "speed")[(FIX[id(R)]["LE"] | FIX[id(R)]["RE"]) & RUN[id(R)]] for R in group])
    k = np.isfinite(s) & np.isfinite(h)
    ax.scatter(s[k], h[k], s=1, alpha=0.05, color=HEAD_COLOR, lw=0, rasterized=True)
    ax.set(ylim=(0, 500), title=title, xlabel="running speed (mm/s)", ylabel="head angular speed (deg/s)")
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "head speed fixation")
