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
EYE_AXIS = {"h": {"LE": (-0.47, -0.52, 0.71), "RE": (-0.47, 0.52, 0.71)},
            "v": {"LE": (0.74, -0.68, 0.0),   "RE": (-0.73, -0.68, 0.0)}}
flip_eye = "LE"          # LE horizontal is flipped to match RE, so both eyes can be pooled
gaze_pad = (12, 20)      # frames excluded before gaze-event onset / after its end (see fixation cell)
nan_margin = 36          # frames excluded around each eye's tracking gaps (NaN), per eye

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

# find "fixation frames", periods of time outside gaze shifts: 
# per eye, frames within nan_margin of tracking gaps are also excluded
# FIX = fixation frames (per eye)

FIX = {}
for R in Results:
    gs = np.zeros(len(R.LE_vx), bool)
    for df in (R.df_LEgaze, R.df_REgaze):
        for o, p in zip(df.onset, df.peak):
            gs[max(0, o - gaze_pad[0]):p + gaze_pad[1] + 1] = True
    FIX[id(R)] = {eye: ~gs & (np.convolve(~np.isfinite(eye_signal(R, eye, "vx")), 
                                          np.ones(2 * nan_margin + 1), "same") == 0)
                  for eye in ("LE", "RE")}


# %% HORIZONTAL: head yaw velocity vs horizontal eye velocity (fixation frames)
# Top: fixation frames pooled over the group's sessions and both eyes. Bottom: median eye velocity per
# head velocity bin (bins with >= min_bin frames), one line per group.

vel_edges = np.arange(-180, 181, 10)    # deg/s, head velocity bins
centers = (vel_edges[:-1] + vel_edges[1:]) / 2
min_bin = 30   

fig, axes = plt.subplots(2, len(groups), figsize=(2.6 * len(groups), 4.8), squeeze=False)

for j, (group, title, c) in enumerate(zip(groups, titles, GROUP_COLORS)):

    # direct data:
    # h = np.concatenate([head_signal(R, "yaw_v")[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    # reprojection onto each eye's horizontal axis
    h = np.concatenate([sum(a * head_signal(R, n) for a, n in zip(EYE_AXIS["h"][eye], ("roll_v", "pitch_v", "yaw_v")))[FIX[id(R)][eye]]
                    for R in group for eye in ("LE", "RE")])
    
    e = np.concatenate([eye_signal(R, eye, "vx", flip_eye)[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    k = np.isfinite(h) & np.isfinite(e)
    h, e = h[k], e[k]
    
    axes[0, j].scatter(h, e, s=1, alpha=0.05, color=EYE_COLOR, lw=0, rasterized=True)
    axes[0, j].set(xlim=(-300, 300), ylim=(-100, 100), title=f"{title} ({len(h)} frames)",
                   xlabel="head velocity, eye horizontal axis (deg/s)", ylabel="eye horizontal velocity (deg/s)")
    
    b = np.digitize(h, vel_edges) - 1
    med = [np.median(e[b == i]) if (b == i).sum() >= min_bin else np.nan for i in range(len(centers))]
    axes[1, 0].plot(centers, med, color=c, marker="o", ms=2, lw=1, label=title)

axes[1, 0].axhline(0, color="0.8", lw=0.5)
axes[1, 0].set(xlabel="head velocity, eye horizontal axis (deg/s)", ylabel="median eye horizontal velocity (deg/s)")
axes[1, 0].legend(fontsize=5)
for ax in axes[1, 1:]:
    ax.axis("off")
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "VOR horizontal fixation")


# %% VERTICAL: head pitch velocity vs vertical eye velocity (fixation frames)
# Same layout as the horizontal cell.

vel_edges = np.arange(-200, 201, 10)    # deg/s, head velocity bins
centers = (vel_edges[:-1] + vel_edges[1:]) / 2
min_bin = 30   

fig, axes = plt.subplots(2, len(groups), figsize=(2.6 * len(groups), 4.8), squeeze=False)

for j, (group, title, c) in enumerate(zip(groups, titles, GROUP_COLORS)):

    # direct data:
    # h = np.concatenate([head_signal(R, "pitch_v")[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    # reprojection onto each eye's vertical axis
    h = np.concatenate([sum(a * head_signal(R, n) for a, n in zip(EYE_AXIS["v"][eye], ("roll_v", "pitch_v", "yaw_v")))[FIX[id(R)][eye]]
                    for R in group for eye in ("LE", "RE")])

    e = np.concatenate([eye_signal(R, eye, "vy", flip_eye)[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    k = np.isfinite(h) & np.isfinite(e)
    h, e = h[k], e[k]
    
    axes[0, j].scatter(h, e, s=1, alpha=0.05, color=EYE_COLOR, lw=0, rasterized=True)
    axes[0, j].set(xlim=(-300, 300), ylim=(-100, 100), title=f"{title} ({len(h)} frames)",
                   xlabel="head velocity, eye vertical axis (deg/s)", ylabel="eye vertical velocity (deg/s)")
    
    b = np.digitize(h, vel_edges) - 1
    med = [np.median(e[b == i]) if (b == i).sum() >= min_bin else np.nan for i in range(len(centers))]
    axes[1, 0].plot(centers, med, color=c, marker="o", ms=2, lw=1, label=title)

axes[1, 0].axhline(0, color="0.8", lw=0.5)
axes[1, 0].set(xlabel="head velocity, eye vertical axis (deg/s)", ylabel="median eye vertical velocity (deg/s)")
axes[1, 0].legend(fontsize=5)

for ax in axes[1, 1:]:
    ax.axis("off")
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "VOR vertical fixation")


# %% counter-pitch: head pitch vs vertical eye position (fixation frames; static, not VOR)
# Same layout as the horizontal cell, binned by head pitch angle.

pitch_edges = np.arange(-40, 61, 15)    # deg, head pitch bins
pcenters = (pitch_edges[:-1] + pitch_edges[1:]) / 2
min_bin = 30   

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


# %% eye-head velocity cross-correlation, horizontal and vertical (fixation frames)
# METHODS: per session and component, cross-correlation of head velocity (reprojected onto each eye's
# horizontal / vertical axis, EYE_AXIS) and eye velocity (horizontal / vertical) over contiguous fixation NaN-free runs (>= min_run frames), z-scored per run,
# length-weighted over runs and both eyes (NaN with < min_frames frames). Top: mean +- SEM across
# sessions per group. Bottom: per session, magnitude (largest |cc|) and its lag (ms, + = eye lags head).
lags = np.arange(-max_lag, max_lag + 1)
comps = (("horizontal", "h", "vx", "o"), ("vertical", "v", "vy", "^"))
XC = {}
for group in groups:
    for R in group:
        for comp, axis, ekey, _ in comps:
            acc, weight = np.zeros(len(lags)), 0
            for eye in ("LE", "RE"):
                h = sum(a * head_signal(R, n) for a, n in zip(EYE_AXIS[axis][eye], ("roll_v", "pitch_v", "yaw_v")))
                e = eye_signal(R, eye, ekey, flip_eye)
                for a, b in clean_runs(FIX[id(R)][eye] & np.isfinite(h) & np.isfinite(e), min_run):
                    cc = run_xcorr(h[a:b], e[a:b], max_lag)
                    if cc is not None:
                        acc += cc * (b - a)
                        weight += b - a
            XC[id(R), comp] = acc / weight if weight >= min_frames else np.full(len(lags), np.nan)

fig, axes = plt.subplots(2, 2, figsize=(6, 4.6))
for ax, (comp, axis, ekey, _) in zip(axes[0], comps):
    for group, title, c in zip(groups, titles, GROUP_COLORS):
        cc = np.array([XC[id(R), comp] for R in group if np.isfinite(XC[id(R), comp]).all()])
        plot_mean_se(ax, lags / FS * 1000, cc, c, title, min_n=2)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set(title=comp, xlabel="eye lag re head (ms)", ylabel=f"correlation (head vs eye {ekey})")
    ax.legend(fontsize=4)
for comp, axis, ekey, marker in comps:
    for group, title, c in zip(groups, titles, GROUP_COLORS):
        cc = [(R.eo, XC[id(R), comp]) for R in group if np.isfinite(XC[id(R), comp]).all()]
        i = [np.argmax(np.abs(x)) for _, x in cc]
        eo = [e for e, _ in cc]
        axes[1, 0].scatter(eo, [abs(x[ii]) for (_, x), ii in zip(cc, i)], s=8, color=c, marker=marker, label=f"{title} {comp}")
        axes[1, 1].scatter(eo, lags[i] / FS * 1000 if i else [], s=8, color=c, marker=marker, label=f"{title} {comp}")
axes[1, 0].set(xlabel="EO (days)", ylabel="peak |correlation|", ylim=(0, None))
axes[1, 1].set(xlabel="EO (days)", ylabel="lag of peak (ms)")
axes[1, 1].axhline(0, color="0.8", lw=0.5)
axes[1, 0].legend(fontsize=4)
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "eye-head xcorr fixation")
