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


speed_threshold = 50     # mm/s, running_mask: stationary = not in a running bout
min_bout = 60            # frames, shortest run of frames counted as running

# RUN = running frames (body speed > speed_threshold for >= min_bout frames)
# STAT = stationary frames (body speed tracked and not running) 
FIX, RUN, STAT = {}, {}, {}
for R in Results:
    gs = np.zeros(len(R.LE_vx), bool)
    for df in (R.df_LEgaze, R.df_REgaze):
        for o, p in zip(df.onset, df.peak):
            gs[max(0, o - gaze_pad[0]):p + gaze_pad[1] + 1] = True
    FIX[id(R)] = {eye: ~gs & (np.convolve(~np.isfinite(eye_signal(R, eye, "vx")), 
                                          np.ones(2 * nan_margin + 1), "same") == 0)
                  for eye in ("LE", "RE")}
    RUN[id(R)] = running_mask(R, speed_threshold, min_bout)
    STAT[id(R)] = np.isfinite(np.asarray(R.speed, float)) & ~RUN[id(R)]

##############
# BELOW IS COPY PASTED FROM OTHER SCRIPTS TO BE REFACTORED LATER
# GOALS:
# 1.  
#

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