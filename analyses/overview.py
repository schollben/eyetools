# %% dataset and behavior overview (Fig 1)
# Set your paths in local_config.py (copy local_config.py.example to get started).
# cd /Users/benjaminscholl/Documents/eyetools/
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import (FS, HEAD_COLOR, LE_COLOR, RE_COLOR, LOCO_COLORS,
                                       load_results, set_style, save_fig, running_mask,
                                       head_signal, eye_signal, unwrap_deg, session_trend,
                                       plot_vs_eo)
set_style()

# LOAD DATA
Results = load_results()

# quiet state = movement_stats.py's "stationary_and_head_still"
speed_threshold = 50     # mm/s, locomotion
min_bout = 30            # frames
head_still_thresh = 20   # deg/s
save_figs = False

rows = []
for R in Results:
    run = running_mask(R, speed_threshold, min_bout)
    hs = head_signal(R, "speed")
    valid = np.isfinite(R.speed)
    rows.append(dict(id=R.id, eo=R.eo, minutes=len(R.LE_vx) / FS / 60,
                     LE_missing=100 * np.mean(~np.isfinite(R.LE_x)),
                     RE_missing=100 * np.mean(~np.isfinite(R.RE_x)),
                     running=100 * run[valid].mean(),
                     quiet=100 * (~run & (hs < head_still_thresh))[valid].mean(),
                     head_speed=np.nanmedian(hs),
                     head_rate=len(R.df_head) / (len(R.LE_vx) / FS)))
B = pd.DataFrame(rows)
print(B.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


# %% 1. sessions, recording length and tracking per ferret

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2))
plot_vs_eo(axes[0], B, "minutes")
axes[0].set_ylabel("recording (min)")
plot_vs_eo(axes[1], B, "LE_missing", color=LE_COLOR)
axes[1].set_ylabel("LE untracked (%)")
plot_vs_eo(axes[2], B, "RE_missing", color=RE_COLOR)
axes[2].set_ylabel("RE untracked (%)")
axes[0].legend(fontsize=4)
sns.despine(fig)
fig.tight_layout()
if save_figs:
    save_fig(fig, "overview_1_dataset")

print(B.groupby("id").agg(sessions=("eo", "size"), eo_min=("eo", "min"),
                          eo_max=("eo", "max"), minutes=("minutes", "sum")))


# %% 2. behavior across development: running, quiet, head movement
# Context for every rate and coupling measure: if behavior changes with age, so will they.

cols = [("running", "% time running", LOCO_COLORS["running"]),
        ("quiet", "% time quiet", LOCO_COLORS["stationary"]),
        ("head_speed", "median head speed (deg/s)", HEAD_COLOR),
        ("head_rate", "head saccades (/s)", HEAD_COLOR)]

fig, axes = plt.subplots(1, len(cols), figsize=(2.2 * len(cols), 2))
for ax, (col, lbl, c) in zip(axes, cols):
    plot_vs_eo(ax, B, col, color=c)
    ax.set_ylabel(lbl)
    session_trend(B, col)
axes[0].legend(fontsize=4)
sns.despine(fig)
fig.tight_layout()
if save_figs:
    save_fig(fig, "overview_2_behavior")


# %% 3. example traces: one young and one old session
# Head frame (flip "LE"): eye + = head yaw +, gaze = yaw + eye. Head saccades are shaded
# onset -> peak; eye saccade onsets are ticks.

examples = [(407, 1), (407, 9)]   # (ferret, EO)
start_s, dur_s = 60, 10

fig, axes = plt.subplots(len(examples), 1, figsize=(6, 2 * len(examples)), squeeze=False)

for ax, (fid, eo) in zip(axes[:, 0], examples):
    R = next(R for R in Results if R.id == fid and R.eo == eo)
    a, b = int(start_s * FS), int((start_s + dur_s) * FS)
    t = np.arange(a, b) / FS
    yaw = unwrap_deg(R.yaw)[a:b]
    yaw = yaw - yaw[0]
    ax.plot(t, yaw, color=HEAD_COLOR, lw=0.8, label="head yaw")
    for eye, c, df in (("LE", LE_COLOR, R.df_LE), ("RE", RE_COLOR, R.df_RE)):
        x = eye_signal(R, eye, "x", "LE")[a:b]
        ax.plot(t, x - np.nanmedian(x), color=c, lw=0.8, label=eye)
        if eye == "LE":
            ax.plot(t, yaw + x - np.nanmedian(x), color="k", lw=0.8, label="gaze (LE)")
        on = df["onset"].to_numpy()
        on = on[(on >= a) & (on < b)]
        top = ax.get_ylim()[1]
        ax.plot(on / FS, np.full(len(on), top), "|", color=c, ms=4)
    for o, p in zip(R.df_head["onset"], R.df_head["peak"]):
        if p > a and o < b:
            ax.axvspan(max(o, a) / FS, min(p, b) / FS, color=HEAD_COLOR, alpha=0.15, lw=0)
    ax.set_title(f"ferret {fid} EO{eo}", fontsize=6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rotation (deg)")
axes[0, 0].legend(fontsize=4)
sns.despine(fig)
fig.tight_layout()
if save_figs:
    save_fig(fig, "overview_3_examples")
