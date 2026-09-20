# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
from utils import create_subplot_grid, load_session_data, process_session, removeBadData, getSesh
from utils import create_subplot_grid
import numpy as np
from scipy.stats import mannwhitneyu as mwu
import plotly.graph_objects as go
from utils.config import SAVELOC
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import EYE_COLOR, eo_groups, logamp_logvel
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 6
plt.rcParams['svg.fonttype'] = 'none'

# LOAD DATA
SESSION = getSesh.by_ferret(402, 405, 407, 420) # 753, 757 -> look carefully at these files
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


# %% settings for every plot below
pool_by_eo = True                     # False: one panel per session | True: one panel per EO range
eo_bins = [(0, 2), (3, 9), (10, 20)]
fit_by = "session"                      # "pooled": one fit per EO bin | "session": one fit per session
min_n = 5                              # minimum number of saccades a group must have before it gets fitted
groups, titles = eo_groups(Results, pool_by_eo, eo_bins)


# %% scatter plots
# amplitude vs peak velocity (log-log), eyes combined

fig, axes = create_subplot_grid(len(groups))
for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    amp = np.concatenate([np.concatenate([R.df_LE["amplitude_deg"].to_numpy(),
                                          R.df_RE["amplitude_deg"].to_numpy()]) for R in group]).astype(float)
    pkv = np.concatenate([np.concatenate([R.df_LE["peak_velocity_deg_s"].to_numpy(),
                                          R.df_RE["peak_velocity_deg_s"].to_numpy()]) for R in group]).astype(float)

    x = np.log10(abs(amp))
    y = np.log10(abs(pkv))

    sns.scatterplot(ax=ax, x=x, y=y, s=3, alpha=0.3)
    
    ax.set_title(title)
    ax.axis([0.25, 1.75, 1, 3])  # [xmin, xmax, ymin, ymax]
    ax.set_xlabel("log10 amplitude (deg)")
    ax.set_ylabel("log10 peak velocity (deg/s)")


# amplitude vs duration (ms), eyes combined
fig, axes = create_subplot_grid(len(groups))
for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    amp = np.concatenate([np.concatenate([R.df_LE["amplitude_deg"].to_numpy(),
                                          R.df_RE["amplitude_deg"].to_numpy()]) for R in group]).astype(float)
    dur = np.concatenate([np.concatenate([(R.df_LE["peak"] - R.df_LE["onset"]).to_numpy(),
                                          (R.df_RE["peak"] - R.df_RE["onset"]).to_numpy()]) for R in group]).astype(float)
    dur_ms = dur / 120.0 * 1000.0

    inds = np.isfinite(dur_ms)
    x = abs(amp[inds])
    y = dur_ms[inds]

    sns.scatterplot(ax=ax, x=x, y=y, s=3, alpha=0.3)
    
    ax.set_title(title)
    ax.axis([0, 40, 0, 500])  # [xmin, xmax, ymin, ymax]
    ax.set_xlabel("Amplitude (deg)")
    ax.set_ylabel("Duration (ms)")


# %% main sequence fit per EO bin

fig, axes = create_subplot_grid(len(groups))

fits = []  # (title, id, slope, intercept, n)

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    x, y = logamp_logvel(group)
    sns.scatterplot(ax=ax, x=x, y=y, s=3, alpha=0.3, color=EYE_COLOR)

    if fit_by == "pooled":
        units = [(None, group)]
    else:
        units = [(R.id, [R]) for R in group]

    for uid, unit in units:

        ux, uy = logamp_logvel(unit)
        if len(ux) < min_n:
            continue

        slope, intercept = np.polyfit(ux, uy, 1)
        fits.append((title, uid, slope, intercept, len(ux)))

        xl = np.array([ux.min(), ux.max()])
        ax.plot(xl, slope * xl + intercept, lw=1)

    ax.set_title(title)
    ax.axis([0.25, 1.75, 1, 3])
    ax.set_xlabel("log10 amplitude (deg)")
    ax.set_ylabel("log10 peak velocity (deg/s)")

sns.despine(fig)

fig, ax = plt.subplots(figsize=(3, 2))
bins = [f[0] for f in fits]
slopes = [f[2] for f in fits]

if fit_by == "pooled":
    sns.barplot(ax=ax, x=bins, y=slopes)
else:
    sns.barplot(ax=ax, x=bins, y=slopes, errorbar="sd", color="0.8")
    sns.stripplot(ax=ax, x=bins, y=slopes, size=3, color="k")

ax.set_ylabel("main sequence slope")
sns.despine(fig)

for f in fits:
    print(f"{f[0]}  id={f[1]}  slope={f[2]:.3f}  intercept={f[3]:.3f}  n={f[4]}")


# %% marginal amplitude / velocity distributions per EO bin

fig, axes = plt.subplots(1, 2, figsize=(6, 2))
group_amp = []
group_vel = []
for group, title in zip(groups, titles):

    if not group:
        continue

    v1, v2 = logamp_logvel(group)
    group_amp.append(v1)
    group_vel.append(v2)

    sns.histplot(ax=axes[0], x=v1, bins=40, element="step", fill=False, stat="density", label=title)
    sns.histplot(ax=axes[1], x=v2, bins=40, element="step", fill=False, stat="density", label=title)

axes[0].set_xlabel("log10 amplitude (deg)")
axes[1].set_xlabel("log10 peak velocity (deg/s)")
axes[0].legend()
sns.despine(fig)

pairs = [(0, 1), (0, 2), (1, 2)]

for name, vals in [("amp", group_amp), ("vel", group_vel)]:
    for i, j in pairs:
        u, p = mwu(vals[i], vals[j])
        r = 1 - 2 * u / (len(vals[i]) * len(vals[j]))
        print(f"{name}  {titles[i]} vs {titles[j]}  "
              f"median {np.median(vals[i]):.3f} vs {np.median(vals[j]):.3f}  "
              f"r={r:+.3f}  p={min(1.0, p * len(pairs)):.3e}")


# %% residual tightness per EO bin

gx, gy = logamp_logvel(Results)
slope, intercept = np.polyfit(gx, gy, 1)
print(f"global fit  slope={slope:.3f}  intercept={intercept:.3f}  n={len(gx)}")

spreads = []  # (title, id, std, n)

for group, title in zip(groups, titles):

    if not group:
        continue

    if fit_by == "pooled":
        units = [(None, group)]
    else:
        units = [(R.id, [R]) for R in group]

    for uid, unit in units:

        ux, uy = logamp_logvel(unit)
        if len(ux) < min_n:
            continue

        res = uy - (slope * ux + intercept)
        spreads.append((title, uid, np.std(res), len(ux)))

fig, ax = plt.subplots(figsize=(3, 2))
bins = [s[0] for s in spreads]
stds = [s[2] for s in spreads]

if fit_by == "pooled":
    sns.barplot(ax=ax, x=bins, y=stds)
else:
    sns.barplot(ax=ax, x=bins, y=stds, errorbar="sd", color="0.8")
    sns.stripplot(ax=ax, x=bins, y=stds, size=3, color="k")

ax.set_ylabel("residual SD (log10 deg/s)")
sns.despine(fig)

for s in spreads:
    print(f"{s[0]}  id={s[1]}  residual SD={s[2]:.3f}  n={s[3]}")
