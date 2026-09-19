# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
# tools
from utils import create_subplot_grid, load_session_data, process_session, removeBadData, getSesh
from utils import create_subplot_grid
import numpy as np
# plotting setup
import plotly.graph_objects as go
from utils.config import SAVELOC
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import EYE_COLOR
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 6
plt.rcParams['svg.fonttype'] = 'none'

# LOAD DATA
# delayed vision: 416,411,403

SESSION = getSesh.by_ferret(402, 420)    # multiple — preserves order by ferret
# SESSION = getSesh.by_ferret(753)          # or load sessions from an inidividual ID
#SESSION = getSesh.by_name("session_2025-07-09_ferret_757_EyeCameras_P41_E13_analyzable_output") # or load a specific session by name
#SESSION = getSesh.by_name("session_2026-03-16_ferret_403_P49_E7_analyzable_output") # or load a specific session by name
#SESSION = getSesh.by_eo(7)      # or load sessions by a single EO number
#SESSION = getSesh.by_eo(10,20)   # or load sessions by an EO range (inclusive)

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

# %% plots
# amplitude vs peak velocity (log-log), eyes combined

pool_by_eo = True  # False: one panel per session | True: one panel per EO range
eo_bins = [(0, 4), (5, 9), (10, 20)]  # early / middle / late, inclusive

if pool_by_eo:

    fig, axes = create_subplot_grid(len(eo_bins))
    groups = [[R for R in Results if lo <= R.eo <= hi] for lo, hi in eo_bins]
    titles = [f"EO {lo}-{hi}" for lo, hi in eo_bins]

else:

    fig, axes = create_subplot_grid(n_sesh)
    groups = [[R] for R in Results]
    titles = [f"Ferret {R.id}" for R in Results]

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
    ax.axis([0.25, 1.75, 1.5, 3])  # [xmin, xmax, ymin, ymax]
    ax.set_xlabel("log10 amplitude (deg)")
    ax.set_ylabel("log10 peak velocity (deg/s)")


# amplitude vs duration (ms), eyes combined

pool_by_eo = True  # False: one panel per session | True: one panel per EO range
eo_bins = [(0, 4), (5, 9), (10, 20)]  # early / middle / late, inclusive

if pool_by_eo:

    fig, axes = create_subplot_grid(len(eo_bins))
    groups = [[R for R in Results if lo <= R.eo <= hi] for lo, hi in eo_bins]
    titles = [f"EO {lo}-{hi}" for lo, hi in eo_bins]
else:

    fig, axes = create_subplot_grid(n_sesh)
    groups = [[R] for R in Results]
    titles = [f"Ferret {R.id}" for R in Results]

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


# %%


# %% main sequence fit per EO bin

fit_by = "pooled"  # "pooled": one fit per EO bin | "session": one fit per session
eo_bins = [(0, 4), (5, 9), (10, 20)]
min_n = 5

def logamp_logvel(group):
    amp = np.concatenate([np.concatenate([R.df_LE["amplitude_deg"].to_numpy(),
                                          R.df_RE["amplitude_deg"].to_numpy()]) for R in group]).astype(float)
    pkv = np.concatenate([np.concatenate([R.df_LE["peak_velocity_deg_s"].to_numpy(),
                                          R.df_RE["peak_velocity_deg_s"].to_numpy()]) for R in group]).astype(float)
    x = np.log10(abs(amp))
    y = np.log10(abs(pkv))
    inds = np.isfinite(x) & np.isfinite(y)
    return x[inds], y[inds]

fig, axes = create_subplot_grid(len(eo_bins))
groups = [[R for R in Results if lo <= R.eo <= hi] for lo, hi in eo_bins]
titles = [f"EO {lo}-{hi}" for lo, hi in eo_bins]

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
    ax.axis([0.25, 1.75, 1.5, 3])
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

eo_bins = [(0, 4), (5, 9), (10, 20)]
groups = [[R for R in Results if lo <= R.eo <= hi] for lo, hi in eo_bins]
titles = [f"EO {lo}-{hi}" for lo, hi in eo_bins]

fig, axes = plt.subplots(1, 2, figsize=(6, 2))

for group, title in zip(groups, titles):

    if not group:
        continue

    x, y = logamp_logvel(group)

    sns.histplot(ax=axes[0], x=x, bins=40, element="step", fill=False, stat="density", label=title)
    sns.histplot(ax=axes[1], x=y, bins=40, element="step", fill=False, stat="density", label=title)

    print(f"{title}  amp median={np.median(x):.3f} IQR={np.subtract(*np.percentile(x, [75, 25])):.3f}"
          f"  vel median={np.median(y):.3f} IQR={np.subtract(*np.percentile(y, [75, 25])):.3f}")

axes[0].set_xlabel("log10 amplitude (deg)")
axes[1].set_xlabel("log10 peak velocity (deg/s)")
axes[0].legend()
sns.despine(fig)


# %% residual tightness per EO bin

fit_by = "pooled"  # "pooled": one spread per EO bin | "session": one spread per session
eo_bins = [(0, 4), (5, 9), (10, 20)]
min_n = 5

groups = [[R for R in Results if lo <= R.eo <= hi] for lo, hi in eo_bins]
titles = [f"EO {lo}-{hi}" for lo, hi in eo_bins]

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
