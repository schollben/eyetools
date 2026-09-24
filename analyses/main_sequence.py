# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
from utils import create_subplot_grid
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu as mwu
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import (EYE_COLOR, AGE_COLORS, EO_BINS, eo_groups, logamp_logvel,
                                       load_results, set_style, session_trend, plot_vs_eo)
set_style()

# LOAD DATA
Results = load_results()  # 753, 757 -> look carefully at these files


# %% settings for every plot below
pool_by_eo = True                     # False: one panel per session | True: one panel per EO range
eo_bins = EO_BINS
fit_by = "session"                      # "pooled": one fit per EO bin | "session": one fit per session
min_n = 5                              # minimum number of saccades a group must have before it gets fitted
groups, titles = eo_groups(Results, pool_by_eo, eo_bins)

for group, title in zip(groups, titles):
    n_pts = sum(len(R.df_LE) + len(R.df_RE) for R in group)
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals, {n_pts} saccades")


# %% scatter plots
# amplitude vs peak velocity (log-log), eyes combined

fig, axes = create_subplot_grid(len(groups))
for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    x, y = logamp_logvel(group)

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

for i, title in enumerate(titles):
    slopes = [f[2] for f in fits if f[0] == title]
    if not slopes:
        continue
    c = AGE_COLORS[i] if pool_by_eo else "k"
    ax.plot(np.full(len(slopes), i - 0.1), slopes, "o", ms=3, color=c)
    ax.plot(i + 0.1, np.median(slopes), "o", ms=7, mfc="white", mew=1.5, color=c)

ax.set_xticks(range(len(titles)), titles)
ax.set_ylabel("main sequence slope")
ax.set(ylim=[0.5, 1])
sns.despine(fig)

for f in fits:
    print(f"{f[0]}  id={f[1]}  slope={f[2]:.3f}  intercept={f[3]:.3f}  n={f[4]}")


# %% marginal amplitude / velocity distributions per EO bin

fig, axes = plt.subplots(1, 2, figsize=(6, 2))
group_amp = []
group_vel = []
names = []
for i, (group, title) in enumerate(zip(groups, titles)):

    if not group:
        continue

    v1, v2 = logamp_logvel(group)
    group_amp.append(v1)
    group_vel.append(v2)
    names.append(title)
    c = AGE_COLORS[i] if pool_by_eo else None

    sns.histplot(ax=axes[0], x=v1, bins=80, element="step", fill=False, stat="density", 
                 label=title, 
                 color=c)

    sns.histplot(ax=axes[1], x=v2, bins=80, element="step", fill=False, stat="density",
                 label=title, 
                 color=c)

axes[0].set_xlabel("log10 amplitude (deg)")
axes[1].set_xlabel("log10 peak velocity (deg/s)")
axes[0].legend()
sns.despine(fig)

# effect size only: rank-biserial r between EO bins on pooled saccades (positive = the later
# bin is larger). No p-values here — saccades are not independent; the tests are per
# session, in the next cell.
for name, vals in [("amp", group_amp), ("vel", group_vel)]:
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            u = mwu(vals[i], vals[j]).statistic
            r = 1 - 2 * u / (len(vals[i]) * len(vals[j]))
            print(f"{name}  {names[i]} vs {names[j]}  "
                  f"median {np.median(vals[i]):.3f} vs {np.median(vals[j]):.3f}  r={r:+.3f}")


# %% per-session measures vs EO (sessions are the unit; one line per ferret)

rows = []
for R in Results:
    x, y = logamp_logvel([R])
    if len(x) < min_n:
        continue
    s_, b_ = np.polyfit(x, y, 1)
    rows.append(dict(id=R.id, eo=R.eo, slope=s_, intercept=b_,
                     resid_sd=np.std(y - (s_ * x + b_)),
                     med_amp=np.median(10 ** x), med_pkv=np.median(10 ** y), n=len(x)))
S = pd.DataFrame(rows)

cols = ["slope", "intercept", "resid_sd", "med_amp", "med_pkv"]
fig, axes = plt.subplots(1, len(cols), figsize=(2.2 * len(cols), 2))
for ax, col in zip(axes, cols):
    plot_vs_eo(ax, S, col, color=EYE_COLOR)
    session_trend(S, col)
axes[0].legend(fontsize=4)
sns.despine(fig)
fig.tight_layout()


# %% residual tightness per EO bin
# scatter around each unit's OWN fit, so slope differences do not count as scatter

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

        us, ub = np.polyfit(ux, uy, 1)
        res = uy - (us * ux + ub)
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
