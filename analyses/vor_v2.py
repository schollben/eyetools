# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started): cd /Users/benjaminscholl/Documents/eyetools/
import sys
import warnings
from itertools import combinations
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, FS, EO_BINS, EYE_COLOR, AGE_COLORS, HEAD_COLOR,
                                       eo_groups, head_signal, eye_signal, clean_runs, run_xcorr,
                                       set_style, save_fig, plot_mean_se)
from scipy.stats import chi2
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False   # True: normal EO 8-20 vs. all delayed sessions

# OTHER SETTINGS
EYE_AXIS = {"h": {"LE": (-0.47, -0.52, 0.71), "RE": (-0.47, 0.52, 0.71)},
            "v": {"LE": (0.74, -0.68, 0.0),   "RE": (-0.73, -0.68, 0.0)}} #regression coefficients for projecting head velocity onto each eye's axis
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
# Top: log10 fraction of fixation frames (2D histogram, contours) pooled over sessions and both eyes. Bottom: median eye velocity per
# head velocity bin (bins with >= min_bin frames), one line per group.

vel_edges = np.arange(-150, 151, 10)    # deg/s, head velocity bins
centers = (vel_edges[:-1] + vel_edges[1:]) / 2
min_bin = 30   
xy_bins = (np.arange(-200, 201, 10), np.arange(-100, 101, 10))   # deg/s, 2D histogram bins (head, eye)
lvlbins = np.arange(-4, 0, 0.5)

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
    
    H = np.histogram2d(h, e, bins=xy_bins)[0].T
    cs = axes[0, j].contourf((xy_bins[0][:-1] + xy_bins[0][1:]) / 2, (xy_bins[1][:-1] + xy_bins[1][1:]) / 2,
                        np.log10(np.where(H >= 3, H / H.sum(), np.nan)), levels=lvlbins, cmap="Purples")
    
    axes[0, j].set(xlim=(-200, 200), ylim=(-100, 100), yticks=np.arange(-100, 101, 50),
                   title=f"{title} ({len(h)} frames)",
                   xlabel="head velocity, eye horizontal axis (deg/s)", ylabel="eye horizontal velocity (deg/s)")

    b = np.digitize(h, vel_edges) - 1
    med = [np.median(e[b == i]) if (b == i).sum() >= min_bin else np.nan for i in range(len(centers))]
    axes[1, 0].plot(centers, med, color=c, marker="o", ms=6, mfc="w", lw=1, label=title)

axes[1, 0].axhline(0, color="0.8", lw=0.5)
axes[1, 0].set(xlabel="head velocity, eye horizontal axis (deg/s)", ylabel="median eye horizontal velocity (deg/s)")
axes[1, 0].legend(fontsize=5)
fig.colorbar(cs, cax=axes[1, 1].inset_axes([0.1, 0, 0.08, 1]), label="log10 fraction of frames")

for ax in axes[1, 1:]:
    ax.axis("off")
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "VOR horizontal fixation")


# %% VERTICAL: head pitch velocity vs vertical eye velocity (fixation frames)
# Same layout as the horizontal cell.

vel_edges = np.arange(-150, 151, 10)    # deg/s, head velocity bins
centers = (vel_edges[:-1] + vel_edges[1:]) / 2
min_bin = 30   
xy_bins = (np.arange(-200, 201, 10), np.arange(-100, 101, 10))   # deg/s, 2D histogram bins (head, eye)
lvlbins = np.arange(-4, 0, 0.5)

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
    
    H = np.histogram2d(h, e, bins=xy_bins)[0].T
    axes[0, j].contourf((xy_bins[0][:-1] + xy_bins[0][1:]) / 2, (xy_bins[1][:-1] + xy_bins[1][1:]) / 2,
                        np.log10(np.where(H >= 3, H / H.sum(), np.nan)), levels=lvlbins, cmap="Purples")
    
    axes[0, j].set(xlim=(-200, 200), ylim=(-100, 100), yticks=np.arange(-100, 101, 50),
                   title=f"{title} ({len(h)} frames)",
                   xlabel="head velocity, eye vertical axis (deg/s)", ylabel="eye vertical velocity (deg/s)")
    
    b = np.digitize(h, vel_edges) - 1
    med = [np.median(e[b == i]) if (b == i).sum() >= min_bin else np.nan for i in range(len(centers))]
    axes[1, 0].plot(centers, med, color=c, marker="o", ms=6, mfc="w", lw=1, label=title)

axes[1, 0].axhline(0, color="0.8", lw=0.5)
axes[1, 0].set(xlabel="head velocity, eye vertical axis (deg/s)", ylabel="median eye vertical velocity (deg/s)")
axes[1, 0].legend(fontsize=5)
fig.colorbar(cs, cax=axes[1, 1].inset_axes([0.1, 0, 0.08, 1]), label="log10 fraction of frames")

for ax in axes[1, 1:]:
    ax.axis("off")
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "VOR vertical fixation")


# %% counter-pitch: head pitch vs vertical eye position (fixation frames; static, not VOR)
# Same layout as the horizontal cell, binned by head pitch angle.

pitch_edges = np.arange(-30, 61, 15)    # deg, head pitch bins
pcenters = (pitch_edges[:-1] + pitch_edges[1:]) / 2
min_bin = 30   
xy_bins = (np.arange(-90, 91, 5), np.arange(-20, 21, 1))         # deg, 2D histogram bins (head, eye)
lvlbins = np.arange(-4, 0, 0.5)

fig, axes = plt.subplots(2, len(groups), figsize=(2.6 * len(groups), 4.8), squeeze=False)

for j, (group, title, c) in enumerate(zip(groups, titles, GROUP_COLORS)):
    
    h = np.concatenate([head_signal(R, "pitch")[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    e = np.concatenate([eye_signal(R, eye, "y", flip_eye)[FIX[id(R)][eye]] for R in group for eye in ("LE", "RE")])
    k = np.isfinite(h) & np.isfinite(e)
    h, e = h[k], e[k]
    
    H = np.histogram2d(h, e, bins=xy_bins)[0].T
    cs = axes[0, j].contourf((xy_bins[0][:-1] + xy_bins[0][1:]) / 2, (xy_bins[1][:-1] + xy_bins[1][1:]) / 2,
                        np.log10(np.where(H > 0, H / H.sum(), np.nan)), levels=lvlbins, cmap="Purples")
    axes[0, j].set(xlim=(-90, 90), ylim=(-20, 20), title=f"{title} ({len(h)} frames)",
                   xlabel="head pitch (deg)", ylabel="eye vertical position (deg)")
    
    b = np.digitize(h, pitch_edges) - 1
    med = [np.median(e[b == i]) if (b == i).sum() >= min_bin else np.nan for i in range(len(pcenters))]
    axes[1, 0].plot(pcenters, med, color=c, marker="o", ms=6, mfc="w", lw=1, label=title)

axes[1, 0].axhline(0, color="0.8", lw=0.5)
axes[1, 0].set(xlabel="head pitch (deg)", ylabel="median eye vertical position (deg)")
axes[1, 0].legend(fontsize=5)
axes[1, 0].set_ylim(-5,5)
axes[1, 0].set_xlim(-30,60)
fig.colorbar(cs, cax=axes[1, 1].inset_axes([0.1, 0, 0.08, 1]), label="log10 fraction of frames")

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
# sessions per group. Bottom: per group, sessions (points) and mean +- SEM (open markers) of magnitude (largest |cc|)
# and its lag (ms, + = eye lags head); o = horizontal, ^ = vertical.

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

rng = np.random.default_rng(0)

for k, (comp, axis, ekey, marker) in enumerate(comps):
    dx = (k - 0.5) * 0.3    # horizontal left, vertical right of each group
    for j, (group, title, c) in enumerate(zip(groups, titles, GROUP_COLORS)):
        cc = np.array([XC[id(R), comp] for R in group if np.isfinite(XC[id(R), comp]).all()])
        if len(cc) == 0:
            continue
        i = np.argmax(np.abs(cc), axis=1)
        for ax, y in zip(axes[1], (np.abs(cc[np.arange(len(cc)), i]), lags[i] / FS * 1000)):
            ax.scatter(j + dx + rng.uniform(-0.05, 0.05, len(y)), y, s=8, color=c, marker=marker, alpha=0.6)
            ax.errorbar(j + dx, y.mean(), yerr=y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0,
                        color="k", marker=marker, ms=5, mfc="w", capsize=2, label=comp if j == 0 else None)

for ax, ylabel in zip(axes[1], ("peak |correlation|", "lag of peak (ms)")):
    ax.set(xticks=range(len(groups)), xticklabels=titles, ylabel=ylabel)
    
axes[1, 0].set(ylim=(0, None))
axes[1, 1].axhline(0, color="0.8", lw=0.5)
axes[1, 0].legend(fontsize=4)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "eye-head xcorr fixation")




# %% VOR gain per session vs EO (horizontal and vertical, fixation frames)
# METHODS: per session, gain = least-squares slope of eye velocity on head velocity (reprojected onto
# each eye's axis, EYE_AXIS) over fixation frames of both eyes pooled with |head| <= gain_max_head
# (NaN with < min_frames frames); -1 = full compensation. Sessions (points) and mean +- SEM per group.
# Stats: each pair of groups, LMM gain ~ group + (1|ferret) vs gain ~ 1 + (1|ferret), likelihood-ratio
# test; Holm-corrected over all pairs and both components.

gain_max_head = 180    # deg/s, fit only the low-speed range (compensation saturates above)

rows = []

for group, title in zip(groups, titles):
    for R in group:
        row = dict(id=R.id, eo=R.eo, group=title)
        for axis, ekey in (("h", "vx"), ("v", "vy")):
            h = np.concatenate([sum(a * head_signal(R, n) for a, n in zip(EYE_AXIS[axis][eye], ("roll_v", "pitch_v", "yaw_v")))[FIX[id(R)][eye]]
                                for eye in ("LE", "RE")])
            e = np.concatenate([eye_signal(R, eye, ekey, flip_eye)[FIX[id(R)][eye]] for eye in ("LE", "RE")])
            k = np.isfinite(h) & np.isfinite(e) & (np.abs(h) <= gain_max_head)
            row[f"gain_{axis}"] = np.polyfit(h[k], e[k], 1)[0] if k.sum() >= min_frames else np.nan
        rows.append(row)

G = pd.DataFrame(rows)

rng = np.random.default_rng(0)

fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.4))

for ax, col, comp in zip(axes, ("gain_h", "gain_v"), ("horizontal", "vertical")):
    for j, (title, c) in enumerate(zip(titles, GROUP_COLORS)):
        y = G[col][G.group == title].dropna().to_numpy()
        ax.scatter(j + rng.uniform(-0.1, 0.1, len(y)), y, s=8, color=c, alpha=0.6)
        ax.errorbar(j, y.mean(), yerr=y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0,
                    color="k", marker="o", ms=5, mfc="w", capsize=2)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set(title=comp, xticks=range(len(titles)), xticklabels=titles, ylabel="VOR gain (eye / head velocity)")
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "VOR gain vs EO")


P = []
for col, comp in (("gain_h", "horizontal"), ("gain_v", "vertical")):
    for a, b in combinations(titles, 2):

        d = G[G.group.isin([a, b])].dropna(subset=[col])
        d = d.assign(gi=(d.group == b).astype(int))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # few ferrets: convergence warnings are routine
            full = smf.mixedlm(f"{col} ~ gi", d, groups=d["id"]).fit(reml=False)
            null = smf.mixedlm(f"{col} ~ 1", d, groups=d["id"]).fit(reml=False)
        
        P.append((comp, a, b, full.params["gi"], chi2.sf(2 * (full.llf - null.llf), 1)))

for (comp, a, b, beta, p), p_holm in zip(P, multipletests([x[-1] for x in P], method="holm")[1]):
    print(f"{comp:10s} {a} vs {b}: Δgain={beta:+.3f}  LMM p={p:.3g}  Holm p={p_holm:.3g}")


#claude explanation: 
# 1. Loop setup

# P = []
# for col, comp in (("gain_h", "horizontal"), ("gain_v", "vertical")):
#     for a, b in combinations(titles, 2):
# P collects one result tuple per test.
# combinations(titles, 2) gives the three unordered pairs: (EO 0-3, EO 4-7), (EO 0-3, EO 8-20) and (EO 4-7, EO 8-20). a is always the earlier group and b the later one.
# 2. Data for this test

# d = G[G.group.isin([a, b])].dropna(subset=[col])
# d = d.assign(gi=(d.group == b).astype(int))
# G has one row per session, with columns id (ferret), eo, group, gain_h and gain_v.
# Only sessions in groups a or b are kept. Sessions with NaN gain for this component (fewer than min_frames frames) are dropped.
# gi is a dummy variable: 0 for sessions in a, 1 for sessions in b.
# Example: for EO 0-3 vs EO 8-20 (horizontal), d has 15 + 15 = 30 rows. Ferrets with sessions in both groups appear under both gi values. Ferrets present in only one group still help estimate the between-ferret variance.
# 3. Two nested models

# full = smf.mixedlm(f"{col} ~ gi", d, groups=d["id"]).fit(reml=False)
# null = smf.mixedlm(f"{col} ~ 1",  d, groups=d["id"]).fit(reml=False)
# For session s of ferret f:

# Full model: gain = β₀ + β₁·gi + u_f + ε
# Null model: gain = β₀ + u_f + ε
# The terms are:

# β₀: the mean gain of group a (in the full model).
# β₁: the group difference, i.e. b minus a, after accounting for ferret identity.
# u_f: a random intercept per ferret, groups=d["id"], drawn from N(0, σ²_ferret). It absorbs the fact that some ferrets have higher or lower gain overall.
# ε: session-to-session noise, N(0, σ²).
# The only difference between the two models is β₁. The random-effect structure is the same in both, so the comparison tests just the group effect.

# reml=False makes the models fit by maximum likelihood. REML likelihoods can't be compared between models whose fixed effects differ, so a likelihood-ratio test on β₁ needs ML fits.

# The warnings are suppressed because with this few ferrets, statsmodels often warns that σ²_ferret is near 0 or that convergence was slow. The fits are still usable.

# 4. The test

# P.append((comp, a, b, full.params["gi"], chi2.sf(2 * (full.llf - null.llf), 1)))
# full.params["gi"] is β₁. It is the Δgain you see printed: the estimated mean difference (later minus earlier) within ferret. Negative means the later group compensates more.
# full.llf and null.llf are the log-likelihoods of the two models.
# 2 * (full.llf - null.llf) is the likelihood-ratio statistic. It measures how much better the data fit when the group difference is allowed.
# Under the null hypothesis (β₁ = 0), this statistic is approximately χ² with 1 degree of freedom, because the full model has one extra parameter.
# chi2.sf(x, 1) gives P(χ²₁ ≥ x), which is the raw p-value.
# 5. Holm correction

# for (...), p_holm in zip(P, multipletests([x[-1] for x in P], method="holm")[1]):
# multipletests(..., method="holm") takes the 6 raw p-values and returns adjusted p-values in the original order. The procedure:

# Sort the 6 p-values from smallest to largest: p₍₁₎ ≤ … ≤ p₍₆₎.
# Multiply the smallest by 6, the next by 5, …, the largest by 1.
# Make the results non-decreasing in that sorted order (each adjusted p is at least the one before it), and cap them at 1.
# Example from your output: horizontal 0-3 vs 8-20 has the smallest raw p, 0.00163, and 0.00163 × 6 = 0.0098. The two vertical/horizontal tests at 4-7 vs 8-20 (raw 0.404 and 0.579) both come out as 0.808, because of the non-decreasing step.

# An adjusted p below 0.05 means the comparison is significant while keeping the chance of any false positive among the 6 tests at 5% or less.

# What a significant result means
# "Sessions in group b have a different mean gain from sessions in group a, by Δgain. This holds after accounting for ferrets differing in baseline gain, and after correcting for 6 comparisons."
