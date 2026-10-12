# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started): cd /Users/benjaminscholl/Documents/eyetools/
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import pickle

import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from utils import non_saccade_mask
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, FS, EO_BINS, AGE_COLORS, HEAD_COLOR,
                                       EYE_COLORS, eo_groups, head_signal, eye_signal, running_mask, clean_runs,
                                       set_style, save_fig, plot_vs_eo)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False   # True: normal EO 8-20 vs. all delayed sessions

# OTHER SETTINGS
# A leaky integrator lets the eye drift back toward center: drift velocity vs eye position has slope
# -1/tau (leak). Frame: flip_eye = None keeps each eye in its own frame (x: nasal + / temporal -,
# y: up + / down -), since each eye drifts toward its own orbital center. Position 0 = neutral from
# the 3D model fit; no per-session re-centering (median clean position is printed as a check).
flip_eye = None
AXES = {"x": ("nasal", "temporal"), "y": ("up", "down")}
sacc_pad = (12, 20)      # frames excluded before eye-saccade onset / after its end (as vor_v2)
nan_pad = 6              # frames excluded on each side of untracked frames
eye_ceiling = 30         # deg/s, max 2D eye speed (either eye) in a clean frame (missed saccades)
head_ceiling = 40        # deg/s, max head angular speed in a clean frame
speed_threshold = 50     # mm/s, running_mask: stationary = not in a running bout
min_bout = 60            # frames, shortest run of frames counted as running
min_period = 60          # frames (0.5 s), shortest clean period
min_periods = 10         # minimum clean periods for a session's per-session measures
min_ecc = 1.0            # deg, periods closer to neutral than this have no defined centripetal direction
pos_edges = np.arange(-12, 13, 4)   # deg, signed position bins (drift vs position cell)
example_session = 0      # index into Results for the clean-period QC figure
qc_window = (60, 120)    # s, time window shown in the QC figure

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals")


# %% clean fixation periods
# METHODS: drift is measured in CLEAN periods: contiguous runs of >= min_period frames (0.5 s), the same
# frames for both eyes. A frame is excluded if it is
#   - inside a detected eye saccade of either eye, padded sacc_pad[0] frames before onset and
#     sacc_pad[1] frames after the end (6 Hz filter ringing, measured in vor_v2);
#   - within nan_pad frames of a frame where either eye's x, y, vx or vy is untracked (NaN);
#   - "fast": 2D eye speed of either eye >= eye_ceiling, or head angular speed >= head_ceiling (or NaN),
#     padded like a saccade. This removes small saccades below the detector threshold (after the 6 Hz
#     filter) and head movements as whole events, not just their peak frames;
#   - running (running_mask: body speed > speed_threshold for >= min_bout frames) or body speed NaN.
# CAVEAT: with head speed < head_ceiling and VOR gain ~ -0.08 (vor_v2), residual VOR is < ~3 deg/s.
CLEAN, PERIODS = {}, {}
for R in Results:
    n = len(R.LE_vx)
    bad = ~non_saccade_mask(R, pad_pre=sacc_pad[0], pad_post=sacc_pad[1])

    nan = np.zeros(n, bool)
    for eye in ("LE", "RE"):
        for key in ("x", "y", "vx", "vy"):
            nan |= ~np.isfinite(eye_signal(R, eye, key))
    for a, b in clean_runs(nan, 1):
        bad[max(0, a - nan_pad):b + nan_pad] = True

    eye_speed = np.fmax(eye_signal(R, "LE", "speed"), eye_signal(R, "RE", "speed"))
    head_speed = head_signal(R, "speed")
    fast = (eye_speed >= eye_ceiling) | ~(head_speed < head_ceiling)
    for a, b in clean_runs(fast, 1):
        bad[max(0, a - sacc_pad[0]):b + sacc_pad[1]] = True

    bad |= running_mask(R, speed_threshold, min_bout) | ~np.isfinite(np.asarray(R.speed, float))
    CLEAN[id(R)] = ~bad
    PERIODS[id(R)] = clean_runs(~bad, min_period)

rows = []
for R in Results:
    m = np.zeros(len(R.LE_vx), bool)
    for a, b in PERIODS[id(R)]:
        m[a:b] = True
    rows.append(dict(id=R.id, eo=R.eo, n_periods=len(PERIODS[id(R)]), clean_s=m.sum() / FS,
                     **{f"{eye}_{ax}_med": np.median(eye_signal(R, eye, ax, flip_eye)[m]) if m.any() else np.nan
                        for ax in AXES for eye in ("LE", "RE")}))
print(pd.DataFrame(rows).round(1).to_string(index=False))

# check: nothing fast inside a period
mx_eye = max([np.fmax(eye_signal(R, "LE", "speed"), eye_signal(R, "RE", "speed"))[a:b].max()
              for R in Results for a, b in PERIODS[id(R)]], default=np.nan)
mx_head = max([head_signal(R, "speed")[a:b].max() for R in Results for a, b in PERIODS[id(R)]], default=np.nan)
print(f"max eye speed in periods {mx_eye:.1f} deg/s, max head speed {mx_head:.1f} deg/s")

# QC figure: one session, eye position, max eye speed, head speed, clean periods shaded
R = Results[example_session]
t = np.arange(len(R.LE_vx)) / FS
w = (t >= qc_window[0]) & (t < qc_window[1])
fig, axes = plt.subplots(4, 1, figsize=(7, 5), sharex=True)
for ax, key in zip(axes[:2], AXES):
    for eye in ("LE", "RE"):
        ax.plot(t[w], eye_signal(R, eye, key, flip_eye)[w], lw=0.6, color=EYE_COLORS[eye], label=eye)
    ax.set_ylabel(f"eye {key} (deg)")
axes[0].legend(fontsize=5)
axes[2].plot(t[w], np.fmax(eye_signal(R, "LE", "speed"), eye_signal(R, "RE", "speed"))[w], lw=0.6, color="k")
axes[2].axhline(eye_ceiling, color="r", lw=0.5, ls="--")
axes[2].set_ylim(0, 4 * eye_ceiling)
axes[2].set_ylabel("max eye speed\n(deg/s)")
axes[3].plot(t[w], head_signal(R, "speed")[w], lw=0.6, color=HEAD_COLOR)
axes[3].axhline(head_ceiling, color="r", lw=0.5, ls="--")
axes[3].set_ylim(0, 4 * head_ceiling)
axes[3].set_ylabel("head speed\n(deg/s)")
axes[3].set_xlabel("time (s)")
for a, b in PERIODS[id(R)]:
    if t[a] < qc_window[1] and t[b - 1] >= qc_window[0]:
        for ax in axes:
            ax.axvspan(t[a], t[b - 1], color="0.85", lw=0, zorder=0)
axes[0].set_title(f"F{R.id} EO{R.eo}: clean periods (grey), {len(PERIODS[id(R)])} in session", fontsize=6)
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "integrator clean periods QC")


# %% per-session table + trends vs EO (sessions are the unit; one line per ferret)
# METHODS: per clean period, eye and axis: pos = mean eye position (deg), drift = OLS slope of eye
# position vs time over the period (deg/s), which averages out frame-to-frame velocity noise.
# METHODS: per session and axis, both eyes pooled (NaN with < min_periods clean periods): leak = OLS slope
# of drift on pos (1/s; negative = drift back toward neutral, tau = -1/leak); cent = fraction of
# eccentric periods (|pos| > min_ecc) whose drift points toward neutral (0.5 = no bias); drift = median
# |drift| (deg/s). clean_s = seconds of clean periods in the session.
rows = []
for gi, (group, title) in enumerate(zip(groups, titles)):
    for R in group:
        for a, b in PERIODS[id(R)]:
            tt = np.arange(b - a) / FS
            for eye in ("LE", "RE"):
                for ax in AXES:
                    x = eye_signal(R, eye, ax, flip_eye)[a:b]
                    rows.append(dict(sess=id(R), id=R.id, eo=R.eo, group=title, gi=gi, eye=eye, axis=ax,
                                     start=a, n=b - a, pos=x.mean(), drift=np.polyfit(tt, x, 1)[0]))
P = pd.DataFrame(rows)

rows = []
for gi, (group, title) in enumerate(zip(groups, titles)):
    for R in group:
        ok = len(PERIODS[id(R)]) >= min_periods
        row = dict(id=R.id, eo=R.eo, group=title, gi=gi, n_periods=len(PERIODS[id(R)]),
                   clean_s=sum(b - a for a, b in PERIODS[id(R)]) / FS)
        for ax in AXES:
            p = P[(P.sess == id(R)) & (P.axis == ax)]
            ecc = p[np.abs(p.pos) > min_ecc]
            row[f"leak_{ax}"] = np.polyfit(p.pos, p.drift, 1)[0] if ok else np.nan
            row[f"cent_{ax}"] = (np.sign(ecc.drift) != np.sign(ecc.pos)).mean() if ok and len(ecc) else np.nan
            row[f"drift_{ax}"] = np.median(np.abs(p.drift)) if ok else np.nan
        rows.append(row)
S = pd.DataFrame(rows)
families = dict(horizontal=["leak_x", "cent_x", "drift_x"],
                vertical=["leak_y", "cent_y", "drift_y"],
                data=["clean_s"])
for title in titles:
    print(f"{title}: {S.leak_x[S.group == title].notna().sum()}/{(S.group == title).sum()} sessions with >= {min_periods} periods")

# METHODS: EO trend per ferret: Pearson r of session value vs EO, sessions treated as independent
# (ferrets with >= 3 sessions). Reported as the number of ferrets whose r has the sign of the global
# trend (r over all sessions pooled), and how many of those reach p < 0.05. Normal cohort only; in
# compare_delayed mode the only comparison is delayed vs oldest normal sessions (group model below).
# METHODS: global trend across EO groups: linear mixed model (LMM), measure ~ group index (EO bin
# 0, 1, 2; or oldest normal = 0, delayed = 1) as a linear fixed effect, random intercept per ferret.
# β = change per group step. p = likelihood-ratio test against the intercept-only model (both fit
# by ML), chi2 with 1 df, Holm-corrected within each family of measures (horizontal, vertical, data).
group_stat = {}
ncol = max(len(c) for c in families.values())   # same panel size in every family figure
for fam, cols in families.items():
    print(f"--- {fam}")

    if not compare_delayed:
        fig, axes = plt.subplots(1, ncol, figsize=(2.2 * ncol, 2.4), squeeze=False)
        for ax in axes[0, len(cols):]:
            ax.axis("off")
        for ax, col in zip(axes[0], cols):
            plot_vs_eo(ax, S, col, color=None)
            d = S.dropna(subset=[col])
            lo = 0 if col.startswith(("drift", "clean")) else d[col].min()   # y limits on whole tick steps
            step = np.diff(MaxNLocator(5, steps=[1, 2, 5, 10]).tick_values(lo, d[col].max())[:2])[0]
            lo, hi = np.floor(lo / step) * step, np.ceil(d[col].max() / step) * step
            ax.set_yticks(np.linspace(lo, hi, round((hi - lo) / step) + 1))
            ax.set_ylim(lo, hi)
            up = pearsonr(d.eo, d[col])[0] > 0
            fr = {fid: pearsonr(g.eo, g[col]) for fid, g in d.groupby("id") if len(g) >= 3}
            same = [fid for fid, (r, p) in fr.items() if (r > 0) == up]
            ax.set_title(f"{col}\n{len(same)}/{len(fr)} ferrets {'↑' if up else '↓'} "
                         f"({sum(fr[fid][1] < 0.05 for fid in same)} with p<0.05)", fontsize=6)
            print(f"{col:15s} " + "  ".join(f"F{fid} r={r:+.2f} p={p:.2g}" for fid, (r, p) in fr.items()))
        axes[0, 0].legend(fontsize=4)
        sns.despine(fig)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.8, wspace=0.5)

        if SAVE_FIGS:
            save_fig(fig, f"per-session {fam} vs EO")

    p_raw, beta = [], []
    for col in cols:
        d = S.dropna(subset=[col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # few ferrets: convergence warnings are routine
            full = smf.mixedlm(f"{col} ~ gi", d, groups=d["id"]).fit(reml=False)
            null = smf.mixedlm(f"{col} ~ 1", d, groups=d["id"]).fit(reml=False)
        p_raw.append(chi2.sf(2 * (full.llf - null.llf), 1))
        beta.append(full.params["gi"])
    for col, b, p in zip(cols, beta, multipletests(p_raw, method="holm")[1]):
        group_stat[col] = f"group trend β={b:+.2g}/group, LMM p={p:.2g} (Holm)"
        desc = "  ".join(f"{t}: {S[col][S.group == t].mean():.3g} ± {S[col][S.group == t].std():.2g}" for t in titles)
        print(f"{col:15s} {desc}  (mean ± SD over sessions)  {group_stat[col]}")
    if compare_delayed:   # per-ferret means (delayed ferrets have one session each)
        print(S.groupby(["group", "id"])[cols].mean().join(S.groupby(["group", "id"]).size().rename("n")).round(3).to_string())


# %% drift vs eye position (illustration; stats are per session above)
# Top: per session, mean drift of the clean periods in each position bin (both eyes pooled, bins with
# >= 2 periods); lines = mean +- SEM across sessions, bins with >= 2 sessions.
# Bottom: clean periods pooled over the group's sessions and both eyes; line = median session leak
# drawn through the pooled mean.
centers = (pos_edges[:-1] + pos_edges[1:]) / 2
fig, axes = plt.subplots(2, len(AXES), figsize=(3 * len(AXES), 4.6), squeeze=False)
for j, (ax_key, (plus, minus)) in enumerate(AXES.items()):
    ax = axes[0, j]
    for group, title, c in zip(groups, titles, GROUP_COLORS):
        G = np.full((len(group), len(centers)), np.nan)
        for si, R in enumerate(group):
            p = P[(P.sess == id(R)) & (P.axis == ax_key)]
            for bi, (lo, hi) in enumerate(zip(pos_edges[:-1], pos_edges[1:])):
                d = p.drift[(p.pos >= lo) & (p.pos < hi)]
                if len(d) >= 2:
                    G[si, bi] = d.mean()
        n = np.isfinite(G).sum(0)
        ok = n >= 2
        mu = np.array([np.nanmean(g) if o else np.nan for g, o in zip(G.T, ok)])
        se = np.array([np.nanstd(g) / np.sqrt(k) if o else np.nan for g, o, k in zip(G.T, ok, n)])
        ax.errorbar(centers[ok], mu[ok], se[ok], marker="o", ms=3, lw=1, capsize=0, color=c, label=title)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_xlabel(f"eye position (deg, {plus} + / {minus} -)")
    ax.set_ylabel("drift (deg/s)")
    ax.set_title(f"leak_{ax_key}: {group_stat[f'leak_{ax_key}']}", fontsize=6)

    ax = axes[1, j]
    for group, title, c in zip(groups, titles, GROUP_COLORS):
        p = P[P.sess.isin([id(R) for R in group]) & (P.axis == ax_key)]
        ax.scatter(p.pos, p.drift, s=4, facecolor="w", edgecolor=c, lw=0.5, alpha=0.5)
        med = S[f"leak_{ax_key}"][S.group == title].median()
        if np.isfinite(med):
            ax.axline((p.pos.mean(), p.drift.mean()), slope=med, color=c, lw=1,
                      label=f"{title} median leak={med:.2f}/s")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_xlim(pos_edges[0], pos_edges[-1])
    ax.set_xlabel(f"eye position (deg, {plus} + / {minus} -)")
    ax.set_ylabel("drift (deg/s)")
    ax.legend(fontsize=4)
axes[0, 0].legend(fontsize=5)
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "integrator drift vs position")
