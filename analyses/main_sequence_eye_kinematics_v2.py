# %% main script to run data loading, cleaning, and saccade extraction for a session
#
# to do: 
# CONTOUR PLOTS FOR EYE KINEMATICS (NOT SCATTER)
#
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
from utils import create_subplot_grid
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu as mwu, pearsonr, chi2
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, EYE_COLOR, AGE_COLORS, FERRET_COLORS, EO_BINS, FS,
                                       eo_groups, logamp_logvel, pooled_intervals, session_rate, event_traces, eye_signal,
                                       set_style, plot_vs_eo, save_fig)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False               # True: normal EO 8-20 vs. all delayed sessions

# OTHER SETTINGS 
min_n = 5                             # minimum number of saccades a session must have before it gets fitted
pre, post = 12, 48                    # frames: -100 to +400 ms from onset (saccade-onset traces)
amp_bins = [0, 4, 8, 12, 20, 45]      # amplitude bins (deg) for the saccade-onset traces
iso_thresh, iso_win = 40, 12          # deg/s, frames: "isolated" saccade = speed < iso_thresh for iso_win before onset and after peak

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

for group, title in zip(groups, titles):
    n_pts = sum(len(R.df_LE) + len(R.df_RE) for R in group)
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals, {n_pts} saccades")


# %% per-session table + trends vs EO (sessions are the unit; one line per ferret)
# METHODS: per session, OLS fit of log10 peak velocity on log10 amplitude (both eyes pooled;
# sessions with >= min_n saccades). pkv10 = fitted peak velocity at 10 deg, an amplitude-matched
# velocity measure. med_amp / med_pkv = session medians.
# METHODS: rate = saccades / tracked-eye time (Hz), averaged over eyes (session_rate). med_isi =
# median gap from one saccade's position peak to the next onset (ms), NaN-free gaps only. All
# detected events are kept (extraction drop_close=False); velocity peaks < 8 frames apart merge.
# METHODS: frac_horiz = fraction of saccades within 45 deg of horizontal (|cos| > |sin| of direction).

rows = []
for gi, (group, title) in enumerate(zip(groups, titles)):
    for R in group:
        x, y = logamp_logvel([R])
        if len(x) < min_n:
            continue
        s_, b_ = np.polyfit(x, y, 1)

        df = pd.concat([d for d in (R.df_LE, R.df_RE) if len(d)])
        th = np.deg2rad(df["direction_deg"].to_numpy(float))
        isi = pooled_intervals([R], "eye")

        rows.append(dict(id=R.id, eo=R.eo, group=title, gi=gi, n=len(x), slope=s_,
                         med_amp=np.median(10 ** x), med_pkv=np.median(10 ** y), pkv10=10 ** (s_ + b_),
                         rate=session_rate(R, "eye"),
                         med_isi=np.median(isi) if len(isi) else np.nan,
                         frac_horiz=np.mean(np.abs(np.cos(th)) > np.abs(np.sin(th)))))
S = pd.DataFrame(rows)
families = dict(kinematics=["slope", "med_amp", "med_pkv", "pkv10"],
                timing=["rate", "med_isi"],
                direction=["frac_horiz"])

# METHODS: EO trend per ferret: Pearson r of session value vs EO, sessions treated as independent
# (ferrets with >= 3 sessions). Reported as the number of ferrets whose r has the sign of the global
# trend (r over all sessions pooled), and how many of those reach p < 0.05. Normal cohort only; in
# compare_delayed mode the only comparison is delayed vs oldest normal sessions (group model below).
# METHODS: global trend across EO groups: linear mixed model (LMM), measure ~ group index (EO bin
# 0, 1, 2; or oldest normal = 0, delayed = 1) as a linear fixed effect, random intercept per ferret.
# β = change per group step. p = likelihood-ratio test against the intercept-only model (both fit
# by ML), chi2 with 1 df, Holm-corrected within each family of measures (kinematics, timing, direction).
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
            lo = 0 if col in ("rate", "med_isi") else d[col].min()   # y limits on whole tick steps
            step = np.diff(MaxNLocator(5, steps=[1, 2, 5, 10]).tick_values(lo, d[col].max())[:2])[0]
            lo, hi = np.floor(lo / step) * step, np.ceil(d[col].max() / step) * step
            ax.set_yticks(np.linspace(lo, hi, round((hi - lo) / step) + 1))
            ax.set_ylim(lo, hi)
            up = pearsonr(d.eo, d[col])[0] > 0
            fr = {fid: pearsonr(g.eo, g[col]) for fid, g in d.groupby("id") if len(g) >= 3}
            same = [fid for fid, (r, p) in fr.items() if (r > 0) == up]
            ax.set_title(f"{col}\n{len(same)}/{len(fr)} ferrets {'↑' if up else '↓'} "
                         f"({sum(fr[fid][1] < 0.05 for fid in same)} with p<0.05)", fontsize=6)
            print(f"{col:10s} " + "  ".join(f"F{fid} r={r:+.2f} p={p:.2g}" for fid, (r, p) in fr.items()))
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
        print(f"{col:10s} {desc}  (mean ± SD over sessions)  {group_stat[col]}")
    if compare_delayed:   # per-ferret means (delayed ferrets have one session each)
        print(S.groupby(["group", "id"])[cols].mean().join(S.groupby(["group", "id"]).size().rename("n")).round(3).to_string())


# %% scatter plots
# amplitude vs peak velocity (log-log), eyes combined

fig, axes = create_subplot_grid(len(groups))
for i, (ax, group, title) in enumerate(zip(axes, groups, titles)):

    if not group:
        ax.set_title(title)
        continue

    x, y = logamp_logvel(group)

    sns.scatterplot(ax=ax, x=x, y=y, s=3, alpha=0.2, color="white", edgecolor="k")
    
    ax.set_title(title, color=GROUP_COLORS[i])
    ax.axis([0.25, 1.75, 1, 3])  # [xmin, xmax, ymin, ymax]
    ax.set_xlabel("log10 amplitude (deg)")
    ax.set_ylabel("log10 peak velocity (deg/s)")

if SAVE_FIGS:
    # fig.set_size_inches(8, 3)
    fig.tight_layout()
    save_fig(fig, "amplitude vs peak velocity")

# amplitude vs duration (ms), eyes combined
fig, axes = create_subplot_grid(len(groups))
for i, (ax, group, title) in enumerate(zip(axes, groups, titles)):

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

    sns.scatterplot(ax=ax, x=x, y=y, s=3, alpha=0.2, color="white", edgecolor="k")
    
    ax.set_title(title, color=GROUP_COLORS[i])
    ax.axis([0, 40, 0, 500])  # [xmin, xmax, ymin, ymax]
    ax.set_xlabel("Amplitude (deg)")
    ax.set_ylabel("Duration (ms)")

if SAVE_FIGS:
    # fig.set_size_inches(8, 3)
    fig.tight_layout()
    save_fig(fig, "amplitude vs duration")


# %% main sequence fits per EO bin (one line per session, coloured by ferret)

fig, axes = create_subplot_grid(len(groups))
for ax, group, title in zip(axes, groups, titles):

    for R in group:
        x, y = logamp_logvel([R])
        if len(x) < min_n:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        xl = np.array([x.min(), x.max()])
        ax.plot(xl, slope * xl + intercept, lw=0.6, alpha=0.6, color=FERRET_COLORS[R.id])

    ax.set_title(title)
    ax.axis([0.25, 1.75, 1, 3])
    ax.set_xlabel("log10 amplitude (deg)")
    ax.set_ylabel("log10 peak velocity (deg/s)")

axes[0].legend(handles=[plt.Line2D([], [], color=FERRET_COLORS[i], label=f"F{i}") for i in sorted(S.id.unique())],
               fontsize=4)
sns.despine(fig)

if SAVE_FIGS:
    fig.tight_layout()
    save_fig(fig, "amplitude vs velocity with fits")

# slope summary: mean ± SD over sessions (descriptive); inference = mixed model in the per-session cell
fig, ax = plt.subplots(figsize=(3, 2))
for i, title in enumerate(titles):
    sl = S.slope[S.group == title]
    if not len(sl):
        continue
    ax.errorbar(i, sl.mean(), yerr=sl.std(), fmt="o", ms=5, capsize=3, color=GROUP_COLORS[i])
    print(f"{title:16s} slope {sl.mean():.3f} ± {sl.std():.3f} (mean ± SD)  "
          f"n={len(sl)} sessions, {S.id[S.group == title].nunique()} ferrets")

ax.set_xticks(range(len(titles)), titles)
ax.set_xlim(-0.5, len(titles) - 0.5)
ax.set_ylabel("main sequence slope")
ax.set_title(group_stat["slope"], fontsize=6)
ax.set(ylim=[0.5, 1])
sns.despine(fig)

if SAVE_FIGS:
    fig.tight_layout()
    save_fig(fig, "main sequence slopes")


# %% marginal amplitude & velocity distributions per EO bin

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

    sns.histplot(ax=axes[0], x=v1, bins=80, element="step", fill=False, stat="density",
                 label=title,
                 color=GROUP_COLORS[i])

    sns.histplot(ax=axes[1], x=v2, bins=80, element="step", fill=False, stat="density",
                 label=title,
                 color=GROUP_COLORS[i])

axes[0].set_xlabel("log10 amplitude (deg)")
axes[1].set_xlabel("log10 peak velocity (deg/s)")
axes[0].set_title(f"session medians\n{group_stat['med_amp']}", fontsize=6)
axes[1].set_title(f"session medians\n{group_stat['med_pkv']}", fontsize=6)
axes[0].legend()
sns.despine(fig)

if SAVE_FIGS:
    fig.tight_layout()
    save_fig(fig, "distributions__amplitude and velocity")

# METHODS: amplitude / peak velocity differences between groups are tested on session medians
# (med_amp, med_pkv) with the group mixed model in the per-session cell (ferret random intercept,
# LR test). Pooled-saccade tests are not used because saccades are not independent; rank-biserial
# r on pooled saccades is given as an effect size only (positive = later group larger).
for name, vals in [("amp", group_amp), ("vel", group_vel)]:
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            r = 1 - 2 * mwu(vals[i], vals[j]).statistic / (len(vals[i]) * len(vals[j]))
            print(f"{name}  {names[i]} vs {names[j]}  "
                  f"median {np.median(vals[i]):.3f} vs {np.median(vals[j]):.3f} log10  r={r:+.3f}")


# %% saccade rate and inter-saccade interval (ISI) distributions per group
# ISI = position peak of one saccade -> onset of the next, NaN-free gaps only (pooled_intervals).
# Test: session rate and session medians (med_isi) in the group-trend LMM, per-session cell.

isi_bins = np.logspace(np.log10(10), np.log10(10000), 30)

fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(6, 2))
for i, (group, title) in enumerate(zip(groups, titles)):

    if not group:
        continue

    isi = pooled_intervals(group, "eye")
    sns.histplot(ax=ax, x=isi, bins=isi_bins, element="step", fill=False, stat="probability",
                 color=GROUP_COLORS[i], label=title)

    m = S.med_isi[S.group == title]
    rt = S.rate[S.group == title]
    ax_r.errorbar(i, rt.mean(), yerr=rt.std(), fmt="o", ms=5, capsize=3, color=GROUP_COLORS[i])
    print(f"{title:16s} med_isi {m.mean():.0f} ± {m.std():.0f} ms (mean ± SD of session medians)  "
          f"rate {rt.mean():.2f} ± {rt.std():.2f} Hz (mean ± SD)  n={m.notna().sum()} sessions")

ax.set_xscale("log")
ax.set_xlabel("inter-saccade interval (ms)")
ax.set_title(f"session medians\n{group_stat['med_isi']}", fontsize=6)
ax.legend(fontsize=6)
ax_r.set_xticks(range(len(titles)), titles)
ax_r.set_xlim(-0.5, len(titles) - 0.5)
ax_r.set_ylabel("saccade rate (Hz)")
ax_r.set_title(group_stat["rate"], fontsize=6)
sns.despine(fig)

if SAVE_FIGS:
    fig.tight_layout()
    save_fig(fig, "inter-saccade intervals")


# %% mean speed / displacement around saccade onset (panel per group, line per amplitude bin)
# METHODS: onset-aligned eye speed (sqrt(vx^2 + vy^2)) and displacement from the onset position,
# -100 to +400 ms; saccades whose window contains NaN are dropped. Traces are averaged within each
# session first (sessions with >= 10 saccades in the bin); band = mean ± SEM across sessions.

t_ms = np.arange(-pre, post) / FS * 1000
bin_colors = sns.color_palette("viridis", len(amp_bins) - 1)

for kind in ("speed", "displacement"):

    fig, axes = create_subplot_grid(len(groups))
    for ax, group, title in zip(axes, groups, titles):

        for lo, hi, c in zip(amp_bins[:-1], amp_bins[1:], bin_colors):
            sess = []
            for R in group:
                traces, _ = event_traces(R, "eye", kind, lo, hi, "amplitude_deg", "all", pre, post)
                if len(traces) >= 10:
                    sess.append(np.mean(traces, axis=0))
            if len(sess) < 2:
                continue
            arr = np.array(sess)
            m = arr.mean(axis=0)
            se = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
            ax.plot(t_ms, m, lw=1, color=c, label=f"{lo}-{hi}° (n={len(arr)})")
            ax.fill_between(t_ms, m - se, m + se, alpha=0.25, color=c, lw=0)

        ax.set_title(title)
        ax.set_xlabel("time from onset (ms)")
        ax.set_ylabel("speed (deg/s)" if kind == "speed" else "displacement (deg)")
        ax.legend(fontsize=4)
    sns.despine(fig)

    if SAVE_FIGS:
        fig.tight_layout()
        save_fig(fig, f"saccade-onset {kind}")


# %% saccade direction per group (check that the direction mix is not driving main-sequence changes)
# Both eyes pooled; the horizontal sign convention may differ between eyes (cf. flip_eye), so read
# this as horizontal vs vertical only. Test: frac_horiz in the group mixed model, per-session cell.

dir_bins = np.linspace(-np.pi, np.pi, 37)

fig, axes = plt.subplots(1, len(groups), figsize=(2.2 * len(groups), 2.4), subplot_kw=dict(projection="polar"))
for i, (ax, group, title) in enumerate(zip(axes, groups, titles)):

    if not group:
        continue

    th = np.deg2rad(np.concatenate([np.concatenate([R.df_LE["direction_deg"].to_numpy(float),
                                                    R.df_RE["direction_deg"].to_numpy(float)]) for R in group]))
    h, _ = np.histogram(th[np.isfinite(th)], dir_bins)
    ax.bar(dir_bins[:-1], h / h.sum(), width=np.diff(dir_bins), align="edge", color=GROUP_COLORS[i])
    ax.set_yticklabels([])

    fh = S.frac_horiz[S.group == title]
    ax.set_title(f"{title}\nfrac horizontal {fh.mean():.2f} ± {fh.std():.2f}", fontsize=6)

fig.suptitle(f"frac_horiz {group_stat['frac_horiz']}", fontsize=6)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "saccade direction")

# %%
