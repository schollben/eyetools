# %% main script to run data loading, cleaning, and saccade extraction for a session
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
from scipy.stats import mannwhitneyu as mwu, spearmanr, chi2
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, EYE_COLOR, AGE_COLORS, FERRET_COLORS, EO_BINS, FS,
                                       eo_groups, logamp_logvel, pooled_intervals, session_rate, event_traces,
                                       set_style, session_trend, plot_vs_eo, save_fig)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False               # True: normal EO 8-20 vs. all delayed sessions

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

min_n = 5                             # minimum number of saccades a session must have before it gets fitted
q_band = (5, 15)                      # amplitude band (deg) for the q_ratio shape index
pre, post = 12, 48                    # frames: -100 to +400 ms from onset (saccade-onset traces)
amp_bins = [0, 4, 8, 12, 20, 45]      # amplitude bins (deg) for the saccade-onset traces

for group, title in zip(groups, titles):
    n_pts = sum(len(R.df_LE) + len(R.df_RE) for R in group)
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals, {n_pts} saccades")


# %% per-session table + trends vs EO (sessions are the unit; one line per ferret)
# METHODS: per session, OLS fit of log10 peak velocity on log10 amplitude (both eyes pooled;
# sessions with >= min_n saccades). pkv10 = fitted peak velocity at 10 deg, an amplitude-matched
# velocity measure. med_amp / med_pkv = session medians.
# METHODS: q_ratio = peak velocity x duration / amplitude (duration = peak - onset frames / FS), a
# velocity-profile shape index; session median over 5-15 deg saccades only, to match amplitude
# across EO (duration is quantized at 1/FS and its relative error depends on saccade size).
# METHODS: rate = saccades / tracked-eye time (Hz), averaged over eyes (session_rate). med_isi =
# median gap from one saccade's position peak to the next onset (ms), NaN-free gaps only; gaps
# < min_inter_event (8 frames, 67 ms) cannot occur by extraction.
# METHODS: frac_horiz = fraction of saccades within 45 deg of horizontal (|cos| > |sin| of direction).

rows = []
for group, title in zip(groups, titles):
    for R in group:
        x, y = logamp_logvel([R])
        if len(x) < min_n:
            continue
        s_, b_ = np.polyfit(x, y, 1)

        df = pd.concat([d for d in (R.df_LE, R.df_RE) if len(d)])
        amp = df["amplitude_deg"].to_numpy(float)
        pkv = df["peak_velocity_deg_s"].to_numpy(float)
        dur = (df["peak"] - df["onset"]).to_numpy(float) / FS
        band = (amp >= q_band[0]) & (amp <= q_band[1]) & (dur > 0)
        q = np.median(pkv[band] * dur[band] / amp[band]) if band.sum() >= min_n else np.nan
        th = np.deg2rad(df["direction_deg"].to_numpy(float))
        isi = pooled_intervals([R], "eye")

        rows.append(dict(id=R.id, eo=R.eo, group=title, n=len(x), slope=s_,
                         med_amp=np.median(10 ** x), med_pkv=np.median(10 ** y), pkv10=10 ** (s_ + b_),
                         q_ratio=q, rate=session_rate(R, "eye"),
                         med_isi=np.median(isi) if len(isi) else np.nan,
                         frac_horiz=np.mean(np.abs(np.cos(th)) > np.abs(np.sin(th)))))
S = pd.DataFrame(rows)
families = dict(kinematics=["slope", "med_amp", "med_pkv", "pkv10", "q_ratio"],
                timing=["rate", "med_isi"],
                direction=["frac_horiz"])

# METHODS: EO trend: linear mixed model, measure ~ EO (days), random intercept per ferret
# (statsmodels mixedlm, REML). p-values Holm-corrected within each family of measures
# (kinematics, timing, direction). Consistency across animals is reported descriptively as the
# number of ferrets whose Spearman rho has the same sign as β (ferrets with >= 4 sessions); no
# per-ferret tests.
# Only for the normal cohort; in compare_delayed mode the only comparison is delayed vs oldest
# normal sessions (group model below).
# METHODS: group comparison: mixed model with group (EO bin, or delayed vs oldest normal) as a
# categorical fixed effect, reference = first group. Omnibus test = likelihood-ratio test against
# the intercept-only model (both fit by ML), chi2 with (n_groups - 1) df.
group_p = {}
for fam, cols in families.items():
    print(f"--- {fam}")

    if not compare_delayed:
        trend_fits = [session_trend(S, col) for col in cols]
        p_holm = multipletests([f.pvalues["eo"] for f in trend_fits], method="holm")[1]

        fig, axes = plt.subplots(1, len(cols), figsize=(2.2 * len(cols), 2.4), squeeze=False)
        for ax, col, fit, p in zip(axes[0], cols, trend_fits, p_holm):
            plot_vs_eo(ax, S, col, color=None)
            rhos = [spearmanr(g.eo, g[col], nan_policy="omit")[0] for _, g in S.groupby("id") if len(g) >= 4]
            up = fit.params["eo"] > 0   # count ferrets whose own trend has the same sign as β
            ax.set_title(f"{col}\nβ={fit.params['eo']:+.3g}/EO day, p={p:.2g} (Holm)\n"
                         f"{sum((r > 0) == up for r in rhos)}/{len(rhos)} ferrets {'↑' if up else '↓'}", fontsize=6)
            print(f"{col:10s} EO trend Holm p={p:.3g}")
        axes[0, 0].legend(fontsize=4)
        sns.despine(fig)
        fig.tight_layout()

        if SAVE_FIGS:
            save_fig(fig, f"per-session {fam} vs EO")

    for col in cols:
        d = S.dropna(subset=[col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # few ferrets: convergence warnings are routine
            full = smf.mixedlm(f"{col} ~ C(group, Treatment('{titles[0]}'))", d, groups=d["id"]).fit(reml=False)
            null = smf.mixedlm(f"{col} ~ 1", d, groups=d["id"]).fit(reml=False)
        group_p[col] = chi2.sf(2 * (full.llf - null.llf), d.group.nunique() - 1)
        contrasts = "  ".join(f"{k.split('T.')[-1].rstrip(']')}: {full.params[k]:+.3g} (p={full.pvalues[k]:.2g})"
                              for k in full.params.index if k.startswith("C("))
        print(f"{col:10s} group LR p={group_p[col]:.3g}  |  vs {titles[0]}: {contrasts}")


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

if SAVE_FIGS:
    # fig.set_size_inches(8, 3)
    fig.tight_layout()
    save_fig(fig, "amplitude vs peak velocity")

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
ax.set_title(f"group LR p={group_p['slope']:.2g}", fontsize=6)
ax.set(ylim=[0.5, 1])
sns.despine(fig)

if SAVE_FIGS:
    fig.tight_layout()
    save_fig(fig, "main sequence slopes")


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

    sns.histplot(ax=axes[0], x=v1, bins=80, element="step", fill=False, stat="density",
                 label=title,
                 color=GROUP_COLORS[i])

    sns.histplot(ax=axes[1], x=v2, bins=80, element="step", fill=False, stat="density",
                 label=title,
                 color=GROUP_COLORS[i])

axes[0].set_xlabel("log10 amplitude (deg)")
axes[1].set_xlabel("log10 peak velocity (deg/s)")
axes[0].set_title(f"session medians, group LR p={group_p['med_amp']:.2g}", fontsize=6)
axes[1].set_title(f"session medians, group LR p={group_p['med_pkv']:.2g}", fontsize=6)
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


# %% inter-saccade interval (ISI) distributions per group
# ISI = position peak of one saccade -> onset of the next, NaN-free gaps only (pooled_intervals).
# Test: session medians (med_isi) in the group mixed model, per-session cell.

isi_bins = np.logspace(np.log10(10), np.log10(10000), 30)

fig, ax = plt.subplots(figsize=(3, 2))
for i, (group, title) in enumerate(zip(groups, titles)):

    if not group:
        continue

    isi = pooled_intervals(group, "eye")
    sns.histplot(ax=ax, x=isi, bins=isi_bins, element="step", fill=False, stat="probability",
                 color=GROUP_COLORS[i], label=title)

    m = S.med_isi[S.group == title]
    print(f"{title:16s} med_isi {m.mean():.0f} ± {m.std():.0f} ms (mean ± SD of session medians)  n={m.notna().sum()} sessions")

ax.set_xscale("log")
ax.set_xlabel("inter-saccade interval (ms)")
ax.set_title(f"session medians, group LR p={group_p['med_isi']:.2g}", fontsize=6)
ax.legend(fontsize=6)
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

fig.suptitle(f"frac_horiz group LR p={group_p['frac_horiz']:.2g}", fontsize=6)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "saccade direction")
