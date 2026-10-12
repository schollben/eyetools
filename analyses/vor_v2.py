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
from utils import create_subplot_grid, non_saccade_mask
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, FS, EO_BINS, EYE_COLOR, AGE_COLORS, HEAD_COLOR,
                                       LOCO_COLORS, eo_groups, head_signal, eye_signal, running_mask, clean_runs, run_xcorr,
                                       set_style, save_fig, plot_mean_se, plot_vs_eo)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False   # True: normal EO 8-20 vs. all delayed sessions

# OTHER SETTINGS
# Head signals: roll_v/pitch_v/yaw_v are stored as rad/s; head_signal converts to deg/s.
# Frame: flip_eye = "LE" is the HEAD frame — eye + = head yaw +, so VOR (counter-rotation) gain is
# NEGATIVE (-1 = full compensation). With "RE" the eye axis points the other way and the sign flips.
flip_eye = "LE"
sacc_pad = (12, 20)      # frames excluded before eye-saccade onset / after its end (see saccade-free cell)
speed_threshold = 50     # mm/s, running_mask: stationary = not in a running bout
min_bout = 60            # frames, shortest run of frames counted as running
min_frames = 500         # minimum saccade-free frames (both eyes pooled) for a session's gain fit
min_bin = 20             # minimum frames for a session's gain within one speed / head-velocity bin
speed_edges = np.array([0, 50, 100, 200, 400, 800])   # mm/s, gain vs running speed
mag_edges = np.arange(0, 220, 20)                      # deg/s, gain vs |head velocity|
max_lag = 15             # frames (125 ms), eye-head velocity cross-correlation
min_run = 4 * max_lag    # frames, shortest saccade-free run worth correlating

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals")


# %% saccade-free frames and locomotor state
# METHODS: every eye-head measure below uses SACCADE-FREE frames only: frames outside every detected
# eye saccade of either eye (saccades are conjugate, so one eye's saccade excludes the frame for both),
# padded by sacc_pad[0] frames before onset and sacc_pad[1] frames after the saccade end (`peak`, the
# position turning point). Padding measured on isolated saccades: eye speed is above baseline from
# ~12 frames (100 ms) before onset to ~20 frames (167 ms) after the end (6 Hz filter ringing / overshoot).
# Head saccades are NOT excluded: compensatory eye movement not detected as a saccade stays in.
# METHODS: running = body speed > speed_threshold for >= min_bout frames (running_mask); stationary =
# body speed tracked and not running.
FREE, RUN, STAT = {}, {}, {}
for R in Results:
    FREE[id(R)] = non_saccade_mask(R, pad_pre=sacc_pad[0], pad_post=sacc_pad[1])
    RUN[id(R)] = running_mask(R, speed_threshold, min_bout)
    STAT[id(R)] = np.isfinite(np.asarray(R.speed, float)) & ~RUN[id(R)]

print(pd.DataFrame([dict(id=R.id, eo=R.eo, free_pct=100 * FREE[id(R)].mean(),
                         free_stat_pct=100 * (FREE[id(R)] & STAT[id(R)]).mean(),
                         free_run_pct=100 * (FREE[id(R)] & RUN[id(R)]).mean())
                    for R in Results]).round(1).to_string(index=False))


# %% per-session table + trends vs EO (sessions are the unit; one line per ferret)
# METHODS: gains = OLS slope of eye on head over the session's saccade-free frames, both eyes pooled
# (NaN with < min_frames finite frames). gain_h: head yaw velocity vs horizontal eye velocity; gain_v:
# head pitch velocity vs vertical eye velocity; gain_h_stat / gain_h_run: gain_h in stationary /
# running frames. -1 = full compensation. counter_pitch: head pitch vs vertical eye POSITION (static
# counter-rotation, not VOR).
# METHODS: head_speed_stat / head_speed_run = median total head angular speed (deg/s) in saccade-free
# stationary / running frames; head_per_run = OLS slope of head angular speed on body speed in
# saccade-free running frames.
# METHODS: xc = cross-correlation of head yaw velocity and horizontal eye velocity over contiguous
# saccade-free NaN-free runs (>= min_run frames), z-scored per run, length-weighted over runs and eyes.
# xc_lag = lag of the largest |cc| (ms, + = eye lags head); xc_r = cc at that lag.
# CAVEAT: eye kinematics are 6 Hz low-passed upstream, head velocity is not: gain is attenuated for
# fast head movement. Gains are small (EO >= 8 median ~ -0.08); yaw_v vs d(yaw)/dt: r ~ 0.7, slope ~ 0.44.
lags = np.arange(-max_lag, max_lag + 1)
rows, XC = [], {}
for gi, (group, title) in enumerate(zip(groups, titles)):
    for R in group:
        free, run, stat = FREE[id(R)], RUN[id(R)], STAT[id(R)]
        row = dict(id=R.id, eo=R.eo, group=title, gi=gi, free_pct=100 * free.mean())
        for col, hname, ekey, m in (("gain_h", "yaw_v", "vx", free), ("gain_v", "pitch_v", "vy", free),
                                    ("gain_h_stat", "yaw_v", "vx", free & stat), ("gain_h_run", "yaw_v", "vx", free & run),
                                    ("counter_pitch", "pitch", "y", free)):
            h = np.concatenate([head_signal(R, hname)[m]] * 2)
            e = np.concatenate([eye_signal(R, eye, ekey, flip_eye)[m] for eye in ("LE", "RE")])
            k = np.isfinite(h) & np.isfinite(e)
            row[col] = np.polyfit(h[k], e[k], 1)[0] if k.sum() >= min_frames else np.nan

        hs, sp = head_signal(R, "speed"), np.asarray(R.speed, float)
        row["head_speed_stat"] = np.nanmedian(hs[free & stat]) if (free & stat).sum() >= min_frames else np.nan
        row["head_speed_run"] = np.nanmedian(hs[free & run]) if (free & run).sum() >= min_frames else np.nan
        k = free & run & np.isfinite(hs) & np.isfinite(sp)
        row["head_per_run"] = np.polyfit(sp[k], hs[k], 1)[0] if k.sum() >= min_frames else np.nan

        h = head_signal(R, "yaw_v")
        acc, weight = np.zeros(len(lags)), 0
        for eye in ("LE", "RE"):
            e = eye_signal(R, eye, "vx", flip_eye)
            for a, b in clean_runs(free & np.isfinite(h) & np.isfinite(e), min_run):
                cc = run_xcorr(h[a:b], e[a:b], max_lag)
                if cc is not None:
                    acc += cc * (b - a)
                    weight += b - a
        XC[id(R)] = acc / weight if weight >= min_frames else np.full(len(lags), np.nan)
        i = np.argmax(np.abs(XC[id(R)])) if weight >= min_frames else 0
        row["xc_lag"] = lags[i] / FS * 1000 if weight >= min_frames else np.nan
        row["xc_r"] = XC[id(R)][i]
        rows.append(row)
S = pd.DataFrame(rows)
families = dict(vor=["gain_h", "gain_v", "gain_h_stat", "gain_h_run"],
                static=["counter_pitch"],
                head=["head_speed_stat", "head_speed_run", "head_per_run"],
                timing=["xc_lag", "xc_r"])

# METHODS: EO trend per ferret: Pearson r of session value vs EO, sessions treated as independent
# (ferrets with >= 3 sessions). Reported as the number of ferrets whose r has the sign of the global
# trend (r over all sessions pooled), and how many of those reach p < 0.05. Normal cohort only; in
# compare_delayed mode the only comparison is delayed vs oldest normal sessions (group model below).
# METHODS: global trend across EO groups: linear mixed model (LMM), measure ~ group index (EO bin
# 0, 1, 2; or oldest normal = 0, delayed = 1) as a linear fixed effect, random intercept per ferret.
# β = change per group step. p = likelihood-ratio test against the intercept-only model (both fit
# by ML), chi2 with 1 df, Holm-corrected within each family of measures (vor, static, head, timing).
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
            lo = 0 if col.startswith("head_speed") else d[col].min()   # y limits on whole tick steps
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
