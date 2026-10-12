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
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, FS, EO_BINS, AGE_COLORS, EYE_COLOR,
                                       LOCO_COLORS, eo_groups, head_signal, eye_signal, running_mask, clean_runs, run_xcorr,
                                       paired_saccades, set_style, save_fig, plot_mean_se, plot_vs_eo)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False   # True: normal EO 8-20 vs. all delayed sessions

# OTHER SETTINGS
# Frame: flip_eye = "RE" is the CONJUGATE frame (both eyes + = same direction of rotation), so
# version = (LE + RE) / 2 and vergence = LE - RE = convergence + (= LE nasal + RE nasal).
# Absolute vergence is NOT calibrated: each eye's 0 is its own 3D-model neutral, and the session median
# vergence is ~0 in every well-tracked session. Only within-session vergence changes are interpreted.
flip_eye = "RE"
sacc_pad = (12, 20)      # frames excluded before eye-saccade onset / after its end (as vor_v2)
speed_threshold = 50     # mm/s, running_mask: stationary = not in a running bout
min_bout = 60            # frames, shortest run of frames counted as running
min_both = 0.5           # fraction of frames with both eyes tracked; sessions below are dropped
pair_window = 12         # frames (100 ms), LE-RE onset separation counted as one saccade
min_pairs = 20           # minimum paired saccades for a session's saccade measures
min_amp = 2              # deg, paired saccades smaller than this (mean of both eyes) are not used
min_frames = 500         # minimum frames for a session's frame-based measures
head_bins = np.array([0, 20, 50, 100, 1000])   # deg/s, head angular speed bins
min_bin = 50             # minimum frames for a session's value within one head-speed bin
max_lag = 15             # frames (125 ms), LE-RE velocity cross-correlation
min_run = 4 * max_lag    # frames, shortest saccade-free run worth correlating

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals")


# %% binocular frames and saccade pairing
# METHODS: binocular frames (BOTH) = both eyes' x, y, vx, vy tracked. Sessions with BOTH in < min_both of
# frames are dropped (one eye untracked for most of the session).
# METHODS: saccade-free frames (FREE) = outside every detected eye saccade of either eye, padded
# sacc_pad[0] frames before onset and sacc_pad[1] after the end (as vor_v2). running = body speed >
# speed_threshold for >= min_bout frames; stationary = body speed tracked and not running.
# METHODS: LE and RE saccades are detected independently and paired by onset (paired_saccades: greedy
# nearest onset within pair_window frames, each RE saccade used once). Displacement = position at the
# position peak minus position at onset, per eye (registration-free). The pairing depends only on onsets,
# so the horizontal (PH, conjugate frame), vertical (PV) and own-frame horizontal (PO) tables have the
# same rows. Unpaired saccades are mostly tracking dropout of the other eye (unpaired_nan).
BOTH, FREE, RUN, STAT, VERG, VVERG, PH, PV, PO = {}, {}, {}, {}, {}, {}, {}, {}, {}
rows = []
for R in Results:
    both = np.ones(len(R.LE_vx), bool)
    for eye in ("LE", "RE"):
        for key in ("x", "y", "vx", "vy"):
            both &= np.isfinite(eye_signal(R, eye, key))
    BOTH[id(R)] = both
    FREE[id(R)] = non_saccade_mask(R, pad_pre=sacc_pad[0], pad_post=sacc_pad[1])
    RUN[id(R)] = running_mask(R, speed_threshold, min_bout)
    STAT[id(R)] = np.isfinite(np.asarray(R.speed, float)) & ~RUN[id(R)]
    VERG[id(R)] = eye_signal(R, "LE", "x", flip_eye) - eye_signal(R, "RE", "x", flip_eye)
    VVERG[id(R)] = eye_signal(R, "LE", "vx", flip_eye) - eye_signal(R, "RE", "vx", flip_eye)
    PH[id(R)] = paired_saccades(R, pair_window, flip_eye, "horizontal")
    PV[id(R)] = paired_saccades(R, pair_window, flip_eye, "vertical")
    PO[id(R)] = paired_saccades(R, pair_window, None, "horizontal")
    P = PH[id(R)]
    rows.append(dict(id=R.id, eo=R.eo, both_pct=100 * both.mean(), free_both_pct=100 * (both & FREE[id(R)]).mean(),
                     n_LE=len(R.df_LE), n_RE=len(R.df_RE), paired_frac=P.paired.mean(),
                     unpaired_nan=P.other_nan[~P.paired].mean(), kept=both.mean() >= min_both))
print(pd.DataFrame(rows).round(2).to_string(index=False))

dropped = {id(R) for R in Results if BOTH[id(R)].mean() < min_both}
groups = [[R for R in group if id(R) not in dropped] for group in groups]
print(f"dropped (both eyes tracked < {min_both:.0%} of frames): "
      + ", ".join(f"F{R.id} EO{R.eo}" for R in Results if id(R) in dropped))
for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions kept, {len(set(R.id for R in group))} animals")


# %% per-session table + trends vs EO (sessions are the unit; one line per ferret)
# METHODS: saccade measures use paired saccades with both eyes' horizontal and vertical displacement finite
# and 2D amplitude (of the two-eye mean) >= min_amp; NaN with < min_pairs. version = mean of the two eyes'
# horizontal displacement (conjugate frame), vertical = mean of the two eyes' vertical displacement.
# "Horizontal" saccades = |version| > |vertical| (within 45 deg of horizontal, as main_sequence frac_horiz).
# CROSS-TALK: each eye's vertical displacement contains +-0.2-0.4 x the horizontal version, with opposite
# sign in the two eyes (mirror image; checked per session), and the horizontal vergence change contains
# ~ -1.6 x the vertical displacement. Frame artifact (eye frames rolled in mirror image) and real
# elevation-linked vergence cannot be separated here. So: horizontal measures use horizontal saccades only,
# vertical conjugacy is a partial r with the horizontal version removed, and both couplings are reported:
# xtalk = (LE - RE slope of vertical displacement on version) / 2; verg_vert = slope of the vergence change
# on the vertical displacement (OLS with version as a second regressor).
# METHODS: r_disp_x = Pearson r of LE vs RE horizontal displacement (horizontal saccades; 1 = conjugate).
# r_disp_y = partial r of LE vs RE vertical displacement, each eye's vertical regressed on the version (all
# saccades). disconj_x = median |LE - RE| / mean(|LE|, |RE|) of horizontal displacement (horizontal
# saccades; 0 = conjugate, 2 = equal and opposite). lag_sd = SD of the RE - LE onset lag (ms, all saccades).
# METHODS: slow measures use saccade-free binocular frames (NaN with < min_frames): r_vel_x = Pearson r of
# LE vs RE horizontal velocity; r_vel_y = partial r of LE vs RE vertical velocity, horizontal version
# velocity removed (as r_disp_y); verg_vel = median |vergence velocity| (deg/s) in stationary frames;
# verg_iqr = interquartile range of the vergence angle (deg; within-session spread, offset-free).
# CAVEAT: eye kinematics are 6 Hz low-passed upstream; displacements are usable, saccade timing is coarse
# (lag quantized at 8.3 ms).
def partial_r(a, b, c):
    """Pearson r of a and b after regressing each on c (OLS with intercept)."""
    ra = a - np.polyval(np.polyfit(c, a, 1), c)
    rb = b - np.polyval(np.polyfit(c, b, 1), c)
    return pearsonr(ra, rb)[0]


rows = []
for gi, (group, title) in enumerate(zip(groups, titles)):
    for R in group:
        both, free, stat = BOTH[id(R)], FREE[id(R)], STAT[id(R)]
        row = dict(id=R.id, eo=R.eo, group=title, gi=gi, both_pct=100 * both.mean(),
                   paired_frac=PH[id(R)].paired.mean())

        H, V = PH[id(R)], PV[id(R)]
        k = H.paired & np.isfinite(H.LE_disp) & np.isfinite(H.RE_disp) & np.isfinite(V.LE_disp) & np.isfinite(V.RE_disp)
        lx, rx, ly, ry = (np.asarray(c[k], float) for c in (H.LE_disp, H.RE_disp, V.LE_disp, V.RE_disp))
        version, vertical = (lx + rx) / 2, (ly + ry) / 2
        big = np.hypot(version, vertical) >= min_amp
        hz = big & (np.abs(version) > np.abs(vertical))
        lx, rx, ly, ry, version, vertical, lag = (a[big] for a in (lx, rx, ly, ry, version, vertical, H.lag_ms[k].to_numpy()))
        hz = hz[big]
        row["n_pairs"], row["n_horiz"] = big.sum(), hz.sum()
        ok, ok_h = big.sum() >= min_pairs, hz.sum() >= min_pairs
        row["r_disp_x"] = pearsonr(lx[hz], rx[hz])[0] if ok_h else np.nan
        row["r_disp_y"] = partial_r(ly, ry, version) if ok else np.nan
        row["disconj_x"] = np.median(np.abs(lx - rx)[hz] / np.abs((np.abs(lx) + np.abs(rx)) / 2)[hz]) if ok_h else np.nan
        row["lag_sd"] = np.std(lag, ddof=1) if ok else np.nan
        row["xtalk"] = (np.polyfit(version, ly, 1)[0] - np.polyfit(version, ry, 1)[0]) / 2 if ok else np.nan
        X = np.c_[version, vertical, np.ones(len(version))]
        row["verg_vert"] = np.linalg.lstsq(X, lx - rx, rcond=None)[0][1] if ok else np.nan

        m = free & both
        lv, rv = eye_signal(R, "LE", "vx", flip_eye)[m], eye_signal(R, "RE", "vx", flip_eye)[m]
        enough = m.sum() >= min_frames
        row["r_vel_x"] = pearsonr(lv, rv)[0] if enough else np.nan
        row["r_vel_y"] = partial_r(eye_signal(R, "LE", "vy", flip_eye)[m], eye_signal(R, "RE", "vy", flip_eye)[m],
                                   (lv + rv) / 2) if enough else np.nan
        k = m & stat
        row["verg_vel"] = np.median(np.abs(VVERG[id(R)][k])) if k.sum() >= min_frames else np.nan
        row["verg_iqr"] = np.subtract(*np.percentile(VERG[id(R)][m], [75, 25])) if enough else np.nan
        rows.append(row)
S = pd.DataFrame(rows)
families = dict(data=["both_pct", "paired_frac", "xtalk", "verg_vert"],
                saccade=["r_disp_x", "r_disp_y", "disconj_x", "lag_sd"],
                slow=["r_vel_x", "r_vel_y", "verg_vel", "verg_iqr"])
print(S[["id", "eo", "n_pairs", "n_horiz"] + [c for cols in families.values() for c in cols]].round(3).to_string(index=False))

# METHODS: EO trend per ferret: Pearson r of session value vs EO, sessions treated as independent
# (ferrets with >= 3 sessions). Reported as the number of ferrets whose r has the sign of the global
# trend (r over all sessions pooled), and how many of those reach p < 0.05. Normal cohort only; in
# compare_delayed mode the only comparison is delayed vs oldest normal sessions (group model below).
# METHODS: global trend across EO groups: linear mixed model (LMM), measure ~ group index (EO bin
# 0, 1, 2; or oldest normal = 0, delayed = 1) as a linear fixed effect, random intercept per ferret.
# β = change per group step. p = likelihood-ratio test against the intercept-only model (both fit
# by ML), chi2 with 1 df, Holm-corrected within each family of measures.
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
            lo = 0 if col.startswith(("both", "paired", "disconj", "lag", "verg")) else d[col].min()   # y limits on whole tick steps
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
            save_fig(fig, f"per-session binocular {fam} vs EO")

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
