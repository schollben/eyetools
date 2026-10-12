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
from utils.find_arcs import find_arcs
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, FS, EO_BINS, EYE_COLOR, AGE_COLORS, HEAD_COLOR,
                                       LE_COLOR, RE_COLOR, eo_groups, unwrap_deg, set_style, save_fig, head_eye_windows,
                                       head_triggered_windows, running_mask,
                                       plot_mean_se, plot_vs_eo)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = False   # True: normal EO 8-20 vs. all delayed sessions

# OTHER SETTINGS
# Frame: flip_eye = "LE" is the HEAD frame — eye + = head yaw +, and world gaze = yaw + eye
# (checked against the pipeline's gaze). With "RE" the eye axis points the other way and a
# counter-rotation would read as positive.
flip_eye = "LE"
pair_window = 30         # frames (250 ms) an eye saccade may lead its head saccade
pre, post = 24, 72       # frames: -200 to +600 ms from eye-saccade onset
pscr_win = 24            # frames (200 ms) after the eye saccade ends: counter-rotation window
head_pre, head_post = 24, 144   # frames: -200 to +1200 ms from head onset (panel B)
amp_classes = [(5, 10), (10, 15), (15, np.inf)]   # Wallace Fig 4A/4C classes, horizontal deg
head_vel_bins = np.arange(50, 551, 100)            # Wallace Fig 4D bins, deg/s
speed_threshold = 20     # mm/s, running_mask: stationary = not in a running bout
min_bout = 30            # frames
min_n = 5                # minimum saccades for a session's per-session PSCR measures
min_ev = 5              # minimum events for a session to contribute a trace (mean +- SEM across sessions)
states = ("all", "outside turns", "stationary")   # head saccades split by state at head onset (D)

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

ARCS = {id(R): find_arcs(R) for R in Results}   # turns (see event tables cell)
for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals, "
          f"{sum(len(R.df_LE) + len(R.df_RE) for R in group)} eye / {sum(len(R.df_head) for R in group)} head saccades, "
          f"{sum(len(ARCS[id(R)]) for R in group)} turns")


# %% event tables: head saccades, turns, eye saccades with their head / turn context
# Head saccades are R.df_head as extracted by process_session. Two columns are recomputed:
# peak velocity (stored in rad/s) and amplitude, taken from UNWRAPPED yaw at the same
# onset/peak frames — raw yaw jumps by 360 deg at the +-180 seam.
# METHODS: turns = arcs in the 2D body path (utils/find_arcs.find_arcs, default fine + coarse scales,
# tuned on synthetic paths in test_findarcs.py): sustained same-direction path curvature while moving
# > 50 mm/s, >= 0.3 s long, turning the heading >= 30 deg. A frame is in a turn from arc onset to end.
# METHODS: stationary = body speed tracked and not in a running bout (running_mask: speed >
# speed_threshold for >= min_bout frames).
HEAD, STATE, E, W = {}, {}, {}, {}
for R in Results:
    df = R.df_head.copy()
    yaw, pitch = unwrap_deg(R.yaw), np.asarray(R.pitch, float)
    on, pk = df["onset"].to_numpy().astype(int), df["peak"].to_numpy().astype(int)
    df["yaw_disp"] = yaw[pk] - yaw[on]
    df["amplitude_deg"] = np.hypot(df["yaw_disp"], pitch[pk] - pitch[on])
    df["peak_velocity_deg_s"] = np.rad2deg(df["peak_velocity_deg_s"].astype(float))
    HEAD[id(R)] = df

    turn = np.zeros(len(R.yaw), bool)
    for s, e in zip(ARCS[id(R)]["onset"], ARCS[id(R)]["end"]):
        turn[s:e] = True
    STATE[id(R)] = {"all": np.ones(len(turn), bool), "outside turns": ~turn,
                    "stationary": np.isfinite(np.asarray(R.speed, float)) & ~running_mask(R, speed_threshold, min_bout)}

    # one row per eye saccade (E) and its onset-aligned traces (W, same row order)
    e, W[id(R)] = head_eye_windows(R, HEAD[id(R)], flip_eye, pre, post, pair_window, pscr_win)
    e["in_turn"] = turn[e["onset"].to_numpy()]
    e["with_head"] = np.sign(e["eye_disp"]) == e["head_sign"]   # eye moved the way the head turned (paired or not)
    E[id(R)] = e

print(pd.DataFrame([dict(id=R.id, eo=R.eo, turns=len(ARCS[id(R)]),
                         turns_per_moving_min=len(ARCS[id(R)]) / max(np.nansum(np.asarray(R.speed, float) > 50) / FS / 60, 1e-9),
                         eye_in_turn=int(E[id(R)].in_turn.sum()), n_eye=len(E[id(R)]))
                    for R in Results]).round(1).to_string(index=False))


# %% per-session table + trends vs EO (sessions are the unit; one line per ferret)
# METHODS: head_rate = head saccades / tracked-yaw time (Hz); head_amp / head_pkv = session medians
# (amplitude from unwrapped yaw + pitch).
# METHODS: PSCR measures use eye saccades whose onset is inside a turn, with a NaN-free window, that
# moved with the head (sign of horizontal eye displacement = sign of head yaw movement). cr_vel =
# median most-negative eye velocity in the pscr_win frames after the saccade (counter-rotation);
# pscr_gain = median of mean eye / mean head velocity over that window (head > 30 deg/s), -1 = full
# compensation. Sessions with < min_n such saccades are NaN. turn_frac = % of eye saccades in turns.

rows = []
for gi, (group, title) in enumerate(zip(groups, titles)):
    for R in group:
        H, e = HEAD[id(R)], E[id(R)]
        cr = e[e.in_turn & e.clean & e.with_head]
        ok = len(cr) >= min_n
        rows.append(dict(id=R.id, eo=R.eo, group=title, gi=gi, n_pscr=len(cr),
                         head_rate=len(H) / (np.isfinite(R.yaw).sum() / FS),
                         head_amp=H.amplitude_deg.median(), head_pkv=H.peak_velocity_deg_s.median(),
                         pscr_gain=cr.pscr_gain.median() if ok else np.nan,
                         cr_vel=cr.eye_cr_vel.median() if ok else np.nan,
                         turn_frac=100 * e.in_turn.mean()))
S = pd.DataFrame(rows)
families = dict(head=["head_rate", "head_amp", "head_pkv"],
                pscr=["pscr_gain", "cr_vel", "turn_frac"])

# METHODS: EO trend per ferret: Pearson r of session value vs EO, sessions treated as independent
# (ferrets with >= 3 sessions). Reported as the number of ferrets whose r has the sign of the global
# trend (r over all sessions pooled), and how many of those reach p < 0.05. Normal cohort only; in
# compare_delayed mode the only comparison is delayed vs oldest normal sessions (group model below).
# METHODS: global trend across EO groups: linear mixed model (LMM), measure ~ group index (EO bin
# 0, 1, 2; or oldest normal = 0, delayed = 1) as a linear fixed effect, random intercept per ferret.
# β = change per group step. p = likelihood-ratio test against the intercept-only model (both fit
# by ML), chi2 with 1 df, Holm-corrected within each family of measures (head, pscr).
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
            lo = 0 if col in ("head_rate", "turn_frac") else d[col].min()   # y limits on whole tick steps
            step = np.diff(MaxNLocator(5, steps=[1, 2, 5, 10]).tick_values(lo, d[col].max())[:2])[0]
            lo, hi = np.floor(lo / step) * step, np.ceil(d[col].max() / step) * step
            ax.set_yticks(np.linspace(lo, hi, round((hi - lo) / step) + 1))
            ax.set_ylim(lo, hi)
            up = pearsonr(d.eo, d[col])[0] > 0
            fr = {fid: pearsonr(g.eo, g[col]) for fid, g in d.groupby("id") if len(g) >= 3}
            same = [fid for fid, (r, p) in fr.items() if (r > 0) == up]
            ax.set_title(f"{col}\n{len(same)}/{len(fr)} ferrets {'↑' if up else '↓'} "
                         f"({sum(fr[fid][1] < 0.05 for fid in same)} with p<0.05)", fontsize=6)
            print(f"{col:20s} " + "  ".join(f"F{fid} r={r:+.2f} p={p:.2g}" for fid, (r, p) in fr.items()))
        axes[0, 0].legend(fontsize=4)
        sns.despine(fig)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.8, wspace=0.5)

        if SAVE_FIGS:
            save_fig(fig, f"eye_head per-session {fam} vs EO")

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
        print(f"{col:20s} {desc}  (mean ± SD over sessions)  {group_stat[col]}")
    if compare_delayed:   # per-ferret means (delayed ferrets have one session each)
        print(S.groupby(["group", "id"])[cols].mean().join(S.groupby(["group", "id"]).size().rename("n")).round(3).to_string())


# %% B. head-onset-triggered average: head and eye-in-head (head frame)
# Sign-aligned so every head saccade turns positive. An eye trace moving negative while the
# head keeps turning is the eye counter-rotating ("saccade and fixate").
# METHODS: all head saccades. Each session's mean trace first (sessions with >= min_ev NaN-free
# windows; eye = LE and RE windows pooled), then mean +- SEM across sessions.

t_h = np.arange(-head_pre, head_post) / FS * 1000
fig, axes = plt.subplots(1, len(groups), figsize=(2.5 * len(groups), 2), squeeze=False)

for ax, group, title in zip(axes[0], groups, titles):
    if not group:
        continue
    head, eye = [], []
    for R in group:
        T = head_triggered_windows(R, HEAD[id(R)], flip_eye, head_pre, head_post)
        hk = np.all(np.isfinite(T["head_pos"]), axis=1)
        if hk.sum() >= min_ev:
            head.append(T["head_pos"][hk].mean(0))
        ev = np.vstack([T[k][hk & np.all(np.isfinite(T[k]), axis=1)] for k in ("LE_pos", "RE_pos")])
        if len(ev) >= min_ev:
            eye.append(ev.mean(0))
    # eye on the left axis, head on the right, as in
    # saccade_andHead_triggered_average
    ax_h = ax.twinx()
    plot_mean_se(ax, t_h, np.array(eye), EYE_COLOR, "eye, sessions", min_n=2)
    plot_mean_se(ax_h, t_h, np.array(head), HEAD_COLOR, "head, sessions", min_n=2)

    ax.axhline(0, color="0.8", lw=0.25)
    ax.axvline(0, color="0.8", lw=0.25)
    
    ax.set_title(title, fontsize=6)
    ax.set_xlabel("time from head onset (ms)")
    ax.set_ylabel("eye rotation (deg)", color=EYE_COLOR)
    ax.tick_params(axis="y", colors=EYE_COLOR)
    ax_h.set_ylabel("head rotation (deg)", color=HEAD_COLOR)
    ax_h.tick_params(axis="y", colors=HEAD_COLOR)

    lines = ax.get_legend_handles_labels()
    lines_h = ax_h.get_legend_handles_labels()
    
    ax.legend(lines[0] + lines_h[0], lines[1] + lines_h[1], fontsize=4)
    fig.tight_layout()

if SAVE_FIGS:
    fig.tight_layout()
    save_fig(fig, "eye_head_B_head_triggered")


# %% B.2 head-onset-triggered averages by head saccade amplitude
# One line per head-amplitude bin (like movement_stats "mean kinematic traces"). Does the eye
# contribute more as head movements get larger? State and eye direction are chosen below.
# METHODS: head saccades in b2_state at head onset. b2_dir splits them by the direction of their first
# paired eye saccade after head onset (earliest eye onset, either eye, within [head onset, head peak];
# same_direction: sign of horizontal eye displacement = sign of head yaw displacement); "all" keeps every
# head saccade, with or without an eye saccade. Session mean per bin (>= min_ev windows), then
# mean +- SEM across sessions.

b2_state = "all"         # "all" or "stationary" (state at head onset)
b2_dir = "with head"     # "with head", "against head" or "all"
head_amp_bins = [0, 40, 60, 80, 360]   # deg, head saccade amplitude (unwrapped)
amp_colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(head_amp_bins) - 1))
head_dirs = {"with head": 1.0, "against head": 0.0}   # same_direction of the first eye saccade after head onset

traces = {}   # (group index, amp bin index) -> dict of head / eye session-mean traces
for gi, group in enumerate(groups):
    for R in group:
        H, e = HEAD[id(R)], E[id(R)]
        T = head_triggered_windows(R, H, flip_eye, head_pre, head_post)
        amp = H["amplitude_deg"].to_numpy(float)
        keep = np.all(np.isfinite(T["head_pos"]), axis=1) & STATE[id(R)][b2_state][H["onset"].to_numpy().astype(int)]
        if b2_dir != "all":
            first = e[e.paired & (e.lag_ms >= 0)].sort_values("onset").groupby("head_idx")["same_direction"].first()
            hdir = np.full(len(H), np.nan)
            hdir[first.index.to_numpy(int)] = first.to_numpy(float)
            keep &= hdir == head_dirs[b2_dir]
        for bi, (lo, hi) in enumerate(zip(head_amp_bins[:-1], head_amp_bins[1:])):
            d = traces.setdefault((gi, bi), {"head": [], "eye": []})
            sel = keep & (amp >= lo) & (amp < hi)
            if sel.sum() >= min_ev:
                d["head"].append(T["head_pos"][sel].mean(0))
            ev = np.vstack([T[k][sel & np.all(np.isfinite(T[k]), axis=1)] for k in ("LE_pos", "RE_pos")])
            if len(ev) >= min_ev:
                d["eye"].append(ev.mean(0))

fig, axes = plt.subplots(2, len(groups), figsize=(2.5 * len(groups), 3.6), squeeze=False,
                         sharex=True, sharey="row")
for ri, sig in enumerate(("head", "eye")):
    for gi, title in enumerate(titles):
        ax = axes[ri, gi]
        for bi, (lo, hi) in enumerate(zip(head_amp_bins[:-1], head_amp_bins[1:])):
            plot_mean_se(ax, t_h, np.array(traces.get((gi, bi), {}).get(sig, [])), amp_colors[bi],
                         f"{lo}-{hi}", min_n=2)
        ax.axhline(0, color="0.8", lw=0.5)
        ax.axvline(0, color="0.8", lw=0.5)
        ax.set_title(title, fontsize=6, color=GROUP_COLORS[gi])
        ax.set_ylabel(f"{sig} rotation (deg)")
        ax.legend(fontsize=4, title="head amp (deg), n sessions", title_fontsize=4)
for ax in axes[-1]:
    ax.set_xlabel("time from head onset (ms)")
fig.suptitle(f"head saccades: {b2_state}, first eye saccade: {b2_dir}", fontsize=7)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, f"eye_head_B2_by_head_amp_{b2_state}_{b2_dir.replace(' ', '_')}")

# %% Wallace Fig 4A / 4C — eye and head position / velocity around eye saccades in turns
# Eye-head dynamics, replicating Wallace, Voit, Martin Machado et al., Kerr lab, Current Biology
# 35:761-775 (Feb 2025), Fig 4, across development. Their term for the return phase is PSCR —
# post-saccadic counter-rotation: the negative eye-velocity lobe after the saccade.
# Rows = groups, columns = eye amplitude classes (amp_classes: HORIZONTAL eye displacement, onset ->
# peak). Every trace is horizontal, zeroed at eye onset, and flipped so the head turns positive
# (head frame: eye + = head +). Head = yaw, eye = horizontal eye; velocities get the same savgol(11, 3).
# METHODS: all eye saccades whose onset is inside a turn (paired with a head saccade or not) with a
# NaN-free window. Each session's mean trace first (>= min_ev saccades for that eye / class), then
# mean +- SEM across sessions.

t_ms = np.arange(-pre, post) / FS * 1000

for kind, unit in (("pos", "rotation (deg)"), ("vel", "velocity (deg/s)")):
    fig, axes = plt.subplots(len(groups), len(amp_classes),
                             figsize=(3 * len(amp_classes), 2 * len(groups)), squeeze=False)
    for row, (group, title) in enumerate(zip(groups, titles)):
        for col, (lo, hi) in enumerate(amp_classes):
            ax = axes[row][col]
            m = {"LE": [], "RE": [], "head": []}   # session-mean traces
            for R in group:
                e, w = E[id(R)], W[id(R)]
                sel = (e.in_turn & e.clean & (e.amp_h >= lo) & (e.amp_h < hi)).to_numpy()
                for eye in ("LE", "RE"):
                    k = sel & (e.eye == eye).to_numpy()
                    if k.sum() >= min_ev:
                        m[eye].append(w[f"eye_{kind}"][k].mean(0))
                if sel.sum() >= min_ev:
                    m["head"].append(w[f"head_{kind}"][sel].mean(0))

            ax_h = ax.twinx()   # eye on the left axis, head on the right
            plot_mean_se(ax, t_ms, np.array(m["LE"]), LE_COLOR, "LE", min_n=2)
            plot_mean_se(ax, t_ms, np.array(m["RE"]), RE_COLOR, "RE", min_n=2)
            plot_mean_se(ax_h, t_ms, np.array(m["head"]), HEAD_COLOR, "head", min_n=2)

            ax_h.set_ylabel(f"head {unit}", color=HEAD_COLOR)
            ax_h.tick_params(axis="y", colors=HEAD_COLOR)
            ax.axhline(0, color="0.8", lw=0.5)
            ax.axvline(0, color="0.8", lw=0.5)
            ax.set_title(f"{title}  eye amp {lo}-{hi} deg  (in turns)", fontsize=6)
            ax.set_xlabel("time from eye onset (ms)")
            ax.set_ylabel(f"eye {unit}")
            lines, lines_h = ax.get_legend_handles_labels(), ax_h.get_legend_handles_labels()
            ax.legend(lines[0] + lines_h[0], lines[1] + lines_h[1], fontsize=4, title="n sessions", title_fontsize=4)
    fig.suptitle(f"pscr_gain: {group_stat['pscr_gain']}", fontsize=6)
    fig.tight_layout()

    if SAVE_FIGS:
        save_fig(fig, f"eye_head_4{'A' if kind == 'pos' else 'C'}_{kind}_in_turns")


# %% Wallace Fig 4D — peak head velocity vs peak counter-rotation eye velocity, per group
# One point per eye saccade in a turn that moved WITH the head (LE and RE pooled), horizontal only,
# in the same flipped frame as 4A/4C (head turns positive):
#   x = peak (positive) head velocity in the pscr_win frames after the eye saccade ends
#   y = peak negative eye velocity in the pscr_win frames after the eye saccade ends
#       (the counter-rotation, not the saccade itself)
# METHODS: black = per-session mean y in 100 deg/s bins of x (head_vel_bins; bins with >= min_n
# saccades), then mean +- SEM across sessions. r over pooled saccades is an effect size only.

fig, axes = plt.subplots(1, len(groups), figsize=(2.6 * len(groups), 2.4), squeeze=False,
                         sharex=True, sharey=True)
centers = (head_vel_bins[:-1] + head_vel_bins[1:]) / 2
for ax, group, title, c in zip(axes[0], groups, titles, GROUP_COLORS):
    xs, ys, bm = [], [], []
    for R in group:
        e = E[id(R)]
        p = e[e.in_turn & e.clean & e.with_head].dropna(subset=["head_peak_vel", "eye_cr_vel"])
        x, y = p.head_peak_vel.to_numpy(), p.eye_cr_vel.to_numpy()
        xs.append(x)
        ys.append(y)
        bm.append([y[(x >= lo) & (x < hi)].mean() if ((x >= lo) & (x < hi)).sum() >= min_n else np.nan
                   for lo, hi in zip(head_vel_bins[:-1], head_vel_bins[1:])])
    x, y, bm = np.concatenate(xs), np.concatenate(ys), np.array(bm)
    n = np.isfinite(bm).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # bins with no sessions stay NaN
        mb, se = np.nanmean(bm, 0), np.nanstd(bm, 0) / np.sqrt(n)
    ax.scatter(x, y, s=1, alpha=0.2, facecolor="white", edgecolor="k")
    ax.errorbar(centers, mb, yerr=se, fmt="o-", ms=3, lw=1, capsize=2, color="k")

    r = np.corrcoef(x, y)[0, 1]
    print(f"{title:15s} n={len(x):5d}  r={r:+.3f}  "
          + "  ".join(f"{cc:.0f}:{mm:.0f} (n={k})" for cc, mm, k in zip(centers, mb, n)))

    ax.set_title(f"{title}  n={len(x)}  r={r:+.2f}  (in turns)", fontsize=6, color=c)
    ax.set_xlabel("peak head velocity (deg/s)")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_ylabel("peak negative eye velocity (deg/s)")
    ax.set_xlim([0, 1000])
    ax.set_ylim([-300, 0])
fig.suptitle(f"cr_vel: {group_stat['cr_vel']}", fontsize=6)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "eye_head_4D_counter_rotation_in_turns")
