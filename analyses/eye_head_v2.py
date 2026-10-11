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
                                       head_triggered_windows, eye_head_coupling, onset_correlogram, running_mask,
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
min_ev = 10              # minimum events for a session to contribute a trace (mean +- SEM across sessions)

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals, "
          f"{sum(len(R.df_LE) + len(R.df_RE) for R in group)} eye / {sum(len(R.df_head) for R in group)} head saccades")


# %% event tables: head saccades, turns, eye saccades with their head / turn context
# Head saccades are R.df_head as extracted by process_session. Two columns are recomputed:
# peak velocity (stored in rad/s) and amplitude, taken from UNWRAPPED yaw at the same
# onset/peak frames — raw yaw jumps by 360 deg at the +-180 seam.
# METHODS: turns = arcs in the 2D body path (utils/find_arcs.find_arcs, default fine + coarse scales,
# tuned on synthetic paths in test_findarcs.py): sustained same-direction path curvature while moving
# > 50 mm/s, >= 0.3 s long, turning the heading >= 30 deg. A frame is in a turn from arc onset to end.
# METHODS: stationary = body speed tracked and not in a running bout (running_mask: speed >
# speed_threshold for >= min_bout frames).
HEAD, ARCS, STATE, E, W = {}, {}, {}, {}, {}
for R in Results:
    df = R.df_head.copy()
    yaw, pitch = unwrap_deg(R.yaw), np.asarray(R.pitch, float)
    on, pk = df["onset"].to_numpy().astype(int), df["peak"].to_numpy().astype(int)
    df["yaw_disp"] = yaw[pk] - yaw[on]
    df["amplitude_deg"] = np.hypot(df["yaw_disp"], pitch[pk] - pitch[on])
    df["peak_velocity_deg_s"] = np.rad2deg(df["peak_velocity_deg_s"].astype(float))
    HEAD[id(R)] = df

    ARCS[id(R)] = find_arcs(R)
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
# METHODS: coupling = % of eye saccades (both eyes) starting inside a head saccade (onset within
# [head onset - pair_window, head peak]) minus chance; head_with_eye_excess = % of head saccades
# containing >= 1 eye onset minus chance. Chance = mean over 100 circular shifts (>= 10 s) of the eye
# onsets against the head, which keeps both rates (eye_head_coupling).
# METHODS: PSCR measures use eye saccades whose onset is inside a turn, with a NaN-free window, that
# moved with the head (sign of horizontal eye displacement = sign of head yaw movement). cr_vel =
# median most-negative eye velocity in the pscr_win frames after the saccade (counter-rotation);
# pscr_gain = median of mean eye / mean head velocity over that window (head > 30 deg/s), -1 = full
# compensation. Sessions with < min_n such saccades are NaN. turn_frac = % of eye saccades in turns.

rows = []
for gi, (group, title) in enumerate(zip(groups, titles)):
    for R in group:
        H, e = HEAD[id(R)], E[id(R)]
        obs_e, chance_e, obs_h, chance_h = eye_head_coupling(R, H, pair_window)
        cr = e[e.in_turn & e.clean & e.with_head]
        ok = len(cr) >= min_n
        rows.append(dict(id=R.id, eo=R.eo, group=title, gi=gi, n_pscr=len(cr),
                         head_rate=len(H) / (np.isfinite(R.yaw).sum() / FS),
                         head_amp=H.amplitude_deg.median(), head_pkv=H.peak_velocity_deg_s.median(),
                         coupling=100 * (obs_e - chance_e), head_with_eye_excess=100 * (obs_h - chance_h),
                         pscr_gain=cr.pscr_gain.median() if ok else np.nan,
                         cr_vel=cr.eye_cr_vel.median() if ok else np.nan,
                         turn_frac=100 * e.in_turn.mean()))
S = pd.DataFrame(rows)
families = dict(head=["head_rate", "head_amp", "head_pkv"],
                coupling=["coupling", "head_with_eye_excess"],
                pscr=["pscr_gain", "cr_vel", "turn_frac"])

# METHODS: EO trend per ferret: Pearson r of session value vs EO, sessions treated as independent
# (ferrets with >= 3 sessions). Reported as the number of ferrets whose r has the sign of the global
# trend (r over all sessions pooled), and how many of those reach p < 0.05. Normal cohort only; in
# compare_delayed mode the only comparison is delayed vs oldest normal sessions (group model below).
# METHODS: global trend across EO groups: linear mixed model (LMM), measure ~ group index (EO bin
# 0, 1, 2; or oldest normal = 0, delayed = 1) as a linear fixed effect, random intercept per ferret.
# β = change per group step. p = likelihood-ratio test against the intercept-only model (both fit
# by ML), chi2 with 1 df, Holm-corrected within each family of measures (head, coupling, pscr).
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
# contribute more as head movements get larger?
# METHODS: session mean per amplitude bin (>= min_ev windows), then mean +- SEM across sessions.

head_amp_bins = [0, 40, 80, 360]   # deg, head saccade amplitude (unwrapped)
amp_colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(head_amp_bins) - 1))

traces = {}   # (group index, amp bin index) -> dict of head / eye session-mean traces
for gi, group in enumerate(groups):
    for R in group:
        T = head_triggered_windows(R, HEAD[id(R)], flip_eye, head_pre, head_post)
        amp = HEAD[id(R)]["amplitude_deg"].to_numpy(float)
        hk = np.all(np.isfinite(T["head_pos"]), axis=1)
        for bi, (lo, hi) in enumerate(zip(head_amp_bins[:-1], head_amp_bins[1:])):
            d = traces.setdefault((gi, bi), {"head": [], "eye": []})
            sel = hk & (amp >= lo) & (amp < hi)
            if sel.sum() >= min_ev:
                d["head"].append(T["head_pos"][sel].mean(0))
            ev = np.vstack([T[k][sel & np.all(np.isfinite(T[k]), axis=1)] for k in ("LE_pos", "RE_pos")])
            if len(ev) >= min_ev:
                d["eye"].append(ev.mean(0))

for sig, ylabel in (("head", "head rotation (deg)"), ("eye", "eye rotation (deg)")):
    fig, axes = plt.subplots(1, len(groups), figsize=(2.5 * len(groups), 2), squeeze=False)
    for gi, (ax, title) in enumerate(zip(axes[0], titles)):
        for bi, (lo, hi) in enumerate(zip(head_amp_bins[:-1], head_amp_bins[1:])):
            arrs = traces.get((gi, bi), {}).get(sig, [])
            plot_mean_se(ax, t_h, np.array(arrs), amp_colors[bi], f"{lo}-{hi}", min_n=2)
        ax.axhline(0, color="0.8", lw=0.5)
        ax.axvline(0, color="0.8", lw=0.5)
        ax.set_title(f"{sig}  {title}", fontsize=6)
        ax.set_xlabel("time from head onset (ms)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=4, title="head amp (deg), n sessions", title_fontsize=4)
    sns.despine(fig)
    fig.tight_layout()

    if SAVE_FIGS:
        fig.tight_layout()
        save_fig(fig, f"eye_head_B2_{sig}_by_head_amp")

# %% D. eye saccade onsets around head saccade onsets (onset cross-correlogram)
# METHODS: for each head saccade, eye onsets (both eyes) are binned by lag = eye onset - head onset
# (bin_ms bins, +-lag_ms; positive = eye followed the head) and divided by the number of head
# saccades: P(eye onset in bin | head onset) (onset_correlogram). Chance (dashed) = the session's eye
# onset rate within frames of that state x bin width. Head saccades are split by state at head onset
# (all / outside turns / stationary). Each session's curve first (sessions with >= min_ev head
# saccades in that state), then mean +- SEM across sessions.

lag_ms, bin_ms = 500, 25
states = ("all", "outside turns", "stationary")
lag_f, bin_f = int(lag_ms / 1000 * FS), int(round(bin_ms / 1000 * FS))

fig, axes = plt.subplots(1, len(states), figsize=(2.6 * len(states), 2.2), squeeze=False, sharey=True)
for ax, state in zip(axes[0], states):
    for group, title, c in zip(groups, titles, GROUP_COLORS):
        obs, exp = [], []
        for R in group:
            m = STATE[id(R)][state]
            H = HEAD[id(R)]
            H = H[m[H["onset"].to_numpy().astype(int)]]
            if len(H) < min_ev:
                continue
            centers_ms, counts, _ = onset_correlogram(R, H, lag_f, bin_f)
            centers_ms, counts = centers_ms[:-1], counts[:-1]   # last histogram bin is closed: one extra lag
            eye_on = np.concatenate([R.df_LE["onset"].to_numpy(), R.df_RE["onset"].to_numpy()]).astype(int)
            obs.append(counts / len(H))
            exp.append(m[eye_on].sum() / m.sum() * bin_f)
        plot_mean_se(ax, centers_ms, np.array(obs), c, title, min_n=2)
        if len(obs) >= 2:
            ax.axhline(np.mean(exp), color=c, lw=0.8, ls="--")
    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_title(f"head saccades: {state}", fontsize=6)
    ax.set_xlabel("eye onset - head onset (ms)")
    ax.legend(fontsize=4, title="n sessions", title_fontsize=4)
axes[0, 0].set_ylabel(f"P(eye saccade onset) per {bin_ms} ms")
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "eye_head_D_onset_correlogram")


# %% Wallace Fig 4A — eye and head rotation, averaged by eye saccade amplitude
# Rows = EO bins, columns = eye amplitude classes (amp_classes: HORIZONTAL eye displacement,
# onset -> peak). Only eye saccades that start inside a head saccade ("paired") and have a
# NaN-free window. Every trace is horizontal, zeroed at eye onset, and flipped so its head
# saccade turns positive (head frame: eye + = head +). Mean +- SE.

t_ms = np.arange(-pre, post) / FS * 1000

fig, axes = plt.subplots(len(groups), len(amp_classes),
                         figsize=(3 * len(amp_classes), 2 * len(groups)), squeeze=False)
for row, (group, title) in enumerate(zip(groups, titles)):
    
    if not group:
        continue
    
    # this EO bin's eye saccades (e) and their traces (w), stacked in the same row order
    e = pd.concat([E[id(R)] for R in group], ignore_index=True)
    w = {k: np.vstack([W[id(R)][k] for R in group]) for k in W[id(group[0])]}

    for col, (lo, hi) in enumerate(amp_classes):
        
        ax = axes[row][col]
        
        sel = (e.paired & e.clean & (e.amp_h >= lo) & (e.amp_h < hi)
               & ((e.loco == loco) | (loco == "all"))).to_numpy()
        
        ax_h = ax.twinx()   # eye on the left axis, head on the right

        plot_mean_se(ax, t_ms, w["eye_pos"][sel & (e.eye == "LE").to_numpy()], LE_COLOR, "LE")
        plot_mean_se(ax, t_ms, w["eye_pos"][sel & (e.eye == "RE").to_numpy()], RE_COLOR, "RE")
        plot_mean_se(ax_h, t_ms, w["head_pos"][sel], HEAD_COLOR, "head")
        
        ax_h.set_ylabel("head rotation (deg)", color=HEAD_COLOR)
        ax_h.tick_params(axis="y", colors=HEAD_COLOR)
        
        ax.axvline(0, color="0.8", lw=0.5)
        ax.set_title(f"{title}  eye amp {lo}-{hi} deg  ({loco})", fontsize=6)
        ax.set_xlabel("time from eye onset (ms)")
        ax.set_ylabel("eye rotation (deg)")
        lines, lines_h = ax.get_legend_handles_labels(), ax_h.get_legend_handles_labels()
        ax.legend(lines[0] + lines_h[0], lines[1] + lines_h[1], fontsize=4)
        
fig.tight_layout()
if SAVE_FIGS:
    fig.tight_layout()
    save_fig(fig, "eye_head_4A_position " + loco)


# %% Wallace Fig 4C — horizontal eye and head velocity, same amplitude classes as 4A
# Same saccades and layout as 4A. Head = yaw velocity, eye = horizontal eye velocity, both
# smoothed with the same savgol(11, 3). The post-saccadic counter-rotation (PSCR) is the
# negative eye-velocity lobe after the saccade.

fig, axes = plt.subplots(len(groups), len(amp_classes),
                         figsize=(3 * len(amp_classes), 2 * len(groups)), squeeze=False)
for row, (group, title) in enumerate(zip(groups, titles)):
    
    if not group:
        continue
    
    # this EO bin's eye saccades (e) and their traces (w), stacked in the same row order
    e = pd.concat([E[id(R)] for R in group], ignore_index=True)
    w = {k: np.vstack([W[id(R)][k] for R in group]) for k in W[id(group[0])]}
    
    for col, (lo, hi) in enumerate(amp_classes):
        
        ax = axes[row][col]
        
        sel = (e.paired & e.clean & (e.amp_h >= lo) & (e.amp_h < hi)
               & ((e.loco == loco) | (loco == "all"))).to_numpy()
        
        ax_h = ax.twinx()   # eye on the left axis, head on the right

        plot_mean_se(ax, t_ms, w["eye_vel"][sel & (e.eye == "LE").to_numpy()], LE_COLOR, "LE")
        plot_mean_se(ax, t_ms, w["eye_vel"][sel & (e.eye == "RE").to_numpy()], RE_COLOR, "RE")
        plot_mean_se(ax_h, t_ms, w["head_vel"][sel], HEAD_COLOR, "head")

        ax_h.set_ylabel("head velocity (deg/s)", color=HEAD_COLOR)
        ax_h.tick_params(axis="y", colors=HEAD_COLOR)
        ax.axhline(0, color="0.8", lw=0.5)
        ax.axvline(0, color="0.8", lw=0.5)
        ax.set_title(f"{title}  eye amp {lo}-{hi} deg  ({loco})", fontsize=6)
        ax.set_xlabel("time from eye onset (ms)")
        ax.set_ylabel("eye velocity (deg/s)")
        lines, lines_h = ax.get_legend_handles_labels(), ax_h.get_legend_handles_labels()
        ax.legend(lines[0] + lines_h[0], lines[1] + lines_h[1], fontsize=4)

if SAVE_FIGS:
    sns.despine(fig)
    fig.tight_layout()
    save_fig(fig, "eye_head_4C_velocity " + loco)


# %% Wallace Fig 4D — peak head velocity vs peak counter-rotation eye velocity, per EO bin
# One point per paired eye saccade that moved WITH its head saccade (LE and RE pooled),
# horizontal only, in the same flipped frame as 4A/4C (head turns positive):
#   x = peak (positive) head velocity from eye onset to the end of the window
#   y = peak negative eye velocity in the pscr_win frames after the eye saccade ends
#       (the counter-rotation, not the saccade itself)
# Black = mean +- SD of y in 100 deg/s bins of x (head_vel_bins).

fig, axes = plt.subplots(1, len(groups), figsize=(2.6 * len(groups), 2.4), squeeze=False,
                         sharex=True, sharey=True)
for ax, group, title, c in zip(axes[0], groups, titles, AGE_COLORS):
    if not group:
        continue
    e = pd.concat([E[id(R)] for R in group], ignore_index=True)
    p = e[e.paired & e.clean & (e.same_direction == 1)
          & ((e.loco == loco) | (loco == "all"))].dropna(subset=["head_peak_vel", "eye_cr_vel"])
    x = p.head_peak_vel.to_numpy()
    y = p.eye_cr_vel.to_numpy()

    ax.scatter(x, y, s=1, alpha=0.2, color=c)

    centers, means, sds = [], [], []
    for lo, hi in zip(head_vel_bins[:-1], head_vel_bins[1:]):
        yb = y[(x >= lo) & (x < hi)]
        if len(yb) > 5:
            centers.append((lo + hi) / 2)
            means.append(yb.mean())
            sds.append(yb.std())
    ax.errorbar(centers, means, yerr=sds, fmt="o-", ms=3, lw=1, capsize=2, color="k")

    r = np.corrcoef(x, y)[0, 1]
    print(f"{title:10s} n={len(x):5d}  r={r:+.3f}  "
          + "  ".join(f"{cc:.0f}:{mm:.0f}" for cc, mm in zip(centers, means)))

    ax.set_title(f"{title}  n={len(x)}  r={r:+.2f}  ({loco})", fontsize=6)
    ax.set_xlabel("peak head velocity (deg/s)")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_ylabel("peak negative eye velocity (deg/s)")
    # ax.set_xscale("log")
    ax.set_xlim([0, 1000])
    ax.set_ylim([-300, 0])

if SAVE_FIGS:
    sns.despine(fig)
    fig.tight_layout()
    save_fig(fig, "eye_head_4D_counter_rotation")


# %% S1. head saccade distributions per EO bin

fig, axes = plt.subplots(1, 3, figsize=(8, 2))

for group, title, c in zip(groups, titles, AGE_COLORS):
    if not group:
        continue
    H = pd.concat([HEAD[id(R)] for R in group])
    if len(H) < 20:
        continue
    dur = (H.peak - H.onset).to_numpy(float) / FS * 1000
    for ax, v, lbl in zip(axes, (H.amplitude_deg.to_numpy(float),
                                 H.peak_velocity_deg_s.to_numpy(float), dur),
                          ("amplitude (deg)", "peak velocity (deg/s)", "duration (ms)")):
        sns.histplot(ax=ax, x=v, bins=40, element="step", fill=False, stat="density",
                     color=c, label=title)
        ax.set_xlabel(lbl)
    print(f"{title:10s} n={len(H):6d}  amp={H.amplitude_deg.median():5.1f}  "
          f"pkv={H.peak_velocity_deg_s.median():6.1f}  dur={np.median(dur):5.0f} ms")

axes[0].legend(fontsize=5)
sns.despine(fig)
fig.tight_layout()


# %% S2. Wallace Fig 4B — amplitude vs peak velocity, head and eye

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))

for group, title, c in zip(groups, titles, AGE_COLORS):
    if not group:
        continue
    H = pd.concat([HEAD[id(R)] for R in group])
    e = pd.concat([E[id(R)] for R in group], ignore_index=True)
    p = e[e.paired]
    axes[0].scatter(H.amplitude_deg, H.peak_velocity_deg_s, s=1, alpha=0.2, color=c,
                    label=title)
    axes[1].scatter(p.amplitude_deg, p.peak_velocity_deg_s, s=1, alpha=0.1, color=c,
                    label=title)

for ax, name in zip(axes, ("head saccades", "eye saccades in a head saccade")):
    ax.set_xlabel("amplitude (deg)")
    ax.set_ylabel("peak velocity (deg/s)")
    ax.set_title(name, fontsize=6)
axes[0].legend(fontsize=4, markerscale=4)
sns.despine(fig)
fig.tight_layout()


# %% S3. eye-head timing categories
# The knobs are here, not in helper_functions, so a boundary can be moved and the effect
# seen immediately. Unpaired = the eye saccade is not inside any head saccade.
# Read the head rate column before comparing percentages: head saccade rate varies across
# sessions and drives occupancy directly, and age and head rate are themselves correlated.
# Panel C (coupling vs chance) is the rate-corrected version.

lead_ms = 50       # |lag| above this = one leads; below = synchronous

rows = []
for R in Results:
    e = E[id(R)]
    lag = e.lag_ms
    rows.append(dict(ferret=R.id, eo=R.eo, n=len(e),
                     head_rate=len(HEAD[id(R)]) / (len(R.LE_vx) / FS),
                     eye_leads=100 * (e.paired & (lag < -lead_ms)).mean(),
                     sync=100 * (e.paired & (lag.abs() <= lead_ms)).mean(),
                     head_leads=100 * (e.paired & (lag > lead_ms)).mean(),
                     unpaired=100 * (~e.paired).mean()))

S = pd.DataFrame(rows)
print(S.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print(f"\ncorr(head_rate, %unpaired) = {S.head_rate.corr(S.unpaired):+.3f}")
print(f"corr(EO, head_rate)        = {S.eo.corr(S.head_rate):+.3f}")
print(f"corr(EO, %sync)            = {S.eo.corr(S['sync']):+.3f}")

fig, axes = plt.subplots(1, 3, figsize=(9, 2.4))

# per-session occupancy, so outliers stay visible instead of averaging away
S.set_index([S.ferret.astype(str) + " EO" + S.eo.astype(str)])[
    ["eye_leads", "sync", "head_leads", "unpaired"]].plot.bar(
    stacked=True, ax=axes[0], width=0.85, legend=True)
axes[0].set_ylabel("% of eye saccades")
axes[0].tick_params(axis="x", labelsize=4, rotation=90)
axes[0].legend(fontsize=4)

# the confound itself: occupancy against head rate, colored by age
sc = axes[1].scatter(S.head_rate, S.unpaired, s=14, c=S.eo, cmap="viridis")
axes[1].set_xlabel("head saccade rate (/s)")
axes[1].set_ylabel("% unpaired")
fig.colorbar(sc, ax=axes[1], label="EO")

# lag distribution of paired eye saccades per EO group, with the lead_ms boundaries drawn
for group, title, c in zip(groups, titles, AGE_COLORS):
    if not group:
        continue
    lag = pd.concat([E[id(R)] for R in group])["lag_ms"].dropna()
    if len(lag) < 20:
        continue
    sns.histplot(ax=axes[2], x=lag.to_numpy(), bins=40, element="step", fill=False,
                 stat="density", color=c, label=f"{title} (n={len(lag)})")
for b in (-lead_ms, lead_ms):
    axes[2].axvline(b, color="0.6", ls=":", lw=0.5)
axes[2].set_xlabel("eye onset - head onset (ms)")
axes[2].legend(fontsize=4)

sns.despine(fig)
fig.tight_layout()

E_all = pd.concat(list(E.values()), ignore_index=True)
lag = E_all["lag_ms"]
E_all["category"] = np.where(~E_all.paired, "unpaired",
                     np.where(lag < -lead_ms, "eye_leads",
                      np.where(lag > lead_ms, "head_leads", "synchronous")))
print("\ncategory kinematics:")
print(E_all.groupby("category").agg(
    n=("amplitude_deg", "size"), med_amp=("amplitude_deg", "median"),
    med_pkv=("peak_velocity_deg_s", "median"),
    frac_same=("same_direction", "mean")).to_string(float_format=lambda v: f"{v:.2f}"))


# %% S4. where in the head movement do eye saccades start?
# phase 0 = head onset, 1 = head saccade end (peak); negative = the eye started first.

bins = np.linspace(-0.6, 1, 33)
fig, axes = plt.subplots(1, len(groups), figsize=(2.5 * len(groups), 2), squeeze=False)

for ax, group, title in zip(axes[0], groups, titles):
    if not group:
        continue
    e = pd.concat([E[id(R)] for R in group], ignore_index=True)
    p = e[e.paired]
    for val, lbl, c in ((1, "with head", EYE_COLOR), (0, "against head", "0.5")):
        ph = p.phase[p.same_direction == val].to_numpy(float)
        sns.histplot(ax=ax, x=ph, bins=bins, element="step", fill=False, stat="count",
                     color=c, label=f"{lbl} (n={len(ph)})")
    ax.set_title(title, fontsize=6)
    ax.set_xlabel("phase in head saccade")
    ax.legend(fontsize=4)

sns.despine(fig)
fig.tight_layout()

#
# %%
