# %% main script to run data loading, cleaning, and saccade extraction for a session
#
# eye + head "main sequence" analysis
# Analyzes the relationship between eye and head movements (positive amplitude/velocity), including saccades and turns.
#
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
                                       head_triggered_windows, running_mask, eye_signal, session_rate,
                                       pooled_intervals, plot_mean_se, plot_vs_eo, FERRET_COLORS)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = True   # True: normal EO 8-20 vs. all delayed sessions
fig_tag = " delayed" if compare_delayed else ""   # appended to saved figure names

# OTHER SETTINGS
# Frame: flip_eye = "LE" is the HEAD frame — eye + = head yaw +, and world gaze = yaw + eye
# (checked against the pipeline's gaze). With "RE" the eye axis points the other way and a
# counter-rotation would read as positive.
flip_eye = "LE"
pair_window = 30         # frames (250 ms) an eye saccade may lead its head saccade
pre, post = 24, 72       # frames: -200 to +600 ms from eye-saccade onset
pscr_win = 24            # frames (200 ms) after the eye saccade ends: counter-rotation window
head_pre, head_post = 24, 144   # frames: -200 to +1200 ms from head onset (panel B)
b3_min_amp = 20          # deg |head yaw displacement|: head saccades used in B.3
b3_min_eye = 2           # deg |horizontal eye displacement|: eye saccades counted in B.3
b3_win = (-60, 180)      # frames: -500 to +1500 ms from head onset (B.3)
b3_bw = 6                # frames (50 ms) per B.3 rate bin
b3_n_shift = 20          # circular shifts (>= 10 s) of the eye onsets for the B.3 chance rate
b3_min_head = 10         # minimum head saccades for a session's B.3 rate
amp_classes = [(5, 10), (10, 15), (15, np.inf)]   # Wallace Fig 4A/4C classes, horizontal deg
speed_threshold = 20     # mm/s, running_mask: stationary = not in a running bout
min_bout = 30            # frames
min_n = 3                # minimum saccades for a session's per-session PSCR measures
min_ev = 3               # minimum events for a session to contribute a trace (mean +- SEM across sessions)
states = ("all", "outside turns", "stationary")   # head saccades split by state at head onset (D)
gaze_state = "all"       # "all" or "stationary": gaze shifts by state at gaze onset (rate / ISI time too)
head_still_deg = 2       # |head yaw displacement| below this = eye-only gaze shift
gaze_amp_bins = [0, 10, 20, 40, 360]   # deg, gaze shift amplitude (unwrapped)
yaw_jump = 10            # deg/frame: a larger single-frame yaw step is a head-pose flip (event excluded)

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
# Gaze shifts are R.df_LEgaze / R.df_REgaze as extracted by process_session. Amplitude is recomputed
# from UNWRAPPED gaze (horizontal gaze wraps at +-180). Displacements are horizontal, onset -> peak, in
# the head frame (gaze = yaw + eye). LE pipeline gaze has the opposite horizontal sign to yaw + eye
# (checked: r = -0.97 vs +0.97 for RE), so it is flipped here.
# METHODS: head-pose flips (single-frame unwrapped yaw step > yaw_jump deg, with roll flipping by ~180
# deg) are tracking artifacts. Head saccades and gaze shifts with a flip in [onset - 5, peak + 5] frames
# are dropped; eye-saccade and head-triggered windows containing a flip are not clean (helpers).
HEAD, STATE, E, W, G = {}, {}, {}, {}, {}
for R in Results:
    df = R.df_head.copy()
    yaw, pitch = unwrap_deg(R.yaw), np.asarray(R.pitch, float)
    on, pk = df["onset"].to_numpy().astype(int), df["peak"].to_numpy().astype(int)
    df["yaw_disp"] = yaw[pk] - yaw[on]
    df["amplitude_deg"] = np.hypot(df["yaw_disp"], pitch[pk] - pitch[on])
    df["peak_velocity_deg_s"] = np.rad2deg(df["peak_velocity_deg_s"].astype(float))
    jump = np.abs(np.diff(yaw)) > yaw_jump   # head-pose flips
    HEAD[id(R)] = df.loc[[not jump[max(o - 5, 0):p + 5].any() for o, p in zip(on, pk)]].reset_index(drop=True)

    turn = np.zeros(len(R.yaw), bool)
    for s, e in zip(ARCS[id(R)]["onset"], ARCS[id(R)]["end"]):
        turn[s:e] = True
    STATE[id(R)] = {"all": np.ones(len(turn), bool), "outside turns": ~turn,
                    "stationary": np.isfinite(np.asarray(R.speed, float)) & ~running_mask(R, speed_threshold, min_bout)}

    # one row per eye saccade (E) and its onset-aligned traces (W, same row order)
    e, W[id(R)] = head_eye_windows(R, HEAD[id(R)], flip_eye, pre, post, pair_window, pscr_win, yaw_jump)
    e["in_turn"] = turn[e["onset"].to_numpy()]
    e["with_head"] = np.sign(e["eye_disp"]) == e["head_sign"]   # eye moved the way the head turned (paired or not)
    E[id(R)] = e

    # one row per gaze shift (G), LE and RE pooled
    gaze_rows = []
    for eye, df in (("LE", R.df_LEgaze), ("RE", R.df_REgaze)):
        g_on, g_pk = df["onset"].to_numpy().astype(int), df["peak"].to_numpy().astype(int)
        gaze_h = unwrap_deg(getattr(R, f"{eye}_gaze_horizontal_deg")) * (-1 if eye == "LE" else 1)
        gaze_v = np.asarray(getattr(R, f"{eye}_gaze_vertical_deg"), float)
        eye_x = eye_signal(R, eye, "x", flip_eye)
        sacc_on = np.sort((R.df_LE if eye == "LE" else R.df_RE)["onset"].to_numpy().astype(int))
        g = pd.DataFrame(dict(onset=g_on, peak=g_pk, eye=eye, gaze_disp=gaze_h[g_pk] - gaze_h[g_on],
                              head_disp=yaw[g_pk] - yaw[g_on], eye_disp=eye_x[g_pk] - eye_x[g_on],
                              peak_velocity_deg_s=df["peak_velocity_deg_s"].to_numpy(float),
                              # an eye saccade (same eye) starts between gaze onset and peak
                              with_eye_sacc=np.searchsorted(sacc_on, g_pk, "right") > np.searchsorted(sacc_on, g_on)))
        g["amplitude_deg"] = np.hypot(g.gaze_disp, gaze_v[g_pk] - gaze_v[g_on])
        g["direction_deg"] = np.degrees(np.arctan2(gaze_v[g_pk] - gaze_v[g_on], g.gaze_disp))
        gaze_rows.append(g)
    g = pd.concat(gaze_rows, ignore_index=True)
    g = g.loc[[not jump[max(o - 5, 0):p + 5].any() for o, p in zip(g.onset, g.peak)]].reset_index(drop=True)
    g["head_contrib"] = g.head_disp / g.gaze_disp   # head share of the shift: 1 = all head, 0 = all eye
    g["eye_only"] = (g.head_disp.abs() < head_still_deg).where(g.head_disp.notna())
    g["in_turn"] = turn[g.onset.to_numpy()]
    g["stationary"] = STATE[id(R)]["stationary"][g.onset.to_numpy()]
    G[id(R)] = g

for group, title in zip(groups, titles):
    print(f"{title}: yaw jump > {yaw_jump} deg/frame removed "
          f"{sum(len(R.df_head) - len(HEAD[id(R)]) for R in group)} head saccades, "
          f"{sum(len(R.df_LEgaze) + len(R.df_REgaze) - len(G[id(R)]) for R in group)} gaze shifts")


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
        T = head_triggered_windows(R, HEAD[id(R)], flip_eye, head_pre, head_post, yaw_jump)
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
    save_fig(fig, "eye_head_B_head_triggered" + fig_tag)


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
        T = head_triggered_windows(R, H, flip_eye, head_pre, head_post, yaw_jump)
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
    save_fig(fig, f"eye_head_B2_by_head_amp_{b2_state}_{b2_dir.replace(' ', '_')}" + fig_tag)


# %% B.3 eye-saccade rate around head onset, above chance
# When do eye saccades follow a head movement? With-head saccades are the gaze shift; against-head
# saccades are mostly the returns of back-and-forth chains. The first-saccade lag (lag_ms) is not used:
# it shortens whenever the saccade rate rises, even at chance.
# METHODS: head saccades with |unwrapped yaw displacement| >= b3_min_amp deg; eye saccades of both eyes with
# |horizontal displacement| >= b3_min_eye deg, with-head = same sign as the head yaw displacement. Rate =
# eye onsets per b3_bw bin around head onset, per head saccade per eye (s^-1). Chance = the same rate with
# the eye onsets circularly shifted against the head by random offsets >= 10 s (mean of b3_n_shift
# shifts, as in eye_head_coupling); excess = rate - chance. Sessions with >= b3_min_head head saccades.
# burst_with_ms / burst_against_ms = centre of mass of the positive excess in [-100, 400) / [200, 700) ms;
# burst_with / burst_against = excess saccades per head saccade per eye in [0, 500) / [200, 700) ms.
# Exploration: the burst timing did not depend on running speed before head onset.

b3_edges = np.arange(b3_win[0], b3_win[1] + 1, b3_bw)
t_b3 = (b3_edges[:-1] + b3_edges[1:]) / 2 / FS * 1000
b3_measures = ["burst_with_ms", "burst_against_ms", "burst_with", "burst_against"]
B3 = {}   # id(R) -> excess rate traces ("with", "against") and the session measures
rng = np.random.default_rng(0)
for R in Results:
    H, e = HEAD[id(R)], E[id(R)]
    H = H[H.yaw_disp.abs() >= b3_min_amp]
    e = e[e.eye_disp.abs() >= b3_min_eye]
    if len(H) < b3_min_head or not len(e):
        continue
    h_on, h_sign = H.onset.to_numpy(int), np.sign(H.yaw_disp.to_numpy())
    e_on, e_with = e.onset.to_numpy(int), np.sign(e.eye_disp.to_numpy())[None, :] == h_sign[:, None]
    n, n_eyes = len(R.LE_vx), e.eye.nunique()

    def b3_rate(onsets):
        lag = onsets[None, :] - h_on[:, None]   # head x eye, frames
        return {d: np.histogram(lag[e_with == (d == "with")], b3_edges)[0] / len(h_on) / n_eyes / (b3_bw / FS)
                for d in ("with", "against")}

    real = b3_rate(e_on)
    chance = [b3_rate((e_on + s) % n) for s in rng.integers(int(10 * FS), n - int(10 * FS), b3_n_shift)]
    ex = {d: real[d] - np.mean([c[d] for c in chance], axis=0) for d in real}

    def centre(x, lo, hi):
        k = (t_b3 >= lo) & (t_b3 < hi)
        w = np.clip(x[k], 0, None)
        return (t_b3[k] * w).sum() / w.sum() if w.sum() > 0 else np.nan

    B3[id(R)] = dict(ex, burst_with_ms=centre(ex["with"], -100, 400), burst_against_ms=centre(ex["against"], 200, 700),
                     burst_with=ex["with"][(t_b3 >= 0) & (t_b3 < 500)].sum() * b3_bw / FS,
                     burst_against=ex["against"][(t_b3 >= 200) & (t_b3 < 700)].sum() * b3_bw / FS)

fig, axes = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
for ax, d in zip(axes, ("with", "against")):
    for gi, (group, title) in enumerate(zip(groups, titles)):
        plot_mean_se(ax, t_b3, np.array([B3[id(R)][d] for R in group if id(R) in B3]), GROUP_COLORS[gi], title, min_n=2)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_title(f"{d}-head eye saccades", fontsize=6)
    ax.set_xlabel("time from head onset (ms)")
axes[0].set_ylabel("rate above chance\n(/s per eye per head saccade)")
axes[0].legend(fontsize=4)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "eye_head_B3_rate_above_chance" + fig_tag)

# %% Wallace Fig 4A / 4C — eye and head position / velocity around eye saccades in turns
# Eye-head dynamics, replicating Wallace, Voit, Martin Machado et al., Kerr lab, Current Biology
# 35:761-775 (Feb 2025), Fig 4, across development. Their term for the return phase is PSCR —
# post-saccadic counter-rotation: the negative eye-velocity lobe after the saccade.
# One panel per group; eye amplitude classes (amp_classes: HORIZONTAL eye displacement, onset -> peak)
# side by side, each in its own -200..600 ms window, with scale bars instead of axes. Every trace is horizontal, zeroed at eye onset, and flipped so the head turns positive
# (head frame: eye + = head +). Head = yaw, eye = horizontal eye; velocities get the same savgol(11, 3).
# METHODS: all eye saccades whose onset is inside a turn (paired with a head saccade or not) with a
# NaN-free window, that moved with the head (with_head; against-head saccades would cancel in the
# mean). Each session's mean trace first (>= min_ev saccades for that eye / class), then mean +- SEM
# across sessions.

t_ms = np.arange(-pre, post) / FS * 1000

seg_ms = t_ms[-1] - t_ms[0] + 200   # each amplitude class shifted by one window + 200 ms gap
bars = {"pos": (20, 40, "deg"), "vel": (100, 100, "deg/s")}   # scale bar sizes: eye, head, unit (vel 1:1, gain readable)
bar_ms = 250

for kind, unit in (("pos", "rotation (deg)"), ("vel", "velocity (deg/s)")):
    
    fig, axes = plt.subplots(1, len(groups), figsize=(2 * len(groups), 2), squeeze=False)
    panels = []
    
    for ax, group, title, c in zip(axes[0], groups, titles, GROUP_COLORS):
        ax_h = ax.twinx()   # eye on the left axis, head on the right
        ns = []
        for ci, (lo, hi) in enumerate(amp_classes):
            m = {"LE": [], "RE": [], "head": []}   # session-mean traces
            for R in group:
                e, w = E[id(R)], W[id(R)]
                sel = (e.in_turn & e.clean & e.with_head & (e.amp_h >= lo) & (e.amp_h < hi)).to_numpy()
                for eye in ("LE", "RE"):
                    k = sel & (e.eye == eye).to_numpy()
                    if k.sum() >= min_ev:
                        m[eye].append(w[f"eye_{kind}"][k].mean(0))
                if sel.sum() >= min_ev:
                    m["head"].append(w[f"head_{kind}"][sel].mean(0))

            t = t_ms + ci * seg_ms
            plot_mean_se(ax, t, np.array(m["LE"]), LE_COLOR, "_LE", min_n=2)
            plot_mean_se(ax, t, np.array(m["RE"]), RE_COLOR, "_RE", min_n=2)
            plot_mean_se(ax_h, t, np.array(m["head"]), HEAD_COLOR, "_head", min_n=2)
            ax.axvline(ci * seg_ms, color="0.8", lw=0.5)
            ax.plot([t[0], t[-1]], [0, 0], color="0.8", lw=0.5)
            ns.append((ci * seg_ms, f"{lo}-{hi} deg\nn = {len(m['LE'])} / {len(m['RE'])} / {len(m['head'])}"))

        panels.append((ax, ax_h, ns, title, c))

    # shared y limits across groups, head axis scaled by hb / eb: zeros aligned and one bar serves both
    eb, hb, u = bars[kind]
    ylim = [min(min(p[0].get_ylim()[0], p[1].get_ylim()[0] * eb / hb) for p in panels),
            max(max(p[0].get_ylim()[1], p[1].get_ylim()[1] * eb / hb) for p in panels)]
    x0 = t_ms[0] - 120   # scale bars left of the first window
    
    for ax, ax_h, ns, title, c in panels:
        ax.set_ylim(ylim)
        ax_h.set_ylim([v * hb / eb for v in ylim])
        ax.plot([x0, x0, x0 + bar_ms], [ylim[0] + eb, ylim[0], ylim[0]], color="k", lw=1.5, clip_on=False)
        ax.text(x0 - 20, ylim[0] + eb / 2, f"eye {eb} {u}", fontsize=5, ha="right", va="center", rotation=90)
        ax.text(x0 - 110, ylim[0] + eb / 2, f"head {hb} {u}", fontsize=5, color=HEAD_COLOR,
                ha="right", va="center", rotation=90)
        ax.text(x0 + bar_ms / 2, ylim[0], f"{bar_ms} ms", fontsize=5, ha="center", va="top")
        for x, txt in ns:
            ax.text(x, ylim[1], txt, fontsize=5, ha="center", va="bottom")
        ax.axis("off")
        ax_h.axis("off")
        # ax.set_title(f"{title}  (in turns, with head)\n", fontsize=6, color=c)
        # ax.legend(handles=[plt.Line2D([], [], color=col, label=lab) for col, lab in
        #                    ((LE_COLOR, "LE"), (RE_COLOR, "RE"), (HEAD_COLOR, "head"))],
        #           fontsize=4, loc="upper right", title="n sessions: LE / RE / head", title_fontsize=4, frameon=False)
    fig.tight_layout()

    if SAVE_FIGS:
        save_fig(fig, f"eye_head_4{'A' if kind == 'pos' else 'C'}_{kind}_in_turns_with_head" + fig_tag)


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

head_vel_bins = np.arange(50, 500, 50)
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

sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "eye_head_4D_counter_rotation_in_turns" + fig_tag)





# %% gaze distributions: amplitude/speed, frequency/ISI (like in saccade rate/ISI)
# basic properites before looking at triggered average stats.
# in addition, should look at contribution of eye and head for each as a distribution
# Figures as in main_sequence_eye_kinematics_v2.py; tests are on session values in the per-session cell.
# METHODS: gaze shifts with onset in gaze_state, LE and RE pooled (event tables cell; amplitude and
# direction from unwrapped gaze, head frame). Distributions pool shifts within a group (descriptive).
# Main sequence: per-session OLS of log10 peak velocity on log10 amplitude (>= min_n shifts), one line
# per session. ISI = position peak of one shift -> onset of the next, NaN-free gaps (pooled_intervals);
# rate = session_rate, mean +- SD over sessions. Eye / head contribution: head_contrib = head yaw /
# horizontal gaze displacement; head and eye displacements sign-aligned so gaze moves positive.

gaze_in_state = {id(R): G[id(R)][STATE[id(R)][gaze_state][G[id(R)].onset.to_numpy()]] for R in Results}
group_gaze = [pd.concat([gaze_in_state[id(R)] for R in group]) for group in groups]

# amplitude & peak velocity distributions
fig, axes = plt.subplots(1, 2, figsize=(6, 2))
for g, title, c in zip(group_gaze, titles, GROUP_COLORS):
    g = g[(g.amplitude_deg > 0) & (g.peak_velocity_deg_s > 0)]
    sns.histplot(ax=axes[0], x=np.log10(g.amplitude_deg), bins=80, element="step", fill=False,
                 stat="density", color=c, label=title)
    sns.histplot(ax=axes[1], x=np.log10(g.peak_velocity_deg_s), bins=80, element="step", fill=False,
                 stat="density", color=c, label=title)
axes[0].set_xlabel("log10 gaze amplitude (deg)")
axes[1].set_xlabel("log10 gaze peak velocity (deg/s)")
axes[0].legend(fontsize=4)
fig.suptitle(f"gaze shifts: {gaze_state}", fontsize=7)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, f"eye_head gaze distributions amplitude and velocity {gaze_state}" + fig_tag)

# inter-shift interval (ISI) distribution and rate per group
isi_bins = np.logspace(np.log10(10), np.log10(10000), 30)
fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(6, 2))
for i, (group, title, c) in enumerate(zip(groups, titles, GROUP_COLORS)):
    sns.histplot(ax=ax, x=pooled_intervals(group, "gaze", gaze_state, speed_threshold, min_bout),
                 bins=isi_bins, element="step", fill=False, stat="probability", color=c, label=title)
    rate = [session_rate(R, "gaze", gaze_state, speed_threshold, min_bout) for R in group]
    ax_r.errorbar(i, np.nanmean(rate), yerr=np.nanstd(rate), fmt="o", ms=5, capsize=3, color=c)
ax.set_xscale("log")
ax.set_xlabel("inter-gaze-shift interval (ms)")
ax.legend(fontsize=4)
ax_r.set_xticks(range(len(titles)), titles)
ax_r.set_xlim(-0.5, len(titles) - 0.5)
ax_r.set_ylabel("gaze shift rate (Hz)")
fig.suptitle(f"gaze shifts: {gaze_state}", fontsize=7)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, f"eye_head gaze intervals and rate {gaze_state}" + fig_tag)

# main sequence fits (one line per session, coloured by ferret)
fig, axes = plt.subplots(1, len(groups), figsize=(2.5 * len(groups), 2.2), squeeze=False, sharex=True, sharey=True)
for ax, group, title, c in zip(axes[0], groups, titles, GROUP_COLORS):
    for R in group:
        g = gaze_in_state[id(R)]
        g = g[(g.amplitude_deg > 0) & (g.peak_velocity_deg_s > 0)]
        if len(g) < min_n:
            continue
        x, y = np.log10(g.amplitude_deg), np.log10(g.peak_velocity_deg_s)
        slope, intercept = np.polyfit(x, y, 1)
        xl = np.array([x.min(), x.max()])
        ax.plot(xl, slope * xl + intercept, lw=0.6, alpha=0.6, color=FERRET_COLORS[R.id])
    ax.set_title(title, fontsize=6, color=c)
    ax.set_xlabel("log10 gaze amplitude (deg)")
    ax.set_ylabel("log10 gaze peak velocity (deg/s)")
axes[0, 0].legend(handles=[plt.Line2D([], [], color=FERRET_COLORS[i], label=f"F{i}")
                           for i in sorted({R.id for group in groups for R in group})], fontsize=4)
fig.suptitle(f"gaze shifts: {gaze_state}", fontsize=7)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, f"eye_head gaze main sequence fits {gaze_state}" + fig_tag)

# direction per group (head frame, both eyes in the same convention)
dir_bins = np.linspace(-np.pi, np.pi, 37)
fig, axes = plt.subplots(1, len(groups), figsize=(2.2 * len(groups), 2.4), squeeze=False,
                         subplot_kw=dict(projection="polar"))
for ax, g, title, c in zip(axes[0], group_gaze, titles, GROUP_COLORS):
    th = np.deg2rad(g.direction_deg.to_numpy(float))
    h, _ = np.histogram(th[np.isfinite(th)], dir_bins)
    ax.bar(dir_bins[:-1], h / h.sum(), width=np.diff(dir_bins), align="edge", color=c)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=6)
fig.suptitle(f"gaze shift direction: {gaze_state}", fontsize=7)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, f"eye_head gaze direction {gaze_state}" + fig_tag)

# eye / head contribution per gaze shift
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2))
for g, title, c in zip(group_gaze, titles, GROUP_COLORS):
    sign = np.sign(g.gaze_disp)
    for ax, x, bins in ((axes[0], g.head_contrib, np.linspace(-1, 2, 61)),
                        (axes[1], g.head_disp * sign, np.linspace(-40, 100, 71)),
                        (axes[2], g.eye_disp * sign, np.linspace(-30, 30, 61))):
        sns.histplot(ax=ax, x=x, bins=bins, element="step", fill=False, stat="density", color=c, label=title)
for ax in axes:
    ax.axvline(0, color="0.8", lw=0.5)
axes[0].axvline(1, color="0.8", lw=0.5)
axes[0].set_xlabel("head share of gaze shift (head_contrib)")
axes[1].set_xlabel("head displacement (deg, gaze +)")
axes[2].set_xlabel("eye displacement (deg, gaze +)")
axes[0].legend(fontsize=4)
fig.suptitle(f"gaze shifts: {gaze_state}", fontsize=7)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, f"eye_head gaze eye and head contribution {gaze_state}" + fig_tag)


# %% C. gaze-shift-triggered averages by gaze amplitude: gaze, head and eye (head frame)
# Does the head carry more of larger gaze shifts? One line per gaze-amplitude bin, as in B.2.
# METHODS: gaze shifts with onset in gaze_state, LE and RE pooled, with a NaN-free window [onset - pre,
# onset + post); traces zeroed at gaze onset and sign-aligned so gaze moves positive (gaze = yaw + eye,
# head frame). Session mean per gaze_amp_bins bin (>= min_ev shifts), then mean +- SEM across sessions.
# Second figure: session median head_contrib per bin (>= min_n shifts), then mean +- SEM across sessions.

t_g = np.arange(-pre, post) / FS * 1000
gaze_amp_colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(gaze_amp_bins) - 1))
gaze_traces = {}   # (group index, amp bin index) -> dict of gaze / head / eye session-mean traces
head_share = {}    # group index -> sessions x bins, session median head_contrib
for gi, group in enumerate(groups):
    head_share[gi] = []
    for R in group:
        g = G[id(R)]
        yaw = unwrap_deg(R.yaw)
        onset = g.onset.to_numpy()
        win = {k: np.full((len(g), pre + post), np.nan) for k in ("gaze", "head", "eye")}
        for eye in ("LE", "RE"):
            rows = np.flatnonzero((g.eye == eye).to_numpy() & (onset >= pre) & (onset + post <= len(yaw)))
            idx = onset[rows, None] + np.arange(-pre, post)
            gaze_h = unwrap_deg(getattr(R, f"{eye}_gaze_horizontal_deg")) * (-1 if eye == "LE" else 1)
            for k, x in (("gaze", gaze_h), ("head", yaw), ("eye", eye_signal(R, eye, "x", flip_eye))):
                win[k][rows] = x[idx] - x[onset[rows], None]
        sign = np.sign(g.gaze_disp.to_numpy())
        for k in win:
            win[k] *= sign[:, None]
        amp = g.amplitude_deg.to_numpy()
        in_state = STATE[id(R)][gaze_state][onset]
        clean = in_state & np.all([np.all(np.isfinite(w), axis=1) for w in win.values()], axis=0)

        share = []
        for bi, (lo, hi) in enumerate(zip(gaze_amp_bins[:-1], gaze_amp_bins[1:])):
            in_bin = (amp >= lo) & (amp < hi)
            d = gaze_traces.setdefault((gi, bi), {"gaze": [], "head": [], "eye": []})
            if (clean & in_bin).sum() >= min_ev:
                for k in win:
                    d[k].append(win[k][clean & in_bin].mean(0))
            hc = g.head_contrib[in_state & in_bin]
            share.append(hc.median() if len(hc) >= min_n else np.nan)
        head_share[gi].append(share)

fig, axes = plt.subplots(3, len(groups), figsize=(2.5 * len(groups), 5), squeeze=False,
                         sharex=True, sharey="row")
for ri, sig in enumerate(("gaze", "head", "eye")):
    for gi, title in enumerate(titles):
        ax = axes[ri, gi]
        for bi, (lo, hi) in enumerate(zip(gaze_amp_bins[:-1], gaze_amp_bins[1:])):
            plot_mean_se(ax, t_g, np.array(gaze_traces.get((gi, bi), {}).get(sig, [])), gaze_amp_colors[bi],
                         f"{lo}-{hi}", min_n=2)
        ax.axhline(0, color="0.8", lw=0.5)
        ax.axvline(0, color="0.8", lw=0.5)
        ax.set_title(title, fontsize=6, color=GROUP_COLORS[gi])
        ax.set_ylabel(f"{sig} rotation (deg)")
        ax.legend(fontsize=4, title="gaze amp (deg), n sessions", title_fontsize=4)
for ax in axes[-1]:
    ax.set_xlabel("time from gaze onset (ms)")
fig.suptitle(f"gaze shifts: {gaze_state}", fontsize=7)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, f"eye_head_C_gaze_triggered_by_amp_{gaze_state}" + fig_tag)

fig, ax = plt.subplots(figsize=(2.4, 2.2))
x = np.arange(len(gaze_amp_bins) - 1)
for gi, (title, c) in enumerate(zip(titles, GROUP_COLORS)):
    share = np.array(head_share[gi], float)
    n = np.isfinite(share).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # bins with no sessions stay NaN
        ax.errorbar(x, np.nanmean(share, 0), yerr=np.nanstd(share, 0) / np.sqrt(n), fmt="o-", ms=3,
                    lw=1, capsize=2, color=c, label=title)
ax.set_xticks(x, [f"{lo}-{hi}" for lo, hi in zip(gaze_amp_bins[:-1], gaze_amp_bins[1:])])
ax.set_xlabel("gaze amplitude (deg)")
ax.set_ylabel("head share of gaze shift")
ax.set_ylim(0, 1)
ax.legend(fontsize=4)
ax.set_title(f"head_contrib, gaze shifts: {gaze_state}", fontsize=6)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, f"eye_head_C_head_share_by_amp_{gaze_state}" + fig_tag)


# %% per-session table + trends vs EO (sessions are the unit; one line per ferret)
# METHODS: head_rate = head saccades / tracked-yaw time (Hz); head_amp / head_pkv = session medians
# (amplitude from unwrapped yaw + pitch).
# METHODS: PSCR measures use eye saccades whose onset is inside a turn, with a NaN-free window, that
# moved with the head (sign of horizontal eye displacement = sign of head yaw movement). cr_peak_vel =
# median most-negative eye velocity in the pscr_win frames after the saccade (counter-rotation);
# pscr_gain = median of mean eye / mean head velocity over that window (head > 30 deg/s), -1 = full
# compensation. Sessions with < min_n such saccades are NaN. pct_eye_in_turn = % of eye saccades in turns.
# METHODS: gaze shifts with onset in gaze_state, LE and RE pooled (event tables cell). gaze_rate = shifts /
# tracked-gaze time in gaze_state (Hz), averaged over eyes (session_rate). gaze_isi = median gap from one
# shift's position peak to the next onset (ms), NaN-free gaps only (pooled_intervals). gaze_amp / gaze_pkv =
# session medians; gaze_slope / gaze_pkv10 = OLS of log10 peak velocity on log10 amplitude, pkv10 = fitted
# peak velocity at 10 deg (sessions with >= min_n shifts). Eye data are 6 Hz filtered upstream, so pkv and
# slope are less reliable than amplitude, rate and ISI.
# METHODS: head_contrib = session median of head yaw / horizontal gaze displacement (onset -> peak).
# frac_eye_only = fraction with |head yaw displacement| < head_still_deg. frac_with_eye_sacc = fraction with
# an eye-saccade onset (same eye) between gaze onset and peak; the rest are gaze "shifts" without a saccade,
# e.g. head movement the eye did not compensate.
# METHODS: match_r_LE / match_r_RE = per-eye Pearson r of gaze displacement vs head + eye displacement (gaze
# should = yaw + eye). A low r flags a session whose pipeline gaze disagrees with its head and eye data; such
# sessions are kept, only flagged (printed below).
# METHODS: timing measures (burst_*): see B.3; NaN for sessions with < b3_min_head head saccades.
# TODO: head_contrib-vs-gaze-amplitude slope test.

session_rows = []
for group_idx, (group, title) in enumerate(zip(groups, titles)):
    for R in group:
        head_sacc, eye_sacc = HEAD[id(R)], E[id(R)]
        pscr_sacc = eye_sacc[eye_sacc.in_turn & eye_sacc.clean & eye_sacc.with_head]
        enough_pscr = len(pscr_sacc) >= min_n
        gaze_shift = G[id(R)][STATE[id(R)][gaze_state][G[id(R)].onset.to_numpy()]]
        gaze_fit = gaze_shift[(gaze_shift.amplitude_deg > 0) & (gaze_shift.peak_velocity_deg_s > 0)]
        ms_slope, ms_icpt = (np.polyfit(np.log10(gaze_fit.amplitude_deg), np.log10(gaze_fit.peak_velocity_deg_s), 1)
                             if len(gaze_fit) >= min_n else (np.nan, np.nan))
        gaze_isi_ms = pooled_intervals([R], "gaze", gaze_state, speed_threshold, min_bout)
        session_rows.append(dict(id=R.id, eo=R.eo, group=title, group_idx=group_idx, n_pscr=len(pscr_sacc),
                                 n_gaze=len(gaze_shift),
                                 head_rate=len(head_sacc) / (np.isfinite(R.yaw).sum() / FS),
                                 head_amp=head_sacc.amplitude_deg.median(),
                                 head_pkv=head_sacc.peak_velocity_deg_s.median(),
                                 pscr_gain=pscr_sacc.pscr_gain.median() if enough_pscr else np.nan,
                                 cr_peak_vel=pscr_sacc.eye_cr_vel.median() if enough_pscr else np.nan,
                                 pct_eye_in_turn=100 * eye_sacc.in_turn.mean(),
                                 gaze_rate=session_rate(R, "gaze", gaze_state, speed_threshold, min_bout),
                                 gaze_isi=np.median(gaze_isi_ms) if len(gaze_isi_ms) else np.nan,
                                 gaze_amp=gaze_shift.amplitude_deg.median(),
                                 gaze_pkv=gaze_shift.peak_velocity_deg_s.median(),
                                 gaze_slope=ms_slope, gaze_pkv10=10 ** (ms_slope + ms_icpt),
                                 head_contrib=gaze_shift.head_contrib.median(),
                                 frac_eye_only=gaze_shift.eye_only.mean(),
                                 frac_with_eye_sacc=gaze_shift.with_eye_sacc.mean(),
                                 **{m: B3[id(R)][m] if id(R) in B3 else np.nan for m in b3_measures},
                                 **{f"match_r_{eye}": d.gaze_disp.corr(d.head_disp + d.eye_disp)
                                    for eye, d in gaze_shift.groupby("eye")}))
session_df = pd.DataFrame(session_rows)
print(f"sessions with gaze vs head + eye r < 0.8 (kept), gaze_state = {gaze_state}:")
print(session_df[(session_df.match_r_LE < 0.8) | (session_df.match_r_RE < 0.8)]
      [["id", "eo", "n_gaze", "match_r_LE", "match_r_RE", "head_contrib"]].round(2).to_string(index=False))
measure_families = dict(head=["head_rate", "head_amp", "head_pkv"],
                        pscr=["pscr_gain", "cr_peak_vel", "pct_eye_in_turn"],
                        gaze=["gaze_rate", "gaze_isi", "gaze_amp", "gaze_pkv", "gaze_slope", "gaze_pkv10"],
                        gaze_comp=["head_contrib", "frac_eye_only", "frac_with_eye_sacc"],
                        timing=b3_measures)

# METHODS: EO trend per ferret: Pearson r of session value vs EO, sessions treated as independent
# (ferrets with >= 3 sessions). Reported as the number of ferrets whose r has the sign of the global
# trend (r over all sessions pooled), and how many of those reach p < 0.05. Normal cohort only; in
# compare_delayed mode the only comparison is delayed vs oldest normal sessions (group model below).
# METHODS: global trend across EO groups: linear mixed model (LMM), measure ~ group index (EO bin
# 0, 1, 2; or oldest normal = 0, delayed = 1) as a linear fixed effect, random intercept per ferret.
# β = change per group step. p = likelihood-ratio test against the intercept-only model (both fit
# by ML), chi2 with 1 df, Holm-corrected within each family of measures (head, pscr, gaze, gaze_comp).
stats_rows = []
ncol = max(len(m) for m in measure_families.values())   # same panel size in every family figure
for family, measures in measure_families.items():
    ferret_trend = {}   # measure -> per-ferret trend columns of stats_df

    if not compare_delayed:
        fig, axes = plt.subplots(1, ncol, figsize=(2.2 * ncol, 2.4), squeeze=False)
        for ax in axes[0, len(measures):]:
            ax.axis("off")
        for ax, measure in zip(axes[0], measures):
            plot_vs_eo(ax, session_df, measure, color=None)
            valid = session_df.dropna(subset=[measure])
            y_lo = (0 if measure in ("head_rate", "pct_eye_in_turn", "gaze_rate", "gaze_isi")
                    else valid[measure].min())   # y limits on whole tick steps
            y_step = np.diff(MaxNLocator(5, steps=[1, 2, 5, 10]).tick_values(y_lo, valid[measure].max())[:2])[0]
            y_lo, y_hi = np.floor(y_lo / y_step) * y_step, np.ceil(valid[measure].max() / y_step) * y_step
            ax.set_yticks(np.linspace(y_lo, y_hi, round((y_hi - y_lo) / y_step) + 1))
            ax.set_ylim(y_lo, y_hi)
            pooled_r = pearsonr(valid.eo, valid[measure])[0]
            pooled_up = pooled_r > 0
            ferret_r = {fid: pearsonr(g.eo, g[measure]) for fid, g in valid.groupby("id") if len(g) >= 3}
            ferrets_same_sign = [fid for fid, (r, p) in ferret_r.items() if (r > 0) == pooled_up]
            ferrets_p05 = sum(ferret_r[fid][1] < 0.05 for fid in ferrets_same_sign)
            ferret_trend[measure] = dict(pooled_r=pooled_r, ferrets_same_sign=
                                         f"{len(ferrets_same_sign)}/{len(ferret_r)} {'↑' if pooled_up else '↓'}",
                                         ferrets_p05=ferrets_p05)
            ax.set_title(f"{measure}\n{len(ferrets_same_sign)}/{len(ferret_r)} ferrets {'↑' if pooled_up else '↓'} "
                         f"({ferrets_p05} with p<0.05)", fontsize=6)
        axes[0, 0].legend(fontsize=4)
        if family.startswith("gaze"):
            fig.suptitle(f"gaze shifts: {gaze_state}", fontsize=7)
        sns.despine(fig)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.8, wspace=0.5)

        if SAVE_FIGS:
            save_fig(fig, f"eye_head per-session {family} vs EO" + (f" {gaze_state}" if family.startswith("gaze") else "") + fig_tag)

    lrt_p, lmm_beta = [], []
    for measure in measures:
        valid = session_df.dropna(subset=[measure])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # few ferrets: convergence warnings are routine
            lmm_full = smf.mixedlm(f"{measure} ~ group_idx", valid, groups=valid["id"]).fit(reml=False)
            lmm_null = smf.mixedlm(f"{measure} ~ 1", valid, groups=valid["id"]).fit(reml=False)
        lrt_p.append(chi2.sf(2 * (lmm_full.llf - lmm_null.llf), 1))
        lmm_beta.append(lmm_full.params["group_idx"])
    for measure, b, p in zip(measures, lmm_beta, multipletests(lrt_p, method="holm")[1]):
        valid = session_df.dropna(subset=[measure])
        stats_rows.append(dict(family=family, measure=measure,
                               **{t: f"{valid[measure][valid.group == t].mean():.3g} ± "
                                     f"{valid[measure][valid.group == t].std():.2g}" for t in titles},
                               n_sessions=len(valid), **ferret_trend.get(measure, {}),
                               lmm_beta=b, lmm_p_holm=p))

# one printout: group mean ± SD over sessions, per-ferret EO trend (normal cohort), group-trend LMM
stats_df = pd.DataFrame(stats_rows)
print(stats_df.to_string(index=False, na_rep="", float_format=lambda v: f"{v:.3g}"))
if compare_delayed:   # per-ferret means (delayed ferrets have one session each)
    all_measures = [m for measures in measure_families.values() for m in measures]
    print(session_df.groupby(["group", "id"])[all_measures].mean()
          .join(session_df.groupby(["group", "id"]).size().rename("n")).round(3).to_string())

