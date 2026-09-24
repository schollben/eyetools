# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import (FS, EO_BINS, EYE_COLOR, AGE_COLORS, HEAD_COLOR, LE_COLOR,
                                       RE_COLOR, eo_groups, unwrap_deg, load_results, set_style,
                                       save_fig, head_eye_windows, head_triggered_windows,
                                       eye_head_coupling, in_head_saccade, running_mask, onset_correlogram, plot_mean_se,
                                       session_trend, plot_vs_eo)
set_style()

# LOAD DATA
Results = load_results()  # FERRETS in helper_functions; 753, 757 -> look carefully at these files

# %% settings for every plot below
# Eye-head dynamics, replicating Wallace, Voit, Martin Machado et al., Kerr lab,
# Current Biology 35:761-775 (Feb 2025), Figure 4, across development.
# Their term for the return phase is PSCR — post-saccadic counter-rotation.
#
# Frame: flip_eye = "LE" is the HEAD frame — eye + = head yaw +, and world gaze = yaw + eye
# (checked against the pipeline's gaze). With "RE" the eye axis points the other way and a
# counter-rotation would read as positive.

# how panels are split: False = one panel per session, True = one panel per EO range
pool_by_eo = True
eo_bins = EO_BINS

flip_eye = "LE"
pair_window = 30         # frames (250 ms) an eye saccade may lead its head saccade
pre, post = 24, 72       # frames: -200 to +600 ms from eye-saccade onset
pscr_win = 24            # frames (200 ms) after the eye saccade ends: counter-rotation window
head_pre, head_post = 24, 144   # frames: -200 to +1200 ms from head onset (panel B)

amp_classes = [(5, 10), (10, 15), (15, np.inf)]   # Wallace Fig 4A/4C classes, horizontal deg
head_vel_bins = np.arange(50, 551, 100)            # Wallace Fig 4D bins, deg/s
save_figs = False

# locomotion flag for the Wallace 4A / 4C / 4D cells: "all" | "stationary" | "running",
# judged at eye saccade onset (running_mask, same thresholds as movement_stats.py)
loco = "all"
speed_threshold = 50     # mm/s
min_bout = 30            # frames

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)
colors = AGE_COLORS if pool_by_eo else [None] * len(groups)

# Head saccades are R.df_head as extracted by process_session. Two columns are recomputed:
# peak velocity (stored in rad/s) and amplitude, taken from UNWRAPPED yaw at the same
# onset/peak frames — raw yaw jumps by 360 deg at the +-180 seam.
HEAD = {}
for R in Results:
    df = R.df_head.copy()
    yaw, pitch = unwrap_deg(R.yaw), np.asarray(R.pitch, float)
    on, pk = df["onset"].to_numpy().astype(int), df["peak"].to_numpy().astype(int)
    df["yaw_disp"] = yaw[pk] - yaw[on]
    df["amplitude_deg"] = np.hypot(df["yaw_disp"], pitch[pk] - pitch[on])
    df["peak_velocity_deg_s"] = np.rad2deg(df["peak_velocity_deg_s"].astype(float))
    HEAD[id(R)] = df

# one row per eye saccade (E) and its onset-aligned traces (W), computed once
E, W = {}, {}
for R in Results:
    E[id(R)], W[id(R)] = head_eye_windows(R, HEAD[id(R)], flip_eye, pre, post,
                                          pair_window, pscr_win)
    # locomotor state at each eye saccade onset; "unknown" where body speed is not tracked
    run = running_mask(R, speed_threshold, min_bout)
    ok = np.isfinite(np.asarray(R.speed, float))
    on = E[id(R)]["onset"].to_numpy()
    E[id(R)]["loco"] = np.where(~ok[on], "unknown", np.where(run[on], "running", "stationary"))


def pool(group):
    """E rows and W traces of a session group, stacked in the same order."""
    e = pd.concat([E[id(R)] for R in group], ignore_index=True)
    w = {k: np.vstack([W[id(R)][k] for R in group]) for k in W[id(group[0])]}
    return e, w


# %% per-session table — every developmental summary below reads from here
rows = []
for R in Results:
    H, e = HEAD[id(R)], E[id(R)]
    obs_e, chance_e, obs_h, chance_h = eye_head_coupling(R, H, pair_window)
    p = e[e.paired]
    cr = p[(p.same_direction == 1) & p.clean]
    rows.append(dict(id=R.id, eo=R.eo,
                     head_rate=len(H) / (np.isfinite(R.yaw).sum() / FS),
                     head_amp=H.amplitude_deg.median(),
                     head_pkv=H.peak_velocity_deg_s.median(),
                     eye_in_head=100 * obs_e, eye_in_head_chance=100 * chance_e,
                     coupling=100 * (obs_e - chance_e),
                     head_with_eye=100 * obs_h, head_with_eye_chance=100 * chance_h,
                     pscr_gain=cr.pscr_gain.median(), n_paired=len(p)))
SESS = pd.DataFrame(rows)

# %% A. head saccades across development, binned by EO (one point per session, bin median)

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2))
for ax, col, lbl in zip(axes, ("head_rate", "head_amp", "head_pkv"),
                        ("head saccades (/s)", "median amplitude (deg)",
                         "median peak velocity (deg/s)")):
    for i, (lo, hi) in enumerate(eo_bins):
        v = SESS.loc[(SESS.eo >= lo) & (SESS.eo <= hi), col].dropna()
        if not len(v):
            continue
        ax.plot(np.full(len(v), i - 0.1), v, "o", ms=3, alpha=0.6, color=AGE_COLORS[i])
        ax.plot(i + 0.1, v.median(), "o", ms=7, mfc="white", mew=1.5, color=AGE_COLORS[i])
    ax.set_xticks(range(len(eo_bins)), [f"EO {lo}-{hi}" for lo, hi in eo_bins])
    ax.set_ylabel(lbl)
    session_trend(SESS, col)
sns.despine(fig)
fig.tight_layout()

# if save_figs:
#     save_fig(fig, "eye_head_A_head_saccades")


# %% B. head-onset-triggered average: head and eye-in-head (head frame)
# Sign-aligned so every head saccade turns positive. An eye trace moving negative while the
# head keeps turning is the eye counter-rotating ("saccade and fixate").

t_h = np.arange(-head_pre, head_post) / FS * 1000
fig, axes = plt.subplots(1, len(groups), figsize=(2.5 * len(groups), 2), squeeze=False)

for ax, group, title in zip(axes[0], groups, titles):
    if not group:
        continue
    head, eye = [], []
    for R in group:
        T = head_triggered_windows(R, HEAD[id(R)], flip_eye, head_pre, head_post)
        hk = np.all(np.isfinite(T["head_pos"]), axis=1)
        head.append(T["head_pos"][hk])
        for k in ("LE_pos", "RE_pos"):
            ok = hk & np.all(np.isfinite(T[k]), axis=1)
            eye.append(T[k][ok])
    # eye on the left axis, head on the right, as in
    # saccade_andHead_triggered_average
    ax_h = ax.twinx()
    plot_mean_se(ax, t_h, np.vstack(eye), EYE_COLOR, "eye")
    plot_mean_se(ax_h, t_h, np.vstack(head), HEAD_COLOR, "head")
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
if save_figs:
    save_fig(fig, "eye_head_B_head_triggered")


# %% B.2 head-onset-triggered averages by head saccade amplitude
# One line per head-amplitude bin (like movement_stats "mean kinematic traces"). Does the eye
# contribute more as head movements get larger?

head_amp_bins = [0, 40, 80, 360]   # deg, head saccade amplitude (unwrapped)
amp_colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(head_amp_bins) - 1))

traces = {}   # (group index, amp bin index) -> dict of head / eye arrays
for gi, group in enumerate(groups):
    for R in group:
        T = head_triggered_windows(R, HEAD[id(R)], flip_eye, head_pre, head_post)
        amp = HEAD[id(R)]["amplitude_deg"].to_numpy(float)
        hk = np.all(np.isfinite(T["head_pos"]), axis=1)
        for bi, (lo, hi) in enumerate(zip(head_amp_bins[:-1], head_amp_bins[1:])):
            d = traces.setdefault((gi, bi), {"head": [], "eye": []})
            sel = hk & (amp >= lo) & (amp < hi)
            d["head"].append(T["head_pos"][sel])
            for k in ("LE_pos", "RE_pos"):
                ok = sel & np.all(np.isfinite(T[k]), axis=1)
                d["eye"].append(T[k][ok])

for sig, ylabel in (("head", "head rotation (deg)"), ("eye", "eye rotation (deg)")):
    fig, axes = plt.subplots(1, len(groups), figsize=(2.5 * len(groups), 2), squeeze=False)
    for gi, (ax, title) in enumerate(zip(axes[0], titles)):
        for bi, (lo, hi) in enumerate(zip(head_amp_bins[:-1], head_amp_bins[1:])):
            arrs = traces.get((gi, bi), {}).get(sig, [])
            if arrs:
                plot_mean_se(ax, t_h, np.vstack(arrs), amp_colors[bi], f"{lo}-{hi}")
        ax.axhline(0, color="0.8", lw=0.5)
        ax.axvline(0, color="0.8", lw=0.5)
        ax.set_title(f"{sig}  {title}", fontsize=6)
        ax.set_xlabel("time from head onset (ms)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=4, title="head amp (deg)", title_fontsize=4)
    sns.despine(fig)
    fig.tight_layout()

    # if save_figs:
    #     save_fig(fig, f"eye_head_B2_{sig}_by_head_amp")


# %% C. coupling by EO bin: observed vs chance (one point per session, bin median)
# left:   % of eye saccades (both eyes) that start inside a head saccade
#         ([head onset - pair_window, head peak]); dark = observed, grey = chance
# middle: % of head saccades that contain at least one eye saccade; same layout
# right:  coupling = observed - chance for the left measure
# Chance = eye onsets circularly shifted against the head (same rates, random timing).
# The raw percentage tracks head saccade rate (more head movement = more eye saccades land
# inside one by chance), so coupling is the measure to compare across age.

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2))

for ax, cols, lbl in ((axes[0], ("eye_in_head", "eye_in_head_chance"),
                       "% eye saccades in a head saccade"),
                      (axes[1], ("head_with_eye", "head_with_eye_chance"),
                       "% head saccades with an eye saccade"),
                      (axes[2], ("coupling",), "coupling (observed - chance, %)")):
    for i, (lo, hi) in enumerate(eo_bins):
        in_bin = (SESS.eo >= lo) & (SESS.eo <= hi)
        for col, dx in zip(cols, (-0.18, 0.18) if len(cols) == 2 else (0,)):
            v = SESS.loc[in_bin, col].dropna()
            if not len(v):
                continue
            c = "0.75" if col.endswith("_chance") else AGE_COLORS[i]
            ax.plot(np.full(len(v), i + dx - 0.07), v, "o", ms=3, alpha=0.6, color=c)
            ax.plot(i + dx + 0.07, v.median(), "o", ms=6, mfc="white", mew=1.2, color=c)
    ax.set_xticks(range(len(eo_bins)), [f"EO {lo}-{hi}" for lo, hi in eo_bins])
    ax.set_ylabel(lbl)

axes[2].axhline(0, color="0.8", lw=0.5)

axes[2].set_ylabel("coupling (observed - chance, %)")

for col in ("eye_in_head", "head_with_eye", "coupling"):
    session_trend(SESS, col)
sns.despine(fig)
fig.tight_layout()

# if save_figs:
#     save_fig(fig, "eye_head_C_coupling")


# %% D. timing: when do eye saccades start relative to head onset?
# Lag = head onset - eye onset (ms): positive = the eye started first.
#
# Left: head-triggered correlogram of eye onsets (both eyes), as a rate relative to chance
#       (1 = no relationship between eye and head timing), pooled per EO bin.
#
# Middle / right: the FIRST eye saccade of each head saccade.
#   For each head saccade, take all eye saccades (either eye) whose onset falls in
#   [head onset - pair_window, head peak], keep the earliest, and compute its lag.
#   A session's value is the median over its head saccades.
#
#   Why a chance level is needed: the window opens pair_window frames (250 ms) BEFORE head
#   onset, so the more eye saccades an animal makes, the sooner the first one lands in it —
#   even if eye and head timing are unrelated. Eye saccade rate roughly doubles with EO, so
#   the raw lag rises with age on rate alone. Chance = the same measure after circularly
#   shifting all eye onsets by a random offset (>= 10 s): same rate, no real timing.
#   excess = observed - chance is the timing effect with rate removed.
#   excess < 0: the first eye saccade comes LATER than chance, i.e. it waits for / is
#   triggered by head onset. excess ~ 0: eye timing is independent of head onset.

max_lag, bin_frames = 60, 3     # correlogram: +-500 ms, 25 ms bins
n_shift = 50                    # random shifts for the chance level

first_lag, first_lag_chance = [], []

rng = np.random.default_rng(0)

for R in Results:

    H = HEAD[id(R)]
    h_on = H["onset"].to_numpy().astype(int)
    eye_on = np.concatenate([R.df_LE["onset"].to_numpy(),
                             R.df_RE["onset"].to_numpy()]).astype(int)
    n_frames = len(R.LE_vx)

    medians = []   # [observed, shift 1, shift 2, ...]
    shifts = [0] + list(rng.integers(int(10 * FS), n_frames - int(10 * FS), n_shift))
    for shift in shifts:
        on = (eye_on + shift) % n_frames
        idx = in_head_saccade(on, H, pair_window)       # head saccade each onset falls in, -1 = none
        inside = idx >= 0
        lags = []
        for h in np.unique(idx[inside]):
            earliest = on[idx == h].min()
            lags.append((h_on[h] - earliest) / FS * 1000)
        medians.append(np.median(lags) if lags else np.nan)

    first_lag.append(medians[0])
    first_lag_chance.append(np.nanmedian(medians[1:]))

SESS["first_lag_ms"] = first_lag
SESS["first_lag_chance_ms"] = first_lag_chance
SESS["first_lag_excess_ms"] = SESS.first_lag_ms - SESS.first_lag_chance_ms

fig, axes = plt.subplots(1, 3, figsize=(9, 2))

for group, title, c in zip(groups, titles, colors):

    if not group:
        continue
    
    counts, expected = 0, 0
    
    for R in group:
        centers, n, ex = onset_correlogram(R, HEAD[id(R)], max_lag, bin_frames)
        counts, expected = counts + n, expected + ex

    axes[0].plot(centers, counts / expected, color=c, lw=1, label=title)

axes[0].axhline(1, color="0.8", lw=0.5)
axes[0].axvline(0, color="0.8", lw=0.5)
axes[0].set_xlabel("head onset - eye onset (ms)")
axes[0].set_ylabel("eye onset rate / chance")
axes[0].legend(fontsize=4)

# per session, grouped by EO bin: middle = observed (color) next to chance (grey),
# right = excess (observed - chance)
for ax, cols, lbl in ((axes[1], ("first_lag_ms", "first_lag_chance_ms"),
                       "first eye saccade lag (ms)"),
                      (axes[2], ("first_lag_excess_ms",), "first-saccade lag - chance (ms)")):
    for i, (lo, hi) in enumerate(eo_bins):
        in_bin = (SESS.eo >= lo) & (SESS.eo <= hi)
        for col, dx in zip(cols, (-0.18, 0.18) if len(cols) == 2 else (0,)):
            v = SESS.loc[in_bin, col].dropna()
            if not len(v):
                continue
            c = "0.75" if col.endswith("_chance_ms") else AGE_COLORS[i]
            ax.plot(np.full(len(v), i + dx - 0.07), v, "o", ms=3, alpha=0.6, color=c)
            ax.plot(i + dx + 0.07, v.median(), "o", ms=6, mfc="white", mew=1.2, color=c)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_xticks(range(len(eo_bins)), [f"EO {lo}-{hi}" for lo, hi in eo_bins])
    ax.set_ylabel(lbl)

for col in ("first_lag_ms", "first_lag_chance_ms", "first_lag_excess_ms"):
    session_trend(SESS, col)
sns.despine(fig)
fig.tight_layout()
if save_figs:
    save_fig(fig, "eye_head_D_timing")


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
    
    e, w = pool(group)

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
if save_figs:
    save_fig(fig, "eye_head_4A_position")


# %% Wallace Fig 4C — horizontal eye and head velocity, same amplitude classes as 4A
# Same saccades and layout as 4A. Head = yaw velocity, eye = horizontal eye velocity, both
# smoothed with the same savgol(11, 3). The post-saccadic counter-rotation (PSCR) is the
# negative eye-velocity lobe after the saccade.

fig, axes = plt.subplots(len(groups), len(amp_classes),
                         figsize=(3 * len(amp_classes), 2 * len(groups)), squeeze=False)
for row, (group, title) in enumerate(zip(groups, titles)):
    if not group:
        continue
    e, w = pool(group)
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
fig.tight_layout()
if save_figs:
    save_fig(fig, "eye_head_4C_velocity")


# %% Wallace Fig 4D — peak head velocity vs peak counter-rotation eye velocity, per EO bin
# One point per paired eye saccade that moved WITH its head saccade (LE and RE pooled),
# horizontal only, in the same flipped frame as 4A/4C (head turns positive):
#   x = peak (positive) head velocity from eye onset to the end of the window
#   y = peak negative eye velocity in the pscr_win frames after the eye saccade ends
#       (the counter-rotation, not the saccade itself)
# log_axes=True plots |y| on log-log axes, so the scaling is easier to compare across bins.
# Black = mean +- SD of y in 100 deg/s bins of x (head_vel_bins).

log_axes = True

fig, axes = plt.subplots(1, len(groups), figsize=(2.6 * len(groups), 2.4), squeeze=False,
                         sharex=True, sharey=True)
for ax, group, title, c in zip(axes[0], groups, titles, colors):
    if not group:
        continue
    e, _ = pool(group)
    p = e[e.paired & e.clean & (e.same_direction == 1)
          & ((e.loco == loco) | (loco == "all"))].dropna(subset=["head_peak_vel", "eye_cr_vel"])
    x = p.head_peak_vel.to_numpy()
    y = -p.eye_cr_vel.to_numpy() if log_axes else p.eye_cr_vel.to_numpy()
    if log_axes:
        keep = (x > 0) & (y > 0)
        x, y = x[keep], y[keep]

    ax.scatter(x, y, s=1, alpha=0.2, color=c)

    centers, means, sds = [], [], []
    for lo, hi in zip(head_vel_bins[:-1], head_vel_bins[1:]):
        yb = y[(x >= lo) & (x < hi)]
        if len(yb) > 5:
            centers.append((lo + hi) / 2)
            means.append(yb.mean())
            sds.append(yb.std())
    ax.errorbar(centers, means, yerr=sds, fmt="o-", ms=3, lw=1, capsize=2, color="k")

    r = np.corrcoef(np.log10(x), np.log10(y))[0, 1] if log_axes else np.corrcoef(x, y)[0, 1]
    print(f"{title:10s} n={len(x):5d}  r={r:+.3f}{' (log-log)' if log_axes else ''}  "
          + "  ".join(f"{cc:.0f}:{mm:.0f}" for cc, mm in zip(centers, means)))

    ax.set_title(f"{title}  n={len(x)}  r={r:+.2f}  ({loco})", fontsize=6)
    ax.set_xlabel("peak head velocity (deg/s)")
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("|peak negative eye velocity| (deg/s)")
    else:
        ax.axhline(0, color="0.8", lw=0.5)
        ax.set_ylabel("peak negative eye velocity (deg/s)")
sns.despine(fig)
fig.tight_layout()
if save_figs:
    save_fig(fig, "eye_head_4D_counter_rotation")


# %% S1. head saccade distributions per EO bin

fig, axes = plt.subplots(1, 3, figsize=(8, 2))

for group, title, c in zip(groups, titles, colors):
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

for group, title, c in zip(groups, titles, colors):
    if not group:
        continue
    H = pd.concat([HEAD[id(R)] for R in group])
    e, _ = pool(group)
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
                     eye_leads=100 * (e.paired & (lag > lead_ms)).mean(),
                     sync=100 * (e.paired & (lag.abs() <= lead_ms)).mean(),
                     head_leads=100 * (e.paired & (lag < -lead_ms)).mean(),
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
for group, title, c in zip(groups, titles, colors):
    if not group:
        continue
    lag = pd.concat([E[id(R)] for R in group])["lag_ms"].dropna()
    if len(lag) < 20:
        continue
    sns.histplot(ax=axes[2], x=lag.to_numpy(), bins=40, element="step", fill=False,
                 stat="density", color=c, label=f"{title} (n={len(lag)})")
for b in (-lead_ms, lead_ms):
    axes[2].axvline(b, color="0.6", ls=":", lw=0.5)
axes[2].set_xlabel("head onset - eye onset (ms)")
axes[2].legend(fontsize=4)

sns.despine(fig)
fig.tight_layout()

E_all = pd.concat(list(E.values()), ignore_index=True)
lag = E_all["lag_ms"]
E_all["category"] = np.where(~E_all.paired, "unpaired",
                     np.where(lag > lead_ms, "eye_leads",
                      np.where(lag < -lead_ms, "head_leads", "synchronous")))
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
    e, _ = pool(group)
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
