# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
# tools
from utils import create_subplot_grid, load_session_data, process_session, removeBadData, getSesh
import numpy as np
# plotting setup
from utils.config import SAVELOC
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 6
plt.rcParams['svg.fonttype'] = 'none'

# LOAD DATA

SESSION = getSesh.by_ferret(402, 420)     # multiple — preserves order by ferret
#SESSION = getSesh.by_ferret(753)         # or load sessions from an individual ID
#SESSION = getSesh.by_eo(10,20)           # or load sessions by an EO range (inclusive)

Results = []
for session in SESSION:

    R = load_session_data(session)
    removeBadData(R)
    process_session(R, window_in_sec=5,
                    velocity_threshold_eye=40, velocity_threshold_gaze=2,
                    velocity_threshold_head=2, min_duration=12, min_inter_event=12)
    Results.append(R)

n_sesh = len(Results)
print(n_sesh, "sessions loaded")



# %% settings for every plot below
# BINOCULARITY. Hypothesis: conjugacy INCREASES with age.
#
# NOTE 1 — LE_x / RE_x are natively nasal/temporal adduction angles, NOT a shared
# left/right head axis. A positive LE-RE correlation REQUIRES negating one eye's
# horizontal signal. flip_eye=None gives the exact negative; "LE" and "RE" are equivalent.
#
# NOTE 2 — the answer depends on which data you look at, which is what data_mode is for:
#   events     (paired saccade displacement)  +0.634 / +0.654 / +0.469  -> against
#   continuous (velocity, saccade frames)     +0.213 / +0.202 / +0.300  -> supports
# Continuous POSITION does not work at all (per-session r spans -0.35..+0.21 and flips
# sign within one animal on consecutive days). No curation tried fixes it: the
# registration error is a per-session DC offset, which differencing removes and averaging
# does not. That is why velocity works and position does not. signal="position" is kept
# so the assumption can be re-tested after any loading or threshold change.
#
# NOTE 3 — the two animals disagree: ferret 420 corr(EO, r) = +0.614 over 12 sessions,
# ferret 402 = -0.328 over 6. 420 supplies every session above EO 7, so pooling hides it.
# Use pool_by_eo = False, and see cell 4.
#
# NOTE 4 — panels pool FRAMES across sessions, so a long session counts more than a short
# one (pooled +0.240/+0.240/+0.294 vs mean-of-session +0.213/+0.202/+0.300 for the same
# data). The per-session table printed below is the unweighted view; cell 4 plots it.

from analyses.helper_functions import (FS, LE_COLOR, RE_COLOR, EYE_COLOR, HEAD_COLOR,
                                       LOCO_COLORS, eo_groups, fit_line, frame_mask,
                                       eye_signal, head_signal, clean_runs, run_xcorr,
                                       paired_saccades, conjugate_samples)
import pandas as pd

# how panels are split: False = one panel per session, True = one panel per EO range
pool_by_eo = True
eo_bins = [(0, 4), (5, 9), (10, 20)]  # early / middle / late, inclusive

# WHAT DATA the conjugacy cells use — the main knob for testing the hypothesis
data_mode = "both"     # "events" | "continuous" | "both"
signal = "velocity"    # "velocity" (works) | "position" (registration-limited, see NOTE 2)
frame_sacc = "saccade"  # continuous: "saccade" | "non_saccade" | "all"
frame_loco = "all"      # continuous: "all" | "stationary" | "running"
head_range = None       # continuous: None, or (lo, hi) deg/s head angular speed gate

flip_eye = "RE"        # common conjugate frame; required by EVERY signed LE-RE measure
axis = "horizontal"    # "horizontal" (x/vx) | "vertical" (y/vy) | "total" (2D magnitude)
pair_window = 12       # frames (100 ms), LE-RE onset separation counted as one event
min_pairs = 50         # fewest paired events a panel will fit
amp_bins = [2, 5, 10, 15, 25, 40]    # deg, saccade amplitude bins for the slope sweep

speed_threshold = 100  # mm/s, stationary vs running
min_bout = 30          # frames, shortest stretch counted as running
head_bins = [0, 20, 50, 100, 1000]   # deg/s edges, head angular speed
tilt_max = 10          # deg, |pitch| and |roll| ceiling for the fixation cell
max_lag = 15           # frames (125 ms), LE-RE cross-correlation
min_run = 4 * max_lag  # frames, shortest NaN-free run worth correlating
disp_lim = 30          # deg, axis limit for displacement panels

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)

for R in Results:
    P = paired_saccades(R, pair_window, flip_eye, axis)
    both = (np.isfinite(R.LE_x) & np.isfinite(R.RE_x)).mean()
    ex, ey = conjugate_samples([R], "events", signal, flip_eye, axis, pair_window)
    cx, cy = conjugate_samples([R], "continuous", signal, flip_eye, axis, pair_window,
                               frame_sacc, frame_loco, head_range)
    r_ev = np.corrcoef(ex, ey)[0, 1] if len(ex) > 2 else np.nan
    r_co = np.corrcoef(cx, cy)[0, 1] if len(cx) > 2 else np.nan
    print(f"ferret {R.id} EO{R.eo:<3d} paired={P.paired.sum():5d} ({P.paired.mean():.2f})"
          f"  both_finite={both:.2f}  r_event={r_ev:+.3f}  r_cont={r_co:+.3f}")


# %% 1. LE vs RE conjugacy — events, continuous, or both

fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    out = conjugate_samples(group, data_mode, signal, flip_eye, axis, pair_window,
                            frame_sacc, frame_loco, head_range)

    if data_mode == "both":
        ex, ey, cx, cy = out
        sns.scatterplot(ax=ax, x=cx, y=cy, s=2, alpha=0.1, color=EYE_COLOR)
        sns.scatterplot(ax=ax, x=ex, y=ey, s=3, alpha=0.4, color=HEAD_COLOR)
        r_ev = np.corrcoef(ex, ey)[0, 1]
        r_co = np.corrcoef(cx, cy)[0, 1]
        ax.set_title(f"{title}  event r={r_ev:+.2f}  cont r={r_co:+.2f}")
        print(f"{title}  event r={r_ev:+.3f} n={len(ex)}   cont r={r_co:+.3f} n={len(cx)}")
    else:
        x, y = out
        sns.scatterplot(ax=ax, x=x, y=y, s=3, alpha=0.3, color=EYE_COLOR)
        slope, _ = fit_line(ax, x, y, min_n=min_pairs)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{title}  r={r:+.2f}  n={len(x)}")
        print(f"{title}  {data_mode}  r={r:+.3f}  slope={slope:.3f}  n={len(x)}")

    lim = disp_lim if data_mode == "events" else None
    if lim:
        ax.axis([-lim, lim, -lim, lim])  # [xmin, xmax, ymin, ymax]
    ax.axline((0, 0), slope=1, color="0.7", lw=0.5)
    ax.set_xlabel("LE")
    ax.set_ylabel("RE")


# conjugacy vs saccade amplitude (events only — amplitude is an event property)
# a threshold artifact should live in the small-amplitude bins and vanish in the large ones

fig, ax = plt.subplots(figsize=(3.5, 2.5))

for group, title in zip(groups, titles):

    if not group:
        continue

    P = pd.concat([paired_saccades(R, pair_window, flip_eye, axis) for R in group])
    P = P[P.paired & np.isfinite(P.LE_disp) & np.isfinite(P.RE_disp)]

    centers, slopes = [], []
    for lo, hi in zip(amp_bins[:-1], amp_bins[1:]):
        sel = P[(P.LE_amp >= lo) & (P.LE_amp < hi)]
        if len(sel) < min_pairs:
            continue
        centers.append((lo + hi) / 2)
        slopes.append(np.polyfit(sel.LE_disp, sel.RE_disp, 1)[0])
        print(f"{title}  amp {lo}-{hi} deg  slope={slopes[-1]:+.3f}  n={len(sel)}")

    ax.plot(centers, slopes, "o-", ms=3, lw=1, label=title)

ax.axhline(1, color="0.7", lw=0.5)
ax.set_xlabel("saccade amplitude (deg)")
ax.set_ylabel("RE / LE displacement slope")
ax.legend()
sns.despine(fig)


# %% 2. conjugacy vs head movement and locomotion

fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    for loco in ("all", "stationary", "running"):

        centers, rs = [], []
        for lo, hi in zip(head_bins[:-1], head_bins[1:]):
            x, y = conjugate_samples(group, "continuous", signal, flip_eye, axis,
                                     pair_window, frame_sacc, loco, (lo, hi),
                                     speed_threshold, min_bout)
            if len(x) < min_pairs:
                continue
            centers.append((lo + hi) / 2)
            rs.append(np.corrcoef(x, y)[0, 1])
            print(f"{title}  {loco:10s} head {lo}-{hi} deg/s  r={rs[-1]:+.3f}  n={len(x)}")

        ax.plot(centers, rs, "o-", ms=3, lw=1, color=LOCO_COLORS[loco], label=loco)

    ax.set_xscale("log")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_title(title)
    ax.set_xlabel("head angular speed (deg/s)")
    ax.set_ylabel("LE-RE correlation")

axes[0].legend()


# %% 3. disconjugate velocity outside saccades

fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    for loco in ("all", "stationary", "running"):

        centers, means, sems = [], [], []
        for lo, hi in zip(head_bins[:-1], head_bins[1:]):

            vals = []
            for R in group:
                m = frame_mask(R, "non_saccade", loco, speed_threshold, min_bout)
                hs = head_signal(R, "speed")
                dv = np.abs(eye_signal(R, "LE", "vx", flip_eye)
                            - eye_signal(R, "RE", "vx", flip_eye))
                m = m & (hs >= lo) & (hs < hi) & np.isfinite(dv)
                vals.append(dv[m])

            v = np.concatenate(vals)
            if len(v) < min_pairs:
                continue
            centers.append((lo + hi) / 2)
            means.append(np.mean(v))
            sems.append(np.std(v) / np.sqrt(len(v)))
            print(f"{title}  {loco:10s} head {lo}-{hi} deg/s  "
                  f"mean={means[-1]:.1f}  sd={np.std(v):.1f}  n={len(v)}")

        ax.errorbar(centers, means, yerr=sems, fmt="o-", ms=3, lw=1,
                    color=LOCO_COLORS[loco], label=loco)

    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("head angular speed (deg/s)")
    ax.set_ylabel("|LE - RE| velocity (deg/s)")

axes[0].legend()


# CW vs CCW yaw, head roughly level
fig, ax = plt.subplots(figsize=(3, 2))

labels, vals = [], []
for group, title in zip(groups, titles):

    if not group:
        continue

    for name, lo, hi in (("CW", 50, 1e6), ("CCW", -1e6, -50)):

        acc = []
        for R in group:
            m = frame_mask(R, "non_saccade", "all", speed_threshold, min_bout)
            yv = head_signal(R, "yaw_v")
            level = (np.abs(head_signal(R, "pitch")) < tilt_max) & \
                    (np.abs(head_signal(R, "roll")) < tilt_max)
            dv = eye_signal(R, "LE", "vx", flip_eye) - eye_signal(R, "RE", "vx", flip_eye)
            m = m & level & (yv >= lo) & (yv < hi) & np.isfinite(dv)
            acc.append(dv[m])

        v = np.concatenate(acc)
        labels.append(f"{title}\n{name}")
        vals.append(np.median(v) if len(v) else np.nan)
        print(f"{title}  {name}  median LE-RE velocity={vals[-1]:+.2f} deg/s  n={len(v)}")

sns.barplot(ax=ax, x=labels, y=vals, color="0.8")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("LE - RE velocity (deg/s)")
ax.tick_params(axis="x", rotation=45)
sns.despine(fig)
fig.tight_layout()


# %% 4. registration and pairing diagnostics — READ BEFORE INTERPRETING ANYTHING ABOVE
# The requested monocular/binocular end-point density is deliberately NOT plotted: the
# "monocular" category is dominated by tracking dropout (other_nan below), so such a map
# would show camera performance vs age, not eye coordination.

fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

for R in Results:

    lx = eye_signal(R, "LE", "x", flip_eye)
    rx = eye_signal(R, "RE", "x", flip_eye)
    lv = eye_signal(R, "LE", "vx", flip_eye)
    rv = eye_signal(R, "RE", "vx", flip_eye)
    m = frame_mask(R, "non_saccade", "all", speed_threshold, min_bout)

    for ax, a, b in ((axes[0], lx, rx), (axes[1], lv, rv)):
        acc = []
        for start, stop in clean_runs(m & np.isfinite(a) & np.isfinite(b), min_run):
            c = run_xcorr(a[start:stop], b[start:stop], max_lag)
            if c is not None:
                acc.append(c)
        if acc:
            ax.plot(np.arange(-max_lag, max_lag + 1) / FS * 1000,
                    np.mean(acc, axis=0), lw=0.5)

axes[0].set_title("position")
axes[1].set_title("velocity")
for ax in axes:
    ax.axvline(0, color="0.7", lw=0.5)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("LE-RE correlation")
sns.despine(fig)
fig.tight_layout()


# conjugacy vs EO, one line per animal, both modes — the direct hypothesis plot
fig, ax = plt.subplots(figsize=(3.5, 2.5))

rows = []
for R in Results:
    ex, ey = conjugate_samples([R], "events", signal, flip_eye, axis, pair_window)
    cx, cy = conjugate_samples([R], "continuous", signal, flip_eye, axis, pair_window,
                               frame_sacc, frame_loco, head_range)
    P = paired_saccades(R, pair_window, flip_eye, axis)
    rows.append(dict(id=R.id, eo=R.eo,
                     r_event=np.corrcoef(ex, ey)[0, 1] if len(ex) > 2 else np.nan,
                     r_cont=np.corrcoef(cx, cy)[0, 1] if len(cx) > 2 else np.nan,
                     paired=P.paired.mean(),
                     other_nan=P.loc[~P.paired, "other_nan"].mean(),
                     other_moving=(P.loc[~P.paired, "other_max_speed"] > 40).mean(),
                     both_finite=(np.isfinite(R.LE_x) & np.isfinite(R.RE_x)).mean()))

D = pd.DataFrame(rows)

for fid, color in zip(sorted(D.id.unique()), (LE_COLOR, RE_COLOR)):
    d = D[D.id == fid].sort_values("eo")
    ax.plot(d.eo, d.r_event, "o-", ms=3, lw=1, color=color, label=f"{fid} events")
    ax.plot(d.eo, d.r_cont, "o--", ms=3, lw=1, color=color, alpha=0.6,
            label=f"{fid} continuous")
    print(f"ferret {fid}  corr(EO, r_event)={np.corrcoef(d.eo, d.r_event)[0,1]:+.3f}"
          f"  corr(EO, r_cont)={np.corrcoef(d.eo, d.r_cont)[0,1]:+.3f}  n={len(d)}")

ax.axhline(0, color="0.7", lw=0.5)
ax.set_xlabel("EO (days)")
ax.set_ylabel("LE-RE correlation")
ax.legend(fontsize=4)
sns.despine(fig)
fig.tight_layout()

print()
print(D.round(3).to_string(index=False))
