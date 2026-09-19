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
# Eye-head dynamics, replicating Wallace, Voit, Martin Machado et al., Kerr lab,
# Current Biology 35:761-775 (Feb 2025), Figure 4, across development.
# Their term for the return phase is PSCR — post-saccadic counter-rotation.

# NOTE — GAZE IS EXCLUDED FROM THIS SCRIPT.
# The gaze angular-velocity axes do not map consistently onto head axes: corr(gaze_x, yaw_v)
# for ferret 402 LE is +0.26 in one session and +0.74 in another, and LE/RE disagree in sign
# on pitch (-0.63 vs +0.71). The identity gaze = eye + head fails outright —
# sd(gaze - eye - yaw) = 94.0 exceeds sd(gaze) = 80.4. Because the inconsistency varies
# BETWEEN SESSIONS OF THE SAME ANIMAL it is not a fixed index bug in load_gaze_kinematics.py
# that this script could correct; it resembles eye-camera registration. Head yaw and eye vx
# are internally consistent, so this script uses head + eye only. This is also the likely
# explanation for the null VOR gain in vor.py.

from analyses.helper_functions import (FS, EYE_COLOR, HEAD_COLOR, LE_COLOR, RE_COLOR, eo_groups,
                                       head_saccades, head_eye_events, head_eye_traces)
import pandas as pd

# how panels are split: False = one panel per session, True = one panel per EO range
pool_by_eo = True
eo_bins = [(0, 4), (5, 9), (10, 20)]  # early / middle / late, inclusive

flip_eye = "RE"          # conjugate frame: required for signed head-vs-eye comparison
head_thresh = 50         # deg/s, head saccade detection (yaw |p90| = 106 deg/s)
pair_window = 30         # frames (250 ms) for an eye saccade to count as head-paired
pre, post = 24, 72       # frames: -200 to +600 ms from eye-saccade onset

amp_classes = [(5, 10), (10, 15), (15, np.inf)]        # Wallace Fig 4A/4C classes
head_vel_bins = np.arange(50, 550, 100)                # Wallace Fig 4D bins, deg/s

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)

# head saccade tables, extracted once (R.df_head is not usable — see head_saccades docstring)
HEAD = {id(R): head_saccades(R, velocity_threshold=head_thresh) for R in Results}
EVENTS = {id(R): head_eye_events(R, HEAD[id(R)], flip_eye, pre, post) for R in Results}

for R in Results:
    print(f"ferret {R.id} EO{R.eo:<3d} head={len(HEAD[id(R)]):5d} "
          f"({len(HEAD[id(R)]) / (len(R.LE_vx) / FS):.2f}/s)  "
          f"eye={len(R.df_LE) + len(R.df_RE):5d} "
          f"({(len(R.df_LE) + len(R.df_RE)) / (len(R.LE_vx) / FS):.2f}/s)")


# %% 1. head saccade characterization
# Confirms the locally-extracted head table is sane before anything is built on it.

fig, axes = plt.subplots(1, 3, figsize=(8, 2))

for group, title in zip(groups, titles):

    if not group:
        continue

    amp = np.concatenate([HEAD[id(R)]["amplitude_deg"].to_numpy() for R in group])
    pkv = np.concatenate([HEAD[id(R)]["peak_velocity_deg_s"].to_numpy() for R in group])
    dur = np.concatenate([(HEAD[id(R)]["peak"].to_numpy() - HEAD[id(R)]["onset"].to_numpy())
                          / FS * 1000 for R in group])
    if len(amp) < 20:
        continue

    for ax, v, lbl in zip(axes, (amp, pkv, dur),
                          ("amplitude (deg)", "peak velocity (deg/s)", "duration (ms)")):
        sns.histplot(ax=ax, x=v, bins=40, element="step", fill=False, stat="density",
                     label=title)
        ax.set_xlabel(lbl)

    rate = sum(len(HEAD[id(R)]) for R in group) / sum(len(R.LE_vx) for R in group) * FS
    print(f"{title:10s} n={len(amp):6d}  amp={np.median(amp):5.1f}  "
          f"pkv={np.median(pkv):6.1f}  dur={np.median(dur):5.0f} ms  rate={rate:.2f}/s")

axes[0].legend(fontsize=5)
sns.despine(fig)
fig.tight_layout()


# %% 2. eye-head onset timing
# "which came first?" — lag = head onset minus eye onset; positive = eye led.

fig, ax = plt.subplots(figsize=(3, 2))

for group, title in zip(groups, titles):

    if not group:
        continue

    lag = pd.concat([EVENTS[id(R)] for R in group])["lag_ms"].dropna()
    lag = lag[lag.abs() <= pair_window / FS * 1000]
    if len(lag) < 20:
        continue

    sns.histplot(ax=ax, x=lag, bins=30, element="step", fill=False, stat="density",
                 label=f"{title} (n={len(lag)})")
    print(f"{title:10s} n={len(lag):6d}  median={lag.median():+6.1f} ms  "
          f"frac eye-first={(lag > 0).mean():.3f}")

ax.axvline(0, color="0.6", ls=":", lw=0.5)
ax.set_xlabel("head onset - eye onset (ms)")
ax.legend(fontsize=5)
sns.despine(fig)
fig.tight_layout()


# %% 3. Wallace Fig 4A — rotation traces by amplitude class
# Head (magenta), LE (blue), RE (green) position, onset-aligned and sign-aligned so the
# head rotation is positive. Amplitude scaling is read off the peak separation.

t_ms = np.arange(-pre, post) / FS * 1000

fig, axes = plt.subplots(len(groups), len(amp_classes),
                         figsize=(3 * len(amp_classes), 2 * len(groups)), squeeze=False)

for row, (group, title) in enumerate(zip(groups, titles)):
    for col, (lo, hi) in enumerate(amp_classes):

        ax = axes[row][col]
        traces = {k: [] for k in ("head_pos", "LE_pos", "RE_pos")}
        for R in group:
            t = head_eye_traces(R, HEAD[id(R)], lo, hi, flip_eye, pair_window, pre, post)
            for k in traces:
                traces[k] += t[k]

        for k, c in (("head_pos", HEAD_COLOR), ("LE_pos", LE_COLOR), ("RE_pos", RE_COLOR)):
            if len(traces[k]) < 10:
                continue
            arr = np.array(traces[k])
            m = arr.mean(axis=0)
            se = arr.std(axis=0) / np.sqrt(len(arr))
            ax.plot(t_ms, m, color=c, lw=1, label=f"{k.split('_')[0]} (n={len(arr)})")
            ax.fill_between(t_ms, m - se, m + se, color=c, alpha=0.25)

        ax.axvline(0, color="0.8", lw=0.5)
        ax.set_title(f"{title}  {lo}-{hi} deg", fontsize=6)
        ax.set_xlabel("time from eye onset (ms)")
        ax.set_ylabel("rotation (deg)")
        ax.legend(fontsize=4)

sns.despine(fig)
fig.tight_layout()


# %% 4. Wallace Fig 4C — rotation velocity traces by amplitude class
# The PSCR appears here as the negative eye-velocity lobe following the primary phase.

fig, axes = plt.subplots(len(groups), len(amp_classes),
                         figsize=(3 * len(amp_classes), 2 * len(groups)), squeeze=False)

for row, (group, title) in enumerate(zip(groups, titles)):
    for col, (lo, hi) in enumerate(amp_classes):

        ax = axes[row][col]
        traces = {k: [] for k in ("head_vel", "LE_vel", "RE_vel")}
        for R in group:
            t = head_eye_traces(R, HEAD[id(R)], lo, hi, flip_eye, pair_window, pre, post)
            for k in traces:
                traces[k] += t[k]

        for k, c in (("head_vel", HEAD_COLOR), ("LE_vel", LE_COLOR), ("RE_vel", RE_COLOR)):
            if len(traces[k]) < 10:
                continue
            arr = np.array(traces[k])
            m = arr.mean(axis=0)
            se = arr.std(axis=0) / np.sqrt(len(arr))
            ax.plot(t_ms, m, color=c, lw=1, label=f"{k.split('_')[0]} (n={len(arr)})")
            ax.fill_between(t_ms, m - se, m + se, color=c, alpha=0.25)
            if k != "head_vel":
                print(f"{title:10s} {lo}-{hi} {k:8s} n={len(arr):5d} "
                      f"peak={m.max():7.1f}  trough={m.min():7.1f} deg/s")

        ax.axhline(0, color="0.8", lw=0.5)
        ax.axvline(0, color="0.8", lw=0.5)
        ax.set_title(f"{title}  {lo}-{hi} deg", fontsize=6)
        ax.set_xlabel("time from eye onset (ms)")
        ax.set_ylabel("rotation velocity (deg/s)")
        ax.legend(fontsize=4)

sns.despine(fig)
fig.tight_layout()


# %% 5. Wallace Fig 4B + 4D — amplitude-velocity scaling and counter-rotation
# 4D is the counter-rotation replication: peak POSITIVE head velocity against peak
# NEGATIVE eye velocity, with mean +- SD in 100 deg/s bins from 50 to 500.

fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))

for group, title in zip(groups, titles):

    if not group:
        continue

    E = pd.concat([EVENTS[id(R)] for R in group])
    H = pd.concat([HEAD[id(R)] for R in group])

    # 4B: amplitude vs peak velocity
    axes[0].scatter(H["amplitude_deg"], H["peak_velocity_deg_s"], s=1, alpha=0.1,
                    color=HEAD_COLOR)
    axes[0].scatter(E["amplitude_deg"], E["peak_velocity_deg_s"], s=1, alpha=0.1,
                    color=EYE_COLOR)

    # 4D: peak positive head velocity vs peak negative eye velocity
    paired = E[E["lag_ms"].abs() <= pair_window / FS * 1000]
    x = paired["head_peak_pos_vel"].to_numpy()
    y = paired["eye_peak_neg_vel"].to_numpy()
    sel = (x >= head_vel_bins[0]) & (x <= head_vel_bins[-1])
    if sel.sum() < 20:
        continue

    axes[1].scatter(x[sel], y[sel], s=1, alpha=0.08)
    centers, means, sds = [], [], []
    for blo in head_vel_bins[:-1]:
        m = (x >= blo) & (x < blo + 100)
        if m.sum() > 5:
            centers.append(blo + 50)
            means.append(y[m].mean())
            sds.append(y[m].std())
    line = axes[1].errorbar(centers, means, yerr=sds, fmt="o-", ms=3, lw=1, capsize=2,
                            label=f"{title} (n={sel.sum()})")

    r = np.corrcoef(x[sel], y[sel])[0, 1]
    slope = np.polyfit(x[sel], y[sel], 1)[0]
    print(f"{title:10s} n={sel.sum():6d}  r={r:+.3f}  slope={slope:+.3f}  "
          + "  ".join(f"{c:.0f}:{m:.0f}" for c, m in zip(centers, means)))

axes[0].set_xlabel("amplitude (deg)")
axes[0].set_ylabel("peak velocity (deg/s)")
axes[0].set_title("Fig 4B  head (magenta) / eye (blue)", fontsize=6)
axes[1].set_xlabel("peak positive head velocity (deg/s)")
axes[1].set_ylabel("peak negative eye velocity (deg/s)")
axes[1].set_title("Fig 4D  counter-rotation", fontsize=6)
axes[1].legend(fontsize=5)
sns.despine(fig)
fig.tight_layout()


# %% 6. eye-head timing exploration
# Exploration cell: the knobs are here, not in helper_functions, so a boundary can be moved
# and the effect seen immediately. One helper call supplies the event table; the rest is
# plain numpy/pandas.
#
# READ THE HEAD RATE COLUMN BEFORE COMPARING PERCENTAGES. Head saccade rate varies 3.3x
# across sessions (0.90-3.01/s) and drives occupancy directly: a session whose head barely
# moves has few paired saccades no matter how tightly coupled eye and head are. Measured on
# 402/420: corr(head_rate, %unpaired) = -0.731, and corr(EO, head_rate) = +0.648 — age and
# head rate are themselves correlated, so a raw trend with age may just be a rate trend.
# The scatter below plots that relationship directly rather than asserting it.

lead_ms = 50       # |lag| above this = one leads; below = synchronous
pair_ms = 250      # no head saccade within this = unpaired

rows = []

for R in Results:
    E = EVENTS[id(R)]
    lag = E["lag_ms"]

    unpaired = lag.isna() | (lag.abs() > pair_ms)
    eye_leads = (~unpaired) & (lag > lead_ms)
    head_leads = (~unpaired) & (lag < -lead_ms)
    sync = (~unpaired) & (lag.abs() <= lead_ms)

    rows.append(dict(ferret=R.id, eo=R.eo, n=len(E),
                     head_rate=len(HEAD[id(R)]) / (len(R.LE_vx) / FS),
                     eye_leads=100 * eye_leads.mean(), sync=100 * sync.mean(),
                     head_leads=100 * head_leads.mean(), unpaired=100 * unpaired.mean()))

S = pd.DataFrame(rows)
print(S.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print(f"\ncorr(head_rate, %unpaired) = {S.head_rate.corr(S.unpaired):+.3f}"
      "   <- occupancy is driven by head rate")
print(f"corr(EO, head_rate)        = {S.eo.corr(S.head_rate):+.3f}"
      "   <- age and head rate are themselves correlated")
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

# lag distribution per EO group, with the lead_ms boundaries drawn
for group, title in zip(groups, titles):
    if not group:
        continue
    lag = pd.concat([EVENTS[id(R)] for R in group])["lag_ms"].dropna()
    lag = lag[lag.abs() <= pair_ms]
    if len(lag) < 20:
        continue
    sns.histplot(ax=axes[2], x=lag, bins=40, element="step", fill=False, stat="density",
                 label=f"{title} (n={len(lag)})")
for b in (-lead_ms, lead_ms):
    axes[2].axvline(b, color="0.6", ls=":", lw=0.5)
axes[2].set_xlabel("head onset - eye onset (ms)")
axes[2].legend(fontsize=4)

sns.despine(fig)
fig.tight_layout()

# category kinematics: flat, and reported as such rather than left to be hunted for
E_all = pd.concat([EVENTS[id(R)].assign(eo=R.eo) for R in Results])
lag = E_all["lag_ms"]
E_all["category"] = np.where(lag.isna() | (lag.abs() > pair_ms), "unpaired",
                     np.where(lag > lead_ms, "eye_leads",
                      np.where(lag < -lead_ms, "head_leads", "synchronous")))
print("\ncategory kinematics (flat across category and age — a stated negative):")
print(E_all.groupby("category").agg(
    n=("amplitude_deg", "size"), med_amp=("amplitude_deg", "median"),
    med_pkv=("peak_velocity_deg_s", "median"),
    frac_same=("same_direction", "mean")).to_string(float_format=lambda v: f"{v:.2f}"))
