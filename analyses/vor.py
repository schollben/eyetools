# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
from utils import create_subplot_grid, load_session_data, process_session, removeBadData, getSesh
from utils import non_saccade_mask
import numpy as np
from scipy.stats import mannwhitneyu as mwu
from utils.config import SAVELOC
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 6
plt.rcParams['svg.fonttype'] = 'none'

# LOAD DATA
SESSION = getSesh.by_ferret(402, 405, 407, 420) # 753, 757 -> look carefully at these files
Results = []
for session in SESSION:
    R = load_session_data(session)
    removeBadData(R)
    process_session(R, window_in_sec=5,
                    velocity_threshold_eye=40, velocity_threshold_gaze=40,
                    velocity_threshold_head=1, min_duration=8, min_inter_event=8)
    Results.append(R)
n_sesh = len(Results)
print(n_sesh, "sessions loaded")


# %% settings for every plot below
# note: roll_v/pitch_v/yaw_v are stored as rad/s in the csv (units column says "rad_s")
# and the loaders do not convert them. Every cell below converts with np.rad2deg;
# do not edit the loading scripts.

from analyses.helper_functions import (FS, EYE_COLOR, HEAD_COLOR, LOCO_COLORS, frame_mask, head_signal,
                                       eye_signal, head_eye_pairs, eo_groups,
                                       fit_line, clean_runs, run_xcorr)

# how panels are split: False = one panel per session, True = one panel per EO range
pool_by_eo = True
eo_bins = [(0, 3), (4, 7), (8, 20)]

flip_eye = "RE"  # put both eyes in a common conjugate frame (None = nasal/temporal)
speed_threshold = 100      # mm/s, stationary vs running
min_bout = 30              # frames, shortest run of frames counted as running

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)

print("head speed deg/s, session 0:",
      np.round(np.nanpercentile(np.rad2deg(Results[0].angVelocities), [50, 99]), 1))
print()
for R in Results:
    n = len(R.LE_vx)
    usable = (non_saccade_mask(R) & np.isfinite(np.asarray(R.LE_vx, float))).sum()
    print(f"ferret {R.id} EO{R.eo:<3d} n={n:6d}  usable non-saccade LE frames={usable:6d}"
          f" ({100*usable/n:4.1f}%)")


# %% 1. head total angular velocity vs eye speed (saccades INCLUDED)

fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    h, e = head_eye_pairs(group, "speed", "speed", "all", "all")
    sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.3, color=EYE_COLOR)
    slope, _ = fit_line(ax, h, e)

    ax.set_title(f"{title}  ratio={slope:.2f}")
    ax.set_xlabel("head angular speed (deg/s)")
    ax.set_ylabel("eye speed (deg/s)")

# all data pooled
fig, ax = plt.subplots(figsize=(2.5, 2.5))
h, e = head_eye_pairs(Results, "speed", "speed", "all", "all")
sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.2, color=EYE_COLOR)
slope, _ = fit_line(ax, h, e)
ax.set_title(f"all sessions  ratio={slope:.2f}")
ax.set_xlabel("head angular speed (deg/s)")
ax.set_ylabel("eye speed (deg/s)")
sns.despine(fig)


# %% 2. head total angular velocity vs eye speed, OUTSIDE saccades

fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    h, e = head_eye_pairs(group, "speed", "speed", "non_saccade", "all")
    sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.3, color=EYE_COLOR)
    slope, _ = fit_line(ax, h, e)

    ax.set_title(f"{title}  ratio={slope:.2f}")
    ax.set_xlabel("head angular speed (deg/s)")
    ax.set_ylabel("eye speed (deg/s)")

fig, ax = plt.subplots(figsize=(2.5, 2.5))
h, e = head_eye_pairs(Results, "speed", "speed", "non_saccade", "all")
sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.2, color=EYE_COLOR)
slope, _ = fit_line(ax, h, e)
ax.set_title(f"all sessions, non-saccade  ratio={slope:.2f}")
ax.set_xlabel("head angular speed (deg/s)")
ax.set_ylabel("eye speed (deg/s)")
sns.despine(fig)


# %% 3. head POSITION vs eye position (static counter-roll, NOT VOR)
# pitch only: roll is too noisy, and yaw has no local position signal in the csv

fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    h, e = head_eye_pairs(group, "pitch", "y", "non_saccade", "all")
    sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.3, color=EYE_COLOR)
    slope, _ = fit_line(ax, h, e)

    ax.set_title(f"{title}  slope={slope:.2f}")
    ax.set_xlabel("head pitch (deg)")
    ax.set_ylabel("eye vertical position (deg)")


# %% 4. head VELOCITY vs eye velocity, signed per-axis (VOR)
# slope here IS signed VOR gain and should be NEGATIVE (eye counter-rotates)

axis = "horizontal"   # "horizontal": yaw_v vs vx | "vertical": pitch_v vs vy

head_attr = "yaw_v" if axis == "horizontal" else "pitch_v"
eye_key = "vx" if axis == "horizontal" else "vy"

fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    h, e = head_eye_pairs(group, head_attr, eye_key, "non_saccade", "all",
                          flip_eye)
    sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.3, color=EYE_COLOR)
    slope, _ = fit_line(ax, h, e)

    ax.set_title(f"{title}  gain={slope:.2f}")
    ax.set_xlabel(f"head {head_attr[:-2]} velocity (deg/s)")
    ax.set_ylabel(f"eye {eye_key} (deg/s)")


# %% 5. VOR split by locomotor state

axis = "horizontal"

head_attr = "yaw_v" if axis == "horizontal" else "pitch_v"
eye_key = "vx" if axis == "horizontal" else "vy"

fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    slopes = {}
    for state in ("stationary", "running"):
        h, e = head_eye_pairs(group, head_attr, eye_key, "non_saccade", state,
                              flip_eye, speed_threshold, min_bout)
        sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.3, color=LOCO_COLORS[state])
        slopes[state], _ = fit_line(ax, h, e, color=LOCO_COLORS[state])

    ax.set_title(f"{title}  stat={slopes['stationary']:.2f} run={slopes['running']:.2f}")
    ax.set_xlabel(f"head {head_attr[:-2]} velocity (deg/s)")
    ax.set_ylabel(f"eye {eye_key} (deg/s)")

# gain vs running speed bin
fig, ax = plt.subplots(figsize=(3, 2))
speed_edges = np.array([0, 50, 100, 200, 400, 800])

for group, title in zip(groups, titles):

    if not group:
        continue

    centers, gains = [], []
    for lo, hi in zip(speed_edges[:-1], speed_edges[1:]):
        hs, es = [], []
        for R in group:
            speed = np.asarray(R.speed, float)
            m = frame_mask(R, "non_saccade", "all") & (speed >= lo) & (speed < hi)
            h = head_signal(R, head_attr)
            for eye in ("LE", "RE"):
                hs.append(h[m])
                es.append(eye_signal(R, eye, eye_key, flip_eye)[m])
        h = np.concatenate(hs)
        e = np.concatenate(es)
        inds = np.isfinite(h) & np.isfinite(e)
        if inds.sum() < 20:
            continue
        centers.append((lo + hi) / 2)
        gains.append(np.polyfit(h[inds], e[inds], 1)[0])

    ax.plot(centers, gains, marker="o", ms=3, label=title)

ax.set_xlabel("running speed (mm/s)")
ax.set_ylabel("VOR gain")
ax.legend()
sns.despine(fig)


# %% 6. head angular speed distributions (head stability) — requires pool_by_eo

fig, ax = plt.subplots(figsize=(3.5, 2.5))

for group, title in zip(groups, titles):

    if not group:
        continue

    for state, ls in (("stationary", "-"), ("running", "--")):
        vals = np.concatenate([head_signal(R, "speed")[frame_mask(R, "all", state,
                                                        speed_threshold, min_bout)]
                               for R in group])
        vals = vals[np.isfinite(vals)]
        if len(vals) < 20:
            continue

        sns.histplot(ax=ax, x=vals, bins=60, element="step", fill=False,
                     stat="density", linestyle=ls, label=f"{title} {state}")
        print(f"{title:10s} {state:10s} mean={np.mean(vals):7.2f} "
              f"var={np.var(vals):9.2f} CV={np.std(vals)/np.mean(vals):.2f} n={len(vals)}")

ax.set_xlabel("head angular speed (deg/s)")
ax.legend(fontsize=5)
sns.despine(fig)

# head angular speed vs running speed, running periods only
fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    hs, sp = [], []
    for R in group:
        m = frame_mask(R, "all", "running", speed_threshold, min_bout)
        hs.append(head_signal(R, "speed")[m])
        sp.append(np.asarray(R.speed, float)[m])
    h = np.concatenate(hs)
    s = np.concatenate(sp)
    inds = np.isfinite(h) & np.isfinite(s)

    sns.scatterplot(ax=ax, x=s[inds], y=h[inds], s=3, alpha=0.3, color=HEAD_COLOR)
    slope, _ = fit_line(ax, s[inds], h[inds])

    ax.set_title(f"{title}  slope={slope:.3f}")
    ax.set_xlabel("running speed (mm/s)")
    ax.set_ylabel("head angular speed (deg/s)")


# %% 7. gain vs EO summary (unsigned ratio and signed per-axis gain)

fit_by = "pooled"  # "pooled": one fit per EO bin | "session": one fit per session
min_n = 500  # frames; sessions with little valid eye data give meaningless fits

rows = []  # (measure, bin, id, slope)

for group, title in zip(groups, titles):

    if not group:
        continue

    units = [(None, group)] if fit_by == "pooled" else [(R.id, [R]) for R in group]

    for uid, unit in units:

        h, e = head_eye_pairs(unit, "speed", "speed", "non_saccade", "all")
        if len(h) >= min_n:
            rows.append(("speed ratio", title, uid, np.polyfit(h, e, 1)[0]))
        else:
            print(f"skip {title} id={uid}: only {len(h)} valid frames")

        for ax_name, hattr, ekey in (("gain horizontal", "yaw_v", "vx"),
                                     ("gain vertical", "pitch_v", "vy")):
            h, e = head_eye_pairs(unit, hattr, ekey, "non_saccade", "all", flip_eye)
            if len(h) >= min_n:
                rows.append((ax_name, title, uid, np.polyfit(h, e, 1)[0]))

measures = ["speed ratio", "gain horizontal", "gain vertical"]
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2))

for ax, measure in zip(axes, measures):
    sub = [r for r in rows if r[0] == measure]
    x = [r[1] for r in sub]
    y = [r[3] for r in sub]

    if fit_by == "pooled":
        sns.barplot(ax=ax, x=x, y=y)
    else:
        sns.barplot(ax=ax, x=x, y=y, errorbar="sd", color="0.8")
        sns.stripplot(ax=ax, x=x, y=y, size=3, color="k")

    ax.set_title(measure)
    ax.set_ylabel("slope")
    ax.tick_params(axis="x", rotation=45)

sns.despine(fig)
fig.tight_layout()

for r in rows:
    print(f"{r[0]:16s} {r[1]:10s} id={r[2]}  slope={r[3]:.3f}")


# %% 8. gain vs head-velocity magnitude

axis = "horizontal"
mag_edges = np.arange(0, 220, 20)  # deg/s

head_attr = "yaw_v" if axis == "horizontal" else "pitch_v"
eye_key = "vx" if axis == "horizontal" else "vy"

fig, ax = plt.subplots(figsize=(3, 2))

for group, title in zip(groups, titles):

    if not group:
        continue

    h, e = head_eye_pairs(group, head_attr, eye_key, "non_saccade", "all", flip_eye)

    centers, gains = [], []
    for lo, hi in zip(mag_edges[:-1], mag_edges[1:]):
        inds = (np.abs(h) >= lo) & (np.abs(h) < hi)
        if inds.sum() < 20:
            continue
        centers.append((lo + hi) / 2)
        gains.append(np.polyfit(h[inds], e[inds], 1)[0])

    ax.plot(centers, gains, marker="o", ms=3, label=title)

ax.set_xlabel("|head velocity| (deg/s)")
ax.set_ylabel("VOR gain")
ax.legend()
sns.despine(fig)


# %% 9. eye-head velocity cross-correlation (VOR latency vs gaze-shift coordination)

# computed over contiguous NaN-free runs, length-weighted — no zero-filling

axis = "horizontal"
max_lag = 15  # frames (125 ms at 120 Hz)
min_run = 4 * max_lag  # frames; shortest run worth correlating

head_attr = "yaw_v" if axis == "horizontal" else "pitch_v"
eye_key = "vx" if axis == "horizontal" else "vy"

fig, axes = create_subplot_grid(len(groups))
lags = np.arange(-max_lag, max_lag + 1)

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    for subset, color in (("all", "#444444"), ("non_saccade", EYE_COLOR)):

        acc = np.zeros(len(lags))
        weight = 0
        n_runs = 0
        for R in group:
            h = head_signal(R, head_attr)
            m = frame_mask(R, subset, "all") & np.isfinite(h)
            for eye in ("LE", "RE"):
                e = eye_signal(R, eye, eye_key, flip_eye)
                for a, b in clean_runs(m & np.isfinite(e), min_run):
                    cc = run_xcorr(h[a:b], e[a:b], max_lag)
                    if cc is None:
                        continue
                    acc += cc * (b - a)
                    weight += b - a
                    n_runs += 1

        if weight == 0:
            print(f"{title:10s} {subset:12s} no usable runs")
            continue

        cc = acc / weight
        ax.plot(lags / FS * 1000, cc, color=color, lw=1, label=subset)
        peak_ms = lags[np.argmax(np.abs(cc))] / FS * 1000
        print(f"{title:10s} {subset:12s} peak lag = {peak_ms:+6.1f} ms  "
              f"r = {cc[np.argmax(np.abs(cc))]:+.3f}  runs={n_runs:5d}  frames={weight}")

    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_title(title)
    ax.set_xlabel("eye lag re head (ms)")
    ax.set_ylabel("correlation")
    ax.legend(fontsize=5)
