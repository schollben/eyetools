# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
# tools
from utils import create_subplot_grid, load_session_data, process_session, removeBadData, getSesh
from utils import non_saccade_mask
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
# delayed vision: 416,411,403

#SESSION = getSesh.by_ferret(402, 420)    # multiple — preserves order by ferret
SESSION = getSesh.by_ferret(753)          # or load sessions from an inidividual ID
#SESSION = getSesh.by_eo(10,20)   # or load sessions by an EO range (inclusive)

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


# %% shared helpers
# NOTE: roll_v/pitch_v/yaw_v are stored as rad/s in the csv (units column says "rad_s")
# and the loaders do not convert them. Convert here; do not edit the loading scripts.

FS = 120.0
LOCO_COLORS = {"all": "#444444", "stationary": "#725EE7", "running": "#E93115"}


def head_v(R, attr):
    return np.rad2deg(np.asarray(getattr(R, attr), float))


def head_speed(R):
    return np.rad2deg(np.asarray(R.angVelocities, float))


def running_mask(R, speed_threshold=100, min_bout=30):
    m = np.asarray(R.speed, float) >= speed_threshold
    edges = np.diff(np.concatenate(([0], m.astype(int), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for a, b in zip(starts, stops):
        if b - a < min_bout:
            m[a:b] = False
    return m


def saccade_frames(R):
    n = len(R.LE_vx)
    sacc = np.zeros(n, bool)
    for df in (R.df_LE, R.df_RE):
        for onset, peak in zip(df["onset"].to_numpy(), df["peak"].to_numpy()):
            sacc[onset:min(n, peak + 1)] = True
    return sacc


def frame_mask(R, sacc_subset="non_saccade", loco_subset="all",
               speed_threshold=100, min_bout=30):
    n = len(R.LE_vx)

    if sacc_subset == "non_saccade":
        m = non_saccade_mask(R)
    elif sacc_subset == "saccade":
        m = saccade_frames(R)
    else:
        m = np.ones(n, bool)

    if loco_subset != "all":
        run = running_mask(R, speed_threshold, min_bout)
        m = m & (run if loco_subset == "running" else ~run & np.isfinite(R.speed))

    return m


def eye_arrays(R, key, flip_RE=True):
    le = np.asarray(getattr(R, f"LE_{key}"), float)
    re = np.asarray(getattr(R, f"RE_{key}"), float)
    if flip_RE and key.endswith("x"):
        re = -re
    return le, re


def head_eye_pairs(group, head_fn, eye_key, sacc_subset="non_saccade", loco_subset="all",
                   flip_RE=True, speed_threshold=100, min_bout=30):
    """Pool a head signal against both eyes, frame-aligned and masked."""
    hs, es = [], []
    for R in group:
        h = head_fn(R)
        m = frame_mask(R, sacc_subset, loco_subset, speed_threshold, min_bout)
        for e in eye_arrays(R, eye_key, flip_RE):
            hs.append(h[m])
            es.append(e[m])
    h = np.concatenate(hs)
    e = np.concatenate(es)
    inds = np.isfinite(h) & np.isfinite(e)
    return h[inds], e[inds]


def eo_groups(Results, pool_by_eo, eo_bins):
    if pool_by_eo:
        groups = [[R for R in Results if lo <= R.eo <= hi] for lo, hi in eo_bins]
        titles = [f"EO {lo}-{hi}" for lo, hi in eo_bins]
    else:
        groups = [[R] for R in Results]
        titles = [f"Ferret {R.id} EO{R.eo}" for R in Results]
    return groups, titles


def fit_line(ax, x, y, color="k", min_n=500):
    if len(x) < min_n:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    xl = np.array([x.min(), x.max()])
    ax.plot(xl, slope * xl + intercept, color=color, lw=1)
    return slope, intercept


# sanity: units and per-session usable-frame counts
print("head speed deg/s, session 0:",
      np.round(np.nanpercentile(head_speed(Results[0]), [50, 99]), 1))
print()
for R in Results:
    n = len(R.LE_vx)
    usable = (non_saccade_mask(R) & np.isfinite(np.asarray(R.LE_vx, float))).sum()
    print(f"ferret {R.id} EO{R.eo:<3d} n={n:6d}  usable non-saccade LE frames={usable:6d}"
          f" ({100*usable/n:4.1f}%)")


# %% 1. head total angular velocity vs eye speed (saccades INCLUDED)

pool_by_eo = True  # False: one panel per session | True: one panel per EO range
eo_bins = [(0, 4), (5, 9), (10, 20)]  # early / middle / late, inclusive
flip_RE_horizontal = True
sacc_subset = "all"   # "all" | "saccade" | "non_saccade"
loco_subset = "all"   # "all" | "stationary" | "running"


def eye_speed_pairs(group, **kw):
    hs, es = [], []
    for R in group:
        h = head_speed(R)
        m = frame_mask(R, kw.get("sacc_subset", "all"), kw.get("loco_subset", "all"))
        vx = np.asarray(R.LE_vx, float), np.asarray(R.RE_vx, float)
        vy = np.asarray(R.LE_vy, float), np.asarray(R.RE_vy, float)
        for ex, ey in zip(vx, vy):
            hs.append(h[m])
            es.append(np.sqrt(ex[m] ** 2 + ey[m] ** 2))
    h = np.concatenate(hs)
    e = np.concatenate(es)
    inds = np.isfinite(h) & np.isfinite(e)
    return h[inds], e[inds]


groups, titles = eo_groups(Results, pool_by_eo, eo_bins)
fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    h, e = eye_speed_pairs(group, sacc_subset=sacc_subset, loco_subset=loco_subset)
    sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.3, color="0.5")
    slope, _ = fit_line(ax, h, e)

    ax.set_title(f"{title}  ratio={slope:.2f}")
    ax.set_xlabel("head angular speed (deg/s)")
    ax.set_ylabel("eye speed (deg/s)")

# all data pooled
fig, ax = plt.subplots(figsize=(2.5, 2.5))
h, e = eye_speed_pairs(Results, sacc_subset=sacc_subset, loco_subset=loco_subset)
sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.2, color="0.5")
slope, _ = fit_line(ax, h, e)
ax.set_title(f"all sessions  ratio={slope:.2f}")
ax.set_xlabel("head angular speed (deg/s)")
ax.set_ylabel("eye speed (deg/s)")
sns.despine(fig)


# %% 2. head total angular velocity vs eye speed, OUTSIDE saccades

pool_by_eo = True
eo_bins = [(0, 4), (5, 9), (10, 20)]
sacc_subset = "non_saccade"
loco_subset = "all"

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)
fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    h, e = eye_speed_pairs(group, sacc_subset=sacc_subset, loco_subset=loco_subset)
    sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.3, color="0.5")
    slope, _ = fit_line(ax, h, e)

    ax.set_title(f"{title}  ratio={slope:.2f}")
    ax.set_xlabel("head angular speed (deg/s)")
    ax.set_ylabel("eye speed (deg/s)")

fig, ax = plt.subplots(figsize=(2.5, 2.5))
h, e = eye_speed_pairs(Results, sacc_subset=sacc_subset, loco_subset=loco_subset)
sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.2, color="0.5")
slope, _ = fit_line(ax, h, e)
ax.set_title(f"all sessions, non-saccade  ratio={slope:.2f}")
ax.set_xlabel("head angular speed (deg/s)")
ax.set_ylabel("eye speed (deg/s)")
sns.despine(fig)


# %% 3. head POSITION vs eye position (static counter-roll, NOT VOR)
# pitch only: roll is too noisy, and yaw has no local position signal in the csv

pool_by_eo = True
eo_bins = [(0, 4), (5, 9), (10, 20)]
sacc_subset = "non_saccade"
loco_subset = "all"

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)
fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    h, e = head_eye_pairs(group, lambda R: np.asarray(R.pitch, float), "y",
                          sacc_subset, loco_subset)
    sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.3, color="0.5")
    slope, _ = fit_line(ax, h, e)

    ax.set_title(f"{title}  slope={slope:.2f}")
    ax.set_xlabel("head pitch (deg)")
    ax.set_ylabel("eye vertical position (deg)")


# %% 4. head VELOCITY vs eye velocity, signed per-axis (VOR)
# slope here IS signed VOR gain and should be NEGATIVE (eye counter-rotates)

pool_by_eo = True
eo_bins = [(0, 4), (5, 9), (10, 20)]
axis = "horizontal"   # "horizontal": yaw_v vs vx | "vertical": pitch_v vs vy
flip_RE_horizontal = True
sacc_subset = "non_saccade"
loco_subset = "all"

head_attr = "yaw_v" if axis == "horizontal" else "pitch_v"
eye_key = "vx" if axis == "horizontal" else "vy"

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)
fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    h, e = head_eye_pairs(group, lambda R: head_v(R, head_attr), eye_key,
                          sacc_subset, loco_subset, flip_RE_horizontal)
    sns.scatterplot(ax=ax, x=h, y=e, s=3, alpha=0.3, color="0.5")
    slope, _ = fit_line(ax, h, e)

    ax.set_title(f"{title}  gain={slope:.2f}")
    ax.set_xlabel(f"head {head_attr[:-2]} velocity (deg/s)")
    ax.set_ylabel(f"eye {eye_key} (deg/s)")


# %% 5. VOR split by locomotor state

pool_by_eo = True
eo_bins = [(0, 4), (5, 9), (10, 20)]
axis = "horizontal"
speed_threshold = 100  # mm/s
min_bout = 30          # frames (250 ms at 120 Hz)
sacc_subset = "non_saccade"

head_attr = "yaw_v" if axis == "horizontal" else "pitch_v"
eye_key = "vx" if axis == "horizontal" else "vy"

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)
fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    slopes = {}
    for state in ("stationary", "running"):
        h, e = head_eye_pairs(group, lambda R: head_v(R, head_attr), eye_key,
                              sacc_subset, state, flip_RE_horizontal,
                              speed_threshold, min_bout)
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
            m = frame_mask(R, sacc_subset, "all") & (np.asarray(R.speed, float) >= lo) \
                & (np.asarray(R.speed, float) < hi)
            h = head_v(R, head_attr)
            for e in eye_arrays(R, eye_key, flip_RE_horizontal):
                hs.append(h[m])
                es.append(e[m])
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

eo_bins = [(0, 4), (5, 9), (10, 20)]
speed_threshold = 100
min_bout = 30

groups, titles = eo_groups(Results, True, eo_bins)

fig, ax = plt.subplots(figsize=(3.5, 2.5))

for group, title in zip(groups, titles):

    if not group:
        continue

    for state, ls in (("stationary", "-"), ("running", "--")):
        vals = np.concatenate([head_speed(R)[frame_mask(R, "all", state,
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
        hs.append(head_speed(R)[m])
        sp.append(np.asarray(R.speed, float)[m])
    h = np.concatenate(hs)
    s = np.concatenate(sp)
    inds = np.isfinite(h) & np.isfinite(s)

    sns.scatterplot(ax=ax, x=s[inds], y=h[inds], s=3, alpha=0.3, color="0.5")
    slope, _ = fit_line(ax, s[inds], h[inds])

    ax.set_title(f"{title}  slope={slope:.3f}")
    ax.set_xlabel("running speed (mm/s)")
    ax.set_ylabel("head angular speed (deg/s)")


# %% 7. gain vs EO summary (unsigned ratio and signed per-axis gain)

fit_by = "pooled"  # "pooled": one fit per EO bin | "session": one fit per session
eo_bins = [(0, 4), (5, 9), (10, 20)]
min_n = 500  # frames; sessions with little valid eye data give meaningless fits
sacc_subset = "non_saccade"

groups, titles = eo_groups(Results, True, eo_bins)

rows = []  # (measure, bin, id, slope)

for group, title in zip(groups, titles):

    if not group:
        continue

    units = [(None, group)] if fit_by == "pooled" else [(R.id, [R]) for R in group]

    for uid, unit in units:

        h, e = eye_speed_pairs(unit, sacc_subset=sacc_subset)
        if len(h) >= min_n:
            rows.append(("speed ratio", title, uid, np.polyfit(h, e, 1)[0]))
        else:
            print(f"skip {title} id={uid}: only {len(h)} valid frames")

        for ax_name, hattr, ekey in (("gain horizontal", "yaw_v", "vx"),
                                     ("gain vertical", "pitch_v", "vy")):
            h, e = head_eye_pairs(unit, lambda R: head_v(R, hattr), ekey, sacc_subset)
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

eo_bins = [(0, 4), (5, 9), (10, 20)]
axis = "horizontal"
sacc_subset = "non_saccade"
mag_edges = np.arange(0, 220, 20)  # deg/s

head_attr = "yaw_v" if axis == "horizontal" else "pitch_v"
eye_key = "vx" if axis == "horizontal" else "vy"

groups, titles = eo_groups(Results, True, eo_bins)
fig, ax = plt.subplots(figsize=(3, 2))

for group, title in zip(groups, titles):

    if not group:
        continue

    h, e = head_eye_pairs(group, lambda R: head_v(R, head_attr), eye_key, sacc_subset)

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

eo_bins = [(0, 4), (5, 9), (10, 20)]
axis = "horizontal"
max_lag = 30  # frames (250 ms at 120 Hz)

head_attr = "yaw_v" if axis == "horizontal" else "pitch_v"
eye_key = "vx" if axis == "horizontal" else "vy"

groups, titles = eo_groups(Results, True, eo_bins)
fig, axes = create_subplot_grid(len(groups))
lags = np.arange(-max_lag, max_lag + 1)


def zscore_fill(v):
    v = v.copy()
    v[~np.isfinite(v)] = 0.0
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v


for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    for subset, color in (("all", "#444444"), ("non_saccade", "#725EE7")):

        acc = np.zeros(len(lags))
        n_used = 0
        for R in group:
            m = frame_mask(R, subset, "all")
            h = head_v(R, head_attr) * m
            for e in eye_arrays(R, eye_key, flip_RE_horizontal):
                hz = zscore_fill(h)
                ez = zscore_fill(e * m)
                cc = np.correlate(ez, hz, mode="full") / len(hz)
                mid = len(hz) - 1
                acc += cc[mid - max_lag: mid + max_lag + 1]
                n_used += 1

        cc = acc / n_used
        ax.plot(lags / FS * 1000, cc, color=color, lw=1, label=subset)
        peak_ms = lags[np.argmax(np.abs(cc))] / FS * 1000
        print(f"{title:10s} {subset:12s} peak lag = {peak_ms:+.1f} ms")

    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_title(title)
    ax.set_xlabel("eye lag re head (ms)")
    ax.set_ylabel("correlation")
    ax.legend(fontsize=5)
