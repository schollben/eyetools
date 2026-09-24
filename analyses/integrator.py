# %% main script to run data loading, cleaning, and saccade extraction for a session
# main init
# Set your paths in local_config.py (copy local_config.py.example to get started).
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
from utils import create_subplot_grid, non_saccade_mask
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import (load_results, set_style, EO_BINS,FS, EYE_COLORS, eo_groups, fit_line, clean_runs,
                                       eye_signal, drift_frames, drift_by_position)
set_style()

# LOAD DATA
Results = load_results()


# %% settings for every plot below
# The integrator holds the eye at an eccentric position. A LEAKY integrator lets the eye
# drift back toward center, so drift velocity should be negative when position is positive.
# Slope of velocity vs position is -1/tau.
#
# TODO — CLEANER FRAME SELECTION (not yet implemented)
# The first-pass results that motivated this list (no centripetal bias, unstable tau) used
# the stored vx, which is VERTICAL velocity (fixed in eye_signal). Re-run cells 2-6 before
# deciding which filters are still needed.
# Candidate filters, roughly in order of expected payoff:
#
# 1. Head-velocity gate. Every frame here still includes head motion, so VOR-driven eye
#    velocity is being counted as integrator drift. Gate on R.angVelocities (rad/s in the
#    csv — convert) below some ceiling to isolate true fixation. Biggest likely confound.
# 2. Position stability within the window. A run spanning a slow drift PLUS a small
#    unlabeled saccade is currently fit as one thing. Require low position variance across
#    the window, or reject runs whose endpoints differ by more than a few degrees.
# 3. Blink / tracking-artifact residue. removeBadData masks on eye quality but fast junk
#    still gets through — that is what vel_ceiling is patching.
# 4. Minimum hold duration. Any contiguous run currently contributes to the binned estimate;
#    requiring ~0.5 s of continuous clean fixation first would bias toward genuine holds.
#
# Each would go in as a flag in this cell and a new axis in the cell 6 sweep.

pool_by_eo = True
eo_bins = EO_BINS

# Each eye's _x is its own adduction angle, i.e. NASAL/TEMPORAL coordinates. Drift is
# toward each eye's own orbital center, so no flip is applied here. (vor.py flips one eye
# to put both in a common conjugate frame — that is the right choice there, not here.)
flip_eye = None

pad_pre = 3          # frames before saccade onset excluded
pad_post = 30        # frames after saccade peak excluded (non_saccade_mask default is 12)
vel_ceiling = 20     # deg/s; drop residual fast frames the saccade detector missed
min_run = 60         # frames, shortest contiguous clean stretch (cell 4)
pos_bins = np.arange(-10, 10, 2)   # signed, never folded to |x|
EYES = ("LE", "RE")

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)

for R in Results:
    n = len(R.LE_vx)
    counts = [len(drift_frames(R, eye, pad_post, pad_pre, vel_ceiling, flip_eye)[0])
              for eye in EYES]


# %% 1. eye position vs velocity (phase plane), one figure per eye
# The pooled slope here is a WEAK estimator — single-frame velocity is noisy.
# Cell 2 (binned drift) is the real measure.

for eye in EYES:

    fig, axes = create_subplot_grid(len(groups))

    for ax, group, title in zip(axes, groups, titles):

        if not group:
            ax.set_title(title)
            continue

        xs, vs = [], []
        for R in group:
            x, v, _ = drift_frames(R, eye, pad_post, pad_pre, vel_ceiling, flip_eye)
            xs.append(x - np.median(x) if len(x) else x)
            vs.append(v)
        x = np.concatenate(xs)
        v = np.concatenate(vs)

        sns.scatterplot(ax=ax, x=x, y=v, s=2, alpha=0.15, color=EYE_COLORS[eye])
        slope, _ = fit_line(ax, x, v, color="k")
        tau = -1 / slope if slope < 0 else np.nan

        ax.set_title(f"{eye} {title}  slope={slope:+.3f}  tau={tau:.1f}s")
        ax.set_xlabel("eye position (deg, nasal + / temporal -)")
        ax.set_ylabel("eye velocity (deg/s)")

    fig.suptitle(f"{eye} phase plane", y=1.02)


# %% 2. drift velocity vs eye position (PRIMARY MEASURE)
# Averaging many frames per position bin removes the single-frame noise that swamps cell 1.

fig, ax = plt.subplots(figsize=(3.5, 2.5))

for group, title in zip(groups, titles):

    if not group:
        continue

    centers, means, sems, counts = drift_by_position(
        group, EYES, pos_bins, pad_post, pad_pre, vel_ceiling, flip_eye)

    ok = counts > 100
    ax.errorbar(centers[ok], means[ok], yerr=sems[ok], marker="o", ms=3, lw=1,
                capsize=2, label=title)

    slope, _ = np.polyfit(centers[ok], means[ok], 1, w=np.sqrt(counts[ok]))
    tau = -1 / slope if slope < 0 else np.nan
    print(f"\n{title}  slope={slope:+.4f}/s  tau={tau:.1f}s")
    for c, m, s, n in zip(centers, means, sems, counts):
        flag = "" if n > 100 else "  (low n, excluded from fit)"
        print(f"   pos {c:+6.1f} deg  n={n:7d}  mean={m:+7.3f}  sem={s:6.3f}{flag}")

ax.axhline(0, color="0.8", lw=0.5)
ax.axvline(0, color="0.8", lw=0.5)
ax.set_xlabel("eye position (deg, nasal + / temporal -)")
ax.set_ylabel("drift velocity (deg/s)")
ax.legend(fontsize=5)
sns.despine(fig)


# %% 3. centripetal vs centrifugal drift
# Centripetal = moving toward center (sign of velocity opposite sign of position).
# A leaky integrator is centripetally biased; a perfect one sits at 0.5.

fig, ax = plt.subplots(figsize=(3, 2))
fracs, labels = [], []

for group, title in zip(groups, titles):

    if not group:
        continue

    xs, vs = [], []
    for R in group:
        for eye in EYES:
            x, v, _ = drift_frames(R, eye, pad_post, pad_pre, vel_ceiling, flip_eye)
            xs.append(x - np.median(x) if len(x) else x)
            vs.append(v)
    x = np.concatenate(xs)
    v = np.concatenate(vs)

    ecc = np.abs(x) > 1.0                      # ignore frames near center, sign is noise
    centripetal = np.sign(v) != np.sign(x)
    frac = centripetal[ecc].mean()
    fracs.append(frac)
    labels.append(title)

    toward = np.abs(v[ecc & centripetal]).mean()
    away = np.abs(v[ecc & ~centripetal]).mean()
    print(f"{title:10s} centripetal={frac:.3f}  |v| toward={toward:.2f}  away={away:.2f}"
          f"  n={ecc.sum()}")

sns.barplot(ax=ax, x=labels, y=fracs)
ax.axhline(0.5, color="k", ls="--", lw=0.5)
ax.set_ylabel("fraction centripetal")
ax.set_ylim(0.4, 0.65)
ax.tick_params(axis="x", rotation=45)
sns.despine(fig)
fig.tight_layout()


# %% 4. per-run slope distribution (SECONDARY)
# Velocity vs position fit WITHIN each saccade-free run. Kept to show run-to-run
# variability, NOT to estimate tau. Cell 2 is the measure to trust.

fig, ax = plt.subplots(figsize=(3.5, 2.5))

for group, title in zip(groups, titles):

    if not group:
        continue

    slopes = []
    for R in group:
        for eye in EYES:
            x, v, m = drift_frames(R, eye, pad_post, pad_pre, vel_ceiling, flip_eye)
            xf = eye_signal(R, eye, "x", flip_eye)
            vf = eye_signal(R, eye, "vx", flip_eye)
            for a, b in clean_runs(m, min_run):
                xx, vv = xf[a:b], vf[a:b]
                if xx.std() < 0.1:
                    continue
                slopes.append(np.polyfit(xx, vv, 1)[0])

    slopes = np.array(slopes)
    if len(slopes) < 10:
        print(f"{title:10s} only {len(slopes)} runs, skipped")
        continue

    sns.histplot(ax=ax, x=np.clip(slopes, -5, 5), bins=50, element="step",
                 fill=False, stat="density", label=title)
    print(f"{title:10s} n_runs={len(slopes):5d}  median={np.median(slopes):+.3f}  "
          f"{100*(slopes < 0).mean():.0f}% negative")

ax.axvline(0, color="k", ls="--", lw=0.5)
ax.set_xlabel("per-run slope (1/s)")
ax.legend(fontsize=5)
sns.despine(fig)


# %% 5. drift measures vs EO (developmental summary)
# One point per session per eye, so LE/RE agreement is visible — a large mismatch flags a
# tracking problem rather than biology.

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2))
min_frames = 500

for R in Results:
    for eye in EYES:

        x, v, _ = drift_frames(R, eye, pad_post, pad_pre, vel_ceiling, flip_eye)
        if len(x) < min_frames:
            print(f"skip ferret {R.id} EO{R.eo} {eye}: only {len(x)} frames")
            continue

        centers, means, sems, counts = drift_by_position(
            [R], [eye], pos_bins, pad_post, pad_pre, vel_ceiling, flip_eye)

        ok = counts > 100
        if ok.sum() < 3:
            print(f"skip ferret {R.id} EO{R.eo} {eye}: only {ok.sum()} usable bins")
            continue

        slope, _ = np.polyfit(centers[ok], means[ok], 1, w=np.sqrt(counts[ok]))
        tau = -1 / slope if slope < 0 else np.nan

        xc = x - np.median(x)
        ecc = np.abs(xc) > 1.0
        frac = (np.sign(v) != np.sign(xc))[ecc].mean()

        band = (np.abs(xc) >= 3) & (np.abs(xc) < 9)
        drift = np.abs(v[band]).mean() if band.sum() > 100 else np.nan

        for ax, val in zip(axes, (tau, frac, drift)):
            ax.plot(R.eo, val, "o", ms=4, color=EYE_COLORS[eye], alpha=0.7)

for ax, lbl in zip(axes, ("tau (s)", "fraction centripetal", "mean |drift| 3-9 deg (deg/s)")):
    ax.set_xlabel("EO day")
    ax.set_ylabel(lbl)
axes[1].axhline(0.5, color="k", ls="--", lw=0.5)
sns.despine(fig)
fig.tight_layout()


# %% 6. threshold sensitivity
# "Saccade-free" is a judgement call. Sweep the two thresholds that define it and check the headline numbers do not swing wildly.

pad_sweep = [12, 24, 48, 72]
vel_sweep = [None, 30, 20, 10]

taus = np.full((len(pad_sweep), len(vel_sweep)), np.nan)
print("pad_post  vel_ceil |    frames |   tau (s) | centripetal")

for i, pp in enumerate(pad_sweep):
    for j, vc in enumerate(vel_sweep):

        centers, means, sems, counts = drift_by_position(
            Results, EYES, pos_bins, pp, pad_pre, vc, flip_eye)

        ok = counts > 100
        if ok.sum() < 3:
            print(f"{pp:8d}  {str(vc):8s} | too few usable bins")
            continue

        slope, _ = np.polyfit(centers[ok], means[ok], 1, w=np.sqrt(counts[ok]))
        tau = -1 / slope if slope < 0 else np.nan
        taus[i, j] = tau

        xs, vs = [], []
        for R in Results:
            for eye in EYES:
                x, v, _ = drift_frames(R, eye, pp, pad_pre, vc, flip_eye)
                xs.append(x - np.median(x) if len(x) else x)
                vs.append(v)
        x = np.concatenate(xs)
        v = np.concatenate(vs)
        ecc = np.abs(x) > 1.0
        frac = (np.sign(v) != np.sign(x))[ecc].mean()

        print(f"{pp:8d}  {str(vc):8s} | {counts.sum():9d} | {tau:9.1f} | {frac:11.3f}")

fig, ax = plt.subplots(figsize=(3, 2.5))
im = ax.imshow(taus, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(vel_sweep)), [str(v) for v in vel_sweep])
ax.set_yticks(range(len(pad_sweep)), [str(p) for p in pad_sweep])
ax.set_xlabel("velocity ceiling (deg/s)")
ax.set_ylabel("pad_post (frames)")
ax.set_title("tau (s)")
fig.colorbar(im, ax=ax)
fig.tight_layout()
