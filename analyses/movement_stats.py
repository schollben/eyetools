# %% init
%load_ext autoreload
%autoreload 2
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
from utils import create_subplot_grid
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu as mwu
from scipy.stats import wilcoxon, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from analyses.helper_functions import (EYE_COLOR, AGE_COLORS, FS, EO_BINS, eo_groups, pooled_events,
                                       pooled_intervals, event_traces, session_rate, session_rates,
                                       load_results, set_style, session_trend)
set_style()

# LOAD DATA
Results = load_results()

# settings for every plot below
# how panels are split: False = one panel per session, True = one panel per EO range
pool_by_eo = True
eo_bins = EO_BINS

flip_eye = "RE"     
amp_bins = [0, 4, 8, 12, 20, 45]

speed_threshold = 50    # mm/s, locomotion
min_bout = 30            # frames
head_still_thresh = 20   # deg/s

# Two conditions throughout: everything, vs the quiet epochs (not locomoting AND head
# below head_still_thresh). The intermediate single-factor conditions are gone — they
# split the data without isolating a state worth naming.
conditions = ["all", "stationary_and_head_still"]
COND_COLORS = dict(zip(conditions, ["#444444", "#1B9E77"]))

pre, post = 12, 48       # frames: -100 to +400 ms from onset (cell 4 only)
bin_by = "amplitude"     # "amplitude" | "peak_velocity" (cell 4 only)

groups, titles = eo_groups(Results, pool_by_eo, eo_bins)



# %% 3. amplitude vs peak velocity by condition

signal = "eye"

fig, axes = create_subplot_grid(len(groups))

for ax, group, title in zip(axes, groups, titles):

    if not group:
        ax.set_title(title)
        continue

    for cond in conditions:
        amp = pooled_events(group, signal, cond, "amplitude_deg",
                            speed_threshold, min_bout, head_still_thresh)
        pkv = pooled_events(group, signal, cond, "peak_velocity_deg_s",
                            speed_threshold, min_bout, head_still_thresh)
        if len(amp) < 20:
            continue

        x = np.log10(np.abs(amp))
        y = np.log10(np.abs(pkv))
        ok = np.isfinite(x) & np.isfinite(y)

        sns.scatterplot(ax=ax, x=x[ok], y=y[ok], s=2, alpha=0.15,
                        color=COND_COLORS[cond])
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        xl = np.array([x[ok].min(), x[ok].max()])
        ax.plot(xl, slope * xl + intercept, color=COND_COLORS[cond], lw=1,
                label=f"{cond} ({slope:.2f})")

    ax.set_title(title)
    ax.set_xlabel("log10 amplitude (deg)")
    ax.set_ylabel("log10 peak velocity (deg/s)")
    ax.legend(fontsize=4)

# %% 4. mean kinematic traces
# Onset-aligned, so displacement traces start at zero by construction.

signal = "eye"      # "eye" | "gaze"
condition = "all"   # "all" | "stationary_and_head_still"
# "head_still" condition gates on head angular speed directly

t_ms = np.arange(-pre, post) / FS * 1000
bin_col = "amplitude_deg" if bin_by == "amplitude" else "peak_velocity_deg_s"
bin_edges = amp_bins if bin_by == "amplitude" else [0, 50, 100, 200, 400, 1000]

for kind in ("speed", "displacement"):

    fig, axes = create_subplot_grid(len(groups))

    for ax, group, title in zip(axes, groups, titles):

        if not group:
            ax.set_title(title)
            continue

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):

            traces, nominal = [], []
            for R in group:
                t, a = event_traces(R, signal, kind, lo, hi, bin_col, condition,
                                    pre, post, flip_eye,
                                    speed_threshold, min_bout, head_still_thresh)
                traces += t
                nominal += a

            if len(traces) < 10:
                continue
            arr = np.array(traces)
            m = arr.mean(axis=0)
            se = arr.std(axis=0) / np.sqrt(len(arr))
            line, = ax.plot(t_ms, m, lw=1, label=f"{lo}-{hi} (n={len(arr)})")
            ax.fill_between(t_ms, m - se, m + se, alpha=0.25)

        ax.set_title(f"{signal} {title}")
        ax.set_xlabel("time from onset (ms)")
        ax.set_ylabel("speed (deg/s)" if kind == "speed" else "displacement (deg)")
        ax.legend(fontsize=4)


# %% event rate (one point per session)
# NOT a sliding window. The old sliding-window rate (pooled_rates, removed) gated only which events were COUNTED while still
# emitting every NaN-free window and dividing by the full window length, so n was the
# window count (identical for every condition) and the rate was the all-condition rate
# scaled by the fraction of time in condition.
# Windowing cannot be repaired here: stationary_and_head_still bouts have a median length
# of ~0.06 s and only a handful reach 2 s, so no window both fits inside the condition and
# is long enough to estimate a ~1 Hz rate. session_rates divides in-condition events by
# in-condition TIME, which uses all the exposure however fragmented it is.
# One point per session, so n is sessions — the window count was pseudo-replication.

fig, axes = plt.subplots(1, 2, figsize=(6, 2))

for ax, signal in zip(axes, ("eye", "gaze")):

    for i, (group, title) in enumerate(zip(groups, titles)):

        if not group:
            continue

        for cond, dx in zip(conditions, (-0.15, 0.15)):

            rate = session_rates(group, signal, cond,
                                 speed_threshold, min_bout, head_still_thresh)
            if not len(rate):
                continue

            ax.plot(np.full(len(rate), i + dx), rate, "o", ms=3, alpha=0.5,
                    color=COND_COLORS[cond])
            ax.plot(i + dx, np.median(rate), "_", ms=12, mew=2,
                    color=COND_COLORS[cond])

            print(f"{signal:5s} {title:10s} {cond:26s} n_sesh={len(rate):3d}  "
                  f"median={np.median(rate):5.2f} Hz  "
                  f"IQR={np.subtract(*np.percentile(rate, [75, 25])):5.2f}")

    ax.set_xticks(range(len(titles)))
    ax.set_xticklabels(titles)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"{signal} event rate (Hz)")

sns.despine(fig)
fig.tight_layout()

# --- stats -------------------------------------------------------------------
# Sessions are the unit. session_trend gives the pooled-session Spearman, a mixed model with
# ferret as a random effect (pooling treats 33 sessions from 4 ferrets as independent when
# they are not; F407/F420 contribute 23 of them) and per-ferret rows. The all-vs-quiet test
# is PAIRED: both rates come from the same session. Rates count only tracked-eye time.

for signal in ("eye", "gaze"):

    print(f"\n--- {signal} ---")

    S = pd.DataFrame(dict(id=[R.id for R in Results], eo=[R.eo for R in Results]))
    for cond, col in zip(conditions, ("rate_all", "rate_quiet")):
        S[col] = [session_rate(R, signal, cond, speed_threshold, min_bout, head_still_thresh)
                  for R in Results]
        session_trend(S, col, f"EO trend {cond}")

        vals = [session_rates(g, signal, cond, speed_threshold, min_bout,
                              head_still_thresh) for g in groups if g]
        if len(vals) > 2:
            print(f"  {'':22s} Kruskal p={kruskal(*vals)[1]:.4g}  "
                  f"youngest-vs-oldest MWU p={mwu(vals[0], vals[-1])[1]:.4g}")

    # paired within session: does the quiet state change rate?
    a, b = S.rate_all.to_numpy(), S.rate_quiet.to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    print(f"  paired all vs quiet  n={ok.sum():3d}  "
          f"median {np.median(a[ok]):.2f} vs {np.median(b[ok]):.2f} Hz  "
          f"diff={np.median(a[ok] - b[ok]):+.3f}  Wilcoxon p={wilcoxon(a[ok], b[ok])[1]:.4g}")


# %% inter-event interval distributions (timing, not magnitude)
# Interval = peak of one event -> onset of the next: the quiescent gap between movements.

condition = "stationary_and_head_still"   # "all" | "stationary_and_head_still"

bins = np.logspace(np.log10(10), np.log10(10000), 30)

fig, axes = plt.subplots(1, 2, figsize=(6, 2))

for ax, signal in zip(axes, ("eye", "gaze")):

    for i, (group, title) in enumerate(zip(groups, titles)):

        if not group:
            continue

        isi = pooled_intervals(group, signal, condition,
                               speed_threshold, min_bout, head_still_thresh)

        sns.histplot(ax=ax, x=isi, bins=bins, element="step", fill=False,
                     stat="probability", color=AGE_COLORS[i],
                     label=f"{title} (n={len(isi)})")

        print(f"{signal:5s} {title:10s} n={len(isi):6d}  "
              f"median={np.median(isi):7.0f} ms  "
              f"IQR={np.subtract(*np.percentile(isi, [75, 25])):7.0f}  ")

    ax.set_xscale("log")
    ax.set_xlabel(f"{signal} inter-event interval (ms)")
    ax.legend(fontsize=6)

sns.despine(fig)
fig.tight_layout()

