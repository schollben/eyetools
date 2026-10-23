# %% exploratory: delayed visual experience — is it age, experience, or session noise?
# Delayed ferrets (403, 411, 416) were recorded once each at P40-43 with 1-2 days of vision (EO 1-2):
# AGE-matched to normal EO 8-20, EXPERIENCE-matched to normal EO 0-3. Per measure, a delayed session that
# looks like EO 0-3 points to a feature that needs visual experience; one that looks like EO 8-20 points
# to a feature that matures with age regardless of vision (intrinsic).
# Set your paths in local_config.py (copy local_config.py.example to get started): cd /Users/benjaminscholl/Documents/eyetools/
import sys
import warnings
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT)
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
import statsmodels.formula.api as smf
from utils.find_arcs import find_arcs
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, FS, FERRET_COLORS, unwrap_deg, set_style, save_fig,
                                       head_eye_windows, head_triggered_windows, head_signal, eye_signal, clean_runs, run_xcorr,
                                       session_rate, pooled_intervals)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False

# OTHER SETTINGS (same values as eye_head_v2 / vor_v2)
flip_eye = "LE"          # head frame: eye + = head yaw +
pair_window = 30         # frames an eye saccade may lead its head saccade
speed_threshold, min_bout = 20, 30   # running_mask, for gaze rate / ISI ("all" state uses neither)
head_still_deg = 2       # |head yaw displacement| below this = eye-only gaze shift
min_n = 20               # minimum events for a per-session median / main-sequence fit
yaw_jump = 10            # deg/frame: a larger single-frame yaw step is a head-pose flip (event excluded)
EYE_AXIS = {"h": {"LE": (-0.47, -0.52, 0.71), "RE": (-0.47, 0.52, 0.71)},
            "v": {"LE": (0.74, -0.68, 0.0),   "RE": (-0.73, -0.68, 0.0)}}   # head velocity onto each eye's axes
gaze_pad = (12, 20)      # frames excluded before gaze onset / after its peak (fixation frames)
nan_margin = 36          # frames excluded around each eye's tracking gaps
min_frames = 500         # minimum frames for a gain / cross-correlation
max_lag = 15             # frames, eye-head cross-correlation
gain_max_head = 180      # deg/s, fixation VOR gain fit range
sacc_pad = (6, 6)        # frames excluded around eye saccades for the in-head-saccade gain (6 Hz filtered eye)
n_draw = 5000            # null draws


def pkv_at_10(amp, pkv):
    """Peak velocity at 10 deg from an OLS fit of log10 pkv on log10 amplitude (NaN with < min_n events)."""
    k = np.isfinite(amp) & np.isfinite(pkv) & (amp > 0) & (pkv > 0)
    if k.sum() < min_n:
        return np.nan
    s, b = np.polyfit(np.log10(amp[k]), np.log10(pkv[k]), 1)
    return 10 ** (s + b)


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.median(x) if len(x) >= min_n else np.nan


def head_onto_eye(R, axis, eye):
    return sum(a * head_signal(R, n) for a, n in zip(EYE_AXIS[axis][eye], ("roll_v", "pitch_v", "yaw_v")))


def slope(x, y, k):
    k = k & np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[k], y[k], 1)[0] if k.sum() >= min_frames else np.nan


# %% per-session measures, every session (normal and delayed)
# METHODS: head = R.df_head, amplitude / peak velocity from unwrapped yaw (+ pitch). Gaze shifts =
# R.df_LEgaze + R.df_REgaze, amplitude from unwrapped gaze, head_contrib = head yaw / gaze horizontal
# displacement (onset -> peak), as in eye_head_v2. eye_lag_ms = median lag of the FIRST eye saccade
# (either eye) paired with each head saccade (eye onset - head onset; the eye may lead by <= pair_window);
# frac_head_w_eye = fraction of head saccades with one. eye_at_150ms / eye_peak_ms = session-mean
# head-onset-triggered eye position (eye_head_v2 cell B: sign-aligned, both eyes) at +150 ms, and the time of
# its peak after head onset. frac_sacc_by_150 = fraction of (head saccade, eye) windows with an eye-saccade onset
# in [-pair_window, +150 ms] of head onset; eye150_sacc / eye150_no_sacc = mean eye position at +150 ms in
# windows with / without one (without = slow phase only). gain_in_head_sacc = slope of eye on head velocity
# (head reprojected onto each eye's horizontal axis) over frames inside head saccades [onset, peak] and
# outside that eye's saccades (+- sacc_pad): stabilization DURING the head movement, before the eye
# saccade. Fixation measures (gain_h, gain_v, xcorr) as in vor_v2; counter_pitch = slope of vertical
# eye position on head pitch over fixation frames. pscr_gain as in eye_head_v2 (saccades in turns, with
# the head). All sessions use the same settings.

rows = []
for R in Results:
    yaw, pitch = unwrap_deg(R.yaw), np.asarray(R.pitch, float)
    jump = np.abs(np.diff(yaw)) > yaw_jump   # head-pose flips, events dropped as in eye_head_v2
    H = R.df_head.loc[[not jump[max(o - 5, 0):p + 5].any()
                   for o, p in zip(R.df_head.onset.astype(int), R.df_head.peak.astype(int))]].reset_index(drop=True)
    on, pk = H["onset"].to_numpy().astype(int), H["peak"].to_numpy().astype(int)
    H["amplitude_deg"] = np.hypot(yaw[pk] - yaw[on], pitch[pk] - pitch[on])
    H["peak_velocity_deg_s"] = np.rad2deg(H["peak_velocity_deg_s"].astype(float))
    E, _ = head_eye_windows(R, H, flip_eye, 24, 72, pair_window, 24, yaw_jump)
    first = E[E.paired].sort_values("onset").groupby("head_idx").first()
    T = head_triggered_windows(R, H, flip_eye, 24, 144, yaw_jump)
    hk = np.all(np.isfinite(T["head_pos"]), axis=1)
    ev = np.vstack([T[k][hk & np.all(np.isfinite(T[k]), axis=1)] for k in ("LE_pos", "RE_pos")])
    eye_tr = ev.mean(0) if len(ev) >= min_n else np.full(168, np.nan)   # session-mean eye trace, head onset at 24
    early = {True: [], False: []}   # eye position at +150 ms, split by an eye saccade onset in [-pair_window, +150 ms]
    for eye, df in (("LE", R.df_LE), ("RE", R.df_RE)):
        es = df["onset"].to_numpy().astype(int)
        for o, x in zip(on, T[f"{eye}_pos"]):
            if np.isfinite(x[:24 + 19]).all():
                early[bool(((es >= o - pair_window) & (es <= o + 18)).any())].append(x[24 + 18])

    turn = np.zeros(len(yaw), bool)
    arcs = find_arcs(R)
    for s, e in zip(arcs["onset"], arcs["end"]):
        turn[s:e] = True
    E["in_turn"] = turn[E["onset"].to_numpy()]
    pscr = E[E.in_turn & E.clean & (np.sign(E.eye_disp) == E.head_sign)]

    gaze = []
    for eye, df in (("LE", R.df_LEgaze), ("RE", R.df_REgaze)):
        df = df.loc[[not jump[max(o - 5, 0):p + 5].any() for o, p in zip(df.onset.astype(int), df.peak.astype(int))]]
        g_on, g_pk = df["onset"].to_numpy().astype(int), df["peak"].to_numpy().astype(int)
        gaze_h = unwrap_deg(getattr(R, f"{eye}_gaze_horizontal_deg")) * (-1 if eye == "LE" else 1)
        gaze_v = np.asarray(getattr(R, f"{eye}_gaze_vertical_deg"), float)
        gaze.append(pd.DataFrame(dict(gaze_disp=gaze_h[g_pk] - gaze_h[g_on], head_disp=yaw[g_pk] - yaw[g_on],
                                      amp=np.hypot(gaze_h[g_pk] - gaze_h[g_on], gaze_v[g_pk] - gaze_v[g_on]),
                                      pkv=df["peak_velocity_deg_s"].to_numpy(float))))
    gaze = pd.concat(gaze, ignore_index=True)
    gaze_isi = pooled_intervals([R], "gaze", "all", speed_threshold, min_bout)

    eye_sacc = pd.concat([R.df_LE, R.df_RE], ignore_index=True)

    # fixation frames per eye (vor_v2), and in-head-saccade, out-of-eye-saccade frames per eye
    gs = np.zeros(len(R.LE_vx), bool)
    for df in (R.df_LEgaze, R.df_REgaze):
        for o, p in zip(df.onset, df.peak):
            gs[max(0, o - gaze_pad[0]):p + gaze_pad[1] + 1] = True
    in_hs = np.zeros(len(R.LE_vx), bool)
    for o, p in zip(on, pk):
        in_hs[o:p + 1] = True
    fix, hs = {}, {}
    for eye, df in (("LE", R.df_LE), ("RE", R.df_RE)):
        ok = np.convolve(~np.isfinite(eye_signal(R, eye, "vx")), np.ones(2 * nan_margin + 1), "same") == 0
        fix[eye] = ~gs & ok
        es = np.zeros(len(R.LE_vx), bool)
        for o, p in zip(df.onset, df.peak):
            es[max(0, o - sacc_pad[0]):p + sacc_pad[1] + 1] = True
        hs[eye] = in_hs & ~es

    def pooled(axis, ekey, mask):
        h = np.concatenate([head_onto_eye(R, axis, eye)[mask[eye]] for eye in ("LE", "RE")])
        e = np.concatenate([eye_signal(R, eye, ekey, flip_eye)[mask[eye]] for eye in ("LE", "RE")])
        return h, e

    h, e = pooled("h", "vx", fix)
    gain_h = slope(h, e, np.abs(h) <= gain_max_head)
    h, e = pooled("v", "vy", fix)
    gain_v = slope(h, e, np.abs(h) <= gain_max_head)
    h, e = pooled("h", "vx", hs)
    gain_hs = slope(h, e, np.ones(len(h), bool))
    hp = np.concatenate([head_signal(R, "pitch")[fix[eye]] for eye in ("LE", "RE")])
    ey = np.concatenate([eye_signal(R, eye, "y", flip_eye)[fix[eye]] for eye in ("LE", "RE")])
    counter_pitch = slope(hp, ey, np.ones(len(hp), bool))

    acc, weight = np.zeros(2 * max_lag + 1), 0
    for eye in ("LE", "RE"):
        hh, ee = head_onto_eye(R, "h", eye), eye_signal(R, eye, "vx", flip_eye)
        for a, b in clean_runs(fix[eye] & np.isfinite(hh) & np.isfinite(ee), 4 * max_lag):
            cc = run_xcorr(hh[a:b], ee[a:b], max_lag)
            if cc is not None:
                acc += cc * (b - a)
                weight += b - a
    cc = acc / weight if weight >= min_frames else np.full(2 * max_lag + 1, np.nan)
    i_pk = np.nanargmax(np.abs(cc)) if np.isfinite(cc).any() else 0

    rows.append(dict(
        id=R.id, eo=R.eo, age=R.age, delayed=R.id in DELAYED_IDS,
        dur_min=len(yaw) / FS / 60,
        frac_eye_tracked=np.mean(np.isfinite(R.LE_x) | np.isfinite(R.RE_x)),
        # head
        head_rate=len(H) / (np.isfinite(R.yaw).sum() / FS),
        head_amp=med(H.amplitude_deg), head_pkv=med(H.peak_velocity_deg_s),
        # eye saccades
        eye_rate=session_rate(R, "eye", "all", speed_threshold, min_bout),
        eye_amp=med(eye_sacc.amplitude_deg), eye_pkv10=pkv_at_10(eye_sacc.amplitude_deg.to_numpy(float),
                                                                eye_sacc.peak_velocity_deg_s.to_numpy(float)),
        # gaze shifts
        gaze_rate=session_rate(R, "gaze", "all", speed_threshold, min_bout),
        gaze_isi=np.median(gaze_isi) if len(gaze_isi) >= min_n else np.nan,
        gaze_amp=med(gaze.amp), gaze_pkv=med(gaze.pkv), gaze_pkv10=pkv_at_10(gaze.amp.to_numpy(), gaze.pkv.to_numpy()),
        head_contrib=med(gaze.head_disp / gaze.gaze_disp),
        frac_eye_only=(gaze.head_disp.abs() < head_still_deg).mean() if len(gaze) >= min_n else np.nan,
        # eye-head coordination
        eye_lag_ms=med(first.lag_ms) if len(first) >= min_n else np.nan,
        frac_head_w_eye=len(first) / len(H) if len(H) >= min_n else np.nan,
        gain_in_head_sacc=gain_hs,
        eye_at_150ms=eye_tr[24 + 18],
        frac_sacc_by_150=len(early[True]) / max(len(early[True]) + len(early[False]), 1),
        eye150_sacc=np.mean(early[True]) if len(early[True]) >= min_n else np.nan,
        eye150_no_sacc=np.mean(early[False]) if len(early[False]) >= min_n else np.nan, eye_peak_ms=(np.nanargmax(eye_tr[24:]) / FS * 1000 if np.isfinite(eye_tr).any() else np.nan),
        pscr_gain=pscr.pscr_gain.median() if len(pscr) >= 3 else np.nan,
        # VOR, fixation
        gain_h=gain_h, gain_v=gain_v, counter_pitch=counter_pitch,
        xcorr_peak=np.abs(cc[i_pk]), xcorr_lag_ms=(i_pk - max_lag) / FS * 1000 if np.isfinite(cc).any() else np.nan))

S = pd.DataFrame(rows)
S["ref"] = np.select([S.delayed, S.eo <= 3, S.eo >= 8], ["delayed", "young", "old"], "mid")

families = {"head": ["head_rate", "head_amp", "head_pkv"],
            "eye saccades": ["eye_rate", "eye_amp", "eye_pkv10"],
            "gaze": ["gaze_rate", "gaze_isi", "gaze_amp", "gaze_pkv", "gaze_pkv10", "head_contrib", "frac_eye_only"],
            "eye-head": ["eye_lag_ms", "frac_head_w_eye", "eye_at_150ms", "frac_sacc_by_150", "eye150_sacc", "eye150_no_sacc", "eye_peak_ms", "gain_in_head_sacc", "pscr_gain"],
            "VOR (fixation)": ["gain_h", "gain_v", "counter_pitch", "xcorr_peak", "xcorr_lag_ms"]}
measures = [m for ms in families.values() for m in ms]

print(S[S.delayed | (S.ref == "old")][["id", "eo", "age", "dur_min", "frac_eye_tracked"]].round(2).to_string(index=False))


# %% stats: delayed vs experience-matched (EO 0-3) and age-matched (EO 8-20) normal sessions
# METHODS: delayed sessions are lumped (all sessions of all delayed ferrets, session = unit). Null for "is the
# delayed mean different from reference group X?": the mean of k sessions drawn without replacement from all
# normal sessions in X, any ferret (k = delayed sessions with a value), n_draw random draws.
# p = fraction of null means at least as far from the null centre as the delayed mean (two-sided).
# dev_p = LMM likelihood-ratio test, EO 0-3 vs EO 8-20 normal sessions, ferret random intercept: does the
# measure develop at all? maturity = (delayed - EO 0-3) / (EO 8-20 - EO 0-3): 0 = like experience-matched,
# 1 = like age-matched. separability = |EO 8-20 - EO 0-3| / SD of the EO 8-20 null: how many noise SDs apart
# the two reference groups are for a 3-session mean (< ~2: 3 sessions cannot tell age from experience).
# Exploratory: no correction over measures; 3 delayed ferrets, 1 session each.

rng = np.random.default_rng(0)


def null_means(ref, measure, k):
    v = ref[measure].dropna()
    return np.array([rng.choice(v.to_numpy(), k, replace=False).mean() for _ in range(n_draw)]), ref.id[v.index].nunique()


stats_rows, NULL = [], {}
for family, ms in families.items():
    for m in ms:
        d = S[S.delayed][m].dropna()
        row = dict(family=family, measure=m, delayed=d.mean(),
                   delayed_sessions=", ".join(f"{i}:{v:.3g}" for i, v in zip(S.id[d.index], d)))
        for ref in ("young", "old"):
            null, n_fer = null_means(S[S.ref == ref], m, len(d))
            NULL[m, ref] = null
            c = null.mean()
            row[ref] = c
            row[f"p_{ref}"] = np.mean(np.abs(null - c) >= np.abs(d.mean() - c))
            row[f"n_fer_{ref}"] = n_fer
        dv = S[S.ref.isin(["young", "old"])].dropna(subset=[m]).assign(gi=lambda x: (x.ref == "old").astype(int))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # few ferrets: convergence warnings are routine
            full = smf.mixedlm(f"{m} ~ gi", dv, groups=dv["id"]).fit(reml=False)
            null_fit = smf.mixedlm(f"{m} ~ 1", dv, groups=dv["id"]).fit(reml=False)
        row["dev_p"] = chi2.sf(2 * (full.llf - null_fit.llf), 1)
        row["maturity"] = (d.mean() - row["young"]) / (row["old"] - row["young"])
        row["separability"] = abs(row["old"] - row["young"]) / np.std(NULL[m, "old"])   # EO 8-20 - EO 0-3, in null SDs
        stats_rows.append(row)

stats = pd.DataFrame(stats_rows)


def verdict(r):
    if r.dev_p >= 0.05:
        return "delayed-specific (no normal development)" if min(r.p_old, r.p_young) < 0.05 else "no development, delayed normal"
    if r.p_old < 0.05 and r.p_young >= 0.05:
        return "EXPERIENCE (delayed ~ EO 0-3)"
    if r.p_old >= 0.05 and r.p_young < 0.05:
        return "AGE / intrinsic (delayed ~ EO 8-20)"
    if r.p_old < 0.05 and r.p_young < 0.05:
        return "delayed differs from both"
    return "undetermined (within noise of both)"


stats["verdict"] = stats.apply(verdict, axis=1)
pd.set_option("display.width", 250)
print(stats[["family", "measure", "young", "old", "delayed", "delayed_sessions", "dev_p", "p_young", "p_old",
             "maturity", "separability", "verdict"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


# %% figure: each measure vs EO, normal ferrets (lines) and delayed sessions (red)
ncol = 6
fig, axes = plt.subplots(int(np.ceil(len(measures) / ncol)), ncol, figsize=(2 * ncol, 1.9 * np.ceil(len(measures) / ncol)))
for ax in axes.ravel()[len(measures):]:
    ax.axis("off")
for ax, m in zip(axes.ravel(), measures):
    for fid, g in S[~S.delayed].groupby("id"):
        g = g.sort_values("eo")
        ax.plot(g.eo, g[m], color="0.6", lw=0.6, marker="o", ms=2, mfc="w")
    d = S[S.delayed]
    ax.scatter(d.eo, d[m], color=[FERRET_COLORS[i] for i in d.id], edgecolor="#d62728", lw=1, s=18, zorder=3)
    r = stats.set_index("measure").loc[m]
    ax.set_title(f"{m}\np(0-3)={r.p_young:.2g} p(8-20)={r.p_old:.2g}", fontsize=5)
    ax.set_xlabel("EO (days)", fontsize=5)
    ax.tick_params(labelsize=5)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "delayed measures vs EO")


# %% figure: delayed vs nulls, standardized by the EO 8-20 session SD
# Bars: 95% of 3-session null means (grey = EO 0-3, black = EO 8-20). Red = delayed mean, dots = sessions.
fig, ax = plt.subplots(figsize=(4, 0.22 * len(measures) + 0.8))
for y, m in enumerate(measures):
    sd = S[S.ref == "old"][m].std()
    c = NULL[m, "old"].mean()
    for ref, col, dy in (("young", "0.6", 0.15), ("old", "k", -0.15)):
        lo, hi = np.percentile((NULL[m, ref] - c) / sd, [2.5, 97.5])
        ax.plot([lo, hi], [y + dy] * 2, color=col, lw=2, solid_capstyle="butt")
    dv = (S[S.delayed][m].dropna() - c) / sd
    ax.scatter(dv, [y] * len(dv), s=6, color="#d62728", alpha=0.5)
    ax.scatter(dv.mean(), y, s=20, color="#d62728", marker="D")
ax.axvline(0, color="0.8", lw=0.5)
ax.set(yticks=range(len(measures)), yticklabels=measures, xlabel="(value - EO 8-20 centre) / EO 8-20 session SD")
ax.invert_yaxis()
ax.tick_params(labelsize=5)
sns.despine(fig)
fig.tight_layout()

if SAVE_FIGS:
    save_fig(fig, "delayed vs nulls standardized")
