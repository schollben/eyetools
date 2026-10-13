# %% main script to run data loading, cleaning, and saccade extraction for a session
# notes:
# Pupil = (major + minor) / 2 of the pupil ellipse (mm, utils/load_eye_kinematics). 
# Apparent size depends on eye position (foreshortening, ellipse fit)
# so the pupil is used only with the eye near the camera axis (+/- eye_max).
# 
# THIS SCRIPT IS OVERLY COMPLEX AND MAY REQUIRE REFACTORING
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
from analyses.helper_functions import (NORMAL_IDS, DELAYED_IDS, delayed_groups, FS, EO_BINS, AGE_COLORS, EYE_COLORS,
                                       LOCO_COLORS, eo_groups, head_signal, eye_signal, running_mask,
                                       set_style, save_fig, plot_mean_se, plot_vs_eo)
set_style()

# LOAD SAVED DATA
with open("./analyses/results.pkl", "rb") as f:
    Results = pickle.load(f)

# SETTINGS
SAVE_FIGS = False
compare_delayed = True   # True: normal EO 8-20 vs. all delayed sessions
fig_tag = " delayed" if compare_delayed else ""   # appended to saved figure names

# OTHER SETTINGS
eye_max = 5              # deg, pupil used only where |x| and |y| < eye_max (that eye); 10 = sensitivity check
nan_margin = 36          # frames excluded around each eye's tracking gaps (blinks / lid closure; as vor_v2)
outlier_sd = 5           # robust SDs (1.4826 x MAD) from the eye's session median -> NaN (as HMM/run_hmm_eye)
win = 120                # frames (1 s), pupil bins
min_valid = 0.5          # fraction of a bin's frames with a valid pupil, else the bin is NaN
state_frac = 0.5         # a bin's state = the state of > state_frac of its frames, else "mixed" (turns are short)
speed_threshold = 50     # mm/s, running_mask (as vor_v2)
min_bout = 60            # frames, shortest run of frames counted as running
min_bins = 20            # minimum valid bins for a session's value
min_state_bins = 10      # minimum valid bins per state for a session's state measures
min_bin = 5              # minimum valid bins for a session's value within one speed / head-speed bin
speed_edges = np.array([0, 25, 50, 100, 200, 400, 800])   # mm/s, pupil vs body speed
head_edges = np.array([0, 10, 20, 40, 80, 160])            # deg/s, pupil vs head angular speed (stationary bins)
trig_pre, trig_post = 3, 5   # bins (s) before / after a state onset
base_s = 2               # s, quiet period required before an onset (and baseline window)
min_onsets = 3           # minimum onsets for a session's onset-triggered trace
example = (420, 8)       # (ferret, EO) for the example-session cell
STATES = ("stationary", "running", "turning")
STATE_COLORS = {**LOCO_COLORS, "turning": "#F2A900"}

if compare_delayed:
    groups, titles = delayed_groups(Results, normal_ids=NORMAL_IDS, delayed_ids=DELAYED_IDS)
    GROUP_COLORS = ["#222222", "#d62728"]
else:
    groups, titles = eo_groups(Results, pool_by_eo=True, eo_bins=EO_BINS, normal_ids=NORMAL_IDS)
    GROUP_COLORS = AGE_COLORS

ARCS = {id(R): find_arcs(R) for R in Results}   # turns (as eye_head_v2)
for group, title in zip(groups, titles):
    print(f"{title}: {len(group)} sessions, {len(set(R.id for R in group))} animals")


# %% pupil gating, locomotor states, 1 s bins
# METHODS: per eye, the pupil is NaN where |eye x| or |eye y| >= eye_max deg, within nan_margin frames of
# that eye's tracking gaps (eye position or pupil NaN), or beyond outlier_sd robust SDs (1.4826 x MAD) of
# that eye's session median. Each eye is expressed as % of its own session median (removes the LE/RE
# camera scale difference); the two eyes are averaged where both are valid, else the valid eye is used.
# pupil_mm reference = mean over valid eyes of the per-eye session median (mm).
# METHODS: states per frame, mutually exclusive: turning = inside an arc of the 2D body path
# (utils/find_arcs.find_arcs, as eye_head_v2: sustained curvature while moving > 50 mm/s, >= 0.5 s, >= 30
# deg); running = running bout (body speed > speed_threshold for >= min_bout frames) and not turning;
# stationary = body speed tracked, not running, not turning.
# METHODS: bins = non-overlapping win-frame (1 s) windows. Bin pupil = mean of valid frames (NaN with <
# min_valid valid); body speed, head angular speed = bin means; state = the state of > state_frac of the
# bin's frames, else "mixed" (excluded from state comparisons). ecc = mean eye eccentricity (deg) of the
# valid-pupil frames (both eyes), to check that states do not differ in eye position within the gate.
def bin_mean(a, nb):
    """Mean of finite values per win-frame bin, and the fraction of finite frames per bin."""
    a = np.asarray(a, float)[:nb * win].reshape(nb, win)
    k = np.isfinite(a)
    return np.where(k.any(1), np.where(k, a, 0).sum(1) / np.maximum(k.sum(1), 1), np.nan), k.mean(1)


PUP, REF, MASK, B = {}, {}, {}, {}
rows = []
for R in Results:
    n = len(R.LE_x)
    pct, ecc, gated, meds = [], [], {}, []
    for eye in ("LE", "RE"):
        p = np.asarray(getattr(R, f"{eye}_pupil"), float).copy()
        x, y = eye_signal(R, eye, "x"), eye_signal(R, eye, "y")
        gap = ~np.isfinite(x) | ~np.isfinite(p)
        p[(np.abs(x) >= eye_max) | (np.abs(y) >= eye_max) | np.isnan(x) | np.isnan(y)
          | (np.convolve(gap, np.ones(2 * nan_margin + 1), "same") > 0)] = np.nan
        if np.isfinite(p).sum() >= min_bins * win:
            med = np.nanmedian(p)
            p[np.abs(p - med) > outlier_sd * 1.4826 * np.nanmedian(np.abs(p - med))] = np.nan
            meds.append(np.nanmedian(p))
            pct.append(100 * p / meds[-1])
        else:
            p[:] = np.nan
            pct.append(p.copy())
        gated[eye] = p
        ecc.append(np.where(np.isfinite(p), np.hypot(x, y), np.nan))
    pct, ecc = np.vstack(pct), np.vstack(ecc)
    k = np.isfinite(pct)
    PUP[id(R)] = np.where(k.any(0), np.where(k, pct, 0).sum(0) / np.maximum(k.sum(0), 1), np.nan)
    REF[id(R)] = np.mean(meds) if meds else np.nan

    turn = np.zeros(n, bool)
    for s, e in zip(ARCS[id(R)]["onset"], ARCS[id(R)]["end"]):
        turn[s:e] = True
    run_all = running_mask(R, speed_threshold, min_bout)
    sp = np.asarray(R.speed, float)
    M = {"stationary": np.isfinite(sp) & ~run_all & ~turn, "running": run_all & ~turn, "turning": turn}
    MASK[id(R)] = dict(M, run_all=run_all, gated=gated)

    nb = n // win
    pup, frac = bin_mean(PUP[id(R)], nb)
    pup[frac < min_valid] = np.nan
    st = np.full(nb, "mixed", object)
    for s in STATES:
        st[M[s][:nb * win].reshape(nb, win).mean(1) > state_frac] = s
    ke = np.isfinite(ecc)
    B[id(R)] = pd.DataFrame(dict(t=np.arange(nb) * win / FS, pupil=pup, speed=bin_mean(sp, nb)[0],
                                 head_speed=bin_mean(head_signal(R, "speed"), nb)[0], state=st,
                                 ecc=bin_mean(np.where(ke.any(0), np.where(ke, ecc, 0).sum(0) / np.maximum(ke.sum(0), 1), np.nan), nb)[0]))

    b = B[id(R)].dropna(subset=["pupil"])
    rows.append(dict(id=R.id, eo=R.eo, **{f"valid_{eye}": 100 * np.isfinite(gated[eye]).mean() for eye in ("LE", "RE")},
                     valid=100 * np.isfinite(PUP[id(R)]).mean(), ref_mm=REF[id(R)],
                     **{f"valid_{s[:4]}": 100 * np.isfinite(PUP[id(R)][M[s]]).mean() if M[s].any() else np.nan for s in STATES},
                     **{f"bins_{s[:4]}": (b.state == s).sum() for s in STATES},
                     **{f"ecc_{s[:4]}": b.ecc[b.state == s].median() for s in STATES}))
print(pd.DataFrame(rows).round(1).to_string(index=False))


# %% onset-triggered pupil (per-session traces, used by the table and the onset figure)
# METHODS: run onsets = starts of running bouts (running_mask) preceded by base_s s of stationary frames
# (not running, not turning). Turn onsets = arc onsets preceded by base_s s without turning (the animal may
# be running before a turn). Per onset, pupil (% of session median) and body speed are averaged in 1 s bins
# from -trig_pre to +trig_post s (pupil bins NaN with < min_valid valid frames). Session trace = mean over
# onsets per bin (NaN-aware), minus its -base_s..0 s baseline; NaN with < min_onsets onsets.
# run_on_dp / turn_on_dp = mean change at +1..+3 s.
tb = np.arange(-trig_pre, trig_post)   # bin start (s)
TRIG = {}
for R in Results:
    M, pc, sp = MASK[id(R)], PUP[id(R)], np.asarray(R.speed, float)
    run_on = np.flatnonzero(np.diff(M["run_all"].astype(int), prepend=0) == 1)
    quiet = {"run": M["stationary"], "turn": ~M["turning"]}
    TRIG[id(R)] = {}
    for kind, ons in (("run", run_on), ("turn", np.asarray(ARCS[id(R)]["onset"], int))):
        ons = [o for o in ons if o - trig_pre * win >= 0 and o + trig_post * win <= len(pc)
               and quiet[kind][o - base_s * win:o].all()]
        P_, S_ = [], []
        for o in ons:
            p, f = bin_mean(pc[o - trig_pre * win:o + trig_post * win], trig_pre + trig_post)
            P_.append(np.where(f >= min_valid, p, np.nan))
            S_.append(bin_mean(sp[o - trig_pre * win:o + trig_post * win], trig_pre + trig_post)[0])
        P_, S_ = np.array(P_).reshape(-1, len(tb)), np.array(S_).reshape(-1, len(tb))
        ok = np.isfinite(P_).sum(0) >= min_onsets
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # all-NaN bins
            tr = np.where(ok, np.nanmean(P_, 0), np.nan) if len(P_) else np.full(len(tb), np.nan)
            spd = np.nanmean(S_, 0) if len(S_) else np.full(len(tb), np.nan)
        tr = tr - tr[(tb >= -base_s) & (tb < 0)].mean()
        TRIG[id(R)][kind] = dict(pupil=tr, speed=spd, n=len(P_),
                                 dp=tr[(tb >= 1) & (tb < 3)].mean())


# %% per-session table + trends vs EO (sessions are the unit; one line per ferret)
# METHODS: valid_pct = % of frames with a valid pupil (either eye); n_bins = valid 1 s bins. Session
# measures are NaN with < min_bins valid bins (< min_state_bins per state for state measures).
# METHODS (size): pupil_mm = reference size (mean over valid eyes of the per-eye session median, mm);
# pupil_mm_stat = pupil_mm x median stationary bin pupil / 100; pupil_cv = SD / mean of the bin pupil (%).
# CAVEAT: absolute mm depends on camera distance / magnification per eye and session; age differences in
# mm are only interpretable if camera geometry was constant across sessions.
# METHODS (state): mean bin pupil per state, as % of the session median; d_run = (running - stationary),
# d_turn = (turning - stationary), d_turn_run = (turning - running), each in % of the stationary mean.
# METHODS (coupling): r_speed = Pearson r of bin pupil vs bin body speed (non-mixed bins); r_head = Pearson
# r of bin pupil vs bin head angular speed in stationary bins (head movement without locomotion).
rows = []
for gi, (group, title) in enumerate(zip(groups, titles)):
    for R in group:
        b = B[id(R)].dropna(subset=["pupil"])
        ok = len(b) >= min_bins
        row = dict(id=R.id, eo=R.eo, group=title, gi=gi, valid_pct=100 * np.isfinite(PUP[id(R)]).mean(), n_bins=len(b))
        row["pupil_mm"] = REF[id(R)] if ok else np.nan
        st = b.pupil[b.state == "stationary"]
        row["pupil_mm_stat"] = REF[id(R)] * st.median() / 100 if len(st) >= min_state_bins else np.nan
        row["pupil_cv"] = 100 * b.pupil.std() / b.pupil.mean() if ok else np.nan
        m = {s: b.pupil[b.state == s].mean() if (b.state == s).sum() >= min_state_bins else np.nan for s in STATES}
        row.update({f"pupil_{s[:4]}": m[s] for s in STATES})
        row["d_run"] = 100 * (m["running"] - m["stationary"]) / m["stationary"]
        row["d_turn"] = 100 * (m["turning"] - m["stationary"]) / m["stationary"]
        row["d_turn_run"] = 100 * (m["turning"] - m["running"]) / m["stationary"]
        k = (b.state != "mixed") & b.speed.notna()
        row["r_speed"] = pearsonr(b.speed[k], b.pupil[k])[0] if k.sum() >= min_bins else np.nan
        k = (b.state == "stationary") & b.head_speed.notna()
        row["r_head"] = pearsonr(b.head_speed[k], b.pupil[k])[0] if k.sum() >= min_state_bins else np.nan
        for kind in ("run", "turn"):
            row[f"n_{kind}_on"] = TRIG[id(R)][kind]["n"]
            row[f"{kind}_on_dp"] = TRIG[id(R)][kind]["dp"] if TRIG[id(R)][kind]["n"] >= min_onsets else np.nan
        rows.append(row)
S = pd.DataFrame(rows)
families = dict(data=["valid_pct", "n_bins"],
                size=["pupil_mm", "pupil_mm_stat", "pupil_cv"],
                state=["d_run", "d_turn", "d_turn_run"],
                coupling=["r_speed", "r_head"],
                dynamics=["run_on_dp", "turn_on_dp"])
print(S[["id", "eo", "pupil_stat", "pupil_runn", "pupil_turn", "n_run_on", "n_turn_on"]
        + [c for cols in families.values() for c in cols]].round(2).to_string(index=False))

# METHODS: EO trend per ferret: Pearson r of session value vs EO, sessions treated as independent
# (ferrets with >= 3 sessions). Reported as the number of ferrets whose r has the sign of the global
# trend (r over all sessions pooled), and how many of those reach p < 0.05. Normal cohort only; in
# compare_delayed mode the only comparison is delayed vs oldest normal sessions (group model below).
# METHODS: global trend across EO groups: linear mixed model (LMM), measure ~ group index (EO bin
# 0, 1, 2; or oldest normal = 0, delayed = 1) as a linear fixed effect, random intercept per ferret.
# β = change per group step. p = likelihood-ratio test against the intercept-only model (both fit
# by ML), chi2 with 1 df, Holm-corrected within each family of measures.
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
            lo = 0 if fam in ("data", "size") else d[col].min()   # y limits on whole tick steps
            step = np.diff(MaxNLocator(5, steps=[1, 2, 5, 10]).tick_values(lo, d[col].max())[:2])[0]
            lo, hi = np.floor(lo / step) * step, np.ceil(d[col].max() / step) * step
            ax.set_yticks(np.linspace(lo, hi, round((hi - lo) / step) + 1))
            ax.set_ylim(lo, hi)
            if fam in ("state", "coupling", "dynamics"):
                ax.axhline(0, color="0.8", lw=0.5)
            up = pearsonr(d.eo, d[col])[0] > 0
            fr = {fid: pearsonr(g.eo, g[col]) for fid, g in d.groupby("id") if len(g) >= 3}
            same = [fid for fid, (r, p) in fr.items() if (r > 0) == up]
            ax.set_title(f"{col}\n{len(same)}/{len(fr)} ferrets {'↑' if up else '↓'} "
                         f"({sum(fr[fid][1] < 0.05 for fid in same)} with p<0.05)", fontsize=6)
            print(f"{col:15s} " + "  ".join(f"F{fid} r={r:+.2f} p={p:.2g}" for fid, (r, p) in fr.items()))
        axes[0, 0].legend(fontsize=4)
        sns.despine(fig)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.8, wspace=0.5)

        if SAVE_FIGS:
            save_fig(fig, f"per-session pupil {fam} vs EO" + fig_tag)

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
        print(f"{col:15s} {desc}  (mean ± SD over sessions)  {group_stat[col]}")
    if compare_delayed:   # per-ferret means (delayed ferrets have one session each)
        print(S.groupby(["group", "id"])[cols].mean().join(S.groupby(["group", "id"]).size().rename("n")).round(3).to_string())


# %% pupil by locomotor state (illustration; stats are per session above)
# Per session, mean bin pupil per state (top: % of session median; bottom: mm = x pupil_mm / 100); grey lines
# join one session's states, black = mean +- SEM across sessions.
fig, axes = plt.subplots(2, len(groups), figsize=(2.2 * len(groups), 4.4), squeeze=False, sharey="row")
for j, title in enumerate(titles):
    d = S[S.group == title]
    for i, scale in enumerate((np.ones(len(d)), d.pupil_mm.to_numpy() / 100)):
        ax = axes[i, j]
        V = d[[f"pupil_{s[:4]}" for s in STATES]].to_numpy() * scale[:, None]
        for v in V:
            ax.plot(range(len(STATES)), v, color="0.85", lw=0.5, zorder=0)
        for si, s in enumerate(STATES):
            ax.scatter(np.full(len(V), si), V[:, si], s=10, facecolor="w", edgecolor=STATE_COLORS[s], lw=0.8)
        n = np.isfinite(V).sum(0)
        ax.errorbar(np.arange(len(STATES)) + 0.2, np.nanmean(V, 0), np.nanstd(V, 0) / np.sqrt(np.maximum(n, 1)),
                    marker="o", ms=3, lw=1, capsize=0, color="k")
        ax.set_xticks(range(len(STATES)), STATES, fontsize=6)
        ax.set_xlim(-0.5, len(STATES) - 0.3)
    axes[0, j].set_title(f"{title}\nd_run: {group_stat['d_run']}\nd_turn: {group_stat['d_turn']}", fontsize=5)
axes[0, 0].set_ylabel("pupil (% of session median)")
axes[1, 0].set_ylabel("pupil (mm)")
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "pupil by state" + fig_tag)


# %% pupil vs body speed and vs head angular speed
# METHODS: per session, mean bin pupil (% of session median) in bins of body speed (all non-mixed bins) and of
# head angular speed (stationary bins only); NaN with < min_bin valid bins. Lines = mean +- SEM across
# sessions, bins with >= 2 sessions.
fig, axes = plt.subplots(1, 2, figsize=(6, 2.2))
for ax, edges, col, sel, xl in ((axes[0], speed_edges, "speed", ("stationary", "running", "turning"), "body speed (mm/s)"),
                                (axes[1], head_edges, "head_speed", ("stationary",), "head angular speed (deg/s), stationary")):
    centers = (edges[:-1] + edges[1:]) / 2
    for group, title, c in zip(groups, titles, GROUP_COLORS):
        G = np.full((len(group), len(centers)), np.nan)
        for si, R in enumerate(group):
            b = B[id(R)]
            b = b[b.state.isin(sel) & b.pupil.notna()]
            for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                k = (b[col] >= lo) & (b[col] < hi)
                if k.sum() >= min_bin:
                    G[si, bi] = b.pupil[k].mean()
        n = np.isfinite(G).sum(0)
        ok = n >= 2
        mu = np.array([np.nanmean(g) if o else np.nan for g, o in zip(G.T, ok)])
        se = np.array([np.nanstd(g) / np.sqrt(k) if o else np.nan for g, o, k in zip(G.T, ok, n)])
        ax.errorbar(centers[ok], mu[ok], se[ok], marker="o", ms=3, lw=1, capsize=0, color=c, label=title)
    ax.axhline(100, color="0.8", lw=0.5)
    ax.set_xlabel(xl)
    ax.set_ylabel("pupil (% of session median)")
axes[0].set_title(f"r_speed: {group_stat['r_speed']}", fontsize=5)
axes[1].set_title(f"r_head: {group_stat['r_head']}", fontsize=5)
axes[0].legend(fontsize=5)
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "pupil vs speed" + fig_tag)


# %% onset-triggered pupil (run onsets, turn onsets)
# Session traces from the onset cell (TRIG); mean +- SEM across sessions with a full trace. Bottom: body speed.
t = tb + 0.5   # bin centres (s)
fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
for j, (kind, name) in enumerate((("run", "run onset"), ("turn", "turn onset"))):
    for group, title, c in zip(groups, titles, GROUP_COLORS):
        use = [R for R in group if TRIG[id(R)][kind]["n"] >= min_onsets and np.isfinite(TRIG[id(R)][kind]["pupil"]).all()]
        plot_mean_se(axes[0, j], t, [TRIG[id(R)][kind]["pupil"] for R in use], c, title, min_n=2)
        plot_mean_se(axes[1, j], t, [TRIG[id(R)][kind]["speed"] for R in use], c, title, min_n=2)
    axes[0, j].axhline(0, color="0.8", lw=0.5)
    axes[0, j].set_title(f"{name}\n{kind}_on_dp: {group_stat[f'{kind}_on_dp']}", fontsize=5)
    axes[1, j].set_xlabel(f"time from {name} (s)")
    for ax in axes[:, j]:
        ax.axvline(0, color="0.8", lw=0.5)
axes[0, 0].set_ylabel("pupil change (% of session median)")
axes[1, 0].set_ylabel("body speed (mm/s)")
axes[0, 0].legend(fontsize=4)
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "onset-triggered pupil" + fig_tag)


# %% example session: gate, gated pupil, bins, speed and state (sanity check)
R = next(R for R in Results if (R.id, R.eo) == example)
M = MASK[id(R)]
i0 = len(R.LE_x) // 2
sl = slice(i0, i0 + int(120 * FS))
tt = np.arange(len(R.LE_x))[sl] / FS
fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
for eye in ("LE", "RE"):
    axes[0].plot(tt, eye_signal(R, eye, "x")[sl], lw=0.5, color=EYE_COLORS[eye], label=f"{eye} x")
    axes[0].plot(tt, eye_signal(R, eye, "y")[sl], lw=0.5, color=EYE_COLORS[eye], ls="--", label=f"{eye} y")
    axes[1].plot(tt, np.asarray(getattr(R, f"{eye}_pupil"), float)[sl], lw=0.5, color="0.8")
    axes[1].plot(tt, M["gated"][eye][sl], lw=0.7, color=EYE_COLORS[eye], label=f"{eye} gated")
axes[0].axhspan(-eye_max, eye_max, color="g", alpha=0.1)
axes[0].set_ylim(-30, 30)
axes[0].set_ylabel("eye position (deg)")
axes[0].legend(fontsize=4, ncol=4)
axes[1].set_ylabel("pupil (mm)")
axes[1].set_ylim(0.7 * REF[id(R)], 1.3 * REF[id(R)])   # raw spikes (lid / fit failures) off scale
axes[1].legend(fontsize=4)
b = B[id(R)]
b = b[(b.t >= tt[0] - 1) & (b.t <= tt[-1])]
axes[2].plot(tt, PUP[id(R)][sl], lw=0.5, color="0.6")
axes[2].step(b.t, b.pupil, where="post", color="k", lw=1)
axes[2].set_ylabel("pupil (% median)")
axes[3].plot(tt, np.asarray(R.speed, float)[sl], lw=0.5, color="k")
axes[3].set_ylabel("body speed (mm/s)")
axes[3].set_xlabel("time (s)")
for ax in axes:
    for s in STATES:
        ax.fill_between(tt, 0, 1, where=M[s][sl], transform=ax.get_xaxis_transform(), color=STATE_COLORS[s],
                        alpha=0.12, lw=0, label=s if ax is axes[3] else None)
axes[3].legend(fontsize=4)
axes[0].set_title(f"F{R.id} EO{R.eo}  (eye_max = {eye_max} deg)", fontsize=6)
sns.despine(fig)
fig.tight_layout()
if SAVE_FIGS:
    save_fig(fig, "pupil example session" + fig_tag)
