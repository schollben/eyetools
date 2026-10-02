# %% init
# Models are fit with ssm (Linderman lab, Stanford; github.com/lindermanlab/ssm): EM (Baum-Welch) for fitting,
# Viterbi for decoding (most_likely_states). Settings come from HMM/config/hmm.yaml.
%load_ext autoreload
%autoreload 2
import sys
sys.path.insert(0, "")  # ensure cwd is on path so local_config.py is found
import local_config  # type: ignore
sys.path.insert(0, local_config.EYETOOLS_ROOT + "/HMM")
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import ks_2samp
from sklearn.metrics import adjusted_rand_score
from utils import create_subplot_grid
from hmm.adapter import load_session
from hmm.sessions import load_config, eligible_sessions
from hmm.features import session_features, fit_scaler, apply_scaler, mirror, FEATURE_NAMES, SIGNED
from hmm.splits import leave_one_session_out, leave_one_animal_out, select
from hmm.fit import fit_model, ll_per_frame, save_run, RESULTS_DIR
from hmm.select import fold_data, grid_scores, best_restarts, plateau
from hmm.evaluate import (run_lengths, dwell_times, transition_freqs, match_states, interior_accuracy,
                          mirror_pairs, simulate, spectrum, changepoint_sweep, boundary_agreement,
                          to_session_frames, event_table)
from hmm.synth import synth_segments, synth_session

cfg = load_config()
ids = eligible_sessions(cfg)
print(len(ids), "eligible sessions")
RESULTS_DIR.mkdir(exist_ok=True)   # cached grids / fits live here; delete a file to refit it


def plot_states(t, X, z, title="", seconds=60, fs=30):
    """Features (rows) under the decoded state sequence (colour band), first `seconds` of a segment."""
    n = min(len(t), int(seconds * fs))
    fig, axes = plt.subplots(len(FEATURE_NAMES) + 1, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw=dict(height_ratios=[0.4] + [1] * len(FEATURE_NAMES)))
    axes[0].imshow(z[None, :n], aspect="auto", cmap="tab20", vmin=0, vmax=19,
                   interpolation="nearest", extent=[t[0], t[n - 1], 0, 1])
    axes[0].set_yticks([])
    axes[0].set_title(title)
    for j, name in enumerate(FEATURE_NAMES):
        axes[j + 1].plot(t[:n], X[:n, j], lw=0.6, color="k")
        axes[j + 1].set_ylabel(name)
    axes[-1].set_xlabel("time (s)")
    plt.show()


# %% load
# Skull kinematics via utils.load_skull_data (bs repo load_kinematics). Frames are marked invalid for
# speed > 800 mm/s, |pitch| > 75 deg (Euler gimbal lock) or |omega| > 1500 deg/s (tracking glitches), padded by 6 frames.
q = cfg["qc"]
S = [load_session(s, max_speed=q["max_speed_mm_s"], max_abs_pitch=q["max_abs_pitch_deg"],
                  max_ang_speed=q["max_ang_speed_deg_s"], pad=q["pad_frames"]) for s in ids]


# %% QC: one row per session + sanity asserts
# One row per session: sampling rate, duration, fraction valid, largest yaw step. The asserts catch broken timestamps / yaw range.
rows = []
for s in S:
    assert np.all(np.diff(s.t) > 0), s.session_id
    assert np.all((s.yaw_deg >= 0) & (s.yaw_deg < 360)), s.session_id
    assert len(s.yaw_deg) == len(s.t) == len(s.pos_xy) == len(s.valid), s.session_id

    dyaw = (np.diff(s.yaw_deg) + 180) % 360 - 180   # wrapped per-frame change
    dyaw = dyaw[s.valid[1:] & s.valid[:-1]]           # valid-to-valid frames only
    rows.append(dict(
        session_id=s.session_id, animal_id=s.animal_id, eo=s.eo,
        fs=s.fs, duration_s=s.t[-1] - s.t[0], frac_valid=s.valid.mean(),
        max_abs_dyaw_valid=np.abs(dyaw).max(),
        x_range=np.ptp(s.pos_xy[:, 0]), y_range=np.ptp(s.pos_xy[:, 1]),
    ))

qc = pd.DataFrame(rows)
print(qc.drop(columns="session_id").round(2).to_string())
print("minutes per animal:\n", (qc.groupby("animal_id").duration_s.sum() / 60).round(1))


# %% inspect one session: yaw, speed, valid
# Raw signals for one session, to check by eye that the valid mask removes the yaw jumps and speed spikes.
n = 0
s = S[n]
speed = np.linalg.norm(s.vel_global, axis=1)

fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
axes[0].plot(s.t, s.yaw_deg, lw=0.5)
axes[0].set_ylabel("yaw (deg)")
axes[1].plot(s.t, speed, lw=0.5)
axes[1].set_ylabel("speed (mm/s)")
axes[2].plot(s.t, s.valid, lw=0.5)
axes[2].set_ylabel("valid")
axes[2].set_xlabel("time (s)")
axes[0].set_title(f"{s.animal_id}  EO{s.eo}")
plt.show()


# %% FEATURES: valid runs -> omega_yaw, log_speed, pitch, v_fwd, v_lat at a common rate
# Per valid run (>= 2 s), from the loaded (upstream-smoothed) data with no extra filter: yaw velocity (np.gradient), log(speed + 1),
# pitch, and world velocity rotated into the head frame by yaw (forward / lateral, signed log); interpolated to 30 Hz.
f = cfg["features"]
segs = [g for s in S for g in session_features(s, **f)]

# sign check: during fast movement the head should mostly move forward (v_fwd > 0); if not, yaw and x/y conventions disagree
Xc = np.concatenate([g["X"] for g in segs])
fast = Xc[:, 1] > np.log(200 + 1)
print(f"fraction v_fwd > 0 when speed > 200 mm/s: {(Xc[fast, 3] > 0).mean():.2f}")

seg_df = pd.DataFrame([dict(session_id=g["session_id"], animal_id=g["animal_id"], eo=g["eo"], n=len(g["t"])) for g in segs])
summary = seg_df.groupby(["animal_id", "eo"]).agg(n_segs=("n", "size"), minutes=("n", "sum"))
summary["minutes"] = (summary["minutes"] / f["fs_out"] / 60).round(1)
print(summary)
print("total minutes:", round(seg_df.n.sum() / f["fs_out"] / 60, 1))


# %% FEATURES: inspect one segment
# The longest segment's features in raw units, to check that turns, runs and rears look plausible.
g = max(segs, key=lambda g: len(g["t"]))   # longest segment
fig, axes = plt.subplots(len(FEATURE_NAMES), 1, figsize=(10, 7), sharex=True)
for j, name in enumerate(FEATURE_NAMES):
    axes[j].plot(g["t"], g["X"][:, j], lw=0.5)
    axes[j].set_ylabel(name)
axes[-1].set_xlabel("time (s)")
axes[0].set_title(f"{g['animal_id']}  EO{g['eo']}")
plt.show()


# %% FEATURES: distributions per animal (raw units: rad/s, log(mm/s+1), rad)
# Feature histograms per animal; large offsets between animals would bias leave-one-animal-out scores.
fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(14, 2.5))
for a in sorted({g["animal_id"] for g in segs}):
    Xa = np.concatenate([g["X"] for g in segs if g["animal_id"] == a])
    for j, name in enumerate(FEATURE_NAMES):
        axes[j].hist(Xa[:, j], bins=100, histtype="step", density=True, label=str(a))
        axes[j].set_title(name)
axes[0].legend()
plt.show()


# %% SPLITS + SCALING: leakage guard on one fold, mirror check
# Cross-validation folds: leave-one-session-out (LOSO) and leave-one-animal-out (LOAO). Features are robust-z-scored
# (median / IQR) using training sessions only; mirroring (omega, v_lat -> -omega, -v_lat) doubles the training data and makes left/right symmetric.
folds_loso = leave_one_session_out(segs)
folds_loao = leave_one_animal_out(segs)
print(len(folds_loso), "LOSO folds,", len(folds_loao), "LOAO folds")

# data halves for reproducibility: alternate sessions (sorted by EO) within each animal
halves = [[], []]
for a in sorted({g["animal_id"] for g in segs}):
    sess = sorted({(g["eo"], g["session_id"]) for g in segs if g["animal_id"] == a})
    for i, (_, sid) in enumerate(sess):
        halves[i % 2].append(sid)

train_ids, test_ids = folds_loso[0]
assert not set(train_ids) & set(test_ids)
train = select(segs, train_ids)
scaler = fit_scaler([g["X"] for g in train])                       # training sessions only
X_train = apply_scaler([g["X"] for g in train], scaler)
X_test = apply_scaler([g["X"] for g in select(segs, test_ids)], scaler)

# scaling stats unchanged if the held-out session is dropped from the pool -> no leakage
assert np.allclose(fit_scaler([g["X"] for g in segs if g["session_id"] in train_ids])["median"], scaler["median"])

# mirror: omega flips, others unchanged; augmented training set = original + mirrored
Xm = mirror(X_train[0])
unsigned = [j for j in range(len(FEATURE_NAMES)) if j not in SIGNED]
assert np.allclose(Xm[:, SIGNED], -X_train[0][:, SIGNED]) and np.allclose(Xm[:, unsigned], X_train[0][:, unsigned])
X_train_aug = X_train + [mirror(X) for X in X_train]

# per-session drift of scaled features (median per session)
drift = pd.DataFrame([dict(session=f"{g['animal_id']} EO{g['eo']}", **dict(zip(FEATURE_NAMES, np.median(X, axis=0).round(2))))
                      for g, X in zip(train, X_train)]).groupby("session").median()
print(drift)


# =============================================================================
# §7 SYNTHETIC CHECKS — test the code, not the science
# =============================================================================

# %% SYNTH 1: round trip — synthetic Session -> features -> fit -> states on the original 120 Hz frames
# Synthetic 3-state session (still / turn left / turn right) at 120 Hz; checks that the feature pipeline and the
# mapping of 30 Hz states back onto the original frames line up (accuracy away from boundaries).
s_syn, z_syn = synth_session()
segs_syn = session_features(s_syn, **f)
X_syn = apply_scaler([g["X"] for g in segs_syn], fit_scaler([g["X"] for g in segs_syn]))
m_syn, _ = fit_model("gaussian", X_syn, 3, num_iters=50)
z_full = to_session_frames(s_syn, segs_syn, [m_syn.most_likely_states(X) for X in X_syn], f["fs_out"])
mapped = z_full >= 0
acc = interior_accuracy(z_syn[mapped], z_full[mapped], 3, margin=12)   # ±100 ms at 120 Hz
print(f"round trip: {mapped.mean():.4f} of frames mapped, interior accuracy {acc:.3f}")
assert len(z_full) == len(s_syn.t) and acc >= 0.85


# %% SYNTH 2: AR-HMM (K=6, L=2) recovers 6 synthetic movement types; Gaussian HMM for comparison
# AR-HMM: each state is a linear autoregressive dynamical system x_t = sum_l A_l x_{t-l} + b + noise, the model behind
# MoSeq (Wiltschko et al. 2015, Datta lab). It can tell apart states with equal means but different dynamics (3 Hz oscillation vs still).
zs_syn, Xs_syn = synth_segments(n_segments=10, seed=1)   # 7 train, 3 held-out segments
m_ar, _ = fit_model("ar", Xs_syn[:7], 6, L=2)
m_g, _ = fit_model("gaussian", Xs_syn[:7], 6)
acc_ar = np.mean([interior_accuracy(z, m_ar.most_likely_states(X), 6) for z, X in zip(zs_syn[7:], Xs_syn[7:])])
acc_g = np.mean([interior_accuracy(z, m_g.most_likely_states(X), 6) for z, X in zip(zs_syn[7:], Xs_syn[7:])])
print(f"held-out interior accuracy: AR-HMM {acc_ar:.3f}, Gaussian HMM {acc_g:.3f} (can't separate still vs oscillation)")
assert acc_ar >= 0.85

z_hat = m_ar.most_likely_states(Xs_syn[7])
z_hat = match_states(zs_syn[7], z_hat, 6)[z_hat]
fig, ax = plt.subplots(figsize=(10, 1.5))
ax.imshow(np.vstack([zs_syn[7][:600], z_hat[:600]]), aspect="auto", cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
ax.set_yticks([0, 1], ["true", "AR-HMM"])
ax.set_xlabel("frame (30 Hz)")
plt.show()


# %% SYNTH 3: mirror augmentation -> turn left / turn right come out as a sign-flipped pair
# With mirrored training data, a left-turn state should have a right-turn partner whose AR parameters are the sign-flipped
# copy (A -> S A S, b -> S b, S = diag(-1, 1, 1)); mirror_pairs finds these by parameter distance.
m_mir, _ = fit_model("ar", Xs_syn[:7] + [mirror(X) for X in Xs_syn[:7]], 6, L=2)
to_true = match_states(np.concatenate(zs_syn[:7]), np.concatenate([m_mir.most_likely_states(X) for X in Xs_syn[:7]]), 6)
pair, dist = mirror_pairs(m_mir, SIGNED)
partner = {int(to_true[k]): int(to_true[pair[k]]) for k in range(6)}   # in true labels
print("mirror partner (true labels):", dict(sorted(partner.items())), " param distance:", dist.round(3))
assert partner[1] == 2 and partner[2] == 1 and all(partner[k] == k for k in (0, 3, 4, 5))


# %% SYNTH 4: gamma dwell times -> HSMM beats the sticky AR-HMM on held-out LL (geometric = control)
# HMM state durations are geometric; an HSMM (hidden semi-Markov model, cf. Johnson & Willsky 2013) models them explicitly.
# ssm's HSMM uses negative-binomial durations; it is warm-started from the sticky AR-HMM's emissions. It should only win when durations are peaked.
gain = {}
for dwell in ["gamma", "geometric"]:
    zs_d, Xs_d = synth_segments(n_segments=8, dwell=dwell, gamma_shape=10, seed=2)
    m_s, _ = fit_model("ar", Xs_d[:6], 6, L=2)
    m_h, _ = fit_model("hsmm", Xs_d[:6], 6, L=2, num_iters=50, init_from=m_s)
    gain[dwell] = ll_per_frame(m_h, Xs_d[6:]) - ll_per_frame(m_s, Xs_d[6:])
    print(f"{dwell:9s} dwell: HSMM - sticky held-out LL = {gain[dwell]:+.4f} nats/frame")
assert gain["gamma"] > 0.005 and abs(gain["geometric"]) < 0.005


# =============================================================================
# §5 TIMESCALE REFERENCE AND KAPPA
# =============================================================================

# %% TIMESCALE: model-free changepoints (PELT, l2) on scaled features, penalty sweep
# PELT (Killick, Fearnhead & Eckley 2012) via the ruptures package (Truong et al. 2020) finds mean shifts in the features with no model.
# Sweeping the penalty and taking the plateau (or elbow) gives a model-free reference duration for movement segments.
ts = cfg["timescale"]
sc_all = fit_scaler([g["X"] for g in segs])   # descriptive only; held-out scores refit scaling per fold
X_all = apply_scaler([g["X"] for g in segs], sc_all)
path = RESULTS_DIR / "changepoints.pkl"
if path.exists():
    cps = pickle.load(open(path, "rb"))
else:
    cps = changepoint_sweep(X_all, ts["penalties"], n_jobs=cfg["selection"]["n_jobs"])
    pickle.dump(cps, open(path, "wb"))
minutes = sum(len(X) for X in X_all) / f["fs_out"] / 60
rate = pd.Series({pen: sum(len(c) for c in cps[pen]) / minutes for pen in ts["penalties"]})

# plateau = flat interior stretch of log(rate) vs log(penalty) (|slope| < 0.2);
# if there is none, use the elbow (largest change in slope)
slope = np.gradient(np.log(rate.values), np.log(rate.index.values))
curv = np.gradient(slope, np.log(rate.index.values))
if np.any(np.abs(slope[1:-1]) < 0.2):
    pen_star, how = rate.index[1 + np.argmin(np.abs(slope[1:-1]))], "plateau"
else:
    pen_star, how = rate.index[1 + np.argmax(np.abs(curv[1:-1]))], "elbow (no plateau)"
cp_dur = np.concatenate([np.diff(np.concatenate([[0], c, [len(X)]])) for c, X in zip(cps[pen_star], X_all)]) / f["fs_out"]
print(pd.DataFrame(dict(changepoints_per_min=rate.round(1), slope=slope.round(2), curvature=curv.round(2))))
print(f"{how}: penalty {pen_star}, median changepoint interval {np.median(cp_dur):.2f} s")

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].loglog(rate.index, rate.values, "o-")
axes[0].axvline(pen_star, color="r", lw=0.8)
axes[0].set_xlabel("PELT penalty")
axes[0].set_ylabel("changepoints / min")
axes[1].hist(cp_dur, bins=np.logspace(-1, 1.5, 40))
axes[1].set_xscale("log")
axes[1].set_xlabel("interval between changepoints (s)")
plt.tight_layout()
plt.show()


# %% KAPPA: sweep kappa (AR-HMM, all sessions, mirrored); pick median state duration ~ changepoint median
# kappa is the 'sticky' bias (Fox, Sudderth, Jordan & Willsky 2011): extra prior counts on self-transitions, so larger kappa
# means longer states. Here kappa is chosen so the median decoded state duration matches the changepoint median.
sel = cfg["selection"]
X_all_aug = X_all + [mirror(X) for X in X_all]
path = RESULTS_DIR / f"kappa_sweep_K{sel['kappa_K']}_L{sel['kappa_L']}.csv"
if path.exists():
    kappa_df = pd.read_csv(path)
else:
    def _median_dwell(kappa):
        m, _ = fit_model("ar", X_all_aug, sel["kappa_K"], L=sel["kappa_L"], kappa=kappa,
                         seed=cfg["seed"], num_iters=sel["num_iters"])
        d = np.concatenate(dwell_times([m.most_likely_states(X) for X in X_all], sel["kappa_K"]))
        return np.median(d) / f["fs_out"]
    meds = Parallel(n_jobs=sel["n_jobs"], verbose=10, pre_dispatch="all")(delayed(_median_dwell)(k) for k in sel["kappas"])
    kappa_df = pd.DataFrame(dict(kappa=sel["kappas"], median_state_dur_s=meds))
    kappa_df.to_csv(path, index=False)

kappa = float(kappa_df.kappa[(np.log(kappa_df.median_state_dur_s) - np.log(np.median(cp_dur))).abs().idxmin()])
print(kappa_df.round(3))
print(f"changepoint median {np.median(cp_dur):.2f} s -> kappa = {kappa:g}")


# =============================================================================
# M1: STICKY GAUSSIAN HMM BASELINE (one animal)
# =============================================================================

# %% M1: held-out LL per frame vs K (leave-one-session-out within one animal)  [cached]
# Baseline: sticky Gaussian HMM (each state = one Gaussian over the 3 features, no dynamics) on one animal.
# Held-out log-likelihood per frame vs number of states K; the 'plateau' is the smallest K within one paired SE of the best (one-SE rule).
segs_m1 = [g for g in segs if g["animal_id"] == sel["m1_animal"]]
folds_m1 = leave_one_session_out(segs_m1)
path = RESULTS_DIR / f"m1_gaussian_{sel['m1_animal']}_kappa{sel['m1_kappa']:g}.csv"
if path.exists():
    m1 = pd.read_csv(path)
else:
    m1 = grid_scores(segs_m1, folds_m1, "gaussian", sel["m1_Ks"], kappas=[sel["m1_kappa"]],
                     num_iters=sel["num_iters"], n_jobs=sel["n_jobs"])
    m1.to_csv(path, index=False)

best_m1 = best_restarts(m1)
K_m1 = plateau(best_m1)
summ = best_m1.groupby("K").test_ll.agg(["mean", "sem"])
fig, ax = plt.subplots(figsize=(4, 3))
ax.errorbar(summ.index, summ["mean"], summ["sem"], marker="o")
ax.axvline(K_m1, color="r", lw=0.8)
ax.set_xlabel("K")
ax.set_ylabel("held-out LL / frame")
ax.set_title(f"M1 Gaussian HMM, ferret {sel['m1_animal']} (plateau K = {K_m1})")
plt.show()


# %% M1: decoded states over the kinematics, held-out session
# Viterbi state sequence of the M1 model on a held-out session, drawn over the features, to see what the states pick up.
train_ids, test_ids = folds_m1[0]
Xtr, Xte, _ = fold_data(segs_m1, train_ids, test_ids)
m, _ = fit_model("gaussian", Xtr, K_m1, kappa=sel["m1_kappa"], seed=cfg["seed"], num_iters=sel["num_iters"])
g_test = select(segs_m1, test_ids)
i = int(np.argmax([len(g["t"]) for g in g_test]))
plot_states(g_test[i]["t"], g_test[i]["X"], m.most_likely_states(Xte[i]), title=f"M1 Gaussian HMM K={K_m1}, held-out")


# =============================================================================
# M2: STICKY AR-HMM (primary model)
# =============================================================================

# %% M2: K x L grid, leave-one-session-out over all sessions  [cached; ~30 min on 8 cores]
# Primary model: sticky AR-HMM (MoSeq-style) over all sessions. Grid of K (states) x L (AR lags = how many past frames
# each state's dynamics use), scored by held-out LL per frame with LOSO; best of several restarts per setting.
path = RESULTS_DIR / f"m2_ar_kappa{kappa:g}.csv"
if path.exists():
    m2 = pd.read_csv(path)
else:
    m2 = grid_scores(segs, folds_loso, "ar", sel["m2_Ks"], sel["m2_Ls"], kappas=[kappa],
                     num_iters=sel["num_iters"], n_jobs=sel["n_jobs"])
    m2.to_csv(path, index=False)


# %% M2: choose K at the plateau for each L, then the smallest L within one SE of the best
# Same one-SE rule as M1: smallest K per L, then smallest L, whose held-out LL is statistically tied with the best.
best_m2 = best_restarts(m2)
K_by_L = {L: plateau(best_m2[best_m2.L == L]) for L in sel["m2_Ls"]}
at_plateau = pd.concat([best_m2[(best_m2.L == L) & (best_m2.K == K)] for L, K in K_by_L.items()])
L_sel = int(plateau(at_plateau, col="L"))
K_sel = int(K_by_L[L_sel])
print("plateau K per L:", K_by_L, f"-> selected L = {L_sel}, K = {K_sel}")

fig, ax = plt.subplots(figsize=(4, 3))
for L in sel["m2_Ls"]:
    summ = best_m2[best_m2.L == L].groupby("K").test_ll.agg(["mean", "sem"])
    ax.errorbar(summ.index, summ["mean"], summ["sem"], marker="o", label=f"L={L}")
ax.set_xlabel("K")
ax.set_ylabel("held-out LL / frame")
ax.set_title(f"M2 sticky AR-HMM, kappa={kappa:g}")
ax.legend()
plt.show()


# %% M2: confirm with leave-one-animal-out  [cached]
# Generalisation to a new animal: train on one ferret, score the other. Only 2 folds, so this is a sanity check, not a selector.
path = RESULTS_DIR / f"m2_ar_loao_L{L_sel}_kappa{kappa:g}.csv"
if path.exists():
    m2_loao = pd.read_csv(path)
else:
    m2_loao = grid_scores(segs, folds_loao, "ar", sel["m2_Ks"], [L_sel], kappas=[kappa],
                          num_iters=sel["num_iters"], n_jobs=sel["n_jobs"])
    m2_loao.to_csv(path, index=False)

held_out_animal = [sorted({g["animal_id"] for g in select(segs, te)})[0] for _, te in folds_loao]
fig, ax = plt.subplots(figsize=(4, 3))
for fold, df in m2_loao.groupby("fold"):
    ax.plot(df.K, df.test_ll, "o-", label=f"held-out ferret {held_out_animal[fold]}")
ax.axvline(K_sel, color="r", lw=0.8)
ax.set_xlabel("K")
ax.set_ylabel("held-out LL / frame")
ax.legend()
plt.show()


# =============================================================================
# DECIDE K AND KAPPA — held-out likelihood + reproducibility (plan: model selection uses only these)
# =============================================================================

# %% DECIDE K: held-out LL, restart / half-split agreement, reproducible states vs K (L=3)  [cached; ~25 min]
# Held-out LL alone keeps rising with K, so also ask whether states are reproducible: ARI (Hubert & Arabie 1985) between restarts
# and between models fit on two halves of the sessions; per-state Jaccard after Hungarian matching (scipy linear_sum_assignment).
L_dec = 3
kappas_dec = [100.0, kappa]                       # moderate vs changepoint-matched kappa
seeds = list(range(cfg["seed"], cfg["seed"] + sel["n_restarts"]))

# held-out LL (LOSO) at each kappa; the M2 grid already has the changepoint-matched kappa
path = RESULTS_DIR / f"m2_ar_L{L_dec}_kappa100.csv"
if path.exists():
    ll_100 = pd.read_csv(path)
else:
    ll_100 = grid_scores(segs, folds_loso, "ar", sel["m2_Ks"], [L_dec], kappas=[100.0],
                         num_iters=sel["num_iters"], n_jobs=sel["n_jobs"])
    ll_100.to_csv(path, index=False)
ll_dec = pd.concat([ll_100, m2[m2.L == L_dec]])

# reproducibility: n_restarts fits on all sessions + one fit per data half, for every K x kappa
path = RESULTS_DIR / f"decide_K_L{L_dec}.csv"
if path.exists():
    dec_K = pd.read_csv(path)
else:
    Xh = [[X for g, X in zip(segs, X_all) if g["session_id"] in h] for h in halves]
    jobs = ([(K, kap, "all", sd) for K in sel["m2_Ks"] for kap in kappas_dec for sd in seeds] +
            [(K, kap, h, cfg["seed"]) for K in sel["m2_Ks"] for kap in kappas_dec for h in (0, 1)])

    def _fit(K, kap, which, sd):
        Xs = X_all if which == "all" else Xh[which]
        m, _ = fit_model("ar", Xs + [mirror(X) for X in Xs], K, L=L_dec, kappa=kap, seed=sd, num_iters=sel["num_iters"])
        return m
    fitted = Parallel(n_jobs=sel["n_jobs"], verbose=10, pre_dispatch="all")(delayed(_fit)(*j) for j in jobs)

    rows = []
    for K in sel["m2_Ks"]:
        for kap in kappas_dec:
            ms = [m for j, m in zip(jobs, fitted) if j[:3] == (K, kap, "all")]
            mh = [m for j, m in zip(jobs, fitted) if j[:2] == (K, kap) and j[2] != "all"]
            zr = [np.concatenate([m.most_likely_states(X) for X in X_all]) for m in ms]
            ref = int(np.argmax([ll_per_frame(m, X_all) for m in ms]))
            jacc = np.zeros((K, len(ms)))
            for r, z in enumerate(zr):
                z = match_states(zr[ref], z, K)[z]
                for k in range(K):
                    jacc[k, r] = np.sum((zr[ref] == k) & (z == k)) / max(np.sum((zr[ref] == k) | (z == k)), 1)
            zh = [np.concatenate([m.most_likely_states(X) for X in X_all]) for m in mh]
            rows.append(dict(K=K, kappa=kap,
                             ari_restarts=np.mean([adjusted_rand_score(zr[a], zr[b]) for a in range(len(zr)) for b in range(a)]),
                             ari_halves=adjusted_rand_score(*zh),
                             frac_states_reproducible=np.mean(np.delete(jacc, ref, axis=1).mean(1) > 0.75)))
    dec_K = pd.DataFrame(rows)
    dec_K.to_csv(path, index=False)
print(dec_K.round(3))

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for kap, c in zip(kappas_dec, ["C0", "C3"]):
    summ = best_restarts(ll_dec[ll_dec.kappa == kap]).groupby("K").test_ll.agg(["mean", "sem"])
    axes[0].errorbar(summ.index, summ["mean"], summ["sem"], marker="o", color=c, label=f"kappa={kap:g}")
    d = dec_K[dec_K.kappa == kap]
    axes[1].plot(d.K, d.ari_restarts, "o-", color=c)
    axes[2].plot(d.K, d.ari_halves, "o-", color=c)
    axes[3].plot(d.K, d.frac_states_reproducible, "o-", color=c)
for fold, df in m2_loao.groupby("fold"):
    axes[0].plot(df.K, df.test_ll, "--", color="0.5", lw=0.8)
axes[0].set_title("held-out LL / frame\n(LOSO mean ± SE; dashed: held-out animal)")
axes[1].set_title("ARI across 5 restarts")
axes[2].set_title("ARI between data halves")
axes[3].set_title("fraction of states reproducible\n(Jaccard > 0.75 across restarts)")
axes[0].legend()
for ax in axes:
    ax.set_xlabel("K")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "decide_K.png", dpi=120, bbox_inches="tight")
plt.show()


# %% DECIDE KAPPA: held-out LL and dwell times (decoded vs implied by the model) vs kappa, K=6, L=3  [cached; ~10 min]
# Effect of kappa on held-out LL and on state durations: decoded (Viterbi) median vs the mean implied by the transition matrix,
# 1 / (1 - p_kk). A kappa where the two disagree badly is one where the model's own dwell times are unrealistic.
K_dec = 6
kappas_sweep = [0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

path = RESULTS_DIR / f"decide_kappa_ll_K{K_dec}_L{L_dec}.csv"
if path.exists():
    ll_kap = pd.read_csv(path)
else:
    ll_kap = grid_scores(segs, folds_loso, "ar", [K_dec], [L_dec], kappas=kappas_sweep,
                         num_iters=sel["num_iters"], n_jobs=sel["n_jobs"])
    ll_kap.to_csv(path, index=False)

path = RESULTS_DIR / f"decide_kappa_dwell_K{K_dec}_L{L_dec}.csv"
if path.exists():
    dw_kap = pd.read_csv(path)
else:
    def _dwell(kap):
        m, _ = fit_model("ar", X_all_aug, K_dec, L=L_dec, kappa=kap, seed=cfg["seed"], num_iters=sel["num_iters"])
        decoded = np.concatenate(dwell_times([m.most_likely_states(X) for X in X_all], K_dec)) / f["fs_out"]
        implied = 1 / (1 - np.diag(m.transitions.transition_matrix)) / f["fs_out"]
        return dict(kappa=kap, decoded_median_s=np.median(decoded), implied_median_s=np.median(implied))
    dw_kap = pd.DataFrame(Parallel(n_jobs=sel["n_jobs"], verbose=10, pre_dispatch="all")(delayed(_dwell)(k) for k in kappas_sweep))
    dw_kap.to_csv(path, index=False)
print(dw_kap.round(3))

summ = ll_kap.groupby("kappa").test_ll.agg(["mean", "sem"])
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].errorbar(summ.index, summ["mean"], summ["sem"], marker="o")
axes[0].set_title(f"held-out LL / frame (LOSO), K={K_dec}, L={L_dec}")
axes[1].plot(dw_kap.kappa, dw_kap.decoded_median_s, "o-", label="decoded (Viterbi)")
axes[1].plot(dw_kap.kappa, dw_kap.implied_median_s, "s-", label="implied by transition matrix")
axes[1].axhline(np.median(cp_dur), color="k", ls="--", lw=0.8, label="changepoint median")
axes[1].set_yscale("log")
axes[1].set_ylabel("median state duration (s)")
axes[1].legend(fontsize=7)
for ax in axes:
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(-30, 3e8)
    ax.set_xlabel("kappa")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "decide_kappa.png", dpi=120, bbox_inches="tight")
plt.show()


# =============================================================================
# §6 VALIDATION OF THE SELECTED MODEL
# =============================================================================

# %% SELECTED: K, L, kappa chosen from the DECIDE plots (2026-10-02); overrides the automatic M2 / KAPPA picks
# K=4 is the largest K whose states reproduce (restarts and halves); kappa <= 1e3 is equivalent on held-out LL, 100 adds mild stickiness.
K_sel, L_sel, kappa = 4, 3, 100.0


# %% FINAL: selected AR-HMM on all sessions, n_restarts seeds; decode; save run  [cached]
# Refit the chosen K / L / kappa on all sessions (mirrored) with several seeds; keep the restart with the best training LL.
# States and posteriors are mapped back onto each session's original frames and saved to HMM/results/<timestamp>_<run_name>/.
seeds = list(range(cfg["seed"], cfg["seed"] + sel["n_restarts"]))
path = RESULTS_DIR / f"final_ar_K{K_sel}_L{L_sel}_kappa{kappa:g}.pkl"
if path.exists():
    models = pickle.load(open(path, "rb"))
else:
    models = [m for m, _ in Parallel(n_jobs=sel["n_jobs"], verbose=10, pre_dispatch="all")(
        delayed(fit_model)("ar", X_all_aug, K_sel, L=L_sel, kappa=kappa, seed=sd, num_iters=sel["num_iters"])
        for sd in seeds)]
    pickle.dump(models, open(path, "wb"))

train_lls = [ll_per_frame(m, X_all_aug) for m in models]
best = models[int(np.argmax(train_lls))]
zs_all = [best.most_likely_states(X) for X in X_all]
print("training LL / frame per restart:", np.round(train_lls, 4))

run_name = f"ar_K{K_sel}_L{L_sel}_kappa{kappa:g}"
if not any(RESULTS_DIR.glob(f"*_{run_name}")):   # save once per selected setting
    states = {s.session_id: to_session_frames(s, segs, zs_all, f["fs_out"]) for s in S}
    post = [best.expected_states(X)[0] for X in X_all]
    posteriors = {s.session_id: to_session_frames(s, segs, post, f["fs_out"], fill=np.nan) for s in S}
    run_dir = save_run(run_name, best, sc_all, cfg,
                       dict(kind="ar", K=K_sel, L=L_sel, kappa=kappa, seed=seeds[int(np.argmax(train_lls))],
                            num_iters=sel["num_iters"], mirror_augmented=True),
                       states, posteriors)
    print("saved", run_dir)


# %% REPRO: agreement across restarts (ARI) and per-state overlap; across data halves  [cached]
# Are the states a property of the data or of the random seed? ARI across restarts, per-state Jaccard (Hungarian-matched),
# and ARI between models fit on alternate-session halves.
z_cat = [np.concatenate([m.most_likely_states(X) for X in X_all]) for m in models]
ari = np.array([[adjusted_rand_score(a, b) for b in z_cat] for a in z_cat])
print(f"ARI across restarts: {ari[np.triu_indices(len(models), 1)].mean():.3f} (mean over pairs)")

z_ref = z_cat[int(np.argmax(train_lls))]
jacc = np.zeros((K_sel, len(models)))
for r, z in enumerate(z_cat):
    z = match_states(z_ref, z, K_sel)[z]
    for k in range(K_sel):
        jacc[k, r] = np.sum((z_ref == k) & (z == k)) / max(np.sum((z_ref == k) | (z == k)), 1)
repro = pd.DataFrame(dict(occupancy=np.bincount(z_ref, minlength=K_sel) / len(z_ref),
                          jaccard_across_restarts=np.delete(jacc, int(np.argmax(train_lls)), axis=1).mean(1)))
print(repro.round(3))

# halves: fit each half of the sessions, decode everything, compare
path = RESULTS_DIR / f"halves_ar_K{K_sel}_L{L_sel}_kappa{kappa:g}.pkl"
if path.exists():
    m_halves = pickle.load(open(path, "rb"))
else:
    Xh = [[X for g, X in zip(segs, X_all) if g["session_id"] in h] for h in halves]
    m_halves = [m for m, _ in Parallel(n_jobs=2, verbose=10, pre_dispatch="all")(
        delayed(fit_model)("ar", Xs + [mirror(X) for X in Xs], K_sel, L=L_sel, kappa=kappa,
                           seed=cfg["seed"], num_iters=sel["num_iters"]) for Xs in Xh)]
    pickle.dump(m_halves, open(path, "wb"))
z_h = [np.concatenate([m.most_likely_states(X) for X in X_all]) for m in m_halves]
print(f"ARI between models fit on the two halves (decoding all data): {adjusted_rand_score(*z_h):.3f}")


# %% GENERATIVE: simulate from the fitted model; compare dwell times, omega spectrum, transitions
# Posterior-predictive check: sample from the model and compare with the data: dwell-time distributions (two-sample KS test),
# omega_yaw power spectrum (Welch 1967) and state-transition frequencies. A dwell misfit is the reason to try the HSMM (M3).
zs_sim, xs_sim = simulate(best, [len(X) for X in X_all], seed=cfg["seed"])
d_real, d_sim = dwell_times(zs_all, K_sel), dwell_times(zs_sim, K_sel)
implied_s = 1 / (1 - np.diag(best.transitions.transition_matrix)) / f["fs_out"]   # mean dwell the model implies

rows = []
for k in range(K_sel):
    a, b = d_real[k] / f["fs_out"], d_sim[k] / f["fs_out"]
    ks = ks_2samp(a, b).statistic if len(a) and len(b) else 1.0   # no complete simulated dwell = maximal misfit
    rows.append(dict(state=k, n_real=len(a), mean_real_s=a.mean() if len(a) else np.nan, implied_mean_s=implied_s[k],
                     mean_sim_s=b.mean() if len(b) else np.nan,
                     cv_real=a.std() / a.mean() if len(a) else np.nan, cv_sim=b.std() / b.mean() if len(b) else np.nan, ks=ks))
dwell_fit = pd.DataFrame(rows).set_index("state")
print(dwell_fit.round(3))
need_hsmm = dwell_fit.ks.median() > 0.1   # dwell-time misfit -> try the HSMM (M3)
print("dwell-time misfit (median KS > 0.1) -> HSMM needed:", need_hsmm)

fig, axes = create_subplot_grid(K_sel)
bins = np.logspace(np.log10(1 / f["fs_out"]), 1.5, 30)
for k in range(K_sel):
    axes[k].hist(d_real[k] / f["fs_out"], bins=bins, density=True, histtype="step", label="real")
    axes[k].hist(d_sim[k] / f["fs_out"], bins=bins, density=True, histtype="step", label="model")
    axes[k].set_xscale("log")
    axes[k].set_title(f"state {k}")
axes[0].legend()
plt.show()

fq, P_real = spectrum(X_all, fs=f["fs_out"])
_, P_sim = spectrum(xs_sim, fs=f["fs_out"])
T_real, T_sim = transition_freqs(zs_all, K_sel), transition_freqs(zs_sim, K_sel)
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
axes[0].semilogy(fq, P_real, label="real")
axes[0].semilogy(fq, P_sim, label="model")
axes[0].set_xlabel("Hz")
axes[0].set_title("omega_yaw power")
axes[0].legend()
for ax, T, name in [(axes[1], T_real, "real"), (axes[2], T_sim, "model")]:
    ax.imshow(T, vmin=0, vmax=max(T_real.max(), T_sim.max()))
    ax.set_title(f"transitions ({name})")
    ax.set_xlabel("to")
    ax.set_ylabel("from")
plt.tight_layout()
plt.show()
print(f"transition-frequency correlation real vs model: {np.corrcoef(T_real.ravel(), T_sim.ravel())[0, 1]:.3f}")


# %% BOUNDARIES: state boundaries vs model-free changepoints (±100 ms), circular-shift null
# Do state switches coincide with PELT changepoints more than chance? Null = state sequences circularly shifted within each
# segment, which keeps their durations but breaks the alignment with the kinematics.
obs, null = boundary_agreement(zs_all, cps[pen_star], tol=ts["tol_frames"])
print(f"{obs:.3f} of state boundaries within ±{ts['tol_frames']} frames of a changepoint "
      f"(null {null.mean():.3f} ± {null.std():.3f}, p = {np.mean(null >= obs):.3f})")
fig, ax = plt.subplots(figsize=(4, 3))
ax.hist(null, bins=30, color="0.7")
ax.axvline(obs, color="r")
ax.set_xlabel("fraction of boundaries near a changepoint")
plt.show()


# %% MIRROR: pair states by sign-flipped emission parameters; per-direction vs merged; vs |omega| fit
# Pair left/right mirror states (sign-flipped AR parameters) and merge them; compare with a model fit on |omega| (direction
# removed). High ARI means the merged states are the same movements regardless of turn direction.
pair, dist = mirror_pairs(best, SIGNED)
merged = np.minimum(np.arange(K_sel), pair)        # each mirror pair labelled by its lower index
raw_all = np.concatenate([g["X"] for g in segs])
z_flat = np.concatenate(zs_all)
desc = pd.DataFrame({name: [raw_all[z_flat == k, j].mean() for k in range(K_sel)] for j, name in enumerate(FEATURE_NAMES)})
desc["occupancy"] = np.bincount(z_flat, minlength=K_sel) / len(z_flat)
desc["mirror_partner"], desc["param_dist"], desc["merged"] = pair, dist, merged
print(desc.round(3))
print(f"{K_sel} per-direction states -> {len(np.unique(merged))} merged states")

X_abs = [X.copy() for X in X_all]
for X in X_abs:
    X[:, SIGNED] = np.abs(X[:, SIGNED])
path = RESULTS_DIR / f"abs_omega_ar_K{len(np.unique(merged))}_L{L_sel}_kappa{kappa:g}.pkl"
if path.exists():
    m_abs = pickle.load(open(path, "rb"))
else:
    m_abs, _ = fit_model("ar", X_abs, len(np.unique(merged)), L=L_sel, kappa=kappa, seed=cfg["seed"], num_iters=sel["num_iters"])
    pickle.dump(m_abs, open(path, "wb"))
z_abs = np.concatenate([m_abs.most_likely_states(X) for X in X_abs])
print(f"ARI merged-mirror states vs |omega| model: {adjusted_rand_score(merged[z_flat], z_abs):.3f}")


# %% DESCRIBE: state-triggered averages of the kinematics (raw units) + event table for video checks
# Mean features ±1 s around each state's onset (like a spike-triggered average) to name the states; the event table
# (session, state, start/stop times) is for pulling video clips of each state.
win = int(1 * f["fs_out"])   # ±1 s around each state onset
lags = np.arange(-win, win) / f["fs_out"]
fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(16, 3))
for k in range(K_sel):
    snips = []
    for g, z in zip(segs, zs_all):
        states, starts, _ = run_lengths(z)
        snips += [g["X"][a - win:a + win] for a in starts[(states == k) & (starts >= win) & (starts + win <= len(z))]]
    if snips:
        for j, name in enumerate(FEATURE_NAMES):
            axes[j].plot(lags, np.mean(snips, axis=0)[:, j], color=plt.cm.tab20(k), label=f"{k} (n={len(snips)})")
for j, name in enumerate(FEATURE_NAMES):
    axes[j].axvline(0, color="k", lw=0.5)
    axes[j].set_title(name)
    axes[j].set_xlabel("time from state onset (s)")
axes[-1].legend(fontsize=6)
plt.tight_layout()
plt.show()

events = event_table(segs, zs_all, f["fs_out"])
events.to_csv(RESULTS_DIR / f"events_ar_K{K_sel}_L{L_sel}.csv", index=False)
print(events.assign(dur=events.t_stop - events.t_start).groupby("state").dur.describe().round(2))

i = int(np.argmax([len(g["t"]) for g in segs]))   # longest segment
plot_states(segs[i]["t"], segs[i]["X"], zs_all[i], title=f"AR-HMM K={K_sel} L={L_sel}, kappa={kappa:g}")


# =============================================================================
# M3: HSMM — only if GENERATIVE shows a dwell-time misfit
# =============================================================================

# %% M3: HSMM (warm-started from the sticky AR-HMM) vs sticky AR-HMM on the same LOSO folds  [cached]
# HSMM with negative-binomial durations (ssm HSMM, r_max = max NB shape) vs the sticky AR-HMM on the same folds.
# A consistent positive held-out LL gain means explicit state durations are worth the extra parameters.
if need_hsmm:
    path = RESULTS_DIR / f"m3_hsmm_K{K_sel}_L{L_sel}_kappa{kappa:g}.csv"
    if path.exists():
        m3 = pd.read_csv(path)
    else:
        def _m3(fold):
            Xtr, Xte, _ = fold_data(segs, *fold)
            ms, _ = fit_model("ar", Xtr, K_sel, L=L_sel, kappa=kappa, seed=cfg["seed"], num_iters=sel["num_iters"])
            mh, _ = fit_model("hsmm", Xtr, K_sel, L=L_sel, num_iters=cfg["hsmm"]["num_iters"],
                              r_max=cfg["hsmm"]["r_max"], init_from=ms)
            return dict(sticky=ll_per_frame(ms, Xte), hsmm=ll_per_frame(mh, Xte))
        m3 = pd.DataFrame(Parallel(n_jobs=sel["n_jobs"], verbose=10, pre_dispatch="all")(delayed(_m3)(fold) for fold in folds_loso))
        m3.to_csv(path, index=False)
    m3["diff"] = m3.hsmm - m3.sticky
    print(m3.round(4))
    print(f"HSMM - sticky: {m3['diff'].mean():+.4f} ± {m3['diff'].sem():.4f} nats/frame; "
          f"HSMM better in {(m3['diff'] > 0).sum()}/{len(m3)} folds")
